#!/usr/bin/env python3
"""
predict.py  —  ShoggoTEh genome annotation

Slides windows across a new genome, generates Hyena-DNA embeddings, and
classifies each window with the trained MLP head. Writes a BED file with
the top predicted class and a TSV with full per-class softmax probabilities.

Usage:
    python predict.py \
        --fasta genome.fa \
        --model_dir models/plant_classifier/ \
        --outdir predictions/

Output (prefix defaults to the FASTA filename stem):
    {outdir}/{prefix}.bed          — top predicted class per window
    {outdir}/{prefix}_probs.tsv    — full softmax probability matrix
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from codecarbon import EmissionsTracker
from pyfaidx import Fasta
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Classifier architecture (must match train_classifier.py) ──────────────────

class TEClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Windowing ──────────────────────────────────────────────────────────────────

def make_windows(fasta: Fasta, chunk_size: int, stride: int) -> list[tuple]:
    windows = []
    for chrom in fasta.keys():
        chrom_len = len(fasta[chrom])
        if chrom_len < chunk_size:
            continue
        start = 0
        while start + chunk_size <= chrom_len:
            windows.append((chrom, start, start + chunk_size))
            start += stride
        remainder = chrom_len - start
        if 0 < remainder >= chunk_size // 2:
            windows.append((chrom, chrom_len - chunk_size, chrom_len))
    return windows


# ── Embedding ──────────────────────────────────────────────────────────────────

def embed_sequences(
    sequences: list[str],
    tokenizer,
    hyena_model,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    all_embeddings = []
    n = len(sequences)
    for batch_start in range(0, n, batch_size):
        batch_seqs = sequences[batch_start : batch_start + batch_size]
        enc = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = hyena_model(**enc)
        hidden = out.last_hidden_state
        if "attention_mask" in enc:
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            emb = hidden.mean(dim=1)
        all_embeddings.append(emb.cpu().float().numpy())
        done = min(batch_start + batch_size, n)
        if done % (batch_size * 10) == 0 or done == n:
            log.info(f"  {done:,} / {n:,} sequences embedded")
    return np.vstack(all_embeddings)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_hyena(model_name: str, device: torch.device):
    log.info(f"Loading Hyena-DNA: {model_name}")
    tokenizer  = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hyena_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    hyena_model.eval().to(device)
    return tokenizer, hyena_model


def load_classifier(model_dir: Path, device: torch.device):
    encoder_path = model_dir / "label_encoder.json"
    weights_path = model_dir / "classifier.pt"

    if not encoder_path.exists():
        log.error(f"label_encoder.json not found in {model_dir}")
        sys.exit(1)
    if not weights_path.exists():
        log.error(f"classifier.pt not found in {model_dir}")
        sys.exit(1)

    with open(encoder_path) as fh:
        enc = json.load(fh)
    labels     = [enc["idx_to_label"][str(i)] for i in range(len(enc["idx_to_label"]))]
    n_classes  = len(labels)

    # Infer input_dim and hidden_dim from saved weights
    state = torch.load(weights_path, map_location=device)
    input_dim  = state["net.0.weight"].shape[1]
    hidden_dim = state["net.0.weight"].shape[0]

    classifier = TEClassifier(input_dim=input_dim, hidden_dim=hidden_dim,
                               n_classes=n_classes)
    classifier.load_state_dict(state)
    classifier.eval().to(device)
    log.info(f"Classifier loaded: {input_dim}→{hidden_dim}→{n_classes} classes")
    log.info(f"Labels: {labels}")
    return classifier, labels


# ── Output writers ─────────────────────────────────────────────────────────────

def write_bed(windows, top_labels, confidences, bed_path: Path) -> None:
    with open(bed_path, "w") as fh:
        for (chrom, start, end), label, score in zip(windows, top_labels, confidences):
            fh.write(f"{chrom}\t{start}\t{end}\t{label}\t{score:.4f}\t.\n")
    log.info(f"BED written to {bed_path}")


def write_probs_tsv(windows, top_labels, confidences, probs, labels,
                    tsv_path: Path) -> None:
    header = ["chrom", "start", "end", "label", "confidence"] + labels
    with open(tsv_path, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for (chrom, start, end), label, score, prob_row in zip(
                windows, top_labels, confidences, probs):
            row = [chrom, str(start), str(end), label, f"{score:.4f}"]
            row += [f"{p:.4f}" for p in prob_row]
            fh.write("\t".join(row) + "\n")
    log.info(f"Probability TSV written to {tsv_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict TE classes for a genome using a trained ShoggoTEh model"
    )
    parser.add_argument("--fasta",      required=True,
                        help="Input genome FASTA (uncompressed or bgzipped)")
    parser.add_argument("--model_dir",  required=True,
                        help="Directory with classifier.pt and label_encoder.json")
    parser.add_argument("--outdir",     required=True,
                        help="Output directory for BED and probability TSV")
    parser.add_argument("--prefix",     default=None,
                        help="Output file prefix (default: FASTA filename stem)")
    parser.add_argument("--hyena_model",
                        default="LongSafari/hyenadna-medium-160k-seqlen-hf",
                        help="Hyena-DNA model on HuggingFace Hub")
    parser.add_argument("--chunk_size",     type=int,   default=5000,
                        help="Window size in bp (default: 5000)")
    parser.add_argument("--overlap",        type=float, default=0.5,
                        help="Fractional overlap between windows (default: 0.5)")
    parser.add_argument("--max_n_fraction", type=float, default=0.1,
                        help="Max N fraction — windows above this are skipped (default: 0.1)")
    parser.add_argument("--batch_size",     type=int,   default=32,
                        help="Embedding batch size (default: 32)")
    parser.add_argument("--device",         default=None,
                        help="Compute device: 'cuda', 'mps', or 'cpu'. Auto-detected if omitted.")
    args = parser.parse_args()

    fasta_path = Path(args.fasta)
    model_dir  = Path(args.model_dir)
    outdir     = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    prefix = args.prefix or fasta_path.stem

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Using device: {device}")

    # ── Load models ────────────────────────────────────────────────────────────
    tokenizer, hyena_model = load_hyena(args.hyena_model, device)
    classifier, labels     = load_classifier(model_dir, device)

    tracker = EmissionsTracker(
        project_name="ShoggoTEh_predict",
        output_dir=str(log_dir),
        log_level="error",
    )
    tracker.start()

    # ── Slide windows ──────────────────────────────────────────────────────────
    log.info(f"Loading genome: {fasta_path}")
    fasta  = Fasta(str(fasta_path))
    stride = int(args.chunk_size * (1 - args.overlap))
    all_windows = make_windows(fasta, args.chunk_size, stride)
    log.info(f"{len(all_windows):,} windows generated (chunk={args.chunk_size}, stride={stride})")

    # Filter high-N windows and extract sequences
    sequences, kept_windows = [], []
    skipped_n = 0
    for chrom, start, end in all_windows:
        seq = str(fasta[chrom][start:end]).upper()
        if seq.count("N") / len(seq) > args.max_n_fraction:
            skipped_n += 1
            continue
        sequences.append(seq)
        kept_windows.append((chrom, start, end))

    log.info(f"{len(kept_windows):,} windows kept | {skipped_n:,} skipped (N content)")

    if not kept_windows:
        log.error("No windows passed the N-content filter. Check your FASTA.")
        sys.exit(1)

    # ── Generate embeddings ────────────────────────────────────────────────────
    log.info("Generating embeddings ...")
    embeddings = embed_sequences(sequences, tokenizer, hyena_model, device, args.batch_size)

    # ── Classify ───────────────────────────────────────────────────────────────
    log.info("Running classifier ...")
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = classifier(X)
        probs  = F.softmax(logits, dim=1).cpu().numpy()

    top_indices  = probs.argmax(axis=1)
    top_labels   = [labels[i] for i in top_indices]
    confidences  = probs[np.arange(len(probs)), top_indices]

    # Summary
    from collections import Counter
    dist = Counter(top_labels)
    log.info("Prediction distribution: " +
             " | ".join(f"{k}: {v:,}" for k, v in sorted(dist.items())))

    # ── Write output ───────────────────────────────────────────────────────────
    write_bed(kept_windows, top_labels, confidences,
              outdir / f"{prefix}.bed")
    write_probs_tsv(kept_windows, top_labels, confidences, probs, labels,
                    outdir / f"{prefix}_probs.tsv")

    emissions = tracker.stop()
    log.info(f"Carbon footprint: {emissions:.6f} kg CO2 equivalent")
    log.info("Done.")


if __name__ == "__main__":
    main()
