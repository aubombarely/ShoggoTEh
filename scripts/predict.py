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

import argparse, warnings
import getpass
import json
import logging
import os
import platform
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyfaidx import Fasta
from transformers import AutoModel, AutoTokenizer

try:
    from codecarbon import EmissionsTracker
    _CODECARBON_AVAILABLE = True
except ImportError:
    _CODECARBON_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

VERSION = "v0.2.0"
SCRIPT_NAME = "predict"


_QUOTE_LINES = [
    "\"It was a terrible, indescribable thing vaster than any subway train —",
    " a shapeless congeries of protoplasmic bubbles, faintly self-luminous,",
    " and with myriads of temporary eyes forming and unforming as pustules",
    " of greenish light all over the tunnel-filling front that bore down upon us.\"",
    "                  — H.P. Lovecraft, At the Mountains of Madness (1931)",
]

def _print_quote() -> None:
    width = max(len(l) for l in _QUOTE_LINES) + 4
    border = "─" * width
    print(f"┌{border}┐", file=sys.stderr)
    for line in _QUOTE_LINES:
        padding = width - len(line) - 1
        print(f"│ {line}{' ' * padding}│", file=sys.stderr)
    print(f"└{border}┘", file=sys.stderr)


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
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")
    _print_quote()
    parser = argparse.ArgumentParser(
        description="Predict TE classes for a genome using a trained ShoggoTEh model"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
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
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate inputs and print what would run, then exit without executing")
    parser.add_argument("--disable_co2_tracking", action="store_true",
                        help="Disable carbon footprint tracking even if codecarbon is installed")
    args = parser.parse_args()

    t_start = time.monotonic()

    fasta_path = Path(args.fasta)
    model_dir  = Path(args.model_dir)
    outdir     = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logs_dir = Path(args.outdir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"Run_{SCRIPT_NAME}.log"
    sep = "=" * 62
    with open(log_path, "w") as _fh:
        _fh.write(f"{sep}\n  {SCRIPT_NAME} {VERSION}  —  Run Log\n{sep}\n")
        _fh.write(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        _fh.write(f"User      : {getpass.getuser()}\n")
        _fh.write(f"Server    : {platform.node()}\n")
        _fh.write(f"OS        : {platform.system()} {platform.release()} ({platform.machine()})\n")
        _fh.write(f"Directory : {os.getcwd()}\n")
        _fh.write(f"Command   : {' '.join(sys.argv)}\n")
        _fh.write(f"{sep}\n\n")
    _fh_handler = logging.FileHandler(log_path, mode="a")
    _fh_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(_fh_handler)

    if not fasta_path.exists():
        print(f"ERROR: --fasta not found: {fasta_path}", file=sys.stderr)
        sys.exit(1)
    if not model_dir.exists():
        print(f"ERROR: --model_dir not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    prefix = args.prefix or fasta_path.stem

    if args.dry_run:
        log.info("Dry run — no steps will be executed")
        log.info(f"  fasta       : {fasta_path}")
        log.info(f"  model_dir   : {model_dir}/")
        log.info(f"  outdir      : {outdir}/")
        log.info(f"  prefix      : {prefix}")
        log.info(f"  hyena_model : {args.hyena_model}")
        log.info(f"  chunk_size  : {args.chunk_size}")
        log.info(f"  overlap     : {args.overlap}")
        log.info(f"  max_n_fraction: {args.max_n_fraction}")
        log.info(f"  batch_size  : {args.batch_size}")
        log.info("  Steps that would run: slide windows → embed → classify → write BED + TSV")
        log.info("  Exiting (--dry_run).")
        sys.exit(0)

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

    _tracker = None
    if _CODECARBON_AVAILABLE and not args.disable_co2_tracking:
        _tracker = EmissionsTracker(
            project_name="ShoggoTEh_predict",
            output_dir=str(logs_dir),
            log_level="error",
        )
        _tracker.start()
        log.info("  codecarbon tracker started")
    elif args.disable_co2_tracking:
        log.info("  Carbon footprint tracking disabled (--disable_co2_tracking)")
    else:
        log.info("  codecarbon not installed — carbon tracking skipped")

    # ── Slide windows ──────────────────────────────────────────────────────────
    log.info(f"Loading genome: {fasta_path}")
    fasta  = Fasta(str(fasta_path))
    stride = int(args.chunk_size * (1 - args.overlap))
    all_windows = make_windows(fasta, args.chunk_size, stride)
    n_windows_total = len(all_windows)
    log.info(f"{n_windows_total:,} windows generated (chunk={args.chunk_size}, stride={stride})")

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

    n_windows_kept = len(kept_windows)
    log.info(f"{n_windows_kept:,} windows kept | {skipped_n:,} skipped (N content)")

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

    emissions_kg = None
    if _tracker is not None:
        try:
            emissions_kg = _tracker.stop()
        except Exception:
            pass

    if emissions_kg is not None:
        log.info(f"Carbon footprint: {emissions_kg:.6f} kg CO2 equivalent")
    log.info("Done.")

    elapsed_s = time.monotonic() - t_start
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_mem_mb = ru.ru_maxrss / (1024 * 1024) if platform.system() == "Darwin" else ru.ru_maxrss / 1024
    log.info(f"Wall-clock time   : {elapsed_s:.1f} s ({elapsed_s/60:.1f} min)")
    log.info(f"Peak memory (RSS) : {peak_mem_mb:.1f} MB")

    summary = {
        "date":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION,
        "input_fasta":       str(args.fasta),
        "n_windows_total":   n_windows_total,
        "n_windows_kept":    n_windows_kept,
        "n_skipped_n":       skipped_n,
        "parameters": {
            "hyena_model":    args.hyena_model,
            "chunk_size":     args.chunk_size,
            "overlap":        args.overlap,
            "max_n_fraction": args.max_n_fraction,
            "batch_size":     args.batch_size,
            "device":         str(device),
        },
        "resource_usage": {
            "wall_clock_s":       round(elapsed_s, 1),
            "peak_mem_mb":        round(peak_mem_mb, 1),
            "emissions_kg_CO2eq": emissions_kg,
        },
    }
    summary_path = Path(args.outdir) / "run_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    log.info(f"Run summary written to {summary_path}")


if __name__ == "__main__":
    main()
