#!/usr/bin/env python3
"""
generate_embeddings.py  —  ShoggoTEh embedding generation

Loads labeled sequence chunks from Parquet files (output of prepare_dataset.py),
runs them through Hyena-DNA to generate per-chunk embeddings via mean pooling
of the last hidden state, and writes one Parquet file per species to the output
directory.

Usage:
    python generate_embeddings.py \
        --chunks_dir data/chunks/ \
        --outdir data/embeddings/

Output columns per Parquet file:
    species, chrom, start, end, label, repeat_fraction, embedding
    (sequence is dropped — it remains in the chunks Parquet)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from codecarbon import EmissionsTracker
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_name: str, device: torch.device):
    log.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"Model ready on {device} ({n_params:.1f}M parameters)")
    return tokenizer, model


# ── Embedding ──────────────────────────────────────────────────────────────────

def embed_sequences(
    sequences: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    Run sequences through Hyena-DNA and return mean-pooled embeddings.
    Returns an array of shape (N, hidden_dim).
    """
    all_embeddings = []
    n = len(sequences)

    for batch_start in range(0, n, batch_size):
        batch_seqs = sequences[batch_start : batch_start + batch_size]

        enc = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)

        hidden = out.last_hidden_state          # (B, L, D)

        if "attention_mask" in enc:
            mask = enc["attention_mask"].unsqueeze(-1).float()   # (B, L, 1)
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (B, D)
        else:
            emb = hidden.mean(dim=1)

        all_embeddings.append(emb.cpu().float().numpy())

        done = min(batch_start + batch_size, n)
        if done % (batch_size * 10) == 0 or done == n:
            log.info(f"  {done:,} / {n:,} sequences embedded")

    return np.vstack(all_embeddings)


# ── Per-species processing ─────────────────────────────────────────────────────

def process_species(
    chunks_path: Path,
    outdir: Path,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
) -> None:
    species_id = chunks_path.stem
    out_path   = outdir / f"{species_id}.parquet"

    if out_path.exists():
        log.info(f"{species_id}: embedding file already exists, skipping")
        return

    log.info(f"{species_id}: loading chunks from {chunks_path}")
    df = pd.read_parquet(chunks_path)
    log.info(f"{species_id}: {len(df):,} chunks to embed")

    sequences = df["sequence"].tolist()

    log.info(f"{species_id}: generating embeddings (batch_size={batch_size})")
    embeddings = embed_sequences(sequences, tokenizer, model, device, batch_size)
    log.info(f"{species_id}: embedding dim = {embeddings.shape[1]}")

    df = df.drop(columns=["sequence"])
    df["embedding"] = [emb.tolist() for emb in embeddings]

    df.to_parquet(out_path, index=False)
    log.info(f"{species_id}: written to {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Hyena-DNA embeddings for ShoggoTEh chunks"
    )
    parser.add_argument(
        "--chunks_dir", required=True,
        help="Directory containing per-species Parquet files from prepare_dataset.py",
    )
    parser.add_argument(
        "--outdir", required=True,
        help="Output directory for per-species embedding Parquet files",
    )
    parser.add_argument(
        "--model",
        default="LongSafari/hyenadna-medium-160k-seqlen-hf",
        help="Hyena-DNA model name on HuggingFace Hub (default: hyenadna-medium-160k)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Number of sequences per forward pass (default: 32)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Compute device: 'cuda', 'mps', or 'cpu'. Auto-detected if omitted.",
    )
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    outdir     = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    chunk_files = sorted(chunks_dir.glob("*.parquet"))
    if not chunk_files:
        log.error(f"No Parquet files found in {chunks_dir}")
        sys.exit(1)
    log.info(f"Found {len(chunk_files)} species to embed: "
             f"{[f.stem for f in chunk_files]}")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Using device: {device}")

    tokenizer, model = load_model(args.model, device)

    tracker = EmissionsTracker(
        project_name="ShoggoTEh_generate_embeddings",
        output_dir=str(log_dir),
        log_level="error",
    )
    tracker.start()

    for chunk_path in chunk_files:
        try:
            process_species(
                chunks_path=chunk_path,
                outdir=outdir,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=args.batch_size,
            )
        except Exception as exc:
            log.error(f"{chunk_path.stem}: FAILED — {exc}")
            continue

    emissions = tracker.stop()
    log.info(f"Carbon footprint: {emissions:.6f} kg CO2 equivalent")
    log.info("Done.")


if __name__ == "__main__":
    main()
