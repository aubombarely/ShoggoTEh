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
import pandas as pd
import torch
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
SCRIPT_NAME = "generate_embeddings"


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
    force: bool = False,
) -> int:
    """Process one species and return embedding dim (0 if skipped)."""
    species_id = chunks_path.stem
    out_path   = outdir / f"{species_id}.parquet"

    if not force and out_path.exists():
        log.info(f"{species_id}: embedding file already exists, skipping")
        return 0

    log.info(f"{species_id}: loading chunks from {chunks_path}")
    df = pd.read_parquet(chunks_path)
    log.info(f"{species_id}: {len(df):,} chunks to embed")

    sequences = df["sequence"].tolist()

    log.info(f"{species_id}: generating embeddings (batch_size={batch_size})")
    embeddings = embed_sequences(sequences, tokenizer, model, device, batch_size)
    embedding_dim = embeddings.shape[1]
    log.info(f"{species_id}: embedding dim = {embedding_dim}")

    df = df.drop(columns=["sequence"])
    df["embedding"] = [emb.tolist() for emb in embeddings]

    df.to_parquet(out_path, index=False)
    log.info(f"{species_id}: written to {out_path}")
    return embedding_dim


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Hyena-DNA embeddings for ShoggoTEh chunks"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
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
    parser.add_argument("--force", action="store_true",
                        help="Rerun all species even if output Parquet files already exist")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate inputs and print what would run, then exit without executing")
    parser.add_argument("--disable_co2_tracking", action="store_true",
                        help="Disable carbon footprint tracking even if codecarbon is installed")
    args = parser.parse_args()

    t_start = time.monotonic()

    chunks_dir = Path(args.chunks_dir)
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

    if not chunks_dir.exists():
        print(f"ERROR: --chunks_dir not found: {chunks_dir}", file=sys.stderr)
        sys.exit(1)

    chunk_files = sorted(chunks_dir.glob("*.parquet"))
    if not chunk_files:
        log.error(f"No Parquet files found in {chunks_dir}")
        sys.exit(1)
    log.info(f"Found {len(chunk_files)} species to embed: "
             f"{[f.stem for f in chunk_files]}")

    if args.dry_run:
        log.info("Dry run — no steps will be executed")
        log.info(f"  chunks_dir  : {chunks_dir}/")
        log.info(f"  outdir      : {outdir}/")
        log.info(f"  model       : {args.model}")
        log.info(f"  batch_size  : {args.batch_size}")
        log.info(f"  force       : {args.force}")
        log.info(f"  Species that would be embedded: {[f.stem for f in chunk_files]}")
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

    tokenizer, model = load_model(args.model, device)

    _tracker = None
    if _CODECARBON_AVAILABLE and not args.disable_co2_tracking:
        _tracker = EmissionsTracker(
            project_name="ShoggoTEh_generate_embeddings",
            output_dir=str(logs_dir),
            log_level="error",
        )
        _tracker.start()
        log.info("  codecarbon tracker started")
    elif args.disable_co2_tracking:
        log.info("  Carbon footprint tracking disabled (--disable_co2_tracking)")
    else:
        log.info("  codecarbon not installed — carbon tracking skipped")

    n_species_processed = 0
    embedding_dim = None

    for chunk_path in chunk_files:
        try:
            dim = process_species(
                chunks_path=chunk_path,
                outdir=outdir,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=args.batch_size,
                force=args.force,
            )
            if dim > 0:
                n_species_processed += 1
                embedding_dim = dim
        except Exception as exc:
            log.error(f"{chunk_path.stem}: FAILED — {exc}")
            continue

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
        "input_chunks_dir": str(args.chunks_dir),
        "n_species_processed": n_species_processed,
        "embedding_dim": embedding_dim,
        "parameters": {
            "model":      args.model,
            "batch_size": args.batch_size,
            "device":     str(device),
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
