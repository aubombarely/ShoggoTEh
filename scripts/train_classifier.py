#!/usr/bin/env python3
"""
train_classifier.py  —  ShoggoTEh classifier training

Loads per-species embedding Parquet files (output of generate_embeddings.py),
trains a one-hidden-layer MLP on top of the frozen Hyena-DNA embeddings, and
saves the model weights and label encoder to the output directory.

Usage:
    python train_classifier.py \
        --embeddings_dir data/embeddings/ \
        --outdir models/plant_classifier/

Output:
    {outdir}/classifier.pt          — best model weights (state dict)
    {outdir}/label_encoder.json     — label → index mapping
    {outdir}/training_metrics.tsv   — per-epoch loss and accuracy
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
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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
SCRIPT_NAME = "train_classifier"

DEFAULT_LABELS = ["LTR", "DNA", "LINE", "SINE", "Unknown_repeat", "Genic", "Intergenic"]


# ── Model ──────────────────────────────────────────────────────────────────────

class TEClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_embeddings(embeddings_dir: Path, labels: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load all embedding Parquet files, filter to configured labels, and return
    (X, y_encoded, label_names) where y_encoded contains integer indices.
    """
    parquet_files = sorted(embeddings_dir.glob("*.parquet"))
    if not parquet_files:
        log.error(f"No Parquet files found in {embeddings_dir}")
        sys.exit(1)

    log.info(f"Loading embeddings from {len(parquet_files)} species: "
             f"{[f.stem for f in parquet_files]}")

    dfs = []
    for p in parquet_files:
        df = pd.read_parquet(p)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    log.info(f"Total chunks loaded: {len(df):,}")

    label_set = set(labels)
    before = len(df)
    df = df[df["label"].isin(label_set)].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        log.warning(f"Dropped {dropped:,} chunks with labels outside the configured set")
    log.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    y = df["label"].map(label_to_idx).values.astype(np.int64)
    X = np.vstack(df["embedding"].values)

    log.info(f"Embedding dim: {X.shape[1]}")
    return X, y, labels


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    outdir: Path,
) -> list[dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    metrics = []

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss   += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total   += len(y_batch)

        # ── validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
                val_loss    += loss.item() * len(y_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total   += len(y_batch)

        t_loss = train_loss / train_total
        v_loss = val_loss   / val_total
        t_acc  = train_correct / train_total
        v_acc  = val_correct   / val_total

        metrics.append({
            "epoch":      epoch,
            "train_loss": round(t_loss, 6),
            "val_loss":   round(v_loss, 6),
            "train_acc":  round(t_acc,  4),
            "val_acc":    round(v_acc,  4),
        })
        log.info(f"Epoch {epoch:3d}/{epochs} | "
                 f"train loss {t_loss:.4f} acc {t_acc:.3f} | "
                 f"val loss {v_loss:.4f} acc {v_acc:.3f}")

        # ── early stopping ──
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), outdir / "classifier.pt")
            log.info(f"  -> best model saved (val_loss={v_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log.info(f"Early stopping after {epoch} epochs (no improvement for {patience})")
                break

    return metrics


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train MLP classifier on Hyena-DNA embeddings for TE classification"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument(
        "--embeddings_dir", required=True,
        help="Directory with per-species embedding Parquet files",
    )
    parser.add_argument(
        "--outdir", required=True,
        help="Output directory for model weights, label encoder, and metrics",
    )
    parser.add_argument(
        "--labels", nargs="+", default=DEFAULT_LABELS,
        help="Ordered list of class labels (default: 7 TE/genic classes)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Mini-batch size for classifier training (default: 256)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Adam learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=512,
        help="Hidden layer width (default: 512)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3,
        help="Dropout probability (default: 0.3)",
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.2,
        help="Fraction of data held out for validation (default: 0.2)",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early-stopping patience in epochs (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Compute device: 'cuda', 'mps', or 'cpu'. Auto-detected if omitted.",
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate inputs and print what would run, then exit without executing")
    parser.add_argument("--disable_co2_tracking", action="store_true",
                        help="Disable carbon footprint tracking even if codecarbon is installed")
    args = parser.parse_args()

    t_start = time.monotonic()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
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

    embeddings_dir = Path(args.embeddings_dir)
    if not embeddings_dir.exists():
        print(f"ERROR: --embeddings_dir not found: {embeddings_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        log.info("Dry run — no steps will be executed")
        log.info(f"  embeddings_dir : {embeddings_dir}/")
        log.info(f"  outdir         : {outdir}/")
        log.info(f"  epochs         : {args.epochs}")
        log.info(f"  batch_size     : {args.batch_size}")
        log.info(f"  lr             : {args.lr}")
        log.info(f"  hidden_dim     : {args.hidden_dim}")
        log.info(f"  dropout        : {args.dropout}")
        log.info(f"  val_fraction   : {args.val_fraction}")
        log.info(f"  patience       : {args.patience}")
        log.info(f"  labels         : {args.labels}")
        log.info("  Steps that would run: load embeddings → train MLP → evaluate → save model")
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

    # ── Load data ──────────────────────────────────────────────────────────────
    X, y, labels = load_embeddings(embeddings_dir, args.labels)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=args.val_fraction,
        stratify=y,
        random_state=args.seed,
    )
    log.info(f"Train: {len(X_train):,} | Val: {len(X_val):,}")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # ── Build model ────────────────────────────────────────────────────────────
    input_dim = X.shape[1]
    model = TEClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        n_classes=len(labels),
        dropout=args.dropout,
    ).to(device)
    log.info(
        f"Model: Linear({input_dim}, {args.hidden_dim}) → ReLU → Dropout({args.dropout}) "
        f"→ Linear({args.hidden_dim}, {len(labels)})"
    )

    # ── Save label encoder ─────────────────────────────────────────────────────
    label_encoder = {"label_to_idx": {lbl: i for i, lbl in enumerate(labels)},
                     "idx_to_label": {str(i): lbl for i, lbl in enumerate(labels)}}
    with open(outdir / "label_encoder.json", "w") as fh:
        json.dump(label_encoder, fh, indent=2)
    log.info(f"Label encoder saved to {outdir / 'label_encoder.json'}")

    # ── Train ──────────────────────────────────────────────────────────────────
    _tracker = None
    if _CODECARBON_AVAILABLE and not args.disable_co2_tracking:
        _tracker = EmissionsTracker(
            project_name="ShoggoTEh_train_classifier",
            output_dir=str(logs_dir),
            log_level="error",
        )
        _tracker.start()
        log.info("  codecarbon tracker started")
    elif args.disable_co2_tracking:
        log.info("  Carbon footprint tracking disabled (--disable_co2_tracking)")
    else:
        log.info("  codecarbon not installed — carbon tracking skipped")

    metrics = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        outdir=outdir,
    )

    emissions_kg = None
    if _tracker is not None:
        try:
            emissions_kg = _tracker.stop()
        except Exception:
            pass

    # ── Save metrics ───────────────────────────────────────────────────────────
    metrics_path = outdir / "training_metrics.tsv"
    pd.DataFrame(metrics).to_csv(metrics_path, sep="\t", index=False)
    log.info(f"Training metrics saved to {metrics_path}")

    # ── Final evaluation on validation set ────────────────────────────────────
    log.info("Loading best model for final evaluation ...")
    model.load_state_dict(torch.load(outdir / "classifier.pt", map_location=device))
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch.to(device))
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_true.extend(y_batch.tolist())

    report = classification_report(
        all_true, all_preds,
        target_names=labels,
        digits=3,
        zero_division=0,
    )
    log.info(f"Validation classification report:\n{report}")

    report_path = outdir / "classification_report.txt"
    report_path.write_text(report)
    log.info(f"Classification report saved to {report_path}")

    if emissions_kg is not None:
        log.info(f"Carbon footprint: {emissions_kg:.6f} kg CO2 equivalent")
    log.info("Done.")

    # ── Resource usage ─────────────────────────────────────────────────────────
    elapsed_s = time.monotonic() - t_start
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_mem_mb = ru.ru_maxrss / (1024 * 1024) if platform.system() == "Darwin" else ru.ru_maxrss / 1024
    log.info(f"Wall-clock time   : {elapsed_s:.1f} s ({elapsed_s/60:.1f} min)")
    log.info(f"Peak memory (RSS) : {peak_mem_mb:.1f} MB")

    # Capture best_val_loss and n_epochs_run from metrics
    best_val_loss = min((m["val_loss"] for m in metrics), default=None)
    n_epochs_run  = len(metrics)

    summary = {
        "date":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION,
        "input_embeddings_dir": str(args.embeddings_dir),
        "n_classes":    len(labels),
        "n_train":      len(X_train),
        "n_val":        len(X_val),
        "best_val_loss": best_val_loss,
        "n_epochs_run": n_epochs_run,
        "parameters": {
            "epochs":       args.epochs,
            "batch_size":   args.batch_size,
            "lr":           args.lr,
            "hidden_dim":   args.hidden_dim,
            "dropout":      args.dropout,
            "val_fraction": args.val_fraction,
            "patience":     args.patience,
            "seed":         args.seed,
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
