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
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from codecarbon import EmissionsTracker
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

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
    X, y, labels = load_embeddings(Path(args.embeddings_dir), args.labels)

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
    tracker = EmissionsTracker(
        project_name="ShoggoTEh_train_classifier",
        output_dir=str(log_dir),
        log_level="error",
    )
    tracker.start()

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

    emissions = tracker.stop()

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

    log.info(f"Carbon footprint: {emissions:.6f} kg CO2 equivalent")
    log.info("Done.")


if __name__ == "__main__":
    main()
