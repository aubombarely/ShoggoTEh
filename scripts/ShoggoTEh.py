#!/usr/bin/env python3
"""
ShoggoTEh.py  —  Deep-learning TE annotation pipeline (Shoggoth + TEh)

Sub-commands
------------
  prepare_dataset        Slide windows across each genome, label every
                          small bin (default 50bp) by the exact repeat/gene
                          interval it falls in (dense, per-bin labeling —
                          no majority-vote window discard).
                          Output: {outdir}/{species}.parquet
  generate_embeddings    Run labeled chunks through Hyena-DNA and pool the
                          per-nucleotide hidden states into per-bin
                          embeddings ([n_bins, hidden_dim] per chunk).
                          Output: {outdir}/{species}.parquet
  train_classifier       Train a 1D-CNN + linear-chain CRF sequence
                          labeler on the per-bin embeddings.
                          Output: {outdir}/classifier.pt, label_encoder.json,
                                  model_config.json
  train_dense_cnn        v2 (see .claude/plans/agile-chasing-willow.md):
                          train a stride-1 dilated-residual CNN + linear-
                          chain CRF end-to-end on raw sequence from a
                          'prepare_dataset --bin_size 1' corpus -- true
                          single-nucleotide resolution, no pretrained
                          backbone, no embedding step.
                          Output: {outdir}/classifier.pt, label_encoder.json,
                                  model_config.json
  predict_dense_cnn      v2: slide overlap-trimmed windows across a new
                          genome, run the dilated CNN+CRF forward pass +
                          Viterbi decode per window at true 1bp resolution,
                          streaming run-length-encode into BED intervals.
                          Output: {outdir}/{prefix}.bed
  predict                Slide windows across a new genome, embed, run the
                          dense CNN+CRF forward pass + Viterbi decode, and
                          run-length-encode consecutive same-label bins into
                          BED intervals.
                          Output: {outdir}/{prefix}.bed,
                                  {outdir}/{prefix}_bin_probs.tsv
  compare_te_annotation  Benchmark predictions against a reference BED
                          annotation (e.g. EarlGrey filteredRepeats.bed).
                          Output: {outdir}/{prefix}_comparison.tsv,
                                  {outdir}/{prefix}_metrics.tsv

Architecture note
------------------
This is the "dense (per-bin)" redesign of ShoggoTEh: instead of
mean-pooling an entire 5kb window into one vector and predicting a single
label for it (which structurally fails for short, dispersed classes like
SINE — see docs/ideas or the project changelog), Hyena-DNA's per-nucleotide
hidden states are pooled into small bins (default 50bp) and every bin gets
its own label and its own prediction. A light 1D-CNN smooths local context
across bins, and a linear-chain CRF (implemented from scratch — transition
matrix + forward algorithm + Viterbi decode) discourages single-bin
"flapping" misclassifications inside otherwise-uniform runs. Consecutive
same-label bins are then run-length-encoded back into BED intervals at
prediction time, so downstream consumers (compare_te_annotation.py,
EarlGrey-style analyses) see no format change.

Usage
-----
  python3 scripts/ShoggoTEh.py prepare_dataset --species_tsv ... --outdir ...
  python3 scripts/ShoggoTEh.py generate_embeddings --chunks_dir ... --outdir ...
  python3 scripts/ShoggoTEh.py train_classifier --embeddings_dir ... --outdir ...
  python3 scripts/ShoggoTEh.py predict --fasta ... --model_dir ... --outdir ...
  python3 scripts/ShoggoTEh.py compare_te_annotation -t ... -r ... --outdir ...
"""

VERSION = "v0.8.0"

import argparse
import getpass
import json
import os
import platform
import resource
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pybedtools
from pyfaidx import Fasta
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

try:
    from codecarbon import EmissionsTracker
    _CODECARBON_AVAILABLE = True
except ImportError:
    _CODECARBON_AVAILABLE = False


# ── Logging ──────────────────────────────────────────────────────────────────

_LOG_FH = None


def _log(msg: str) -> None:
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, file=sys.stderr)
    if _LOG_FH is not None:
        print(line, file=_LOG_FH, flush=True)


def _banner(title: str) -> None:
    bar = "─" * (len(title) + 4)
    _log(f"┌{bar}┐")
    _log(f"│  {title}  │")
    _log(f"└{bar}┘")


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
    _log(f"┌{border}┐")
    for line in _QUOTE_LINES:
        padding = width - len(line) - 1
        _log(f"│ {line}{' ' * padding}│")
    _log(f"└{border}┘")


# ── Checkpoint / external tools ────────────────────────────────────────────────

def _checkpoint(path: Path, label: str, force: bool) -> bool:
    if not force and path.exists() and path.stat().st_size > 0:
        _log(f"  [checkpoint] {label} — {path.name} already exists, skipping")
        return True
    return False


# ── Input validation ──────────────────────────────────────────────────────────

def _validate_inputs(pairs: list) -> None:
    ok = True
    for flag, path in pairs:
        if path is not None and not Path(path).exists():
            print(f"ERROR: {flag} not found: {path}", file=sys.stderr)
            ok = False
    if not ok:
        sys.exit(1)


# ── Log file / run infrastructure ──────────────────────────────────────────────

def _open_log(logs_dir: Path, command: str) -> None:
    """Open the per-subcommand run log. Filename includes the subcommand
    name since ShoggoTEh.py is a single multi-subcommand entry point and
    each subcommand typically writes to its own --outdir anyway."""
    global _LOG_FH
    log_path = logs_dir / f"Run_ShoggoTEh_{command}.log"
    _LOG_FH  = open(log_path, "w")
    sep = "=" * 62
    _LOG_FH.write(f"{sep}\n  ShoggoTEh {VERSION}  —  {command}  —  Run Log\n{sep}\n")
    _LOG_FH.write(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _LOG_FH.write(f"User      : {getpass.getuser()}\n")
    _LOG_FH.write(f"Server    : {platform.node()}\n")
    _LOG_FH.write(f"OS        : {platform.system()} {platform.release()} "
                  f"({platform.machine()})\n")
    _LOG_FH.write(f"Directory : {os.getcwd()}\n")
    _LOG_FH.write(f"Command   : {' '.join(sys.argv)}\n")
    _LOG_FH.write(f"{sep}\n\n")
    _LOG_FH.flush()


def _close_log() -> None:
    global _LOG_FH
    if _LOG_FH is not None:
        _LOG_FH.close()
        _LOG_FH = None


def _peak_mem_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return (ru.ru_maxrss / (1024 * 1024) if platform.system() == "Darwin"
            else ru.ru_maxrss / 1024)


def _start_tracker(project_name: str, logs_dir: Path, disable: bool):
    if disable:
        _log("  Carbon footprint tracking disabled (--disable_co2_tracking)")
        return None
    if not _CODECARBON_AVAILABLE:
        _log("  codecarbon not installed — carbon tracking skipped "
             "(conda install -c conda-forge codecarbon)")
        return None
    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=str(logs_dir),
        log_level="error",
    )
    tracker.start()
    _log("  codecarbon tracker started")
    return tracker


def _stop_tracker(tracker):
    if tracker is None:
        return None
    try:
        return tracker.stop()
    except Exception:
        return None


def _write_summary(outdir: Path, summary: dict) -> Path:
    summary_path = Path(outdir) / "run_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    _log(f"Run summary written to {summary_path}")
    return summary_path


# ── Repeat class mapping (shared across prepare_dataset / compare_te_annotation) ──

REPEAT_CLASS_MAP = {
    "LTR":            "LTR",
    "DNA":            "DNA",
    "LINE":           "LINE",
    "SINE":           "SINE",
    "RC":             "DNA",             # Rolling Circle / Helitron
    "Unknown":        "Unknown_repeat",
    "Satellite":      "Other_repeat",
    "Simple_repeat":  "Other_repeat",
    "Low_complexity": "Other_repeat",
}

# Full label vocabulary shared by train_classifier (default) and
# compare_te_annotation (reference-label normalisation target).
DEFAULT_LABELS = ["LTR", "DNA", "LINE", "SINE", "Unknown_repeat",
                  "Other_repeat", "Genic", "Intergenic"]


def map_repeat_class(classification: str) -> str:
    top = str(classification).split("/")[0]
    return REPEAT_CLASS_MAP.get(top, "Other_repeat")


# Nucleotide encoding for the v2 end-to-end dense CNN (train_dense_cnn /
# predict_dense_cnn): raw sequence -> int64 indices, no pretrained backbone.
_NT_LOOKUP = np.full(256, 4, dtype=np.int64)  # default: N (or any other IUPAC code)
for _i, _c in enumerate("ACGT"):
    _NT_LOOKUP[ord(_c)] = _i
del _i, _c


def encode_sequence(seq: str) -> np.ndarray:
    """Vectorized encode of an uppercase nucleotide string to int64 indices
    (A=0, C=1, G=2, T=3, everything else incl. N/ambiguity codes=4) via a
    256-entry ASCII lookup table -- avoids a per-character Python loop,
    which would be prohibitively slow at training-corpus scale (millions
    of positions)."""
    raw = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return _NT_LOOKUP[raw]


# ── Windowing (shared: prepare_dataset + predict) ──────────────────────────────

def make_windows(fasta: Fasta, chunk_size: int, stride: int) -> list:
    """Generate sliding windows across all scaffolds long enough."""
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


###############################################################################
# prepare_dataset
###############################################################################

def load_repeats(bed_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        bed_path, sep="\t", header=None, usecols=[0, 1, 2, 3],
        names=["chrom", "start", "end", "classification"],
        dtype={"chrom": str, "start": int, "end": int, "classification": str},
    )
    df["label"] = df["classification"].apply(map_repeat_class)
    return df


def _check_chroms(species_id: str, fasta: Fasta,
                  repeats_df: pd.DataFrame, genes_df) -> None:
    fasta_chroms = set(fasta.keys())

    bed_chroms = set(repeats_df["chrom"].unique())
    shared_bed = fasta_chroms & bed_chroms
    if not shared_bed:
        raise ValueError(
            f"No chromosome names in common between FASTA and BED.\n"
            f"  FASTA examples : {sorted(fasta_chroms)[:5]}\n"
            f"  BED examples   : {sorted(bed_chroms)[:5]}\n"
            f"Check that both files use the same naming convention (e.g. 'Chr1' vs '1')."
        )
    unmatched_bed = bed_chroms - fasta_chroms
    if unmatched_bed:
        _log(f"  WARNING {species_id}: {len(unmatched_bed)} BED chromosome(s) "
             f"not found in FASTA (e.g. {sorted(unmatched_bed)[:3]}) — ignored.")

    if genes_df is not None:
        gff_chroms = set(genes_df["chrom"].unique())
        shared_gff = fasta_chroms & gff_chroms
        if not shared_gff:
            raise ValueError(
                f"No chromosome names in common between FASTA and GFF3.\n"
                f"  FASTA examples : {sorted(fasta_chroms)[:5]}\n"
                f"  GFF3 examples  : {sorted(gff_chroms)[:5]}\n"
            )
        unmatched_gff = gff_chroms - fasta_chroms
        if unmatched_gff:
            _log(f"  WARNING {species_id}: {len(unmatched_gff)} GFF3 chromosome(s) "
                 f"not found in FASTA (e.g. {sorted(unmatched_gff)[:3]}) — ignored.")


def label_bins(windows: list, bin_size: int, chunk_size: int,
               repeats_df: pd.DataFrame, genes_df) -> tuple:
    """
    Dense per-bin labeling. Every bin (default 50bp) is assigned the label
    of whichever repeat interval it overlaps most (no window-level
    min_fraction threshold — the majority-vote/mixed-window-discard logic
    of the old window-level label_windows() is retired). Bins with no
    repeat overlap are refined into Genic/Intergenic by gene-model overlap
    dominance within the bin.

    Returns (window_bin_labels, n_bins) where window_bin_labels maps
    (chrom, start, end) -> list[str] of length n_bins.
    """
    if chunk_size % bin_size != 0:
        raise ValueError(
            f"--chunk_size ({chunk_size}) must be a multiple of --bin_size "
            f"({bin_size}) so bins tile each window exactly."
        )
    n_bins = chunk_size // bin_size

    bin_rows = []  # (chrom, start, end, uid)
    uid = 0
    for chrom, start, end in windows:
        for i in range(n_bins):
            bstart = start + i * bin_size
            bend = bstart + bin_size
            bin_rows.append((chrom, bstart, bend, uid))
            uid += 1
    n_total_bins = uid

    bins_df = pd.DataFrame(bin_rows, columns=["chrom", "start", "end", "uid"])
    bins_bed = pybedtools.BedTool.from_dataframe(bins_df)
    rep_bed = pybedtools.BedTool.from_dataframe(
        repeats_df[["chrom", "start", "end", "label"]]
    )

    intersect = bins_bed.intersect(rep_bed, wao=True)
    overlaps: dict = {}
    for feat in intersect:
        bin_uid   = int(feat.fields[3])
        rep_class = feat.fields[7]
        bp        = int(feat.fields[8])
        if rep_class != "." and bp > 0:
            overlaps.setdefault(bin_uid, {})
            overlaps[bin_uid][rep_class] = overlaps[bin_uid].get(rep_class, 0) + bp

    bin_labels = ["Non_repeat"] * n_total_bins
    for u, ovlp in overlaps.items():
        bin_labels[u] = max(ovlp, key=ovlp.get)

    non_rep_uids = [u for u in range(n_total_bins) if bin_labels[u] == "Non_repeat"]
    if genes_df is not None and non_rep_uids:
        nr_rows = [bin_rows[u] for u in non_rep_uids]
        nr_df   = pd.DataFrame(nr_rows, columns=["chrom", "start", "end", "uid"])
        nr_bed  = pybedtools.BedTool.from_dataframe(nr_df)
        gene_bed = pybedtools.BedTool.from_dataframe(genes_df[["chrom", "start", "end"]])
        gene_intersect = nr_bed.intersect(gene_bed, wao=True)
        gene_bp: dict = {}
        for feat in gene_intersect:
            bin_uid = int(feat.fields[3])
            bp = int(feat.fields[-1])
            if bp > 0:
                gene_bp[bin_uid] = gene_bp.get(bin_uid, 0) + bp
        for u in non_rep_uids:
            bstart, bend = bin_rows[u][1], bin_rows[u][2]
            bin_len = bend - bstart
            bin_labels[u] = "Genic" if gene_bp.get(u, 0) >= bin_len / 2 else "Intergenic"
    else:
        for u in non_rep_uids:
            bin_labels[u] = "Intergenic"

    window_bin_labels = {}
    idx = 0
    for w in windows:
        window_bin_labels[tuple(w)] = bin_labels[idx: idx + n_bins]
        idx += n_bins

    pybedtools.cleanup()
    return window_bin_labels, n_bins


def _paint_chrom_labels(chrom_len: int, repeats_chrom: pd.DataFrame,
                        genes_chrom, label_to_idx: dict) -> np.ndarray:
    """Rasterize repeat + gene intervals for one chromosome into a per-base
    int8 label array via vectorized slice-assignment. label_bins()'s
    bedtools per-bin-row intersect needs one row per bin -- at bin_size=1
    that's one row per base (a single 5000bp window alone would need 5000
    rows; genome-wide, billions), so this paints directly into a numpy
    array instead. Precedence matches the project convention used
    elsewhere (repeat > genic > intergenic, see assign_reference_labels()):
    genes are painted first, then repeats overwrite them. Overlapping
    repeat intervals resolve by BED file order (last write wins) -- a
    reasonable approximation given real repeat annotations rarely have
    deeply nested overlaps at the same base."""
    intergenic_idx = label_to_idx["Intergenic"]
    labels = np.full(chrom_len, intergenic_idx, dtype=np.int8)

    if genes_chrom is not None and len(genes_chrom):
        genic_idx = label_to_idx["Genic"]
        for s, e in zip(genes_chrom["start"].to_numpy(), genes_chrom["end"].to_numpy()):
            labels[max(0, s):min(chrom_len, e)] = genic_idx

    for s, e, lbl in zip(repeats_chrom["start"].to_numpy(),
                         repeats_chrom["end"].to_numpy(),
                         repeats_chrom["label"].to_numpy()):
        labels[max(0, s):min(chrom_len, e)] = label_to_idx[lbl]

    return labels


def label_bases_dense(windows: list, repeats_df: pd.DataFrame, genes_df,
                      labels: list) -> dict:
    """Per-base (bin_size=1) labeling: paint each chromosome once into a
    dense int8 label array, then slice out each window's bases from it --
    replaces label_bins()'s per-bin bedtools intersect, which does not
    scale to 1bp resolution (see _paint_chrom_labels docstring). Processes
    one chromosome at a time and copies out each window's slice before
    discarding the full painted array, so memory is bounded by one
    chromosome at a time rather than the whole genome. Returns
    {(chrom, start, end): np.ndarray(int8)}."""
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    windows_by_chrom: dict = {}
    for w in windows:
        windows_by_chrom.setdefault(w[0], []).append(w)

    rep_groups  = {c: g for c, g in repeats_df.groupby("chrom")}
    gene_groups = {c: g for c, g in genes_df.groupby("chrom")} if genes_df is not None else {}

    window_base_labels = {}
    for chrom, chrom_windows in windows_by_chrom.items():
        chrom_len = max(w[2] for w in chrom_windows)
        rep_chrom   = rep_groups.get(chrom, repeats_df.iloc[0:0])
        genes_chrom = gene_groups.get(chrom) if genes_df is not None else None
        painted = _paint_chrom_labels(chrom_len, rep_chrom, genes_chrom, label_to_idx)
        for w in chrom_windows:
            _, start, end = w
            window_base_labels[w] = painted[start:end].copy()
        del painted

    return window_base_labels


def process_species_prepare(species_id: str, fasta_path: str, bed_path: str,
                            gff3_path, chunk_size: int, stride: int,
                            bin_size: int, max_n_fraction: float,
                            outdir: Path, force: bool) -> int:
    out_path = outdir / f"{species_id}.parquet"
    if _checkpoint(out_path, f"prepare_dataset:{species_id}", force):
        return 0

    _log(f"{species_id}: loading genome")
    fasta = Fasta(fasta_path)

    _log(f"{species_id}: generating windows (chunk={chunk_size}, stride={stride})")
    windows = make_windows(fasta, chunk_size, stride)
    _log(f"{species_id}: {len(windows):,} windows generated")

    _log(f"{species_id}: loading repeat annotations")
    repeats_df = load_repeats(bed_path)

    genes_df = None
    if gff3_path and Path(gff3_path).exists():
        _log(f"{species_id}: loading gene annotations")
        gff = pd.read_csv(
            gff3_path, sep="\t", comment="#", header=None,
            names=["chrom", "source", "feature", "start", "end",
                   "score", "strand", "frame", "attributes"],
            dtype={"chrom": str},
        )
        genes_df = gff[gff["feature"] == "gene"][["chrom", "start", "end"]].copy()
        genes_df["start"] = genes_df["start"] - 1  # GFF3 is 1-based

    _check_chroms(species_id, fasta, repeats_df, genes_df)

    dense_1bp = (bin_size == 1)
    if dense_1bp:
        # True single-nucleotide labeling for the v2 end-to-end dense CNN
        # path (see docs/plans/agile-chasing-willow.md) -- fixed label
        # vocabulary (DEFAULT_LABELS) encoded as int8, stored as raw bytes
        # (base_labels_bytes) rather than a Python list of label strings:
        # a naive list<str> column at 1bp resolution would need one string
        # object per base per window (e.g. 5000 per window), which is the
        # same class of Python-object-boxing memory blowup already fixed
        # once this session for embeddings (see embedding_bytes).
        _log(f"{species_id}: labeling bases (bin_size=1, dense per-chromosome painting)")
        label_to_idx = {lbl: i for i, lbl in enumerate(DEFAULT_LABELS)}
        window_base_labels = label_bases_dense(windows, repeats_df, genes_df, DEFAULT_LABELS)
        n_bins = chunk_size
        genic_idx, intergenic_idx = label_to_idx["Genic"], label_to_idx["Intergenic"]
        bincount_total = np.zeros(len(DEFAULT_LABELS), dtype=np.int64)
    else:
        _log(f"{species_id}: labeling bins (bin_size={bin_size})")
        window_bin_labels, n_bins = label_bins(windows, bin_size, chunk_size,
                                               repeats_df, genes_df)

    _log(f"{species_id}: extracting sequences")
    records = []
    skipped_n = 0
    for win in windows:
        chrom, start, end = win
        seq = str(fasta[chrom][start:end]).upper()
        n_frac = seq.count("N") / len(seq)
        if n_frac > max_n_fraction:
            skipped_n += 1
            continue

        if dense_1bp:
            base_labels = window_base_labels[win]
            non_context = (base_labels != genic_idx) & (base_labels != intergenic_idx)
            repeat_fraction = float(non_context.mean())
            bincount_total += np.bincount(base_labels, minlength=len(DEFAULT_LABELS))
            records.append({
                "species":           species_id,
                "chrom":             chrom,
                "start":             start,
                "end":               end,
                "sequence":          seq,
                "base_labels_bytes": base_labels.astype(np.int8).tobytes(),
                "n_bins":            n_bins,
                "bin_size":          bin_size,
                "repeat_fraction":   round(repeat_fraction, 4),
            })
        else:
            bin_labels = window_bin_labels[tuple(win)]
            repeat_fraction = sum(1 for l in bin_labels if l not in ("Genic", "Intergenic")) / n_bins
            records.append({
                "species":         species_id,
                "chrom":           chrom,
                "start":           start,
                "end":             end,
                "sequence":        seq,
                "bin_labels":      bin_labels,
                "n_bins":          n_bins,
                "bin_size":        bin_size,
                "repeat_fraction": round(repeat_fraction, 4),
            })

    _log(f"{species_id}: {len(records):,} chunks kept | "
         f"{skipped_n:,} skipped (N content)")

    df = pd.DataFrame(records)
    if len(df):
        if dense_1bp:
            dist = {lbl: int(c) for lbl, c in zip(DEFAULT_LABELS, bincount_total)}
            _log(f"{species_id}: base label distribution — {dist}")
        else:
            all_bin_labels = [l for row in df["bin_labels"] for l in row]
            dist = Counter(all_bin_labels)
            _log(f"{species_id}: bin label distribution — {dict(dist)}")
        df = df.sort_values(["chrom", "start"]).reset_index(drop=True)

    df.to_parquet(out_path, index=False)
    _log(f"{species_id}: written to {out_path}")

    pybedtools.cleanup()
    return len(records)


def run_prepare_dataset(args) -> None:
    args.species_tsv = args.species_tsv.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _open_log(logs_dir, "prepare_dataset")

    _validate_inputs([("--species_tsv", args.species_tsv)])

    if args.chunk_size % args.bin_size != 0:
        print(f"ERROR: --chunk_size ({args.chunk_size}) must be a multiple of "
              f"--bin_size ({args.bin_size})", file=sys.stderr)
        sys.exit(1)

    stride = int(args.chunk_size * (1 - args.overlap))

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  species_tsv    : {args.species_tsv}")
        _log(f"  outdir         : {outdir}/")
        _log(f"  chunk_size     : {args.chunk_size}")
        _log(f"  bin_size       : {args.bin_size}")
        _log(f"  overlap        : {args.overlap}")
        _log(f"  max_n_fraction : {args.max_n_fraction}")
        _log(f"  force          : {args.force}")
        _log("  Steps that would run: process each species from species_tsv "
             "(dense per-bin labeling)")
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    if args.force:
        _log("--force set: all species will be reprocessed regardless of "
             "existing outputs")
    elif outdir.exists() and any(outdir.iterdir()):
        _log("Existing outdir found — resuming from checkpoints "
             "(use --force to rerun all species from scratch)")

    species_df = pd.read_csv(
        args.species_tsv, sep="\t", comment="#", header=None,
        names=["species_id", "fasta", "bed", "gff3"],
    )

    t_start = time.monotonic()
    tracker = _start_tracker("ShoggoTEh_prepare_dataset", logs_dir, args.disable_co2_tracking)

    n_species_processed = 0
    total_records = 0
    for _, row in species_df.iterrows():
        gff3 = row.get("gff3")
        gff3 = None if pd.isna(gff3) or str(gff3).strip().upper() == "NA" else str(gff3)
        try:
            n_records = process_species_prepare(
                species_id=str(row["species_id"]),
                fasta_path=str(row["fasta"]),
                bed_path=str(row["bed"]),
                gff3_path=gff3,
                chunk_size=args.chunk_size,
                stride=stride,
                bin_size=args.bin_size,
                max_n_fraction=args.max_n_fraction,
                outdir=outdir,
                force=args.force,
            )
            n_species_processed += 1
            total_records += n_records
        except Exception as exc:
            _log(f"  ERROR {row['species_id']}: FAILED — {exc}")
            continue

    emissions_kg = _stop_tracker(tracker)
    elapsed_s   = time.monotonic() - t_start
    peak_mem_mb = _peak_mem_mb()
    _log(f"Wall-clock time   : {elapsed_s:.1f} s ({elapsed_s/60:.1f} min)")
    _log(f"Peak memory (RSS) : {peak_mem_mb:.1f} MB")

    summary = {
        "date":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION,
        "input_species_tsv":    str(args.species_tsv),
        "n_species_processed":  n_species_processed,
        "total_records_written": total_records,
        "parameters": {
            "chunk_size":     args.chunk_size,
            "bin_size":       args.bin_size,
            "overlap":        args.overlap,
            "max_n_fraction": args.max_n_fraction,
        },
        "resource_usage": {
            "wall_clock_s":       round(elapsed_s, 1),
            "peak_mem_mb":        round(peak_mem_mb, 1),
            "emissions_kg_CO2eq": emissions_kg,
        },
    }
    _write_summary(outdir, summary)
    _close_log()


###############################################################################
# generate_embeddings
###############################################################################

def load_hyena_backbone(model_name: str, device):
    """AutoModel/AutoTokenizer, single-nucleotide tokenization (1 token/bp)
    plus a trailing EOS token. Hidden states via out.last_hidden_state."""
    from transformers import AutoModel, AutoTokenizer
    _log(f"Loading tokenizer and model (hyena): {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    _log(f"Model ready on {device} ({n_params:.1f}M parameters)")
    return tokenizer, model


def load_nucleotide_transformer_backbone(model_name: str, device):
    """Standard AutoModel/AutoTokenizer, 6-mer tokenization (each token =
    6bp, with standalone single-nucleotide fallback tokens for
    non-multiple-of-6 remainders) plus a leading CLS token. Hidden states
    via out.last_hidden_state (same pattern as Hyena-DNA)."""
    from transformers import AutoModel, AutoTokenizer
    _log(f"Loading tokenizer and model (nucleotide_transformer): {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    _log(f"Model ready on {device} ({n_params:.1f}M parameters)")
    return tokenizer, model


def load_plantcaduceus_backbone(model_name: str, device):
    """AutoModelForMaskedLM (NOT plain AutoModel) + AutoTokenizer, both with
    trust_remote_code=True. Hidden states are NOT exposed via
    .last_hidden_state on this wrapper -- must call with
    output_hidden_states=True and read out.hidden_states[-1] instead.
    Tokenization scheme (single-nucleotide vs k-mer) is handled generically
    by the dynamic tokens-per-bp logic in embed_sequences_dense(), not
    assumed here."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    _log(f"Loading tokenizer and model (plantcaduceus): {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    _log(f"Model ready on {device} ({n_params:.1f}M parameters)")
    return tokenizer, model


def _extract_last_hidden_state(out):
    return out.last_hidden_state


def _extract_last_layer_hidden_states(out):
    return out.hidden_states[-1]


# Backbone registry: name -> {default_model, load_fn, hidden_state_fn,
# forward_kwargs}. To add a 4th backbone: write a load_<name>_backbone(name,
# device) -> (tokenizer, model) function, pick the right hidden_state_fn
# (out.last_hidden_state for a plain AutoModel, or
# _extract_last_layer_hidden_states + forward_kwargs={"output_hidden_states":
# True} for an AutoModelForMaskedLM-style wrapper), and add one entry below.
# embed_sequences_dense() needs no per-backbone changes -- tokens-per-bp is
# always computed dynamically at runtime, never hardcoded per backbone.
BACKBONE_REGISTRY = {
    "hyena": {
        "default_model":   "LongSafari/hyenadna-medium-160k-seqlen-hf",
        "load_fn":         load_hyena_backbone,
        "hidden_state_fn": _extract_last_hidden_state,
        "forward_kwargs":  {},
    },
    "nucleotide_transformer": {
        "default_model":   "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        "load_fn":         load_nucleotide_transformer_backbone,
        "hidden_state_fn": _extract_last_hidden_state,
        "forward_kwargs":  {},
    },
    "plantcaduceus": {
        "default_model":   "kuleshov-group/PlantCaduceus_l20",
        "load_fn":         load_plantcaduceus_backbone,
        "hidden_state_fn": _extract_last_layer_hidden_states,
        "forward_kwargs":  {"output_hidden_states": True},
    },
}


def resolve_backbone(backbone: str, backbone_model) -> tuple:
    """Resolve a --backbone name + optional --backbone_model override into
    (backbone, model_name), validating the backbone name and falling back
    to that backbone's own sensible default checkpoint when no override is
    given."""
    if backbone not in BACKBONE_REGISTRY:
        print(f"ERROR: unknown --backbone '{backbone}'. Choices: "
              f"{sorted(BACKBONE_REGISTRY)}", file=sys.stderr)
        sys.exit(1)
    model_name = backbone_model or BACKBONE_REGISTRY[backbone]["default_model"]
    return backbone, model_name


def load_backbone(backbone: str, model_name: str, device):
    entry = BACKBONE_REGISTRY[backbone]
    return entry["load_fn"](model_name, device)


def _pool_bins_from_hidden(content_hidden, n_bins: int, bin_size: int,
                           tokens_per_bp: float):
    """Mean-pool a (content_length, D) slice of hidden states into
    (n_bins, D) using a dynamically-computed tokens_per_bp ratio (rather
    than a hardcoded per-backbone token/bp assumption). Bin token
    boundaries are recomputed from scratch at each bin edge
    (round(i * bin_size * tokens_per_bp)) instead of accumulating a fixed
    per-bin token count, so rounding error cannot drift across bins. The
    last bin absorbs any remainder up to the actual content length. Bins
    that end up with zero tokens (e.g. a short trailing chunk under a
    high-tokens-per-bp backbone) fall back to the previous bin's pooled
    vector, or an all-zero vector for the very first bin."""
    import torch

    total_tokens = content_hidden.shape[0]
    hidden_dim   = content_hidden.shape[1]
    pooled = torch.zeros(n_bins, hidden_dim, dtype=content_hidden.dtype,
                         device=content_hidden.device)

    prev_end = 0
    for i in range(n_bins):
        end_tok = min(round((i + 1) * bin_size * tokens_per_bp), total_tokens)
        start_tok = min(prev_end, total_tokens)
        if end_tok > start_tok:
            pooled[i] = content_hidden[start_tok:end_tok].mean(dim=0)
        elif i > 0:
            pooled[i] = pooled[i - 1]
        # else: leave as zeros (no tokens at all for this sequence's first bin)
        prev_end = end_tok

    return pooled


def embed_sequences_dense(sequences: list, n_bins: int, bin_size: int,
                          tokenizer, model, device, batch_size: int,
                          hidden_state_fn=_extract_last_hidden_state,
                          forward_kwargs: dict = None) -> list:
    """
    Run sequences through a frozen embedding backbone and pool the
    per-token hidden states into per-bin embeddings. Returns a list of
    np.ndarray, each of shape (n_bins, hidden_dim), one per input sequence.

    Generalises the original Hyena-DNA-only implementation (1 token/bp +
    trailing EOS) to any tokenization scheme: the effective tokens-per-bp
    ratio is computed dynamically per sequence as
    valid_content_token_len / len(raw_sequence_bp), after generically
    dropping whichever of CLS/BOS (leading) and EOS/SEP (trailing) special
    tokens the tokenizer added (checked via tokenizer.cls_token_id /
    bos_token_id / eos_token_id / sep_token_id, not hardcoded to
    EOS-only). Bin pooling then uses bin_size * tokens_per_bp tokens per
    bin (see _pool_bins_from_hidden). For Hyena-DNA (no CLS, 1 token/bp,
    trailing EOS) this reduces to the original reshape-into-bins behaviour.
    """
    import torch

    forward_kwargs = forward_kwargs or {}

    cls_id = getattr(tokenizer, "cls_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    special_ids = {
        cls_id, bos_id,
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "sep_token_id", None),
    }
    special_ids.discard(None)

    results = []
    n = len(sequences)

    for batch_start in range(0, n, batch_size):
        batch_seqs = sequences[batch_start: batch_start + batch_size]
        enc = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, **forward_kwargs)
        hidden = hidden_state_fn(out)  # (B, L, D)
        attn = enc.get("attention_mask")
        input_ids = enc["input_ids"]

        for i in range(len(batch_seqs)):
            valid_len = int(attn[i].sum().item()) if attn is not None else hidden.shape[1]
            ids_row = input_ids[i][:valid_len].tolist()

            n_special = sum(1 for tid in ids_row if tid in special_ids)
            leading_offset = 1 if (ids_row and ids_row[0] in (cls_id, bos_id)
                                   and ids_row[0] is not None) else 0
            content_len = max(valid_len - n_special, 0)

            raw_len = len(batch_seqs[i])
            tokens_per_bp = content_len / raw_len if raw_len > 0 else 0.0

            content_hidden = hidden[i, leading_offset: leading_offset + content_len, :]
            pooled = _pool_bins_from_hidden(content_hidden, n_bins, bin_size, tokens_per_bp)
            results.append(pooled.cpu().float().numpy())

        done = min(batch_start + batch_size, n)
        if done % (batch_size * 10) == 0 or done == n:
            _log(f"  {done:,} / {n:,} sequences embedded")

    return results


def process_species_embed(chunks_path: Path, outdir: Path, tokenizer, model,
                          device, bin_size: int, batch_size: int,
                          backbone: str, backbone_model: str,
                          hidden_state_fn, forward_kwargs: dict,
                          force: bool) -> int:
    species_id = chunks_path.stem
    out_path   = outdir / f"{species_id}.parquet"
    if _checkpoint(out_path, f"generate_embeddings:{species_id}", force):
        return 0

    _log(f"{species_id}: loading chunks from {chunks_path}")
    df = pd.read_parquet(chunks_path)
    _log(f"{species_id}: {len(df):,} chunks to embed")

    n_bins_vals = df["n_bins"].unique()
    if len(n_bins_vals) > 1:
        raise ValueError(f"{species_id}: inconsistent n_bins values in chunks "
                         f"file: {n_bins_vals}")
    n_bins = int(n_bins_vals[0])

    bin_size_vals = df["bin_size"].unique()
    if len(bin_size_vals) > 1 or int(bin_size_vals[0]) != bin_size:
        raise ValueError(f"{species_id}: chunk bin_size ({bin_size_vals}) does not "
                         f"match --bin_size ({bin_size})")

    sequences = df["sequence"].tolist()
    _log(f"{species_id}: generating dense per-bin embeddings "
         f"(backbone={backbone}, model={backbone_model}, n_bins={n_bins}, "
         f"bin_size={bin_size}, batch_size={batch_size})")
    pooled_list = embed_sequences_dense(sequences, n_bins, bin_size,
                                        tokenizer, model, device, batch_size,
                                        hidden_state_fn=hidden_state_fn,
                                        forward_kwargs=forward_kwargs)
    hidden_dim = pooled_list[0].shape[1]
    _log(f"{species_id}: hidden dim = {hidden_dim}")

    df = df.drop(columns=["sequence"])
    # Store as raw float32 bytes rather than Python float lists -- .tolist()
    # boxes every value into an individual Python float object (~28 bytes
    # each), which at genome scale (hundreds of millions of bin embeddings)
    # inflates RAM usage far beyond the raw array size and can OOM-kill the
    # process during DataFrame construction. Raw bytes carry none of that
    # per-element overhead.
    df["embedding_bytes"] = [p.astype(np.float32).tobytes() for p in pooled_list]
    df["hidden_dim"] = hidden_dim
    # Record which backbone + exact checkpoint produced these embeddings so
    # train_classifier/predict can detect a mixed-backbone corpus or a
    # stale --backbone flag instead of silently mixing incompatible
    # embedding spaces (see model_config.json backbone-mismatch handling).
    df["backbone"] = backbone
    df["backbone_model"] = backbone_model

    df.to_parquet(out_path, index=False)
    _log(f"{species_id}: written to {out_path}")
    return hidden_dim


def run_generate_embeddings(args) -> None:
    chunks_dir = args.chunks_dir.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _open_log(logs_dir, "generate_embeddings")

    _validate_inputs([("--chunks_dir", chunks_dir)])

    chunk_files = sorted(chunks_dir.glob("*.parquet"))
    if not chunk_files:
        print(f"ERROR: no Parquet files found in {chunks_dir}", file=sys.stderr)
        sys.exit(1)
    _log(f"Found {len(chunk_files)} species to embed: {[f.stem for f in chunk_files]}")

    backbone, backbone_model = resolve_backbone(args.backbone, args.backbone_model)

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  chunks_dir     : {chunks_dir}/")
        _log(f"  outdir         : {outdir}/")
        _log(f"  backbone       : {backbone}")
        _log(f"  backbone_model : {backbone_model}")
        _log(f"  bin_size       : {args.bin_size}")
        _log(f"  batch_size     : {args.batch_size}")
        _log(f"  force          : {args.force}")
        _log(f"  Species that would be embedded: {[f.stem for f in chunk_files]}")
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    if args.force:
        _log("--force set: all species will be reprocessed regardless of "
             "existing outputs")
    elif outdir.exists() and any(outdir.iterdir()):
        _log("Existing outdir found — resuming from checkpoints "
             "(use --force to rerun all species from scratch)")

    import torch
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    _log(f"Using device: {device}")

    tokenizer, model = load_backbone(backbone, backbone_model, device)
    registry_entry = BACKBONE_REGISTRY[backbone]

    t_start = time.monotonic()
    tracker = _start_tracker("ShoggoTEh_generate_embeddings", logs_dir, args.disable_co2_tracking)

    n_species_processed = 0
    hidden_dim = None
    for chunk_path in chunk_files:
        try:
            dim = process_species_embed(
                chunks_path=chunk_path, outdir=outdir,
                tokenizer=tokenizer, model=model, device=device,
                bin_size=args.bin_size, batch_size=args.batch_size,
                backbone=backbone, backbone_model=backbone_model,
                hidden_state_fn=registry_entry["hidden_state_fn"],
                forward_kwargs=registry_entry["forward_kwargs"],
                force=args.force,
            )
            if dim:
                n_species_processed += 1
                hidden_dim = dim
        except Exception as exc:
            _log(f"  ERROR {chunk_path.stem}: FAILED — {exc}")
            continue

    emissions_kg = _stop_tracker(tracker)
    elapsed_s   = time.monotonic() - t_start
    peak_mem_mb = _peak_mem_mb()
    _log(f"Wall-clock time   : {elapsed_s:.1f} s ({elapsed_s/60:.1f} min)")
    _log(f"Peak memory (RSS) : {peak_mem_mb:.1f} MB")

    summary = {
        "date":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION,
        "input_chunks_dir":    str(chunks_dir),
        "n_species_processed": n_species_processed,
        "hidden_dim":          hidden_dim,
        "parameters": {
            "backbone":       backbone,
            "backbone_model": backbone_model,
            "bin_size":       args.bin_size,
            "batch_size":     args.batch_size,
            "device":         str(device),
        },
        "resource_usage": {
            "wall_clock_s":       round(elapsed_s, 1),
            "peak_mem_mb":        round(peak_mem_mb, 1),
            "emissions_kg_CO2eq": emissions_kg,
        },
    }
    _write_summary(outdir, summary)
    _close_log()


###############################################################################
# train_classifier — dense CNN + linear-chain CRF
###############################################################################

def _build_dense_model_classes():
    """Import torch lazily and define the model + CRF classes. Returns
    (torch, nn, DenseTEClassifier, LinearChainCRF)."""
    import torch
    import torch.nn as nn

    class DenseTEClassifier(nn.Module):
        """Light 1D-CNN local-context smoothing + linear per-bin head on
        top of frozen per-bin Hyena-DNA embeddings."""

        def __init__(self, input_dim: int, cnn_channels: int, kernel_size: int,
                    n_classes: int, dropout: float):
            super().__init__()
            self.conv = nn.Conv1d(input_dim, cnn_channels, kernel_size,
                                  padding=kernel_size // 2)
            self.act = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(cnn_channels, n_classes)

        def forward(self, x):
            # x: (B, n_bins, input_dim)
            x = x.transpose(1, 2)              # (B, input_dim, n_bins)
            x = self.act(self.conv(x))
            x = self.dropout(x)
            x = x.transpose(1, 2)              # (B, n_bins, cnn_channels)
            return self.classifier(x)          # (B, n_bins, n_classes) — emissions

    class LinearChainCRF(nn.Module):
        """Linear-chain CRF implemented from scratch (transition matrix +
        forward algorithm for the partition function + Viterbi decode).
        Avoids adding a new pip dependency (e.g. pytorch-crf) since
        envs/shoggoTEh.yaml deliberately keeps a minimal dependency set."""

        def __init__(self, n_classes: int):
            super().__init__()
            self.n_classes = n_classes
            self.transitions       = nn.Parameter(torch.randn(n_classes, n_classes) * 0.01)
            self.start_transitions = nn.Parameter(torch.randn(n_classes) * 0.01)
            self.end_transitions   = nn.Parameter(torch.randn(n_classes) * 0.01)

        def _score(self, emissions, tags):
            B, T, _ = emissions.shape
            idx = torch.arange(B, device=emissions.device)
            score = self.start_transitions[tags[:, 0]] + emissions[idx, 0, tags[:, 0]]
            for t in range(1, T):
                score = (score + self.transitions[tags[:, t - 1], tags[:, t]]
                         + emissions[idx, t, tags[:, t]])
            score = score + self.end_transitions[tags[:, -1]]
            return score

        def _forward_alg(self, emissions):
            B, T, _ = emissions.shape
            alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
            for t in range(1, T):
                broadcast = (alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
                            + emissions[:, t].unsqueeze(1))
                alpha = torch.logsumexp(broadcast, dim=1)
            alpha = alpha + self.end_transitions.unsqueeze(0)
            return torch.logsumexp(alpha, dim=1)

        def neg_log_likelihood(self, emissions, tags, sample_weights=None):
            """sample_weights (optional, shape (B,)): per-sequence NLL
            reweighting used to port --class_weight balanced from
            per-window to per-bin frequency (mean weight of the gold bin
            tags in that sequence)."""
            gold = self._score(emissions, tags)
            logZ = self._forward_alg(emissions)
            nll = logZ - gold
            if sample_weights is not None:
                nll = nll * sample_weights
            return nll.mean()

        def decode(self, emissions):
            """Viterbi decode. Returns a list (len B) of lists (len T) of
            predicted class indices.

            The backtrace is vectorized over the batch dimension via numpy
            fancy indexing after a single .cpu().numpy() transfer of the
            backpointers and final tags, rather than looping in Python with
            per-element GPU-tensor indexing (tag = int(bp[b, tag])). Each
            such scalar extraction forces a synchronous GPU->CPU round trip;
            doing that B*T times per call (previously called on every
            training AND validation batch) was the dominant cost of
            training, far exceeding the actual forward/backward pass -- the
            GPU was mostly idle waiting on these tiny synchronous transfers,
            not compute-bound.
            """
            B, T, _ = emissions.shape
            backpointers = []
            score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
            for t in range(1, T):
                broadcast = score.unsqueeze(2) + self.transitions.unsqueeze(0)
                best_score, best_idx = broadcast.max(dim=1)
                score = best_score + emissions[:, t]
                backpointers.append(best_idx)
            score = score + self.end_transitions.unsqueeze(0)
            best_final_tag = score.argmax(dim=1)

            best_paths_arr = np.zeros((B, T), dtype=np.int64)
            best_paths_arr[:, T - 1] = best_final_tag.cpu().numpy()
            if backpointers:
                bp_stack = torch.stack(backpointers, dim=0).cpu().numpy()  # (T-1, B, C)
                batch_idx = np.arange(B)
                for i in range(T - 2, -1, -1):
                    best_paths_arr[:, i] = bp_stack[i, batch_idx, best_paths_arr[:, i + 1]]
            return best_paths_arr.tolist()

    return torch, nn, DenseTEClassifier, LinearChainCRF


###############################################################################
# train_dense_cnn -- v2 end-to-end dense CNN+CRF on raw sequence (no
# pretrained backbone). See .claude/plans/agile-chasing-willow.md.
###############################################################################

def _build_dilated_cnn_model():
    """Import torch lazily and define the v2 model: a stride-1 dilated
    residual CNN tower operating directly on raw nucleotide sequence (no
    pretrained backbone, no pooling -- every input base keeps its own
    position-aligned feature vector throughout the network). Reuses
    LinearChainCRF unchanged from _build_dense_model_classes() rather than
    redefining it -- it is architecture-agnostic (only needs per-position
    emissions of shape (B, T, C)) and was already validated this session
    (200-trial randomized equivalence test for the vectorized Viterbi
    decode). Returns (torch, nn, DilatedResidualCNN, LinearChainCRF)."""
    torch, nn, _DenseTEClassifier, LinearChainCRF = _build_dense_model_classes()

    class ResidualDilatedBlock(nn.Module):
        """Conv1d -> GroupNorm -> GELU -> Conv1d -> GroupNorm -> residual
        add -> GELU, both convs at the same dilation, same-padding (stride
        1) so sequence length is preserved exactly -- this is the
        resolution-preserving mechanism: no block in the tower ever
        downsamples, so there is no reinflation step for predict to
        undo, unlike the legacy 50bp-bin path's pool-then-merge design."""

        def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
            super().__init__()
            pad = (kernel_size - 1) * dilation // 2
            self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                                   padding=pad, dilation=dilation)
            self.norm1 = nn.GroupNorm(min(8, channels), channels)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                                   padding=pad, dilation=dilation)
            self.norm2 = nn.GroupNorm(min(8, channels), channels)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # x: (B, channels, T)
            residual = x
            x = self.act(self.norm1(self.conv1(x)))
            x = self.dropout(x)
            x = self.norm2(self.conv2(x))
            return self.act(x + residual)

    class DilatedResidualCNN(nn.Module):
        """Embedding -> stem conv -> N cycles of exponentially-dilated
        residual blocks -> per-position linear classifier. Structurally
        analogous to Helixer's CNN front end / ANNEVO's local module /
        SegmentNT's validated single-nucleotide segmentation approach --
        a single end-to-end model, no alignment-heavy classical tools and
        no large pretrained genomic LM in the loop (see plan Context for
        why both were ruled out: LTRharvest/LTR_FINDER-based cascades
        reproduce EarlGrey's own bottleneck since LTR is the dominant
        genome fraction; Hyena-DNA embedding was the single largest real
        measured cost this project hit, ~11.8h/genome)."""

        def __init__(self, n_classes: int, channels: int = 128,
                    kernel_size: int = 9, n_cycles: int = 3,
                    dilation_schedule=(1, 2, 4, 8, 16, 32, 64, 128),
                    embed_dim: int = 16, dropout: float = 0.1):
            super().__init__()
            self.embedding = nn.Embedding(5, embed_dim)  # A,C,G,T,N
            self.stem = nn.Conv1d(embed_dim, channels, kernel_size=1)
            blocks = []
            for _ in range(n_cycles):
                for d in dilation_schedule:
                    blocks.append(ResidualDilatedBlock(channels, kernel_size, d, dropout))
            self.blocks = nn.ModuleList(blocks)
            self.classifier = nn.Linear(channels, n_classes)

        def forward(self, x):
            # x: (B, T) int64 nucleotide indices
            x = self.embedding(x)      # (B, T, embed_dim)
            x = x.transpose(1, 2)      # (B, embed_dim, T)
            x = self.stem(x)           # (B, channels, T)
            for block in self.blocks:
                x = block(x)
            x = x.transpose(1, 2)      # (B, T, channels)
            return self.classifier(x)  # (B, T, n_classes) -- emissions

    return torch, nn, DilatedResidualCNN, LinearChainCRF


def load_dense_sequences(chunks_dir: Path, labels: list) -> tuple:
    """Load a --bin_size 1 prepare_dataset corpus (raw sequence + per-base
    int8 labels) for train_dense_cnn, pooling every species' Parquet file
    the same way load_dense_embeddings() does for the legacy backbone-
    embeddings corpus. Returns (X, y, seq_len) where X is (n_chunks,
    seq_len) int64 nucleotide indices and y is (n_chunks, seq_len) int64
    label indices."""
    parquet_files = sorted(chunks_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"ERROR: no Parquet files found in {chunks_dir}", file=sys.stderr)
        sys.exit(1)

    _log(f"Loading raw-sequence chunks from {len(parquet_files)} species: "
         f"{[f.stem for f in parquet_files]}")
    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    _log(f"Total chunks loaded: {len(df):,}")

    if "base_labels_bytes" not in df.columns:
        print(f"ERROR: {chunks_dir} does not look like a --bin_size 1 "
              f"prepare_dataset corpus (missing base_labels_bytes column). "
              f"Regenerate with: prepare_dataset --bin_size 1 ...", file=sys.stderr)
        sys.exit(1)

    bin_size_vals = df["bin_size"].unique()
    if len(bin_size_vals) > 1 or int(bin_size_vals[0]) != 1:
        print(f"ERROR: corpus does not have a uniform bin_size of 1: "
              f"{bin_size_vals}. train_dense_cnn requires a --bin_size 1 "
              f"prepare_dataset corpus.", file=sys.stderr)
        sys.exit(1)

    n_bins_vals = df["n_bins"].unique()
    if len(n_bins_vals) > 1:
        print(f"ERROR: inconsistent chunk length across the corpus: "
              f"{n_bins_vals}. Regenerate all species with the same "
              f"--chunk_size.", file=sys.stderr)
        sys.exit(1)
    seq_len = int(n_bins_vals[0])

    if labels != DEFAULT_LABELS:
        _log("  WARNING: --labels differs from DEFAULT_LABELS, but "
             "base_labels_bytes was encoded using DEFAULT_LABELS order at "
             "prepare_dataset time -- results will be wrong unless the "
             "label sets match exactly, in the same order.")

    X = np.stack([encode_sequence(s) for s in df["sequence"]])
    y = np.stack([
        np.frombuffer(b, dtype=np.int8).astype(np.int64)
        for b in df["base_labels_bytes"]
    ])

    return X, y, seq_len


def run_train_dense_cnn(args) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _open_log(logs_dir, "train_dense_cnn")

    chunks_dir = args.chunks_dir.resolve()
    _validate_inputs([("--chunks_dir", chunks_dir)])

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  chunks_dir     : {chunks_dir}/")
        _log(f"  outdir         : {outdir}/")
        _log(f"  epochs         : {args.epochs}")
        _log(f"  batch_size     : {args.batch_size}")
        _log(f"  lr             : {args.lr}")
        _log(f"  channels       : {args.channels}")
        _log(f"  kernel_size    : {args.kernel_size}")
        _log(f"  n_cycles       : {args.n_cycles}")
        _log(f"  embed_dim      : {args.embed_dim}")
        _log(f"  dropout        : {args.dropout}")
        _log(f"  val_fraction   : {args.val_fraction}")
        _log(f"  patience       : {args.patience}")
        _log(f"  class_weight   : {args.class_weight}")
        _log(f"  balanced_corpus: {args.balanced_corpus}")
        if args.balanced_corpus:
            _log(f"  target_bins_per_class: {args.target_bins_per_class}")
        _log(f"  labels         : {args.labels}")
        _log("  Steps that would run: load raw-sequence chunks -> train "
             "dilated-CNN+CRF sequence labeler (1bp resolution) -> "
             "evaluate -> save model")
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    _log(f"Using device: {device}")

    X, y, seq_len = load_dense_sequences(chunks_dir, args.labels)
    _log(f"Loaded {X.shape[0]:,} chunks | seq_len={seq_len}")

    if args.balanced_corpus:
        _log(f"Building balanced multi-genome corpus (target: "
             f"{args.target_bins_per_class:,} bases/class, rarest-first chunk selection) ...")
        sel_idx, achieved = _select_balanced_chunks(
            y, len(args.labels), args.target_bins_per_class, args.seed)
        _log(f"Selected {len(sel_idx):,} / {X.shape[0]:,} chunks")
        for lbl, n_achieved in zip(args.labels, achieved):
            flag = "" if n_achieved >= args.target_bins_per_class else "  [data-limited]"
            _log(f"    {lbl:15s} {int(n_achieved):>10,} / {args.target_bins_per_class:,} bases{flag}")
        X, y = X[sel_idx], y[sel_idx]

    # Stratify the chunk-level split by each chunk's dominant base label
    # (a per-chunk proxy — the classifier itself trains at base resolution).
    strat_key = [Counter(row).most_common(1)[0][0] for row in y]
    idx_train, idx_val = train_test_split(
        np.arange(len(X)), test_size=args.val_fraction,
        stratify=strat_key, random_state=args.seed,
    )
    X_train, X_val = X[idx_train], X[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]
    _log(f"Train: {len(X_train):,} chunks | Val: {len(X_val):,} chunks")

    torch_mod, nn, DilatedResidualCNN, LinearChainCRF = _build_dilated_cnn_model()

    class_weights = None
    if args.class_weight == "balanced":
        y_train_flat = y_train.reshape(-1)
        present_classes = np.unique(y_train_flat)
        w_present = compute_class_weight(class_weight="balanced",
                                         classes=present_classes, y=y_train_flat)
        full_weights = np.ones(len(args.labels), dtype=np.float32)
        for cls_idx, w in zip(present_classes, w_present):
            full_weights[cls_idx] = w
        class_weights = torch.tensor(full_weights, dtype=torch.float32).to(device)
        _log("Class weights (balanced, inverse per-base training frequency):")
        for lbl, w in zip(args.labels, full_weights):
            _log(f"    {lbl:15s} weight={w:.4f}")
    else:
        _log("Class weighting disabled (--class_weight none)")

    from torch.utils.data import DataLoader, TensorDataset
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.long),
                     torch.tensor(y_train, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.long),
                     torch.tensor(y_val, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=False)

    model = DilatedResidualCNN(n_classes=len(args.labels), channels=args.channels,
                               kernel_size=args.kernel_size, n_cycles=args.n_cycles,
                               embed_dim=args.embed_dim, dropout=args.dropout).to(device)
    crf = LinearChainCRF(n_classes=len(args.labels)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f"Model: Embedding(5->{args.embed_dim}) -> Conv1d(1x1) -> "
         f"{args.n_cycles} x 8 ResidualDilatedBlock(channels={args.channels}, "
         f"k={args.kernel_size}) -> Linear({args.channels}, {len(args.labels)}) "
         f"-> LinearChainCRF  ({n_params:,} params)")

    label_encoder = {"label_to_idx": {lbl: i for i, lbl in enumerate(args.labels)},
                     "idx_to_label": {str(i): lbl for i, lbl in enumerate(args.labels)}}
    with open(outdir / "label_encoder.json", "w") as fh:
        json.dump(label_encoder, fh, indent=2)

    model_config = {
        "architecture": "dilated_cnn_v2", "n_classes": len(args.labels),
        "channels": args.channels, "kernel_size": args.kernel_size,
        "n_cycles": args.n_cycles, "embed_dim": args.embed_dim,
        "dropout": args.dropout, "seq_len_trained_on": seq_len,
    }
    with open(outdir / "model_config.json", "w") as fh:
        json.dump(model_config, fh, indent=2)
    _log(f"Label encoder + model config saved to {outdir}")

    t_start = time.monotonic()
    tracker = _start_tracker("ShoggoTEh_train_dense_cnn", logs_dir, args.disable_co2_tracking)

    metrics = train_dense(torch_mod, nn, model, crf, train_loader, val_loader, device,
                          args.epochs, args.lr, args.patience, outdir, class_weights)

    emissions_kg = _stop_tracker(tracker)

    metrics_path = outdir / "training_metrics.tsv"
    pd.DataFrame(metrics).to_csv(metrics_path, sep="\t", index=False)
    _log(f"Training metrics saved to {metrics_path}")

    _log("Loading best model for final (base-level) evaluation ...")
    state = torch.load(outdir / "classifier.pt", map_location=device)
    model.load_state_dict(state["model"])
    crf.load_state_dict(state["crf"])
    model.eval(); crf.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            emissions = model(Xb.to(device))
            preds = crf.decode(emissions)
            all_preds.extend([p for seq in preds for p in seq])
            all_true.extend(yb.reshape(-1).tolist())

    report = classification_report(all_true, all_preds, labels=list(range(len(args.labels))),
                                   target_names=args.labels, digits=3, zero_division=0)
    _log(f"Validation base-level classification report:\n{report}")
    (outdir / "classification_report.txt").write_text(report)

    if emissions_kg is not None:
        _log(f"Carbon footprint: {emissions_kg:.6f} kg CO2 equivalent")

    elapsed_s   = time.monotonic() - t_start
    peak_mem_mb = _peak_mem_mb()
    _log(f"Wall-clock time   : {elapsed_s:.1f} s ({elapsed_s/60:.1f} min)")
    _log(f"Peak memory (RSS) : {peak_mem_mb:.1f} MB")

    best_val_loss = min((m["val_loss"] for m in metrics), default=None)
    summary = {
        "date":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION,
        "input_chunks_dir": str(chunks_dir),
        "n_classes":  len(args.labels),
        "seq_len":    seq_len,
        "n_params":   n_params,
        "n_train_chunks": len(X_train),
        "n_val_chunks":    len(X_val),
        "best_val_loss": best_val_loss,
        "n_epochs_run":  len(metrics),
        "parameters": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "channels": args.channels, "kernel_size": args.kernel_size,
            "n_cycles": args.n_cycles, "embed_dim": args.embed_dim,
            "dropout": args.dropout, "val_fraction": args.val_fraction,
            "patience": args.patience, "class_weight": args.class_weight,
            "balanced_corpus": args.balanced_corpus,
            "target_bins_per_class": args.target_bins_per_class if args.balanced_corpus else None,
            "seed": args.seed,
        },
        "resource_usage": {
            "wall_clock_s":       round(elapsed_s, 1),
            "peak_mem_mb":        round(peak_mem_mb, 1),
            "emissions_kg_CO2eq": emissions_kg,
        },
    }
    _write_summary(outdir, summary)
    _close_log()


def load_dense_embeddings(embeddings_dir: Path, labels: list) -> tuple:
    parquet_files = sorted(embeddings_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"ERROR: no Parquet files found in {embeddings_dir}", file=sys.stderr)
        sys.exit(1)

    _log(f"Loading embeddings from {len(parquet_files)} species: "
         f"{[f.stem for f in parquet_files]}")
    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    _log(f"Total chunks loaded: {len(df):,}")

    n_bins_vals    = df["n_bins"].unique()
    hidden_dim_vals = df["hidden_dim"].unique()
    if len(n_bins_vals) > 1:
        print(f"ERROR: inconsistent n_bins across the embeddings corpus: "
              f"{n_bins_vals}. Regenerate all species with the same "
              f"--chunk_size/--bin_size.", file=sys.stderr)
        sys.exit(1)
    if len(hidden_dim_vals) > 1:
        print(f"ERROR: inconsistent hidden_dim across the embeddings corpus: "
              f"{hidden_dim_vals}. Regenerate all species with the same "
              f"--backbone/--backbone_model.", file=sys.stderr)
        sys.exit(1)
    n_bins     = int(n_bins_vals[0])
    hidden_dim = int(hidden_dim_vals[0])

    # Backbone/backbone_model provenance: read back off the embeddings
    # themselves (written by process_species_embed), not trusted from any
    # CLI flag, so a mixed-backbone corpus (different species embedded with
    # different backbones -- semantically incompatible embedding spaces) is
    # caught here rather than silently producing a nonsense-trained model.
    if "backbone" in df.columns and "backbone_model" in df.columns:
        backbone_vals = df["backbone"].unique()
        backbone_model_vals = df["backbone_model"].unique()
        if len(backbone_vals) > 1 or len(backbone_model_vals) > 1:
            print(f"ERROR: mixed-backbone embeddings corpus detected — "
                  f"backbone(s): {list(backbone_vals)}, "
                  f"backbone_model(s): {list(backbone_model_vals)}. "
                  f"All species must be embedded with the same "
                  f"--backbone/--backbone_model before training. Regenerate "
                  f"the mismatched species' embeddings.", file=sys.stderr)
            sys.exit(1)
        backbone       = str(backbone_vals[0])
        backbone_model = str(backbone_model_vals[0])
    else:
        _log("  WARNING: embeddings corpus has no recorded backbone/"
             "backbone_model columns (generated by an older ShoggoTEh "
             "version) — assuming 'hyena' (legacy default) for "
             "model_config.json provenance.")
        backbone, backbone_model = "hyena", BACKBONE_REGISTRY["hyena"]["default_model"]

    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    def _encode(bl):
        return [label_to_idx.get(l, -1) for l in bl]

    bin_idx = df["bin_labels"].apply(_encode)
    valid_mask = bin_idx.apply(lambda idxs: all(i >= 0 for i in idxs))
    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        _log(f"Dropped {n_dropped:,} chunks containing a bin label outside "
             f"the configured --labels set")
    df = df[valid_mask].reset_index(drop=True)
    bin_idx = bin_idx[valid_mask].reset_index(drop=True)

    all_bins = [l for row in df["bin_labels"] for l in row]
    _log(f"Bin label distribution:\n{pd.Series(all_bins).value_counts().to_string()}")

    X = np.stack([
        np.frombuffer(e, dtype=np.float32).reshape(n_bins, hidden_dim)
        for e in df["embedding_bytes"]
    ])
    y = np.stack([np.asarray(idxs, dtype=np.int64) for idxs in bin_idx])

    return X, y, n_bins, hidden_dim, backbone, backbone_model


def _select_balanced_chunks(y: np.ndarray, n_classes: int, target_per_class: int,
                            seed: int) -> tuple:
    """Greedy, multi-genome chunk selection for a balanced training corpus.

    Whole chunks (not individual bins) are the atomic selection unit -- the
    CRF needs contiguous bin runs to learn transition structure, so
    cherry-picking individual bins across the genome would destroy exactly
    the sequence context it depends on. Classes are processed rarest-first
    (by total corpus-wide bin count) so scarce classes get first claim on
    the chunks that contain them; a class's cumulative *bin* coverage (not
    chunk count) is compared against target_per_class, since one chunk
    typically contributes to several classes at once (its own gold labels
    are still whatever they are -- selection doesn't relabel anything).
    Candidate order is shuffled with `seed`, so chunks from every genome in
    the pooled corpus (load_dense_embeddings already concatenates all
    species' Parquet files) mix freely with no per-genome preference.

    Returns (selected_indices, achieved_bin_counts) where achieved_bin_counts
    is the actual per-class bin count in the selected subset -- classes with
    fewer than target_per_class total bins in the whole corpus will fall
    short of the target by construction; the caller should log this."""
    rng = np.random.default_rng(seed)
    n_chunks = y.shape[0]

    chunk_class_counts = np.zeros((n_chunks, n_classes), dtype=np.int64)
    for c in range(n_classes):
        chunk_class_counts[:, c] = (y == c).sum(axis=1)

    total_per_class = chunk_class_counts.sum(axis=0)
    class_order = np.argsort(total_per_class)  # rarest first

    selected = np.zeros(n_chunks, dtype=bool)
    covered_bins = np.zeros(n_classes, dtype=np.int64)

    for c in class_order:
        if covered_bins[c] >= target_per_class:
            continue
        candidates = np.where((chunk_class_counts[:, c] > 0) & (~selected))[0]
        rng.shuffle(candidates)
        for idx in candidates:
            if covered_bins[c] >= target_per_class:
                break
            selected[idx] = True
            covered_bins += chunk_class_counts[idx]

    return np.where(selected)[0], covered_bins


def train_dense(torch, nn, model, crf, train_loader, val_loader, device,
                epochs: int, lr: float, patience: int, outdir: Path,
                class_weights) -> list:
    optimizer = torch.optim.Adam(list(model.parameters()) + list(crf.parameters()), lr=lr)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    metrics = []

    for epoch in range(1, epochs + 1):
        model.train(); crf.train()
        train_loss, train_total = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            emissions = model(Xb)
            sw = class_weights[yb].mean(dim=1) if class_weights is not None else None
            loss = crf.neg_log_likelihood(emissions, yb, sw)
            loss.backward()
            optimizer.step()
            train_loss  += loss.item() * len(yb)
            train_total += len(yb)
            # No training-batch accuracy metric: a cheap on-GPU proxy
            # (raw per-position emissions.argmax(), ignoring the CRF's
            # transition structure) was tried here, but under real class
            # imbalance (e.g. a ~378x sample weight on the rarest class)
            # it diverged wildly from the actual CRF-decoded prediction --
            # observed on a real training run collapsing to ~0.004 "train
            # bin-acc" while the true (CRF-decoded) val bin-acc held at
            # ~0.58-0.61 the whole time. The mismatch is real and
            # inherent to how the CRF loss optimizes a joint sequence
            # likelihood (leaning on learned transitions to fix up weak
            # emissions), not a bug in the proxy computation itself -- so
            # there's no cheap fix that stays honest. Train loss (below)
            # is the real, correct training-time signal; val bin-acc
            # (still the full CRF decode, paid once per validation batch
            # per epoch, not once per training batch) is the real,
            # correct accuracy signal.

        model.eval(); crf.eval()
        val_loss, val_total, val_correct, val_bins = 0.0, 0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                emissions = model(Xb)
                sw = class_weights[yb].mean(dim=1) if class_weights is not None else None
                loss = crf.neg_log_likelihood(emissions, yb, sw)
                val_loss  += loss.item() * len(yb)
                val_total += len(yb)
                preds = torch.tensor(crf.decode(emissions), device=device)
                val_correct += (preds == yb).sum().item()
                val_bins    += yb.numel()

        t_loss = train_loss / train_total
        v_loss = val_loss / val_total
        v_acc  = val_correct / val_bins

        metrics.append({
            "epoch": epoch, "train_loss": round(t_loss, 6), "val_loss": round(v_loss, 6),
            "val_bin_acc": round(v_acc, 4),
        })
        _log(f"Epoch {epoch:3d}/{epochs} | train loss {t_loss:.4f} "
             f"| val loss {v_loss:.4f} bin-acc {v_acc:.3f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "crf": crf.state_dict()},
                      outdir / "classifier.pt")
            _log(f"  -> best model saved (val_loss={v_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                _log(f"Early stopping after {epoch} epochs (no improvement for {patience})")
                break

    return metrics


def run_train_classifier(args) -> None:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _open_log(logs_dir, "train_classifier")

    embeddings_dir = args.embeddings_dir.resolve()
    _validate_inputs([("--embeddings_dir", embeddings_dir)])

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  embeddings_dir : {embeddings_dir}/")
        _log(f"  outdir         : {outdir}/")
        _log(f"  epochs         : {args.epochs}")
        _log(f"  batch_size     : {args.batch_size}")
        _log(f"  lr             : {args.lr}")
        _log(f"  cnn_channels   : {args.cnn_channels}")
        _log(f"  kernel_size    : {args.kernel_size}")
        _log(f"  dropout        : {args.dropout}")
        _log(f"  val_fraction   : {args.val_fraction}")
        _log(f"  patience       : {args.patience}")
        _log(f"  class_weight   : {args.class_weight}")
        _log(f"  balanced_corpus: {args.balanced_corpus}")
        if args.balanced_corpus:
            _log(f"  target_bins_per_class: {args.target_bins_per_class}")
        _log(f"  labels         : {args.labels}")
        _log("  Steps that would run: load per-bin embeddings -> train "
             "CNN+CRF sequence labeler -> evaluate -> save model")
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    _log(f"Using device: {device}")

    X, y, n_bins, hidden_dim, backbone, backbone_model = load_dense_embeddings(
        embeddings_dir, args.labels)
    _log(f"Loaded {X.shape[0]:,} chunks | n_bins={n_bins} | hidden_dim={hidden_dim} | "
         f"backbone={backbone} | backbone_model={backbone_model}")

    if args.balanced_corpus:
        _log(f"Building balanced multi-genome corpus (target: "
             f"{args.target_bins_per_class:,} bins/class, rarest-first chunk selection) ...")
        sel_idx, achieved = _select_balanced_chunks(
            y, len(args.labels), args.target_bins_per_class, args.seed)
        _log(f"Selected {len(sel_idx):,} / {X.shape[0]:,} chunks")
        for lbl, n_achieved in zip(args.labels, achieved):
            flag = "" if n_achieved >= args.target_bins_per_class else "  [data-limited]"
            _log(f"    {lbl:15s} {int(n_achieved):>10,} / {args.target_bins_per_class:,} bins{flag}")
        X, y = X[sel_idx], y[sel_idx]

    # Stratify the chunk-level split by each chunk's dominant bin label
    # (a per-chunk proxy — the classifier itself trains at bin resolution).
    strat_key = [Counter(row).most_common(1)[0][0] for row in y]
    idx_train, idx_val = train_test_split(
        np.arange(len(X)), test_size=args.val_fraction,
        stratify=strat_key, random_state=args.seed,
    )
    X_train, X_val = X[idx_train], X[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]
    _log(f"Train: {len(X_train):,} chunks | Val: {len(X_val):,} chunks")

    torch_mod, nn, DenseTEClassifier, LinearChainCRF = _build_dense_model_classes()

    class_weights = None
    if args.class_weight == "balanced":
        y_train_flat = y_train.reshape(-1)
        present_classes = np.unique(y_train_flat)
        w_present = compute_class_weight(class_weight="balanced",
                                         classes=present_classes, y=y_train_flat)
        full_weights = np.ones(len(args.labels), dtype=np.float32)
        for cls_idx, w in zip(present_classes, w_present):
            full_weights[cls_idx] = w
        class_weights = torch.tensor(full_weights, dtype=torch.float32).to(device)
        _log("Class weights (balanced, inverse per-bin training frequency):")
        for lbl, w in zip(args.labels, full_weights):
            _log(f"    {lbl:15s} weight={w:.4f}")
    else:
        _log("Class weighting disabled (--class_weight none)")

    from torch.utils.data import DataLoader, TensorDataset
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                     torch.tensor(y_train, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                     torch.tensor(y_val, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=False)

    model = DenseTEClassifier(input_dim=hidden_dim, cnn_channels=args.cnn_channels,
                              kernel_size=args.kernel_size, n_classes=len(args.labels),
                              dropout=args.dropout).to(device)
    crf = LinearChainCRF(n_classes=len(args.labels)).to(device)
    _log(f"Model: Conv1d({hidden_dim}->{args.cnn_channels}, k={args.kernel_size}) -> "
         f"ReLU -> Dropout({args.dropout}) -> Linear({args.cnn_channels}, "
         f"{len(args.labels)}) -> LinearChainCRF")

    label_encoder = {"label_to_idx": {lbl: i for i, lbl in enumerate(args.labels)},
                     "idx_to_label": {str(i): lbl for i, lbl in enumerate(args.labels)}}
    with open(outdir / "label_encoder.json", "w") as fh:
        json.dump(label_encoder, fh, indent=2)

    model_config = {
        "input_dim": hidden_dim, "cnn_channels": args.cnn_channels,
        "kernel_size": args.kernel_size, "n_classes": len(args.labels),
        "dropout": args.dropout, "bin_size_trained_on": None,
        # Backbone/backbone_model actually used to produce the embeddings
        # this model was trained on (read back from the embeddings
        # themselves in load_dense_embeddings, not from any CLI flag) --
        # predict reads these back to auto-correct/warn on a mismatched
        # --backbone/--backbone_model instead of silently mixing embedding
        # spaces.
        "backbone": backbone, "backbone_model": backbone_model,
    }
    with open(outdir / "model_config.json", "w") as fh:
        json.dump(model_config, fh, indent=2)
    _log(f"Label encoder + model config saved to {outdir}")

    t_start = time.monotonic()
    tracker = _start_tracker("ShoggoTEh_train_classifier", logs_dir, args.disable_co2_tracking)

    metrics = train_dense(torch_mod, nn, model, crf, train_loader, val_loader, device,
                          args.epochs, args.lr, args.patience, outdir, class_weights)

    emissions_kg = _stop_tracker(tracker)

    metrics_path = outdir / "training_metrics.tsv"
    pd.DataFrame(metrics).to_csv(metrics_path, sep="\t", index=False)
    _log(f"Training metrics saved to {metrics_path}")

    _log("Loading best model for final (bin-level) evaluation ...")
    state = torch.load(outdir / "classifier.pt", map_location=device)
    model.load_state_dict(state["model"])
    crf.load_state_dict(state["crf"])
    model.eval(); crf.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            emissions = model(Xb.to(device))
            preds = crf.decode(emissions)
            all_preds.extend([p for seq in preds for p in seq])
            all_true.extend(yb.reshape(-1).tolist())

    report = classification_report(all_true, all_preds, labels=list(range(len(args.labels))),
                                   target_names=args.labels, digits=3, zero_division=0)
    _log(f"Validation bin-level classification report:\n{report}")
    (outdir / "classification_report.txt").write_text(report)

    if emissions_kg is not None:
        _log(f"Carbon footprint: {emissions_kg:.6f} kg CO2 equivalent")

    elapsed_s   = time.monotonic() - t_start
    peak_mem_mb = _peak_mem_mb()
    _log(f"Wall-clock time   : {elapsed_s:.1f} s ({elapsed_s/60:.1f} min)")
    _log(f"Peak memory (RSS) : {peak_mem_mb:.1f} MB")

    best_val_loss = min((m["val_loss"] for m in metrics), default=None)
    summary = {
        "date":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION,
        "input_embeddings_dir": str(embeddings_dir),
        "n_classes":    len(args.labels),
        "n_bins":       n_bins,
        "hidden_dim":   hidden_dim,
        "backbone":       backbone,
        "backbone_model": backbone_model,
        "n_train_chunks": len(X_train),
        "n_val_chunks":    len(X_val),
        "best_val_loss": best_val_loss,
        "n_epochs_run":  len(metrics),
        "parameters": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "cnn_channels": args.cnn_channels, "kernel_size": args.kernel_size,
            "dropout": args.dropout, "val_fraction": args.val_fraction,
            "patience": args.patience, "class_weight": args.class_weight,
            "balanced_corpus": args.balanced_corpus,
            "target_bins_per_class": args.target_bins_per_class if args.balanced_corpus else None,
            "seed": args.seed,
        },
        "resource_usage": {
            "wall_clock_s":       round(elapsed_s, 1),
            "peak_mem_mb":        round(peak_mem_mb, 1),
            "emissions_kg_CO2eq": emissions_kg,
        },
    }
    _write_summary(outdir, summary)
    _close_log()


###############################################################################
# predict — dense forward pass + CRF decode + run-length-encode to BED
###############################################################################

def load_classifier_dense(model_dir: Path, device):
    encoder_path = model_dir / "label_encoder.json"
    config_path  = model_dir / "model_config.json"
    weights_path = model_dir / "classifier.pt"
    for p, flag in ((encoder_path, "label_encoder.json"), (config_path, "model_config.json"),
                    (weights_path, "classifier.pt")):
        if not p.exists():
            print(f"ERROR: {flag} not found in {model_dir}", file=sys.stderr)
            sys.exit(1)

    with open(encoder_path) as fh:
        enc = json.load(fh)
    labels = [enc["idx_to_label"][str(i)] for i in range(len(enc["idx_to_label"]))]

    with open(config_path) as fh:
        cfg = json.load(fh)

    _, _, DenseTEClassifier, LinearChainCRF = _build_dense_model_classes()
    model = DenseTEClassifier(input_dim=cfg["input_dim"], cnn_channels=cfg["cnn_channels"],
                              kernel_size=cfg["kernel_size"], n_classes=cfg["n_classes"],
                              dropout=0.0)
    crf = LinearChainCRF(n_classes=cfg["n_classes"])

    import torch
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state["model"])
    crf.load_state_dict(state["crf"])
    model.eval().to(device)
    crf.eval().to(device)
    _log(f"Classifier loaded: input_dim={cfg['input_dim']} cnn_channels={cfg['cnn_channels']} "
         f"n_classes={cfg['n_classes']}")
    _log(f"Labels: {labels}")
    return model, crf, labels


def load_dense_cnn_classifier(model_dir: Path, device):
    """Load a train_dense_cnn (v2) model — mirrors load_classifier_dense()
    but reconstructs DilatedResidualCNN from model_config.json instead of
    DenseTEClassifier, and requires no backbone/embedding provenance
    (there is none — the v2 model trains end-to-end on raw sequence)."""
    encoder_path = model_dir / "label_encoder.json"
    config_path  = model_dir / "model_config.json"
    weights_path = model_dir / "classifier.pt"
    for p, flag in ((encoder_path, "label_encoder.json"), (config_path, "model_config.json"),
                    (weights_path, "classifier.pt")):
        if not p.exists():
            print(f"ERROR: {flag} not found in {model_dir}", file=sys.stderr)
            sys.exit(1)

    with open(encoder_path) as fh:
        enc = json.load(fh)
    labels = [enc["idx_to_label"][str(i)] for i in range(len(enc["idx_to_label"]))]

    with open(config_path) as fh:
        cfg = json.load(fh)
    if cfg.get("architecture") != "dilated_cnn_v2":
        print(f"ERROR: {model_dir} does not look like a train_dense_cnn (v2) "
              f"model (model_config.json architecture={cfg.get('architecture')!r}, "
              f"expected 'dilated_cnn_v2'). Use the legacy 'predict' subcommand "
              f"for train_classifier models.", file=sys.stderr)
        sys.exit(1)

    _, _, DilatedResidualCNN, LinearChainCRF = _build_dilated_cnn_model()
    model = DilatedResidualCNN(n_classes=cfg["n_classes"], channels=cfg["channels"],
                               kernel_size=cfg["kernel_size"], n_cycles=cfg["n_cycles"],
                               embed_dim=cfg["embed_dim"], dropout=0.0)
    crf = LinearChainCRF(n_classes=cfg["n_classes"])

    import torch
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state["model"])
    crf.load_state_dict(state["crf"])
    model.eval().to(device)
    crf.eval().to(device)
    _log(f"Classifier loaded: channels={cfg['channels']} n_cycles={cfg['n_cycles']} "
         f"kernel_size={cfg['kernel_size']} n_classes={cfg['n_classes']}")
    _log(f"Labels: {labels}")
    return model, crf, labels


def make_predict_segments(chrom_len: int, window_size: int, overlap: int) -> list:
    """Tile a chromosome into non-overlapping output segments, each backed
    by a (possibly larger) inference window extending past the segment on
    both sides by up to overlap//2 bases for extra CRF/receptive-field
    context -- the standard overlap-trim pattern for sliding-window
    genomic inference. Window length is clamped to the chromosome length,
    never padded: the fully-convolutional DilatedResidualCNN has no
    fixed-length assumption anywhere, so short scaffolds are handled
    natively. Returns [(seg_start, seg_end, win_start, win_end), ...]."""
    stride = window_size - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be smaller than window_size ({window_size})")
    segments = []
    pos = 0
    while pos < chrom_len:
        seg_start = pos
        seg_end = min(pos + stride, chrom_len)
        win_start = max(0, seg_start - overlap // 2)
        win_end = min(chrom_len, win_start + window_size)
        win_start = max(0, win_end - window_size)
        segments.append((seg_start, seg_end, win_start, win_end))
        pos = seg_end
    return segments


def _stream_rle_step(intervals: list, open_iv, chrom: str, pos: int,
                     label: str, prob: float):
    """Incrementally run-length-encode one more (chrom, pos, label, prob)
    observation into `intervals`, extending or closing `open_iv` as
    needed. Genome-scale predict_dense_cnn produces one prediction per
    base (potentially billions across a real genome) -- materializing one
    dict per base before merging (as the legacy bin-level rle_encode_bins
    does, fine at 50bp-bin scale) would reproduce the exact class of
    Python-object-boxing memory blowup already fixed once this session for
    embeddings. This keeps only the single currently-open interval plus
    the final merged-interval list (bounded by real TE element count, not
    genome length) in memory. Caller must flush the final `open_iv` after
    the loop ends (it is never auto-flushed on the last position)."""
    if (open_iv is not None and open_iv["chrom"] == chrom
            and open_iv["label"] == label and open_iv["end"] == pos):
        open_iv["end"] += 1
        open_iv["prob_sum"] += prob
        open_iv["prob_n"] += 1
        return open_iv
    if open_iv is not None:
        open_iv["confidence"] = open_iv["prob_sum"] / open_iv["prob_n"]
        del open_iv["prob_sum"], open_iv["prob_n"]
        intervals.append(open_iv)
    return {"chrom": chrom, "start": pos, "end": pos + 1, "label": label,
           "prob_sum": prob, "prob_n": 1}


def run_predict_dense_cnn(args) -> None:
    fasta_path = args.fasta.resolve()
    model_dir  = args.model_dir.resolve()
    outdir     = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _open_log(logs_dir, "predict_dense_cnn")

    _validate_inputs([("--fasta", fasta_path), ("--model_dir", model_dir)])

    if args.overlap >= args.window_size:
        print(f"ERROR: --overlap ({args.overlap}) must be smaller than "
              f"--window_size ({args.window_size})", file=sys.stderr)
        sys.exit(1)

    prefix = args.prefix or fasta_path.stem

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  fasta          : {fasta_path}")
        _log(f"  model_dir      : {model_dir}/")
        _log(f"  outdir         : {outdir}/")
        _log(f"  prefix         : {prefix}")
        _log(f"  window_size    : {args.window_size}")
        _log(f"  overlap        : {args.overlap}")
        _log(f"  max_n_fraction : {args.max_n_fraction}")
        _log(f"  batch_size     : {args.batch_size}")
        _log("  Steps that would run: tile genome into overlap-trimmed "
             "segments -> batched dilated CNN+CRF forward pass + Viterbi "
             "decode -> keep each segment's core -> streaming run-length-"
             "encode -> BED")
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    import torch
    import torch.nn.functional as F
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    _log(f"Using device: {device}")

    model, crf, labels = load_dense_cnn_classifier(model_dir, device)

    t_start = time.monotonic()
    tracker = _start_tracker("ShoggoTEh_predict_dense_cnn", logs_dir, args.disable_co2_tracking)

    _log(f"Loading genome: {fasta_path}")
    fasta = Fasta(str(fasta_path))

    intervals = []
    open_iv = None
    label_counts = Counter()
    n_segments_total = 0
    n_segments_skipped_n = 0

    def _flush_batch(pending):
        """pending: list of (chrom, seg_start, seg_end, win_start, win_end,
        seq) tuples, all sharing the same window length (batching requires
        a rectangular tensor) and already in ascending genomic order. Runs
        one batched forward pass + CRF decode (batching amortizes the
        CRF's sequential Python-loop decode cost across every sequence in
        the batch, not just the model's own compute), then streams each
        window's core region into the running RLE state in the same
        order the windows were generated."""
        nonlocal open_iv
        if not pending:
            return
        Xb = torch.tensor(np.stack([encode_sequence(p[5]) for p in pending]),
                          dtype=torch.long, device=device)
        with torch.no_grad():
            emissions = model(Xb)
            probs = F.softmax(emissions, dim=2).cpu().numpy()
            paths = crf.decode(emissions)
        for (chrom, seg_start, seg_end, win_start, win_end, _seq), path, prob_row in \
                zip(pending, paths, probs):
            core_start_local = seg_start - win_start
            core_end_local   = seg_end - win_start
            for i in range(core_start_local, core_end_local):
                pos = win_start + i
                cls_idx = path[i]
                label = labels[cls_idx]
                label_counts[label] += 1
                open_iv = _stream_rle_step(intervals, open_iv, chrom, pos,
                                           label, float(prob_row[i, cls_idx]))

    pending, pending_len = [], None
    for chrom in fasta.keys():
        chrom_len = len(fasta[chrom])
        if chrom_len == 0:
            continue
        segments = make_predict_segments(chrom_len, args.window_size, args.overlap)
        _log(f"{chrom}: {chrom_len:,}bp -> {len(segments):,} segments")

        for seg_start, seg_end, win_start, win_end in segments:
            n_segments_total += 1
            win_seq = str(fasta[chrom][win_start:win_end]).upper()
            n_frac = win_seq.count("N") / len(win_seq) if win_seq else 1.0
            if n_frac > args.max_n_fraction:
                n_segments_skipped_n += 1
                continue

            win_len = win_end - win_start
            if pending and (win_len != pending_len or len(pending) >= args.batch_size):
                _flush_batch(pending)
                pending = []
            pending_len = win_len
            pending.append((chrom, seg_start, seg_end, win_start, win_end, win_seq))

    _flush_batch(pending)

    if open_iv is not None:
        open_iv["confidence"] = open_iv["prob_sum"] / open_iv["prob_n"]
        del open_iv["prob_sum"], open_iv["prob_n"]
        intervals.append(open_iv)

    n_kept = n_segments_total - n_segments_skipped_n
    _log(f"{n_segments_total:,} segments total | {n_kept:,} kept | "
         f"{n_segments_skipped_n:,} skipped (N content)")

    if not intervals:
        print("ERROR: no segments passed the N-content filter. Check your FASTA.",
              file=sys.stderr)
        sys.exit(1)

    _log("Prediction distribution (per-base): " +
         " | ".join(f"{k}: {v:,}" for k, v in sorted(label_counts.items())))

    write_bed_intervals(intervals, outdir / f"{prefix}.bed")

    emissions_kg = _stop_tracker(tracker)
    if emissions_kg is not None:
        _log(f"Carbon footprint: {emissions_kg:.6f} kg CO2 equivalent")

    elapsed_s   = time.monotonic() - t_start
    peak_mem_mb = _peak_mem_mb()
    _log(f"Wall-clock time   : {elapsed_s:.1f} s ({elapsed_s/60:.1f} min)")
    _log(f"Peak memory (RSS) : {peak_mem_mb:.1f} MB")

    summary = {
        "date":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION,
        "input_fasta": str(fasta_path),
        "n_segments_total": n_segments_total,
        "n_segments_kept":  n_kept,
        "n_segments_skipped_n": n_segments_skipped_n,
        "n_bed_intervals": len(intervals),
        "parameters": {
            "window_size": args.window_size, "overlap": args.overlap,
            "max_n_fraction": args.max_n_fraction, "batch_size": args.batch_size,
            "device": str(device),
        },
        "resource_usage": {
            "wall_clock_s":       round(elapsed_s, 1),
            "peak_mem_mb":        round(peak_mem_mb, 1),
            "emissions_kg_CO2eq": emissions_kg,
        },
    }
    _write_summary(outdir, summary)
    _close_log()


def resolve_predict_backbone(model_dir: Path, cli_backbone, cli_backbone_model) -> tuple:
    """model_config.json (written by train_classifier from the embeddings'
    own recorded provenance) is the authoritative source of truth for which
    backbone + checkpoint the loaded classifier was trained on -- a model
    trained on backbone X's embeddings produces garbage predictions if the
    input genome is embedded with backbone Y instead (different,
    semantically incompatible embedding spaces), with no error to signal
    it. If the user's --backbone/--backbone_model CLI flags disagree with
    what's recorded, auto-correct to the recorded values with a clear
    warning (the least-surprising choice -- predict then always matches the
    model it's using, rather than failing on an easy-to-make CLI typo)."""
    config_path = model_dir / "model_config.json"
    recorded_backbone = None
    recorded_model = None
    if config_path.exists():
        with open(config_path) as fh:
            cfg = json.load(fh)
        recorded_backbone = cfg.get("backbone")
        recorded_model = cfg.get("backbone_model")

    if recorded_backbone is None:
        _log("  model_config.json has no recorded backbone (model trained "
             "with an older ShoggoTEh version) — assuming 'hyena' (legacy default)")
        recorded_backbone = "hyena"
        recorded_model = recorded_model or BACKBONE_REGISTRY["hyena"]["default_model"]
    if recorded_backbone not in BACKBONE_REGISTRY:
        print(f"ERROR: model_config.json records an unknown backbone "
              f"'{recorded_backbone}'. Choices: {sorted(BACKBONE_REGISTRY)}",
              file=sys.stderr)
        sys.exit(1)
    recorded_model = recorded_model or BACKBONE_REGISTRY[recorded_backbone]["default_model"]

    # Compared unconditionally (not just when the user explicitly passed
    # --backbone) because --backbone's CLI default ("hyena") is itself a
    # real value indistinguishable from an explicit choice -- and a
    # default-hyena predict run against a model trained on a different
    # backbone is exactly the silent-mismatch failure mode this check
    # exists to catch.
    if cli_backbone != recorded_backbone:
        _log(f"  WARNING: --backbone {cli_backbone} does not match the backbone "
             f"model_config.json says this classifier was trained on "
             f"({recorded_backbone}). Auto-correcting to {recorded_backbone} "
             f"to avoid silently mixing incompatible embedding spaces.")
    if cli_backbone_model is not None and cli_backbone_model != recorded_model:
        _log(f"  WARNING: --backbone_model {cli_backbone_model} does not match "
             f"the checkpoint model_config.json says this classifier was "
             f"trained on ({recorded_model}). Auto-correcting to {recorded_model}.")

    return recorded_backbone, recorded_model


def rle_encode_bins(bin_records: list) -> list:
    """Run-length-encode a stream of per-bin predictions (sorted by chrom,
    start) into merged BED intervals. Breaks the run on chrom change, label
    change, or a coordinate discontinuity (non-adjacent bins)."""
    intervals = []
    cur = None
    for rec in bin_records:
        if (cur is not None and rec["chrom"] == cur["chrom"]
                and rec["label"] == cur["label"] and rec["start"] == cur["end"]):
            cur["end"] = rec["end"]
            cur["probs"].append(rec["prob"])
        else:
            if cur is not None:
                intervals.append(cur)
            cur = {"chrom": rec["chrom"], "start": rec["start"], "end": rec["end"],
                  "label": rec["label"], "probs": [rec["prob"]]}
    if cur is not None:
        intervals.append(cur)
    for iv in intervals:
        iv["confidence"] = sum(iv["probs"]) / len(iv["probs"])
        del iv["probs"]
    return intervals


def write_bed_intervals(intervals: list, bed_path: Path) -> None:
    intervals = sorted(intervals, key=lambda iv: (iv["chrom"], iv["start"]))
    with open(bed_path, "w") as fh:
        for iv in intervals:
            fh.write(f"{iv['chrom']}\t{iv['start']}\t{iv['end']}\t{iv['label']}\t"
                     f"{iv['confidence']:.4f}\t.\n")
    _log(f"BED written to {bed_path} ({len(intervals):,} merged intervals)")


def write_bin_probs_tsv(bin_records: list, labels: list, tsv_path: Path) -> None:
    bin_records = sorted(bin_records, key=lambda r: (r["chrom"], r["start"]))
    header = ["chrom", "start", "end", "label", "confidence"] + labels
    with open(tsv_path, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for r in bin_records:
            row = [r["chrom"], str(r["start"]), str(r["end"]), r["label"], f"{r['prob']:.4f}"]
            row += [f"{p:.4f}" for p in r["probs_all"]]
            fh.write("\t".join(row) + "\n")
    _log(f"Per-bin probability TSV written to {tsv_path}")


def run_predict(args) -> None:
    fasta_path = args.fasta.resolve()
    model_dir  = args.model_dir.resolve()
    outdir     = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _open_log(logs_dir, "predict")

    _validate_inputs([("--fasta", fasta_path), ("--model_dir", model_dir)])

    if args.chunk_size % args.bin_size != 0:
        print(f"ERROR: --chunk_size ({args.chunk_size}) must be a multiple of "
              f"--bin_size ({args.bin_size})", file=sys.stderr)
        sys.exit(1)

    prefix = args.prefix or fasta_path.stem

    backbone, backbone_model = resolve_predict_backbone(
        model_dir, args.backbone, args.backbone_model)

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  fasta          : {fasta_path}")
        _log(f"  model_dir      : {model_dir}/")
        _log(f"  outdir         : {outdir}/")
        _log(f"  prefix         : {prefix}")
        _log(f"  backbone       : {backbone}")
        _log(f"  backbone_model : {backbone_model}")
        _log(f"  chunk_size     : {args.chunk_size}")
        _log(f"  bin_size       : {args.bin_size}")
        _log(f"  overlap        : {args.overlap}")
        _log(f"  max_n_fraction : {args.max_n_fraction}")
        _log(f"  batch_size     : {args.batch_size}")
        _log("  Steps that would run: slide windows -> dense embed -> "
             "CNN+CRF forward + Viterbi decode -> run-length-encode -> BED")
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    import torch
    import torch.nn.functional as F
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    _log(f"Using device: {device}")

    tokenizer, backbone_encoder = load_backbone(backbone, backbone_model, device)
    registry_entry = BACKBONE_REGISTRY[backbone]
    model, crf, labels = load_classifier_dense(model_dir, device)

    t_start = time.monotonic()
    tracker = _start_tracker("ShoggoTEh_predict", logs_dir, args.disable_co2_tracking)

    _log(f"Loading genome: {fasta_path}")
    fasta  = Fasta(str(fasta_path))
    stride = int(args.chunk_size * (1 - args.overlap)) if args.overlap < 1.0 else args.chunk_size
    all_windows = make_windows(fasta, args.chunk_size, stride)
    n_windows_total = len(all_windows)
    _log(f"{n_windows_total:,} windows generated (chunk={args.chunk_size}, stride={stride})")

    n_bins = args.chunk_size // args.bin_size
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
    _log(f"{n_windows_kept:,} windows kept | {skipped_n:,} skipped (N content)")
    if not kept_windows:
        print("ERROR: no windows passed the N-content filter. Check your FASTA.",
              file=sys.stderr)
        sys.exit(1)

    _log("Generating dense per-bin embeddings ...")
    pooled_list = embed_sequences_dense(sequences, n_bins, args.bin_size,
                                        tokenizer, backbone_encoder, device, args.batch_size,
                                        hidden_state_fn=registry_entry["hidden_state_fn"],
                                        forward_kwargs=registry_entry["forward_kwargs"])

    _log("Running CNN + CRF forward pass and Viterbi decode ...")
    bin_records = []
    with torch.no_grad():
        for b_start in range(0, len(pooled_list), args.batch_size):
            batch_pooled = pooled_list[b_start: b_start + args.batch_size]
            batch_windows = kept_windows[b_start: b_start + args.batch_size]
            Xb = torch.tensor(np.stack(batch_pooled), dtype=torch.float32).to(device)
            emissions = model(Xb)
            probs = F.softmax(emissions, dim=2).cpu().numpy()   # (B, n_bins, C)
            decoded = crf.decode(emissions)                     # list[list[int]]

            for (chrom, start, end), path, prob_row in zip(batch_windows, decoded, probs):
                for i in range(n_bins):
                    bstart = start + i * args.bin_size
                    bend   = bstart + args.bin_size
                    cls_idx = path[i]
                    bin_records.append({
                        "chrom": chrom, "start": bstart, "end": bend,
                        "label": labels[cls_idx], "prob": float(prob_row[i, cls_idx]),
                        "probs_all": prob_row[i].tolist(),
                    })

    bin_records.sort(key=lambda r: (r["chrom"], r["start"]))
    dist = Counter(r["label"] for r in bin_records)
    _log("Prediction distribution (per-bin): " +
         " | ".join(f"{k}: {v:,}" for k, v in sorted(dist.items())))

    intervals = rle_encode_bins(bin_records)
    write_bed_intervals(intervals, outdir / f"{prefix}.bed")
    write_bin_probs_tsv(bin_records, labels, outdir / f"{prefix}_bin_probs.tsv")

    emissions_kg = _stop_tracker(tracker)
    if emissions_kg is not None:
        _log(f"Carbon footprint: {emissions_kg:.6f} kg CO2 equivalent")

    elapsed_s   = time.monotonic() - t_start
    peak_mem_mb = _peak_mem_mb()
    _log(f"Wall-clock time   : {elapsed_s:.1f} s ({elapsed_s/60:.1f} min)")
    _log(f"Peak memory (RSS) : {peak_mem_mb:.1f} MB")

    summary = {
        "date":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION,
        "input_fasta":     str(fasta_path),
        "n_windows_total": n_windows_total,
        "n_windows_kept":  n_windows_kept,
        "n_skipped_n":     skipped_n,
        "n_bed_intervals": len(intervals),
        "parameters": {
            "backbone": backbone, "backbone_model": backbone_model,
            "chunk_size": args.chunk_size,
            "bin_size": args.bin_size, "overlap": args.overlap,
            "max_n_fraction": args.max_n_fraction, "batch_size": args.batch_size,
            "device": str(device),
        },
        "resource_usage": {
            "wall_clock_s":       round(elapsed_s, 1),
            "peak_mem_mb":        round(peak_mem_mb, 1),
            "emissions_kg_CO2eq": emissions_kg,
        },
    }
    _write_summary(outdir, summary)
    _close_log()


###############################################################################
# compare_te_annotation — unchanged logic, moved into the merged CLI
###############################################################################

def load_target(bed_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        bed_path, sep="\t", header=None,
        names=["chrom", "start", "end", "label", "confidence", "strand"],
        dtype={"chrom": str, "start": int, "end": int, "label": str, "confidence": float},
    )
    _log(f"Target: {len(df):,} intervals loaded from {bed_path}")
    _log(f"Target label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def load_reference_repeats(bed_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        bed_path, sep="\t", header=None, usecols=[0, 1, 2, 3],
        names=["chrom", "start", "end", "classification"],
        dtype={"chrom": str, "start": int, "end": int, "classification": str},
    )
    df["label"] = df["classification"].apply(map_repeat_class)
    _log(f"Reference: {len(df):,} intervals loaded from {bed_path}")
    _log(f"Reference label distribution:\n{df['label'].value_counts().to_string()}")
    return df[["chrom", "start", "end", "label"]]


def load_genes_for_compare(gff3_path: Path) -> pd.DataFrame:
    gff = pd.read_csv(
        gff3_path, sep="\t", comment="#", header=None,
        names=["chrom", "source", "feature", "start", "end",
               "score", "strand", "frame", "attributes"],
        dtype={"chrom": str},
    )
    genes = gff[gff["feature"] == "gene"][["chrom", "start", "end"]].copy()
    genes["start"] = genes["start"] - 1
    _log(f"Gene models: {len(genes):,} gene intervals loaded from {gff3_path}")
    return genes


def assign_reference_labels(target_df: pd.DataFrame, repeats_df: pd.DataFrame,
                            genes_df, min_fraction: float) -> list:
    """Operates on arbitrary (non-fixed-size) target intervals — this is
    why compare_te_annotation needs no changes for the dense architecture:
    the merged BED intervals from predict's run-length-encoding are just
    another set of arbitrary-length intervals."""
    win_bed = pybedtools.BedTool.from_dataframe(target_df[["chrom", "start", "end"]])
    rep_bed = pybedtools.BedTool.from_dataframe(repeats_df)

    rep_intersect = win_bed.intersect(rep_bed, wao=True)
    rep_overlaps: dict = {}
    for feat in rep_intersect:
        key = (feat.chrom, int(feat.start), int(feat.end))
        rep_class = feat.fields[-2]
        bp = int(feat.fields[-1])
        if rep_class != "." and bp > 0:
            rep_overlaps.setdefault(key, {})
            rep_overlaps[key][rep_class] = rep_overlaps[key].get(rep_class, 0) + bp

    gene_overlaps: dict = {}
    if genes_df is not None:
        gene_bed = pybedtools.BedTool.from_dataframe(genes_df)
        gene_intersect = win_bed.intersect(gene_bed, wao=True)
        for feat in gene_intersect:
            key = (feat.chrom, int(feat.start), int(feat.end))
            bp = int(feat.fields[-1])
            if bp > 0:
                gene_overlaps[key] = gene_overlaps.get(key, 0) + bp

    results = []
    for _, row in target_df.iterrows():
        key = (row["chrom"], int(row["start"]), int(row["end"]))
        win_size = int(row["end"]) - int(row["start"])
        rep_ovlp = rep_overlaps.get(key, {})
        total_rep_bp = sum(rep_ovlp.values())
        rep_fraction = total_rep_bp / win_size if win_size else 0.0

        if rep_fraction >= min_fraction:
            dominant_class = max(rep_ovlp, key=rep_ovlp.get)
            dominant_bp = rep_ovlp[dominant_class]
            if dominant_bp / win_size >= min_fraction:
                ref_label, overlap_bp, overlap_frac = dominant_class, dominant_bp, dominant_bp / win_size
            else:
                ref_label, overlap_bp, overlap_frac = "Ambiguous", total_rep_bp, rep_fraction
        else:
            gene_bp = gene_overlaps.get(key, 0)
            gene_fraction = gene_bp / win_size if win_size else 0.0
            overlap_bp, overlap_frac = 0, 0.0
            if genes_df is not None and gene_fraction >= min_fraction:
                ref_label = "Genic"
            else:
                ref_label = "Intergenic"

        results.append({"ref_label": ref_label, "ref_overlap_bp": overlap_bp,
                        "ref_overlap_fraction": round(overlap_frac, 4)})

    pybedtools.cleanup()
    return results


def write_comparison(target_df: pd.DataFrame, ref_results: list, out_path: Path) -> None:
    rows = []
    for (_, trow), rrow in zip(target_df.iterrows(), ref_results):
        rows.append({
            "chrom": trow["chrom"], "start": trow["start"], "end": trow["end"],
            "target_label": trow["label"], "target_confidence": round(trow["confidence"], 4),
            "ref_label": rrow["ref_label"], "ref_overlap_bp": rrow["ref_overlap_bp"],
            "ref_overlap_fraction": rrow["ref_overlap_fraction"],
            "match": trow["label"] == rrow["ref_label"],
        })
    out_df = pd.DataFrame(rows).sort_values(["chrom", "start"])
    out_df.to_csv(out_path, sep="\t", index=False)
    _log(f"Per-interval comparison written to {out_path}")


def write_compare_metrics(target_labels: list, ref_labels: list, n_ambiguous: int,
                          used_gff3: bool, out_path: Path) -> float:
    present_labels = sorted(set(target_labels) | set(ref_labels))
    acc = accuracy_score(ref_labels, target_labels)
    report = classification_report(ref_labels, target_labels, labels=present_labels,
                                   target_names=present_labels, digits=4,
                                   zero_division=0, output_dict=True)

    rows = []
    for label in present_labels:
        m = report.get(label, {})
        rows.append({"class": label, "precision": round(m.get("precision", 0), 4),
                     "recall": round(m.get("recall", 0), 4),
                     "f1_score": round(m.get("f1-score", 0), 4),
                     "support": int(m.get("support", 0))})
    for avg in ("macro avg", "weighted avg"):
        m = report.get(avg, {})
        rows.append({"class": avg, "precision": round(m.get("precision", 0), 4),
                     "recall": round(m.get("recall", 0), 4),
                     "f1_score": round(m.get("f1-score", 0), 4),
                     "support": int(m.get("support", 0))})
    rows.append({"class": "overall_accuracy", "precision": round(acc, 4),
                 "recall": "", "f1_score": "", "support": len(target_labels)})
    metrics_df = pd.DataFrame(rows)

    cm = confusion_matrix(ref_labels, target_labels, labels=present_labels)
    cm_df = pd.DataFrame(cm, index=present_labels, columns=present_labels)
    cm_df.index.name = "ref \\ target"

    with open(out_path, "w") as fh:
        fh.write("# ShoggoTEh annotation comparison metrics\n")
        fh.write(f"# Gene annotations used: {used_gff3}\n")
        fh.write(f"# Intervals evaluated: {len(target_labels):,}\n")
        fh.write(f"# Intervals excluded (ambiguous reference): {n_ambiguous:,}\n")
        fh.write("#\n## Per-class and overall metrics\n")
        fh.write(metrics_df.to_csv(sep="\t", index=False))
        fh.write("\n## Confusion matrix (rows=reference, cols=target)\n")
        fh.write(cm_df.to_csv(sep="\t"))

    _log(f"Metrics written to {out_path}")
    _log(f"Overall accuracy: {acc:.4f}")
    return acc


def run_compare_te_annotation(args) -> None:
    target_path    = args.target.resolve()
    reference_path = args.reference.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or target_path.stem
    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _open_log(logs_dir, "compare_te_annotation")

    _validate_inputs([("--target", target_path), ("--reference", reference_path)])
    if args.gff3:
        args.gff3 = args.gff3.resolve()
        _validate_inputs([("--gff3", args.gff3)])

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  target      : {target_path}")
        _log(f"  reference   : {reference_path}")
        _log(f"  gff3        : {args.gff3 or '(none)'}")
        _log(f"  outdir      : {outdir}/")
        _log(f"  prefix      : {prefix}")
        _log(f"  min_fraction: {args.min_fraction}")
        _log("  Steps that would run: load inputs -> intersect -> write "
             "comparison TSV -> write metrics TSV")
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    t_start = time.monotonic()

    target_df  = load_target(target_path)
    repeats_df = load_reference_repeats(reference_path)
    genes_df   = load_genes_for_compare(args.gff3) if args.gff3 else None
    if genes_df is None:
        _log("No GFF3 provided — non-repeat intervals will be labelled Intergenic")

    _log("Intersecting target intervals with reference annotations ...")
    ref_results = assign_reference_labels(target_df, repeats_df, genes_df, args.min_fraction)

    write_comparison(target_df, ref_results, outdir / f"{prefix}_comparison.tsv")

    target_labels, ref_labels = [], []
    n_ambiguous = 0
    for (_, trow), rrow in zip(target_df.iterrows(), ref_results):
        if rrow["ref_label"] == "Ambiguous":
            n_ambiguous += 1
            continue
        target_labels.append(trow["label"])
        ref_labels.append(rrow["ref_label"])

    n_evaluated = len(target_labels)
    _log(f"Intervals evaluated: {n_evaluated:,} | excluded (ambiguous reference): {n_ambiguous:,}")

    if not target_labels:
        print("ERROR: no evaluable intervals — check that target and reference "
              "share the same chromosome names and coordinate space.", file=sys.stderr)
        sys.exit(1)

    overall_accuracy = write_compare_metrics(
        target_labels, ref_labels, n_ambiguous=n_ambiguous,
        used_gff3=genes_df is not None, out_path=outdir / f"{prefix}_metrics.tsv")

    elapsed_s   = time.monotonic() - t_start
    peak_mem_mb = _peak_mem_mb()
    _log(f"Wall-clock time   : {elapsed_s:.1f} s ({elapsed_s/60:.1f} min)")
    _log(f"Peak memory (RSS) : {peak_mem_mb:.1f} MB")

    summary = {
        "date":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION,
        "target":    str(target_path),
        "reference": str(reference_path),
        "n_intervals_evaluated": n_evaluated,
        "n_ambiguous":           n_ambiguous,
        "overall_accuracy":      round(overall_accuracy, 4) if overall_accuracy is not None else None,
        "parameters": {"min_fraction": args.min_fraction, "gff3": str(args.gff3) if args.gff3 else None},
        "resource_usage": {"wall_clock_s": round(elapsed_s, 1), "peak_mem_mb": round(peak_mem_mb, 1)},
    }
    _write_summary(outdir, summary)
    _close_log()


###############################################################################
# Argument parser
###############################################################################

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ShoggoTEh",
        description="Deep-learning TE annotation pipeline (Shoggoth + TEh) — "
                    "dense per-bin sequence labeling with a CNN+CRF head "
                    "over frozen Hyena-DNA embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    sub = ap.add_subparsers(
        dest="command",
        metavar="{prepare_dataset,generate_embeddings,train_classifier,predict,compare_te_annotation}",
        required=True,
    )

    # ── prepare_dataset ─────────────────────────────────────────────────────
    pp = sub.add_parser("prepare_dataset",
                        help="Slide windows, label every bin by the exact "
                             "repeat/gene interval it falls in (dense)")
    req = pp.add_argument_group("required")
    req.add_argument("--species_tsv", required=True, type=Path, metavar="FILE",
                     help="TSV: species_id, fasta, bed, gff3 (gff3 optional / NA)")
    req.add_argument("--outdir", required=True, type=Path, metavar="DIR",
                     help="Output directory for per-species Parquet files")
    opt = pp.add_argument_group("optional")
    opt.add_argument("--chunk_size", type=int, default=5000, metavar="BP",
                     help="Window size in bp (default: 5000)")
    opt.add_argument("--bin_size", type=int, default=50, metavar="BP",
                     help="Per-bin label/embedding resolution in bp; must "
                          "evenly divide --chunk_size (default: 50). "
                          "--bin_size 1 switches to true single-nucleotide "
                          "labeling for the v2 end-to-end dense CNN "
                          "architecture: labels are painted directly per "
                          "chromosome (not via bedtools, which does not "
                          "scale to 1bp resolution) and stored as compact "
                          "int8 bytes (base_labels_bytes) instead of the "
                          "legacy bin_labels string-list column, using the "
                          "fixed DEFAULT_LABELS vocabulary.")
    opt.add_argument("--overlap", type=float, default=0.5, metavar="F",
                     help="Fractional overlap between windows (default: 0.5)")
    opt.add_argument("--max_n_fraction", type=float, default=0.1, metavar="F",
                     help="Max N fraction allowed in a chunk (default: 0.1)")
    opt.add_argument("--force", action="store_true",
                     help="Rerun all species even if output Parquet already exists")
    opt.add_argument("--dry_run", action="store_true",
                     help="Validate inputs and print planned steps, then exit")
    opt.add_argument("--disable_co2_tracking", action="store_true",
                     help="Disable carbon footprint tracking")

    # ── generate_embeddings ─────────────────────────────────────────────────
    gp = sub.add_parser("generate_embeddings",
                        help="Run chunks through Hyena-DNA and pool into "
                             "per-bin embeddings")
    req = gp.add_argument_group("required")
    req.add_argument("--chunks_dir", required=True, type=Path, metavar="DIR",
                     help="Directory with per-species Parquet files from prepare_dataset")
    req.add_argument("--outdir", required=True, type=Path, metavar="DIR",
                     help="Output directory for per-species embedding Parquet files")
    opt = gp.add_argument_group("optional")
    opt.add_argument("--backbone", choices=sorted(BACKBONE_REGISTRY), default="hyena",
                     help="Embedding backbone to use (default: hyena). "
                          "A/B-test alternative pretrained genomic language "
                          "models against the Hyena-DNA default.")
    opt.add_argument("--backbone_model", default=None, metavar="NAME",
                     help="HuggingFace Hub checkpoint override for --backbone "
                          "(default: the backbone's own sensible default -- "
                          "see BACKBONE_REGISTRY in the script)")
    opt.add_argument("--bin_size", type=int, default=50, metavar="BP",
                     help="Per-bin pooling resolution in bp; must match the "
                          "chunks' bin_size (default: 50)")
    opt.add_argument("--batch_size", type=int, default=32, metavar="N",
                     help="Sequences per forward pass (default: 32)")
    opt.add_argument("--device", default=None, metavar="DEV",
                     help="'cuda', 'mps', or 'cpu'. Auto-detected if omitted.")
    opt.add_argument("--force", action="store_true",
                     help="Rerun all species even if output Parquet already exists")
    opt.add_argument("--dry_run", action="store_true",
                     help="Validate inputs and print planned steps, then exit")
    opt.add_argument("--disable_co2_tracking", action="store_true",
                     help="Disable carbon footprint tracking")

    # ── train_classifier ─────────────────────────────────────────────────────
    tp = sub.add_parser("train_classifier",
                        help="Train a CNN + linear-chain CRF sequence "
                             "labeler on per-bin embeddings")
    req = tp.add_argument_group("required")
    req.add_argument("--embeddings_dir", required=True, type=Path, metavar="DIR",
                     help="Directory with per-species embedding Parquet files")
    req.add_argument("--outdir", required=True, type=Path, metavar="DIR",
                     help="Output directory for model weights, config, metrics")
    opt = tp.add_argument_group("optional")
    opt.add_argument("--labels", nargs="+", default=DEFAULT_LABELS, metavar="LBL",
                     help=f"Ordered list of class labels (default: {DEFAULT_LABELS})")
    opt.add_argument("--epochs", type=int, default=50, metavar="N",
                     help="Maximum training epochs (default: 50)")
    opt.add_argument("--batch_size", type=int, default=64, metavar="N",
                     help="Mini-batch size in chunks/sequences (default: 64)")
    opt.add_argument("--lr", type=float, default=1e-3, metavar="F",
                     help="Adam learning rate (default: 0.001)")
    opt.add_argument("--cnn_channels", type=int, default=128, metavar="N",
                     help="1D-CNN channel width (default: 128)")
    opt.add_argument("--kernel_size", type=int, default=5, metavar="N",
                     help="1D-CNN kernel size in bins (default: 5)")
    opt.add_argument("--dropout", type=float, default=0.3, metavar="F",
                     help="Dropout probability (default: 0.3)")
    opt.add_argument("--val_fraction", type=float, default=0.2, metavar="F",
                     help="Fraction of chunks held out for validation (default: 0.2)")
    opt.add_argument("--patience", type=int, default=10, metavar="N",
                     help="Early-stopping patience in epochs (default: 10)")
    opt.add_argument("--class_weight", choices=["balanced", "none"], default="balanced",
                     help="'balanced' (default) weights the CRF's per-sequence "
                          "NLL by inverse per-bin training-set class frequency "
                          "(mean weight of the gold bin tags in that sequence), "
                          "so rare TE classes are not drowned out. 'none' disables it.")
    opt.add_argument("--balanced_corpus", action="store_true",
                     help="Build the training set via quota-capped, multi-genome "
                          "chunk selection instead of using every pooled chunk: "
                          "process classes rarest-first and pull in whole chunks "
                          "(any species in --embeddings_dir) containing that "
                          "class until its cumulative bin count reaches "
                          "--target_bins_per_class. Fixes rare-class scarcity by "
                          "exposure (more real examples, pooled across all "
                          "genomes) rather than only reweighting the loss. Off "
                          "by default; combine with --class_weight balanced.")
    opt.add_argument("--target_bins_per_class", type=int, default=20000, metavar="N",
                     help="Target per-class bin count for --balanced_corpus "
                          "(default: 20000). Classes with fewer bins available "
                          "in the whole pooled corpus fall short of this target "
                          "by construction -- the actual achieved count is "
                          "logged per class. Ignored unless --balanced_corpus.")
    opt.add_argument("--seed", type=int, default=42, metavar="N",
                     help="Random seed (default: 42)")
    opt.add_argument("--device", default=None, metavar="DEV",
                     help="'cuda', 'mps', or 'cpu'. Auto-detected if omitted.")
    opt.add_argument("--dry_run", action="store_true",
                     help="Validate inputs and print planned steps, then exit")
    opt.add_argument("--disable_co2_tracking", action="store_true",
                     help="Disable carbon footprint tracking")

    # ── train_dense_cnn (v2: end-to-end, no pretrained backbone) ────────────────
    tdc = sub.add_parser("train_dense_cnn",
                         help="v2: train a stride-1 dilated-residual CNN + "
                              "linear-chain CRF end-to-end on raw sequence "
                              "at true single-nucleotide (1bp) resolution "
                              "-- no pretrained backbone, no embedding step")
    req = tdc.add_argument_group("required")
    req.add_argument("--chunks_dir", required=True, type=Path, metavar="DIR",
                     help="Directory with per-species Parquet files from "
                          "'prepare_dataset --bin_size 1'")
    req.add_argument("--outdir", required=True, type=Path, metavar="DIR",
                     help="Output directory for model weights, config, metrics")
    opt = tdc.add_argument_group("optional")
    opt.add_argument("--labels", nargs="+", default=DEFAULT_LABELS, metavar="LBL",
                     help=f"Ordered list of class labels (default: {DEFAULT_LABELS}). "
                          f"Must match DEFAULT_LABELS order -- base_labels_bytes was "
                          f"encoded with that order at prepare_dataset time.")
    opt.add_argument("--epochs", type=int, default=50, metavar="N",
                     help="Maximum training epochs (default: 50)")
    opt.add_argument("--batch_size", type=int, default=16, metavar="N",
                     help="Mini-batch size in chunks/sequences (default: 16 -- "
                          "smaller than train_classifier's 64 since sequences "
                          "here are chunk_size long, e.g. 5000bp, not n_bins)")
    opt.add_argument("--lr", type=float, default=1e-3, metavar="F",
                     help="Adam learning rate (default: 0.001)")
    opt.add_argument("--channels", type=int, default=128, metavar="N",
                     help="Dilated residual tower channel width (default: 128)")
    opt.add_argument("--kernel_size", type=int, default=9, metavar="N",
                     help="Conv kernel size in bases (default: 9)")
    opt.add_argument("--n_cycles", type=int, default=3, metavar="N",
                     help="Number of times the dilation schedule "
                          "(1,2,4,8,16,32,64,128) repeats (default: 3, "
                          "~12kb receptive field). Trivially extensible -- "
                          "increase if benchmarking shows insufficient "
                          "receptive field for the longest LTRs.")
    opt.add_argument("--embed_dim", type=int, default=16, metavar="N",
                     help="Nucleotide embedding dimension (default: 16)")
    opt.add_argument("--dropout", type=float, default=0.1, metavar="F",
                     help="Dropout probability (default: 0.1)")
    opt.add_argument("--val_fraction", type=float, default=0.2, metavar="F",
                     help="Fraction of chunks held out for validation (default: 0.2)")
    opt.add_argument("--patience", type=int, default=10, metavar="N",
                     help="Early-stopping patience in epochs (default: 10)")
    opt.add_argument("--class_weight", choices=["balanced", "none"], default="balanced",
                     help="'balanced' (default) weights the CRF's per-sequence "
                          "NLL by inverse per-base training-set class frequency. "
                          "'none' disables it.")
    opt.add_argument("--balanced_corpus", action="store_true",
                     help="Build the training set via quota-capped, multi-genome "
                          "chunk selection instead of using every pooled chunk "
                          "(see train_classifier's flag of the same name -- "
                          "identical logic, reused unchanged).")
    opt.add_argument("--target_bins_per_class", type=int, default=20000, metavar="N",
                     help="Target per-class base count for --balanced_corpus "
                          "(default: 20000). Ignored unless --balanced_corpus.")
    opt.add_argument("--seed", type=int, default=42, metavar="N",
                     help="Random seed (default: 42)")
    opt.add_argument("--device", default=None, metavar="DEV",
                     help="'cuda', 'mps', or 'cpu'. Auto-detected if omitted.")
    opt.add_argument("--dry_run", action="store_true",
                     help="Validate inputs and print planned steps, then exit")
    opt.add_argument("--disable_co2_tracking", action="store_true",
                     help="Disable carbon footprint tracking")

    # ── predict ───────────────────────────────────────────────────────────────
    dp = sub.add_parser("predict",
                        help="Dense CNN+CRF forward pass + Viterbi decode "
                             "over a new genome, RLE-merged into BED intervals")
    req = dp.add_argument_group("required")
    req.add_argument("--fasta", required=True, type=Path, metavar="FASTA",
                     help="Input genome FASTA (uncompressed or bgzipped)")
    req.add_argument("--model_dir", required=True, type=Path, metavar="DIR",
                     help="Directory with classifier.pt, label_encoder.json, model_config.json")
    req.add_argument("--outdir", required=True, type=Path, metavar="DIR",
                     help="Output directory for BED and per-bin probability TSV")
    opt = dp.add_argument_group("optional")
    opt.add_argument("--prefix", default=None, metavar="STR",
                     help="Output filename prefix (default: FASTA filename stem)")
    opt.add_argument("--backbone", choices=sorted(BACKBONE_REGISTRY), default="hyena",
                     help="Embedding backbone to use (default: hyena). "
                          "model_config.json in --model_dir is authoritative: "
                          "if this conflicts with what the classifier was "
                          "actually trained on, ShoggoTEh auto-corrects and "
                          "logs a WARNING rather than silently mixing "
                          "incompatible embedding spaces.")
    opt.add_argument("--backbone_model", default=None, metavar="NAME",
                     help="HuggingFace Hub checkpoint override for --backbone "
                          "(default: the backbone's own sensible default). "
                          "Same model_config.json auto-correction applies.")
    opt.add_argument("--chunk_size", type=int, default=5000, metavar="BP",
                     help="Window size in bp (default: 5000)")
    opt.add_argument("--bin_size", type=int, default=50, metavar="BP",
                     help="Per-bin resolution in bp; must match the trained "
                          "model's bin_size and evenly divide --chunk_size (default: 50)")
    opt.add_argument("--overlap", type=float, default=0.0, metavar="F",
                     help="Fractional overlap between windows (default: 0.0 — "
                          "the dense architecture does not need the redundant "
                          "50%% overlap the old window-level pipeline used)")
    opt.add_argument("--max_n_fraction", type=float, default=0.1, metavar="F",
                     help="Windows above this N fraction are skipped (default: 0.1)")
    opt.add_argument("--batch_size", type=int, default=32, metavar="N",
                     help="Embedding + forward-pass batch size (default: 32)")
    opt.add_argument("--device", default=None, metavar="DEV",
                     help="'cuda', 'mps', or 'cpu'. Auto-detected if omitted.")
    opt.add_argument("--dry_run", action="store_true",
                     help="Validate inputs and print planned steps, then exit")
    opt.add_argument("--disable_co2_tracking", action="store_true",
                     help="Disable carbon footprint tracking")

    # ── predict_dense_cnn (v2: end-to-end, no pretrained backbone) ──────────────
    pdc = sub.add_parser("predict_dense_cnn",
                         help="v2: dilated-CNN+CRF forward pass + Viterbi "
                              "decode over a new genome at true 1bp "
                              "resolution, streaming run-length-encoded "
                              "into BED intervals -- no embedding step")
    req = pdc.add_argument_group("required")
    req.add_argument("--fasta", required=True, type=Path, metavar="FASTA",
                     help="Input genome FASTA (uncompressed or bgzipped)")
    req.add_argument("--model_dir", required=True, type=Path, metavar="DIR",
                     help="Directory with classifier.pt, label_encoder.json, "
                          "model_config.json from train_dense_cnn")
    req.add_argument("--outdir", required=True, type=Path, metavar="DIR",
                     help="Output directory for the predicted BED")
    opt = pdc.add_argument_group("optional")
    opt.add_argument("--prefix", default=None, metavar="STR",
                     help="Output filename prefix (default: FASTA filename stem)")
    opt.add_argument("--window_size", type=int, default=5000, metavar="BP",
                     help="Inference window size in bp (default: 5000, "
                          "matching prepare_dataset's default --chunk_size). "
                          "The v2 architecture plan's genome-scale target was "
                          "24kb windows, but the CRF's forward/decode are "
                          "sequential Python loops over sequence length -- "
                          "defaulting to 5000 (already smaller than the "
                          "model's own ~12kb receptive field at the default "
                          "--n_cycles 3, so little context is actually lost) "
                          "keeps predict practical until that loop is "
                          "optimized (windowed/chunked Viterbi). Increase "
                          "once benchmarked.")
    opt.add_argument("--overlap", type=int, default=500, metavar="BP",
                     help="Context overlap between adjacent inference "
                          "windows in bp, trimmed from each window's edges "
                          "before merging (default: 500). Must be smaller "
                          "than --window_size.")
    opt.add_argument("--max_n_fraction", type=float, default=0.1, metavar="F",
                     help="Windows above this N fraction are skipped (default: 0.1)")
    opt.add_argument("--batch_size", type=int, default=16, metavar="N",
                     help="Forward-pass batch size (default: 16). Windows "
                          "are batched only with other windows of identical "
                          "length (true for every window in a chromosome "
                          "except its last) -- batching amortizes the CRF's "
                          "sequential decode cost across the batch, not "
                          "just the model's own compute.")
    opt.add_argument("--device", default=None, metavar="DEV",
                     help="'cuda', 'mps', or 'cpu'. Auto-detected if omitted.")
    opt.add_argument("--dry_run", action="store_true",
                     help="Validate inputs and print planned steps, then exit")
    opt.add_argument("--disable_co2_tracking", action="store_true",
                     help="Disable carbon footprint tracking")

    # ── compare_te_annotation ────────────────────────────────────────────────
    cp = sub.add_parser("compare_te_annotation",
                        help="Benchmark predictions against a reference BED annotation")
    req = cp.add_argument_group("required")
    req.add_argument("-t", "--target", required=True, type=Path, metavar="BED",
                     help="Target BED file (ShoggoTEh predict output)")
    req.add_argument("-r", "--reference", required=True, type=Path, metavar="BED",
                     help="Reference BED file (e.g. EarlGrey filteredRepeats.bed)")
    req.add_argument("--outdir", required=True, type=Path, metavar="DIR",
                     help="Output directory")
    opt = cp.add_argument_group("optional")
    opt.add_argument("--gff3", default=None, type=Path, metavar="FILE",
                     help="Gene annotation GFF3 for Genic/Intergenic resolution (optional)")
    opt.add_argument("--prefix", default=None, metavar="STR",
                     help="Output filename prefix (default: target BED stem)")
    opt.add_argument("--min_fraction", type=float, default=0.5, metavar="F",
                     help="Min overlap fraction to assign a reference label (default: 0.5)")
    opt.add_argument("--dry_run", action="store_true",
                     help="Validate inputs and print planned steps, then exit")

    return ap


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    _print_quote()
    ap = _build_parser()
    args = ap.parse_args()

    if args.command == "prepare_dataset":
        run_prepare_dataset(args)
    elif args.command == "generate_embeddings":
        run_generate_embeddings(args)
    elif args.command == "train_classifier":
        run_train_classifier(args)
    elif args.command == "train_dense_cnn":
        run_train_dense_cnn(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "predict_dense_cnn":
        run_predict_dense_cnn(args)
    elif args.command == "compare_te_annotation":
        run_compare_te_annotation(args)


if __name__ == "__main__":
    main()
