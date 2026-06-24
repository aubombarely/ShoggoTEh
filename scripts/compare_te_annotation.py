#!/usr/bin/env python3
"""
compare_te_annotation.py  —  ShoggoTEh annotation comparison

Compares a target TE annotation (e.g. ShoggoTEh predictions) against a
reference annotation (e.g. EarlGrey filteredRepeats.bed). For each target
window the dominant reference class is determined by majority bp overlap,
using the same label-normalization and precedence rules as prepare_dataset.py
(repeats > genic > intergenic).

Usage:
    python compare_te_annotation.py \
        -t predictions/genome.bed \
        -r /path/to/filteredRepeats.bed \
        --outdir comparisons/

    # With gene annotations for Genic/Intergenic resolution:
    python compare_te_annotation.py \
        -t predictions/genome.bed \
        -r /path/to/filteredRepeats.bed \
        --gff3 /path/to/annotation.gff3 \
        --outdir comparisons/

Output:
    {outdir}/{prefix}_comparison.tsv   — per-window detail
    {outdir}/{prefix}_metrics.tsv      — per-class and overall metrics
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

import pandas as pd
import pybedtools
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

VERSION = "v0.2.0"
SCRIPT_NAME = "compare_te_annotation"

# ── Label normalization (mirrors prepare_dataset.py) ──────────────────────────

REPEAT_CLASS_MAP = {
    "LTR":            "LTR",
    "DNA":            "DNA",
    "LINE":           "LINE",
    "SINE":           "SINE",
    "RC":             "DNA",
    "Unknown":        "Unknown_repeat",
    "Satellite":      "Other_repeat",
    "Simple_repeat":  "Other_repeat",
    "Low_complexity": "Other_repeat",
}

ALL_LABELS = ["LTR", "DNA", "LINE", "SINE", "Unknown_repeat",
              "Other_repeat", "Genic", "Intergenic"]


def map_repeat_class(classification: str) -> str:
    top = str(classification).split("/")[0]
    return REPEAT_CLASS_MAP.get(top, "Other_repeat")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_target(bed_path: Path) -> pd.DataFrame:
    """Load ShoggoTEh BED: chrom start end label confidence strand."""
    df = pd.read_csv(
        bed_path, sep="\t", header=None,
        names=["chrom", "start", "end", "label", "confidence", "strand"],
        dtype={"chrom": str, "start": int, "end": int,
               "label": str, "confidence": float},
    )
    log.info(f"Target: {len(df):,} windows loaded from {bed_path}")
    log.info(f"Target label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def load_reference_repeats(bed_path: Path) -> pd.DataFrame:
    """Load reference BED (EarlGrey or similar) and normalise labels."""
    df = pd.read_csv(
        bed_path, sep="\t", header=None, usecols=[0, 1, 2, 3],
        names=["chrom", "start", "end", "classification"],
        dtype={"chrom": str, "start": int, "end": int, "classification": str},
    )
    df["label"] = df["classification"].apply(map_repeat_class)
    log.info(f"Reference: {len(df):,} intervals loaded from {bed_path}")
    log.info(f"Reference label distribution:\n{df['label'].value_counts().to_string()}")
    return df[["chrom", "start", "end", "label"]]


def load_genes(gff3_path: Path) -> pd.DataFrame:
    """Extract gene intervals from a GFF3 file (0-based BED coordinates)."""
    gff = pd.read_csv(
        gff3_path, sep="\t", comment="#", header=None,
        names=["chrom", "source", "feature", "start", "end",
               "score", "strand", "frame", "attributes"],
        dtype={"chrom": str},
    )
    genes = gff[gff["feature"] == "gene"][["chrom", "start", "end"]].copy()
    genes["start"] = genes["start"] - 1   # GFF3 is 1-based
    log.info(f"Gene models: {len(genes):,} gene intervals loaded from {gff3_path}")
    return genes


# ── Reference label assignment ────────────────────────────────────────────────

def assign_reference_labels(
    target_df: pd.DataFrame,
    repeats_df: pd.DataFrame,
    genes_df: pd.DataFrame | None,
    min_fraction: float,
) -> list[dict]:
    """
    For each target window determine the dominant reference label and overlap
    statistics. Returns a list of dicts (one per window, same order as target_df).

    Precedence: repeat > genic > intergenic.
    Windows where no single repeat class reaches min_fraction but total repeat
    coverage does are marked as 'Ambiguous' and excluded from metrics.
    """
    win_bed = pybedtools.BedTool.from_dataframe(
        target_df[["chrom", "start", "end"]]
    )
    rep_bed = pybedtools.BedTool.from_dataframe(repeats_df)

    # ── Repeat overlap ─────────────────────────────────────────────────────────
    # wao: write the original A entry plus each overlapping B feature + overlap bp
    rep_intersect = win_bed.intersect(rep_bed, wao=True)

    # Accumulate bp per repeat class per window
    rep_overlaps: dict = {}
    for feat in rep_intersect:
        key = (feat.chrom, int(feat.start), int(feat.end))
        rep_class = feat.fields[3]   # target has 3 cols → ref label is fields[3]
        bp = int(feat.fields[4])     # overlap bp is last field
        if rep_class != "." and bp > 0:
            rep_overlaps.setdefault(key, {})
            rep_overlaps[key][rep_class] = rep_overlaps[key].get(rep_class, 0) + bp

    # ── Gene overlap (optional) ────────────────────────────────────────────────
    gene_overlaps: dict = {}
    if genes_df is not None:
        gene_bed = pybedtools.BedTool.from_dataframe(genes_df)
        gene_intersect = win_bed.intersect(gene_bed, wao=True)
        for feat in gene_intersect:
            key = (feat.chrom, int(feat.start), int(feat.end))
            bp = int(feat.fields[-1])
            if bp > 0:
                gene_overlaps[key] = gene_overlaps.get(key, 0) + bp

    # ── Assign label per window ────────────────────────────────────────────────
    results = []
    for _, row in target_df.iterrows():
        key       = (row["chrom"], int(row["start"]), int(row["end"]))
        win_size  = int(row["end"]) - int(row["start"])
        rep_ovlp  = rep_overlaps.get(key, {})
        total_rep_bp = sum(rep_ovlp.values())
        rep_fraction = total_rep_bp / win_size

        if rep_fraction >= min_fraction:
            dominant_class = max(rep_ovlp, key=rep_ovlp.get)
            dominant_bp    = rep_ovlp[dominant_class]
            if dominant_bp / win_size >= min_fraction:
                ref_label    = dominant_class
                overlap_bp   = dominant_bp
                overlap_frac = dominant_bp / win_size
            else:
                # Multiple repeat classes, none dominant enough
                ref_label    = "Ambiguous"
                overlap_bp   = total_rep_bp
                overlap_frac = rep_fraction
        else:
            gene_bp      = gene_overlaps.get(key, 0)
            gene_fraction = gene_bp / win_size
            overlap_bp   = 0
            overlap_frac = 0.0
            if genes_df is not None and gene_fraction >= min_fraction:
                ref_label = "Genic"
            else:
                ref_label = "Intergenic"

        results.append({
            "ref_label":          ref_label,
            "ref_overlap_bp":     overlap_bp,
            "ref_overlap_fraction": round(overlap_frac, 4),
        })

    pybedtools.cleanup()
    return results


# ── Output writers ─────────────────────────────────────────────────────────────

def write_comparison(target_df: pd.DataFrame, ref_results: list[dict],
                     out_path: Path) -> None:
    rows = []
    for (_, trow), rrow in zip(target_df.iterrows(), ref_results):
        rows.append({
            "chrom":               trow["chrom"],
            "start":               trow["start"],
            "end":                 trow["end"],
            "target_label":        trow["label"],
            "target_confidence":   round(trow["confidence"], 4),
            "ref_label":           rrow["ref_label"],
            "ref_overlap_bp":      rrow["ref_overlap_bp"],
            "ref_overlap_fraction": rrow["ref_overlap_fraction"],
            "match":               trow["label"] == rrow["ref_label"],
        })
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    log.info(f"Per-window comparison written to {out_path}")


def write_metrics(target_labels: list, ref_labels: list,
                  n_ambiguous: int, used_gff3: bool,
                  out_path: Path) -> None:
    present_labels = sorted(set(target_labels) | set(ref_labels))

    acc = accuracy_score(ref_labels, target_labels)
    report = classification_report(
        ref_labels, target_labels,
        labels=present_labels,
        target_names=present_labels,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    # ── Per-class metrics ──────────────────────────────────────────────────────
    rows = []
    for label in present_labels:
        m = report.get(label, {})
        rows.append({
            "class":     label,
            "precision": round(m.get("precision", 0), 4),
            "recall":    round(m.get("recall",    0), 4),
            "f1_score":  round(m.get("f1-score",  0), 4),
            "support":   int(m.get("support",     0)),
        })
    for avg in ("macro avg", "weighted avg"):
        m = report.get(avg, {})
        rows.append({
            "class":     avg,
            "precision": round(m.get("precision", 0), 4),
            "recall":    round(m.get("recall",    0), 4),
            "f1_score":  round(m.get("f1-score",  0), 4),
            "support":   int(m.get("support",     0)),
        })
    rows.append({
        "class":     "overall_accuracy",
        "precision": round(acc, 4),
        "recall":    "",
        "f1_score":  "",
        "support":   len(target_labels),
    })

    metrics_df = pd.DataFrame(rows)

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(ref_labels, target_labels, labels=present_labels)
    cm_df = pd.DataFrame(cm, index=present_labels, columns=present_labels)
    cm_df.index.name = "ref \\ target"

    with open(out_path, "w") as fh:
        fh.write("# ShoggoTEh annotation comparison metrics\n")
        fh.write(f"# Gene annotations used: {used_gff3}\n")
        fh.write(f"# Windows evaluated: {len(target_labels):,}\n")
        fh.write(f"# Windows excluded (ambiguous reference): {n_ambiguous:,}\n")
        fh.write("#\n")
        fh.write("## Per-class and overall metrics\n")
        fh.write(metrics_df.to_csv(sep="\t", index=False))
        fh.write("\n## Confusion matrix (rows=reference, cols=target)\n")
        fh.write(cm_df.to_csv(sep="\t"))

    log.info(f"Metrics written to {out_path}")
    log.info(f"Overall accuracy: {acc:.4f}")
    return acc


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare ShoggoTEh TE predictions against a reference annotation"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("-t", "--target",    required=True,
                        help="Target BED file (ShoggoTEh predict.py output)")
    parser.add_argument("-r", "--reference", required=True,
                        help="Reference BED file (e.g. EarlGrey filteredRepeats.bed)")
    parser.add_argument("--gff3",            default=None,
                        help="Gene annotation GFF3 for Genic/Intergenic resolution (optional)")
    parser.add_argument("--outdir",          required=True,
                        help="Output directory")
    parser.add_argument("--prefix",          default=None,
                        help="Output filename prefix (default: target BED stem)")
    parser.add_argument("--min_fraction",    type=float, default=0.5,
                        help="Min overlap fraction to assign a reference label (default: 0.5)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate inputs and print what would run, then exit without executing")
    args = parser.parse_args()

    t_start = time.monotonic()

    target_path = Path(args.target)
    outdir      = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or target_path.stem

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

    if not target_path.exists():
        print(f"ERROR: --target not found: {target_path}", file=sys.stderr)
        sys.exit(1)
    reference_path = Path(args.reference)
    if not reference_path.exists():
        print(f"ERROR: --reference not found: {reference_path}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        log.info("Dry run — no steps will be executed")
        log.info(f"  target      : {target_path}")
        log.info(f"  reference   : {reference_path}")
        log.info(f"  gff3        : {args.gff3 or '(none)'}")
        log.info(f"  outdir      : {outdir}/")
        log.info(f"  prefix      : {prefix}")
        log.info(f"  min_fraction: {args.min_fraction}")
        log.info("  Steps that would run: load inputs → intersect → write comparison TSV → write metrics TSV")
        log.info("  Exiting (--dry_run).")
        sys.exit(0)

    # ── Load inputs ────────────────────────────────────────────────────────────
    target_df  = load_target(target_path)
    repeats_df = load_reference_repeats(reference_path)
    genes_df   = load_genes(Path(args.gff3)) if args.gff3 else None

    if genes_df is None:
        log.info("No GFF3 provided — non-repeat windows will be labelled Intergenic")

    # ── Assign reference labels ────────────────────────────────────────────────
    log.info("Intersecting target windows with reference annotations ...")
    ref_results = assign_reference_labels(
        target_df, repeats_df, genes_df, args.min_fraction
    )

    # ── Write per-window comparison ────────────────────────────────────────────
    write_comparison(target_df, ref_results,
                     outdir / f"{prefix}_comparison.tsv")

    # ── Compute metrics (exclude ambiguous windows) ────────────────────────────
    target_labels, ref_labels = [], []
    n_ambiguous = 0
    for (_, trow), rrow in zip(target_df.iterrows(), ref_results):
        if rrow["ref_label"] == "Ambiguous":
            n_ambiguous += 1
            continue
        target_labels.append(trow["label"])
        ref_labels.append(rrow["ref_label"])

    n_windows_evaluated = len(target_labels)
    log.info(f"Windows evaluated: {n_windows_evaluated:,} | "
             f"excluded (ambiguous reference): {n_ambiguous:,}")

    if not target_labels:
        log.error("No evaluable windows — check that target and reference share "
                  "the same chromosome names and coordinate space.")
        sys.exit(1)

    # ── Write metrics ──────────────────────────────────────────────────────────
    overall_accuracy = write_metrics(
        target_labels, ref_labels,
        n_ambiguous=n_ambiguous,
        used_gff3=genes_df is not None,
        out_path=outdir / f"{prefix}_metrics.tsv",
    )

    log.info("Done.")

    elapsed_s = time.monotonic() - t_start
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_mem_mb = ru.ru_maxrss / (1024 * 1024) if platform.system() == "Darwin" else ru.ru_maxrss / 1024
    log.info(f"Wall-clock time   : {elapsed_s:.1f} s ({elapsed_s/60:.1f} min)")
    log.info(f"Peak memory (RSS) : {peak_mem_mb:.1f} MB")

    summary = {
        "date":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION,
        "target":    str(args.target),
        "reference": str(args.reference),
        "n_windows_evaluated": n_windows_evaluated,
        "n_ambiguous":         n_ambiguous,
        "overall_accuracy":    round(overall_accuracy, 4) if overall_accuracy is not None else None,
        "parameters": {
            "min_fraction": args.min_fraction,
            "gff3":         args.gff3,
        },
        "resource_usage": {
            "wall_clock_s": round(elapsed_s, 1),
            "peak_mem_mb":  round(peak_mem_mb, 1),
        },
    }
    summary_path = Path(args.outdir) / "run_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    log.info(f"Run summary written to {summary_path}")


if __name__ == "__main__":
    main()
