#!/usr/bin/env python3
"""
analyze_te_lengths.py  —  ShoggoTEh diagnostic: TE length distribution per class

Reads the same species_tsv used by `ShoggoTEh.py prepare_dataset`, pools
every BED interval across all listed species, maps each to its broad repeat
class (reusing ShoggoTEh.py's own REPEAT_CLASS_MAP so this stays consistent
with the actual labeling pipeline), and reports the real length distribution
per class -- plus a data-driven suggested window size per class, based on
the geometric rule that a window must be < 2x an element's length for that
element to have any chance of covering >50% of the window on its own.

This is a lightweight, one-off analysis helper, not a pipeline stage --
it does not follow the full project blueprint (no checkpointing, no
codecarbon tracking, no structured run_summary.json). It only reads BED
files, so it is cheap and fast to rerun.

Usage:
    python analyze_te_lengths.py --species_tsv data/species.tsv --outdir results_te_lengths/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the exact same class mapping as the main pipeline, so window-size
# decisions are made against the same classes ShoggoTEh.py prepare_dataset
# will actually assign.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ShoggoTEh import REPEAT_CLASS_MAP, map_repeat_class  # noqa: E402

VERSION = "v0.1.0"


def load_all_lengths(species_tsv: Path) -> pd.DataFrame:
    """Read every species' BED file and return one long DataFrame of
    (species, label, length) rows."""
    species_df = pd.read_csv(
        species_tsv, sep="\t", comment="#", header=None,
        names=["species_id", "fasta", "bed", "gff3"],
    )

    frames = []
    for _, row in species_df.iterrows():
        species_id = str(row["species_id"])
        bed_path = str(row["bed"])
        if not Path(bed_path).exists():
            print(f"WARNING: {species_id}: BED not found ({bed_path}), skipping", file=sys.stderr)
            continue
        df = pd.read_csv(
            bed_path, sep="\t", header=None, usecols=[0, 1, 2, 3],
            names=["chrom", "start", "end", "classification"],
            dtype={"chrom": str, "start": int, "end": int, "classification": str},
        )
        df["label"] = df["classification"].apply(map_repeat_class)
        df["length"] = df["end"] - df["start"]
        df["species"] = species_id
        frames.append(df[["species", "label", "length"]])
        print(f"  {species_id}: {len(df):,} intervals loaded", file=sys.stderr)

    if not frames:
        print("ERROR: no BED files could be read", file=sys.stderr)
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


def summarize(all_lengths: pd.DataFrame) -> pd.DataFrame:
    """Per-class length distribution summary, plus a suggested window size."""
    rows = []
    for label, grp in all_lengths.groupby("label"):
        lengths = grp["length"].values
        pcts = np.percentile(lengths, [10, 25, 50, 75, 90, 95])
        # Geometric rule: window < 2 x element length for single-copy
        # majority to be possible. Suggest a window that clears this for
        # the class's 75th-percentile length (so most instances of the
        # class, not just the shortest ones, could in principle dominate
        # a window), rounded up to a clean number.
        p75 = pcts[3]
        suggested_window = int(np.ceil((2 * p75) / 50.0) * 50)  # round up to nearest 50bp
        rows.append({
            "label":            label,
            "n_intervals":      len(lengths),
            "mean_bp":          round(float(np.mean(lengths)), 1),
            "median_bp":        round(float(pcts[2]), 1),
            "std_bp":           round(float(np.std(lengths)), 1),
            "min_bp":           int(np.min(lengths)),
            "p10_bp":           round(float(pcts[0]), 1),
            "p25_bp":           round(float(pcts[1]), 1),
            "p75_bp":           round(float(pcts[3]), 1),
            "p90_bp":           round(float(pcts[4]), 1),
            "p95_bp":           round(float(pcts[5]), 1),
            "max_bp":           int(np.max(lengths)),
            "suggested_window_bp": suggested_window,
        })
    summary = pd.DataFrame(rows).sort_values("median_bp")
    return summary


def plot_distributions(all_lengths: pd.DataFrame, outdir: Path, formats: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "figure.dpi": 150, "savefig.dpi": 150, "figure.facecolor": "white",
    })

    # Clip at the 99th percentile per class for a readable plot -- TE
    # length distributions are heavily right-skewed (a few very long
    # elements would otherwise squash the whole plot).
    clip_val = all_lengths["length"].quantile(0.99)
    order = all_lengths.groupby("label")["length"].median().sort_values().index
    data_by_class = [
        all_lengths.loc[all_lengths["label"] == lbl, "length"].clip(upper=clip_val).values
        for lbl in order
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot(data_by_class, showmedians=True, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor("#4C9BE8")
        body.set_alpha(0.7)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel(f"Length (bp, clipped at 99th pct = {clip_val:.0f} bp)")
    ax.set_xlabel("Repeat class")
    ax.set_title("TE length distribution per class (all species pooled)")
    fig.tight_layout()

    for fmt in formats:
        out_path = outdir / f"te_length_distribution.{fmt}"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"  Plot saved: {out_path}", file=sys.stderr)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze TE length distributions per class to inform window-size choices"
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    ap.add_argument("--species_tsv", required=True, help="Same species TSV used by 'ShoggoTEh.py prepare_dataset'")
    ap.add_argument("--outdir", required=True, help="Output directory for summary TSV and plot")
    ap.add_argument("--format", default="pdf", help="Plot format(s), comma-separated (default: pdf)")
    ap.add_argument("--no_plot", action="store_true", help="Skip plotting, TSV summary only")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip().lstrip(".") for f in args.format.split(",")]

    print("Loading BED intervals for all species ...", file=sys.stderr)
    all_lengths = load_all_lengths(Path(args.species_tsv))
    print(f"\nTotal intervals loaded: {len(all_lengths):,}\n", file=sys.stderr)

    summary = summarize(all_lengths)
    summary_path = outdir / "te_length_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)

    print(summary.to_string(index=False))
    print(f"\nSummary written to {summary_path}")

    print(
        "\nNote: 'suggested_window_bp' = ceil(2 x p75_length / 50) x 50 -- the "
        "smallest window (rounded to 50bp) for which most instances of that "
        "class (not just the shortest ones) could in principle cover >50% of "
        "a window on their own. This is a starting point, not a hard rule --  "
        "wider windows remain fine for classes where the current 5kb pipeline "
        "already works well; this is most useful for classes far below 5000bp."
    )

    if not args.no_plot:
        print("\nGenerating length-distribution plot ...", file=sys.stderr)
        plot_distributions(all_lengths, outdir, formats)


if __name__ == "__main__":
    main()
