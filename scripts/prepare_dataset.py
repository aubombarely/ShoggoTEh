#!/usr/bin/env python3
"""
prepare_dataset.py  —  ShoggoTEh data preparation

Slides a fixed-size window across each genome, labels each chunk by majority
overlap with EarlGrey repeat annotations and (optionally) gene annotations,
and writes one Parquet file per species ready for embedding generation.

Usage:
    python prepare_dataset.py --species_tsv data/species.tsv --outdir data/chunks/

Output columns per Parquet file:
    species, chrom, start, end, sequence, label, repeat_fraction
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pybedtools
from codecarbon import EmissionsTracker
from pyfaidx import Fasta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Repeat class mapping ───────────────────────────────────────────────────────
REPEAT_CLASS_MAP = {
    "LTR":           "LTR",
    "DNA":           "DNA",
    "LINE":          "LINE",
    "SINE":          "SINE",
    "RC":            "DNA",            # Rolling Circle / Helitron
    "Unknown":       "Unknown_repeat",
    "Satellite":     "Other_repeat",
    "Simple_repeat": "Other_repeat",
    "Low_complexity":"Other_repeat",
}

def map_repeat_class(classification: str) -> str:
    top = classification.split("/")[0]
    return REPEAT_CLASS_MAP.get(top, "Other_repeat")


# ── Window generation ──────────────────────────────────────────────────────────
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
        # Include a final window anchored to the end if remainder >= half chunk
        remainder = chrom_len - start
        if 0 < remainder >= chunk_size // 2:
            windows.append((chrom, chrom_len - chunk_size, chrom_len))
    return windows


# ── Repeat labeling ────────────────────────────────────────────────────────────
def load_repeats(bed_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        bed_path, sep="\t", header=None, usecols=[0, 1, 2, 3],
        names=["chrom", "start", "end", "classification"],
        dtype={"chrom": str, "start": int, "end": int, "classification": str},
    )
    df["label"] = df["classification"].apply(map_repeat_class)
    return df


def label_windows(windows: list, repeats_df: pd.DataFrame,
                  genes_df: pd.DataFrame | None,
                  min_fraction: float, chunk_size: int) -> dict:
    """
    Return a dict mapping (chrom, start, end) → label string.
    Windows where no single class reaches min_fraction are mapped to None
    and will be discarded.
    """
    win_df = pd.DataFrame(windows, columns=["chrom", "start", "end"])
    win_bed = pybedtools.BedTool.from_dataframe(win_df)
    rep_bed = pybedtools.BedTool.from_dataframe(
        repeats_df[["chrom", "start", "end", "label"]]
    )

    # wao: write original A entry + overlap bp for each B feature
    intersect = win_bed.intersect(rep_bed, wao=True)

    # Accumulate bp per repeat class per window
    overlaps: dict = {}
    for feat in intersect:
        key = (feat.chrom, int(feat.start), int(feat.end))
        rep_class = feat.fields[6]
        bp = int(feat.fields[7])
        if rep_class != "." and bp > 0:
            overlaps.setdefault(key, {})
            overlaps[key][rep_class] = overlaps[key].get(rep_class, 0) + bp

    # Assign labels
    window_labels: dict = {}
    for win in windows:
        key = tuple(win)
        ovlp = overlaps.get(key, {})
        total_repeat_bp = sum(ovlp.values())
        repeat_fraction = total_repeat_bp / chunk_size

        if repeat_fraction >= min_fraction:
            dominant_class = max(ovlp, key=ovlp.get)
            dominant_bp = ovlp[dominant_class]
            if dominant_bp / chunk_size >= min_fraction:
                window_labels[key] = (dominant_class, repeat_fraction)
            else:
                window_labels[key] = None  # mixed — discard
        else:
            window_labels[key] = ("Non_repeat", repeat_fraction)

    # Refine Non_repeat → Genic / Intergenic if gene annotations available
    if genes_df is not None:
        non_rep = [w for w in windows if (window_labels.get(tuple(w)) or (None,))[0] == "Non_repeat"]
        if non_rep:
            nr_bed = pybedtools.BedTool.from_dataframe(
                pd.DataFrame(non_rep, columns=["chrom", "start", "end"])
            )
            gene_bed = pybedtools.BedTool.from_dataframe(
                genes_df[["chrom", "start", "end"]]
            )
            gene_intersect = nr_bed.intersect(gene_bed, wao=True)
            gene_overlap_bp: dict = {}
            for feat in gene_intersect:
                key = (feat.chrom, int(feat.start), int(feat.end))
                bp = int(feat.fields[-1])
                if bp > 0:
                    gene_overlap_bp[key] = gene_overlap_bp.get(key, 0) + bp

            for win in non_rep:
                key = tuple(win)
                _, rep_frac = window_labels[key]
                bp = gene_overlap_bp.get(key, 0)
                if bp / chunk_size >= min_fraction:
                    window_labels[key] = ("Genic", rep_frac)
                else:
                    window_labels[key] = ("Intergenic", rep_frac)

    return window_labels


# ── Main per-species processing ────────────────────────────────────────────────
def process_species(species_id: str, fasta_path: str, bed_path: str,
                    gff3_path: str | None, chunk_size: int, stride: int,
                    min_fraction: float, max_n_fraction: float,
                    outdir: Path) -> None:

    out_path = outdir / f"{species_id}.parquet"
    if out_path.exists():
        log.info(f"{species_id}: already exists, skipping")
        return

    log.info(f"{species_id}: loading genome")
    fasta = Fasta(fasta_path)

    log.info(f"{species_id}: generating windows (chunk={chunk_size}, stride={stride})")
    windows = make_windows(fasta, chunk_size, stride)
    log.info(f"{species_id}: {len(windows):,} windows generated")

    log.info(f"{species_id}: loading repeat annotations")
    repeats_df = load_repeats(bed_path)

    genes_df = None
    if gff3_path and Path(gff3_path).exists():
        log.info(f"{species_id}: loading gene annotations")
        gff = pd.read_csv(
            gff3_path, sep="\t", comment="#", header=None,
            names=["chrom","source","feature","start","end",
                   "score","strand","frame","attributes"],
            dtype={"chrom": str},
        )
        genes_df = gff[gff["feature"] == "gene"][["chrom", "start", "end"]].copy()
        genes_df["start"] = genes_df["start"] - 1  # GFF3 is 1-based

    log.info(f"{species_id}: labeling windows")
    window_labels = label_windows(windows, repeats_df, genes_df,
                                  min_fraction, chunk_size)

    log.info(f"{species_id}: extracting sequences")
    records = []
    skipped_n, skipped_mixed = 0, 0

    for win in windows:
        key = tuple(win)
        result = window_labels.get(key)
        if result is None:
            skipped_mixed += 1
            continue
        label, repeat_fraction = result
        chrom, start, end = win
        seq = str(fasta[chrom][start:end]).upper()
        n_frac = seq.count("N") / len(seq)
        if n_frac > max_n_fraction:
            skipped_n += 1
            continue
        records.append({
            "species":          species_id,
            "chrom":            chrom,
            "start":            start,
            "end":              end,
            "sequence":         seq,
            "label":            label,
            "repeat_fraction":  round(repeat_fraction, 4),
        })

    log.info(
        f"{species_id}: {len(records):,} chunks kept | "
        f"{skipped_n:,} skipped (N content) | "
        f"{skipped_mixed:,} skipped (mixed repeat)"
    )

    # Label distribution summary
    df = pd.DataFrame(records)
    dist = df["label"].value_counts().to_dict()
    log.info(f"{species_id}: label distribution — {dist}")

    df.to_parquet(out_path, index=False)
    log.info(f"{species_id}: written to {out_path}")

    pybedtools.cleanup()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Prepare labeled sequence chunks for ShoggoTEh"
    )
    parser.add_argument("--species_tsv",    required=True,
                        help="TSV file: species_id, fasta, bed, gff3 (gff3 optional / NA)")
    parser.add_argument("--outdir",         required=True,
                        help="Output directory for per-species Parquet files")
    parser.add_argument("--chunk_size",     type=int,   default=5000,
                        help="Window size in bp (default: 5000)")
    parser.add_argument("--overlap",        type=float, default=0.5,
                        help="Fractional overlap between windows (default: 0.5)")
    parser.add_argument("--min_fraction",   type=float, default=0.5,
                        help="Min fraction of window to assign a label (default: 0.5)")
    parser.add_argument("--max_n_fraction", type=float, default=0.1,
                        help="Max N fraction allowed in a chunk (default: 0.1)")
    args = parser.parse_args()

    stride = int(args.chunk_size * (1 - args.overlap))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    species_df = pd.read_csv(
        args.species_tsv, sep="\t", comment="#", header=None,
        names=["species_id", "fasta", "bed", "gff3"],
    )

    tracker = EmissionsTracker(
        project_name="ShoggoTEh_prepare_dataset",
        output_dir=str(log_dir),
        log_level="error",
    )
    tracker.start()

    for _, row in species_df.iterrows():
        gff3 = row.get("gff3")
        gff3 = None if pd.isna(gff3) or str(gff3).strip().upper() == "NA" else str(gff3)
        try:
            process_species(
                species_id=str(row["species_id"]),
                fasta_path=str(row["fasta"]),
                bed_path=str(row["bed"]),
                gff3_path=gff3,
                chunk_size=args.chunk_size,
                stride=stride,
                min_fraction=args.min_fraction,
                max_n_fraction=args.max_n_fraction,
                outdir=outdir,
            )
        except Exception as exc:
            log.error(f"{row['species_id']}: FAILED — {exc}")
            continue

    emissions = tracker.stop()
    log.info(f"Carbon footprint: {emissions:.6f} kg CO2 equivalent")
    log.info("All species processed.")


if __name__ == "__main__":
    main()
