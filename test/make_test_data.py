#!/usr/bin/env python3
"""
make_test_data.py — Generate synthetic ShoggoTEh test data.

Creates a small deterministic dataset (random.seed(42)) suitable for testing
prepare_dataset.py without requiring any real genomes or large downloads.
All Hyena-DNA / GPU steps (generate_embeddings.py, train_classifier.py,
predict.py) still require the model weights and are not covered by this
quick test.

Output (written next to this script in test/):
    genomes/test_species.fa         3 chromosomes, ~75 kb total
    repeats/test_species.bed        LTR / DNA / LINE / SINE / Unknown_repeat intervals
    gff3/test_species.gff3          12 gene models
    species.tsv                     Single-row species table pointing to the above
"""

import random
from pathlib import Path

random.seed(42)

HERE = Path(__file__).parent

# ── Helpers ───────────────────────────────────────────────────────────────────

def rand_seq(n: int, gc: float = 0.45) -> str:
    at = (1 - gc) / 2
    weights = [at, gc / 2, gc / 2, at]
    return "".join(random.choices("ACGT", weights=weights, k=n))


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i+60] + "\n")


def write_bed(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for row in rows:
            fh.write("\t".join(str(x) for x in row) + "\n")


def write_gff3(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("##gff-version 3\n")
        for row in rows:
            fh.write("\t".join(str(x) for x in row) + "\n")


# ── Genome ────────────────────────────────────────────────────────────────────

CHROMS = [("Chr1", 40_000), ("Chr2", 25_000), ("Chr3", 15_000)]
genome = [(name, rand_seq(size)) for name, size in CHROMS]
write_fasta(HERE / "genomes" / "test_species.fa", genome)
print(f"  genome → {HERE/'genomes'/'test_species.fa'}  "
      f"({sum(s for _,s in CHROMS):,} bp, {len(CHROMS)} chromosomes)")

# ── Repeat annotations (BED) ──────────────────────────────────────────────────

REPEAT_TYPES = [
    "LTR/Gypsy", "LTR/Copia",
    "DNA/hAT", "DNA/Tc1",
    "LINE/L1",
    "SINE/MIR",
    "Unknown",
]

bed_rows = []
for chrom, chrom_len in CHROMS:
    pos = 500
    while pos + 1000 < chrom_len:
        length = random.randint(500, 3000)
        end = min(pos + length, chrom_len - 100)
        rtype = random.choice(REPEAT_TYPES)
        score = random.randint(200, 1000)
        strand = random.choice(["+", "-"])
        bed_rows.append((chrom, pos, end, rtype, score, strand))
        pos = end + random.randint(200, 2000)

write_bed(HERE / "repeats" / "test_species.bed", bed_rows)
print(f"  repeats → {HERE/'repeats'/'test_species.bed'}  ({len(bed_rows)} intervals)")

# ── Gene models (GFF3) ────────────────────────────────────────────────────────

gff_rows = []
gene_id = 0
for chrom, chrom_len in CHROMS:
    n_genes = 4 if chrom_len >= 25_000 else 2
    step = chrom_len // (n_genes + 1)
    for i in range(n_genes):
        gene_id += 1
        gstart = step * (i + 1)
        gend   = gstart + random.randint(1500, 4000)
        gend   = min(gend, chrom_len - 100)
        strand = random.choice(["+", "-"])
        gff_rows.append((
            chrom, "test", "gene",
            gstart + 1, gend,               # GFF3 is 1-based
            ".", strand, ".",
            f"ID=gene{gene_id:04d};Name=gene{gene_id:04d}",
        ))
        # Single exon / CDS for simplicity
        cds_s = gstart + 100
        cds_e = gend   - 100
        gff_rows.append((
            chrom, "test", "mRNA",
            gstart + 1, gend, ".", strand, ".",
            f"ID=mRNA{gene_id:04d};Parent=gene{gene_id:04d}",
        ))
        gff_rows.append((
            chrom, "test", "exon",
            cds_s + 1, cds_e, ".", strand, ".",
            f"ID=exon{gene_id:04d};Parent=mRNA{gene_id:04d}",
        ))
        gff_rows.append((
            chrom, "test", "CDS",
            cds_s + 1, cds_e, ".", strand, "0",
            f"ID=CDS{gene_id:04d};Parent=mRNA{gene_id:04d}",
        ))

write_gff3(HERE / "gff3" / "test_species.gff3", gff_rows)
print(f"  GFF3    → {HERE/'gff3'/'test_species.gff3'}  ({gene_id} genes)")

# ── species.tsv ───────────────────────────────────────────────────────────────

tsv_path = HERE / "species.tsv"
with open(tsv_path, "w") as fh:
    fh.write("# species_id\tfasta\tbed\tgff3\n")
    fh.write(
        "test_species\t"
        f"{(HERE / 'genomes' / 'test_species.fa').resolve()}\t"
        f"{(HERE / 'repeats' / 'test_species.bed').resolve()}\t"
        f"{(HERE / 'gff3' / 'test_species.gff3').resolve()}\n"
    )
print(f"  species → {tsv_path}")
print("\nDone. Run 'python scripts/prepare_dataset.py --species_tsv test/species.tsv --outdir test/chunks/' to test.")
