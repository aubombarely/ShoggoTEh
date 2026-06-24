# ShoggoTEh

<p align="center">
  <img src="assets/shoggoteh_logo.svg" width="260" alt="ShoggoTEh logo"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-v0.2.0-teal" alt="Version v0.2.0"/>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey" alt="Platform"/>
</p>

<p align="center">
  <a href="CHANGELOG.md">Changelog</a>
</p>

A deep learning pipeline for identifying and classifying transposable elements (TEs) in plant genomes using DNA language model embeddings.

ShoggoTEh uses [Hyena-DNA](https://github.com/HazyResearch/hyena-dna) to generate sequence embeddings from genomic windows and trains a classifier to distinguish TE classes, genic, and intergenic regions — without relying on alignment-based repeat detection.

> *Named after the Shoggoths of H.P. Lovecraft's mythos — shapeshifting entities, much like transposable elements that continuously reshape the genome.*

## Overview

```
species.tsv  (species_id, fasta, bed, gff3)
      │
      ▼
prepare_dataset.py          Slide 5 kb windows across each genome,
                            label by majority overlap with EarlGrey
                            repeat annotations and gene models.
                            Output: data/chunks/{species}.parquet
      │
      ▼
generate_embeddings.py      Run labeled chunks through Hyena-DNA
                            (mean-pooled last hidden state).
                            Output: data/embeddings/{species}.parquet
      │
      ▼
train_classifier.py         Train a one-hidden-layer MLP on the
                            embeddings. Stratified 80/20 split,
                            early stopping, per-class report.
                            Output: models/plant_classifier/
      │
      ▼
predict.py                  Slide windows across a new genome,
                            embed, classify, and write results.
                            Output: {prefix}.bed, {prefix}_probs.tsv


compare_te_annotation.py    Compare predictions against a reference
  -t target.bed               annotation (e.g. EarlGrey). Assigns a
  -r reference.bed            reference label per window by majority
  [--gff3 genes.gff3]         bp overlap and computes per-class and
                              overall metrics.
                              Output: {prefix}_comparison.tsv,
                                      {prefix}_metrics.tsv
```

## Labels

| Label | Description |
|-------|-------------|
| `LTR` | LTR retrotransposons (Gypsy, Copia, etc.) |
| `DNA` | DNA transposons (hAT, Helitron, Tc1-Mariner, etc.) |
| `LINE` | Long interspersed nuclear elements |
| `SINE` | Short interspersed nuclear elements |
| `Unknown_repeat` | Unclassified repetitive elements |
| `Other_repeat` | Satellites, simple repeats, low complexity (training only) |
| `Genic` | Regions overlapping gene models (requires GFF3) |
| `Intergenic` | Non-repetitive, non-genic regions |

## Installation

```bash
conda env create -f envs/shoggoTEh.yaml
conda activate shoggoTEh
```

## Usage

### 1. Prepare `data/species.tsv`

Tab-separated file with one species per line:

```
# species_id<TAB>fasta<TAB>bed<TAB>gff3
Arabidopsis_thaliana    /path/to/genome.fa    /path/to/filteredRepeats.bed    /path/to/annotation.gff3
Oryza_sativa            /path/to/genome.fa    /path/to/filteredRepeats.bed    NA
```

- `bed` should be the EarlGrey `filteredRepeats.bed` output
- `gff3` is optional — use `NA` if not available

### 2. Prepare the dataset

```bash
python scripts/prepare_dataset.py \
    --species_tsv data/species.tsv \
    --outdir data/chunks/
```

| Option | Default | Description |
|--------|---------|-------------|
| `--chunk_size` | `5000` | Window size in bp |
| `--overlap` | `0.5` | Fractional overlap between windows |
| `--min_fraction` | `0.5` | Min fraction of window covered to assign a label |
| `--max_n_fraction` | `0.1` | Max N fraction allowed per chunk |
| `--force` | off | Reprocess all species even if output Parquet already exists |
| `--dry_run` | off | Validate inputs, print steps, exit without running |
| `--disable_co2_tracking` | off | Skip codecarbon energy/CO₂ tracking for this run |

Output: one Parquet file per species under `data/chunks/` with columns `species`, `chrom`, `start`, `end`, `sequence`, `label`, `repeat_fraction`.

### 3. Generate embeddings

```bash
python scripts/generate_embeddings.py \
    --chunks_dir data/chunks/ \
    --outdir data/embeddings/
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `LongSafari/hyenadna-medium-160k-seqlen-hf` | Hyena-DNA model on HuggingFace Hub |
| `--batch_size` | `32` | Sequences per forward pass |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--force` | off | Reprocess all species even if output Parquet already exists |
| `--dry_run` | off | Validate inputs, print steps, exit without running |
| `--disable_co2_tracking` | off | Skip codecarbon energy/CO₂ tracking for this run |

Output: one Parquet file per species under `data/embeddings/` with columns `species`, `chrom`, `start`, `end`, `label`, `repeat_fraction`, `embedding`. The `sequence` column is dropped to save space; it remains in the chunks Parquet.

### 4. Train classifier

```bash
python scripts/train_classifier.py \
    --embeddings_dir data/embeddings/ \
    --outdir models/plant_classifier/
```

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | `50` | Maximum training epochs |
| `--batch_size` | `256` | Mini-batch size |
| `--lr` | `0.001` | Adam learning rate |
| `--hidden_dim` | `512` | Hidden layer width |
| `--dropout` | `0.3` | Dropout probability |
| `--val_fraction` | `0.2` | Fraction held out for validation (stratified) |
| `--patience` | `10` | Early-stopping patience in epochs |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--dry_run` | off | Validate inputs, print steps, exit without running |
| `--disable_co2_tracking` | off | Skip codecarbon energy/CO₂ tracking for this run |

Output files in `--outdir`:

| File | Description |
|------|-------------|
| `classifier.pt` | Best model weights (state dict) |
| `label_encoder.json` | Label ↔ index mapping |
| `training_metrics.tsv` | Per-epoch train/val loss and accuracy |
| `classification_report.txt` | Per-class precision, recall, F1 on the validation set |

### 5. Predict on a new genome

```bash
python scripts/predict.py \
    --fasta new_genome.fa \
    --model_dir models/plant_classifier/ \
    --outdir predictions/
```

| Option | Default | Description |
|--------|---------|-------------|
| `--prefix` | FASTA stem | Output filename prefix |
| `--hyena_model` | `LongSafari/hyenadna-medium-160k-seqlen-hf` | Hyena-DNA model |
| `--chunk_size` | `5000` | Window size in bp |
| `--overlap` | `0.5` | Fractional overlap between windows |
| `--max_n_fraction` | `0.1` | Windows above this N fraction are skipped |
| `--batch_size` | `32` | Embedding batch size |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--dry_run` | off | Validate inputs, print steps, exit without running |
| `--disable_co2_tracking` | off | Skip codecarbon energy/CO₂ tracking for this run |

Output files in `--outdir`:

| File | Description |
|------|-------------|
| `{prefix}.bed` | Top predicted class and confidence per window (`chrom start end label score .`) |
| `{prefix}_probs.tsv` | Full softmax probability for every class per window |

### 6. Compare against a reference annotation

```bash
# Repeat classes only
python scripts/compare_te_annotation.py \
    -t predictions/genome.bed \
    -r filteredRepeats.bed \
    --outdir comparisons/

# With gene models for Genic/Intergenic resolution
python scripts/compare_te_annotation.py \
    -t predictions/genome.bed \
    -r filteredRepeats.bed \
    --gff3 annotation.gff3 \
    --outdir comparisons/
```

| Option | Default | Description |
|--------|---------|-------------|
| `-t / --target` | required | ShoggoTEh BED output (`predict.py`) |
| `-r / --reference` | required | Reference BED (e.g. EarlGrey `filteredRepeats.bed`) |
| `--gff3` | None | Gene annotation GFF3 — resolves `Genic` vs `Intergenic` for non-repeat windows |
| `--prefix` | target BED stem | Output filename prefix |
| `--min_fraction` | `0.5` | Min overlap fraction to assign a reference label |
| `--dry_run` | off | Validate inputs, print steps, exit without running |

Reference labels are normalised to the ShoggoTEh label set using the same `REPEAT_CLASS_MAP` as `prepare_dataset.py` (`LTR/Gypsy` → `LTR`, `DNA/hAT` → `DNA`, etc.). Windows where no single repeat class reaches `--min_fraction` are marked `Ambiguous`, reported in the comparison file, and excluded from the metrics.

Output files in `--outdir`:

| File | Description |
|------|-------------|
| `{prefix}_comparison.tsv` | Per-window detail: target label, reference label, overlap bp and fraction, match (True/False) |
| `{prefix}_metrics.tsv` | Per-class precision, recall, F1 and support; macro and weighted averages; overall accuracy; confusion matrix |

## Run log

Every script writes a run log to `{outdir}/logs/Run_{script}.log`. The header records:

```
## Run: prepare_dataset.py
## Date: 2026-06-24 10:00:00
## User: aubombarely
## Server: my-workstation
## OS: Linux-5.15.0
## Working directory: /home/aubombarely/ShoggoTEh
## Command: python scripts/prepare_dataset.py --species_tsv data/species.tsv --outdir data/chunks/
```

All subsequent `log.info()` messages are captured in the same file.

## Run summary

Every script writes `{outdir}/run_summary.json` at the end of a successful run. It records date, script version, key inputs, result counts, parameters, and resource usage:

```json
{
  "script": "prepare_dataset.py",
  "version": "v0.2.0",
  "date": "2026-06-24T10:05:12",
  "n_species": 3,
  "n_windows_total": 12450,
  "wall_time_s": 47.3,
  "peak_rss_mb": 284.1,
  "co2_eq_kg": 0.000021
}
```

## Carbon footprint tracking

ShoggoTEh tracks energy consumption and CO₂ equivalent emissions for scripts that run ML workloads (`prepare_dataset.py`, `generate_embeddings.py`, `train_classifier.py`, `predict.py`) using [CodeCarbon](https://github.com/mlco2/codecarbon).

- `codecarbon` is an **optional dependency** — the pipeline runs normally without it.
- Pass `--disable_co2_tracking` to skip tracking for a specific run (e.g. on cluster nodes that block network access).
- Emissions are reported in the run log and included in `run_summary.json` as `co2_eq_kg`.

## Test data

A small synthetic dataset is provided in `test/` for validating `prepare_dataset.py` without downloading Hyena-DNA model weights.

```bash
# Generate (or regenerate) test files
python3 test/make_test_data.py

# Quick test — no GPU required, runs in < 30 s
conda activate shoggoTEh
python3 scripts/prepare_dataset.py \
    --species_tsv test/species.tsv \
    --outdir test/chunks/ \
    --chunk_size 5000 \
    --overlap 0.5
```

Expected output: `test/chunks/test_species.parquet` (~15–25 labeled windows), `test/chunks/logs/Run_prepare_dataset.log`, `test/chunks/run_summary.json`.

Steps 2–5 require the Hyena-DNA model weights (~1.5 GB from HuggingFace) and benefit from a GPU; see `test/README.md` for the full end-to-end test sequence.

## Configuration

All default settings are in `config/config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | `5000` | Window size in bp |
| `overlap` | `0.5` | Fractional overlap between windows |
| `min_fraction` | `0.5` | Min fraction to assign a label |
| `max_n_fraction` | `0.1` | Max N fraction per chunk |
| `hyena_model` | `LongSafari/hyenadna-medium-160k-seqlen-hf` | Hyena-DNA model variant |
| `embedding_batch_size` | `32` | Batch size for embedding generation |

## Training data

ShoggoTEh v0.1.0 is trained on EarlGrey repeat annotations from 185 plant species spanning diverse clades and repeat content profiles.

## Third-party tools

| Tool | Repository |
|------|-----------|
| Hyena-DNA | https://github.com/HazyResearch/hyena-dna |
| EarlGrey | https://github.com/TobyBaril/EarlGrey |
| CodeCarbon | https://github.com/mlco2/codecarbon |
| pyfaidx | https://github.com/mdshw5/pyfaidx |
| pybedtools | https://github.com/daler/pybedtools |

## Related tools from our group

- [DeepTE](https://github.com/LiLabAtVT/DeepTE) — TE classification using CNNs
- [annotseba](https://github.com/aubombarely/annotseba) — genome download and QC pipeline
