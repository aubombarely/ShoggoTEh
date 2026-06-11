# ShoggoTEh v0.1.0

A deep learning pipeline for identifying and classifying transposable elements (TEs) in plant genomes using DNA language model embeddings.

ShoggoTEh uses [Hyena-DNA](https://github.com/HazyResearch/hyena-dna) to generate sequence embeddings from genomic windows and trains a classifier to distinguish TE classes, genic, and intergenic regions — without relying on alignment-based repeat detection.

> *Named after the Shoggoths of H.P. Lovecraft's mythos — shapeshifting entities, much like transposable elements that continuously reshape the genome.*

## Overview

```
species.tsv  (species_id, fasta, bed, gff3)
      │
      ▼
prepare_dataset.py          Slide 5 kb windows across genome,
                            label by majority overlap with EarlGrey
                            repeat annotations and gene models
      │
      ▼
generate_embeddings.py      Run labeled chunks through Hyena-DNA
                            to produce sequence embeddings
      │
      ▼
train_classifier.py         Train a classifier head on top of
                            the embeddings
      │
      ▼
predict.py                  Annotate a new genome with the
                            trained model
```

## Labels

| Label | Description |
|-------|-------------|
| `LTR` | LTR retrotransposons (Gypsy, Copia, etc.) |
| `DNA` | DNA transposons (hAT, Helitron, Tc1-Mariner, etc.) |
| `LINE` | Long interspersed nuclear elements |
| `SINE` | Short interspersed nuclear elements |
| `Unknown_repeat` | Unclassified repetitive elements |
| `Other_repeat` | Satellites, simple repeats, low complexity |
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
    --outdir data/chunks/ \
    --chunk_size 5000 \
    --overlap 0.5
```

Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `--chunk_size` | `5000` | Window size in bp |
| `--overlap` | `0.5` | Fractional overlap between windows |
| `--min_fraction` | `0.5` | Min fraction of window to assign a label |
| `--max_n_fraction` | `0.1` | Max N fraction allowed per chunk |

Output: one Parquet file per species under `data/chunks/`, with columns `species`, `chrom`, `start`, `end`, `sequence`, `label`, `repeat_fraction`.

### 3. Generate embeddings *(coming soon)*

```bash
python scripts/generate_embeddings.py \
    --chunks_dir data/chunks/ \
    --outdir data/embeddings/
```

### 4. Train classifier *(coming soon)*

```bash
python scripts/train_classifier.py \
    --embeddings_dir data/embeddings/ \
    --outdir models/plant_classifier/
```

## Carbon footprint tracking

ShoggoTEh tracks energy consumption and CO2 equivalent emissions for each script using [CodeCarbon](https://github.com/mlco2/codecarbon). Emissions are logged to `logs/emissions.csv` after each run and printed at the end of the log output.

## Configuration

All default settings are in `config/config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | `5000` | Window size in bp |
| `overlap` | `0.5` | Fractional overlap between windows |
| `min_fraction` | `0.5` | Min fraction to assign a label |
| `max_n_fraction` | `0.1` | Max N fraction per chunk |
| `hyena_model` | `hyenadna-medium-160k` | Hyena-DNA model variant |
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
