# Changelog

All notable changes to ShoggoTEh are documented here.
Dates follow ISO 8601 (YYYY-MM-DD). Changes are grouped by version and type.

---

## [v0.2.0] — 2026-06-24

### Added

- **`VERSION = "v0.2.0"`** and **`--version`** argument added to all five
  scripts (`prepare_dataset.py`, `generate_embeddings.py`,
  `train_classifier.py`, `predict.py`, `compare_te_annotation.py`).
- **Run log** (`{outdir}/logs/Run_{script}.log`) written on every run;
  header contains date, user, server hostname, OS, working directory, and
  the exact command. All subsequent `log.info()` messages are also captured
  in the file via a root-logger `FileHandler`.
- **`--dry_run` flag** for all five scripts — validates inputs, prints the
  steps that would execute, and exits without running anything. Useful for
  testing command syntax before submitting to a cluster scheduler.
- **`--disable_co2_tracking` flag** for `prepare_dataset.py`,
  `generate_embeddings.py`, `train_classifier.py`, and `predict.py` — opt
  out of carbon footprint tracking per run without uninstalling `codecarbon`.
- **Optional codecarbon import** — all four scripts that use `codecarbon`
  now guard the import with `try/except ImportError`; the pipeline runs
  normally if `codecarbon` is not installed.
- **`--force` flag** for `prepare_dataset.py` and `generate_embeddings.py`
  — bypasses the per-species skip logic so all species are reprocessed even
  if their output Parquet already exists.
- **Wall-clock time and peak RSS memory** logged at the end of every script
  and included in `run_summary.json`.
- **`run_summary.json`** written to `{outdir}/run_summary.json` at the end
  of every run; records date, version, key inputs, result counts, parameters,
  and resource usage (wall-clock time, peak memory, CO₂eq emissions).
- **Test data** (`test/`) — synthetic three-chromosome genome (~75 kb) with
  matching repeat BED and gene GFF3; `make_test_data.py` regenerates all
  files with `random.seed(42)`; `test/README.md` documents expected outputs
  and explains the GPU dependency for steps 2–5.

---

## [v0.1.0] — 2026-06-20

### Added

- **Initial release** of ShoggoTEh.
- **`prepare_dataset.py`** — sliding-window genome labeling using EarlGrey
  BED (repeat classes) and optional GFF3 (Genic/Intergenic); per-species
  Parquet output; chromosome-name consistency check; N-content filter.
- **`generate_embeddings.py`** — batch Hyena-DNA embedding generation
  (mean-pooled last hidden state); supports CUDA, MPS, and CPU; per-species
  Parquet output.
- **`train_classifier.py`** — one-hidden-layer MLP; stratified 80/20 split;
  Adam + early stopping; per-epoch metrics TSV; final classification report
  and confusion matrix.
- **`predict.py`** — sliding-window prediction on a new genome; BED output
  with top class and confidence; full softmax probability TSV.
- **`compare_te_annotation.py`** — compares ShoggoTEh predictions against
  a reference BED (e.g. EarlGrey); per-window detail TSV; per-class
  precision/recall/F1 and confusion matrix metrics TSV.
- Conda environment (`envs/shoggoTEh.yaml`).
- `config/config.yaml` with default pipeline parameters.
