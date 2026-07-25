# Changelog

All notable changes to ShoggoTEh are documented here.
Dates follow ISO 8601 (YYYY-MM-DD). Changes are grouped by version and type.

---

## [v0.3.1] — 2026-07-25

### Fixed

- **`generate_embeddings` OOM on genome-scale corpora** — `df["embedding_flat"]
  = [p.flatten().tolist() for p in pooled_list]` boxed every per-bin float32
  value into an individual Python `float` object (~28 bytes each vs. 4 bytes
  raw) before writing to Parquet. At genome scale (hundreds of millions of
  bin embeddings — e.g. 870,816 Zea_mays chunks x 100 bins x 256 dims) this
  inflated RAM usage far beyond the raw array size and silently OOM-killed
  the process right after embedding finished, with no traceback (a kernel
  `SIGKILL` gives no chance to log an exception). Fixed by storing/loading
  embeddings as raw `float32` bytes (`np.ndarray.tobytes()` /
  `np.frombuffer()`) instead of nested Python float lists. Parquet column
  renamed `embedding_flat` -> `embedding_bytes` accordingly. Validated via a
  direct write/read roundtrip through real Parquet I/O (500 synthetic
  embeddings, byte-identical after roundtrip) and confirmed on the real
  Zea_mays corpus (870,816 chunks) on production hardware.
- **`train_classifier` CRF Viterbi decode dominating wall-clock time** — the
  CRF's `decode()` was called on every training batch (not just validation)
  purely to display a training-accuracy metric, and its backtrace looped in
  Python with per-element GPU-tensor indexing (`tag = int(bp[b, tag])`) --
  each such scalar extraction forces a synchronous GPU->CPU round trip,
  ~6,400 of them per batch (batch_size x n_bins). Across ~10,885 training
  batches/epoch on the real Zea_mays corpus this left the GPU mostly idle
  waiting on the CPU rather than computing (observed: 21% GPU utilization,
  641MB/15360MB VRAM used, ~59 min/epoch). Fixed two ways: (1) training-batch
  accuracy now uses a cheap on-GPU emissions-argmax proxy instead of a full
  CRF decode (loss, used for backprop, is unaffected); (2) `decode()`'s
  backtrace itself is now vectorized -- backpointers and final tags are
  transferred to CPU/numpy once each, then the backtrace uses vectorized
  numpy fancy-indexing across the batch dimension instead of nested Python
  loops with per-element GPU sync. This also speeds up validation and real
  `predict` usage, not just training. Validated as exactly equivalent to the
  original nested-loop implementation across 200 randomized trials (varying
  batch size, sequence length, class count, including edge cases).

## [v0.3.0] — 2026-07-23

### Added

- **Dense (per-bin) architecture** — replaces whole-window majority-vote
  classification with per-bin (default 50bp) sequence labeling, fixing a
  structural precision failure: a window can only be majority-labeled by a
  single repeat copy if `window_size < 2 x repeat_length`, which made the old
  5kb-window pipeline blind to short, dispersed classes (proved concretely —
  2,996 real EarlGrey-confirmed SINE copies in Zea_mays, median ~226bp,
  yielded zero SINE-labeled windows).
  - `prepare_dataset`: every bin gets the exact label of whichever
    repeat/gene interval it falls in dominantly — the old `min_fraction`
    threshold and "mixed window" discard are retired. Output rows now carry
    a `bin_labels` array (length `n_bins`) instead of a single `label`.
  - `generate_embeddings`: Hyena-DNA's per-nucleotide hidden states (already
    base-pair resolution) are pooled into small bins instead of mean-pooled
    into one window vector, producing a `[n_bins, hidden_dim]` embedding per
    chunk. The tokenizer's trailing EOS/SEP token is dropped before pooling
    so bin boundaries map correctly to genomic coordinates. New `--bin_size`
    flag (default 50).
  - `train_classifier`: the single-window MLP head is replaced by a light
    1D-CNN (local-context smoothing) + a from-scratch **linear-chain CRF**
    (transition matrix, forward-algorithm loss, Viterbi decode) — no new pip
    dependency. `--class_weight balanced` now reweights per-bin (not
    per-window) class frequency. New `--cnn_channels` and `--kernel_size`
    flags.
  - `predict`: runs the dense CNN+CRF forward pass and Viterbi decode, then
    run-length-encodes consecutive same-label bins into BED intervals
    (confidence = mean per-bin softmax probability of the assigned label
    across the merged interval). Default `--overlap` changed from `0.5` to
    `0.0` — the dense architecture does not need the old pipeline's
    redundant window overlap. Output renamed `{prefix}_bin_probs.tsv`
    (per-bin, not per-window, probability table).
  - `compare_te_annotation`: **no logic changes** — `assign_reference_labels()`
    already operates on arbitrary-length intervals, so predict's merged BED
    output needs no special handling.

### Changed

- **CLI consolidation** — `prepare_dataset.py`, `generate_embeddings.py`,
  `train_classifier.py`, `predict.py`, and `compare_te_annotation.py` are
  merged into a single entry point, `scripts/ShoggoTEh.py`, dispatched via
  argparse subparsers (matching the `YogsoPROT.py` pattern):
  `python3 scripts/ShoggoTEh.py <prepare_dataset|generate_embeddings|train_classifier|predict|compare_te_annotation> ...`.
  The five old scripts are deleted (no backwards-compatibility shims).
- **De-duplicated shared code** — `REPEAT_CLASS_MAP` / `map_repeat_class`
  (previously duplicated between `prepare_dataset.py` and
  `compare_te_annotation.py`) and the blueprint logging/checkpoint/carbon
  helpers are now single module-level definitions shared by all subcommands.
- **`DEFAULT_LABELS`** for `train_classifier` now includes `Other_repeat`
  (8 classes total, matching `compare_te_annotation`'s label set) so
  Satellite/Simple_repeat/Low_complexity bins are trained on instead of
  silently dropping the whole chunk that contains them.
- **Run log filename** — `{outdir}/logs/Run_ShoggoTEh_{command}.log` (was
  `Run_{script}.log`); the subcommand name is included since ShoggoTEh.py is
  now a single multi-subcommand entry point.
- **`analyze_te_lengths.py`** now imports `REPEAT_CLASS_MAP` /
  `map_repeat_class` from `ShoggoTEh` instead of `prepare_dataset` (module
  renamed); its role as a standalone diagnostic is unchanged.
- **`envs/shoggoTEh.yaml`**: added explicit `bedtools` conda dependency
  (the pip `pybedtools` package does not bundle the `bedtools` binary it
  shells out to); pinned `torch==2.5.1` (was unpinned `torch>=2.0`, which
  caused resolver drift in production on Salvia).

### Known limitations

- The hand-rolled linear-chain CRF is unit-tested for forward/backward
  gradient flow and Viterbi-decode correctness, but has not been validated
  against a real trained model — no GPU or Hyena-DNA weights were available
  during this rewrite.
- The CRF's `--class_weight balanced` port uses a per-sequence NLL
  reweighting heuristic (mean weight of the gold bin tags in that sequence),
  not a fully principled weighted-CRF objective; documented in code comments.

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
