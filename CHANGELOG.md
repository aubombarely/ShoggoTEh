# Changelog

All notable changes to ShoggoTEh are documented here.
Dates follow ISO 8601 (YYYY-MM-DD). Changes are grouped by version and type.

---

## [v0.6.0] — 2026-07-26

### Added

- **`prepare_dataset --bin_size 1`: true single-nucleotide labeling**, the
  first step of the v2 architecture redesign (see
  `.claude/plans/agile-chasing-willow.md`) — an end-to-end dense CNN+CRF
  trained directly on raw nucleotide sequence, no pretrained backbone, no
  `generate_embeddings` step. `label_bins()`'s bedtools-per-bin-row
  intersect does not scale to 1bp resolution (a single 5000bp window alone
  would need 5000 intersect rows; genome-wide, billions), so a new
  `bin_size == 1` path was added instead: `_paint_chrom_labels()` /
  `label_bases_dense()` rasterize repeat + gene intervals directly into a
  per-chromosome `int8` numpy array via vectorized slice-assignment
  (repeat > genic > intergenic precedence, matching
  `assign_reference_labels()`), processing one chromosome at a time so
  memory is bounded rather than holding the whole genome painted at once.
  Labels are stored as raw `int8` bytes (`base_labels_bytes`, using the
  fixed `DEFAULT_LABELS` vocabulary) rather than a Python list of label
  strings — a naive `list<str>` column at 1bp resolution would need one
  string object per base per window, the same class of Python-object-
  boxing memory blowup already fixed once this session for embeddings
  (`embedding_bytes`, v0.3.1). The legacy `bin_size > 1` path
  (`label_bins()`, bedtools-based) is unchanged. Verified against real
  `process_species_prepare()` runs on synthetic data: exact base-pair
  boundary correctness (no off-by-one at interval edges), correct
  repeat-over-gene precedence at overlapping positions, and performance
  (~10,850 chunks/s on a 2Mb synthetic genome with 1,096 repeat intervals,
  extrapolating to ~80s genome-wide versus the ~11.8h/genome the eliminated
  `generate_embeddings` step took).

---

## [v0.5.0] — 2026-07-26

### Added

- **`train_classifier` gains `--balanced_corpus` / `--target_bins_per_class`**
  — a quota-capped, multi-genome chunk-selection mode addressing the real
  root cause found on the Zea_mays pilot: SINE recall was 0.000 in
  validation not because loss weighting was insufficient, but because a
  single species only had 5,955 SINE bins total to learn from in the first
  place. `load_dense_embeddings` already pools every species' Parquet file
  under `--embeddings_dir` — this mode selects a balanced training subset
  from that full multi-genome pool: process classes rarest-first (by
  corpus-wide bin count) and pull in whole chunks containing that class
  until its cumulative bin count reaches `--target_bins_per_class` (default
  20,000; classes with less data available across the whole corpus fall
  short by construction, logged per class with a `[data-limited]` flag).
  Selection operates on **whole chunks**, not individual bins — the CRF
  needs contiguous bin runs to learn transition structure, so cherry-picking
  isolated bins across the genome would destroy exactly the sequence
  context it depends on. Off by default (opt-in); combine with the existing
  `--class_weight balanced` for both exposure- and loss-level correction.
  `balanced_corpus`/`target_bins_per_class` are recorded in
  `run_summary.json`'s `parameters` block for provenance. Verified via a
  synthetic unit test: rarest-class selection exhausts all available bins
  when scarcer than the target, and the returned achieved-bin-counts match
  an independent recount of the selected subset exactly.

---

## [v0.4.1] — 2026-07-26

### Fixed

- **`compare_te_annotation` reported near-zero accuracy (0.18%) on a real
  Zea_mays EarlGrey benchmark due to a `bedtools intersect -wao` field-index
  bug in `assign_reference_labels()`.** The target BED has 3 columns
  (chrom, start, end) and the reference repeats BED has 4 (chrom, start,
  end, label), so a `-wao` intersection row is
  `[A.chrom, A.start, A.end, B.chrom, B.start, B.end, B.label, overlap_bp]`
  (8 fields). The code read `fields[3]` (`B.chrom`, e.g. `"ZmB73C01"`) as
  the repeat class and `fields[4]` (`B.start`, a coordinate) as the overlap
  bp — neither is ever `"."`/`0`, so the check that should skip non-overlaps
  never fired, and every target interval was scored against a bogus
  "reference class" (a chromosome name) with a bogus bp count. Fixed to
  `fields[-2]`/`fields[-1]`, matching the (already-correct) pattern used a
  few lines below for `gene_overlaps`. Confirmed the real reference BED
  columns match this analysis exactly (`chrom, start, end,
  classification, score, strand`) via a user-supplied `head` of the file.
  This bug predates the dense-architecture rewrite and affected every
  `compare_te_annotation` run to date — re-run any prior comparison to get
  a valid accuracy number.

---

## [v0.4.0] — 2026-07-26

### Added

- **Pluggable embedding-backbone support** — `generate_embeddings` and
  `predict` gain a `--backbone {hyena,nucleotide_transformer,plantcaduceus}`
  flag (default: `hyena`, preserving current behaviour exactly when
  unspecified) so different pretrained genomic language models can be
  A/B-tested against each other (this is explicitly an experimentation
  feature, not a replacement of Hyena-DNA as the default — motivated by a
  real, measured problem: SINE recall was 0% on a real Zea_mays pilot run
  with Hyena-DNA, prompting interest in whether an alternative backbone
  does better).
  - New `BACKBONE_REGISTRY` in `scripts/ShoggoTEh.py`: `hyena` (`AutoModel`
    + `AutoTokenizer`, hidden states via `out.last_hidden_state`, default
    checkpoint `LongSafari/hyenadna-medium-160k-seqlen-hf`),
    `nucleotide_transformer` (same `AutoModel` pattern, default checkpoint
    `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`), and
    `plantcaduceus` (`AutoModelForMaskedLM` — **not** plain `AutoModel` —
    hidden states via `output_hidden_states=True` + `out.hidden_states[-1]`,
    default checkpoint `kuleshov-group/PlantCaduceus_l20`). Adding a 4th
    backbone is one registry entry (a `load_fn`, a `hidden_state_fn`, and
    optional `forward_kwargs`) — no changes needed to the shared
    batching/pooling loop.
  - **`embed_sequences_dense()` generalised to any tokenization scheme**:
    the effective tokens-per-bp ratio is now computed *dynamically per
    sequence* (`valid_content_token_len / len(raw_sequence_bp)`, after
    generically dropping whichever of CLS/BOS (leading) and EOS/SEP
    (trailing) special tokens the tokenizer added — checked via
    `tokenizer.cls_token_id` / `bos_token_id` / `eos_token_id` /
    `sep_token_id`, not hardcoded to EOS-only as before) rather than
    assuming a fixed tokens/bp ratio per backbone. This correctly handles
    Hyena-DNA's 1 token/bp, Nucleotide Transformer's ~6 tokens/bp (6-mer
    tokenization with single-nucleotide fallback tokens for
    non-multiple-of-6 remainders), and whatever scheme a future backbone
    uses, without per-backbone special-casing. New helper
    `_pool_bins_from_hidden()` recomputes each bin's token boundary from
    scratch (`round(i * bin_size * tokens_per_bp)`) so rounding error
    cannot drift across bins, and falls back gracefully (repeats the
    previous bin, or zero-fills the first) when a short trailing chunk
    yields zero tokens for a bin.
  - **`--model` (generate_embeddings) / `--hyena_model` (predict) renamed to
    `--backbone_model`** (default: `None`, meaning "use the resolved
    `--backbone`'s own sensible default checkpoint"). These old flag names
    no longer exist.
- **Backbone/checkpoint provenance tracking, to prevent silently mixing
  incompatible embedding spaces** — a classifier trained on backbone X's
  embeddings produces meaningless predictions if a later `predict` run
  embeds the input genome with backbone Y instead, with no error to signal
  the mismatch:
  - `generate_embeddings` now writes `backbone` and `backbone_model`
    columns into each species' embeddings Parquet (alongside the existing
    `hidden_dim` column), recording exactly what produced that species'
    embeddings.
  - `train_classifier`'s `load_dense_embeddings()` reads `backbone`/
    `backbone_model` back off the loaded embeddings (not from any CLI
    flag) and **errors out clearly** if a mixed-backbone corpus is detected
    (different species embedded with different backbones/checkpoints).
    `model_config.json` now records the backbone/checkpoint actually used,
    sourced from this validated embeddings metadata.
  - `predict` reads `model_config.json` and compares it against the
    resolved `--backbone`/`--backbone_model`; on any mismatch (including
    the CLI default `--backbone hyena` silently disagreeing with a model
    trained on a different backbone) it logs a `WARNING` and
    **auto-corrects** to the values recorded in `model_config.json` — the
    least-surprising choice, since predict then always matches the model
    it is actually using.
  - Older embeddings/models without recorded backbone metadata are handled
    gracefully: `load_dense_embeddings()` and `predict` both assume
    `hyena` (the previous hardcoded/legacy default) and log a note rather
    than failing.

### Changed

- `envs/shoggoTEh.yaml` unchanged — all three backbones load via the
  already-present `transformers`/`torch` dependencies; no new packages
  required.

## [v0.3.2] — 2026-07-25

### Fixed
- **`train_classifier`'s "train bin-acc" metric was actively misleading**,
  not just imprecise. It was a cheap on-GPU proxy (raw per-position
  `emissions.argmax()`, ignoring the CRF's learned transition structure)
  introduced in v0.3.1 to avoid the expensive Viterbi decode on every
  training batch. On a real training run (Zea_mays, ~870K chunks, extreme
  class imbalance with a ~378x sample weight on the rarest class) it
  collapsed to ~0.004 after epoch 1 and stayed there, while the true
  (CRF-decoded) val bin-acc held steady at ~0.58-0.61 the whole time —
  the CRF loss optimizes a joint sequence likelihood that leans on learned
  transitions to correct weak/skewed emissions, so raw emission-argmax
  diverges wildly from the actual decoded prediction under real class
  imbalance. There's no cheap fix that stays honest (e.g. reweighting the
  proxy would just be a different, still-potentially-misleading
  approximation), so the training-batch accuracy metric is removed
  entirely. `train_dense()` now only tracks train loss (the real signal
  driving backprop); val bin-acc (still the real, full CRF decode, paid
  once per validation batch per epoch, not once per training batch)
  remains the correct accuracy signal, in both the log output and
  `training_metrics.tsv` (column `train_bin_acc` removed).

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
