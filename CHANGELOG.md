# Changelog

All notable changes to ShoggoTEh are documented here.
Dates follow ISO 8601 (YYYY-MM-DD). Changes are grouped by version and type.

---

## [v0.10.1] — 2026-08-04

### Fixed

- **`classify_taxonomic_group` had three real clade-coverage gaps**, found
  from a real full-scale run of `build_taxonomy_manifest` against the
  177-genome collection: 9/177 species (all `status=ok`, i.e. real,
  successful NCBI lookups) fell through to `"Unclassified"` because none
  of the 17 marker clade names matched their lineage.
  - 7 species (*Annona* ×2, *Asimina*, *Cinnamomum*, *Litsea*, *Magnolia*,
    *Persea*) use NCBI's formal class name `Magnoliidae` instead of the
    informal clade name `magnoliids` the marker list checked for — the
    same kind of clade-name-vs-rank inconsistency already documented for
    `Liliopsida` in v0.10.0. Added `("Magnoliidae", "Magnoliid")`.
  - *Schisandra chinensis* is in `Austrobaileyales`, the third
    ANITA-grade (earliest-diverging angiosperm) order — only
    `Nymphaeales` and `Amborellales` were covered. Added
    `("Austrobaileyales", "Basal_angiosperm")`.
  - *Zygnema circumcarinatum* is in `Zygnematophyceae`, a charophyte-grade
    algal class not covered by the `Charophyta` marker. Added
    `("Zygnematophyceae", "Charophyte_algae")`.
  All three verified directly against the real lineage strings from the
  177-genome run (all 9 previously-unclassified species now resolve
  correctly), plus a regression check confirming `zeaMays`/
  `arabidopsisThaliana`/`physcomitrellaPatens` still classify unchanged.

## [v0.10.0] — 2026-07-29

### Added

- **New `build_taxonomy_manifest` subcommand** — first step toward a
  multi-genome balanced corpus across the real 177-genome, EarlGrey-
  annotated collection. Reverse-engineers a real binomial species name
  from the dataset's camelCase file-naming convention (`camel_to_
  binomial()`, e.g. `zeaMays` -> `Zea mays`), looks each one up against
  real NCBI Taxonomy (`query_ncbi_taxonomy()`, esearch + efetch), and
  classifies it into a broad plant taxonomic group (Monocot, Eudicot,
  Gymnosperm, Bryophyte, ...) via real NCBI lineage clade names
  (`classify_taxonomic_group()`) -- membership-based on clade *name*,
  not NCBI's own rank label, since rank assignment for these clades is
  inconsistent across records (confirmed directly: `Liliopsida`
  appears as rank "clade" for some species, "class" for others).
  Verified against real, live NCBI Taxonomy records for four real
  species spanning the groups that actually matter here (Zea mays ->
  Monocot, Oryza sativa -> Monocot, Arabidopsis thaliana -> Eudicot,
  Physcomitrella patens -> Bryophyte_moss), not synthetic data.
  Takes the same `species_tsv` format `prepare_dataset` already uses,
  so the same file drives both. Per-species checkpointed (failed
  lookups always retried on rerun, successful ones skipped unless
  `--force`) and rate-limited to NCBI's usage policy (3 req/s, 10 with
  `--ncbi_api_key`). Output is a separate `taxonomy_manifest.tsv`
  (species_id, binomial_name, taxid, order, family, taxonomic_group,
  lineage, status) meant to be joined with the species_tsv by
  `species_id` -- deliberately doesn't touch `prepare_dataset`'s
  existing format at all.
  Along the way, fixed a real `CERTIFICATE_VERIFY_FAILED` error hit on
  first real run against NCBI's API: Python's `ssl` module doesn't use
  the OS trust store the way `curl` does on this machine, so
  `urlopen()` failed against a perfectly valid public endpoint;
  `_ssl_context()` now explicitly uses `certifi`'s CA bundle (new
  dependency) rather than falling back to unverified SSL, which would
  have been a real security regression for no good reason.

## [v0.9.5] — 2026-07-28

### Added

- **Mixed-precision training for `train_dense_cnn`** (`--disable_amp` to
  turn off, on by default on CUDA). v0.9.4's cross-entropy switch was
  validated on a real run: 0.16-0.18 -> 0.57 batch/s (~3.2x), confirming
  the CRF loop was the right thing to remove -- but the remaining
  ~28h/epoch is now genuine GPU compute (24 residual blocks' Conv1d
  work at channels=128 over T=5000), not overhead. A Tesla T4's Tensor
  Cores only engage in fp16, not the fp32 this ran in by default (~65 vs
  ~8 TFLOPS), so autocast + gradient scaling (`torch.amp.autocast` /
  `torch.amp.GradScaler`) is the standard next lever for exactly this
  GPU/workload mismatch. Applied to the training batch loop (scaled
  backward + step) and the validation forward pass; Viterbi decode
  always runs in fp32 regardless of autocast (cheap relative to
  training, no need to trade precision for speed there). Auto-disables
  on non-CUDA devices (no `torch.amp` benefit on CPU/MPS). Verified
  end-to-end on CPU (correctly no-ops, no crash) -- real GPU speedup
  unmeasured, needs validation on Salvia.

## [v0.9.4] — 2026-07-28

### Changed (breaking: `train_dense_cnn`'s training objective)

- **`train_dense_cnn` now trains via plain per-base cross-entropy, not
  the CRF's `neg_log_likelihood`.** v0.9.3's fused + TorchScript-compiled
  CRF loop was validated on a real run and only helped 0.16 -> 0.18
  batch/s (~12%) -- confirming the bottleneck was never Python/dispatch
  overhead (which compilation fixes) but the forward algorithm's genuine
  *sequential dependency* (`alpha_t` depends on `alpha_{t-1}`, so 5,000
  GPU kernel launches must execute strictly in order, un-fixable by
  compiling each one better). Discussed with the user directly (two
  options: keep the exact CRF objective via a from-scratch parallel-scan
  reformulation, higher effort/risk and unvalidatable without a GPU here;
  or switch to cross-entropy, fully vectorized, zero sequential loop) --
  went with cross-entropy.
  - The CRF module and `decode()` (Viterbi) are unchanged and still used
    at **inference** time (`predict_dense_cnn` only ever called
    `decode()`, never `neg_log_likelihood` -- unaffected). Decode's own
    per-window Viterbi loop is comparatively cheap (once per prediction
    window, not once per training batch across many epochs).
  - Since nothing trains `crf.transitions`/`start_transitions`/
    `end_transitions` via gradient descent anymore, they're instead set
    once from **empirical consecutive-label statistics** in the
    (RC-augmented) training corpus (`compute_empirical_transitions()`,
    Laplace-smoothed, single vectorized pass, no training-time cost) and
    frozen (`requires_grad_(False)`) -- still gives Viterbi decode a
    real, data-driven transition-smoothing prior, just not a gradient-
    learned one. Verified: forward pass, frozen-transitions-stay-frozen
    after `optimizer.step()`, and model-parameters-do-update, all checked
    directly; `--class_weight balanced` now uses `F.cross_entropy`'s
    native per-class `weight` natively (more correct than the previous
    per-sequence-mean-weight approximation the CRF path needed).
  - `train_dense()` (the shared training loop) takes a new `compute_loss`
    callable instead of hardcoding the CRF loss, so the legacy
    `train_classifier` path (bin resolution, T~100, never a real
    bottleneck) keeps its original CRF-trained objective unchanged.
  - **Scientific impact unknown until re-measured**: this changes what
    the model optimizes for during training. Needs a full retrain +
    `compare_te_annotation` run to see whether SINE recall/overall
    accuracy hold up, improve, or regress relative to the CRF-trained
    v0.9.0-v0.9.3 baseline (overall_accuracy=0.3756, SINE recall=0.393).

## [v0.9.3] — 2026-07-28

### Changed

- **`LinearChainCRF`'s forward-algorithm loop found to be the real
  training bottleneck, not batch size.** The v0.9.2 progress logging
  immediately paid off: a real run at `--batch_size 24` on Salvia showed
  a rock-steady 0.16 batch/s (ETA ~6,067 min/epoch) regardless of batch
  size, which pointed at a per-batch, not per-sample, cost. Root cause:
  `_score`/`_forward_alg` each ran a Python `for t in range(1, T)` loop
  with `T=5000` (the v2 architecture's 1bp resolution -- 50x more
  iterations than the old bin_size=50 path's T~100), on *every* training
  batch, and that cost is roughly independent of `--batch_size` since
  each iteration processes the whole batch as one op.
  - **Fused `_score` + `_forward_alg`** into one `_forward_alg_and_score`
    loop (half the Python-loop iterations for the same math) --
    verified bit-exact against the original two-loop version (forward
    values, gradients w.r.t. both `emissions` and `transitions`, all
    `torch.allclose` to 1e-6).
  - **TorchScript-compiled the fused loop** (`torch.jit.script`) as a
    standalone module-level function, since `torch.compile()` was tried
    first and failed outright -- it tries to unroll the whole loop into
    a static graph and hit a Python `RecursionError` at `T=1000` on this
    exact loop (would be worse at the real `T=5000`). `jit.script`
    compiles the `for` loop as a real loop, not an unroll.
  - **Honesty about what's verified**: correctness (forward + gradients)
    is verified; a CPU-only isolated benchmark showed a modest ~1.25x
    speedup at `T=1000`, well short of what real training needs. GPU
    kernel-launch overhead (the actual thing being reduced) is typically
    more expensive than CPU dispatch overhead, so the real speedup on
    Salvia's T4 is unmeasured and needs validation before trusting it
    for a full run -- there was no GPU available to test this change
    directly. If this isn't enough on its own, the next lever is
    reconsidering whether the CRF's sequence-level training objective is
    worth its O(T) sequential cost at 1bp resolution versus a plain
    per-base cross-entropy loss (fully vectorized, no loop) -- discussed
    with the user but not implemented, pending real numbers from this fix.

## [v0.9.2] — 2026-07-28

### Added

- **Periodic in-epoch progress logging in `train_dense()`** (every 30s,
  matching `predict_dense_cnn`'s v0.8.1 fix). Raised when a real
  Zea_mays training run at `--batch_size 16` on the RC-augmented corpus
  (~87,000 batches in epoch 1 alone) sat completely silent for 7.5+
  hours — `train_dense()` previously only logged once a *whole* epoch
  (train + val) finished, giving no way to tell "just slow" from
  "stuck" for a run that can genuinely take hours per epoch.

### Changed

- **Vectorized two pre-training steps that were pure-Python loops over
  the whole corpus**, both real, measured bottlenecks on the same run:
  stratification-key computation (`compute_strat_key()`, replacing
  `Counter(row).most_common(1)` — measured ~10 min on 870k chunks) and
  `--class_weight balanced`'s weight computation (`compute_balanced_
  class_weights()`, replacing `sklearn.compute_class_weight` with direct
  `np.bincount` — measured ~20 min on the RC-augmented ~7-billion-
  element flattened label array, since RC augmentation doubles this
  computation's input size). Both verified to match their original
  implementations' output before replacing them (`compute_
  balanced_class_weights` matches `sklearn` exactly; `compute_strat_key`
  differs from `Counter`-based tie-breaking only on adversarial
  near-uniform synthetic data, confirmed immaterial for real, heavily
  skewed per-base label distributions where exact ties don't occur).
  Applied to both `train_dense_cnn` and the legacy `train_classifier`.

## [v0.9.1] — 2026-07-27

### Changed

- **`train_dense_cnn`'s numpy-side sequence/label arrays are now `int8`,
  not `int64`.** Raised when the user reported the real Zea_mays training
  run using ~20% of Salvia's RAM just loading 870,816 chunks (`X`+`y` at
  `int64` is ~70GB for that corpus size, before the train/val split or RC
  augmentation even run). Nucleotide indices only ever take 5 values
  (0-4) and label indices at most 8 (0-7), and `torch.tensor(...,
  dtype=torch.long)` already does the int64 upcast right at `DataLoader`
  construction — carrying `int64` through loading, splitting, RC
  augmentation, and balanced-corpus selection was pure 8x waste.
  `_NT_LOOKUP` and `_RC_COMPLEMENT_MAP` switched to `int8`; the
  `.astype(np.int64)` dropped from `load_dense_sequences`'s label
  construction. Verified downstream compatibility (`sklearn`'s
  `compute_class_weight`, `torch.tensor(..., dtype=torch.long)` cast) with
  a standalone check before pushing.

## [v0.9.0] — 2026-07-27

### Added

- **`train_dense_cnn` reverse-complement training augmentation**
  (`--disable_rc_augment` to turn off, on by default). Real
  `compare_te_annotation` results against the full Zea_mays B73 genome
  (`overall_accuracy=0.3756` with gene models, SINE recall 0.393 but
  precision only 0.020) prompted the question of whether the model sees
  DNA strand at all — it doesn't: the FASTA is read forward-strand only,
  with no reverse-complement generation or strand feature anywhere in the
  pipeline. Real repeat families (LTRs especially, with directional
  internal structure 5'LTR-gag-pol-env-3'LTR) occur on either genomic
  strand, so a strand-blind model has to learn every motif twice over from
  a fixed parameter budget. Fix: `reverse_complement_encoded()` doubles
  the *training* split only (validation stays unaugmented so reported
  metrics reflect real single-strand inference) by complementing each
  chunk's encoded nucleotide indices (A<->T, C<->G, N->N via
  `_RC_COMPLEMENT_MAP`) and reversing both the sequence and its per-base
  label array along the sequence axis (label identity is strand-invariant;
  only position order flips). Verified with a hand-checked round-trip
  (`ACGTN` -> `NACGT`, labels `[0,1,2,3,4]` -> `[4,3,2,1,0]`).

## [v0.8.1] — 2026-07-27

### Added

- **`predict_dense_cnn` now logs periodic progress** (segments processed,
  %, seg/s, elapsed, ETA) at most once every 30s, regardless of
  `--batch_size`. Found live on a real genome-scale Zea_mays run: the only
  prior signal between per-chromosome log lines was total silence, and at
  the CRF-decode-bound speeds already flagged in v0.8.0, a single
  chromosome can take a long time — there was no way to distinguish
  slow-but-working from stalled, and no way to compare throughput before
  and after a `--batch_size` change without waiting for a whole
  chromosome to finish. Total segment count is now computed up front
  (tiling every chromosome before processing starts — cheap, pure
  arithmetic) so the progress line has a real denominator from the start.

### Changed

- `run_predict_dense_cnn()` tiles all chromosomes into `chrom_segments`
  once, up front, and reuses those segments in the main processing loop
  instead of recomputing `make_predict_segments()` per chromosome.

---

## [v0.8.0] — 2026-07-26

### Added

- **`predict_dense_cnn`: v2 genome-scale inference, no embedding step**
  (third and final piece of the v2 pipeline restructuring — the
  train/predict side is now complete; see
  `.claude/plans/agile-chasing-willow.md`). `make_predict_segments()`
  tiles each chromosome/scaffold into non-overlapping *output* segments,
  each backed by a (possibly larger) inference window extending past the
  segment by up to `--overlap`/2 bases on each side for extra CRF/
  receptive-field context — the standard overlap-trim pattern, verified
  to give full, gap-free, non-overlapping coverage across a range of
  chromosome lengths including ones shorter than the window itself (the
  fully-convolutional `DilatedResidualCNN` has no fixed-length assumption
  anywhere, so short scaffolds need no padding). Same-length windows
  (true for every window in a chromosome except its last) are batched
  together before the forward pass + CRF decode, which amortizes the
  CRF's sequential Python-loop decode cost across the whole batch, not
  just the model's own compute. Predictions are merged into BED intervals
  via a new **streaming** run-length-encoder (`_stream_rle_step()`)
  instead of the legacy `rle_encode_bins()`, which requires materializing
  one dict per base first — at genome scale (potentially billions of
  bases) that would reproduce the exact Python-object-boxing memory
  blowup already fixed once this session for embeddings; the streaming
  version holds only the single currently-open interval plus the final
  merged-interval list (bounded by real TE element count, not genome
  length). Verified equivalent to `rle_encode_bins()` on 500 randomized
  synthetic records (byte-identical output), and verified end-to-end on
  real trained-model output: predicted a full 200kb synthetic genome in
  1.2s wall-clock, recovering the injected ground-truth LTR/DNA/Intergenic
  pattern at 99.81% base-level agreement (per-class recall ~0.998).
  `--window_size` defaults to 5000 (not the plan's original 24kb target)
  with the reasoning logged directly in the flag's own help text: the
  CRF's forward/decode are sequential Python loops over sequence length,
  so wall-clock cost scales with window size; 5000 already exceeds little
  useful context relative to the model's own ~12kb receptive field at the
  default `--n_cycles 3`, and keeps predict practical until that loop is
  optimized (windowed/chunked Viterbi, already flagged as a known
  follow-up in v0.7.0). Increase once benchmarked on real data.

---

## [v0.7.0] — 2026-07-26

### Added

- **`train_dense_cnn`: v2 end-to-end dense CNN+CRF, no pretrained backbone**
  (second step of the v2 architecture redesign, see
  `.claude/plans/agile-chasing-willow.md`). A new `DilatedResidualCNN`
  (`_build_dilated_cnn_model()`) operates directly on raw nucleotide
  sequence (a small learned embedding, `Embedding(5, embed_dim)` for
  A/C/G/T/N) through stride-1/same-padding dilated residual blocks
  (`Conv1d -> GroupNorm -> GELU -> Conv1d -> GroupNorm -> residual add ->
  GELU`, dilation schedule `1,2,4,8,16,32,64,128` repeated `--n_cycles`
  times, ~12kb receptive field at the default 3 cycles) — no block ever
  downsamples, so there is no reinflation step to undo at prediction time,
  unlike the legacy 50bp-bin path. Feeds the **existing, unmodified**
  `LinearChainCRF` (already validated this session) at true 1bp resolution.
  New `load_dense_sequences()` loads a `--bin_size 1` `prepare_dataset`
  corpus (raw sequence + `base_labels_bytes`, decoded via the new
  `encode_sequence()` — a vectorized 256-entry ASCII lookup table, not a
  per-character Python loop). Reuses `train_dense()`, `_select_balanced_chunks()`
  (`--balanced_corpus`), and `--class_weight balanced` unchanged, since all
  three are already architecture-agnostic. Verified via a real (not just
  shape-check) CPU training run on synthetic data with a designed learnable
  signal (GC-rich = LTR, AT-rich = DNA): loss dropped monotonically
  (2805.8 -> 43.9 over 8 epochs), val bin-accuracy reached 0.998, and
  per-class precision/recall for the injected classes were ~0.998 —
  confirming correct forward/backward pass and CRF integration at 1bp
  resolution end-to-end, not just that the code runs.
- Known follow-up (not blocking, already flagged in the v2 plan): the
  CRF's `_forward_alg`/`decode` are sequential Python loops over sequence
  length, so wall-clock cost scales with `chunk_size` — measured ~91s for
  8 epochs on 160 tiny chunks at `seq_len=2000` on CPU. Mitigation
  (windowed/chunked Viterbi) deferred until real Zea_mays pilot numbers are
  in hand, per the plan's sequencing.

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
