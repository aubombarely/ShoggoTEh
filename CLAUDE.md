# ShoggoTEh — Project Notes

Deep learning pipeline for transposable element (TE) classification in plant
genomes using Hyena-DNA sequence embeddings and a dense (per-bin) CNN+CRF
sequence labeler.

**Current version:** v0.3.0

This project follows the shared coding blueprint at `../CLAUDE.md`.
Apply those standards to any new work here.

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/ShoggoTEh.py` | Single-entry-point CLI (argparse subparsers, matching YogsoPROT.py's pattern). Subcommands: `prepare_dataset` (dense per-bin labeling by exact repeat/gene overlap → `{outdir}/{species}.parquet`), `generate_embeddings` (Hyena-DNA per-nucleotide hidden states pooled into per-bin embeddings, EOS token dropped → `{outdir}/{species}.parquet`), `train_classifier` (1D-CNN + linear-chain CRF sequence labeler over frozen embeddings → `{outdir}/classifier.pt`, `label_encoder.json`, `model_config.json`), `predict` (dense forward pass + Viterbi decode + run-length-encoding of consecutive same-label bins → `{prefix}.bed`, `{prefix}_bin_probs.tsv`), `compare_te_annotation` (benchmark predictions against a reference BED annotation, unchanged logic — operates on arbitrary-length intervals). |
| `scripts/analyze_te_lengths.py` | Standalone diagnostic (not part of the merged CLI) — TE length distribution per class, informs window/bin-size choices. Imports `REPEAT_CLASS_MAP`/`map_repeat_class` from `ShoggoTEh.py`. |

### Architecture: dense (per-bin) vs the old window-level pipeline

v0.3.0 replaced whole-window majority-vote classification with per-bin
(default 50bp) sequence labeling — see `../../.claude/plans/agile-chasing-willow.md`
for the full rationale. The old pipeline mean-pooled an entire 5kb window
into one vector and predicted one label for it, which structurally fails for
short, dispersed classes (proved concretely: 2,996 real SINE copies in
Zea_mays, median ~226bp, yielded zero SINE-labeled windows under the old
architecture, because a window can only be majority-labeled by a single
repeat copy if `window_size < 2 x repeat_length`). The dense redesign:

- **prepare_dataset**: every bin gets the exact label of whichever BED
  interval it falls in — no min_fraction threshold, no mixed-window discard.
- **generate_embeddings**: Hyena-DNA's per-nucleotide hidden states (already
  base-pair resolution — no BPE) are pooled into small bins instead of
  mean-pooled into one window vector; the tokenizer's trailing EOS/SEP token
  is dropped before pooling so bin coordinates map correctly to genomic
  positions.
- **train_classifier**: a light 1D-CNN (local-context smoothing) + a
  from-scratch linear-chain CRF (transition matrix + forward algorithm +
  Viterbi decode) replace the single-window MLP head. `--class_weight
  balanced` now reweights per-bin (not per-window) class frequency.
- **predict**: dense forward pass + Viterbi decode, then consecutive
  same-bin-label runs are merged (run-length-encoded) into BED intervals —
  no format change for downstream consumers.
- **compare_te_annotation**: unchanged — `assign_reference_labels()` already
  operates on arbitrary-length intervals.

## External tools / models required

- **Hyena-DNA** model weights (HuggingFace: `LongSafari/hyenadna-*`)
- `torch`, `transformers`, `pandas`, `pyarrow`, `pyfaidx`, `pybedtools`, `scikit-learn`
- System `bedtools` binary (required by `pybedtools`; not bundled by the pip
  package — install via `conda install -c bioconda bedtools` or your OS
  package manager)

## Blueprint compliance status

All blueprint compliance gaps resolved as of v0.3.0:

- [x] `VERSION = "v0.3.0"` and `--version` argument on the merged CLI
- [x] Structured output (`logs/`) — `{outdir}/logs/Run_ShoggoTEh_{command}.log`
      (filename includes the subcommand since each subcommand has its own `--outdir`)
- [x] Run log with date/user/server/OS/command
- [x] Optional codecarbon tracking + `--disable_co2_tracking` (`try/except ImportError` guard)
- [x] Checkpoint/resume logic + `--force` flag (`prepare_dataset`, `generate_embeddings`)
- [x] `run_summary.json` with resource usage (all five subcommands)
- [x] `test/` directory with synthetic data (`make_test_data.py`, `test/README.md`)
- [x] `CHANGELOG.md`
- [x] README updated with version badge and options tables (all new flags documented)

## Notes

- **logs/ location**: `Path(args.outdir) / "logs"` — self-contained per run, not relative to cwd.
- **codecarbon import**: guarded by `try/except ImportError` → `_CODECARBON_AVAILABLE` flag; all
  ML-workload subcommands run normally without `codecarbon` installed.
- **--force scope**: per-species skip only (`prepare_dataset`, `generate_embeddings`);
  `train_classifier` and `predict` always overwrite their outputs.
- **GPU dependency**: `generate_embeddings`, `train_classifier`, and `predict` require Hyena-DNA
  model weights; only `prepare_dataset` is covered by the quick test in `test/` (and additionally
  requires a system `bedtools` binary, which may not be present on all machines).
- **CRF implementation**: hand-rolled linear-chain CRF (no new pip dependency) — unit-tested for
  forward/backward gradient flow and Viterbi decode correctness, but not yet validated against a
  real trained model (no GPU/Hyena-DNA weights available during this rewrite).

---

## FAIR compliance status

- [x] `LICENSE` — MIT, added 2026-06-24
- [x] `CITATION.cff` — author, ORCID, version, keywords, repository URL
- [ ] Zenodo DOI — mint after first public release; add `doi:` field to `CITATION.cff`
- [ ] bio.tools registration — register after first public release
