# ShoggoTEh — Project Notes

Deep learning pipeline for transposable element (TE) classification in plant
genomes using Hyena-DNA sequence embeddings and a trained MLP classifier.

**Current version:** v0.2.0

This project follows the shared coding blueprint at `../CLAUDE.md`.
Apply those standards to any new work here.

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/prepare_dataset.py` | Slide 5 kb windows, label by majority overlap with EarlGrey + gene models → `data/chunks/{species}.parquet` |
| `scripts/generate_embeddings.py` | Run labeled windows through Hyena-DNA (mean-pooled last hidden state) → `data/embeddings/{species}.parquet` |
| `scripts/train_classifier.py` | Train MLP on embeddings (stratified 80/20, early stopping) → `models/plant_classifier/` |
| `scripts/predict.py` | Classify new genome windows → `{prefix}.bed`, `{prefix}_probs.tsv` |
| `scripts/compare_te_annotation.py` | Benchmark predictions against a reference BED annotation |

## External tools / models required

- **Hyena-DNA** model weights (HuggingFace: `LongSafari/hyenadna-*`)
- `torch`, `transformers`, `datasets`, `pandas`, `scikit-learn`
- `mafft` or `muscle` (not required for core pipeline)

## Blueprint compliance status

All blueprint compliance gaps resolved in v0.2.0:

- [x] Add `VERSION = "v0.2.0"` to each script
- [x] Add `--version` argument to each script
- [x] Add structured output (`logs/`) — `{outdir}/logs/Run_{script}.log`
- [x] Add run log with date/user/server/OS/command
- [x] Add optional codecarbon tracking + `--disable_co2_tracking` (`try/except ImportError` guard)
- [x] Add checkpoint/resume logic + `--force` flag (`prepare_dataset.py`, `generate_embeddings.py`)
- [x] Add `run_summary.json` with resource usage (all five scripts)
- [x] Add `test/` directory with synthetic data (`make_test_data.py`, `test/README.md`)
- [x] Add `CHANGELOG.md`
- [x] Update README with version badge and options tables (all new flags documented)

## Notes

- **logs/ location**: `Path(args.outdir) / "logs"` — self-contained per run, not relative to cwd.
- **codecarbon import**: guarded by `try/except ImportError` → `_CODECARBON_AVAILABLE` flag; all four
  applicable scripts run normally without `codecarbon` installed.
- **--force scope**: per-species skip only (`prepare_dataset.py`, `generate_embeddings.py`);
  `train_classifier.py` and `predict.py` always overwrite their outputs.
- **GPU dependency**: steps 2–5 require Hyena-DNA model weights; only step 1 is covered by the
  quick test in `test/`.
