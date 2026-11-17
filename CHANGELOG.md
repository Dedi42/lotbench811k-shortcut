# Changelog

## 1.2.0 (2026-11-17)
* `code/revision2_experiments.py --parts noise`: default refit of SALE repeated with the same training seed and with three
  further training seeds (seeds 10–12), the noise baseline of the tuning check (Online Resource 1, Table S16).
* Results: `results/public/results_revision2/noise_summary.json`.
* Reproduction: 114 checks (noise baseline added); `--without-gbdt` still reproduces the 91 checks of v1.0.0.

## 1.1.0 (2026-11-15)
Revision 2 of the manuscript (new title: "Benchmarking lot-level abnormality detection in semiconductor manufacturing").
* `code/revision2_experiments.py`: lot-summary GBDT (`B3-GB`, default settings, same features, splits and folds as the
  lot-summary forest) and a tuning check of the forest and SALE with the same budget (seeds 10–12, 20 trials each).
* `code/core_experiments.py`: `B3-GB` added; forest settings and SALE learning rate, weight decay and hidden width can be
  passed as options (defaults unchanged, so the paper run is reproduced as before).
* Results: `results/public/results_revision2/` (GBDT scores of every lot and seed; `tuning_summary.json`).
* Reproduction: GBDT in the Holm families (19 / 18 detectors) and the tuning check, 108 checks; `--without-gbdt`
  reproduces the 91 checks of v1.0.0.

## 1.0.0 (2026-11-04)
First release accompanying the submission to the Journal of Intelligent Manufacturing.
* Experiment code of the paper run (code version v20).
* Public results: score of every LotBench-811K and WM-811K lot for every detector and seed.
* One-minute reproduction of all public numbers (91 checks) and of Figs. 4–7.
* Final figures of the paper (Fig1–Fig7, TIFF, 600 dpi).
* WM-811K rerun of SALE and SALE-lite without wafer-label pretraining (`code/rerun_wmlot_without_wafer_pretraining.py`).
  The paper reports this rerun (`results/public/results_wmlot_nopretrain/`); the reproduction uses it by default and checks the
  first-run values with `--with-wafer-pretraining`.
* `core_experiments.py`: "v20" added to the version list of the report check after the paper run (no effect on results); one code comment anonymized.
