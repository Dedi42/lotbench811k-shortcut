# Lot-level abnormality detection: shortcut-aware protocol, LotBench-811K and stacking

Code and public results for the paper

> Surahman, D., & Hsieh, Y.-M. *Benchmarking lot-level abnormality detection in semiconductor manufacturing.* Journal of Intelligent
> Manufacturing (submitted).

The package lets anyone

1. **check the public-data results of the paper in about a minute**, without raw data, GPU or training, from the
   released score of every test lot, detector and seed (`reproduce/`); and
2. **rerun every experiment on public data** (LotBench-811K and the WM-811K lot task) from the raw WM-811K files
   (`code/`).

The three production lines of the paper (Plant B coat/develop, Plant B wafer saw, Fab A PECVD) are confidential and
are **not** included; see [Confidentiality](#confidentiality).

---

## 1. Quick check of the paper (about 1 minute, laptop)

```bash
python -m pip install -r requirements.txt          # numpy, pandas, scipy, scikit-learn, matplotlib (tested versions)
cd reproduce
python reproduce_public_results.py                 # recomputes the public numbers and compares them with the paper
python far_theory.py                               # Beta-binomial theory of the realized false-alarm rate (Sect. 5.5)
python make_figures.py                             # redraws Figs. 4, 5 and 7 into reproduce/figures/
python make_fig6.py                                # Fig. 6 from the aggregated production values of the paper
python unweighted_average.py                       # exploratory unweighted average of the base detectors (Online Resource 1, S10)
```

`reproduce_public_results.py` ends with `114 of 114 checks passed` when the 114 public-data values listed in `reproduce/expected_values.json` (Table 3, Sections 5.1–5.2, Figs. 4–5, Online Resource S2–S5, Table S12 and Section S12) are reproduced
(see `reproduce/expected_values.json` for the list and `reproduce/reproduced/check_against_paper.txt` for the result).
It recomputes

| Paper item | What is recomputed |
|---|---|
| Table 3, Online Resource S2 | mean AUROC over ten seeds, difference to the lot-summary forest, paired *t* test and corrected resampled *t* test (Nadeau & Bengio, 2003), Holm over the 19 / 18 detectors (incl. the lot-summary GBDT) |
| Section 5.1, S3, S5 | confirmatory seeds 13–19; sensitivity without the context-revealed faults |
| Table 3, Fig. 4c, 4d | recall and realized false-alarm rate at the split-conformal threshold for ε = 5%; pooled hold/release decisions; recall per fault mechanism |
| S4 | AUROC per fault scenario |
| Table S12 | rank correlation of detector scores on abnormal lots (why stacking helps) |
| Section 5.2, S12 | tuning check: default vs tuned forest and SALE, forest − SALE with paired *t* test (from `results_revision2/tuning_summary.json`); variation of SALE between training seeds, Table S16 (from `results_revision2/noise_summary.json`) |
| Section 5.2, Fig. 5 | learning curve: ensemble minus forest per training fraction |
| Section 5.5, S11, Fig. 7 | Beta-binomial band of the realized false-alarm rate |

## 2. Full rerun on public data

Download the public data (not redistributed here; see their own licences):

* **WM-811K** `LSWMD.pkl` — Kaggle dataset `qingyi/wm811k-wafer-map` (Wu, Jang & Chen, 2015)
* **MixedWM38** `MixedWM38.npz` (or `Wafer_Map_Datasets.npz`) — from its authors (Wang et al., 2020)

```bash
cd code
cp local_config_example.py local_config.py          # set PUBLIC_DATA_DIR to the folder with the two files
python -m pip install -r requirements.txt torch       # CUDA build of PyTorch recommended for the neural detectors
python run_all.py --quick-test                      # 20–40 min smoke test on small synthetic data, must end with QUICK TEST PASSED
python run_all.py --ignore-checks                   # public parts; production steps are skipped without the partner data
```

`run_all.py` runs, in order, `core_experiments.py` (LotBench-811K generator, WM-811K lot task, all detectors,
localization, ablations), `extended_experiments.py` (MCE, variants, ten seeds, time-ordered tests),
`revision_experiments.py` (SALE-lite, MCE without SALE, three-seed ablations, learning curve) and `export_csv.py`
(per-lot score files). Interrupted runs resume. Seeds, configuration and SHA-256 hashes of all scripts are written
before a run reads data. The paper run took about 55–60 h on one GPU workstation (the estimate printed by `run_all.py`, 3–4 days, allows for slower GPUs). `SHA256SUMS.txt` lists the hashes of the released scripts; `core_experiments.py` differs from the paper run only in comments (version list of its report check, one anonymized comment, dates removed) and in the options added for revision 2 (lot-summary GBDT, tunable forest and SALE settings; defaults unchanged).

**WM-811K lot task without wafer-label pretraining.** In the paper run, the map encoder of SALE and SALE-lite was
pretrained on the wafer pattern labels of the training lots (and MixedWM38). `code/rerun_wmlot_without_wafer_pretraining.py`
repeats the same ten seeds with this pretraining switched off, so that SALE sees lot labels only like every other
detector; `reproduce/compare_wmlot_pretraining.py` compares both versions. The paper reports the rerun (its scores are in
`results/public/results_wmlot_nopretrain/`), and `reproduce_public_results.py` uses it by default; run it with
`--with-wafer-pretraining` to check the values of the first run instead (`reproduce/expected_values_with_wafer_pretraining.json`).

`results/public/paper_tables/` keeps the file names of the run (see its `README.md` for the mapping to the tables of
the paper). `code/export_csv.py` still carries internal comments of earlier manuscript drafts; they do not affect results.

**Revision 2 (lot-summary GBDT and tuning check).** `code/revision2_experiments.py` adds the lot-summary GBDT (`B3-GB`,
gradient boosting with default settings on exactly the features of the lot-summary forest) on the splits of the paper run
(`--parts gbdt_public`; scores in `results/public/results_revision2/gbdt_public/`) and on the production folds
(`--parts gbdt_prod`, partner data only), and a tuning check of the forest and SALE with the same budget
(`--parts tune`: seeds 10–12 of LotBench-811K, 15% of the training lots for validation, 20 random-search trials each,
selected setting refitted on all training lots; about 2–3 days on one GPU workstation, almost all for SALE). Its
result is `results/public/results_revision2/tuning_summary.json`. `--parts noise` repeats the default refit of SALE with the
same and with three further training seeds (about 12 h; `noise_summary.json`, Online Resource 1, Table S16). The GBDT is part of the default quick check; run
`--without-gbdt` for the Holm families of v1.0.0 (`reproduce/expected_values_without_gbdt.json`).

To re-check a fresh run against the paper, point the quick check at it:
`python reproduce/reproduce_public_results.py --results <csv_package>/public/03_output`.

## 3. Contents

```
code/                      experiment code (MIT licence)
  core_experiments.py      LotBench-811K, WM-811K lot task, the 14 single detectors (incl. SALE), shortcut tests, statistics
  extended_experiments.py  stacked ensemble (MCE), variants, ten seeds, time-ordered evaluation
  revision_experiments.py  SALE-lite, MCE without SALE, ablations, learning curve, same-coater analysis
  revision2_experiments.py lot-summary GBDT, tuning check of the forest and SALE (revision 2)
  rerun_wmlot_without_wafer_pretraining.py  WM-811K rerun without wafer-label pretraining
  export_csv.py            per-lot score and fold files (public / confidential split)
  run_all.py               one command for the whole pipeline (resumable)
  build_release.py         assembles a release and scans it for confidential strings
  production_loaders.py    documented STUB: the readers of the confidential production files are not released
  local_config_example.py  paths and settings (copy to local_config.py)
reproduce/                 one-minute reproduction of the public results and figures
  reproduce_public_results.py, expected_values.json   recompute and check every public number of the paper
  far_theory.py                                       Beta-binomial false-alarm theory (Sect. 5.5)
  make_figures.py, make_fig6.py, figstyle.py          redraw Figs. 4–7
figures/                   final figures of the paper, Fig1–Fig7 (TIFF, 600 dpi)
results/public/            released outputs of the paper run on public data (version v20)
  results_extended_v20/{lotbench,wmlot}/test_scores_seed1*.csv.gz   test score of every lot, detector, seed
  results_revision_v20/{lotbench,wmlot}/scores_seed1*.csv.gz        calibration, blending and test scores
  results_revision_v20/learning_curve/, ablation/                   learning curve and ablation runs
  results_core_v20/, paper_tables/                                  core tables and paper tables
  results_wmlot_nopretrain/wmlot/scores_seed1*.csv.gz             WM-811K lot task, SALE/SALE-lite/MCE/MCE-lite without wafer-label pretraining (used in the paper)
  results_revision2/gbdt_public/{lotbench,wmlot}/scores_seed1*.csv.gz  lot-summary GBDT (revision 2)
  results_revision2/tuning_summary.json                            tuning check, seeds 10–12 (revision 2)
  results_revision2/noise_summary.json                             SALE between training seeds, seeds 10–12 (revision 2)
docs/CONFIDENTIALITY.md    what is excluded and how the package was checked
docs/DETECTORS.md          detector codes used in the files and their names in the paper
tools/release_scan.py      confidentiality scan run before every upload
```

Score files: one row per lot; `label` (1 = abnormal), `scenario` (`N` normal, `F1`–`F6` fault mechanism on
LotBench-811K), `role` (`cal`, `blend`, `test`), `lot_index` (index of the WM-811K host lot) and one column
`score_<detector>` per detector (higher = more abnormal).

## 4. Confidentiality

The production data are confidential under agreements with the manufacturing partners. This package contains no
production data, no per-lot production results, no tool, chamber, lot, recipe or product identifiers, and no partner
names. `code/production_loaders.py` is a stub that only documents the interface; the real readers stay with the
authors. All files were scanned before release (`docs/CONFIDENTIALITY.md`). Access to the production data requires
the written approval of the partners.

## 5. Figures

`figures/` holds the final figures of the paper (Fig1–Fig7, TIFF, 600 dpi). Figs. 1–3 are schematic drawings of the
framework, the LotBench-811K construction and the evaluation protocol. Figs. 4, 5 and 7 are recomputed from the
released public files by `reproduce/make_figures.py`; Fig. 6 is drawn by `reproduce/make_fig6.py` from the aggregated
production values reported in the paper, because the production data are confidential.

## 6. Git and Zenodo

```bash
git init && git add . && git commit -m "Release 1.2.0"
git remote add origin <repository URL> && git push -u origin main
```

`.gitignore` keeps local and confidential files (`code/local_config.py`, `code/secret_list.txt`, run outputs) out of the
repository. `.zenodo.json` and `CITATION.cff` carry the metadata for a Zenodo record; with the GitHub–Zenodo
integration, every GitHub release is archived with its own DOI. Run `tools/release_scan.py` before every push.

## 7. Citation and licence

Code: MIT licence (`LICENSE`). Released result files: CC BY 4.0. Please cite the paper and this record
(`CITATION.cff`). WM-811K and MixedWM38 remain under the terms of their providers.

Contact: Yu-Ming Hsieh (corresponding author), Academy of Innovative Semiconductor and Sustainable Manufacturing,
National Cheng Kung University, Tainan, Taiwan.
