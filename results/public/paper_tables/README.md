# Paper tables written by the run (file names of the run)

| File | Table in the paper |
|---|---|
| table06a_lotbench_detection.csv | Table 3 (LotBench-811K columns) and Online Resource 1, Table S2 |
| table06b_wmlot_detection.csv | first run of the WM-811K lot task (with wafer-label pretraining of SALE/SALE-lite); see the note below |
| table07_lotbench_per_scenario.csv | Online Resource 1, Table S4 |
| table08_mce_design_vs_confirmatory.csv | Online Resource 1, Table S3 (LotBench-811K rows; WM-811K rows: first run) |
| sensitivity_lotbench_without_F5_F4.csv | Online Resource 1, Table S5 |

Column `delta_vs_B3RF` is the difference to the lot-summary forest (B3-RF); detector codes are explained in
docs/DETECTORS.md.

**WM-811K lot task.** The paper reports SALE, SALE-lite, MCE and MCE-lite on this task from the rerun without wafer-label
pretraining (`results/public/results_wmlot_nopretrain/`). The WM-811K columns of Table 3 and Table S2 and the WM-811K rows
of Table S3 are therefore recomputed by `reproduce/reproduce_public_results.py` (`reproduce/reproduced/`), not taken from
the files above, which keep the first run (compared in Online Resource 1, Table S13).
