#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unweighted_average.py - exploratory analysis of Online Resource 1, Section S10: an unweighted average of the
standardized scores of the MCE base detectors (no abnormal lots needed to combine them), compared with the
lot-summary forest and with MCE on both public tasks (corrected resampled t test over the ten seeds, unadjusted).
Usage: python unweighted_average.py [--results ../results/public]
"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from reproduce_public_results import SEEDS, RATIO, corrected_t, conformal_tau

HERE = os.path.dirname(os.path.abspath(__file__))
BASES = ["B1", "B3-RF", "B4", "B8", "B9", "B10", "SALE", "SALE-S", "TI-RF", "GB-U"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default=os.path.join(HERE, "..", "results", "public")); a = ap.parse_args()
    src = {"lotbench": os.path.join(a.results, "results_revision_v20", "lotbench"),
           "wmlot": os.path.join(a.results, "results_wmlot_nopretrain", "wmlot")}      # WM-811K: rerun without wafer-label pretraining
    for task, d in src.items():
        A, F, M, rec, recf = [], [], [], [], []
        for s in SEEDS:
            x = pd.read_csv(os.path.join(d, f"scores_seed{s}.csv.gz"))
            cols = ["score_" + b for b in BASES if "score_" + b in x]
            cal = x[(x.role == "cal") & (x.label == 0)]
            x["u"] = np.mean([(x[c] - cal[c].mean()) / (cal[c].std() + 1e-12) for c in cols], 0)
            t = x[x.role == "test"]
            A.append(roc_auc_score(t.label, t.u)); F.append(roc_auc_score(t.label, t["score_B3-RF"])); M.append(roc_auc_score(t.label, t["score_MCE"]))
            for m, r in (("u", rec), ("score_B3-RF", recf)):
                tau = conformal_tau(x[(x.role == "cal") & (x.label == 0)][m]); r.append((t[t.label == 1][m] > tau).mean())
        A, F, M = map(np.array, (A, F, M))
        print(f"{task}: {len(cols)} base detectors; unweighted average AUROC {A.mean():.3f}, forest {F.mean():.3f}, MCE {M.mean():.3f}")
        print(f"   vs forest {np.mean(A - F):+.3f} (corrected p = {corrected_t(A - F, RATIO[task])[1]:.3f}, {int((A > F).sum())}/10 seeds higher); "
              f"vs MCE {np.mean(A - M):+.3f} (p = {corrected_t(A - M, RATIO[task])[1]:.3f}); recall at 5%: {np.mean(rec):.2f} (forest {np.mean(recf):.2f})")


if __name__ == "__main__":
    main()
