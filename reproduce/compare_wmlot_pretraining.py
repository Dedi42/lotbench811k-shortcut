#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_wmlot_pretraining.py - WM-811K lot task with and without wafer-label pretraining of the SALE map encoder.

Reads the released scores of the paper run and the scores written by code/rerun_wmlot_without_wafer_pretraining.py,
and recomputes the WM-811K columns of Table 3 (mean AUROC over ten seeds, difference to the lot-summary forest,
corrected resampled t test with Holm over the 17 detectors, recall at the conformal threshold for eps = 5%) plus the
comparisons of Section 5.1/5.2 (SALE-lite vs MCE, MCE without SALE, MCE-lite) for both versions.
Usage: python compare_wmlot_pretraining.py --nopretrain <out folder of the rerun> [--results ../results/public]
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from reproduce_public_results import SEEDS, RATIO, holm, corrected_t, conformal_tau

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = ("SALE", "SALE-lite", "MCE", "MCE-lite")


def table(E, R):
    dets = [c[6:] for c in E[10].columns if c.startswith("score_") and c[6:] != "CTX"] + ["SALE-lite", "MCE-lite", "MCE-without-SALE"]
    def test(m, s):
        if m in NEW or f"score_{m}" not in E[s]:
            d = R[s]; return d[d.role == "test"]
        return E[s]
    A = {m: np.array([roc_auc_score(test(m, s).label, test(m, s)["score_" + m]) for s in SEEDS]) for m in dets}
    ref = A["B3-RF"]; fam = [m for m in dets if m not in ("B3-RF", "SALE-lite", "MCE-lite", "MCE-without-SALE")]
    rows = {}
    for m in dets:
        r = {"AUROC": A[m].mean()}
        if m != "B3-RF":
            r["diff"] = (A[m] - ref).mean(); r["p_nb"] = corrected_t(A[m] - ref, RATIO["wmlot"])[1]
        rows[m] = r
    for m, p in zip(fam, holm([rows[m]["p_nb"] for m in fam])):
        rows[m]["p_holm"] = p
    for m in ("B3-RF", "SALE", "SALE-lite", "MCE", "MCE-lite"):
        rr = []
        for s in SEEDS:
            x = R[s]; tau = conformal_tau(x[(x.role == "cal") & (x.label == 0)]["score_" + m]); t = x[x.role == "test"]
            rr.append((t[t.label == 1]["score_" + m] > tau).mean())
        rows[m]["recall"] = float(np.mean(rr))
    for a_, b_ in (("SALE-lite", "MCE"), ("MCE-without-SALE", "MCE"), ("MCE-lite", "MCE"), ("SALE-lite", "SALE")):
        d = A[a_] - A[b_]
        rows[f"{a_} vs {b_}"] = {"diff": d.mean(), "p_nb": corrected_t(d, RATIO["wmlot"])[1], "seeds_above": int((d > 0).sum())}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "..", "results", "public"))
    ap.add_argument("--nopretrain", required=True)
    a = ap.parse_args()
    E = {s: pd.read_csv(os.path.join(a.results, "results_extended_v20", "wmlot", f"test_scores_seed{s}.csv.gz")) for s in SEEDS}
    R0 = {s: pd.read_csv(os.path.join(a.results, "results_revision_v20", "wmlot", f"scores_seed{s}.csv.gz")) for s in SEEDS}
    R1 = {s: pd.read_csv(os.path.join(a.nopretrain, "wmlot", f"scores_seed{s}.csv.gz")) for s in SEEDS}
    T0, T1 = table(E, R0), table(E, R1)
    fmt = lambda v: "" if v is None else f"{v:.3f}"
    fp = lambda v: "" if v is None else ("< 0.001" if v < 0.001 else f"{v:.3f}")
    fd = lambda v: "" if v is None else f"{v:+.3f}"
    lines = ["WM-811K lot task: paper run (map encoder pretrained on wafer labels) vs rerun without wafer-label pretraining", "",
             f"{'detector':22s} {'AUROC':>13s} {'diff':>15s} {'Holm p':>15s} {'recall':>13s}"]
    for m in ("B3-RF", "B9", "SALE", "SALE-lite", "MCE", "MCE-lite", "MCE-without-SALE"):
        r0, r1 = T0[m], T1[m]
        lines.append(f"{m:22s} {fmt(r0['AUROC'])}->{fmt(r1['AUROC'])}  {fd(r0.get('diff'))}->{fd(r1.get('diff'))}  "
                     f"{fp(r0.get('p_holm'))}->{fp(r1.get('p_holm'))}  {fmt(r0.get('recall'))}->{fmt(r1.get('recall'))}")
    lines.append("")
    for k in ("SALE-lite vs MCE", "MCE-without-SALE vs MCE", "MCE-lite vs MCE", "SALE-lite vs SALE"):
        r0, r1 = T0[k], T1[k]
        lines.append(f"{k:24s} diff {fd(r0['diff'])}->{fd(r1['diff'])}, corrected p {fp(r0['p_nb'])}->{fp(r1['p_nb'])}, "
                     f"seeds above {r0['seeds_above']}->{r1['seeds_above']}")
    rep = "\n".join(lines); print(rep)
    os.makedirs(os.path.join(HERE, "reproduced"), exist_ok=True)
    open(os.path.join(HERE, "reproduced", "wmlot_pretraining_comparison.txt"), "w").write(rep + "\n")
    json.dump({"paper_run": T0, "without_wafer_pretraining": T1}, open(os.path.join(HERE, "reproduced", "wmlot_pretraining_comparison.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
