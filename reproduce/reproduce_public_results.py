#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduce_public_results.py - recompute every public-data number of the paper from the released per-lot scores.

No raw data, GPU or model training is needed: the script reads the test (and calibration) scores of every detector
and seed in results/public/ and recomputes
  * mean AUROC per detector over ten seeds, the difference to the lot-summary forest, the paired t test and the
    corrected resampled t test (Nadeau & Bengio, 2003) with Holm adjustment          -> Table 3, Online Resource S2
  * the confirmatory seeds 13-19 and the sensitivity analysis without the context-revealed faults -> S3, S5
  * recall and realized false-alarm rate at the split-conformal threshold for eps = 5% -> Table 3, Fig. 4c, 4d
  * the pooled hold/release decisions of Fig. 4c, the AUROC per fault scenario (S4)
  * the rank correlation of detector scores on abnormal lots (S12) and the learning curve (Fig. 5a, b)
It then compares the results with the values printed in the paper (expected_values.json) and reports PASS/FAIL.

Usage (about 1-3 minutes on a laptop):
    python reproduce_public_results.py                     # results/public next to this folder
    python reproduce_public_results.py --results <dir> --out <dir>
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(10, 20))
RATIO = {"lotbench": 3000 / 12600, "wmlot": 668 / 2804}      # n_test / n_train of the stacker (conservative for single detectors)
SECOND_RUN = ("SALE-lite", "MCE-lite", "MCE-without-SALE", "MCE-without-SALE-and-SALE-S")
# WM-811K lot task: the paper reports SALE and SALE-lite WITHOUT wafer-label pretraining of the map encoder
# (results/public/results_wmlot_nopretrain); MCE and MCE-lite contain them and were re-fitted accordingly.
NOPRETRAIN = ("SALE", "SALE-lite", "MCE", "MCE-lite")
USE_NOPRETRAIN = True
# revision 2: lot-summary GBDT (B3-GB, same features as the forest, default settings); joins the Holm family (19 / 18)
USE_GBDT = True


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); n = len(p); a = np.empty(n); r = 0.0
    for k, i in enumerate(o):
        r = max(r, min(1.0, (n - k) * p[i])); a[i] = r
    return a


def corrected_t(d, ratio):
    d = np.asarray(d, float); J = len(d); t = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1))
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def conformal_tau(cal_normal, eps=0.05):
    c = np.sort(np.asarray(cal_normal, float)); n = len(c); k = int(np.ceil((n + 1) * (1 - eps)))
    return np.inf if k > n else c[k - 1]


def load(results, kind):
    E = {s: pd.read_csv(os.path.join(results, "results_extended_v20", kind, f"test_scores_seed{s}.csv.gz")) for s in SEEDS}
    R = {s: pd.read_csv(os.path.join(results, "results_revision_v20", kind, f"scores_seed{s}.csv.gz")) for s in SEEDS}
    npd = os.path.join(results, "results_wmlot_nopretrain", "wmlot")
    if kind == "wmlot" and USE_NOPRETRAIN and os.path.isdir(npd):
        for s in SEEDS:
            N = pd.read_csv(os.path.join(npd, f"scores_seed{s}.csv.gz"))
            if not np.array_equal(N.lot_index.values, R[s].lot_index.values):
                sys.exit(f"rerun scores of seed {s} do not match the paper run")
            for m in NOPRETRAIN:
                R[s]["score_" + m] = N["score_" + m].values
                E[s] = E[s].drop(columns=["score_" + m], errors="ignore")    # take the test scores from the rerun
    gd = os.path.join(results, "results_revision2", "gbdt_public", kind)
    if USE_GBDT and os.path.isdir(gd):
        for s in SEEDS:
            G = pd.read_csv(os.path.join(gd, f"scores_seed{s}.csv.gz"))
            if not np.array_equal(G.lot_index.values, R[s].lot_index.values) or np.abs(G["score_B3-RF_check"].values - R[s]["score_B3-RF"].values).max() > 1e-5:
                sys.exit(f"GBDT scores of seed {s} do not match the splits of the paper run")
            R[s]["score_B3-GB"] = G["score_B3-GB"].values
            t = G[G.role == "test"].set_index("lot_index")["score_B3-GB"]
            E[s]["score_B3-GB"] = t.reindex(E[s].lot_index.values).values
    return E, R


def analyse(results, kind):
    E, R = load(results, kind)
    def frame(m, s):
        return E[s] if f"score_{m}" in E[s] else R[s][R[s].role == "test"]
    first = [c[6:] for c in E[10].columns if c.startswith("score_")]
    first += [m for m in ("SALE", "MCE") if m not in first and f"score_{m}" in R[10]]   # WM-811K rerun columns
    dets = first + list(SECOND_RUN)
    subsets = {"all": []}
    if kind == "lotbench":
        subsets.update({"noF4F5": ["F4", "F5"], "noF5": ["F5"]})
    out = {}
    for sub, excl in subsets.items():
        A = {}
        for m in dets:
            v = []
            for s in SEEDS:
                d = frame(m, s); keep = ~d.scenario.isin(excl) if excl else np.ones(len(d), bool)
                v.append(roc_auc_score(d.label[keep], d["score_" + m][keep]))
            A[m] = np.array(v)
        ref = A["B3-RF"]; family = [m for m in dets if m != "B3-RF" and m not in SECOND_RUN and m != "CTX"]
        rows = {}
        for m in dets:
            r = {"mean": float(A[m].mean()), "sd": float(A[m].std(ddof=1)), "per_seed": A[m].tolist()}
            if m != "B3-RF":
                d = A[m] - ref; h = stats.t.ppf(.975, 9) * np.sqrt((1 / 10 + RATIO[kind]) * d.var(ddof=1))
                t_nb, p_nb = corrected_t(d, RATIO[kind]); _, p_nb_conf = corrected_t(d[3:], RATIO[kind])
                r.update(diff=float(d.mean()), lo_nb=float(d.mean() - h), hi_nb=float(d.mean() + h), p=float(stats.ttest_rel(A[m], ref).pvalue),
                         t_nb=t_nb, p_nb=p_nb, diff_conf=float(d[3:].mean()), npos_conf=int((d[3:] > 0).sum()), p_nb_conf=p_nb_conf, p_conf=float(stats.ttest_rel(A[m][3:], ref[3:]).pvalue),
                         npos=int((d > 0).sum()))
            rows[m] = r
        for m, a, b in zip(family, holm([rows[m]["p"] for m in family]), holm([rows[m]["p_nb"] for m in family])):
            rows[m]["p_holm"] = float(a); rows[m]["p_nb_holm"] = float(b)
        out[sub] = rows
    a = np.array(out["all"]["SALE-lite"]["per_seed"]); b = np.array(out["all"]["MCE"]["per_seed"])
    out["salelite_vs_mce"] = {"diff": float((a - b).mean()), "p_nb": corrected_t(a - b, RATIO[kind])[1], "npos": int((a > b).sum())}
    # recall / FAR at the conformal threshold (revision scores carry the calibration lots)
    rec = {}; decisions = {}
    for m in ("B3-RF", "MCE", "SALE", "SALE-lite", "SALE-S", "B9", "B1", "B4") + (("B3-GB",) if "score_B3-GB" in R[10] else ()):
        rr, ff, per, tn = [], [], {}, np.zeros(4, int)
        for s in SEEDS:
            x = R[s]; tau = conformal_tau(x[(x.role == "cal") & (x.label == 0)]["score_" + m])
            t = x[x.role == "test"]; hold = t["score_" + m] > tau; ab = t.label == 1
            rr.append(hold[ab].mean()); ff.append(hold[~ab].mean())
            tn += np.array([(~hold & ~ab).sum(), (hold & ~ab).sum(), (~hold & ab).sum(), (hold & ab).sum()])
            if kind == "lotbench":
                for sc in ("F1", "F2", "F3", "F4", "F5", "F6"):
                    k = t.scenario == sc; per.setdefault(sc, []).append(hold[k].mean())
        rec[m] = {"recall": float(np.mean(rr)), "far": float(np.mean(ff)), "far_per_seed": [float(v) for v in ff],
                  "per": {k: float(np.mean(v)) for k, v in per.items()}}
        decisions[m] = {"release_normal": int(tn[0]), "hold_normal": int(tn[1]), "release_abnormal": int(tn[2]), "hold_abnormal": int(tn[3])}
    out["recall"] = rec; out["decisions"] = decisions
    # mean ROC curves (vertical averaging) for Fig. 4a, b
    grid = np.linspace(0, 1, 201); roc = {"fpr": grid.tolist()}
    for m in ("MCE", "B3-RF", "SALE", "B9", "B1", "SALE-lite"):
        tp = []
        for s in SEEDS:
            d = frame(m, s); f, t_, _ = roc_curve(d.label, d["score_" + m]); tp.append(np.interp(grid, f, t_))
        roc[m] = np.mean(tp, 0).tolist()
    out["roc"] = roc
    if kind == "lotbench":
        # AUROC per fault scenario (each scenario against all normal lots)
        ps = {}
        for m in [c[6:] for c in E[10].columns if c.startswith("score_")]:
            ps[m] = {}
            for sc in ("F1", "F2", "F3", "F4", "F5", "F6"):
                v = []
                for s in SEEDS:
                    d = E[s]; k = d.scenario.isin([sc, "N"])
                    v.append(roc_auc_score(d.label[k], d["score_" + m][k]))
                ps[m][sc] = float(np.mean(v))
        out["per_scenario_auroc"] = ps
        # Spearman correlation of scores on abnormal lots (mean over seeds)
        ms = ["B1", "B3-RF", "B4", "B9", "SALE", "B10"]; C = []
        for s in SEEDS:
            d = E[s][E[s].label == 1]; C.append(stats.spearmanr(d[["score_" + m for m in ms]].values).correlation)
        out["spearman_abnormal"] = {"detectors": ms, "matrix": np.round(np.mean(C, 0), 4).tolist()}
    return out


def tuning(results):
    """Revision 2 tuning check (Online Resource 1, Section S12): default vs tuned test AUROC of the forest and SALE, seeds 10-12."""
    from scipy import stats
    p = os.path.join(results, "results_revision2", "tuning_summary.json")
    if not os.path.exists(p):
        return None
    T = json.load(open(p)); ks = sorted(k for k in T if k.startswith("lotbench_") and "tuned" in T[k]["sale"])
    g = lambda det, w: np.array([T[k][det][w]["test_AUROC"] for k in ks])
    out = {"seeds": ks}
    for det in ("forest", "sale"):
        out[det] = {"mean_default": float(g(det, "default").mean()), "mean_tuned": float(g(det, "tuned").mean()),
                    "search_minutes_mean": float(np.mean([T[k][det]["search_minutes"] for k in ks]))}
    for w in ("default", "tuned"):
        d = g("forest", w) - g("sale", w)
        out[f"forest_minus_sale_{w}"] = {"mean": float(d.mean()), "p_paired_t": float(stats.ttest_rel(g("forest", w), g("sale", w)).pvalue)}
    n = os.path.join(results, "results_revision2", "noise_summary.json")
    if os.path.exists(n):
        N = json.load(open(n)); m = []; dev = []
        for k in ks:
            r = np.array([x["test_AUROC"] for x in N[k]["runs"]])          # four distinct runs (same-seed repeat = default refit)
            m.append(r.mean()); dev += list(r - r.mean())
        m = np.array(m); tu = g("sale", "tuned")
        out["noise"] = {"pooled_sd": float(np.sqrt(np.sum(np.square(dev)) / (len(dev) - len(ks)))),
                        "same_seed_max_abs_diff": float(max(abs(N[k]["runs"][0]["test_AUROC"] - T[k]["sale"]["default"]["test_AUROC"]) for k in ks)),
                        "tuned_minus_mean": {"mean": float((tu - m).mean()), "p_paired_t": float(stats.ttest_rel(tu, m).pvalue)},
                        "forest_minus_sale_mean": {"mean": float((g("forest", "default") - m).mean()), "p_paired_t": float(stats.ttest_rel(g("forest", "default"), m).pvalue)}}
    return out


def learning_curve(results):
    p = os.path.join(results, "results_revision_v20", "learning_curve", "learning_curve_runs.csv")
    lc = pd.read_csv(p); rows = []
    for fr, g in lc.groupby("train_fraction"):
        a = g[g.detector == "MCE-prod"].sort_values("seed"); b = g[g.detector == "B3-RF"].sort_values("seed")
        d = a.AUROC.values - b.AUROC.values
        rows.append({"fraction": float(fr), "abnormal_train": float(a.n_abnormal_train.mean()), "auroc_forest": float(b.AUROC.mean()),
                     "auroc_ensemble": float(a.AUROC.mean()), "diff": float(d.mean()), "p_paired_t": float(stats.ttest_rel(a.AUROC, b.AUROC).pvalue),
                     "recall_forest": float(b.recall5.mean()), "recall_ensemble": float(a.recall5.mean())})
    return rows


def lookup(res, path):
    x = res
    for k in path.split("."):
        x = x[int(k)] if isinstance(x, list) else x[k]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "..", "results", "public"))
    ap.add_argument("--out", default=os.path.join(HERE, "reproduced"))
    ap.add_argument("--with-wafer-pretraining", action="store_true",
                    help="WM-811K: use the scores of the paper run in which SALE's map encoder was pretrained on wafer labels")
    ap.add_argument("--without-gbdt", action="store_true", help="leave out the lot-summary GBDT (Holm families of 18 / 17 as in v1.0.0)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    global USE_NOPRETRAIN, USE_GBDT
    USE_NOPRETRAIN = not a.with_wafer_pretraining
    USE_GBDT = not (a.without_gbdt or a.with_wafer_pretraining)
    res = {k: analyse(a.results, k) for k in ("lotbench", "wmlot")}
    res["learning_curve"] = learning_curve(a.results)
    tu = tuning(a.results)
    if tu is not None and USE_GBDT:
        res["tuning"] = tu
    with open(os.path.join(a.out, "public_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    # Table 3 as CSV
    rows = []
    for m in ("B1", "B4", "B9", "B3-RF") + (("B3-GB",) if USE_GBDT and "B3-GB" in res["lotbench"]["all"] else ()) + ("SALE", "SALE-S", "SALE-lite", "MCE"):
        r = []
        for k in ("lotbench", "wmlot"):
            x = res[k]["all"][m]; rc = res[k]["recall"].get(m, {}).get("recall", np.nan)
            r += [x["mean"], x.get("diff", 0.0), x.get("p_nb_holm", x.get("p_nb", np.nan)), rc]
        r.insert(4, res["lotbench"]["noF4F5"][m]["mean"]); rows.append([m] + r)
    pd.DataFrame(rows, columns=["detector", "LB_AUROC", "LB_diff", "LB_p", "LB_recall", "LB_noctx_AUROC", "WM_AUROC", "WM_diff", "WM_p", "WM_recall"]) \
        .round(4).to_csv(os.path.join(a.out, "table3_reproduced.csv"), index=False)
    # compare with the paper
    exp = json.load(open(os.path.join(HERE, "expected_values.json" if USE_GBDT else ("expected_values_without_gbdt.json" if USE_NOPRETRAIN else "expected_values_with_wafer_pretraining.json"))))
    lines, fails = [], 0
    for e in exp["checks"]:
        try:
            v = lookup(res, e["path"]); v = v * e.get("scale", 1)
            ok = abs(round(v, e["digits"]) - e["paper"]) <= e.get("tol", 0) + 1e-9
        except Exception as ex:                           # noqa: BLE001
            v, ok = f"error: {ex}", False
        fails += (not ok)
        lines.append(f"{'PASS' if ok else 'FAIL'}  {e['label']}: paper {e['paper']}, reproduced {v if isinstance(v, str) else round(v, e['digits'] + 1)}")
    rep = "\n".join(lines + ["", f"{len(exp['checks']) - fails} of {len(exp['checks'])} checks passed"])
    open(os.path.join(a.out, "check_against_paper.txt"), "w").write(rep + "\n"); print(rep)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
