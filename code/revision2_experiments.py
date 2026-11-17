#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
revision2_experiments.py - analyses added in revision 2 of the manuscript.

Parts (run all, or choose with --parts):
  gbdt_public   Lot-summary GBDT (B3-GB): gradient boosting on EXACTLY the lot-summary features of the lot-summary
                forest, fixed default settings (sklearn HistGradientBoostingClassifier, balanced class weights).
                Ten seeds (10-19) of LotBench-811K and of the WM-811K lot task, same splits as the paper run
                (fit = 90% of the training lots, blending lots held out, calibration and test lots unchanged).
                The lot-summary forest is refitted on the same split as a check that the splits are identical.
  gbdt_prod     B3-GB on the three production lines with the folds of the main run (stratified 5-fold CV, 20 repetitions,
                same seeds); Plant B over all tools and on the same coaters. Only aggregated values are written.
  tune          Tuning sensitivity, forest vs. SALE with the same budget: seeds 10, 11, 12 of LotBench-811K (and of the
                WM-811K lot task with --tune-wm). 15% of the training lots are held out for validation (calibration
                and test lots are not touched); the same number of random-search trials for both (default 20); the
                best trial (validation AUROC) is refitted on all training lots and evaluated on the test lots, together
                with the default setting. Wall-clock time is recorded.
  noise         Run-to-run variation of SALE with its DEFAULT setting (noise baseline for the tuning check): seeds 10, 11,
                12 of LotBench-811K, SALE refitted on all training lots exactly as the 'default' refit of 'tune', once more
                with the SAME training seed (non-determinism of GPU operations) and with 3 DIFFERENT training seeds.
                Same splits, calibration and test lots. About 1 h per run, 12 runs.

Run (GPU recommended for 'tune'; every part is resumable, finished seeds/trials are skipped):
    python revision2_experiments.py --quick-test                     # a few minutes on synthetic data (pipeline check)
    python revision2_experiments.py --parts gbdt_public,gbdt_prod    # about 1-2 h
    python revision2_experiments.py --parts tune                     # about 2-3 days on one GPU workstation
    python revision2_experiments.py --parts tune --tune-wm           # adds the WM-811K lot task
    python revision2_experiments.py --parts noise                    # about 12 h on one GPU workstation
Output folder (default results_revision2/): per-lot score files of the PUBLIC tasks and summary JSON files.
Production results contain only aggregated values (AUROC, intervals, p-values); no lot, tool or date.
"""
import argparse, glob, hashlib, json, os, sys, time
from types import SimpleNamespace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import revision_experiments as RV                     # noqa: E402

SEEDS = list(range(10, 20))
TUNE_SEEDS = [10, 11, 12]
RF_DEFAULT = dict(n_estimators=300, min_samples_leaf=3, max_features="sqrt")
SALE_DEFAULT = dict(lr=3e-4, wd=1e-4, bdrop=0.3, width=(64, 128), epochs=30)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def jload(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def jsave(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


# ------------------------------------------------------------------ random search spaces (fixed seed per detector/seed)
def sample_rf(rng):
    return dict(n_estimators=int(round(np.exp(rng.uniform(np.log(100), np.log(1000))))),
                min_samples_leaf=int(rng.choice([1, 2, 3, 5, 10, 20])),
                max_features=[ "sqrt", "log2", 0.2, 0.33, 0.5][int(rng.integers(5))])


def sample_sale(rng):
    return dict(lr=float(np.exp(rng.uniform(np.log(1e-4), np.log(1e-3)))),
                wd=float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-3)))),
                bdrop=float(rng.uniform(0.0, 0.5)),
                width=[(32, 64), (64, 128), (128, 256)][int(rng.integers(3))],
                epochs=int(rng.choice([15, 30, 45, 60])))


def score_fold(J, d, fit, cal, te, method, seed, opts, cfg=None):
    """Score cal + te with one detector trained on 'fit' (no validation lots, as in the paper's benchmark)."""
    r = J.run_fold(d, fit, np.array([], int), cal, te, cfg or {}, [method], seed=seed, opts=opts)
    m = r.get(method, {})
    return (np.asarray(m["score"]) if "score" in m else None), m.get("sec", np.nan)


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


# ------------------------------------------------------------------ part 1: lot-summary GBDT on the public tasks
def part_gbdt_public(a, out):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    J, EK = RV.J, RV.EK
    meta, maps, MIX = RV.load_public(a)
    if meta is None:
        sys.exit("WM-811K (LSWMD.pkl) not found: set PUBLIC_DATA_DIR in local_config.py.")
    wmd = J.make_wmlot(meta, maps)
    seeds = [0, 1] if a.quick_test else SEEDS
    summ = jload(os.path.join(out, "gbdt_public_summary.json")) or {}
    for kind in ("lotbench", "wmlot"):
        for seed in seeds:
            dst = os.path.join(out, "gbdt_public", kind, f"scores_seed{seed}.csv.gz")
            if os.path.exists(dst):
                print(f"[gbdt {kind} seed {seed}] done; skipped", flush=True); continue
            t0 = time.time()
            d, tr, cal, te, _ = RV.prep_bench(kind, seed, meta, maps, MIX, a, wmd)
            yb = d.ybin
            fit, blend = train_test_split(tr, test_size=0.10, stratify=yb[tr], random_state=seed)
            r = J.run_fold(d, fit, np.array([], int), np.r_[blend, cal], te, {}, ["B3-RF", "B3-GB"], seed=seed, opts={"verbose": False})
            E = np.r_[blend, cal, te]; nb = len(blend)
            df = pd.DataFrame({"lot_index": E, "role": ["blend"] * nb + ["cal"] * len(cal) + ["test"] * len(te), "label": yb[E]})
            df["score_B3-GB"] = np.asarray(r["B3-GB"]["score"]); df["score_B3-RF_check"] = np.asarray(r["B3-RF"]["score"])
            os.makedirs(os.path.dirname(dst), exist_ok=True); df.to_csv(dst, index=False, compression="gzip")
            t = df.role.values == "test"
            summ[f"{kind}_{seed}"] = {"AUROC_B3-GB": auc(df.label[t], df["score_B3-GB"][t]), "AUROC_B3-RF": auc(df.label[t], df["score_B3-RF_check"][t]),
                                      "sec_B3-GB": r["B3-GB"].get("sec"), "sec_B3-RF": r["B3-RF"].get("sec"), "minutes": (time.time() - t0) / 60}
            jsave(summ, os.path.join(out, "gbdt_public_summary.json"))
            s = summ[f"{kind}_{seed}"]
            print(f"[gbdt {kind} seed {seed}] B3-GB {s['AUROC_B3-GB']:.3f}  forest {s['AUROC_B3-RF']:.3f}  ({s['minutes']:.1f} min)", flush=True)


# ------------------------------------------------------------------ part 2: lot-summary GBDT on the production lines
def part_gbdt_prod(a, out):
    J, JL = RV.J, RV.JL
    from types import SimpleNamespace as NS
    pa = NS(repeat=2 if a.quick_test else 20, repeat_deep=0, epochs=30, pre_epochs_prod=100)
    res = jload(os.path.join(out, "gbdt_production_summary.json")) or {}
    lines = ["demo"] if a.quick_test else ["plantb", "diesaw", "faba"]
    for nm in lines:
        if nm in res:
            print(f"[gbdt production {nm}] done; skipped", flush=True); continue
        d = JL.demo_data(0, 300) if a.quick_test else JL.load_prod(nm, a)
        if d is None:
            print(f"[gbdt production {nm}] data not found; skipped", flush=True); continue
        vp = os.path.join(a.v20_core, f"{nm}.json") if a.v20_core else None
        v20 = jload(vp) if vp else None
        print(f"[gbdt production {nm}] main-run JSON: {vp} -> {'found' if v20 else 'NOT FOUND'}", flush=True)
        subsets = [("all", d)]
        if "subset_same" in d.lots and d.lots["subset_same"].sum() < len(d.lots):
            subsets.append(("same", J.subset_data(d, d.lots["subset_same"].values)))
        res[nm] = {}
        for tag, dd in subsets:
            t0 = time.time()
            reps = J.cv_production(dd, pa, ["B3-RF", "B3-GB"], [], tag=f" {tag}")
            S = J.summarize_cv(dd, reps)
            e = {m: {k: S[m][k] for k in ("AUROC", "CI", "PR-AUC", "R@5%", "n_rep", "FAR@0.05", "FAR_CI@0.05") if k in S[m]} for m in S}
            # paired DeLong in repetition 1 against the reference of the main run (highest mean AUROC), with the Holm family
            # of the main run (raw p of every detector, taken from the main-run JSON) extended by B3-GB
            ref_v20 = (v20 or {}).get("delong_table" if tag == "all" else "delong_table_same")
            cvv = (v20 or {}).get("cv" if tag == "all" else "same", {})
            if ref_v20 and cvv:
                ref = ref_v20["strongest_baseline"]; s0 = cvv.get(ref, {}).get("_s0"); idx0 = cvv.get(ref, {}).get("_idx")
                rp0 = reps[0]
                if s0 is not None and idx0 is not None:
                    sref = dict(zip(idx0, s0)); ii = np.array(sorted(set(sref) & set(rp0["B3-GB"]["score"])))
                    y = dd.ybin[ii]
                    a1, a2, z, p = J.delong_paired(y, [rp0["B3-GB"]["score"][i] for i in ii], [sref[i] for i in ii])
                    lo_hi = J.auroc_ci_delong(y, [rp0["B3-GB"]["score"][i] for i in ii])
                    fam = {m: r["p"] for m, r in ref_v20["rows"].items() if "p" in r}
                    fam["B3-GB"] = p
                    names = list(fam); ph = J.holm([fam[m] for m in names])
                    e["delong_rep1"] = {"reference": ref, "AUROC_rep1_B3-GB": a1, "CI": [lo_hi[1], lo_hi[2]], "AUROC_rep1_ref": a2, "z": z, "p": p,
                                        "p_holm_B3-GB": float(ph[names.index("B3-GB")]),
                                        "p_holm_family": {m: float(h) for m, h in zip(names, ph)},
                                        "check_rf_rep1_vs_main_run": None}
                    # check that the folds are those of the main run: forest AUROC of repetition 1 here vs main run
                    rf0 = cvv.get("B3-RF", {})
                    if rf0.get("_s0") is not None:
                        srf = dict(zip(rf0["_idx"], rf0["_s0"])); jj = np.array(sorted(set(srf) & set(rp0["B3-RF"]["score"])))
                        e["delong_rep1"]["check_rf_rep1_vs_main_run"] = {"here": J.auroc(dd.ybin[jj], [rp0["B3-RF"]["score"][i] for i in jj]),
                                                                         "main_run": J.auroc(dd.ybin[jj], [srf[i] for i in jj])}
            # within-tool AUROC (Eq. 4) for both detectors
            try:
                wt = J.within_tool(dd, S)
                e["within_tool"] = {m: {k: v for k, v in wt[m].items() if not str(k).startswith("_")} for m in ("B3-RF", "B3-GB") if m in wt}
            except Exception as ex:                                            # noqa: BLE001
                e["within_tool"] = f"not computed: {type(ex).__name__}"
            e["main_run_forest_AUROC"] = cvv.get("B3-RF", {}).get("AUROC") if cvv else None
            # repetition-1 scores of both detectors (integer lot positions only, no identifiers), so that the paired DeLong
            # test against any detector of the main run can be recomputed from the main-run JSON
            rp0 = reps[0]; ii = sorted(rp0["B3-GB"]["score"])
            e["rep1"] = {"idx": [int(i) for i in ii], "B3-GB": [float(rp0["B3-GB"]["score"][i]) for i in ii],
                         "B3-RF": [float(rp0["B3-RF"]["score"][i]) for i in ii]}
            e["minutes"] = (time.time() - t0) / 60
            res[nm][tag] = e
            print(f"[gbdt production {nm} {tag}] B3-GB {S['B3-GB']['AUROC']:.3f}  forest {S['B3-RF']['AUROC']:.3f}"
                  f" (main run forest {e['main_run_forest_AUROC']})", flush=True)
        jsave(res, os.path.join(out, "gbdt_production_summary.json"))


# ------------------------------------------------------------------ part 3: tuning sensitivity, forest vs. SALE
def part_tune(a, out):
    from sklearn.model_selection import train_test_split
    from scipy import stats
    J = RV.J
    if not J.HAS_TORCH and not a.quick_test:
        sys.exit("STOP: PyTorch is not installed; SALE cannot be trained.")
    meta, maps, MIX = RV.load_public(a)
    wmd = J.make_wmlot(meta, maps) if a.tune_wm else None
    seeds = [0] if a.quick_test else TUNE_SEEDS
    trials = 2 if a.quick_test else a.trials
    path = os.path.join(out, "tuning_summary.json"); res = jload(path) or {}
    base_opts = RV.bench_opts(SimpleNamespace(quick_test=a.quick_test))
    for kind in (["lotbench", "wmlot"] if a.tune_wm else ["lotbench"]):
        for seed in seeds:
            key = f"{kind}_{seed}"; R = res.setdefault(key, {"forest": {"trials": []}, "sale": {"trials": []}})
            d, tr, cal, te, _ = RV.prep_bench(kind, seed, meta, maps, MIX, a, wmd)
            if kind == "wmlot":
                d.pre_maps = d.pre_y = None; d.mix_maps = d.mix_y = None          # no wafer-level labels (as in the paper's WM-811K results)
            yb = d.ybin
            fit_t, val = train_test_split(tr, test_size=0.15, stratify=yb[tr], random_state=seed)
            for det, sampler, method in (("forest", sample_rf, "B3-RF"), ("sale", sample_sale, "Ours")):
                rng = np.random.default_rng(10_000 * seed + (1 if det == "sale" else 0))
                cands = [sampler(rng) for _ in range(trials)]
                done = len(R[det]["trials"])
                for k in range(done, trials):
                    hp = cands[k]; t0 = time.time()
                    if det == "forest":
                        s, _ = score_fold(J, d, fit_t, np.array([], int), val, method, seed, dict(base_opts, rf_params=hp))
                    else:
                        o = dict(base_opts, lr=hp["lr"], wd=hp["wd"], epochs=2 if a.quick_test else hp["epochs"],
                                 pre_epochs=base_opts["pre_epochs"] if a.full_pretrain_in_search else min(base_opts["pre_epochs"], a.search_pre_epochs))
                        if kind == "wmlot":
                            o["pre_menc"] = False
                        s, _ = score_fold(J, d, fit_t, np.array([], int), val, method, seed, o, cfg={"bdrop": hp["bdrop"], "width": tuple(hp["width"])})
                    v = auc(yb[val], s) if s is not None else float("nan")
                    R[det]["trials"].append({"params": hp, "val_AUROC": v, "minutes": (time.time() - t0) / 60}); jsave(res, path)
                    print(f"[tune {key} {det} trial {k + 1}/{trials}] val AUROC {v:.3f}  {hp}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
                # final: default and best setting, refitted on ALL training lots, evaluated on the test lots
                for which in ("default", "tuned"):
                    if which in R[det]:
                        continue
                    if which == "default":
                        hp = dict(RF_DEFAULT) if det == "forest" else dict(SALE_DEFAULT)
                    else:
                        ok = [t for t in R[det]["trials"] if np.isfinite(t["val_AUROC"])]
                        if not ok:
                            print(f"[tune {key} {det}] no valid trial; tuned = default", flush=True)
                        hp = max(ok, key=lambda t: t["val_AUROC"])["params"] if ok else (dict(RF_DEFAULT) if det == "forest" else dict(SALE_DEFAULT))
                    t0 = time.time()
                    if det == "forest":
                        s, _ = score_fold(J, d, tr, cal, te, method, seed, dict(base_opts, rf_params=hp))
                    else:
                        o = dict(base_opts, lr=hp["lr"], wd=hp["wd"], epochs=2 if a.quick_test else hp["epochs"])
                        if kind == "wmlot":
                            o["pre_menc"] = False
                        s, _ = score_fold(J, d, tr, cal, te, method, seed, o, cfg={"bdrop": hp["bdrop"], "width": tuple(hp["width"])})
                    ta = auc(yb[te], s[len(cal):]) if s is not None else float("nan")
                    R[det][which] = {"params": hp, "test_AUROC": ta, "minutes": (time.time() - t0) / 60}; jsave(res, path)
                    print(f"[tune {key} {det} {which}] test AUROC {R[det][which]['test_AUROC']:.3f}", flush=True)
                R[det]["search_minutes"] = float(sum(t["minutes"] for t in R[det]["trials"])); jsave(res, path)
    # summary: default vs tuned for both, forest - SALE with a paired t test over the seeds
    summ = {}
    for kind in (["lotbench", "wmlot"] if a.tune_wm else ["lotbench"]):
        ks = [f"{kind}_{s}" for s in seeds if f"{kind}_{s}" in res and "tuned" in res[f"{kind}_{s}"]["sale"]]
        if not ks:
            continue
        g = lambda det, w: np.array([res[k][det][w]["test_AUROC"] for k in ks])
        row = {"seeds": ks}
        for det in ("forest", "sale"):
            row[det] = {"default": g(det, "default").tolist(), "tuned": g(det, "tuned").tolist(),
                        "mean_default": float(g(det, "default").mean()), "mean_tuned": float(g(det, "tuned").mean()),
                        "search_minutes": [res[k][det]["search_minutes"] for k in ks]}
        for w in ("default", "tuned"):
            dd = g("forest", w) - g("sale", w)
            row[f"forest_minus_sale_{w}"] = {"per_seed": dd.tolist(), "mean": float(dd.mean()),
                                             "p_paired_t": float(stats.ttest_rel(g("forest", w), g("sale", w)).pvalue) if len(ks) > 1 else None}
        summ[kind] = row
    res["_summary"] = summ; jsave(res, path)
    print(json.dumps(summ, indent=1, default=str))


# ------------------------------------------------------------------ part 4: run-to-run variation of SALE (default setting)
def part_noise(a, out):
    from scipy import stats
    J = RV.J
    if not J.HAS_TORCH and not a.quick_test:
        sys.exit("STOP: PyTorch is not installed; SALE cannot be trained.")
    meta, maps, MIX = RV.load_public(a)
    seeds = [0] if a.quick_test else TUNE_SEEDS
    path = os.path.join(out, "noise_summary.json"); res = jload(path) or {}
    base_opts = RV.bench_opts(SimpleNamespace(quick_test=a.quick_test))
    hp = dict(SALE_DEFAULT)
    o = dict(base_opts, lr=hp["lr"], wd=hp["wd"], epochs=2 if a.quick_test else hp["epochs"])
    for seed in seeds:
        key = f"lotbench_{seed}"; R = res.setdefault(key, {"runs": []})
        d, tr, cal, te, _ = RV.prep_bench("lotbench", seed, meta, maps, MIX, a, None)
        yb = d.ybin
        # training seeds: the seed of the 'default' refit of 'tune' (same seed again), then a.noise_reps different ones
        plan = [("same", seed)] + [("different", 1000 * seed + r) for r in range(1, a.noise_reps + 1)]
        for k, (kind, ts) in enumerate(plan):
            if k < len(R["runs"]):
                continue
            t0 = time.time()
            s, _ = score_fold(J, d, tr, cal, te, "Ours", ts, o, cfg={"bdrop": hp["bdrop"], "width": tuple(hp["width"])})
            ta = auc(yb[te], s[len(cal):]) if s is not None else float("nan")
            R["runs"].append({"train_seed": ts, "kind": kind, "test_AUROC": ta, "minutes": (time.time() - t0) / 60}); jsave(res, path)
            print(f"[noise {key} run {k + 1}/{len(plan)} ({kind} training seed {ts})] test AUROC {ta:.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # summary, compared with the 'default' and 'tuned' refits of the tuning check when tuning_summary.json is present
    T = jload(os.path.join(out, "tuning_summary.json")) or {}
    summ = {"seeds": {}}
    allr, ranges, devs = [], [], []
    for seed in seeds:
        key = f"lotbench_{seed}"
        runs = res.get(key, {}).get("runs", [])
        if not runs:
            continue
        v = [r["test_AUROC"] for r in runs]
        tdef = T.get(key, {}).get("sale", {}).get("default", {}).get("test_AUROC")
        ttun = T.get(key, {}).get("sale", {}).get("tuned", {}).get("test_AUROC")
        allv = v + ([tdef] if tdef is not None else [])                   # all default-setting runs of this seed
        row = {"runs": v, "same_seed_run": runs[0]["test_AUROC"], "tune_default": tdef, "tune_tuned": ttun,
               "default_runs_all": allv, "mean": float(np.mean(allv)), "sd": float(np.std(allv, ddof=1)) if len(allv) > 1 else None,
               "range": float(max(allv) - min(allv))}
        if tdef is not None:
            row["same_seed_abs_diff"] = abs(runs[0]["test_AUROC"] - tdef)   # identical setting and seed: GPU non-determinism only
        if ttun is not None:
            row["tuned_minus_default_mean"] = ttun - row["mean"]
            row["tuned_z"] = (ttun - row["mean"]) / row["sd"] if row["sd"] else None
            row["tuned_within_default_range"] = bool(min(allv) <= ttun <= max(allv))
        summ["seeds"][key] = row
        allr += allv; ranges.append(row["range"]); devs += [x - row["mean"] for x in allv]
    if summ["seeds"]:
        n = len(devs); k = len(summ["seeds"])
        summ["pooled_within_seed_sd"] = float(np.sqrt(np.sum(np.square(devs)) / (n - k))) if n > k else None
        summ["max_range"] = float(max(ranges))
        if all("tuned_minus_default_mean" in r for r in summ["seeds"].values()):
            dd = [r["tuned_minus_default_mean"] for r in summ["seeds"].values()]
            summ["tuned_minus_default_mean"] = float(np.mean(dd))
        F = [T.get(k, {}).get("forest", {}).get("default", {}).get("test_AUROC") for k in summ["seeds"]]
        if all(f is not None for f in F) and len(F) > 1:
            m = [r["mean"] for r in summ["seeds"].values()]
            summ["forest_minus_sale_default_mean"] = {"mean": float(np.mean(np.array(F) - np.array(m))),
                                                      "p_paired_t": float(stats.ttest_rel(F, m).pvalue)}
    res["_summary"] = summ; jsave(res, path)
    print(json.dumps(summ, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--parts", default="gbdt_public,gbdt_prod,tune")
    ap.add_argument("--out", default="results_revision2")
    ap.add_argument("--trials", type=int, default=20, help="random-search trials per detector and seed (same for forest and SALE)")
    ap.add_argument("--tune-wm", action="store_true", help="also tune on the WM-811K lot task (seeds 10-12)")
    ap.add_argument("--search-pre-epochs", type=int, default=100, help="Stage-1 pretraining epochs of SALE during the search (final refits always use 100)")
    ap.add_argument("--full-pretrain-in-search", action="store_true")
    ap.add_argument("--v20-core", default=None, help="folder with the main-run JSONs plantb.json, diesaw.json, faba.json (results_core_v20)")
    ap.add_argument("--noise-reps", type=int, default=3, help="different training seeds per seed in --parts noise")
    ap.add_argument("--quick-test", action="store_true")
    a = ap.parse_args()
    RV.init(); RV.PKG = None
    a.folder, a.label = RV.J.PARTNER_DATA_DIR, RV.J.LABEL_FINAL           # from local_config.py, as in the paper run
    if a.v20_core is None:
        a.v20_core = RV.J.OUT                                              # main-run JSONs (results_core_v20)
    out = a.out + ("_DEMO" if a.quick_test else ""); os.makedirs(out, exist_ok=True)
    cfg = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "parts": a.parts, "trials": a.trials, "tune_wm": a.tune_wm,
           "search_pre_epochs": a.search_pre_epochs, "rf_default": RF_DEFAULT, "sale_default": SALE_DEFAULT,
           "sha256": {f: sha(os.path.join(HERE, f)) for f in ("core_experiments.py", "extended_experiments.py", "revision_experiments.py",
                                                            os.path.basename(__file__))}}
    jsave(cfg, os.path.join(out, "CONFIGURATION_revision2.json"))
    parts = a.parts.split(",")
    if "gbdt_public" in parts:
        part_gbdt_public(a, out)
    if "gbdt_prod" in parts:
        part_gbdt_prod(a, out)
    if "tune" in parts:
        part_tune(a, out)
    if "noise" in parts:
        part_noise(a, out)
    print(f"done: {out}" + ("  (QUICK TEST: synthetic data, numbers meaningless)" if a.quick_test else ""))


if __name__ == "__main__":
    main()
