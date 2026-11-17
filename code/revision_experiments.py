#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
revision_experiments.py - additional experiments for the revision of the JIM paper.

Parts (--parts):
  bench         LotBench-811K and WM-811K on the SAME ten seeds as the paper (10-19):
                (a) SALE-lite: SALE without the components that the ablations showed to be unnecessary (CFG["lite_remove"]),
                (b) MCE-lite: MCE with SALE-lite in place of SALE,
                (c) MCE-without-SALE and MCE-without-SALE-and-SALE-S: does the MCE gain come from the structure-aware
                    representation or only from stacking the baselines?
                The original MCE is recomputed and compared with the earlier run (reproduction check).
  ablation      SALE ablations on seeds 10, 11 and 12 (seed 10 can be reused from an earlier run) plus SALE-lite.
  curve         Learning curve on LotBench-811K: 2%-100% of the training lots (nested subsamples), B3-RF vs the
                production-style MCE (meta-learner on out-of-fold scores of an inner CV). Optional --curve-with-sale:
                the benchmark-style MCE with SALE as well.
  samecoater    MCE and all extended models on the Plant B "same coaters" subset (main Plant B analysis of the paper).

Put this file next to extended_experiments.py and core_experiments.py. Examples:
  python revision_experiments.py --quick-test                       # checks the pipeline on small/synthetic data
  python revision_experiments.py --parts samecoater                 # minutes
  python revision_experiments.py --parts curve                      # a few hours (without SALE)
  python revision_experiments.py --parts bench                      # slow: ~1.5 h per LotBench seed, ~15 min per WM seed
  python revision_experiments.py --parts ablation                   # ~3 x 9 h (complete per-lot scores)
  python revision_experiments.py --parts ablation --reuse-ablation results_extended/raw/ablation.json
Every part can be RESUMED: seeds/fractions already stored in raw/*.json are skipped.
Output: results_revision/REPORT_REVISION.md, raw/*.json, PREREGISTRATION_REVISION.json; CSV files (input, process,
output) go to csv_package/public (public data) and csv_package/confidential (production, anonymized), see export_csv.py.
Before uploading anything: python build_release.py
"""
import argparse, hashlib, importlib.util, json, os, sys, time, warnings
from collections import defaultdict
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))

# ======================= FROZEN CONFIGURATION (do not change after seeing results) =======================
CFG = {
    "version": "revision-1.1",
    # SALE components removed in SALE-lite (keywords matched against the names in core.ABLATIONS).
    # KEPT: chamber-relative normalization, route branch, context branch, polar map encoder, focal loss.
    "lite_remove": ["hierarchy", "slot Transformer", "slot and chamber embeddings", "peer contrast", "gated fusion",
                    "pretraining", "ℒord"],
    "bench_seeds": list(range(10, 20)),
    "confirmatory_seeds": list(range(13, 20)),
    "ablation_seeds": [10, 11, 12],
    "curve": {"fractions": [0.02, 0.05, 0.10, 0.20, 0.50, 1.00], "seeds": [10, 11, 12, 13, 14], "seeds_with_sale": [10, 11, 12]},
    "samecoater": {"repeats": 10, "min_normal_tool": 5},
    "expectations": {
        "E1": "MCE-without-SALE is lower than MCE on LotBench-811K (paired t test over 10 seeds, p < 0.05)",
        "E2": "MCE-lite is not worse than MCE by more than 0.005 AUROC (lower bound of the 95% CI of the difference > -0.005)",
        "E3": "removing chamber-relative normalization gives the largest ablation drop on every ablation seed",
        "E4": "the production-style MCE is not better than B3-RF at a production-sized training fraction (2%, ~20 abnormal lots)",
        "E5": "on Plant B same coaters, MCE does not differ significantly from B3-RF (Holm p >= 0.05)",
    },
}
# earlier ablation names -> names that follow the equation numbers of the paper
LEGACY_ABLATION_NAMES = {"Flat MIL instead of hierarchy (5)–(6)": "Flat MIL instead of hierarchy (7)–(8)",
                         "Cartesian CNN instead of polar encoder (4a)–(4b)": "Cartesian CNN instead of polar encoder (4)–(5)"}
# =======================================================================================================

JL = J = EK = PKG = None
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def init():
    """Load extended_experiments.py (which loads core_experiments.py) and export_csv.py from this folder."""
    global JL, J, EK
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import export_csv as EK_
    EK = EK_
    p = os.path.join(HERE, "extended_experiments.py")
    if not os.path.exists(p):
        sys.exit("extended_experiments.py must be in the same folder.")
    spec = importlib.util.spec_from_file_location("extended_experiments", p)
    JL = importlib.util.module_from_spec(spec); spec.loader.exec_module(JL); J = JL.J


# ============================================================ utilities
def load_json(p):
    if p and os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def dump(obj, p):
    J.jdump(obj, p)


def fmt(v, d=3):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def fp(p):
    return "—" if p is None or not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def lite_changes(ablations):
    """Merge the changes of several ablations into one SALE-lite configuration. Fails loudly on ambiguity or conflict."""
    ch, used = {}, []
    for kw in CFG["lite_remove"]:
        keys = [k for k in ablations if kw in k]
        if len(keys) != 1:
            sys.exit(f"SALE-lite: keyword '{kw}' matches {keys} in core.ABLATIONS; fix CFG['lite_remove'].")
        for kk, v in ablations[keys[0]].items():
            if kk in ch and ch[kk] != v:
                sys.exit(f"SALE-lite: conflict on '{kk}' ({ch[kk]} vs {v}) from '{keys[0]}'.")
            ch[kk] = v
        used.append(keys[0])
    return ch, used


def nested_sub(tr, y, f, seed):
    """Class-stratified subsample that is NESTED: a smaller fraction is always a subset of a larger one."""
    if f >= 1:
        return np.sort(np.asarray(tr))
    rng = np.random.default_rng(seed); out = []
    for c in (0, 1):
        idx = np.asarray(tr)[y[tr] == c]; idx = idx[rng.permutation(len(idx))]
        out.append(idx[:max(2 if c == 1 else 1, int(round(f * len(idx))))])
    return np.sort(np.concatenate(out))


def paired_summary(v, ref, ratio=None):
    """Paired difference over seeds: means, 95% t interval, paired t test, Wilcoxon (>= 6 seeds) and, if ratio
    (= n_test / n_train) is given, the corrected resampled t test of Nadeau & Bengio (2003) with its 95% interval."""
    v = np.asarray(v, float); ref = np.asarray(ref, float); d = v - ref; n = len(d)
    r = {"n": n, "mean": float(v.mean()), "ref_mean": float(ref.mean()), "diff": float(d.mean()),
         "min": float(d.min()), "max": float(d.max()), "n_pos": int((d > 0).sum())}
    if n >= 2 and d.std(ddof=1) > 0:
        h = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n); tt = stats.ttest_rel(v, ref)
        r.update(lo=float(d.mean() - h), hi=float(d.mean() + h), t=float(tt.statistic), p_t=float(tt.pvalue), df=n - 1)
        if ratio is not None and np.isfinite(ratio):
            t_nb, p_nb = JL.nb_corrected_t(d, ratio); se = np.sqrt((1.0 / n + ratio) * d.var(ddof=1)); hq = stats.t.ppf(.975, n - 1) * se
            r.update(t_nb=t_nb, p_nb=p_nb, lo_nb=float(d.mean() - hq), hi_nb=float(d.mean() + hq))
    if n >= 6:
        try:
            r["p_wilcoxon"] = float(stats.wilcoxon(v, ref).pvalue)
        except ValueError:
            pass
    return r


def bench_opts(a):
    return dict(epochs=2 if a.quick_test else 30, pre_epochs=1 if a.quick_test else 100, menc_epochs=1 if a.quick_test else 3,
                attr=False, verbose=False)


def load_public(a):
    wa = JL._wm_args(a); meta, maps = J.get_wm811k(wa); MIX = (None, None)
    if meta is not None and wa.mixedwm38 and not a.quick_test:
        MIX = J.load_mixedwm38(wa.mixedwm38)
    return meta, maps, MIX


def prep_bench(kind, seed, meta, maps, MIX, a, wmd):
    """Identical to extended_experiments.run_benchmark (splits, map pretraining), so a seed gives the same data."""
    if kind == "lotbench":
        d = J.make_lotbench(meta, maps, seed, n_lots=600 if a.quick_test else 20000)
        tr = np.where(d.lots["split"] == "train")[0]; cal = np.where(d.lots["split"] == "cal")[0]; te = np.where(d.lots["split"] == "test")[0]
        hs = set(d.lots.loc[tr, "host"]); L_ = meta[meta["ftype"].notna() & meta["lot"].isin(hs)].sample(frac=1, random_state=seed).head(30000)
        cls = sorted(L_["ftype"].unique()); d.pre_maps = np.stack([J.to_polar(maps[i]) for i in L_.index]).astype(np.float16) if len(L_) else None
        d.pre_y = L_["ftype"].map({c: k for k, c in enumerate(cls)}).values if len(L_) else None; scen = d.lots["scenario"].values
    else:
        d = wmd; L = len(d.ybin); perm = np.random.default_rng(1000 + seed).permutation(L)
        tr, cal, te = perm[:int(.7 * L)], perm[int(.7 * L):int(.85 * L)], perm[int(.85 * L):]; scen = None
        trw = np.isin(d.W["li"].values, tr); mi = np.random.default_rng(seed).permutation(d.W["map_i"].values[trw])[:30000]
        ft = d.ftype_w[mi]; cl = sorted(set(ft)); d.pre_maps = d.maps[mi]; d.pre_y = np.array([cl.index(f) for f in ft])
    d.mix_maps, d.mix_y = MIX
    return d, tr, cal, te, scen


def write_scores(path, lot_index, role, label, scores, desc, extra=None):
    """Write or update a per-lot score CSV (columns score_<detector>); existing columns are kept when resuming."""
    if PKG is None:
        return
    import pandas as pd
    df = pd.DataFrame({"lot_index": np.asarray(lot_index, int), "role": role, "label": np.asarray(label, int)})
    for k, v in (extra or {}).items():
        df[k] = v
    if os.path.exists(path):
        old = pd.read_csv(path)
        if len(old) == len(df) and (old["lot_index"].values == df["lot_index"].values).all():
            df = old
    for m, v in scores.items():
        df[f"score_{EK.det(m)}"] = np.asarray(v, float)
    PKG.write(df, path, desc)


def run_name(out_dir):
    return os.path.basename(os.path.normpath(out_dir))


def score_of(r, m="Ours"):
    return np.asarray(r[m]["score"]) if m in r and isinstance(r[m], dict) and "score" in r[m] else None


# ============================================================ 1. benchmarks: SALE-lite, MCE-lite, MCE without SALE
def bench_part(kind, a, out_dir, previous_dir):
    meta, maps, MIX = load_public(a)
    if meta is None:
        log(f"[bench {kind}] WM-811K not found; skipped"); return None
    from sklearn.model_selection import train_test_split
    seeds = [0, 1] if a.quick_test else CFG["bench_seeds"]; opts = bench_opts(a)
    wmd = J.make_wmlot(meta, maps) if kind == "wmlot" else None
    fpath = os.path.join(out_dir, "raw", f"bench_{kind}.json"); spath = os.path.join(out_dir, "raw", f"bench_{kind}_scores.json")
    st = load_json(fpath) or {"per_seed": {}, "vs": {}, "weights": {}, "reproduction": {}}
    st.setdefault("ratios", {})
    sc_all = load_json(spath) or {}
    old = EK.normalize_legacy(load_json(os.path.join(previous_dir, "raw", f"{kind}.json"))) if previous_dir else None
    lite, lite_used = lite_changes(J.ABLATIONS); st["lite_components_removed"] = lite_used
    log(f"[bench {kind}] SALE-lite removes: {lite_used}")
    for seed in seeds:
        if str(seed) in st["per_seed"]:
            log(f"[bench {kind}] seed {seed} already done; skipped"); continue
        t0 = time.time()
        d, tr, cal, te, scen = prep_bench(kind, seed, meta, maps, MIX, a, wmd); yb = d.ybin
        fit, blend = train_test_split(tr, test_size=0.10, stratify=yb[tr], random_state=seed)
        if PKG is not None:
            EK.export_bench_input(PKG, J, d, kind, seed, {"fit": fit, "blend": blend, "cal": cal, "test": te}); PKG.flush()
        out, E, w = JL.bench_fold(d, fit, blend, cal, te, seed, opts)
        s = score_of(J.run_fold(d, fit, np.array([], int), np.r_[blend, cal], te, lite, ["Ours"], seed=seed, opts=opts))
        if s is not None:
            out["SALE-lite"] = s
        nb = len(blend); W = {"MCE": w}
        if "MCE" in out:
            base = [b for b in JL.CFG["MCE"]["bases_benchmark"] if b in out]; var = {}
            if "SALE-lite" in out and "Ours" in base:
                var["MCE-lite"] = ["SALE-lite" if b == "Ours" else b for b in base]
            var["MCE-without-SALE"] = [b for b in base if b != "Ours"]
            var["MCE-without-SALE-and-SALE-S"] = [b for b in base if b not in ("Ours", "Ours-S")]
            for nm, bs in var.items():
                M = np.column_stack([out[b] for b in bs]); out[nm], ww = JL.meta_combine(M[:nb], yb[blend], M)
                W[nm] = dict(zip(bs, np.round(ww, 3).tolist()))
        ev = JL.bench_eval(d, out, E, nb, cal, te, scen)
        # direct comparisons: MCE variants vs MCE, SALE-lite vs SALE (DeLong, Holm within the seed)
        n_bc = nb + len(cal); y = yb[te]; vs = {}; ps = []
        pairs = [(k, "MCE") for k in out if k.startswith("MCE-")] + ([("SALE-lite", "Ours")] if "SALE-lite" in out else [])
        for nm, ref in pairs:
            if ref in out:
                a1, a2, z, p = J.delong_paired(y, np.asarray(out[nm])[n_bc:], np.asarray(out[ref])[n_bc:])
                vs[nm] = {"vs": ref, "AUROC": a1, "AUROC_ref": a2, "delta": a1 - a2, "z": z, "p": p}; ps.append((nm, p))
        for (nm, _), ph in zip(ps, J.holm([p for _, p in ps])):
            vs[nm]["p_holm"] = float(ph)
        if old and str(seed) in old.get("per_seed", {}) and "MCE" in ev:
            o = old["per_seed"][str(seed)]
            st["reproduction"][str(seed)] = {m: {"previous": o[m]["AUROC"], "new": ev[m]["AUROC"], "difference": ev[m]["AUROC"] - o[m]["AUROC"]}
                                             for m in ("B3-RF", "Ours", "MCE") if m in o and m in ev}
            log(f"[bench {kind}] seed {seed} reproduction: " + ", ".join(f"{m} {v['previous']:.3f}->{v['new']:.3f}" for m, v in st["reproduction"][str(seed)].items()))
        st["per_seed"][str(seed)] = ev; st["vs"][str(seed)] = vs; st["weights"][str(seed)] = W
        st["ratios"][str(seed)] = len(te) / len(fit)
        sc_all[str(seed)] = {"E": np.asarray(E).tolist(), "n_blend": nb, "n_cal": len(cal), "y_E": yb[E].tolist(),
                             "scen_E": scen[E].tolist() if scen is not None else None,
                             "scores_E": {m: np.asarray(v).tolist() for m, v in out.items()}}
        dump(st, fpath); dump(sc_all, spath)
        log(f"[bench {kind}] seed {seed} ({(time.time() - t0) / 60:.0f} min): " +
            ", ".join(f"{m} {ev[m]['AUROC']:.3f}" for m in ["B3-RF", "Ours", "SALE-lite", "MCE", "MCE-lite", "MCE-without-SALE",
                                                           "MCE-without-SALE-and-SALE-S"] if m in ev))
    st["summary"] = bench_summary_variants(st)
    dump(st, fpath)
    return st


def bench_summary_variants(st):
    P = st["per_seed"]; V = st["vs"]; seeds = sorted(P, key=int); R = {"all": {}, "confirmatory": {}}
    ratio = float(np.mean(list(st.get("ratios", {}).values()))) if st.get("ratios") else None
    R["test_train_ratio"] = ratio
    for label, ss in (("all", seeds), ("confirmatory", [s for s in seeds if int(s) in CFG["confirmatory_seeds"]])):
        if len(ss) < 2:
            continue
        for nm in sorted({k for s in ss for k in V.get(s, {})}):
            sel = [s for s in ss if nm in V.get(s, {})]
            if len(sel) < 2:
                continue
            r = paired_summary([V[s][nm]["AUROC"] for s in sel], [V[s][nm]["AUROC_ref"] for s in sel], ratio)
            r["vs"] = V[sel[0]][nm]["vs"]
            r["seeds_better"] = sum(1 for s in sel if V[s][nm].get("p_holm", 1) < .05 and V[s][nm]["z"] > 0)
            r["seeds_worse"] = sum(1 for s in sel if V[s][nm].get("p_holm", 1) < .05 and V[s][nm]["z"] < 0)
            R[label][nm] = r
        for nm in ("SALE-lite", "MCE-lite", "MCE-without-SALE", "MCE-without-SALE-and-SALE-S", "MCE"):
            sel = [s for s in ss if nm in P[s] and "B3-RF" in P[s]]
            if len(sel) >= 2:
                R[label][nm + "|vs B3-RF"] = paired_summary([P[s][nm]["AUROC"] for s in sel], [P[s]["B3-RF"]["AUROC"] for s in sel], ratio)
        if label == "all":          # SALE-lite against MCE (best single detector vs ensemble)
            sel = [s for s in ss if "SALE-lite" in P[s] and "MCE" in P[s]]
            if len(sel) >= 2:
                R[label]["SALE-lite|vs MCE"] = paired_summary([P[s]["SALE-lite"]["AUROC"] for s in sel], [P[s]["MCE"]["AUROC"] for s in sel], ratio)
    for nm in ("MCE", "MCE-lite"):
        ws = [W[nm] for W in st["weights"].values() if W.get(nm)]
        if ws:
            R["weights_" + nm] = {k: [round(float(np.mean([w[k] for w in ws if k in w])), 3), round(float(np.std([w[k] for w in ws if k in w])), 3)] for k in ws[0]}
    sc = {}
    for nm in ("B3-RF", "Ours", "SALE-lite", "MCE", "MCE-lite", "MCE-without-SALE"):
        v = [P[s][nm]["per_scen"] for s in seeds if nm in P[s] and "per_scen" in P[s][nm]]
        if v:
            sc[nm] = {k: float(np.nanmean([x[k] for x in v])) for k in v[0]}
    if sc:
        R["per_scenario"] = sc
    return R


# ============================================================ 2. ablations over several seeds (+ SALE-lite)
def ablation_part(a, out_dir):
    if not J.HAS_TORCH:
        log("[ablation] PyTorch not available; skipped"); return None
    from sklearn.model_selection import train_test_split
    meta, maps, _ = load_public(a)
    if meta is None:
        log("[ablation] WM-811K not found; skipped"); return None
    seeds = [0, 1] if a.quick_test else CFG["ablation_seeds"]
    fpath = os.path.join(out_dir, "raw", "ablation_multi.json"); st = load_json(fpath) or {"per_seed": {}}
    if a.reuse_ablation and not a.quick_test:
        old = load_json(a.reuse_ablation) or {}
        for s, rows in old.items():
            if int(s) in seeds and s not in st["per_seed"]:
                rows = {LEGACY_ABLATION_NAMES.get(k, k): v for k, v in rows.items()}
                st["per_seed"][s] = {"rows": rows, "source": os.path.basename(a.reuse_ablation)}
                log(f"[ablation] seed {s}: {len(rows)} variants reused from {a.reuse_ablation} (no per-lot scores)")
    lite, lite_used = lite_changes(J.ABLATIONS); st["lite_components_removed"] = lite_used
    wanted = list(J.ABLATIONS) + ["SALE-lite"]
    for seed in seeds:
        S = st["per_seed"].setdefault(str(seed), {"rows": {}})
        todo = [v for v in wanted if v not in S["rows"]]
        if not todo:
            continue
        d = J.make_lotbench(meta, maps, seed, n_lots=600 if a.quick_test else 20000)
        tr = np.where(d.lots["split"] == "train")[0]; cal = np.where(d.lots["split"] == "cal")[0]; te = np.where(d.lots["split"] == "test")[0]
        fit, va = train_test_split(tr, test_size=0.10, stratify=d.ybin[tr], random_state=seed)
        opts = dict(epochs=2 if a.quick_test else 30, pre_epochs=1 if a.quick_test else 100, attr=False, verbose=False)   # as in run_ablation
        full = np.asarray(J.run_fold(d, fit, va, cal, te, {}, ["Ours"], seed=seed, opts=opts)["Ours"]["score"])[len(cal):]
        y = d.ybin[te]; S["AUROC_full"] = float(J.auroc(y, full))
        spath = PKG.pub("03_output", run_name(out_dir), "ablation", f"test_scores_seed{seed}.csv.gz") if PKG else None
        sdesc = f"ablation seed {seed}: scores of the full model (SALE) and of every variant for every test lot"
        write_scores(spath, te, "test", y, {"SALE_full": full}, sdesc)
        old_full = [r.get("AUROC_full") for r in S["rows"].values() if r.get("AUROC_full") is not None]
        if old_full:
            log(f"[ablation] seed {seed}: full model {S['AUROC_full']:.3f} (earlier run {old_full[0]:.3f})")
        for nm in todo:
            ch = lite if nm == "SALE-lite" else J.ABLATIONS[nm]
            s = score_of(J.run_fold(d, fit, va, cal, te, ch, ["Ours"], seed=seed, opts=opts))
            if s is None:
                continue
            s = s[len(cal):]; a1, a2, z, p = J.delong_paired(y, s, full)
            write_scores(spath, te, "test", y, {nm: s}, sdesc)
            S["rows"][nm] = {"AUROC": a1, "AUROC_full": a2, "delta": a1 - a2, "z": z, "p": p}
            log(f"[ablation] seed {seed}: {nm}: Δ {a1 - a2:+.3f} (p {p:.3g})"); dump(st, fpath)
            if PKG:
                PKG.flush()
    # Holm per seed over the 12 original ablations (as in the paper); SALE-lite is reported separately
    for s, S in st["per_seed"].items():
        names = [n for n in S["rows"] if n != "SALE-lite"]
        for nm, ph in zip(names, J.holm([S["rows"][n]["p"] for n in names])):
            S["rows"][nm]["p_holm"] = float(ph)
    agg = {}
    for nm in wanted:
        sel = [s for s in sorted(st["per_seed"], key=int) if nm in st["per_seed"][s]["rows"]]
        if not sel:
            continue
        rows = [st["per_seed"][s]["rows"][nm] for s in sel]; dl = np.array([r["delta"] for r in rows])
        r = {"seeds": [int(s) for s in sel], "delta_mean": float(dl.mean()), "delta_min": float(dl.min()), "delta_max": float(dl.max()),
             "significant_drops": sum(1 for r_ in rows if r_.get("p_holm", r_["p"]) < .05 and r_["delta"] < 0),
             "significant_gains": sum(1 for r_ in rows if r_.get("p_holm", r_["p"]) < .05 and r_["delta"] > 0)}
        if len(dl) >= 2 and dl.std(ddof=1) > 0:
            tt = stats.ttest_1samp(dl, 0.0); r.update(t=float(tt.statistic), p_t=float(tt.pvalue), df=len(dl) - 1)
        agg[nm] = r
    st["summary"] = agg
    st["largest_drop_per_seed"] = {s: min(((n, r["delta"]) for n, r in S["rows"].items() if n != "SALE-lite"), key=lambda x: x[1])[0]
                                   for s, S in st["per_seed"].items() if S["rows"]}
    dump(st, fpath)
    return st


# ============================================================ 3. learning curve
def eval_ev(out, yb, cal, te):
    ncal = len(cal); y = yb[te]; R = {}; sref = np.asarray(out["B3-RF"])[ncal:] if "B3-RF" in out else None
    for m, s in out.items():
        s = np.asarray(s); st = s[ncal:]; tau = J.conformal_tau(s[:ncal][yb[cal] == 0], 0.05)
        e = {"AUROC": J.auroc(y, st), "PR-AUC": J.prauc(y, st), "recall5": float((st[y == 1] > tau).mean()), "FAR5": float((st[y == 0] > tau).mean())}
        if sref is not None and m != "B3-RF":
            a1, a2, z, p = J.delong_paired(y, st, sref); e.update(z=z, p=p)
        R[m] = e
    return R


def curve_part(a, out_dir):
    from sklearn.model_selection import train_test_split
    meta, maps, MIX = load_public(a)
    if meta is None:
        log("[curve] WM-811K not found; skipped"); return None
    K = CFG["curve"]; seeds = [0] if a.quick_test else K["seeds"]; fr = [0.2, 1.0] if a.quick_test else K["fractions"]
    seeds_sale = ([0] if a.quick_test else K["seeds_with_sale"]) if a.curve_with_sale else []
    opts = bench_opts(a); bases = JL.CFG["MCE"]["bases_production"]; cls = [m for m in bases if m in J.CLASSIC]
    fpath = os.path.join(out_dir, "raw", "learning_curve.json"); st = load_json(fpath) or {"runs": {}}
    for seed in seeds:
        need = [f for f in fr if f"{seed}|{f}" not in st["runs"] or (seed in seeds_sale and "sale" not in st["runs"][f"{seed}|{f}"])]
        if not need:
            continue
        d, tr, cal, te, scen = prep_bench("lotbench", seed, meta, maps, MIX, a, None); yb = d.ybin
        tools = d.lots["tool"].astype(str).values; ev = np.r_[cal, te]
        for f in need:
            key = f"{seed}|{f}"; t0 = time.time()
            sub = nested_sub(tr, yb, f, 777 + seed)
            if PKG is not None:
                import pandas as pd
                PKG.write(pd.DataFrame({"lot_index": sub}), PKG.pub("02_process", "learning_curve", f"subsample_seed{seed}_frac{f:g}.csv"),
                          f"learning curve seed {seed}: training lots at fraction {f:g} (nested; original split in 02_process/lotbench/splits_seed{seed}.csv)")
            kdir = PKG.pub("03_output", run_name(out_dir), "learning_curve") if PKG else None
            R = st["runs"].get(key) or {"seed": seed, "fraction": f, "n_train": int(len(sub)), "n_abnormal": int(yb[sub].sum())}
            if "prod" not in R:
                r = J.run_fold(d, sub, np.array([], int), cal, te, {}, cls, seed=seed, opts=opts)
                out = {m: np.asarray(v["score"]) for m, v in r.items() if isinstance(v, dict) and "score" in v}
                X, *_ = JL.features(d, sub)
                m = JL.fit_ti_rf(X[sub], yb[sub], tools[sub], seed)
                if m is not None:
                    out["TI-RF"] = JL.logit(m.predict_proba(X[ev])[:, list(m.classes_).index(1)])
                out["GB-U"] = JL.logit(JL.fit_gbu(X[sub], yb[sub], seed).predict_proba(X[ev])[:, 1])
                if all(b in out for b in bases) and yb[sub].sum() >= JL.CFG["MCE"]["inner_folds"]:
                    oof = JL.mce_inner(d, sub, seed); ok = ~np.isnan(oof).any(1)
                    if ok.sum() > 10 and yb[sub][ok].sum() >= 2:
                        out["MCE-prod"], w = JL.meta_combine(oof[ok], yb[sub][ok], np.column_stack([out[b] for b in bases]))
                        R["weights_prod"] = dict(zip(bases, np.round(w, 3).tolist()))
                R["prod"] = {k: v for k, v in eval_ev(out, yb, cal, te).items() if k in ("B3-RF", "MCE-prod", "Ours-S", "B4", "B1")}
                if kdir:
                    write_scores(os.path.join(kdir, f"scores_production_style_seed{seed}_frac{f:g}.csv.gz"), ev, ["cal"] * len(cal) + ["test"] * len(te),
                                 yb[ev], out, f"learning curve seed {seed}, fraction {f:g}: scores of the base detectors and MCE-prod for cal and test lots")
            if seed in seeds_sale and "sale" not in R:
                fit, blend = train_test_split(sub, test_size=0.10, stratify=yb[sub], random_state=seed)
                if yb[blend].sum() >= JL.CFG["MCE"]["min_abnormal_blend"]:
                    outB, E, w = JL.bench_fold(d, fit, blend, cal, te, seed, opts)
                    evB = JL.bench_eval(d, outB, E, len(blend), cal, te, None)
                    if kdir:
                        write_scores(os.path.join(kdir, f"scores_with_SALE_seed{seed}_frac{f:g}.csv.gz"), E,
                                     ["blend"] * len(blend) + ["cal"] * len(cal) + ["test"] * len(te), yb[E], outB,
                                     f"learning curve seed {seed}, fraction {f:g}: scores of all detectors (incl. SALE, MCE) for blend, cal and test lots")
                    R["sale"] = {k: {kk: v[kk] for kk in ("AUROC", "PR-AUC", "recall5", "FAR5", "z", "p") if kk in v} for k, v in evB.items() if k in ("B3-RF", "Ours", "Ours-S", "MCE")}
                else:
                    R["sale"] = {"note": f"the blending set has only {int(yb[blend].sum())} abnormal lots; benchmark-style MCE not run"}
            st["runs"][key] = R; dump(st, fpath)
            p = R["prod"]
            log(f"[curve] seed {seed}, fraction {f}: {R['n_train']} lots ({R['n_abnormal']} abnormal), B3-RF {p['B3-RF']['AUROC']:.3f}" +
                (f", MCE-prod {p['MCE-prod']['AUROC']:.3f}" if "MCE-prod" in p else "") +
                (f", MCE(SALE) {R['sale']['MCE']['AUROC']:.3f}" if isinstance(R.get("sale"), dict) and "MCE" in R["sale"] else "") +
                f" ({(time.time() - t0) / 60:.1f} min)")
    S = {}
    for f in sorted({r["fraction"] for r in st["runs"].values()}):
        rs = [r for r in st["runs"].values() if r["fraction"] == f]
        e = {"n_train": float(np.mean([r["n_train"] for r in rs])), "n_abnormal": float(np.mean([r["n_abnormal"] for r in rs])), "n_seeds": len(rs)}
        for arm, a_, b_ in (("prod", "MCE-prod", "B3-RF"), ("sale", "MCE", "B3-RF"), ("sale_SALE", "Ours", "B3-RF")):
            src = "sale" if arm.startswith("sale") else "prod"
            ok = [r for r in rs if isinstance(r.get(src), dict) and a_ in r[src] and b_ in r[src]]
            if ok:
                v = [r[src][a_]["AUROC"] for r in ok]; ref = [r[src][b_]["AUROC"] for r in ok]
                ps = paired_summary(v, ref) if len(ok) >= 2 else {"n": 1, "mean": v[0], "ref_mean": ref[0], "diff": v[0] - ref[0]}
                ps["recall5"] = float(np.mean([r[src][a_]["recall5"] for r in ok])); ps["recall5_ref"] = float(np.mean([r[src][b_]["recall5"] for r in ok]))
                ps["seeds_delong_sig_up"] = sum(1 for r in ok if r[src][a_].get("p", 1) < .05 and r[src][a_].get("z", 0) > 0)
                ps["seeds_delong_sig_down"] = sum(1 for r in ok if r[src][a_].get("p", 1) < .05 and r[src][a_].get("z", 0) < 0)
                e[arm] = ps
        S[str(f)] = e
    st["summary"] = S; dump(st, fpath)
    return st


# ============================================================ 4. Plant B same coaters
def same_coater_idx(d):
    L = d.lots; yb = d.ybin
    if "subset_same" in L and not L["subset_same"].astype(bool).all():
        idx = np.where(L["subset_same"].astype(bool).values)[0]; rule = "column subset_same"
    else:
        tools = L["tool"].astype(str).values; keep = []
        for t in np.unique(tools):
            k = tools == t
            if (yb[k] == 0).sum() >= CFG["samecoater"]["min_normal_tool"] and yb[k].sum() >= 1:
                keep.append(t)
        idx = np.where(np.isin(tools, keep))[0]; rule = f"tools with >= {CFG['samecoater']['min_normal_tool']} normal and >= 1 abnormal lot ({len(keep)} tools)"
    return idx, rule


def samecoater_part(a, out_dir):
    from sklearn.model_selection import StratifiedKFold
    d = JL.demo_data() if a.quick_test else JL.load_prod("plantb", a)
    if d is None:
        log("[samecoater] Plant B data not found; skipped"); return None
    yb = d.ybin; tools = d.lots["tool"].astype(str).values
    sub, rule = same_coater_idx(d); nb = int(yb[sub].sum())
    log(f"[samecoater] {rule}: {len(sub)} lots, {nb} abnormal (paper: 222 lots, 7 abnormal)")
    if not a.quick_test and (len(sub) != 222 or nb != 7):
        log("[samecoater] WARNING: the subset differs from the paper; check the definition before using these results.")
    R_ = 1 if a.quick_test else CFG["samecoater"]["repeats"]; reps = []; folds = []
    stem = "demo" if a.quick_test else "plantb"
    if PKG is not None and not os.path.exists(PKG.conf("01_input", EK.PROD_NAME[stem], "lots.csv.gz")):
        EK.export_dataset(PKG, d, EK.PROD_NAME[stem], False, stem=stem)
    for r in range(R_):
        skf = StratifiedKFold(min(JL.CFG["cv"]["folds"], nb), shuffle=True, random_state=5000 + r)
        sc = defaultdict(dict); far = defaultdict(lambda: defaultdict(list)); infos = []
        for k, (trp, tep) in enumerate(skf.split(np.zeros(len(sub)), yb[sub])):
            tr, te = sub[trp], sub[tep]          # global indices; lots outside the subset are never used
            fit, va, cal = J.split_train(tr, yb, seed=r * 100 + k, cal_frac=JL.CFG["cv"]["frac_calibration_normal"])
            folds += [(r + 1, k + 1, int(i), role) for role, ii in (("fit", fit), ("validation", va), ("cal", cal), ("test", te)) for i in ii]
            out, info, ev = JL.run_fold(d, fit, va, cal, te, r * 100 + k)
            infos.append(info); res = JL.eval_scores(d, te, cal, out, len(cal))
            for m, v in res.items():
                for i, x in zip(te, v["score_te"]):
                    sc[m][int(i)] = x
                for e in JL.CFG["eps"]:
                    far[m][e].append(v[f"far{e}"])
        log(f"[samecoater] repetition {r + 1}/{R_} done")
        reps.append({"scores": {m: dd for m, dd in sc.items()}, "far": {m: {str(e): float(np.nanmean(v)) for e, v in dd.items()} for m, dd in far.items()}, "info": infos})
    if PKG is not None:
        EK.export_folds(PKG, folds, "samecoater", "revision", "CV folds of the same-coaters subset (fit, validation, cal, test)")
    S = JL.summarize(d, reps)            # within-tool AUROC of Eq. 14 (extended_experiments.within_tool_auc)
    mw = [i["MCE_weights"] for rp in reps for i in rp["info"] if "MCE_weights" in i]
    n_fold = sum(len(rp["info"]) for rp in reps)
    R = {"subset_rule": rule, "n_lots": int(len(sub)), "n_abnormal": nb, "cv": S, "mce_folds_run": len(mw), "folds_total": n_fold,
         "mce_w": {k: round(float(np.mean([w[k] for w in mw])), 3) for k in mw[0]} if mw else None,
         "per_lot_rep1": {"idx": sorted(reps[0]["scores"]["B3-RF"]), "y": yb.tolist(), "tool": tools.tolist(), "scores": reps[0]["scores"]}}
    dump(R, os.path.join(out_dir, "raw", "same_coaters.json"))
    for m in ["B3-RF", "B6", "MCE"] + [x for x in JL.NEW if x != "MCE"]:
        if m in S:
            log(f"[samecoater] {m}: AUROC {S[m]['AUROC']:.3f}" + (f", DeLong Holm p {fp(S[m].get('delong_p_holm'))}" if m != "B3-RF" else ""))
    return R


# ============================================================ report
def report(out_dir, res):
    md = [f"# REPORT OF THE REVISION EXPERIMENTS ({time.strftime('%Y-%m-%d %H:%M')}, {CFG['version']})", "",
          "Expectations fixed before the runs:", ""] + [f"- **{k}**: {v}" for k, v in CFG["expectations"].items()] + [""]
    ev = {}
    for kind in ("lotbench", "wmlot"):
        B = res.get("bench_" + kind)
        if not B or "summary" not in B:
            continue
        R = B["summary"]; n = len(B["per_seed"])
        md += [f"## {kind}: SALE-lite, MCE-lite and MCE without SALE ({n} seeds)", "", f"SALE-lite removes: {', '.join(B.get('lite_components_removed', []))}.", ""]
        for label in ("all", "confirmatory"):
            T = R.get(label, {})
            if not isinstance(T, dict) or not T:
                continue
            md += [f"### {'All seeds' if label == 'all' else 'Confirmatory seeds 13–19'}", "",
                   "| Model | Reference | AUROC model | AUROC reference | Δ [95% CI] | t (df), p | corrected t, p (Nadeau-Bengio) | Wilcoxon p | seeds > / < (DeLong-Holm) |", "|---|---|---|---|---|---|---|---|---|"]
            for nm, r in T.items():
                m, refl = (nm.split("|vs ")[0], nm.split("|vs ")[1]) if "|vs " in nm else (nm, r.get("vs", ""))
                md.append(f"| {EK.det(m)} | {EK.det(refl)} | {fmt(r['mean'])} | {fmt(r['ref_mean'])} | {fmt(r['diff'], 4)} [{fmt(r.get('lo'), 4)}, {fmt(r.get('hi'), 4)}] | "
                          f"{fmt(r.get('t'), 2)} ({r.get('df', '—')}), {fp(r.get('p_t'))} | {fmt(r.get('t_nb'), 2)}, {fp(r.get('p_nb'))} | {fp(r.get('p_wilcoxon'))} | "
                          f"{r.get('seeds_better', '—')} / {r.get('seeds_worse', '—')} |")
            md.append("")
        if "per_scenario" in R:
            ks = list(next(iter(R["per_scenario"].values())).keys())
            md += ["AUROC per scenario (mean over seeds):", "", "| Model | " + " | ".join(ks) + " |", "|---|" + "---|" * len(ks)]
            md += [f"| {EK.det(m)} | " + " | ".join(fmt(v[k]) for k in ks) + " |" for m, v in R["per_scenario"].items()] + [""]
        for nm in ("weights_MCE", "weights_MCE-lite"):
            if nm in R:
                md += [f"{nm} (mean, sd): " + ", ".join(f"{EK.det(k)} {v[0]:+.2f} ({v[1]:.2f})" for k, v in R[nm].items()), ""]
        if B.get("reproduction"):
            dmax = max(abs(v["difference"]) for s in B["reproduction"].values() for v in s.values())
            md += [f"Reproduction of the earlier run: largest AUROC difference {dmax:.4f} (B3-RF, SALE, MCE).", ""]
        if kind == "lotbench":
            r1 = R.get("all", {}).get("MCE-without-SALE"); r2 = R.get("all", {}).get("MCE-lite")
            if r1:
                ev["E1"] = ("supported" if (r1["diff"] < 0 and r1.get("p_t", 1) < .05) else "not supported") + f" (Δ {r1['diff']:+.4f}, p {fp(r1.get('p_t'))})"
            if r2 and "lo" in r2:
                ev["E2"] = ("supported" if r2["lo"] > -0.005 else "not supported") + f" (Δ {r2['diff']:+.4f} [{r2['lo']:+.4f}, {r2['hi']:+.4f}])"
    A = res.get("ablation")
    if A and A.get("summary"):
        seeds = sorted(A["per_seed"], key=int)
        md += [f"## SALE ablations on seeds {', '.join(seeds)} (DeLong vs full model; Holm over the 12 ablations within each seed)", "",
               "| Variant | " + " | ".join(f"Δ seed {s}" for s in seeds) + " | mean Δ [min, max] | t (df), p | significant drops / gains |", "|---|" + "---|" * (len(seeds) + 3)]
        for nm, r in A["summary"].items():
            cells = [fmt(A["per_seed"][s]["rows"][nm]["delta"]) if nm in A["per_seed"][s]["rows"] else "—" for s in seeds]
            md.append(f"| {nm} | " + " | ".join(cells) + f" | {fmt(r['delta_mean'])} [{fmt(r['delta_min'])}, {fmt(r['delta_max'])}] | "
                      f"{fmt(r.get('t'), 2)} ({r.get('df', '—')}), {fp(r.get('p_t'))} | {r['significant_drops']} / {r['significant_gains']} |")
        md += ["", "Largest drop per seed: " + ", ".join(f"seed {s}: {v}" for s, v in A["largest_drop_per_seed"].items()), ""]
        ev["E3"] = "supported" if all("normalization" in v for v in A["largest_drop_per_seed"].values()) else "not supported"
    K = res.get("curve")
    if K and K.get("summary"):
        md += ["## Learning curve on LotBench-811K (fraction of training lots, nested; full calibration and test splits)", "",
               "| Fraction | Training lots | Abnormal lots | B3-RF | MCE-prod | Δ [95% CI] | p (t) | R@5% B3-RF → MCE-prod | seeds DeLong up/down | SALE | MCE (with SALE) | Δ vs B3-RF |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for f, e in K["summary"].items():
            p = e.get("prod", {}); s = e.get("sale", {}); ss = e.get("sale_SALE", {})
            md.append(f"| {float(f):.0%} | {e['n_train']:.0f} | {e['n_abnormal']:.0f} | {fmt(p.get('ref_mean'))} | {fmt(p.get('mean'))} | "
                      f"{fmt(p.get('diff'), 4)} [{fmt(p.get('lo'), 4)}, {fmt(p.get('hi'), 4)}] | {fp(p.get('p_t'))} | "
                      f"{fmt(p.get('recall5_ref'), 2)} → {fmt(p.get('recall5'), 2)} | {p.get('seeds_delong_sig_up', '—')}/{p.get('seeds_delong_sig_down', '—')} | "
                      f"{fmt(ss.get('mean'))} | {fmt(s.get('mean'))} | {fmt(s.get('diff'), 4)} |")
        md.append("")
        f0 = min(K["summary"], key=float); p0 = K["summary"][f0].get("prod", {})
        if p0:
            ev["E4"] = ("supported" if not (p0.get("diff", 0) > 0 and p0.get("p_t", 1) < .05) else "not supported") + f" (fraction {float(f0):.0%}: Δ {p0['diff']:+.4f}, p {fp(p0.get('p_t'))})"
    C = res.get("samecoater")
    if C:
        S = C["cv"]
        md += [f"## Plant B same coaters ({C['n_lots']} lots, {C['n_abnormal']} abnormal; {C['subset_rule']})", "",
               f"MCE was run in {C['mce_folds_run']} of {C['folds_total']} folds (requires >= 5 abnormal lots in the training fold).", "",
               "| Method | AUROC [range over reps] | AUROC pooled p-values | DeLong vs B3-RF z (Holm p), rep 1 | reps sig. better / worse / tested | within-tool AUROC (Eq. 14) | PR-AUC | FAR ε=5% |", "|---|---|---|---|---|---|---|---|"]
        order = ["B3-RF"] + [m for m in ["B1", "B5", "B6", "Ours-S"] if m in S] + [m for m in JL.NEW if m in S]
        for m in order:
            e = S[m]
            md.append(f"| {EK.det(m)} | {fmt(e['AUROC'])} [{fmt(e['AUROC_pct'][0], 2)}, {fmt(e['AUROC_pct'][1], 2)}] | {fmt(e.get('AUROC_pval'))} | "
                      f"{'reference' if m == 'B3-RF' else fmt(e.get('delong_z'), 2) + ' (' + fp(e.get('delong_p_holm')) + ')'} | "
                      f"{'' if m == 'B3-RF' else str(e.get('reps_sig_better')) + ' / ' + str(e.get('reps_sig_worse')) + ' / ' + str(e.get('reps_tested'))} | "
                      f"{fmt(e.get('withinAUROC'))} | {fmt(e['PR-AUC'])} | {fmt(100 * e['FAR5'], 1)}% |")
        if C.get("mce_w"):
            md += ["", f"MCE meta-learner weights (mean over folds): {C['mce_w']}"]
        if "MCE" in S:
            ev["E5"] = (("supported" if S["MCE"].get("delong_p_holm", 1) >= .05 else "not supported") +
                        f" (repetition 1: AUROC {S['MCE']['AUROC']:.3f} vs B3-RF {S['B3-RF']['AUROC']:.3f}, Holm p {fp(S['MCE'].get('delong_p_holm'))}; "
                        f"significantly worse in {S['MCE'].get('reps_sig_worse')} of {S['MCE'].get('reps_tested')} repetitions)")
        md.append("")
    if ev:
        md += ["## Outcome of the expectations", ""] + [f"- **{k}**: {v}" for k, v in sorted(ev.items())] + [""]
    with open(os.path.join(out_dir, "REPORT_REVISION.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    log(f"Report: {os.path.join(out_dir, 'REPORT_REVISION.md')}")


# ============================================================ main
def main():
    global PKG
    init()
    ap = argparse.ArgumentParser(description="Revision experiments of the JIM paper")
    ap.add_argument("--folder", default=J.PARTNER_DATA_DIR); ap.add_argument("--label", default=J.LABEL_FINAL)
    ap.add_argument("--out", default=J._setting("OUT_REVISION", "results_revision"))
    ap.add_argument("--previous", default=J._setting("OUT_EXTENDED", "results_extended"), help="earlier extended run, for the reproduction check")
    ap.add_argument("--parts", default="samecoater,curve,bench,ablation", help="samecoater,curve,bench,ablation")
    ap.add_argument("--reuse-ablation", default=None, help="earlier ablation.json (seed 10) to reuse (its per-lot scores are not available)")
    ap.add_argument("--curve-with-sale", action="store_true", help="learning curve also with SALE and the benchmark-style MCE (slow)")
    ap.add_argument("--package", default=J._setting("CSV_PACKAGE", "csv_package"), help="CSV package folder (public + confidential)")
    ap.add_argument("--no-csv", action="store_true", help="do not write CSV files (not recommended)")
    ap.add_argument("--quick-test", action="store_true")
    a = ap.parse_args()
    PKG = None if a.no_csv else EK.Package(a.package + ("_DEMO" if a.quick_test else ""))
    out = a.out + ("_DEMO" if a.quick_test else ""); os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    if a.quick_test:
        JL.CFG["TI-NN"]["epochs"] = 30; JL.CFG["TI-NN"]["ensemble"] = 1; JL.CFG["PU-RF"]["n_bags"] = 5; JL.CFG["bootstrap"] = 200
        JL.CFG["MCE"]["inner_folds"] = 3; JL.CFG["SSL-RF"]["max_iter"] = 50
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    pre = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "sha256_script": sha(os.path.abspath(__file__)),
           "sha256_extended": sha(os.path.join(HERE, "extended_experiments.py")), "sha256_core": sha(os.path.join(HERE, "core_experiments.py")),
           "configuration": CFG, "configuration_extended": JL.CFG, "parts": a.parts, "quick_test": a.quick_test}
    pp = os.path.join(out, "PREREGISTRATION_REVISION.json"); old = load_json(pp)
    if old and not a.quick_test and any(old.get(k) != pre[k] for k in ("sha256_script", "sha256_extended", "sha256_core")):
        sys.exit(f"STOP: {pp} was written by different scripts (SHA-256 differs). Resuming would mix code versions: "
                 f"set a NEW output folder (OUT_REVISION in local_config.py) and run again.")
    if not old or a.quick_test:
        dump(pre, pp)
    if not J.HAS_TORCH and not a.quick_test and any(p in a.parts for p in ("bench", "ablation", "samecoater")):
        sys.exit("STOP: PyTorch is not installed; SALE / SALE-lite / B6-B9 would be missing. Install PyTorch (see CARA_RUN.md).")
    log(f"revision_experiments.py {CFG['version']}; torch: {J.HAS_TORCH}; parts: {a.parts}")
    parts = [p.strip() for p in a.parts.split(",")]; res = {}
    previous = None if a.quick_test else a.previous
    if "samecoater" in parts:
        res["samecoater"] = samecoater_part(a, out)
    if "curve" in parts:
        res["curve"] = curve_part(a, out)
    if "bench" in parts:
        for kind in ("lotbench", "wmlot"):
            res["bench_" + kind] = bench_part(kind, a, out, previous)
    if "ablation" in parts:
        res["ablation"] = ablation_part(a, out)
    # include results of other parts that are already stored, so the report is complete
    for k, f in (("samecoater", "same_coaters.json"), ("curve", "learning_curve.json"), ("bench_lotbench", "bench_lotbench.json"),
                 ("bench_wmlot", "bench_wmlot.json"), ("ablation", "ablation_multi.json")):
        if not res.get(k):
            res[k] = load_json(os.path.join(out, "raw", f))
    report(out, res)
    with open(os.path.join(out, "log_revision.txt"), "a", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    if PKG is not None:
        EK.export_run(PKG, out, paper_source=False); PKG.flush()
        log(f"CSV: {PKG.root} (public = may be uploaded after build_release.py; confidential = never)")


if __name__ == "__main__":
    main()
