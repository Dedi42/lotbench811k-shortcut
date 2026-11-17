#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
extended_experiments.py - extended experiments: tool-invariant variants, the mechanism-complementary ensemble (MCE),
ten-seed benchmarks, SALE ablations and time-ordered evaluation.

Background (from the paper): supervised detectors (B3-RF, B4, SALE) absorb tool identity on Plant B, whereas
detectors referenced to normal lots hold up better. This script tests models that address this directly, with a
protocol frozen before the runs.

New models (all scores depend on training lots only -> the conformal threshold stays valid):
  MCE     Mechanism-complementary ensemble: B1 (chamber), B4 (wafer level, strong on F2), B8/B9 (sets, strong on F3),
          SALE (route, strong on F4), B10 (commonality), B3-RF, SALE-S, TI-RF and GB-U combined by logistic regression
          on standardized scores. Benchmarks: meta-learner fitted on a blending set (10% of training lots) that never
          trains a base detector; production: meta-learner fitted on out-of-fold scores of an inner CV (classical
          base detectors).
  TI-RF   Random forest with within-tool weights: every tool with normal AND abnormal lots gets class-balanced weights;
          tools with one class carry no within-tool information and are not used for training.
          Features: lot summary (B3) + per-tool standardized structure features (SALE-S).
  TI-NN   Small network with (a) a pairwise ranking loss that compares lots of the SAME tool only and
          (b) a tool-identity adversary through gradient reversal (Ganin & Lempitsky, 2015). Needs PyTorch.
  HYB     Stacking of a supervised RF score, the SALE-S one-class score and an Isolation Forest, combined by logistic
          regression fitted on out-of-fold predictions of a nested inner CV on training lots only.
  GB-U    Histogram gradient boosting on the union of B3 and SALE-S features.
  PU-RF   Positive-unlabelled bagging (Mordelet & Vert, 2014): "normal" lots are treated as unlabelled.
  SSL-RF  Denoising autoencoder (no labels) on structure features of all training lots (+ optional unlabelled lots);
          latent code + reconstruction error + original features -> random forest.
References (from core_experiments.py): B1, B3-RF, B5, SALE-S, and B6 if PyTorch is available.

Protocol (v2.2; frozen in PREREGISTRATION_<parts>.json BEFORE any data are read; the file is never overwritten):
  - PRIMARY metric: pooled AUROC (as in the paper). Co-primary check for production lines: within-tool AUROC (Eq. 14)
    -- a model counts as "better" on a production line only if it is better on BOTH and neither is significantly worse
  - production: stratified 5-fold CV repeated R times; paired DeLong vs B3-RF (Holm) on EVERY repetition (the count of
    significant repetitions is reported, not only repetition 1); stratified bootstrap for the within-tool difference;
    pooled conformal p-values as a scale-free sensitivity check; leave-one-tool-out; time split; monthly chaining
  - benchmarks (10 seeds): paired t test over seeds AND the corrected resampled t test (Nadeau & Bengio, 2003), which
    accounts for the overlap of training sets between seeds; Holm over detectors; the corrected test decides
  - LotBench sensitivity: AUROC without F5 and without F4+F5 (scenarios that context/route identify by design) and a
    context-only model (CTX)
  - per-lot scores (index, label, tool, month, score) are STORED so that every test can be repeated without rerunning

Additions v2.0: LotBench-811K and WM-811K over 10 seeds (paired t test df = 9, Wilcoxon, per-seed DeLong with Holm),
SALE ablations with DeLong tests against the full model, and monthly forward chaining for all production lines.

Put this file next to core_experiments.py. Run:
  python extended_experiments.py --quick-test                       # synthetic data with tool confounding (10-15 min)
  python extended_experiments.py --parts plantb,diesaw,faba         # production (CV, LOTO, time split, monthly)
  python extended_experiments.py --parts lotbench,wmlot             # 10-seed benchmarks (slow: SALE ~45 min per seed)
  python extended_experiments.py --parts ablation                   # SALE ablations on seed 10 (12 variants, ~9 h)
  python extended_experiments.py --parts plantb --extra-unlabeled unlabelled_lots.npy   (optional, for SSL-RF)
Output: results_extended/REPORT.md, results_extended/raw/*.json, results_extended/PREREGISTRATION_<parts>.json
"""
import argparse, hashlib, importlib.util, json, os, sys, time, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = "results_extended"            # overridden by OUT_EXTENDED in local_config.py

# ======================= FROZEN CONFIGURATION (do not change after seeing results) =======================
CFG = {
    "version": "extended-2.2",
    "primary_metric": "pooled AUROC (paper, Table 6 and Table 9)",
    "secondary_metrics": ["within-tool AUROC (Eq. 14, weighted by abnormal lots per tool)", "PR-AUC", "recall and FAR at eps = 5%",
                          "AUROC of pooled conformal p-values (scale-free)"],
    "claim_rule": ("benchmarks: better than B3-RF only if the Holm-adjusted corrected resampled t test (Nadeau & Bengio) "
                   "gives p < 0.05; production: only if the Holm-adjusted DeLong p < 0.05 on most CV repetitions AND the "
                   "within-tool AUROC is not significantly lower; otherwise 'not different'"),
    "time_protocols_skip": ["plantb"],       # Plant B lot dates are export dates, not processing dates (paper, Data section)
    "cv": {"folds": 5, "repeats": 10, "frac_calibration_normal": 0.15},
    "time_split": {"frac_train": 0.70},
    "eps": [0.05, 0.01],
    "TI-RF": {"n_estimators": 500, "min_samples_leaf": 3, "max_features": "sqrt"},
    "TI-NN": {"hidden": 64, "dropout": 0.2, "epochs": 300, "lr": 1e-3, "weight_decay": 1e-3, "lambda_rank": 1.0,
              "lambda_adv": 0.5, "ensemble": 5, "large_batch": 512},
    "HYB": {"inner_folds": 5, "meta_C": 1.0},
    "GB-U": {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    "PU-RF": {"n_bags": 50, "unlabeled_ratio": 5, "n_estimators": 100},
    "SSL-RF": {"hidden": [64, 16, 64], "noise": 0.1, "max_iter": 500},
    "bootstrap": 2000,
    "MCE": {"bases_benchmark": ["B1", "B3-RF", "B4", "B8", "B9", "B10", "Ours", "Ours-S", "TI-RF", "GB-U"],
            "bases_production": ["B1", "B3-RF", "B4", "B5", "Ours-S", "TI-RF", "GB-U"], "C": 1.0, "inner_folds": 5,
            "min_abnormal_blend": 5},
    "benchmark": {"seeds": list(range(10, 20)), "ablation_seeds": [10]},
    "monthly": {"min_abnormal_train": 3, "min_normal_train": 20},
}
NEW = ["MCE", "TI-RF", "TI-NN", "HYB", "GB-U", "PU-RF", "SSL-RF"]
# =======================================================================================================


def load_J():
    p = os.path.join(HERE, "core_experiments.py")
    if not os.path.exists(p):
        sys.exit("core_experiments.py must be in the same folder.")
    spec = importlib.util.spec_from_file_location("core_experiments", p)
    m = importlib.util.module_from_spec(spec); sys.argv_backup = sys.argv; spec.loader.exec_module(m)
    return m


J = load_J()
from sklearn.ensemble import RandomForestClassifier, IsolationForest, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


# ============================================================ metrics and tests
def within_tool_auc(y, s, tools, min_normal=5, min_abnormal=2):
    """Within-tool AUROC of Eq. 14 (tools with >= 5 normal and >= 2 abnormal lots, weighted by abnormal lots).
    min_normal=1, min_abnormal=1 gives the earlier sensitivity version (every tool with both classes)."""
    return J.within_tool_auroc(y, s, tools, min_normal=min_normal, min_abnormal=min_abnormal)[0]


def boot_within_diff(y, s1, s2, tools, B=2000, seed=0):
    """Stratified bootstrap (resampling lots within each tool and class) for a difference in within-tool AUROC."""
    rng = np.random.default_rng(seed); y = np.asarray(y); tools = np.asarray(tools).astype(str)
    groups = [np.where((tools == t) & (y == c))[0] for t in np.unique(tools) for c in (0, 1)]
    groups = [g for g in groups if len(g)]
    obs = within_tool_auc(y, s1, tools) - within_tool_auc(y, s2, tools); d = []
    for _ in range(B):
        i = np.concatenate([rng.choice(g, len(g)) for g in groups])
        d.append(within_tool_auc(y[i], np.asarray(s1)[i], tools[i]) - within_tool_auc(y[i], np.asarray(s2)[i], tools[i]))
    d = np.array(d); d = d[np.isfinite(d)]
    if not len(d) or not np.isfinite(obs):
        return {"diff": obs, "lo": np.nan, "hi": np.nan, "p": np.nan}
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diff": float(obs), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "p": float(min(1.0, max(p, 1.0 / len(d))))}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def balanced_w(y):
    y = np.asarray(y); w = np.ones(len(y), float)
    for c in (0, 1):
        k = y == c
        if k.any():
            w[k] = 0.5 / k.sum()
    return w * len(y)


def tool_weights(y, tools):
    """Within-tool weights: each tool with both classes gets a weight proportional to its size, split equally between
    the classes; a tool with a single class gets weight 0 (it carries no within-tool information)."""
    y = np.asarray(y); tools = np.asarray(tools).astype(str); w = np.zeros(len(y)); N = len(y)
    mixed = [t for t in np.unique(tools) if 0 < y[tools == t].sum() < (tools == t).sum()]
    if not mixed:
        return balanced_w(y), False
    for t in mixed:
        k = tools == t; nt = k.sum()
        for c in (0, 1):
            kc = k & (y == c); w[kc] = 0.5 * (nt / N) / kc.sum()
    w = w / w[w > 0].mean()
    return w, True


# ============================================================ new models
def fit_ti_rf(X, y, tools, seed):
    w, ok = tool_weights(y, tools); k = w > 0
    if y[k].sum() < 2:
        return None
    c = CFG["TI-RF"]
    return RandomForestClassifier(c["n_estimators"], min_samples_leaf=c["min_samples_leaf"], max_features=c["max_features"],
                                  n_jobs=-1, random_state=seed).fit(X[k], y[k], sample_weight=w[k])


def fit_gbu(X, y, seed):
    c = CFG["GB-U"]
    return HistGradientBoostingClassifier(max_iter=c["max_iter"], learning_rate=c["learning_rate"], max_leaf_nodes=c["max_leaf_nodes"],
                                          l2_regularization=c["l2_regularization"], random_state=seed).fit(X, y, sample_weight=balanced_w(y))


def pu_bagging(X, y, Xev, seed):
    c = CFG["PU-RF"]; rng = np.random.default_rng(seed); P = np.where(y == 1)[0]; U = np.where(y == 0)[0]; out = np.zeros(len(Xev))
    for b in range(c["n_bags"]):
        u = rng.choice(U, min(len(U), c["unlabeled_ratio"] * len(P)), replace=False); i = np.r_[P, u]
        m = RandomForestClassifier(c["n_estimators"], min_samples_leaf=3, max_features="sqrt", class_weight="balanced",
                                   n_jobs=-1, random_state=seed + b).fit(X[i], y[i])
        out += m.predict_proba(Xev)[:, list(m.classes_).index(1)]
    return out / c["n_bags"]


def ssl_features(Zfit, Zall, seed):
    c = CFG["SSL-RF"]; rng = np.random.default_rng(seed)
    sc = StandardScaler().fit(Zfit); A = np.clip(sc.transform(Zfit), -10, 10); Aall = np.clip(sc.transform(Zall), -10, 10)
    dae = MLPRegressor(hidden_layer_sizes=tuple(c["hidden"]), activation="relu", max_iter=c["max_iter"], random_state=seed,
                       early_stopping=True, n_iter_no_change=20).fit(A + c["noise"] * rng.standard_normal(A.shape), A)
    h = Aall
    for W, b in zip(dae.coefs_[:2], dae.intercepts_[:2]):
        h = np.maximum(h @ W + b, 0)
    rec = ((dae.predict(Aall) - Aall) ** 2).mean(1, keepdims=True)
    return np.hstack([Aall, h, rec])


def iforest_score(Zfit_norm, Zev, seed):
    return -IsolationForest(n_estimators=300, random_state=seed).fit(Zfit_norm).score_samples(Zev)


def fit_ti_nn(X, y, tools, Xev, seed):
    if not J.HAS_TORCH:
        return None
    import torch, torch.nn as nn
    c = CFG["TI-NN"]; dev = J.DEV
    tv = {t: i for i, t in enumerate(sorted(set(map(str, tools))))}; tid = np.array([tv[str(t)] for t in tools], dtype=np.int64)
    sc = StandardScaler().fit(X); Xs = np.clip(sc.transform(X), -10, 10).astype(np.float32); Xe = np.clip(sc.transform(Xev), -10, 10).astype(np.float32)

    class GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, l):
            ctx.l = l; return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.l * g, None

    class Net(nn.Module):
        def __init__(s, d, nt):
            super().__init__()
            s.f = nn.Sequential(nn.Linear(d, c["hidden"]), nn.GELU(), nn.Dropout(c["dropout"]), nn.Linear(c["hidden"], c["hidden"]), nn.GELU())
            s.h = nn.Linear(c["hidden"], 1); s.a = nn.Sequential(nn.Linear(c["hidden"], c["hidden"]), nn.GELU(), nn.Linear(c["hidden"], nt))

        def forward(s, x, l=0.0):
            z = s.f(x); return s.h(z).squeeze(-1), s.a(GRL.apply(z, l))

    preds = []
    Xt = torch.as_tensor(Xs, device=dev); yt = torch.as_tensor(y, dtype=torch.float32, device=dev); tt = torch.as_tensor(tid.astype(np.int64), dtype=torch.long, device=dev)   # Windows: numpy int = int32
    wpos = float((y == 0).sum() / max(1, (y == 1).sum()))
    for e in range(c["ensemble"]):
        torch.manual_seed(seed * 100 + e); rng = np.random.default_rng(seed * 100 + e)
        net = Net(X.shape[1], len(tv)).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(wpos, device=dev)); ce = nn.CrossEntropyLoss()
        for ep in range(c["epochs"]):
            net.train()
            idx = np.arange(len(y)) if len(y) <= c["large_batch"] else rng.choice(len(y), c["large_batch"], replace=False)
            bi = torch.as_tensor(np.asarray(idx, np.int64), dtype=torch.long, device=dev); lam = c["lambda_adv"] * min(1.0, ep / (0.3 * c["epochs"]))
            s, a = net(Xt[bi], lam); yb = yt[bi]; tb = tt[bi]
            loss = bce(s, yb) + ce(a, tb)
            # pairwise ranking ONLY within the same tool
            same = (tb[:, None] == tb[None, :]) & (yb[:, None] == 1) & (yb[None, :] == 0)
            if same.any():
                d = s[:, None] - s[None, :]
                loss = loss + c["lambda_rank"] * torch.nn.functional.softplus(-d[same]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            preds.append(net(torch.as_tensor(Xe, device=dev))[0].cpu().numpy())
    return np.mean(preds, 0)


def hyb_scores(data, A, Sstd, fit, va, ev, seed):
    """Stacking with an inner CV on training lots only (no calibration or test information)."""
    yb = data.ybin; trf = np.concatenate([fit, va]) if len(va) else fit
    if yb[trf].sum() < CFG["HYB"]["inner_folds"]:
        return None
    def base(fit_i, ev_i):
        Z, OH, aw, s_oc = J.struct_features(data, A, fit_i)
        Xs = np.hstack([Sstd, Z, OH]); nf = fit_i[yb[fit_i] == 0]
        rf = RandomForestClassifier(300, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1,
                                    random_state=seed).fit(Xs[fit_i], yb[fit_i])
        p = rf.predict_proba(Xs[ev_i])[:, list(rf.classes_).index(1)]
        return np.c_[logit(p), s_oc[ev_i], iforest_score(Z[nf], Z[ev_i], seed)]
    oof = np.zeros((len(trf), 3))
    skf = StratifiedKFold(CFG["HYB"]["inner_folds"], shuffle=True, random_state=seed)
    for a, b in skf.split(trf, yb[trf]):
        oof[b] = base(trf[a], trf[b])
    sc = StandardScaler().fit(oof)
    meta = LogisticRegression(C=CFG["HYB"]["meta_C"], class_weight="balanced", max_iter=2000).fit(sc.transform(oof), yb[trf])
    return meta.decision_function(sc.transform(base(trf, ev))), dict(zip(["w_rf", "w_oneclass", "w_iforest"], meta.coef_[0].tolist()))


# ============================================================ one fold
REF = ["B1", "B3-RF", "B4", "B5", "Ours-S"]


def features(data, fit):
    """Lot features referenced to the normal 'fit' lots only: B3 summary + SALE-S structure features + one-hot route/context."""
    yb = data.ybin
    Xn = J.crn(data, fit[yb[fit] == 0]); Xp = J.peer_contrast(data, Xn); A = J.lot_arrays(data, Xn, Xp)
    S, _ = J.lot_summary(data, Xn); Sstd = np.clip(StandardScaler().fit(S[fit]).transform(S), -10, 10)
    Z, OH, aw, s_oc = J.struct_features(data, A, fit)
    return np.hstack([Sstd, Z, OH]), A, Sstd, Z


def meta_combine(Mb, yblend, Mall):
    """Logistic regression on base-detector scores standardized on the blending set."""
    mu = np.nanmean(Mb, 0); sd = np.nanstd(Mb, 0) + 1e-9
    f = lambda M: np.nan_to_num(np.clip((M - mu) / sd, -10, 10))
    lr = LogisticRegression(C=CFG["MCE"]["C"], class_weight="balanced", max_iter=5000).fit(f(Mb), yblend)
    return lr.decision_function(f(Mall)), lr.coef_[0]


def mce_inner(data, trf, seed):
    """Out-of-fold scores of the classical base detectors on training lots (inner CV), used to fit the MCE meta-learner
    on production lines."""
    yb = data.ybin; tools = data.lots["tool"].astype(str).values; bases = CFG["MCE"]["bases_production"]
    oof = np.full((len(trf), len(bases)), np.nan)
    skf = StratifiedKFold(CFG["MCE"]["inner_folds"], shuffle=True, random_state=seed)
    for a, b in skf.split(trf, yb[trf]):
        fi, vi = trf[a], trf[b]
        r = J.run_fold(data, fi, np.array([], int), np.array([], int), vi, {}, [m for m in bases if m in J.CLASSIC], seed=seed, opts={"verbose": False})
        X, *_ = features(data, fi)
        for j, m in enumerate(bases):
            if m in r and isinstance(r[m], dict) and "score" in r[m]:
                oof[b, j] = r[m]["score"]
            elif m == "TI-RF":
                mm = fit_ti_rf(X[fi], yb[fi], tools[fi], seed)
                if mm is not None:
                    oof[b, j] = logit(mm.predict_proba(X[vi])[:, list(mm.classes_).index(1)])
            elif m == "GB-U":
                oof[b, j] = logit(fit_gbu(X[fi], yb[fi], seed).predict_proba(X[vi])[:, 1])
    return oof


def run_fold(data, fit, va, cal, te, seed, extra_unlabeled=None):
    yb = data.ybin; ev = np.concatenate([cal, te]); trf = np.concatenate([fit, va]) if len(va) else fit
    tools = data.lots["tool"].astype(str).values
    ref = J.run_fold(data, fit, va, cal, te, {}, REF + (["B6"] if J.HAS_TORCH else []), seed=seed, opts={"verbose": False, "epochs": 30})
    out = {m: np.asarray(r["score"]) for m, r in ref.items() if isinstance(r, dict) and "score" in r}
    Xn = J.crn(data, fit[yb[fit] == 0]); Xp = J.peer_contrast(data, Xn); A = J.lot_arrays(data, Xn, Xp)
    S, _ = J.lot_summary(data, Xn); Sstd = np.clip(StandardScaler().fit(S[fit]).transform(S), -10, 10)
    Z, OH, aw, s_oc = J.struct_features(data, A, fit)
    X = np.hstack([Sstd, Z, OH]); info = {}
    if yb[trf].sum() < 2:
        return out, info, ev
    t0 = time.time()
    m = fit_ti_rf(X[trf], yb[trf], tools[trf], seed)
    if m is not None:
        out["TI-RF"] = logit(m.predict_proba(X[ev])[:, list(m.classes_).index(1)])
    s = fit_ti_nn(X[trf], yb[trf], tools[trf], X[ev], seed)
    if s is not None:
        out["TI-NN"] = s
    h = hyb_scores(data, A, Sstd, fit, va, ev, seed)
    if h is not None:
        out["HYB"], info["HYB_weights"] = h
    out["GB-U"] = logit(fit_gbu(X[trf], yb[trf], seed).predict_proba(X[ev])[:, 1])
    out["PU-RF"] = logit(pu_bagging(X[trf], yb[trf], X[ev], seed))
    Zall = Z if extra_unlabeled is None else np.vstack([Z, extra_unlabeled])
    F = ssl_features(Z[trf] if extra_unlabeled is None else np.vstack([Z[trf], extra_unlabeled]), Z, seed)
    rf = RandomForestClassifier(300, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1,
                                random_state=seed).fit(F[trf], yb[trf])
    out["SSL-RF"] = logit(rf.predict_proba(F[ev])[:, list(rf.classes_).index(1)])
    # MCE (production): meta-learner on out-of-fold scores of an inner CV
    bases = CFG["MCE"]["bases_production"]
    if yb[trf].sum() >= max(CFG["MCE"]["inner_folds"], CFG["MCE"]["min_abnormal_blend"]) and all(m in out for m in bases):
        oof = mce_inner(data, trf, seed); ok = ~np.isnan(oof).any(1)
        if ok.sum() > 10 and yb[trf][ok].sum() >= 2:
            out["MCE"], w = meta_combine(oof[ok], yb[trf][ok], np.column_stack([out[m] for m in bases]))
            info["MCE_weights"] = dict(zip(bases, np.round(w, 3).tolist()))
    info["sec_new"] = time.time() - t0
    return out, info, ev


# ============================================================ protocols
def eval_scores(data, idx_te, cal, scores_ev, ncal_used):
    yb = data.ybin; tools = data.lots["tool"].astype(str).values; res = {}
    for m, s in scores_ev.items():
        s = np.asarray(s); st = s[ncal_used:]; sc = s[:ncal_used][yb[cal] == 0]
        r = {"score_te": st.tolist(), "pval_te": J.conformal_pvalues(sc, st).tolist()}
        for e in CFG["eps"]:
            tau = J.conformal_tau(sc, e); nt = yb[idx_te] == 0
            r[f"far{e}"] = float((st[nt] > tau).mean()) if nt.any() else np.nan
        res[m] = r
    return res


def cv_protocol(data, R, seed0=0, extra=None):
    yb = data.ybin; L = len(yb); nb = int(yb.sum()); reps = []
    for r in range(R):
        skf = StratifiedKFold(min(CFG["cv"]["folds"], nb), shuffle=True, random_state=5000 + r)
        sc = defaultdict(dict); pv = defaultdict(dict); far = defaultdict(lambda: defaultdict(list)); infos = []
        for k, (tr, te) in enumerate(skf.split(np.zeros(L), yb)):
            fit, va, cal = J.split_train(tr, yb, seed=seed0 + r * 100 + k, cal_frac=CFG["cv"]["frac_calibration_normal"])
            out, info, ev = run_fold(data, fit, va, cal, te, seed0 + r * 100 + k, extra)
            infos.append(info)
            res = eval_scores(data, te, cal, out, len(cal))
            for m, v in res.items():
                for i, x, q in zip(te, v["score_te"], v["pval_te"]):
                    sc[m][int(i)] = x; pv[m][int(i)] = q
                for e in CFG["eps"]:
                    far[m][e].append(v[f"far{e}"])
        log(f"   {data.name}: repetition {r + 1}/{R} done")
        reps.append({"scores": {m: d for m, d in sc.items()}, "pvals": {m: d for m, d in pv.items()},
                     "far": {m: {str(e): float(np.nanmean(v)) for e, v in d.items()} for m, d in far.items()}, "info": infos})
    return reps


def summarize(data, reps):
    yb = data.ybin; tools = data.lots["tool"].astype(str).values; S = {}
    methods = sorted({m for rp in reps for m in rp["scores"]})
    for m in methods:
        au, wt, pr, aup = [], [], [], []
        for rp in reps:
            if m not in rp["scores"]:
                continue
            idx = np.array(sorted(rp["scores"][m])); s = np.array([rp["scores"][m][i] for i in idx])
            au.append(J.auroc(yb[idx], s)); wt.append(within_tool_auc(yb[idx], s, tools[idx])); pr.append(J.prauc(yb[idx], s))
            if m in rp.get("pvals", {}):
                aup.append(J.auroc(yb[idx], -np.array([rp["pvals"][m][i] for i in idx])))
        far5 = [rp["far"][m]["0.05"] for rp in reps if m in rp["far"]]
        S[m] = {"AUROC": float(np.nanmean(au)), "AUROC_pct": [float(np.nanpercentile(au, 2.5)), float(np.nanpercentile(au, 97.5))],
                "withinAUROC": float(np.nanmean(wt)), "within_pct": [float(np.nanpercentile(wt, 2.5)), float(np.nanpercentile(wt, 97.5))] if np.isfinite(wt).any() else [np.nan, np.nan],
                "PR-AUC": float(np.nanmean(pr)), "FAR5": float(np.nanmean(far5)), "FAR5_reps": far5,
                "AUROC_pval": float(np.nanmean(aup)) if aup else np.nan}
    # paired DeLong vs B3-RF on EVERY repetition (Holm over detectors within a repetition): how often is a detector
    # significantly better / worse? (a single repetition is one arbitrary partition of the lots)
    ref = "B3-RF"
    for m in methods:
        if m != ref:
            S[m].update(reps_sig_better=0, reps_sig_worse=0, reps_tested=0)
    for rp in reps:
        if ref not in rp["scores"]:
            continue
        idx = np.array(sorted(rp["scores"][ref])); y = yb[idx]; s_ref = np.array([rp["scores"][ref][i] for i in idx])
        comps = [m for m in methods if m != ref and m in rp["scores"] and set(rp["scores"][m]) >= set(idx.tolist())]
        zs, ps_ = [], []
        for m in comps:
            _, _, z, p = J.delong_paired(y, np.array([rp["scores"][m][i] for i in idx]), s_ref); zs.append(z); ps_.append(p if np.isfinite(p) else 1.0)
        for m, z, ph in zip(comps, zs, J.holm(ps_)):
            S[m]["reps_tested"] += 1
            if ph < .05 and z > 0:
                S[m]["reps_sig_better"] += 1
            elif ph < .05 and z < 0:
                S[m]["reps_sig_worse"] += 1
    # detailed tests on repetition 1 (z, p, bootstrap of the within-tool difference)
    rp = reps[0]; ref = "B3-RF"
    if ref in rp["scores"]:
        idx = np.array(sorted(rp["scores"][ref])); y = yb[idx]; t = tools[idx]; s_ref = np.array([rp["scores"][ref][i] for i in idx])
        comps = [m for m in methods if m != ref and m in rp["scores"]]; pd_, pw_ = [], []
        for m in comps:
            s = np.array([rp["scores"][m][i] for i in idx])
            a1, a2, z, p = J.delong_paired(y, s, s_ref); b = boot_within_diff(y, s, s_ref, t, B=CFG["bootstrap"], seed=1)
            S[m].update({"delong_z": z, "delong_p": p, "within_diff": b["diff"], "within_lo": b["lo"], "within_hi": b["hi"], "within_p": b["p"]})
            pd_.append(p); pw_.append(b["p"] if np.isfinite(b["p"]) else 1.0)
        for m, a, b in zip(comps, J.holm(pd_), J.holm(pw_)):
            S[m]["delong_p_holm"] = float(a); S[m]["within_p_holm"] = float(b)
    return S


def loto_protocol(data, seed=11):
    yb = data.ybin; res = {}
    for tl, g in data.lots.groupby("tool"):
        nb_, ng_ = int(g["y"].gt(0).sum()), int((g["y"] == 0).sum())
        if nb_ < 2 or ng_ < 5:
            continue
        te = g.index.values; tr = np.setdiff1d(np.arange(len(yb)), te)
        if yb[tr].sum() < 2:
            continue
        fit, va, cal = J.split_train(tr, yb, seed=seed)
        out, _, ev = run_fold(data, fit, va, cal, te, seed)
        res[str(tl)] = {"bad": nb_, "good": ng_, **{m: J.auroc(yb[te], np.asarray(s)[len(cal):]) for m, s in out.items()}}
        res[str(tl)]["_stats"] = {m: J.auroc_hanley(v, nb_, ng_) for m, v in res[str(tl)].items() if m not in ("bad", "good")}
    return res


def time_protocol(data, seed=21):
    if "time" not in data.lots or pd.to_datetime(data.lots["time"]).notna().mean() < .8:
        return {"note": "lot dates not complete enough; time split skipped"}
    yb = data.ybin; t = pd.to_datetime(data.lots["time"]); ok = np.where(t.notna())[0]; o = ok[np.argsort(t.values[ok])]
    n_tr = int(CFG["time_split"]["frac_train"] * len(o)); tr, te = o[:n_tr], o[n_tr:]
    if yb[te].sum() < 2 or yb[tr].sum() < 2:
        return {"note": f"too few abnormal lots in one of the periods (train {int(yb[tr].sum())}, test {int(yb[te].sum())})"}
    nrm = tr[yb[tr] == 0]; ncal = max(10, int(CFG["cv"]["frac_calibration_normal"] * len(nrm)))
    cal = nrm[-ncal:]; fit = np.setdiff1d(tr, cal); va = np.array([], int)        # calibration = the MOST RECENT normal lots of the training period
    out, _, ev = run_fold(data, fit, va, cal, te, seed)
    res = eval_scores(data, te, cal, out, len(cal)); tools = data.lots["tool"].astype(str).values
    return {"train": [str(t.values[tr[0]])[:10], str(t.values[tr[-1]])[:10]], "test": [str(t.values[te[0]])[:10], str(t.values[te[-1]])[:10]],
            "abnormal_test": int(yb[te].sum()), "normal_test": int((yb[te] == 0).sum()),
            **{m: {**J.auroc_hanley(J.auroc(yb[te], np.asarray(v["score_te"])), int(yb[te].sum()), int((yb[te] == 0).sum())),
                   "withinAUROC": within_tool_auc(yb[te], np.asarray(v["score_te"]), tools[te]), "FAR5": v["far0.05"]} for m, v in res.items()}}



def month_protocol(data, seed=31):
    """Monthly forward chaining: train on all earlier months, test on the following month."""
    if "month" not in data.lots:
        return {"note": "no month column"}
    yb = data.ybin; mon = data.lots["month"].astype(str).values; tools = data.lots["tool"].astype(str).values
    t = pd.to_datetime(data.lots["time"]) if "time" in data.lots else None
    months = sorted(m for m in set(mon) if m not in ("NA", "NaT", "nan"))
    sc = defaultdict(dict); pv = defaultdict(dict); far = defaultdict(list); used = []
    for m in months:
        tr = np.where((mon < m) & ~np.isin(mon, ["NA", "NaT", "nan"]))[0]; te = np.where(mon == m)[0]
        if yb[tr].sum() < CFG["monthly"]["min_abnormal_train"] or (yb[tr] == 0).sum() < CFG["monthly"]["min_normal_train"] or not len(te):
            continue
        nrm = tr[yb[tr] == 0]
        if t is not None and t.notna().all():
            nrm = nrm[np.argsort(t.values[nrm])]
        ncal = max(10, int(CFG["cv"]["frac_calibration_normal"] * len(nrm))); cal = nrm[-ncal:]; fit = np.setdiff1d(tr, cal)
        out, _, ev = run_fold(data, fit, np.array([], int), cal, te, seed)
        res = eval_scores(data, te, cal, out, len(cal)); used.append(m)
        for k, v in res.items():
            for i, x, q in zip(te, v["score_te"], v["pval_te"]):
                sc[k][int(i)] = x; pv[k][int(i)] = q
            far[k].append(v["far0.05"])
        log(f"   {data.name}: test month {m} ({int(yb[te].sum())} abnormal / {int((yb[te] == 0).sum())} normal)")
    if not used or "B3-RF" not in sc:
        return {"note": "too few abnormal lots for monthly forward chaining"}
    idx = np.array(sorted(sc["B3-RF"])); y = yb[idx]; tl = tools[idx]; sr = np.array([sc["B3-RF"][i] for i in idx]); R = {"test_months": used,
         "abnormal_test": int(y.sum()), "normal_test": int((y == 0).sum())}
    comps = [k for k in sc if k != "B3-RF"]; ps = []
    for k in ["B3-RF"] + comps:
        s = np.array([sc[k].get(i, np.nan) for i in idx]); ok = np.isfinite(s)
        q = np.array([pv[k].get(i, np.nan) for i in idx]); okq = np.isfinite(q)
        R[k] = {**J.auroc_hanley(J.auroc(y[ok], s[ok]), int(y[ok].sum()), int((y[ok] == 0).sum())),
                "withinAUROC": within_tool_auc(y[ok], s[ok], tl[ok]), "FAR5": float(np.nanmean(far[k])),
                "AUROC_pval": J.auroc(y[okq], -q[okq])}     # pooled conformal p-values: no mixing of monthly score scales
        if k != "B3-RF" and ok.all() and y.sum() >= 2:
            a1, a2, z, p = J.delong_paired(y, s, sr); R[k].update(z=z, p=p); ps.append((k, p))
    for (k, _), ph in zip(ps, J.holm([p for _, p in ps])):
        R[k]["p_holm"] = float(ph)
    R["per_lot"] = {"idx": idx.tolist(), "scores": {k: [sc[k].get(int(i)) for i in idx] for k in sc}}
    return R


# ============================================================ benchmark (LotBench-811K, WM-811K lot task)
def _wm_args(a):
    from types import SimpleNamespace
    wm = J.find_public(J.PUBLIC_DATA_DIR, ["LSWMD.pkl"]); mx = J.find_public(J.PUBLIC_DATA_DIR, ["MixedWM38.npz", "Wafer_Map_Datasets.npz"])
    return SimpleNamespace(demo=a.quick_test, demo_lots=400 if a.quick_test else 600, wm811k=wm, mixedwm38=mx)


def bench_fold(data, fit, blend, cal, te, seed, opts):
    yb = data.ybin; tools = data.lots["tool"].astype(str).values
    methods = J.CLASSIC + (J.NEURAL if J.HAS_TORCH else [])
    r = J.run_fold(data, fit, np.array([], int), np.r_[blend, cal], te, {}, methods, seed=seed, opts=opts)
    E = np.r_[blend, cal, te]; out = {m: np.asarray(v["score"]) for m, v in r.items() if isinstance(v, dict) and "score" in v}
    X, A, Sstd, Z = features(data, fit)
    m = fit_ti_rf(X[fit], yb[fit], tools[fit], seed)
    if m is not None:
        out["TI-RF"] = logit(m.predict_proba(X[E])[:, list(m.classes_).index(1)])
    out["GB-U"] = logit(fit_gbu(X[fit], yb[fit], seed).predict_proba(X[E])[:, 1])
    out["PU-RF"] = logit(pu_bagging(X[fit], yb[fit], X[E], seed))
    F = ssl_features(Z[fit], Z, seed)
    rf = RandomForestClassifier(300, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=seed).fit(F[fit], yb[fit])
    out["SSL-RF"] = logit(rf.predict_proba(F[E])[:, list(rf.classes_).index(1)])
    s = fit_ti_nn(X[fit], yb[fit], tools[fit], X[E], seed)
    if s is not None:
        out["TI-NN"] = s
    if len(data.ctx_cols):                # context-only shortcut check (CTX; never a base detector of MCE)
        rc = J.run_fold(data, fit, np.array([], int), np.r_[blend, cal], te, {}, ["CTX"], seed=seed, opts=dict(opts, verbose=False))
        if isinstance(rc.get("CTX"), dict) and "score" in rc["CTX"]:
            out["CTX"] = np.asarray(rc["CTX"]["score"])
    bases = [b for b in CFG["MCE"]["bases_benchmark"] if b in out]; nb = len(blend); w = None
    if yb[blend].sum() >= CFG["MCE"]["min_abnormal_blend"] and len(bases) >= 2:
        M = np.column_stack([out[b] for b in bases]); out["MCE"], w = meta_combine(M[:nb], yb[blend], M)
    return out, E, (dict(zip(bases, np.round(w, 3).tolist())) if w is not None else None)


def bench_eval(data, out, E, nb, cal, te, scen=None):
    yb = data.ybin; n_bc = nb + len(cal); y = yb[te]; R = {}
    sref = out["B3-RF"][n_bc:]; ps = []
    for m, s in out.items():
        st = np.asarray(s)[n_bc:]; sc = np.asarray(s)[nb:n_bc][yb[cal] == 0]; tau = J.conformal_tau(sc, 0.05)
        e = {"AUROC": J.auroc(y, st), "PR-AUC": J.prauc(y, st), "FAR5": float((st[y == 0] > tau).mean()), "recall5": float((st[y == 1] > tau).mean())}
        n_fa, n_n = int((st[y == 0] > tau).sum()), int((y == 0).sum())       # exact binomial test of the realized FAR vs 5%
        e.update(n_false_alarms=n_fa, n_normal_test=n_n, p_binom_FAR5=float(stats.binomtest(n_fa, n_n, 0.05).pvalue) if n_n else np.nan)
        if scen is not None:
            sct = scen[te]; e["per_scen"] = {k: J.auroc(y[(sct == k) | (y == 0)], st[(sct == k) | (y == 0)]) for k in J.SCEN}
            # sensitivity: F5 (material batch) is identified by the context 'batch' and F4 (upstream tool) by route + time
            # BY DESIGN of the generator; these AUROCs show the ranking without them
            k5 = ~np.isin(sct, ["F5"]); k45 = ~np.isin(sct, ["F4", "F5"])
            e["AUROC_excl_F5"] = J.auroc(y[k5], st[k5]); e["AUROC_excl_F4F5"] = J.auroc(y[k45], st[k45])
        if m != "B3-RF":
            a1, a2, z, p = J.delong_paired(y, st, sref); e.update(z=z, p=p)
            if m not in EXCLUDE_FROM_HOLM:
                ps.append((m, p))
        R[m] = e
    for (m, _), ph in zip(ps, J.holm([p for _, p in ps])):
        R[m]["p_holm"] = float(ph)
    return R


def nb_corrected_t(d, ratio):
    """Corrected resampled t test (Nadeau & Bengio, 2003, Machine Learning 52:239-281) for paired differences d over
    J resampled splits; ratio = n_test / n_train. Variance factor (1/J + ratio) instead of 1/J."""
    d = np.asarray(d, float); J_ = len(d)
    if J_ < 2 or not np.isfinite(ratio) or d.std(ddof=1) == 0:
        return np.nan, np.nan
    t = d.mean() / np.sqrt((1.0 / J_ + ratio) * d.var(ddof=1))
    return float(t), float(2 * stats.t.sf(abs(t), J_ - 1))


EXCLUDE_FROM_HOLM = {"CTX"}          # the context-only model is a shortcut check, not a candidate detector


def bench_summary(per_seed, ratio=np.nan):
    """Summary over seeds. ratio = n_test / n_train (for the corrected resampled t test).
    For 'AUROC' and, on LotBench, 'AUROC_excl_F5' and 'AUROC_excl_F4F5': mean, sd, difference to B3-RF, paired t,
    corrected t (Nadeau-Bengio), Holm over detectors (CTX excluded)."""
    methods = sorted({m for s in per_seed.values() for m in s}); ref = [s["B3-RF"]["AUROC"] for s in per_seed.values()]; S = {}; ps = []
    for m in methods:
        v = [s[m]["AUROC"] for s in per_seed.values() if m in s]
        S[m] = {"AUROC_mean": float(np.mean(v)), "AUROC_sd": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan, "n_seeds": len(v)}
        if m != "B3-RF" and len(v) == len(ref) and len(v) >= 2:
            d = np.array(v) - np.array(ref); tt = stats.ttest_rel(v, ref); h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            S[m].update(diff=float(d.mean()), lo=float(d.mean() - h), hi=float(d.mean() + h), t=float(tt.statistic), p_t=float(tt.pvalue), df=len(d) - 1)
            if len(d) >= 6:
                S[m]["p_wilcoxon"] = float(stats.wilcoxon(v, ref).pvalue)
            S[m]["t_nb"], S[m]["p_nb"] = nb_corrected_t(d, ratio)
            S[m]["seeds_better"] = int(sum(1 for s in per_seed.values() if m in s and s[m].get("p_holm", 1) < .05 and s[m]["z"] > 0))
            S[m]["seeds_worse"] = int(sum(1 for s in per_seed.values() if m in s and s[m].get("p_holm", 1) < .05 and s[m]["z"] < 0))
            if m not in EXCLUDE_FROM_HOLM:
                ps.append((m, S[m]["p_t"], S[m]["p_nb"]))
        if all("per_scen" in s[m] for s in per_seed.values() if m in s):
            S[m]["per_scen"] = {k: float(np.nanmean([s[m]["per_scen"][k] for s in per_seed.values() if m in s])) for k in J.SCEN}
    for (m, _, _), ph, pn in zip(ps, J.holm([p for _, p, _ in ps]), J.holm([1.0 if not np.isfinite(q) else q for _, _, q in ps])):
        S[m]["p_t_holm"] = float(ph); S[m]["p_nb_holm"] = float(pn)
    # sensitivity without the scenarios that context/route identify by design (LotBench only)
    for key in ("AUROC_excl_F5", "AUROC_excl_F4F5"):
        if not all(key in s.get("B3-RF", {}) for s in per_seed.values()):
            continue
        r0 = np.array([s["B3-RF"][key] for s in per_seed.values()]); ps2 = []
        for m in methods:
            v = [s[m][key] for s in per_seed.values() if m in s and key in s[m]]
            if len(v) != len(r0):
                continue
            S[m][key + "_mean"] = float(np.mean(v))
            if m != "B3-RF":
                d = np.array(v) - r0; S[m][key + "_diff"] = float(d.mean())
                S[m][key + "_t_nb"], S[m][key + "_p_nb"] = nb_corrected_t(d, ratio)
                if m not in EXCLUDE_FROM_HOLM:
                    ps2.append((m, S[m][key + "_p_nb"]))
        for (m, _), ph in zip(ps2, J.holm([1.0 if not np.isfinite(p) else p for _, p in ps2])):
            S[m][key + "_p_nb_holm"] = float(ph)
    return S


def run_benchmark(kind, a, outdir):
    meta, maps = J.get_wm811k(_wm_args(a))
    if meta is None:
        log(f"[{kind}] WM-811K not found"); return None
    MIX = (None, None)
    wa = _wm_args(a)
    if wa.mixedwm38 and not a.quick_test:
        MIX = J.load_mixedwm38(wa.mixedwm38)
    seeds = [0, 1] if a.quick_test else CFG["benchmark"]["seeds"]
    opts = dict(epochs=2 if a.quick_test else 30, pre_epochs=1 if a.quick_test else 100, menc_epochs=1 if a.quick_test else 3, attr=False, verbose=False)
    per_seed, raw, mw = {}, {}, {}; ratios = []
    wmd = J.make_wmlot(meta, maps) if kind == "wmlot" else None
    from sklearn.model_selection import train_test_split
    for seed in seeds:
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
        fit, blend = train_test_split(tr, test_size=0.10, stratify=d.ybin[tr], random_state=seed)
        out, E, w = bench_fold(d, fit, blend, cal, te, seed, opts)
        per_seed[seed] = bench_eval(d, out, E, len(blend), cal, te, scen); mw[seed] = w; ratios.append(len(te) / len(fit))
        raw[seed] = {"te": te.tolist(), "y_te": d.ybin[te].tolist(), "scen_te": scen[te].tolist() if scen is not None else None,
                     "scores_te": {m: np.asarray(s)[len(blend) + len(cal):].tolist() for m, s in out.items()}}
        log(f"[{kind}] seed {seed}: " + ", ".join(f"{m} {v['AUROC']:.3f}" for m, v in per_seed[seed].items()))
        J.jdump({"per_seed": per_seed, "mce_weights": mw}, os.path.join(outdir, "raw", f"{kind}.json"))
    ratio = float(np.mean(ratios)) if ratios else np.nan
    R = {"per_seed": per_seed, "summary": bench_summary(per_seed, ratio), "mce_weights": mw, "test_train_ratio": ratio}
    J.jdump(R, os.path.join(outdir, "raw", f"{kind}.json")); J.jdump(raw, os.path.join(outdir, "raw", f"{kind}_scores.json"))
    return R


def run_ablation(a, outdir):
    if not J.HAS_TORCH:
        log("[ablation] PyTorch not available; skipped"); return None
    meta, maps = J.get_wm811k(_wm_args(a)); res = {}
    from sklearn.model_selection import train_test_split
    for seed in ([0] if a.quick_test else CFG["benchmark"]["ablation_seeds"]):
        d = J.make_lotbench(meta, maps, seed, n_lots=600 if a.quick_test else 20000)
        tr = np.where(d.lots["split"] == "train")[0]; cal = np.where(d.lots["split"] == "cal")[0]; te = np.where(d.lots["split"] == "test")[0]
        fit, va = train_test_split(tr, test_size=0.10, stratify=d.ybin[tr], random_state=seed)
        opts = dict(epochs=2 if a.quick_test else 30, pre_epochs=1 if a.quick_test else 100, attr=False, verbose=False)
        full = J.run_fold(d, fit, va, cal, te, {}, ["Ours"], seed=seed, opts=opts)["Ours"]["score"][len(cal):]; y = d.ybin[te]; rows = {}; ps = []
        for nm, ch in J.ABLATIONS.items():
            s = J.run_fold(d, fit, va, cal, te, ch, ["Ours"], seed=seed, opts=opts)["Ours"].get("score")
            if s is None:
                continue
            s = np.asarray(s)[len(cal):]; a1, a2, z, p = J.delong_paired(y, s, full)
            rows[nm] = {"AUROC": a1, "AUROC_full": a2, "delta": a1 - a2, "z": z, "p": p}; ps.append((nm, p))
            log(f"[ablation] seed {seed}: {nm}: Δ {a1 - a2:+.3f} (p {p:.3g})")
            J.jdump(res | {seed: rows}, os.path.join(outdir, "raw", "ablation.json"))
        for (nm, _), ph in zip(ps, J.holm([p for _, p in ps])):
            rows[nm]["p_holm"] = float(ph)
        res[seed] = rows
    J.jdump(res, os.path.join(outdir, "raw", "ablation.json")); return res

# ============================================================ data
def demo_data(seed=0, n=700):
    """SYNTHETIC data with tool confounding (mimics Plant B): one small tool contains almost only abnormal lots."""
    meta, maps = J.demo_wm811k(n, seed); d = J.make_lotbench(meta, maps, seed, n_lots=n)
    L = d.lots.copy(); L["y"] = (L["y"] > 0).astype(int); rng = np.random.default_rng(seed)
    tool = np.where(rng.random(len(L)) < .5, "T1", "T2").astype(object)
    ab = np.where(L["y"] == 1)[0]; tool[rng.choice(ab, len(ab) // 2, replace=False)] = "T9"
    tool[rng.choice(np.where(L["y"] == 0)[0], 2, replace=False)] = "T9"
    L["tool"] = tool; L["subset_same"] = True
    W = d.W.drop(columns=["ch_id", "ch_loc", "map_i"]).copy()
    # tool T9 has a sensor offset (a tool-identity signal that can be memorized)
    shift = np.isin(W["li"].values, np.where(tool == "T9")[0])
    for c in d.groups["FDC"][:8]:
        W.loc[shift, c] = W.loc[shift, c] + 2.0
    out = J.LotData("demo", L, W, d.groups, d.keys, [], ["abnormal"], notes={"lot": len(L), "abnormal": int(L.y.sum())})
    out.ctx_cols = []
    return out


def load_prod(nm, a):
    if nm == "plantb":
        d = J.load_plantb(a.folder, a.label) if os.path.exists(a.label) else None
    elif nm == "diesaw":
        d = J.load_diesaw(a.folder, None, k_bad=3)
    else:
        d = J.load_faba(a.folder)
    if d is not None:
        d.ctx_cols = []                       # as in the JIM paper: no context in production models
    return d


# ============================================================ report
def fmt(v, d=3):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def fp(p):
    return "—" if p is None or not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def report(outdir, results):
    md = [f"# REPORT OF THE EXTENDED EXPERIMENTS ({time.strftime('%Y-%m-%d %H:%M')}, {CFG['version']})", "",
          f"Primary metric: {CFG['primary_metric']}.", f"Claim rule: {CFG['claim_rule']}.", ""]
    for nm, R in results.items():
        if nm.startswith("_"):
            continue
        S = R["cv"]; md += [f"## {nm} ({R['notes']})", "",
                            "| Method | pooled AUROC [range over reps] | AUROC of pooled p-values | DeLong vs B3-RF z (Holm p), rep 1 | reps sig. better / worse / tested | within-tool AUROC | within-tool Δ vs B3-RF [95% CI] (Holm p) | PR-AUC | FAR ε=5% |",
                            "|---|---|---|---|---|---|---|---|---|"]
        order = ["B3-RF"] + [m for m in ["B1", "B5", "B6", "Ours-S"] if m in S] + [m for m in NEW if m in S]
        for m in order:
            e = S[m]; ref = m == "B3-RF"
            md.append(f"| {m} | {fmt(e['AUROC'])} [{fmt(e['AUROC_pct'][0], 2)}, {fmt(e['AUROC_pct'][1], 2)}] | {fmt(e.get('AUROC_pval'))} | "
                      f"{'reference' if ref else fmt(e.get('delong_z'), 2) + ' (' + fp(e.get('delong_p_holm')) + ')'} | "
                      f"{'' if ref else str(e.get('reps_sig_better')) + ' / ' + str(e.get('reps_sig_worse')) + ' / ' + str(e.get('reps_tested'))} | "
                      f"{fmt(e['withinAUROC'])} [{fmt(e['within_pct'][0], 2)}, {fmt(e['within_pct'][1], 2)}] | "
                      f"{'reference' if ref else fmt(e.get('within_diff'), 3) + ' [' + fmt(e.get('within_lo'), 3) + ', ' + fmt(e.get('within_hi'), 3) + '] (' + fp(e.get('within_p_holm')) + ')'} | "
                      f"{fmt(e['PR-AUC'])} | {fmt(100 * e['FAR5'], 1)}% |")
        nrep = max([S[m].get("reps_tested", 0) for m in S] + [1])
        win = [m for m in NEW if m in S and S[m].get("reps_sig_better", 0) > nrep / 2 and not (S[m].get("within_p_holm", 1) < .05 and S[m].get("within_diff", 0) < 0)]
        lose = [m for m in S if m != "B3-RF" and S[m].get("reps_sig_worse", 0) > nrep / 2]
        md += ["", "**Decision (frozen rule):** " + (f"better than B3-RF: {', '.join(win)}." if win else
                                                    "no new model is better than B3-RF (Holm-DeLong p < 0.05 on most repetitions)."),
               f"Significantly WORSE than B3-RF on most repetitions: {', '.join(lose) if lose else 'none'}.",
               "Within-tool AUROC significantly lower than B3-RF (repetition 1, bootstrap, Holm): " +
               (", ".join(m for m in S if S[m].get("within_p_holm", 1) < .05 and S[m].get("within_diff", 0) < 0) or "none") + "."]
        if R.get("loto"):
            md += ["", "Leave-one-tool-out (AUROC): " + "; ".join(f"{t} ({v['bad']}/{v['good']}): " + ", ".join(f"{m} {fmt(v.get(m), 2)}" for m in order if m in v) for t, v in R["loto"].items())]
        if R.get("time"):
            T = R["time"]
            if "note" in T:
                md += ["", f"Time split: {T['note']}"]
            else:
                md += ["", f"Time split (train {T['train']}, test {T['test']}, {T['abnormal_test']} abnormal / {T['normal_test']} normal): " +
                       "; ".join(f"{m} AUROC {fmt(T[m]['AUROC'])}, within-tool {fmt(T[m]['withinAUROC'])}" for m in order if m in T)]
        if R.get("month"):
            Mo = R["month"]
            if "note" in Mo:
                md += ["", f"Monthly (forward chaining): {Mo['note']}"]
            else:
                md += ["", f"Monthly (forward chaining; {len(Mo['test_months'])} test months, {Mo['abnormal_test']} abnormal / {Mo['normal_test']} normal): " +
                       "; ".join(f"{m} AUROC {fmt(Mo[m]['AUROC'])}" + (f" (z {fmt(Mo[m].get('z'), 2)}, Holm p {fp(Mo[m].get('p_holm'))})" if m != "B3-RF" and "z" in Mo[m] else "") +
                                 f", within-tool {fmt(Mo[m]['withinAUROC'])}" for m in order if m in Mo)]
        if R.get("mce_w"):
            md += ["", f"MCE meta-learner weights (mean over folds): {R['mce_w']}"]
        if R.get("hyb_w"):
            md += ["", f"HYB meta-learner weights (mean over folds): {R['hyb_w']}"]
        md.append("")
    for kind in ("lotbench", "wmlot"):
        B = results.get("_" + kind)
        if not B:
            continue
        S = B["summary"]; n = len(B["per_seed"])
        md += [f"## {kind} ({n} seed; n_test/n_train = {fmt(B.get('test_train_ratio'))})", "",
               "| Method | AUROC mean ± sd | Δ vs B3-RF [95% CI] | paired t (df), Holm p | corrected t (Nadeau-Bengio), Holm p | Wilcoxon p | seeds better / worse (DeLong-Holm) |",
               "|---|---|---|---|---|---|---|"]
        for m in sorted(S, key=lambda k: -S[k]["AUROC_mean"]):
            e = S[m]
            md.append(f"| {m} | {fmt(e['AUROC_mean'])} ± {fmt(e['AUROC_sd'])} | " + ("reference | | | | |" if m == "B3-RF" else
                      f"{fmt(e.get('diff'))} [{fmt(e.get('lo'))}, {fmt(e.get('hi'))}] | {fmt(e.get('t'), 2)} ({e.get('df')}), {fp(e.get('p_t_holm'))} | "
                      f"{fmt(e.get('t_nb'), 2)}, {fp(e.get('p_nb_holm'))} | {fp(e.get('p_wilcoxon'))} | {e.get('seeds_better')} / {e.get('seeds_worse')} |"))
        if any("AUROC_excl_F5_mean" in e for e in S.values()):
            md += ["", "Sensitivity without the scenarios that context/route identify by design (corrected t vs B3-RF, Holm):", "",
                   "| Method | all | without F5 | Δ, Holm p | without F4+F5 | Δ, Holm p |", "|---|---|---|---|---|---|"]
            for m in sorted(S, key=lambda k: -S[k].get("AUROC_excl_F4F5_mean", 0)):
                e = S[m]
                if "AUROC_excl_F5_mean" in e:
                    md.append(f"| {m} | {fmt(e['AUROC_mean'])} | {fmt(e['AUROC_excl_F5_mean'])} | " +
                              ("reference |" if m == "B3-RF" else f"{fmt(e.get('AUROC_excl_F5_diff'))}, {fp(e.get('AUROC_excl_F5_p_nb_holm'))} |") +
                              f" {fmt(e.get('AUROC_excl_F4F5_mean'))} | " +
                              ("reference |" if m == "B3-RF" else f"{fmt(e.get('AUROC_excl_F4F5_diff'))}, {fp(e.get('AUROC_excl_F4F5_p_nb_holm'))} |"))
        if kind == "lotbench" and "per_scen" in S.get("B3-RF", {}):
            md += ["", "AUROC per scenario (mean over seeds):", "", "| Method | " + " | ".join(J.SCEN) + " |", "|---|" + "---|" * len(J.SCEN)]
            for m in sorted(S, key=lambda k: -S[k]["AUROC_mean"]):
                if "per_scen" in S[m]:
                    md.append(f"| {m} | " + " | ".join(fmt(S[m]["per_scen"][k]) for k in J.SCEN) + " |")
        win = [m for m in S if m != "B3-RF" and m not in EXCLUDE_FROM_HOLM and S[m].get("p_nb_holm", 1) < .05 and S[m].get("diff", 0) > 0]
        md += ["", "**Decision (corrected resampled t test, Holm):** " + (f"better than B3-RF: {', '.join(win)}." if win else "no detector is better than B3-RF."),
               "CTX = context-only model (shortcut check, not a candidate; excluded from Holm).", ""]
    A = results.get("_ablation")
    if A:
        for seed, rows in A.items():
            md += [f"## SALE ablations (seed {seed}; DeLong vs full model, Holm)", "", "| Variant | AUROC | Δ vs full | z | Holm p |", "|---|---|---|---|---|"]
            for nm, r in rows.items():
                md.append(f"| {nm} | {fmt(r['AUROC'])} | {fmt(r['delta'])} | {fmt(r['z'], 2)} | {fp(r.get('p_holm'))} |")
            md.append("")
    open(os.path.join(outdir, "REPORT.md"), "w", encoding="utf-8").write("\n".join(md))
    log(f"Report: {os.path.join(outdir, 'REPORT.md')}")


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description="Extended study: tool-invariant detectors, MCE, ten-seed benchmarks, ablations")
    ap.add_argument("--folder", default=J.PARTNER_DATA_DIR); ap.add_argument("--label", default=J.LABEL_FINAL)
    ap.add_argument("--out", default=J._setting("OUT_EXTENDED", OUT_DEFAULT)); ap.add_argument("--parts", default="plantb,diesaw,faba", help="plantb,diesaw,faba,lotbench,wmlot,ablation")
    ap.add_argument("--repeats", type=int, default=CFG["cv"]["repeats"])
    ap.add_argument("--extra-unlabeled", default=None, help="(optional) .npy with structure features of unlabelled lots for SSL-RF")
    ap.add_argument("--quick-test", action="store_true")
    a = ap.parse_args()
    out = a.out + ("_DEMO" if a.quick_test else ""); os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    if a.quick_test:
        CFG["TI-NN"]["epochs"] = 30; CFG["TI-NN"]["ensemble"] = 1; CFG["PU-RF"]["n_bags"] = 5; CFG["bootstrap"] = 200; a.repeats = 1
        CFG["MCE"]["inner_folds"] = 3; CFG["SSL-RF"]["max_iter"] = 50
    # pre-registration: configuration + script fingerprint BEFORE any data are read. One file per set of parts;
    # it is written ONCE and never overwritten (a restart with the same scripts keeps the original time stamp).
    h = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    hj = hashlib.sha256(open(os.path.join(HERE, "core_experiments.py"), "rb").read()).hexdigest()
    pre = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "sha256_script": h, "sha256_core": hj, "configuration": CFG,
           "parts": a.parts, "repeats": a.repeats, "quick_test": a.quick_test}
    tag = "_".join(sorted(p.strip() for p in a.parts.split(",") if p.strip())) if not a.quick_test else "quicktest"
    pp = os.path.join(out, f"PREREGISTRATION_{tag}.json")
    if os.path.exists(pp) and not a.quick_test:
        old = json.load(open(pp, encoding="utf-8"))
        if old.get("sha256_script") != h or old.get("sha256_core") != hj:
            sys.exit(f"STOP: {pp} was written by different scripts (SHA-256 differs). Results of different code versions must "
                     f"not be mixed: set a NEW output folder (OUT_EXTENDED in local_config.py) and run again.")
        log(f"pre-registration {pp} exists (same scripts, {old.get('time')}); kept unchanged")
    else:
        J.jdump(pre, pp)
    log(f"extended_experiments.py {CFG['version']}; torch: {J.HAS_TORCH}; pre-registration: {pp}")
    if not J.HAS_TORCH and not a.quick_test:
        sys.exit("STOP: PyTorch is not installed; SALE, B6-B9 and TI-NN would be missing. Install PyTorch (see CARA_RUN.md).")
    extra = np.load(a.extra_unlabeled) if a.extra_unlabeled else None
    results = {}
    parts = [p.strip() for p in a.parts.split(",")]
    names = ["demo"] if a.quick_test else [p for p in parts if p in ("plantb", "diesaw", "faba")]
    if a.quick_test:
        parts = ["lotbench"]
    for kind in ("lotbench", "wmlot"):
        if kind in parts:
            R = run_benchmark(kind, a, out)
            if R:
                results["_" + kind] = R
    if "ablation" in parts:
        results["_ablation"] = run_ablation(a, out)
    for nm in names:
        d = demo_data() if nm == "demo" else load_prod(nm, a)
        if d is None:
            log(f"[{nm}] data not found, skipped"); continue
        log(f"[{nm}] {d.notes}")
        reps = cv_protocol(d, a.repeats, extra=extra)
        S = summarize(d, reps)
        hw = [i["HYB_weights"] for rp in reps for i in rp["info"] if "HYB_weights" in i]
        mw = [i["MCE_weights"] for rp in reps for i in rp["info"] if "MCE_weights" in i]
        skip_t = nm in CFG["time_protocols_skip"]
        note_t = {"note": "skipped: lot dates of this line are export dates, not processing dates (frozen in CFG)"}
        R = {"notes": d.notes, "cv": S, "loto": loto_protocol(d), "time": note_t if skip_t else time_protocol(d),
             "month": note_t if skip_t else month_protocol(d),
             "mce_w": {k: round(float(np.mean([w[k] for w in mw])), 3) for k in mw[0]} if mw else None,
             "hyb_w": {k: round(float(np.mean([w[k] for w in hw])), 3) for k in hw[0]} if hw else None,
             "per_lot_rep1": {"idx": sorted(reps[0]["scores"]["B3-RF"]), "y": d.ybin.tolist(), "tool": d.lots["tool"].astype(str).tolist(),
                              "month": d.lots["month"].astype(str).tolist() if "month" in d.lots else None,
                              "scores": reps[0]["scores"]}}
        J.jdump(R, os.path.join(out, "raw", f"{nm}.json")); results[nm] = R
        for m in ["B3-RF"] + [x for x in NEW if x in S]:
            log(f"   {m}: AUROC {S[m]['AUROC']:.3f}, within-tool {S[m]['withinAUROC']:.3f}" +
                (f", within-tool Δ {S[m].get('within_diff', np.nan):+.3f} (Holm p {fp(S[m].get('within_p_holm'))})" if m != "B3-RF" else ""))
    report(out, results)
    open(os.path.join(out, "log_extended.txt"), "w", encoding="utf-8").write("\n".join(LOG))


if __name__ == "__main__":
    main()
