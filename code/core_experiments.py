#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
core_experiments.py - core experiments of the paper "From wafers to lots": LotBench-811K generator, WM-811K lot task,
production protocols, all detectors (B1-B10, SALE, SALE-S), shortcut tests, calibration, attribution and the core report.

Contents:
  A. LotBench-811K : benchmark generator built on WM-811K (LSWMD.pkl) with known causes (F1-F6)
  B. Production    : Plant B coat/develop, Plant B wafer saw and Fab A PECVD through production_loaders.py
                     (confidential; the public release contains a stub that explains the expected interface)
  C. Models        : SALE (structure-aware lot encoder: chamber-relative normalization + peer contrast + hierarchical
                     MIL + slot Transformer + route + context + gated fusion; three pretraining losses; focal loss;
                     conformal threshold; occlusion / integrated gradients), SALE-S and baselines B1-B10, ablations,
                     shortcut/leakage tests, calibration, attribution, retrieval
  D. Report        : <out>/table_*.csv, fig_*.png, hypotheses.json, REPORT_core.md

Inside the code SALE is called "Ours" and SALE-S "Ours-S".

Changes v17 over v16:
  1. Conformal score: standardization from normal training lots ('fit'), not from calibration lots -> valid threshold
  2. Test "tool aliases permuted" (always 0 by design) replaced by the within-tool AUROC for every method
  3. Test "deliberate leak": 2 % of labels flipped (formerly 10 %, too weak)
  4. Plant B same coaters = main Plant B result (DeLong and H2 use this subset)
  5. Attribution example: reasonably large lot with mixed outcomes; gate figure from LotBench when available
  6. Chamber hit@1 of SALE: occlusion keyed by the GLOBAL chamber id (formerly the local index)
  7. Route branch receives the TOOL of each step (formerly only the step number, so upstream tools were invisible)
  8. Step hit@1: integrated gradients over the whole step token (features + tool embedding)
  9. B10 tool commonality produces a lot score and is evaluated
 10. Injection strengths eta set in SETTINGS (ETA_LEVELS) or --eta; stored in lotbench.json
 11. Within-chamber slot shuffling (order loss) vectorized on the GPU (faster, statistically equivalent)
 12. (v18) FAR averaged over all CV repetitions; per-fold FAR stored
 13. (v18) New method SALE-S (Ours-S): explicit structure features (chamber contrast, slot trend, early post-PM wafers,
     outlying wafers, peer contrast, maps) standardized PER TOOL against normal lots + RF (OOB) and one-class fusion.
     Design frozen before it was run; no tuning on test data. SALE (neural) is still reported.
 14. (v18) New WM-811K lot task with REAL defects (--parts wmlot): abnormal lot = >= 3 wafers with a defect pattern
     according to the original labels; definition fixed in SETTINGS before it was run
 15. (v19) English release; paths and partner-specific settings in local_config.py; production loaders separated
     into production_loaders.py (confidential; a stub with the same interface is public)
 16. (v19) Within-tool AUROC follows Eq. 14 (tools with >= 5 normal and >= 2 abnormal lots); the earlier version
     (every tool with both classes) is kept as 'within_all_mixed'; the drop is tested by a stratified bootstrap and
     is computed against the pooled AUROC of the same repetition
 17. (v19) Statistics reported in the paper are now computed in code: DeLong confidence intervals (logit scale) and
     Holm-adjusted paired DeLong tests for every detector (Appendix Table A2), Hanley-McNeil intervals and tests
     against 0.5 for shortcut tests and leave-one-tool-out, minimal detectable AUROC difference, McNemar tests for
     chamber hit@1 and retrieval, a lot-level bootstrap for wafer attribution
 18. (v19) Ablation names follow the equation numbers of the paper ((7)-(8) hierarchy, (4)-(5) map encoder)
 19. (v20) Fab A is called "faba" everywhere (part name, JSON file, loader function)
 20. (v20) Route target encoding is leave-one-out for training lots (a training lot no longer sees its own label)
 21. (v20) Reproducibility: python/numpy/torch/CUDA seeds, cuDNN deterministic, deterministic algorithms (warn only)
 22. (v20) Shortcut test "normal lots matched by tool and month": difference against the pooled AUROC of the SAME
     repetition (formerly against the mean over repetitions)
 23. (v20) Label permutation test additionally WITHIN tool ("null_B3L1_within_tool"), which keeps tool confounding
 24. (v20) Production CV: conformal p-value of every test lot against the calibration lots of its own fold; the
     AUROC of the pooled p-values ("AUROC_pval") is a sensitivity check that does not mix score scales across folds

Install (once; an NVIDIA GPU is recommended):
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
  python -m pip install numpy pandas scipy scikit-learn matplotlib openpyxl xlrd
Data:
  WM-811K: download LSWMD.pkl (Kaggle: qingyi/wm811k-wafer-map)
Run:
  python core_experiments.py --quick-test                  # checks that all code runs (DEMO results, not for the paper)
  python core_experiments.py --parts lotbench              # LotBench (seeds from --seeds, eta from ETA_LEVELS)
  python core_experiments.py --parts wmlot                 # WM-811K lot task with real defects
  python core_experiments.py --parts plantb,diesaw,faba    # production lines (needs production_loaders.py)
  python core_experiments.py --parts report                # tables, figures, REPORT_core.md from raw/*.json
"""
# ======================= SETTINGS (override in local_config.py, which is never uploaded) =======================
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
try:
    import local_config as _LC
except ImportError:
    _LC = None


def _setting(name, default):
    return getattr(_LC, name, default) if _LC is not None else default


VERSION = "v20"
PARTNER_DATA_DIR = _setting("PARTNER_DATA_DIR", "")        # confidential partner exports (production lines)
LABEL_FINAL = _setting("PLANTB_LABEL_CSV", "")             # Plant B lot labels (lot_id,status), confidential
PUBLIC_DATA_DIR = _setting("PUBLIC_DATA_DIR", "data")      # public data folder (searched automatically)
WM811K_PKL = _setting("WM811K_PKL", "")                    # empty = search LSWMD.pkl in PUBLIC_DATA_DIR
MIXEDWM38_NPZ = _setting("MIXEDWM38_NPZ", "")              # empty = search MixedWM38.npz / Wafer_Map_Datasets.npz
OUT = _setting("OUT_CORE", "results_core")
LOADER_DIR = _setting("LOADER_DIR", "")                    # folder of production_loaders.py (default: next to this file)
# context that MAY enter production models. Month/recipe on Plant B and month on Fab A proved to be shortcuts
# (context-only AUROC 0.97 / 0.79), so the default is no context. Fill e.g. {"diesaw": ["device", "blade"]} for a sensitivity test.
CTX_PROD = {"plantb": [], "diesaw": [], "faba": []}
# chamber embedding: False = chamber index WITHIN the lot (no tool identity); True = global id tool|chamber
CH_GLOBAL = False
# LotBench injection strengths (eta). v16 used 0.3-0.9 and all methods saturated; v17: sweep from very hard
ETA_LEVELS = [0.05, 0.1, 0.2, 0.3]
# WM-811K lot task with REAL defects (wmlot) - DEFINITION FIXED BEFORE IT WAS RUN (do not change after seeing results):
#   lot   = WM-811K lot with >= WMLOT_MIN_W wafers (slots 1-25) whose wafers ALL carry a failureType label
#   label = lot abnormal if >= WMLOT_K wafers carry a defect pattern (failureType != 'none'); no injection at all
#   input = wafer maps (polar map + 17 map features per wafer); no FDC, chamber, route or context
#   split = per lot 70 / 15 / 15 (train / calibration / test) for each seed; all methods; run once
#   wafer attribution is measured against the (original) per-wafer pattern labels
WMLOT_K = 3
WMLOT_MIN_W = 20
# =========================================================================

import argparse, glob, json, math, os, random, sys, time, warnings, importlib.util
from collections import defaultdict
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # needed for deterministic cuBLAS (must be set before torch)
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, IsolationForest, HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, f1_score
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.covariance import LedoitWolf
except ImportError:
    sys.exit("scikit-learn is required: python -m pip install scikit-learn")
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

S_MAX = 25        # slot per lot
DEMO = False      # True with --demo / --quick-test: results are stamped DEMO and must not be used in the paper
ETAS = list(ETA_LEVELS)
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def seed_all(s):
    """Seeds of python, numpy and torch (CPU and every GPU); cuDNN in deterministic mode. Some GPU operations have no
    deterministic version: torch then only warns (warn_only=True) instead of stopping the run."""
    random.seed(s); np.random.seed(s)
    if HAS_TORCH:
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# ============================================================ statistics
def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def prauc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, s))


def recall_at_fpr(y, s, fpr=0.05):
    y, s = np.asarray(y), np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return np.nan
    f, t, _ = roc_curve(y, s)
    ok = f <= fpr
    return float(t[ok].max()) if ok.any() else 0.0


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong_paired(y, s1, s2):
    """Paired DeLong test (DeLong et al., 1988). Returns: auc1, auc2, z, p."""
    y = np.asarray(y).astype(int); s1 = np.asarray(s1, float); s2 = np.asarray(s2, float)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    if m < 2 or n < 2:
        return np.nan, np.nan, np.nan, np.nan
    V10, V01, A = [], [], []
    for s in (s1, s2):
        X, Y = s[pos], s[neg]
        tx, ty, tz = _midrank(X), _midrank(Y), _midrank(np.concatenate([X, Y]))
        a = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        V10.append((tz[:m] - tx) / n); V01.append(1 - (tz[m:] - ty) / m); A.append(a)
    S10, S01 = np.cov(np.vstack(V10)), np.cov(np.vstack(V01))
    S = S10 / m + S01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return A[0], A[1], 0.0, 1.0
    z = (A[0] - A[1]) / np.sqrt(var)
    return float(A[0]), float(A[1]), float(z), float(2 * stats.norm.sf(abs(z)))


def holm(ps):
    ps = np.asarray(ps, float); n = len(ps); o = np.argsort(ps); adj = np.empty(n); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (n - k) * ps[i])); adj[i] = run
    return adj


def boot_ci(y, s, fn=auroc, B=2000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); s = np.asarray(s, float); v = []
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return [np.nan, np.nan]
    for _ in range(B):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        v.append(fn(y[i], s[i]))
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def conformal_tau(cal_scores, eps):
    s = np.sort(np.asarray(cal_scores, float)); n = len(s)
    k = int(math.ceil((n + 1) * (1 - eps)))
    return float(s[min(k, n) - 1]) if n else np.nan


def conformal_pvalues(cal_scores, test_scores):
    """Split-conformal p-value of each test score against the normal calibration scores of the same fold:
    p = (1 + #{cal >= s}) / (n + 1). Small p = abnormal. Pooling p-values (instead of raw scores) across folds or months
    puts every fold on the same scale."""
    c = np.sort(np.asarray(cal_scores, float)); n = len(c); t = np.asarray(test_scores, float)
    if not n:
        return np.full(len(t), np.nan)
    ge = n - np.searchsorted(c, t, side="left")
    return (1.0 + ge) / (n + 1.0)


def _logit_ci(a, se, alpha=0.05):
    a_ = float(np.clip(a, 1e-6, 1 - 1e-6)); z = stats.norm.ppf(1 - alpha / 2)
    lg = math.log(a_ / (1 - a_)); sl = se / (a_ * (1 - a_))
    return float(1 / (1 + math.exp(-(lg - z * sl)))), float(1 / (1 + math.exp(-(lg + z * sl))))


def auroc_ci_delong(y, s, alpha=0.05):
    """AUROC with a DeLong confidence interval on the logit scale (Appendix Table A2). Returns auc, lo, hi."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float); ok = ~np.isnan(s); y, s = y[ok], s[ok]
    pos, neg = y == 1, y == 0; m, n = int(pos.sum()), int(neg.sum())
    if m < 2 or n < 2:
        return np.nan, np.nan, np.nan
    X, Y = s[pos], s[neg]
    tx, ty, tz = _midrank(X), _midrank(Y), _midrank(np.concatenate([X, Y]))
    a = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    var = np.var((tz[:m] - tx) / n, ddof=1) / m + np.var(1 - (tz[m:] - ty) / m, ddof=1) / n
    return (float(a),) + _logit_ci(a, math.sqrt(max(var, 1e-12)), alpha)


def hanley_mcneil_se(a, n1, n0):
    """Standard error of an AUROC (Hanley & McNeil, 1982); n1 abnormal, n0 normal."""
    q1 = a / (2 - a); q2 = 2 * a * a / (1 + a)
    return math.sqrt(max(a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a), 0) / (n1 * n0))


def auroc_hanley(a, n1, n0, alpha=0.05):
    """AUROC with a Hanley-McNeil interval on the logit scale and a two-sided test against 0.5 (SE under H0)."""
    if a is None or not np.isfinite(a) or n1 < 1 or n0 < 1:
        return {"AUROC": np.nan, "lo": np.nan, "hi": np.nan, "p_vs_0.5": np.nan, "n_abnormal": int(n1), "n_normal": int(n0)}
    lo, hi = _logit_ci(a, hanley_mcneil_se(a, n1, n0), alpha)
    se0 = math.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
    return {"AUROC": float(a), "lo": lo, "hi": hi, "p_vs_0.5": float(2 * stats.norm.sf(abs(a - 0.5) / se0)),
            "n_abnormal": int(n1), "n_normal": int(n0)}


def min_detectable_difference(n1, n0, auc=0.85, r=0.5, alpha=0.05, power=0.8):
    """Smallest paired AUROC difference detectable with the given power (Hanley & McNeil SE at AUROC 'auc',
    correlation 'r' between the two detectors' AUROC estimates). Defaults reproduce the values in the paper."""
    se = hanley_mcneil_se(auc, n1, n0)
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se * math.sqrt(2 * (1 - r)))


def within_tool_auroc(y, s, tools, min_normal=5, min_abnormal=2):
    """Within-tool AUROC (Eq. 14): AUROC within each tool with >= min_normal normal and >= min_abnormal abnormal lots,
    averaged with weights = number of abnormal lots. Returns (value, number of abnormal lots used)."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float); tools = np.asarray(tools).astype(str); num = den = 0.0
    for t in np.unique(tools):
        k = tools == t; na = int(y[k].sum()); nn = int(k.sum()) - na
        if na >= min_abnormal and nn >= min_normal:
            num += auroc(y[k], s[k]) * na; den += na
    return (num / den if den else np.nan), int(den)


def within_drop_bootstrap(y, s, tools, B=2000, seed=0, **kw):
    """Drop from the pooled to the within-tool AUROC with a stratified bootstrap (lots resampled within tool and
    class). Returns drop, 95% interval and a two-sided p-value."""
    rng = np.random.default_rng(seed); y = np.asarray(y).astype(int); s = np.asarray(s, float); tools = np.asarray(tools).astype(str)
    groups = [np.where((tools == t) & (y == c))[0] for t in np.unique(tools) for c in (0, 1)]
    groups = [g for g in groups if len(g)]
    obs = auroc(y, s) - within_tool_auroc(y, s, tools, **kw)[0]; d = []
    for _ in range(B):
        i = np.concatenate([rng.choice(g, len(g)) for g in groups])
        d.append(auroc(y[i], s[i]) - within_tool_auroc(y[i], s[i], tools[i], **kw)[0])
    d = np.array(d, float); d = d[np.isfinite(d)]
    if not len(d) or not np.isfinite(obs):
        return {"drop": float(obs), "drop_lo": np.nan, "drop_hi": np.nan, "drop_p": np.nan}
    p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / len(d))
    return {"drop": float(obs), "drop_lo": float(np.percentile(d, 2.5)), "drop_hi": float(np.percentile(d, 97.5)), "drop_p": float(min(1.0, p))}


def mcnemar_exact(a, b):
    """Exact McNemar test for two paired binary outcomes (e.g. hit@1 of two methods on the same lots)."""
    a = np.asarray(a, bool); b = np.asarray(b, bool); n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
    p = float(stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
    return {"n": int(len(a)), "rate_a": float(a.mean()) if len(a) else np.nan, "rate_b": float(b.mean()) if len(b) else np.nan,
            "only_a": n01, "only_b": n10, "p": p}


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{d}f}"


# ============================================================ data structure
class LotData:
    """One collection of lots. lots: 1 row per lot; W: 1 row per wafer (sorted by li, slot).
    lots columns: lot (alias), y (0 = normal, 1..D = type), tool, time, month, ctx..., subset_same, split, ...
    W columns   : li (lot index), slot (1..25), chamber (global id), pm, features..., key_<group>_<level>, wbad, map_i
    route       : dict rt_tool [L,K] (step number), rt_id [L,K] (tool per step), rt_feat [L,K,F], rt_mask [L,K]"""

    def __init__(self, name, lots, W, groups, keys, ctx_cols, types, maps=None, route=None, mapfeat=None, notes=None):
        self.name, self.lots, self.groups, self.keys, self.ctx_cols, self.types = name, lots.reset_index(drop=True), groups, keys, ctx_cols, types
        W = W.copy(); W["slot"] = pd.to_numeric(W["slot"], errors="coerce")
        n0 = len(W)
        W = W[(W["slot"] >= 1) & (W["slot"] <= S_MAX)].copy(); W["slot"] = W["slot"].astype(int)
        W = W.drop_duplicates(["li", "slot"])
        if len(W) < n0:
            print(f"   [{name}] {n0 - len(W)} wafer rows dropped (slot outside 1-{S_MAX} or duplicate slot within a lot)")
        W = W.sort_values(["li", "slot"]).reset_index(drop=True)
        self.W, self.maps, self.route, self.mapfeat, self.notes = W, maps, route, mapfeat, dict(notes or {})
        if len(W) < n0:
            self.notes["wafers_dropped_slot"] = int(n0 - len(W))
        self.feats = [f for g in groups.values() for f in g]
        self.off = np.zeros(len(self.lots) + 1, int)
        cnt = W.groupby("li").size().reindex(range(len(self.lots)), fill_value=0).values
        self.off[1:] = np.cumsum(cnt)
        ch = W["chamber"].astype(str)
        self.ch_vocab = {c: i + 1 for i, c in enumerate(sorted(ch.unique()))}
        self.W["ch_id"] = ch.map(self.ch_vocab).astype(int)
        self.W["ch_loc"] = self.W.groupby("li")["ch_id"].transform(lambda s: pd.factorize(s)[0])
        self.ctx_vocab = {c: {v: i + 1 for i, v in enumerate(sorted(self.lots[c].astype(str).unique()))} for c in ctx_cols}
        self.ctx_all = list(ctx_cols)

    @property
    def y(self):
        return self.lots["y"].values.astype(int)

    @property
    def ybin(self):
        return (self.lots["y"].values > 0).astype(int)

    def rows(self, i):
        return slice(self.off[i], self.off[i + 1])


def _production_loaders():
    """Import production_loaders.py. The real module is confidential; the public release ships a stub with the same
    interface (see production_loaders_stub.py)."""
    for d in (LOADER_DIR, os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        p = os.path.join(d, "production_loaders.py") if d else ""
        if p and os.path.exists(p):
            spec = importlib.util.spec_from_file_location("production_loaders", p)
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            return m
    sys.exit("production_loaders.py not found. Put it next to core_experiments.py or set LOADER_DIR in local_config.py.")


def _as_lotdata(d):
    return None if d is None else LotData(**d)


def load_plantb(folder, label_csv, **kw):
    """Plant B coat/develop (photoresist bubble); see production_loaders.py."""
    return _as_lotdata(_production_loaders().load_plantb(folder, label_csv, **kw))


def load_diesaw(folder, zip_path=None, k_bad=3, pct=95):
    """Plant B wafer saw (chipping); see production_loaders.py."""
    return _as_lotdata(_production_loaders().load_diesaw(folder, zip_path, k_bad=k_bad, pct=pct))


def load_faba(folder, zthr=2.5):
    """Fab A PECVD (metrology excursion); see production_loaders.py."""
    return _as_lotdata(_production_loaders().load_faba(folder, zthr=zthr))


# ============================================================ WM-811K and LotBench-811K
NR, NT = 16, 32          # polar map: 16 rings x 32 angles
_POLAR_CACHE = {}


def polar_index(h, w):
    k = (h, w)
    if k not in _POLAR_CACHE:
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = (h - 1) / 2, (w - 1) / 2
        r = np.sqrt(((yy - cy) / (h / 2)) ** 2 + ((xx - cx) / (w / 2)) ** 2)
        th = (np.arctan2(yy - cy, xx - cx) + np.pi) / (2 * np.pi)
        ri = np.clip((r * NR).astype(int), 0, NR - 1); ti = np.clip((th * NT).astype(int), 0, NT - 1)
        _POLAR_CACHE[k] = (ri * NT + ti).ravel()
    return _POLAR_CACHE[k]


def to_polar(m):
    """WM-811K map (0 = no die, 1 = pass, 2 = fail) -> failing fraction per polar cell [16, 32]."""
    m = np.asarray(m)
    idx = polar_index(*m.shape)
    die = (m.ravel() > 0).astype(float); fail = (m.ravel() == 2).astype(float)
    nd = np.bincount(idx, die, NR * NT); nf = np.bincount(idx, fail, NR * NT)
    return (nf / np.maximum(nd, 1)).reshape(NR, NT).astype(np.float32)


def load_wm811k(path):
    import pandas as _pd
    sys.modules.setdefault("pandas.indexes", _pd.core.indexes)
    df = pd.read_pickle(path)
    def ft(v):
        try:
            v = np.asarray(v).ravel()
            return str(v[0]) if len(v) else None
        except Exception:
            return None
    out = pd.DataFrame({"lot": df["lotName"].astype(str), "slot": pd.to_numeric(df["waferIndex"], errors="coerce").astype(int),
                        "die": pd.to_numeric(df["dieSize"], errors="coerce"), "ftype": df["failureType"].map(ft)})
    out["ftype"] = out["ftype"].where(out["ftype"].isin(["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]))
    return out, df["waferMap"].values


def find_public(folder, names):
    """Return the first file (recursive search) that has one of the given names."""
    if not folder or not os.path.isdir(folder):
        return None
    for n in names:
        c = sorted(glob.glob(os.path.join(folder, "**", n), recursive=True), key=len)
        if c:
            return c[0]
    return None


def load_mixedwm38(path, max_n=40000, seed=0):
    """MixedWM38 (Wang et al., 2020): arr_0 = 52x52 maps (0 no die, 1 pass, 2 fail), arr_1 = one-hot labels of 8 basic patterns.
    Returns: polar maps [N,16,32] float16 and multi-label targets [N,8]."""
    z = np.load(path, allow_pickle=True)
    keys = list(z.keys())
    X = z["arr_0"] if "arr_0" in keys else z[keys[0]]
    Y = z["arr_1"] if "arr_1" in keys else z[keys[1]]
    idx = np.random.default_rng(seed).permutation(len(X))[:max_n]
    P = np.stack([to_polar(X[i]) for i in idx]).astype(np.float16)
    return P, np.asarray(Y)[idx].astype(np.float32)


def demo_wm811k(n_lots=400, seed=0):
    """FAKE data shaped like WM-811K, only for --quick-test / --demo (never for the paper)."""
    rng = np.random.default_rng(seed); rows, maps = [], []
    kinds = ["none", "Center", "Edge-Ring", "Scratch", "Loc", "Random"]
    for l in range(n_lots):
        h = int(rng.choice([26, 32, 40])); n = int(rng.integers(20, 26))
        for s in range(1, n + 1):
            yy, xx = np.mgrid[0:h, 0:h]; r = np.hypot(yy - h / 2 + .5, xx - h / 2 + .5) / (h / 2)
            m = np.where(r <= 1, 1, 0); k = rng.choice(kinds, p=[.7, .06, .06, .06, .06, .06])
            fail = rng.random(m.shape) < .03
            if k == "Center": fail |= r < .3
            if k == "Edge-Ring": fail |= (r > .85) & (r <= 1)
            if k == "Scratch": fail |= np.abs(yy - xx) < 1
            if k == "Loc": fail |= np.hypot(yy - h * .3, xx - h * .6) < h * .15
            if k == "Random": fail |= rng.random(m.shape) < .3
            m = np.where((m == 1) & fail, 2, m)
            rows.append({"lot": f"lot{l}", "slot": s, "die": h * h, "ftype": k if rng.random() < .3 else None}); maps.append(m)
    return pd.DataFrame(rows), np.array(maps, dtype=object)


def map_feats(P):
    """Hand-crafted map features for the baselines: ring profile (16) + failing fraction (1)."""
    P = np.asarray(P, np.float32)
    return np.concatenate([P.mean(-1), P.mean((-1, -2))[..., None]], -1)


SCEN = ["F1", "F2", "F3", "F4", "F5", "F6"]
PATTERN = {"F1": 1, "F2": 3, "F3": 2, "F4": 0, "F5": 0, "F6": 3}     # 0 whole lot, 1 chamber alternation, 2 trend, 3 local
PATTERN_NAMES = ["whole lot", "chamber alternation", "slot trend", "local excursion"]


def make_lotbench(meta, maps, seed, n_lots=20000, frac=0.08, J=20, K=8, polar_cache=None):
    """LotBench-811K (Table 4): host = WM-811K lot with >= 20 wafers; simulated process context; F1-F6 injected."""
    rng = np.random.default_rng(seed)
    cnt = meta.groupby("lot").size()
    hosts = cnt[cnt >= 20].index.values
    hosts = rng.permutation(hosts)[:min(n_lots, len(hosts))]
    split = np.array(["train"] * len(hosts), dtype=object)
    n = len(hosts); split[int(.70 * n):int(.85 * n)] = "cal"; split[int(.85 * n):] = "test"
    dsz = meta.groupby("lot")["die"].median()
    qs = np.quantile(dsz.loc[hosts], [.2, .4, .6, .8])
    mu = rng.normal(0, 1, (4, 3, J)); sig = rng.uniform(.5, 1.5, J); a_s = rng.normal(0, 1, J); d_r = rng.normal(0, .5, J)
    t_win = 150.0; k_star = 3; b_star = 0
    gidx = meta.groupby("lot").indices
    # defect templates per split and die-size group
    lab = meta[meta["ftype"].notna() & (meta["ftype"] != "none")]
    host_split = dict(zip(hosts, split))
    lab = lab[lab["lot"].isin(host_split)]
    lab_split = {s: lab[lab["lot"].map(host_split) == s] for s in ("train", "cal", "test")}
    def polar(i):
        if polar_cache is not None:
            return polar_cache[i]
        return to_polar(maps[i])
    abn = rng.random(n) < frac
    scen = np.where(abn, rng.integers(0, 6, n), -1)
    lots, Wrows, P = [], [], []
    rt_tool = np.zeros((n, K), int); rt_id = np.zeros((n, K), int); rt_feat = np.zeros((n, K, 3), np.float32)
    for li, h in enumerate(hosts):
        ii = gidx[h]; sub = meta.iloc[ii].sort_values("slot")
        ii = sub.index.values; slots = sub["slot"].values.clip(1, 25)
        _, uq = np.unique(slots, return_index=True); ii, slots = ii[uq], slots[uq]
        prod = int(np.searchsorted(qs, dsz.loc[h])); tool = int(rng.integers(0, 4)); kk = 2 if tool < 2 else 3
        t = float(rng.uniform(0, 365)); pm_lot = float(rng.uniform(0, 2000))
        recipe = prod * 3 + int(rng.integers(0, 3)); batch = int(rng.integers(0, 40))
        steps = rng.integers(0, 3, K)
        sc = SCEN[scen[li]] if scen[li] >= 0 else "N"; eta = float(rng.choice(ETAS)) if sc != "N" else 0.0
        if sc == "F2":
            pm_lot = float(rng.uniform(0, 200))
        if sc == "F4":
            t = float(rng.uniform(t_win, t_win + 60)); steps[k_star] = 0
        elif t_win <= t <= t_win + 60 and steps[k_star] == 0:
            steps[k_star] = int(rng.integers(1, 3))
        if sc == "F5":
            batch = b_star
        elif batch == b_star:
            batch = int(rng.integers(1, 40))
        order = np.arange(len(ii)); ch = (slots - 1) % kk; pmw = pm_lot + order
        x = mu[tool, ch] + a_s * np.exp(-pmw / 150)[:, None] + d_r * (t / 365) + rng.normal(0, 1, (len(ii), J)) * sig * .3
        Pw = np.stack([polar(i) for i in ii]).astype(np.float32)
        w_out = np.zeros(len(ii)); aff = np.zeros(len(ii), bool); fj = rng.choice(J, 5, replace=False); tch = -1
        if sc == "F1":
            c = int(rng.choice(np.unique(ch))); aff = ch == c; x[np.ix_(aff, fj)] += eta * 3 * sig[fj]; w_out[aff] = eta; tch = tool * 3 + c
        elif sc == "F2":
            na = 1 + int(round(4 * eta)); aff = order < na; dec = np.exp(-order / 2.0)
            x[:, fj] += (eta * 3 * sig[fj])[None, :] * (dec * aff)[:, None]; w_out = eta * dec * aff; tch = tool * 3 + int(ch[0])
        elif sc == "F3":
            r = order / max(1, len(ii) - 1); x[:, fj] += (eta * 3 * sig[fj])[None, :] * r[:, None]; w_out = eta * r; aff = r >= .5
        elif sc in ("F4", "F5"):
            w_out[:] = eta * (.6 if sc == "F4" else .4); aff[:] = True
        elif sc == "F6":
            na = int(rng.integers(1, 4)); pick = rng.choice(len(ii), na, replace=False); aff[pick] = True
            x[np.ix_(aff, fj)] += eta * 4 * sig[fj]; w_out[aff] = eta; tch = tool * 3 + int(ch[pick[0]])
        if sc != "N":
            L_ = lab_split[split[li]]; cand = L_[(L_["die"] <= np.r_[qs, np.inf][prod]) & (L_["die"] > np.r_[-np.inf, qs][prod])]
            cand = cand if len(cand) else L_
            tmpl = polar(int(rng.choice(cand.index.values))) if len(cand) else np.ones((NR, NT), np.float32) * .3
            Pw = np.clip(Pw + w_out[:, None, None] * tmpl[None], 0, 1)
        q = rng.lognormal(0, .5, K); pmk = rng.uniform(0, 2000, K); dur = rng.normal(1, .1, K)
        rt_tool[li] = np.arange(K); rt_id[li] = np.arange(K) * 3 + steps; rt_feat[li] = np.stack([np.log1p(q), pmk / 2000, dur], 1)
        base = len(P)
        P.append(Pw)
        for j in range(len(ii)):
            Wrows.append([li, int(slots[j]), f"{tool}|{int(ch[j])}", float(pmw[j]), float(aff[j]), base + j, str(tool)] + list(x[j]))
        lots.append({"lot": f"H{li:05d}", "y": (SCEN.index(sc) + 1) if sc != "N" else 0, "scenario": sc, "eta": eta,
                     "tool": f"V{tool}", "rot": kk, "product": prod, "recipe": recipe, "batch": batch, "week": int(t // 7),
                     "time": pd.Timestamp("2025-01-01") + pd.Timedelta(days=t), "split": split[li],
                     "pattern": PATTERN.get(sc, -1), "true_ch": f"{tool}|{tch - tool * 3}" if tch >= 0 else "",
                     "true_step": k_star if sc == "F4" else -1, "subset_same": True, "host": h})
    P = np.concatenate(P).astype(np.float16)
    fcols = [f"FDC:{j:02d}" for j in range(J)]
    W = pd.DataFrame(Wrows, columns=["li", "slot", "chamber", "pm", "wbad", "map_i", "tool"] + fcols)
    W["key_FDC_0"] = W["chamber"] + "|" + W["li"].map(lambda i: str(lots[i]["recipe"])); W["key_FDC_1"] = W["chamber"]
    L = pd.DataFrame(lots); L["month"] = L["time"].dt.to_period("M").astype(str)
    route = {"rt_tool": rt_tool, "rt_id": rt_id, "rt_feat": rt_feat, "rt_mask": np.ones((n, K), bool)}
    return LotData("lotbench", L, W, {"FDC": fcols}, {"FDC": ["key_FDC_0", "key_FDC_1"]}, ["product", "recipe", "batch", "week"],
                   SCEN, maps=P, route=route, mapfeat=map_feats(P), notes={"lot": n, "abnormal": int(abn.sum()), "seed": seed, "etas": list(ETAS)})


# ============================================================ per-fold preprocessing
def crn(data, ref_lots, use=True, min_n=20):
    """Chamber-relative normalization (Eq. 2): robust z against wafers of NORMAL lots in the training fold, per key
    tool|chamber|recipe (falling back to tool|chamber and then to global if the reference has < min_n wafers).
    use=False (ablation): one global reference per feature."""
    W = data.W
    ref = W["li"].isin(set(ref_lots)).values
    out = []
    for g, fs in data.groups.items():
        X = W[fs].values.astype(np.float64)
        gmed = np.nanmedian(X[ref], 0); gsd = np.nanstd(X[ref], 0) + 1e-6
        if not use:
            mad = np.nanmedian(np.abs(X[ref] - gmed), 0)
            Z = (X - gmed) / np.where(mad > 0, 1.4826 * mad, gsd)
        else:
            Z = np.full_like(X, np.nan); done = np.zeros(len(W), bool)
            for k in data.keys[g] + [None]:
                kv = np.zeros(len(W), int) if k is None else pd.factorize(W[k].astype(str))[0]
                df = pd.DataFrame(X[ref]); kr = kv[ref]
                med = df.groupby(kr).median()
                cnt = df.groupby(kr).count().min(1)
                dev = (df - med.reindex(kr).values).abs()
                mad = dev.groupby(kr).median()
                okk = cnt.index[cnt >= (min_n if k is not None else 1)]
                rows = (~done) & np.isin(kv, okk)
                if rows.any():
                    m_ = med.reindex(kv[rows]).values; d_ = mad.reindex(kv[rows]).values
                    den = np.where(d_ > 0, 1.4826 * d_, gsd[None, :])
                    Z[rows] = (X[rows] - m_) / den
                    done |= rows
        out.append(np.clip(Z, -20, 20))
    Zall = np.hstack(out).astype(np.float32)
    data.obs = ~np.isnan(Zall)                      # features that were actually measured per wafer
    return np.nan_to_num(Zall, nan=0.0)


def peer_contrast(data, Xn):
    """Eq. 3: x̄ = x̃ - median x̃ of the other wafers of the same lot that ran in other chambers (0 if there are none)."""
    Xp = np.zeros_like(Xn); ch = data.W["ch_id"].values
    for i in range(len(data.lots)):
        r = data.rows(i); X = Xn[r]; c = ch[r]
        for cc in np.unique(c):
            o = c != cc
            if o.any():
                Xp[r.start:r.stop][c == cc] = X[c == cc] - np.median(X[o], 0)
    return Xp


def lot_summary(data, Xn, with_ctx=True, ctx_only=False, extra=None):
    """Lot summary features (B3, B5, B6): mean/std/min/max of x̃, chamber deviation, mean |x̃|, map features, route, context."""
    L = len(data.lots); W = data.W; li = W["li"].values; parts, names = [], []
    if not ctx_only:
        df = pd.DataFrame(Xn); df["li"] = li
        g = df.groupby("li")
        for fn in ("mean", "std", "min", "max"):
            v = getattr(g, fn)().reindex(range(L)).fillna(0).values; parts.append(v); names += [f"{fn}_{j}" for j in range(Xn.shape[1])]
        a = pd.DataFrame({"li": li, "ch": W["ch_id"].values, "a": np.abs(Xn).mean(1)})
        chm = a.groupby(["li", "ch"])["a"].mean()
        parts.append(np.c_[chm.groupby(level=0).max().reindex(range(L)).fillna(0).values,
                           a.groupby("li")["a"].mean().reindex(range(L)).fillna(0).values,
                           a.groupby("li")["a"].max().reindex(range(L)).fillna(0).values,
                           W.groupby("li").size().reindex(range(L)).fillna(0).values])
        names += ["chamber_absz", "mean_absz", "max_absz", "n_wafer"]
        if data.mapfeat is not None:
            mf = pd.DataFrame(data.mapfeat[W["map_i"].values].astype(np.float32)); mf["li"] = li
            for fn in ("mean", "max"):
                parts.append(getattr(mf.groupby("li"), fn)().reindex(range(L)).fillna(0).values); names += [f"map{fn}_{j}" for j in range(17)]
        if data.route is not None:
            R = data.route
            parts.append(R["rt_feat"].reshape(L, -1)); names += [f"rt{k}" for k in range(R["rt_feat"].reshape(L, -1).shape[1])]
            oh = np.zeros((L, int(R["rt_id"].max()) + 1)); oh[np.arange(L)[:, None], R["rt_id"]] = 1
            parts.append(oh); names += [f"tool{k}" for k in range(oh.shape[1])]
    if with_ctx or ctx_only:
        for c in data.ctx_cols:
            d = pd.get_dummies(data.lots[c].astype(str), prefix=c).astype(float)
            parts.append(d.values); names += list(d.columns)
    if extra is not None:
        parts.append(extra); names += [f"extra{j}" for j in range(extra.shape[1])]
    return np.hstack(parts).astype(np.float32), names


def lot_arrays(data, Xn, Xp):
    """Dense arrays [L, 25, ...] per fold (slot = position)."""
    L = len(data.lots); J = Xn.shape[1]; W = data.W
    li = W["li"].values; sl = W["slot"].values.astype(int) - 1
    A = {"x": np.zeros((L, S_MAX, J), np.float32), "xp": np.zeros((L, S_MAX, J), np.float32),
         "m": np.zeros((L, S_MAX), bool), "ch": np.zeros((L, S_MAX), np.int64), "cl": np.zeros((L, S_MAX), np.int64),
         "pm": np.zeros((L, S_MAX), np.float32), "wbad": np.full((L, S_MAX), np.nan, np.float32),
         "mi": np.full((L, S_MAX), -1, np.int64)}
    A["x"][li, sl] = Xn; A["xp"][li, sl] = Xp; A["m"][li, sl] = True
    A["ch"][li, sl] = W["ch_id"].values; A["cl"][li, sl] = W["ch_loc"].values
    A["pm"][li, sl] = np.log1p(np.maximum(W["pm"].values.astype(float), 0)) / 8.0
    A["wbad"][li, sl] = W["wbad"].values.astype(np.float32)
    if "map_i" in W:
        A["mi"][li, sl] = W["map_i"].values
    A["ctx"] = np.stack([data.lots[c].astype(str).map(data.ctx_vocab[c]).fillna(0).astype(np.int64).values for c in data.ctx_cols], 1) \
        if data.ctx_cols else np.zeros((L, 0), np.int64)
    A["y"] = data.y.astype(np.int64); A["pat"] = data.lots["pattern"].values.astype(np.int64) if "pattern" in data.lots else np.full(L, -1, np.int64)
    if data.route is not None:
        for k, v in data.route.items():
            A[k] = v
    if data.mapfeat is not None:
        mf = np.zeros((L, S_MAX, 17), np.float32); mf[li, sl] = data.mapfeat[W["map_i"].values]; A["mf"] = mf
    return A


def target_encoding(data, A, train_idx, prior=10):
    """Route: smoothed abnormal rate per (step, tool) from training lots only (route branch).
    v20: leave-one-out for the training lots themselves (their own label is removed from the count), so the encoder
    sees the same kind of value in training as for calibration and test lots (no in-sample target leakage)."""
    if data.route is None:
        return None
    ids = A["rt_id"]; yb = (A["y"] > 0).astype(float)
    tr = np.asarray(train_idx); p0 = yb[tr].mean()
    num = defaultdict(float); den = defaultdict(float)
    for k in range(ids.shape[1]):
        for t_, y_ in zip(ids[tr, k], yb[tr]):
            num[(k, t_)] += y_; den[(k, t_)] += 1
    te = np.zeros(ids.shape, np.float32)
    own = np.zeros(len(yb), bool); own[tr] = True
    for k in range(ids.shape[1]):
        n_ = np.array([num[(k, t_)] for t_ in ids[:, k]]); d_ = np.array([den[(k, t_)] for t_ in ids[:, k]])
        n_ = n_ - own * yb; d_ = d_ - own                    # leave-one-out for training lots
        te[:, k] = (n_ + prior * p0) / (d_ + prior)
    return te


# ============================================================ model (PyTorch)
DEV = "cpu"
if HAS_TORCH:
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    class GatedAttn(nn.Module):
        """Ilse et al. (2018): a = softmax(w^T [tanh(V h) * sigmoid(U h)])"""
        def __init__(s, d, h=64):
            super().__init__(); s.V = nn.Linear(d, h); s.U = nn.Linear(d, h); s.w = nn.Linear(h, 1)

        def logits(s, H):
            return s.w(torch.tanh(s.V(H)) * torch.sigmoid(s.U(H))).squeeze(-1)

        def forward(s, H, mask):
            a = s.logits(H).masked_fill(~mask, -1e9)
            a = torch.softmax(a, -1) * mask
            a = a / (a.sum(-1, keepdim=True) + 1e-9)
            return (a.unsqueeze(-1) * H).sum(-2), a

    class MapEnc(nn.Module):
        """Eqs. 4-5: CNN on the polar map with circular padding along the angle -> u (shape, rotation invariant), v (position)."""
        def __init__(s, polar=True, n_cls=9):
            super().__init__(); s.polar = polar
            s.c1 = nn.Conv2d(1, 16, 3, 1); s.c2 = nn.Conv2d(16, 32, 3, 2); s.c3 = nn.Conv2d(32, 32, 3, 2)
            s.vl = nn.Linear(6, 16); s.cart = nn.Linear(32 * 4 * 8, 32); s.cls = nn.Linear(32, n_cls)
            s.mcls = nn.Linear(32, 8)          # multi-label head for MixedWM38 (pretraining only)

        def pad(s, x):
            if s.polar:
                return F.pad(F.pad(x, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1))
            return F.pad(x, (1, 1, 1, 1))

        def forward(s, P):
            x = P.unsqueeze(1).float()
            for c in (s.c1, s.c2, s.c3):
                x = F.gelu(c(s.pad(x)))
            rad = x.mean((1, 3))
            if s.polar:
                f = x.mean(2); w = torch.softmax(f.mean(1), -1)
                u = (f * w.unsqueeze(1)).sum(-1)
                th = torch.arange(f.shape[-1], device=P.device, dtype=torch.float32) * (2 * math.pi / f.shape[-1])
                ang = torch.stack([(w * torch.cos(th)).sum(-1), (w * torch.sin(th)).sum(-1)], -1)
            else:
                u = s.cart(x.flatten(1)); ang = torch.zeros(len(P), 2, device=P.device)
            return u, s.vl(torch.cat([ang, rad], -1))

    def tf_layer(d, L=2, h=4):
        return nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, 2 * d, 0.1, batch_first=True, norm_first=True), L)

    class LotEncoder(nn.Module):
        """Structure-aware lot encoder SALE (Method section): wafer branch (tokens + hierarchical MIL + slot Transformer),
    route, context, gated fusion."""
        def __init__(s, cfg):
            super().__init__(); s.c = c = cfg; d, D = c.get("width", (64, 128))     # width tunable (revision 2); default as in the paper
            tin = c["J"] * (2 if c["peer"] else 1) + (48 if c["maps"] else 0)
            s.mask_vec = nn.Parameter(torch.zeros(tin))
            s.tok = nn.Sequential(nn.Linear(tin, d), nn.GELU(), nn.Linear(d, d))
            if c["maps"]:
                s.menc = MapEnc(polar=c["polar"])
            if c["emb"]:
                s.e_slot = nn.Embedding(S_MAX, d); s.e_ch = nn.Embedding(c["n_ch"] + 1, d); s.e_pm = nn.Linear(1, d)
            s.att_w = GatedAttn(d)
            if c["hier"]:
                s.att_c = GatedAttn(2 * d); zw = 2 * d
            else:
                zw = d
            if c["slot_tf"]:
                s.cls_tok = nn.Parameter(torch.zeros(1, 1, d)); s.stf = tf_layer(d); zw += d
            zw += 48 if c["maps"] else 0
            s.Pw = nn.Linear(zw, D); s.rec_w = nn.Linear(d, c["J"])
            nb = 1
            if c["route"]:
                # e_tt = embedding of the TOOL used at each step (rt_id); e_step = embedding of the step position
                s.r_in = nn.Linear(c["Fr"] + 1, d); s.e_tt = nn.Embedding(c["n_tt"] + 1, d); s.e_step = nn.Embedding(c["K"], d)
                s.rtf = tf_layer(d); s.r_att = GatedAttn(d); s.Pr = nn.Linear(d, D); s.rec_r = nn.Linear(d, c["Fr"]); nb += 1
                s.r_mask_vec = nn.Parameter(torch.zeros(c["Fr"] + 1))
            if c["ctx"]:
                s.e_ctx = nn.ModuleList([nn.Embedding(n + 1, 16) for n in c["ctx_sizes"]])
                s.Pc = nn.Sequential(nn.Linear(16 * len(c["ctx_sizes"]), D), nn.GELU(), nn.Linear(D, D)); nb += 1
            s.nb = nb
            s.proj = nn.ModuleList([nn.Linear(D, D) for _ in range(nb)])
            s.gate = nn.Linear(nb * D, nb) if c["fusion"] == "gate" else None
            s.cat = nn.Linear(nb * D, D) if c["fusion"] != "gate" else None
            s.head = nn.Linear(D, c["n_cls"]); s.pat = nn.Linear(D, 4); s.ordh = nn.Linear(D, 1); s.ph = nn.Linear(D, 64)

        def forward(s, b, tokmask=None, rtmask=None, rt_scale=None):
            c = s.c; m = b["m"]; B = m.shape[0]
            parts = [b["x"]] + ([b["xp"]] if c["peer"] else [])
            u = v = None
            if c["maps"]:
                mi = b["maps"]; P = mi.reshape(-1, NR, NT)
                u, v = s.menc(P); u = u.view(B, S_MAX, -1); v = v.view(B, S_MAX, -1)
                parts += [u, v]
            tin = torch.cat(parts, -1)
            if tokmask is not None:
                tin = torch.where(tokmask.unsqueeze(-1), s.mask_vec.expand_as(tin), tin)
            h = s.tok(tin)
            if c["emb"]:
                pos = torch.arange(S_MAX, device=m.device).expand(B, -1)
                h = h + s.e_slot(pos) + s.e_ch(b["ch"]) + s.e_pm(b["pm"].unsqueeze(-1))
            out = {}
            if c["hier"]:
                cl = b["cl"]; Cm = int(cl.max().item()) + 1
                Mc = (cl.unsqueeze(1) == torch.arange(Cm, device=m.device).view(1, -1, 1)) & m.unsqueeze(1)
                a = s.att_w.logits(h).unsqueeze(1).expand(-1, Cm, -1).masked_fill(~Mc, -1e9)
                a = torch.softmax(a, -1) * Mc; a = a / (a.sum(-1, keepdim=True) + 1e-9)
                g = torch.einsum("bcs,bsd->bcd", a, h)
                pres = Mc.any(-1)
                gm = (g * pres.unsqueeze(-1)).sum(1) / pres.sum(1, keepdim=True).clamp(min=1)
                gx = torch.cat([g, g - gm.unsqueeze(1)], -1)
                zw, bc = s.att_c(gx, pres)
                out["a_s"] = (a * bc.unsqueeze(-1)).sum(1); out["b_c"] = bc
            else:
                zw, a = s.att_w(h, m); out["a_s"] = a
            zs = [zw]
            if c["slot_tf"]:
                hh = torch.cat([s.cls_tok.expand(B, -1, -1), h], 1)
                kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=m.device), ~m], 1)
                o = s.stf(hh, src_key_padding_mask=kpm)
                zs.append(o[:, 0]); out["tok_ctx"] = o[:, 1:]
            else:
                out["tok_ctx"] = h
            if c["maps"]:
                mm = m.unsqueeze(-1).float(); den = mm.sum(1).clamp(min=1)
                zs += [(u * mm).sum(1) / den, (v * mm).sum(1) / den]
            br = [s.Pw(torch.cat(zs, -1))]
            if c["route"]:
                rin = torch.cat([b["rt_feat"], b["te"].unsqueeze(-1)], -1)
                if rtmask is not None:
                    rin = torch.where(rtmask.unsqueeze(-1), s.r_mask_vec.expand_as(rin), rin)
                K = rin.shape[1]
                rt = s.r_in(rin) + s.e_tt(b["rt_tool"])
                if rt_scale is not None:                     # scale of the step-token content (integrated gradients)
                    rt = rt * rt_scale
                rt = rt + s.e_step(torch.arange(K, device=m.device).expand(B, -1))
                ro = s.rtf(rt, src_key_padding_mask=~b["rt_mask"])
                zr, _ = s.r_att(ro, b["rt_mask"]); br.append(s.Pr(zr)); out["rt_ctx"] = ro
            if c["ctx"]:
                e = torch.cat([emb(b["ctx"][:, j]) for j, emb in enumerate(s.e_ctx)], -1); br.append(s.Pc(e))
            pz = [p(z) for p, z in zip(s.proj, br)]
            out["logits_b"] = [s.head(p) for p in pz]          # the same head for every branch (deep supervision, attribution per branch)
            if s.gate is not None:
                gl = s.gate(torch.cat(pz, -1))
                if s.training and s.nb > 1 and c.get("bdrop", 0.3) > 0:
                    # branch dropout: every branch must be informative on its own (prevents the gate from collapsing onto one branch)
                    drop = torch.rand(gl.shape, device=gl.device) < c.get("bdrop", 0.3)
                    drop[drop.all(1)] = False
                    gl = gl.masked_fill(drop, -1e4)
                pi = torch.softmax(gl, -1)
                z = (torch.stack(pz, 1) * pi.unsqueeze(-1)).sum(1)
            else:
                pi = torch.full((B, s.nb), 1.0 / s.nb, device=m.device); z = s.cat(torch.cat(pz, -1))
            out.update(z=z, logits=s.head(z), pat=s.pat(z), ordl=s.ordh(z).squeeze(-1), proj=F.normalize(s.ph(z), dim=-1), gates=pi)
            return out

    class SetModel(nn.Module):
        """B7 Deep Sets, B8 Set Transformer, B9 flat attention MIL (no slot/chamber information)."""
        def __init__(s, kind, tin, n_cls, d=64):
            super().__init__(); s.kind = kind
            s.phi = nn.Sequential(nn.Linear(tin, d), nn.GELU(), nn.Linear(d, d))
            if kind == "settf":
                s.enc = tf_layer(d); s.seed = nn.Parameter(torch.randn(1, 1, d) * .1); s.pma = nn.MultiheadAttention(d, 4, batch_first=True)
            if kind == "flatmil":
                s.att = GatedAttn(d)
            s.rho = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_cls))

        def forward(s, b, tokmask=None, **k):
            m = b["m"]; h = s.phi(b["tin"]); out = {}
            if s.kind == "deepsets":
                z = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1); out["a_s"] = None
            elif s.kind == "settf":
                h = s.enc(h, src_key_padding_mask=~m)
                z, w = s.pma(s.seed.expand(len(h), -1, -1), h, h, key_padding_mask=~m); z = z[:, 0]; out["a_s"] = w[:, 0]
            else:
                z, a = s.att(h, m); out["a_s"] = a
            out["logits"] = s.rho(z); out["z"] = z
            return out

    class DeepSVDD(nn.Module):
        def __init__(s, din):
            super().__init__(); s.f = nn.Sequential(nn.Linear(din, 64, bias=False), nn.ReLU(), nn.Linear(64, 32, bias=False))

        def forward(s, x):
            return s.f(x)


def to_dev(A, idx, data, te=None, mode="ours"):
    idx = np.asarray(idx); b = {}
    for k in ("x", "xp", "m", "ch", "cl", "pm", "ctx"):
        b[k] = torch.as_tensor(A[k][idx]).to(DEV)
    for k in ("ch", "cl", "ctx"):
        b[k] = b[k].long()                         # Windows: numpy int = int32, PyTorch needs int64
    if not CH_GLOBAL:
        b["ch"] = (b["cl"] + 1) * b["m"].long()    # chamber embedding = local index (no tool identity)
    if data.maps is not None and mode == "ours":
        mi = A["mi"][idx]; Pm = np.zeros(mi.shape + (NR, NT), np.float32); ok = mi >= 0
        Pm[ok] = data.maps[mi[ok]].astype(np.float32); b["maps"] = torch.as_tensor(Pm).to(DEV)
    if data.route is not None:
        b["rt_feat"] = torch.as_tensor(A["rt_feat"][idx]).to(DEV)
        b["rt_tool"] = torch.as_tensor(A["rt_id"][idx]).long().to(DEV)     # TOOL used at each step
        b["rt_mask"] = torch.as_tensor(A["rt_mask"][idx]).to(DEV)
        b["te"] = torch.as_tensor(te[idx] if te is not None else np.zeros(A["rt_id"][idx].shape, np.float32)).to(DEV)
    if mode == "set":
        parts = [A["x"][idx]] + ([A["mf"][idx]] if "mf" in A else [])
        b["tin"] = torch.as_tensor(np.concatenate(parts, -1)).to(DEV)
    return b


def focal_loss(logits, y, alpha, gamma=2.0, use_focal=True):
    y = y.long(); lp = F.log_softmax(logits, -1); lpy = lp.gather(1, y.view(-1, 1)).squeeze(1); py = lpy.exp()
    w = alpha[y]
    if not use_focal:
        return -(w * lpy).mean()
    return -(w * (1 - py) ** gamma * lpy).mean()


def info_nce(p1, p2, t=0.2):
    lg = p1 @ p2.t() / t; lab = torch.arange(len(p1), device=p1.device)
    return (F.cross_entropy(lg, lab) + F.cross_entropy(lg.t(), lab)) / 2


def shuffle_within_chamber(b):
    """Copy of a batch with slots shuffled WITHIN chambers (chamber composition unchanged; L_ord). Vectorized:
    empty slots form their own group (they do not move); within each chamber the order is shuffled."""
    m = b["m"]; cl = b["cl"]; B, S = m.shape; dev = m.device
    pos = torch.arange(S, device=dev).expand(B, -1)
    grp = torch.where(m, cl, 1000 + pos)
    o_orig = torch.argsort(grp * (S + 1) + pos, dim=1)
    o_rand = torch.argsort(grp.double() * (S + 1) + torch.rand(B, S, device=dev, dtype=torch.float64) * S, dim=1)
    perm = torch.empty_like(o_orig); perm.scatter_(1, o_orig, o_rand)
    bi = torch.arange(B, device=dev).unsqueeze(1)
    nb = dict(b)
    for k in ("x", "xp", "pm", "ch", "maps"):
        if k in b:
            nb[k] = b[k][bi, perm]
    return nb


def view(b, drop=0.2, jit=0.05):
    nb = dict(b); m = b["m"].clone()
    r = torch.rand(m.shape, device=m.device) < (torch.rand(m.shape[0], 1, device=m.device) * drop)
    m2 = m & ~r
    m2[m2.sum(1) == 0] = m[m2.sum(1) == 0]
    nb["m"] = m2; nb["x"] = b["x"] + jit * torch.randn_like(b["x"]); nb["xp"] = b["xp"] + jit * torch.randn_like(b["xp"])
    return nb


def pre_loss(model, b, cfg, lam1=0.5, lam2=0.2):
    m = b["m"]
    tm = (torch.rand(m.shape, device=m.device) < 0.15) & m
    rtm = None
    if cfg["route"]:
        rtm = (torch.rand(b["rt_mask"].shape, device=m.device) < 0.15) & b["rt_mask"]
    o = model(b, tokmask=tm, rtmask=rtm)
    L = torch.tensor(0.0, device=m.device)
    if tm.any():
        L = L + F.mse_loss(model.rec_w(o["tok_ctx"][tm]), b["x"][tm])
    if rtm is not None and rtm.any():
        L = L + F.mse_loss(model.rec_r(o["rt_ctx"][rtm]), b["rt_feat"][rtm])
    if len(m) > 1:
        L = L + lam1 * info_nce(model(view(b))["proj"], model(view(b))["proj"])
    if cfg.get("ord", True):
        zs = model(shuffle_within_chamber(b))["ordl"]; zr = o["ordl"]
        L = L + lam2 * (F.binary_cross_entropy_with_logits(zr, torch.zeros_like(zr)) + F.binary_cross_entropy_with_logits(zs, torch.ones_like(zs)))
    return L


def iterate(idx, bs, rng, shuffle=True):
    idx = np.asarray(idx)
    if shuffle:
        idx = rng.permutation(idx)
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


@torch.no_grad() if HAS_TORCH else (lambda f: f)
def predict(model, A, idx, data, te=None, mode="ours", bs=128):
    model.eval(); outs = defaultdict(list)
    for bi in iterate(idx, bs, None, shuffle=False):
        o = model(to_dev(A, bi, data, te, mode))
        for k in ("z", "logits", "pat", "gates", "a_s"):
            if k in o and o[k] is not None:
                outs[k].append(o[k].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def train_neural(model, A, data, tr, va, cfg, te=None, mode="ours", pre_epochs=100, epochs=30, bs=64, seed=0, lr_pre=3e-4, lr=3e-4, wd=1e-4):
    rng = np.random.default_rng(seed); model.to(DEV)
    y = A["y"]; ncls = cfg["n_cls"]
    cnt = np.bincount(y[tr], minlength=ncls).astype(float)
    alpha = torch.tensor(np.where(cnt > 0, cnt.sum() / (ncls * np.maximum(cnt, 1)), 0.0), dtype=torch.float32, device=DEV)
    if mode == "ours" and cfg.get("pretrain", True) and pre_epochs > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=lr_pre, weight_decay=1e-4)
        nrm = [i for i in tr if y[i] == 0]
        for ep in range(pre_epochs):
            model.train()
            for bi in iterate(nrm, bs, rng):
                b = to_dev(A, bi, data, te, mode); loss = pre_loss(model, b, cfg)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    opt = torch.optim.AdamW(model.parameters(), lr=lr if mode == "ours" else 3e-4, weight_decay=wd if mode == "ours" else 1e-4)
    best, best_state, wait = -1, None, 0
    for ep in range(epochs):
        model.train()
        for bi in iterate(tr, bs, rng):
            b = to_dev(A, bi, data, te, mode); o = model(b)
            yt = torch.as_tensor(y[bi]).long().to(DEV)
            loss = focal_loss(o["logits"], yt, alpha, use_focal=cfg.get("focal", True))
            if mode == "ours" and "logits_b" in o and len(o["logits_b"]) > 1:
                loss = loss + 0.5 * sum(focal_loss(lb, yt, alpha, use_focal=cfg.get("focal", True)) for lb in o["logits_b"]) / len(o["logits_b"])
            if mode == "ours":
                pt = torch.as_tensor(A["pat"][bi]).long().to(DEV); ok = pt >= 0
                if ok.any():
                    loss = loss + F.cross_entropy(o["pat"][ok], pt[ok])
                if cfg.get("pretrain", True):
                    loss = loss + 0.1 * pre_loss(model, b, cfg)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if len(va) and len(np.unique((y[va] > 0).astype(int))) > 1:
            p = predict(model, A, va, data, te, mode)
            sc = logodds(p["logits"]); v = prauc((y[va] > 0).astype(int), sc)
            if v > best + 1e-4:
                best, wait = v, 0; best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                wait += 1
                if wait >= 8:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def logodds(lg):
    lg = np.asarray(lg, np.float64)
    if lg.shape[1] == 1:
        return lg[:, 0]
    mx = lg.max(1, keepdims=True)
    lse_ab = mx[:, 0] + np.log(np.exp(lg[:, 1:] - mx).sum(1))
    return lse_ab - lg[:, 0]


class Scorer:
    """s(q) = standardized Mahalanobis(z) + standardized log-odds (abnormality score).
    The standardization uses NORMAL TRAINING lots (not calibration lots), so the score function does not depend on the
    calibration set and the split-conformal threshold stays valid."""
    def __init__(self, z_ref, lo_ref, use_maha=True):
        self.use_maha = use_maha
        lw = LedoitWolf().fit(z_ref); self.mu = lw.location_; self.P = lw.precision_
        mh = self.maha(z_ref); self.m1, self.s1 = mh.mean(), mh.std() + 1e-9
        self.m2, self.s2 = lo_ref.mean(), lo_ref.std() + 1e-9

    def maha(self, z):
        d = z - self.mu
        return np.einsum("ij,jk,ik->i", d, self.P, d)

    def __call__(self, z, lo):
        s = (lo - self.m2) / self.s2
        return s + ((self.maha(z) - self.m1) / self.s1 if self.use_maha else 0)


def occlusion(model, A, data, idx, scorer, te=None):
    """Eq. 13: φ_s = s(q) - s(q without wafer s), φ_c = s(q) - s(q without chamber c). Keys of φ_c = GLOBAL chamber id."""
    model.eval(); res = {}
    with torch.no_grad():
        for i in idx:
            b = to_dev(A, [i], data, te); m = b["m"][0]; obs = torch.where(m)[0]
            cls_ = torch.unique(b["cl"][0][m]); n = 1 + len(obs) + len(cls_)
            bb = {k: v.repeat(n, *([1] * (v.dim() - 1))) for k, v in b.items()}
            for j, s_ in enumerate(obs):
                bb["m"][1 + j, s_] = False
            for j, c in enumerate(cls_):
                bb["m"][1 + len(obs) + j] = m & (b["cl"][0] != c)
            ok = bb["m"].sum(1) > 0
            o = model(bb)
            lw = o["logits_b"][0] if "logits_b" in o else o["logits"]       # wafer-branch score: only wafer data decide
            sc = (torch.logsumexp(lw[:, 1:], -1) - lw[:, 0]).cpu().numpy(); sc[~ok.cpu().numpy()] = sc[0]
            phi_s = np.full(S_MAX, np.nan); phi_s[obs.cpu().numpy()] = sc[0] - sc[1:1 + len(obs)]
            chs = np.asarray(A["ch"][i]); clv = np.asarray(A["cl"][i]); mm = m.cpu().numpy()
            phi_c = {}
            for j, c in enumerate(cls_.cpu().numpy()):
                gid = chs[(clv == c) & mm][0]; phi_c[int(gid)] = float(sc[0] - sc[1 + len(obs) + j])
            res[i] = {"phi_s": phi_s, "phi_c": phi_c, "a_s": o["a_s"][0].cpu().numpy()}
    return res


def route_ig(model, A, data, idx, scorer, te, steps=32):
    """Integrated gradients (Sundararajan et al., 2017) of the route-branch score with respect to the step tokens
    (numerical features + tool embedding; baseline = empty token, step position only)."""
    model.eval(); out = {}
    for i in idx:
        b = to_dev(A, [i], data, te); tot = None; K_ = b["rt_feat"].shape[1]
        for a in np.linspace(1.0 / steps, 1.0, steps):
            al_t = torch.full((1, K_, 1), float(a), device=b["rt_feat"].device).requires_grad_(True)
            o = model(b, rt_scale=al_t)
            lr_ = o["logits_b"][1] if "logits_b" in o and len(o["logits_b"]) > 1 else o["logits"]   # route branch
            s_ = (torch.logsumexp(lr_[:, 1:], -1) - lr_[:, 0]).sum()
            g = torch.autograd.grad(s_, al_t)[0]
            tot = g if tot is None else tot + g
        out[i] = (tot / steps).sum(-1)[0].detach().cpu().numpy()
    return out


# ============================================================ methods (B1-B10 and SALE) within one fold
CLASSIC = ["B1", "B2", "B3-L1", "B3-RF", "B4", "B5", "B10", "Ours-S"]
NEURAL = ["B6", "B7", "B8", "B9", "Ours"]
NAMES = {"B1": "B1 Hotelling T²", "B2": "B2 Chamber deviation", "B3-L1": "B3 Lot summary + L1-logistic",
         "B3-RF": "B3 Lot summary + random forest", "B4": "B4 Wafer-level + aggregation", "B5": "B5 Isolation Forest",
         "B6": "B6 Deep one-class", "B7": "B7 Deep Sets", "B8": "B8 Set Transformer", "B9": "B9 Flat attention MIL",
         "B10": "B10 Tool commonality", "Ours": "Ours (neural encoder)", "Ours-S": "Ours-S (structured, normal-referenced)"}
BASE_CFG = dict(crn=True, peer=True, hier=True, emb=True, polar=True, slot_tf=True, route=True, ctx=True,
                fusion="gate", pretrain=True, ord=True, focal=True, maha=True)
ABLATIONS = {"w/o chamber-relative normalization (2)": dict(crn=False), "w/o peer contrast (3)": dict(peer=False),
             "Flat MIL instead of hierarchy (7)–(8)": dict(hier=False), "w/o slot and chamber embeddings": dict(emb=False),
             "Cartesian CNN instead of polar encoder (4)–(5)": dict(polar=False), "w/o slot Transformer (zslot)": dict(slot_tf=False),
             "w/o route branch": dict(route=False), "w/o context branch": dict(ctx=False),
             "Concatenation instead of gated fusion": dict(fusion="concat"), "w/o pretraining (Stage 1)": dict(pretrain=False),
             "w/o ℒord (within-chamber shuffle)": dict(ord=False), "Cross-entropy instead of focal loss": dict(focal=False)}


def _lo(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def chamber_stats(Xn, data, fn):
    W = data.W
    df = pd.DataFrame(Xn); df["li"] = W["li"].values; df["ch"] = W["ch_id"].values
    if fn != "absmean":
        return df.groupby(["li", "ch"]).agg(fn)
    obs = getattr(data, "obs", np.ones_like(Xn, bool))
    a = (np.abs(Xn) * obs).sum(1) / np.maximum(obs.sum(1), 1)          # mean |x̃| over measured features only
    a = np.where(obs.sum(1) > 0, a, np.nan)
    return pd.DataFrame({"li": df["li"], "ch": df["ch"], "a": a}).groupby(["li", "ch"])["a"].mean()


# ------------------------------------------------------------ Ours-S / SALE-S (v18): explicit structure features + normal-referenced fusion
# DESIGN FROZEN before it was run, based on a diagnosis of the failures of the neural model.
# Fixed hyperparameters (not tuned on test data): top-k = 5, |z| > 3, 3 early wafers, RF with 300 trees, fusion weight 0.5.
def _mmean(v, M):
    Mx = M[..., None] if v.ndim == 3 else M
    return (v * Mx).sum(1) / np.maximum(M.sum(1), 1)[..., None] if v.ndim == 3 else (v * M).sum(1) / np.maximum(M.sum(1), 1)


def _mmax(v, M):
    Mx = M[..., None] if v.ndim == 3 else M
    r = np.where(Mx, v, -np.inf).max(1)
    return np.where(np.isfinite(r), r, 0.0)


def _struct_block(v, M, cl, order):
    """Structure statistics of a tensor v [L,S,F]: mean, max|.|, sd, largest chamber contrast,
    slope against slot order (drift), difference between the first 3 wafers and the rest (post-PM)."""
    L, S, Fd = v.shape; Mf = M[..., None].astype(float)
    mu = _mmean(v, M); mx = _mmax(np.abs(v), M)
    sd = np.sqrt(np.maximum(_mmean(v ** 2, M) - mu ** 2, 0))
    cmax = np.zeros((L, Fd))
    for c in range(int(cl.max()) + 1 if cl.size else 0):
        Mc = M & (cl == c); nc = Mc.sum(1)
        muc = (v * Mc[..., None]).sum(1) / np.maximum(nc, 1)[:, None]
        cmax = np.maximum(cmax, np.where((nc > 0)[:, None], np.abs(muc - mu), 0))
    n = M.sum(1); ob = np.where(M, order - ((n - 1) / 2)[:, None], 0.0)
    var = (ob ** 2).sum(1)
    slope = (ob[..., None] * (v - mu[:, None, :]) * Mf).sum(1) / np.maximum(var, 1e-9)[:, None]
    e1 = M & (order < 3); e2 = M & (order >= 3)
    early = np.where((e2.sum(1) > 0)[:, None], _mmean(v, e1) - _mmean(v, e2), 0.0)
    return [mu, mx, sd, cmax, slope, early]


def struct_features(data, A, fit):
    """Lot features of Ours-S (SALE-S). Returns: Z (features standardized per tool against normal training lots),
    onehot (route/context), aw [L,S] anomaly score per wafer (mean of top-5 |z|), and s_oc (one-class score per lot)."""
    x = A["x"]; xp = A["xp"]; M = A["m"]; cl = A["cl"]; L, S, J = x.shape
    order = np.where(M, np.cumsum(M, 1) - 1, 0).astype(float)
    absx = np.abs(x); k = min(5, J)
    aw = np.where(M, -np.sort(-absx, axis=2)[..., :k].mean(2), np.nan)           # anomaly per wafer
    f3 = np.where(M, (absx > 3).mean(2), np.nan)
    awz = np.nan_to_num(aw); med = np.nanmedian(np.where(M, aw, np.nan), 1); med = np.nan_to_num(med)
    mad = np.nan_to_num(np.nanmedian(np.abs(np.where(M, aw, np.nan) - med[:, None]), 1))
    nout = (M & (awz > (med + 3 * 1.4826 * mad + 1e-6)[:, None])).sum(1)
    top3 = -np.sort(-np.where(M, awz, -1), 1)[:, :3]; top3 = np.where(top3 < 0, 0, top3).mean(1)
    parts = _struct_block(x, M, cl, order)
    parts += _struct_block(awz[..., None], M, cl, order)
    parts += [_struct_block(xp, M, cl, order)[3], _mmax(np.abs(xp), M)]           # contrast between chambers (peer contrast)
    if "mf" in A:
        parts += _struct_block(A["mf"], M, cl, order)
    parts += [np.c_[_mmean(awz, M), _mmax(awz, M), top3, _mmax(awz, M) - med, nout, _mmean(np.nan_to_num(f3), M),
                    _mmax(np.nan_to_num(f3), M), M.sum(1)]]
    X = np.nan_to_num(np.hstack(parts).astype(np.float64))
    # per-tool standardization against NORMAL training lots (fallback: all normal training lots)
    yb = data.ybin; nf = fit[yb[fit] == 0]; tools = data.lots["tool"].astype(str).values
    def ref(idx):
        md = np.median(X[idx], 0); s = 1.4826 * np.median(np.abs(X[idx] - md), 0)
        s = np.where(s > 1e-9, s, X[idx].std(0) + 1e-6)
        return md, s
    g_md, g_s = ref(nf); Z = (X - g_md) / g_s
    for tl in np.unique(tools):
        nt = nf[tools[nf] == tl]
        if len(nt) >= 10:
            md, s = ref(nt); rows = tools == tl; Z[rows] = (X[rows] - md) / s
    Z = np.clip(Z, -20, 20)
    s_oc = -np.sort(-np.abs(Z), 1)[:, :5].mean(1)                                    # top-5 deviations from normal
    oh = []
    if data.route is not None:
        R_ = data.route; o_ = np.zeros((L, int(R_["rt_id"].max()) + 1)); o_[np.arange(L)[:, None], R_["rt_id"]] = 1; oh.append(o_)
    for c in data.ctx_cols:
        oh.append(pd.get_dummies(data.lots[c].astype(str), prefix=c).astype(float).values)
    OH = np.hstack(oh) if oh else np.zeros((L, 0))
    return Z.astype(np.float32), OH.astype(np.float32), aw, s_oc


def run_ours_s(data, A, TE, fit, va, cal, te, ev, seed):
    y = data.y; yb = data.ybin; ncls = len(data.types) + 1
    trf = np.concatenate([fit, va]) if len(va) else fit
    Z, OH, aw, s_oc = struct_features(data, A, fit)
    Xs = np.hstack([Z, OH])
    rf = RandomForestClassifier(300, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample",
                                oob_score=True, n_jobs=-1, random_state=seed).fit(Xs[trf], y[trf])
    P = np.zeros((len(ev), ncls)); P[:, rf.classes_] = rf.predict_proba(Xs[ev]); P /= P.sum(1, keepdims=True)
    oob = np.zeros((len(trf), ncls)); od = np.nan_to_num(rf.oob_decision_function_, nan=0.0); oob[:, rf.classes_] = od
    s_sup = _lo(1 - P[:, 0]); ref_sup = _lo(1 - oob[:, 0][yb[trf] == 0])            # reference = OOB scores of normal training lots
    nf = fit[yb[fit] == 0]
    m_s, sd_s = ref_sup.mean(), ref_sup.std() + 1e-9; m_o, sd_o = s_oc[nf].mean(), s_oc[nf].std() + 1e-9
    r = {"score": 0.5 * (s_sup - m_s) / sd_s + 0.5 * (s_oc[ev] - m_o) / sd_o, "prob": P,
         "feat": Xs[ev], "feat_fit_ab": Xs[fit[yb[fit] == 1]], "fit_ab": fit[yb[fit] == 1]}
    r["w_attr"] = {i: aw[i] for i in te}
    chr_ = {}
    for i in te:
        m = A["m"][i]; s_ = pd.Series(np.nan_to_num(aw[i])[m]).groupby(A["ch"][i][m]).mean()
        chr_[i] = list((s_ - s_.mean()).sort_values(ascending=False).index)
    r["ch_rank"] = chr_
    if TE is not None:
        r["step_rank"] = {i: list(np.argsort(-TE[i])) for i in te}
    return r


def run_fold(data, fit, va, cal, te, cfg, methods, seed=0, opts=None):
    """Train every method on 'fit', calibrate on the normal 'cal' lots, score cal + te."""
    opts = opts or {}; c = dict(BASE_CFG); c.update(cfg)
    y = data.y; yb = data.ybin; ncls = len(data.types) + 1
    fit, va, cal, te = (np.asarray(v, int) for v in (fit, va, cal, te))
    ev = np.concatenate([cal, te])
    Xn = crn(data, fit[yb[fit] == 0], use=c["crn"])
    Xp = peer_contrast(data, Xn) if c["peer"] else np.zeros_like(Xn)
    A = lot_arrays(data, Xn, Xp)
    TE = target_encoding(data, A, fit)
    out = {"ev": ev, "cal": cal, "te": te}
    Sstd = None
    need_S = any(m in methods for m in ("B3-L1", "B3-RF", "B3-GB", "B5", "B6", "CTX"))
    if need_S:
        S, _ = lot_summary(data, Xn, ctx_only=("CTX" in methods and len(methods) == 1), extra=opts.get("extra"))
        sc = StandardScaler().fit(S[fit]); Sstd = np.clip(sc.transform(S), -10, 10)
    trf = np.concatenate([fit, va]) if len(va) else fit
    sup_ok = len(np.unique(yb[trf])) > 1
    for m in methods:
        t0 = time.time(); r = {}
        if not sup_ok and m in ("B3-L1", "B3-RF", "B3-GB", "B4", "B7", "B8", "B9", "Ours", "Ours-S", "CTX", "B10"):
            log(f"   - {m} skipped: no abnormal lots in the training data"); out[m] = {}; continue
        try:
            if m == "B1":
                cm = chamber_stats(Xn, data, "mean")
                fl = cm.index.get_level_values(0).isin(fit[yb[fit] == 0])
                lw = LedoitWolf().fit(cm.values[fl]); d = cm.values - lw.location_
                t2 = pd.Series(np.einsum("ij,jk,ik->i", d, lw.precision_, d), index=cm.index)
                r["score"] = t2.groupby(level=0).max().reindex(ev).fillna(0).values
                r["ch_rank"] = {i: list(t2.loc[i].sort_values(ascending=False).index) for i in te if i in t2.index.get_level_values(0)}
            elif m == "B2":
                a = chamber_stats(Xn, data, "absmean")
                r["score"] = a.groupby(level=0).max().reindex(ev).fillna(0).values
                r["ch_rank"] = {i: list(a.loc[i].sort_values(ascending=False).index) for i in te if i in a.index.get_level_values(0)}
            elif m in ("B3-L1", "B3-RF", "B3-GB", "CTX"):
                if m == "B3-RF":
                    rfp = dict(n_estimators=300, min_samples_leaf=3, max_features="sqrt"); rfp.update(opts.get("rf_params", {}))   # tunable (revision 2)
                    mdl = RandomForestClassifier(**rfp, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
                elif m == "B3-GB":
                    # revision 2: gradient boosting on exactly the lot-summary features of the forest, fixed default settings
                    mdl = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
                else:
                    mdl = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, class_weight="balanced", max_iter=3000)
                    if len(np.unique(y[trf])) > 2:
                        from sklearn.multiclass import OneVsRestClassifier
                        mdl = OneVsRestClassifier(mdl)
                mdl.fit(Sstd[trf], y[trf])
                P = np.zeros((len(ev), ncls)); P[:, mdl.classes_] = mdl.predict_proba(Sstd[ev])
                P = P / P.sum(1, keepdims=True)
                r["score"] = _lo(1 - P[:, 0]); r["prob"] = P; r["feat"] = Sstd[ev]
                fab = fit[yb[fit] == 1]; r["feat_fit_ab"] = Sstd[fab]; r["fit_ab"] = fab
                if m == "B3-L1":
                    r["coef"] = np.abs(mdl.coef_ if hasattr(mdl, "coef_") else np.vstack([e.coef_ for e in mdl.estimators_])).mean(0)
            elif m == "B4":
                mf = data.mapfeat[data.W["map_i"].values] if data.mapfeat is not None else np.zeros((len(Xn), 0))
                Xw = np.hstack([Xn, mf]); li = data.W["li"].values
                wf = np.isin(li, trf); yw = y[li[wf]]
                cw = np.bincount(yw, minlength=ncls).astype(float); sw = (cw.sum() / (ncls * np.maximum(cw, 1)))[yw]
                gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed).fit(Xw[wf], yw, sample_weight=sw)
                we = np.isin(li, ev); Pw = np.zeros((we.sum(), ncls)); Pw[:, gb.classes_] = gb.predict_proba(Xw[we])
                dfw = pd.DataFrame(Pw); dfw["li"] = li[we]; dfw["pab"] = 1 - Pw[:, 0]; dfw["slot"] = data.W["slot"].values[we]
                dfw["ch"] = data.W["ch_id"].values[we]
                top = dfw.sort_values("pab", ascending=False).groupby("li").head(3)
                sc = top.groupby("li")["pab"].mean().reindex(ev).fillna(0).values; r["score"] = _lo(sc)
                pm = top.groupby("li")[list(range(ncls))].mean().reindex(ev).fillna(0).values; r["prob"] = pm / pm.sum(1, keepdims=True).clip(1e-9)
                r["w_attr"] = {i: g.set_index("slot")["pab"].reindex(range(1, 26)).values for i, g in dfw[dfw.li.isin(te)].groupby("li")}
                r["ch_rank"] = {i: list(g.groupby("ch")["pab"].mean().sort_values(ascending=False).index) for i, g in dfw[dfw.li.isin(te)].groupby("li")}
            elif m == "B5":
                nf = fit[yb[fit] == 0]
                iso = IsolationForest(n_estimators=300, random_state=seed).fit(Sstd[nf]); r["score"] = -iso.score_samples(Sstd[ev])
            elif m == "B10":
                if data.route is None:
                    continue
                ids = A["rt_id"]; K = ids.shape[1]; pv = {}
                for k in range(K):
                    for t_ in np.unique(ids[trf, k]):
                        a_ = int(((ids[trf, k] == t_) & (yb[trf] == 1)).sum()); b_ = int(((ids[trf, k] == t_) & (yb[trf] == 0)).sum())
                        c_ = int(((ids[trf, k] != t_) & (yb[trf] == 1)).sum()); d_ = int(((ids[trf, k] != t_) & (yb[trf] == 0)).sum())
                        pv[(k, t_)] = stats.fisher_exact([[a_, b_], [c_, d_]], alternative="greater")[1]
                r["step_rank"] = {i: list(np.argsort([pv.get((k, ids[i, k]), 1.0) for k in range(K)])) for i in te}
                # lot score = -log10 of the smallest Fisher p among the steps the lot visited
                r["score"] = np.array([-np.log10(max(1e-300, min(pv.get((k, ids[i, k]), 1.0) for k in range(K)))) for i in ev])
            elif m == "B6":
                if not HAS_TORCH:
                    continue
                seed_all(seed); net = DeepSVDD(Sstd.shape[1]).to(DEV); nf = fit[yb[fit] == 0]
                Xt = torch.as_tensor(Sstd[nf], dtype=torch.float32, device=DEV)
                with torch.no_grad():
                    cen = net(Xt).mean(0)
                opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-5)
                for ep in range(opts.get("epochs", 30) * 2):
                    for bi in iterate(np.arange(len(nf)), 64, np.random.default_rng(ep)):
                        loss = ((net(Xt[bi]) - cen) ** 2).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    r["score"] = ((net(torch.as_tensor(Sstd[ev], dtype=torch.float32, device=DEV)) - cen) ** 2).sum(1).cpu().numpy()
            elif m in ("B7", "B8", "B9"):
                if not HAS_TORCH:
                    continue
                seed_all(seed)
                kind = {"B7": "deepsets", "B8": "settf", "B9": "flatmil"}[m]
                tin = Xn.shape[1] + (17 if "mf" in A else 0)
                net = SetModel(kind, tin, ncls).to(DEV)
                net = train_neural(net, A, data, fit, va, {"n_cls": ncls, "focal": True}, mode="set", epochs=opts.get("epochs", 30), seed=seed)
                p = predict(net, A, ev, data, mode="set")
                r["score"] = logodds(p["logits"]); pr = torch.softmax(torch.as_tensor(p["logits"]), -1).numpy(); r["prob"] = pr
                if "a_s" in p:
                    pos = {i: k for k, i in enumerate(ev)}
                    r["w_attr"] = {i: p["a_s"][pos[i]] for i in te}
                    r["ch_rank"] = {i: _chrank(p["a_s"][pos[i]], A, i) for i in te}
            elif m == "Ours-S":
                r = run_ours_s(data, A, TE, fit, va, cal, te, ev, seed)
            elif m == "Ours":
                if not HAS_TORCH:
                    continue
                r = run_ours(data, A, TE, fit, va, cal, te, ev, c, seed, opts)
        except Exception as e:
            log(f"   ! {m} failed: {type(e).__name__}: {e}"); r = {}
        r["sec"] = time.time() - t0
        out[m] = r
        if opts.get("verbose", True):
            log(f"      {m}: {r['sec']:.0f} s")
    return out


def _chrank(w, A, i):
    m = A["m"][i]; ch = A["ch"][i]
    s = pd.Series(np.nan_to_num(w)[m]).groupby(ch[m]).sum()
    return list(s.sort_values(ascending=False).index)


def run_ours(data, A, TE, fit, va, cal, te, ev, c, seed, opts):
    seed_all(seed)
    ncls = len(data.types) + 1
    cfg = dict(c, J=A["x"].shape[2], maps=data.maps is not None, route=bool(c["route"] and data.route is not None),
               ctx=bool(c["ctx"] and len(data.ctx_cols) > 0), n_ch=len(data.ch_vocab), n_cls=ncls,
               ctx_sizes=[len(data.ctx_vocab[k]) for k in data.ctx_cols])
    if data.route is not None:
        cfg.update(Fr=A["rt_feat"].shape[2], K=A["rt_feat"].shape[1], n_tt=int(A["rt_id"].max()) + 1)
    if not cfg["ctx"]:
        cfg["ctx_sizes"] = []
    model = LotEncoder(cfg).to(DEV)
    if cfg["maps"] and opts.get("pre_menc", True) and (getattr(data, "pre_maps", None) is not None or getattr(data, "mix_maps", None) is not None):
        pretrain_menc(model.menc, getattr(data, "pre_maps", None), getattr(data, "pre_y", None), epochs=opts.get("menc_epochs", 3),
                      Pm=getattr(data, "mix_maps", None), ym=getattr(data, "mix_y", None))
    model = train_neural(model, A, data, fit, va, cfg, te=TE, mode="ours", pre_epochs=opts.get("pre_epochs", 100),
                         epochs=opts.get("epochs", 30), seed=seed, lr=opts.get("lr", 3e-4), wd=opts.get("wd", 1e-4))
    yb = data.ybin
    nf = fit[yb[fit] == 0]
    pn = predict(model, A, nf, data, TE); pe = predict(model, A, ev, data, TE)
    lo_e = logodds(pe["logits"])
    scorer = Scorer(pn["z"], logodds(pn["logits"]), use_maha=c.get("maha", True))     # reference = NORMAL TRAINING lots
    r = {"score": scorer(pe["z"], lo_e), "z": pe["z"], "gates": pe["gates"], "pat_pred": pe["pat"].argmax(1)}
    fab = fit[yb[fit] == 1]
    if len(fab):
        r["z_fit_ab"] = predict(model, A, fab, data, TE)["z"]; r["fit_ab"] = fab
    r["prob"] = torch.softmax(torch.as_tensor(pe["logits"]), -1).numpy()
    r["a_s_att"] = {i: pe["a_s"][k] for k, i in enumerate(ev) if i in set(te)}
    att_idx = [i for i in te if yb[i] == 1][: opts.get("max_attr", 400)]
    if opts.get("attr", True) and att_idx:
        oc = occlusion(model, A, data, att_idx, scorer, TE)
        r["w_attr"] = {i: v["phi_s"] for i, v in oc.items()}
        r["ch_rank"] = {i: sorted(v["phi_c"], key=lambda k: -v["phi_c"][k]) for i, v in oc.items()}
        r["phi_c"] = {i: v["phi_c"] for i, v in oc.items()}
        if cfg["route"]:
            f4 = [i for i in att_idx if data.lots.loc[i].get("true_step", -1) >= 0] if "true_step" in data.lots else []
            ig = route_ig(model, A, data, f4[: opts.get("max_ig", 150)], scorer, TE)
            r["step_rank"] = {i: list(np.argsort(-v)) for i, v in ig.items()}
    r["n_param"] = int(sum(p.numel() for p in model.parameters()))
    # scoring time per lot
    t0 = time.time(); predict(model, A, te[:256], data, TE, bs=1); r["ms_per_lot"] = 1000 * (time.time() - t0) / max(1, min(256, len(te)))
    return r


def pretrain_menc(menc, P, ycls, epochs=3, bs=256, Pm=None, ym=None):
    """Pretraining of the map encoder (Training, Stage 1, map part): labelled WM-811K maps of training hosts
    (cross-entropy, 9 classes) and, if available, MixedWM38 (multi-label BCE, 8 patterns). Random rotation = shift along
    the angle of the polar map."""
    if (P is None or len(P) == 0) and (Pm is None or len(Pm) == 0):
        return
    opt = torch.optim.AdamW(menc.parameters(), 1e-3); rng = np.random.default_rng(0)
    sets = []
    if P is not None and len(P):
        sets.append(("wm", P, torch.as_tensor(np.asarray(ycls, np.int64))))
    if Pm is not None and len(Pm):
        sets.append(("mixed", Pm, torch.as_tensor(ym)))
    for ep in range(epochs):
        for nm, P_, y_ in sets:
            for bi in iterate(np.arange(len(P_)), bs, rng):
                x = torch.as_tensor(P_[bi].astype(np.float32)).to(DEV)
                x = torch.roll(x, int(rng.integers(0, NT)), -1)
                u, _ = menc(x)
                if nm == "wm":
                    loss = F.cross_entropy(menc.cls(u), y_[bi].to(DEV))
                else:
                    loss = F.binary_cross_entropy_with_logits(menc.mcls(u), y_[bi].to(DEV))
                opt.zero_grad(); loss.backward(); opt.step()


# ============================================================ evaluation
def truth_matrix(data):
    T = np.full((len(data.lots), S_MAX), np.nan)
    W = data.W; T[W["li"].values, W["slot"].values.astype(int) - 1] = W["wbad"].values
    return T


def wafer_attr_bootstrap(attr, T, lots, B=1000, seed=0):
    """Wafer-attribution AUROC with a bootstrap over lots (wafers of one lot are not independent):
    95% interval and two-sided p-value against 0.5."""
    ys, ss = [], []
    for i in lots:
        if i not in attr:
            continue
        t = T[i]; a = np.asarray(attr[i], float); ok = ~np.isnan(t) & ~np.isnan(a)
        if ok.sum() >= 2 and 0 < np.nansum(t[ok]) < ok.sum():
            ys.append(t[ok]); ss.append(a[ok])
    if len(ys) < 3:
        return {"AUROC": np.nan, "lo": np.nan, "hi": np.nan, "p_vs_0.5": np.nan, "n_lots": len(ys)}
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        k = rng.integers(0, len(ys), len(ys))
        v.append(auroc(np.concatenate([ys[j] for j in k]), np.concatenate([ss[j] for j in k])))
    v = np.array(v, float); v = v[np.isfinite(v)]; a = auroc(np.concatenate(ys), np.concatenate(ss))
    p = max(2 * min((v <= 0.5).mean(), (v >= 0.5).mean()), 1.0 / max(len(v), 1))
    return {"AUROC": a, "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)), "p_vs_0.5": float(min(1.0, p)), "n_lots": len(ys)}


def wafer_attr_auc(attr, T, lots):
    ys, ss = [], []
    for i in lots:
        if i not in attr:
            continue
        t = T[i]; a = np.asarray(attr[i], float); ok = ~np.isnan(t) & ~np.isnan(a)
        if ok.sum() >= 2 and 0 < np.nansum(t[ok]) < ok.sum():
            ys.append(t[ok]); ss.append(a[ok])
    if not ys:
        return np.nan, 0
    return auroc(np.concatenate(ys), np.concatenate(ss)), len(ys)


def split_train(tr, yb, seed, cal_frac=0.15, va_frac=0.10):
    rng = np.random.default_rng(seed); tr = np.asarray(tr)
    nrm = tr[yb[tr] == 0]; nc = max(10, int(round(cal_frac * len(nrm)))) if len(nrm) > 30 else max(3, len(nrm) // 5)
    cal = rng.choice(nrm, nc, replace=False); fit = np.setdiff1d(tr, cal)
    va = np.array([], int)
    if (yb[fit] == 1).sum() >= 8:
        fit, va = train_test_split(fit, test_size=va_frac, stratify=yb[fit], random_state=seed)
    return np.asarray(fit), np.asarray(va), cal


def eval_single(data, out, eps=0.05, rot3=None):
    """Metrics on one fixed split (LotBench)."""
    te, cal = out["te"], out["cal"]; yb, y = data.ybin, data.y; T = truth_matrix(data); res = {}
    for m, r in out.items():
        if not isinstance(r, dict) or "score" not in r:
            continue
        s = np.asarray(r["score"]); st, sc = s[len(cal):], s[:len(cal)][yb[cal] == 0]
        tau = conformal_tau(sc, eps); e = {"AUROC": auroc(yb[te], st), "PR-AUC": prauc(yb[te], st), "R@5%": recall_at_fpr(yb[te], st)}
        if "prob" in r:
            P = np.asarray(r["prob"])[len(cal):]
            pred = np.where(st > tau, P[:, 1:].argmax(1) + 1, 0)
            e["Macro-F1"] = float(f1_score(y[te], pred, average="macro", labels=list(range(len(data.types) + 1))))
        e["realized_FAR"] = float((st[yb[te] == 0] > tau).mean()); e["recall_eps"] = float((st[yb[te] == 1] > tau).mean())
        if rot3 is not None:
            n3 = (yb[te] == 0) & rot3[te]; e["FAR_N_rot3"] = float((st[n3] > tau).mean()) if n3.any() else np.nan
        if "scenario" in data.lots:
            sc_ = data.lots["scenario"].values[te]; e["per_scen"] = {}
            for k in SCEN:
                msk = (sc_ == k) | (yb[te] == 0)
                e["per_scen"][k] = auroc(yb[te][msk], st[msk])
            eta = data.lots["eta"].values[te]; e["per_eta"] = {}; e["eta_all"] = {}
            for et in ETAS:
                msk = (np.isclose(eta, et)) | (yb[te] == 0)
                e["eta_all"][f"{et}"] = auroc(yb[te][msk], st[msk])
                for k in SCEN:
                    msk = ((sc_ == k) & (np.isclose(eta, et))) | (yb[te] == 0)
                    e["per_eta"][f"{k}_{et}"] = auroc(yb[te][msk], st[msk])
        if "w_attr" in r:
            sub = [i for i in te if data.lots.loc[i, "scenario"] not in ("F4", "F5")] if "scenario" in data.lots else list(te)
            e["Wafer attr. AUROC"], _ = wafer_attr_auc(r["w_attr"], T, sub)
            e["Wafer attr. bootstrap"] = wafer_attr_bootstrap(r["w_attr"], T, sub)
            if "a_s_att" in r:
                e["Wafer attr. AUROC (attention)"], _ = wafer_attr_auc(r["a_s_att"], T, sub)
        if "ch_rank" in r and "true_ch" in data.lots:
            hits = []
            for i, rk in r["ch_rank"].items():
                tc = data.lots.loc[i, "true_ch"]
                if tc and data.lots.loc[i, "scenario"] in ("F1", "F2", "F6"):
                    hits.append(len(rk) > 0 and rk[0] == data.ch_vocab.get(tc, -1))
            e["Chamber hit@1"] = float(np.mean(hits)) if hits else np.nan
            h1 = {int(i): bool(len(rk) > 0 and rk[0] == data.ch_vocab.get(data.lots.loc[i, "true_ch"], -1)) for i, rk in r["ch_rank"].items() if data.lots.loc[i, "scenario"] == "F1"}
            e["Chamber hit@1 F1"] = float(np.mean(list(h1.values()))) if h1 else np.nan
            e["_chamber_hits_F1"] = h1                                  # per lot, for paired McNemar tests
        if "step_rank" in r and "true_step" in data.lots:
            hs = [rk[0] == data.lots.loc[i, "true_step"] for i, rk in r["step_rank"].items() if data.lots.loc[i, "true_step"] >= 0]
            e["Step hit@1"] = float(np.mean(hs)) if hs else np.nan
        if "pat_pred" in r and "pattern" in data.lots:
            pt = data.lots["pattern"].values[te]; pp = np.asarray(r["pat_pred"])[len(cal):]; ok = pt >= 0
            e["Slot-pattern macro-F1"] = float(f1_score(pt[ok], pp[ok], average="macro")) if ok.any() else np.nan
        if "gates" in r:
            G = np.asarray(r["gates"])[len(cal):]; e["gates_mean"] = G.mean(0).tolist()
            if "scenario" in data.lots:
                e["gates_scen"] = {k: G[data.lots["scenario"].values[te] == k].mean(0).tolist() for k in ["N"] + SCEN if (data.lots["scenario"].values[te] == k).any()}
        e["sec"] = r.get("sec"); e["n_param"] = r.get("n_param"); e["ms_per_lot"] = r.get("ms_per_lot")
        res[m] = e
    return res


def retrieval_hits5(Z, fit_ab, te_ab, cause):
    """Per test lot: does one of the five most similar abnormal training lots have the same cause?"""
    if len(fit_ab) == 0 or len(te_ab) == 0:
        return {}
    Zf = Z["fit"]; Zt = Z["te"]
    Zf = Zf / (np.linalg.norm(Zf, axis=1, keepdims=True) + 1e-9); Zt = Zt / (np.linalg.norm(Zt, axis=1, keepdims=True) + 1e-9)
    sim = Zt @ Zf.T; top = np.argsort(-sim, 1)[:, :5]
    return {int(q): bool((cause[fit_ab[t]] == cause[q]).any()) for q, t in zip(te_ab, top)}


def retrieval_hit5(Z, fit_ab, te_ab, cause):
    h = retrieval_hits5(Z, fit_ab, te_ab, cause)
    return float(np.mean(list(h.values()))) if h else np.nan


# ------------------------------------------------------------ A. LotBench-811K
def exp_lotbench(args, outdir):
    if args.demo:
        meta, maps = demo_wm811k(args.demo_lots, 0)
    else:
        if not args.wm811k or not os.path.exists(args.wm811k):
            log(f"WM-811K (LSWMD.pkl) not found (searched in {args.wm811k or PUBLIC_DATA_DIR}); part lotbench skipped"); return None
        log(f"Reading WM-811K: {args.wm811k} ...")
        try:
            meta, maps = load_wm811k(args.wm811k)
        except Exception as ex:
            log(f"Failed to read LSWMD.pkl: {type(ex).__name__}: {ex}"); return None
        log(f"   {len(meta)} maps, {meta['lot'].nunique()} lots, {int((meta.groupby('lot').size() >= 20).sum())} lots with >= 20 wafers")
    log(f"   injection strengths eta: {ETAS}")
    MIX = (None, None)
    if args.mixedwm38 and os.path.exists(args.mixedwm38) and not args.demo:
        try:
            MIX = load_mixedwm38(args.mixedwm38); log(f"   MixedWM38 for map-encoder pretraining: {len(MIX[0])} maps")
        except Exception as ex:
            log(f"   MixedWM38 could not be read ({type(ex).__name__}: {ex}); map pretraining with WM-811K only")
    meta = meta.reset_index(drop=True)
    R = {"seeds": {}, "ablation": {}, "etas": list(ETAS), "version": VERSION}
    methods = CLASSIC + NEURAL
    for si, seed in enumerate(args.seeds):
        log(f"[LotBench] seed {seed}: building the benchmark ...")
        data = make_lotbench(meta, maps, seed, n_lots=args.n_lots)
        tr = np.where(data.lots["split"] == "train")[0]; cal = np.where(data.lots["split"] == "cal")[0]; te = np.where(data.lots["split"] == "test")[0]
        fit, va = train_test_split(tr, test_size=0.10, stratify=data.ybin[tr], random_state=seed)
        hs = set(data.lots.loc[tr, "host"])
        L_ = meta[meta["ftype"].notna() & meta["lot"].isin(hs)]
        L_ = L_.sample(min(len(L_), 30000), random_state=seed)
        cls = sorted(L_["ftype"].unique())
        data.pre_maps = np.stack([to_polar(maps[i]) for i in L_.index]).astype(np.float16) if len(L_) else None
        data.pre_y = L_["ftype"].map({c: k for k, c in enumerate(cls)}).values if len(L_) else None
        data.mix_maps, data.mix_y = MIX
        opts = dict(epochs=args.epochs, pre_epochs=args.pre_epochs, max_attr=args.max_attr, menc_epochs=args.menc_epochs)
        log(f"[LotBench] seed {seed}: {len(data.lots)} lots, {int(data.ybin.sum())} abnormal; training {len(methods)} methods ...")
        out = run_fold(data, fit, va, cal, te, {}, methods, seed=seed, opts=opts)
        ev = eval_single(data, out, rot3=(data.lots["rot"].values == 3))
        # retrieval: abnormal training lots act as a memory of confirmed causes
        cause = data.lots["scenario"].values; tab = [i for i in te if data.ybin[i] == 1]
        for m_, zkey, fkey in (("Ours", "z", "z_fit_ab"), ("B3-L1", "feat", "feat_fit_ab"), ("Ours-S", "feat", "feat_fit_ab")):
            r_ = out.get(m_, {})
            if zkey in r_ and fkey in r_:
                pos = {i: k for k, i in enumerate(out["ev"])}
                Zt = np.asarray(r_[zkey])[[pos[i] for i in tab]]
                h_ = retrieval_hits5({"fit": np.asarray(r_[fkey]), "te": Zt}, np.asarray(r_["fit_ab"]), np.asarray(tab), cause)
                ev.setdefault(m_, {})["Retrieval hit@5"] = float(np.mean(list(h_.values()))) if h_ else np.nan
                ev[m_]["_retrieval_hits"] = h_
        R["seeds"][seed] = ev
        if si == 0:
            R["roc"] = {m: {"y": data.ybin[te].tolist(), "s": np.asarray(out[m]["score"])[len(cal):].tolist()} for m in out if isinstance(out[m], dict) and "score" in out[m]}
        # summary of this seed in the log (so progress can be followed without waiting for all seeds)
        log(f"[LotBench] seed {seed} AUROC: " + ", ".join(f"{m} {ev[m]['AUROC']:.3f}" for m in ev if "AUROC" in ev[m]))
        jdump(R, os.path.join(outdir, "raw", "lotbench.json"))     # save after every seed (before the ablations)
        if si < args.ablation_seeds:
            for nm, ch in ABLATIONS.items():
                log(f"[LotBench] ablation seed {seed}: {nm}")
                o2 = run_fold(data, fit, va, cal, te, ch, ["Ours"], seed=seed, opts=dict(opts, attr=False))
                e2 = eval_single(data, o2)
                R["ablation"].setdefault(nm, []).append(e2.get("Ours", {}).get("AUROC", np.nan))
                jdump(R, os.path.join(outdir, "raw", "lotbench.json"))
    return R


# ------------------------------------------------------------ A2. WM-811K lot task with real defects
_WM_CACHE = {}


def get_wm811k(args):
    if args.demo:
        meta, maps = demo_wm811k(args.demo_lots, 0)
        meta["ftype"] = meta["ftype"].fillna("none")          # fake data: every wafer counts as labelled
        return meta.reset_index(drop=True), maps
    if "wm" not in _WM_CACHE:
        if not args.wm811k or not os.path.exists(args.wm811k):
            return None, None
        log(f"Reading WM-811K: {args.wm811k} ...")
        meta, maps = load_wm811k(args.wm811k); _WM_CACHE["wm"] = (meta.reset_index(drop=True), maps)
    return _WM_CACHE["wm"]


def make_wmlot(meta, maps):
    g = meta.groupby("lot")
    full = g["ftype"].apply(lambda s: bool(s.notna().all())); n = g["slot"].apply(lambda s: int(((s >= 1) & (s <= S_MAX)).sum()))
    ok = full.index[full.values & (n.values >= WMLOT_MIN_W)]
    sub = meta[meta["lot"].isin(set(ok)) & (meta["slot"] >= 1) & (meta["slot"] <= S_MAX)].sort_values(["lot", "slot"])
    sub = sub[~sub.duplicated(["lot", "slot"])].copy()
    log(f"   [wmlot] {len(ok)} fully labelled lots with >= {WMLOT_MIN_W} wafers; converting {len(sub)} maps to polar ...")
    P = np.stack([to_polar(maps[i]) for i in sub.index]).astype(np.float16)
    mf = map_feats(P); sub["map_i"] = np.arange(len(sub)); sub["wbad"] = (sub["ftype"] != "none").astype(float)
    names = sorted(sub["lot"].unique()); li = {l: k for k, l in enumerate(names)}
    dsz = sub.groupby("lot")["die"].median().reindex(names)
    qs = np.nanquantile(dsz.values, [.2, .4, .6, .8])
    def dom(s):
        s = s[s != "none"]
        return s.value_counts().index[0] if len(s) else "none"
    lots = pd.DataFrame({"lot": names, "nbad": sub.groupby("lot")["wbad"].sum().reindex(names).values,
                         "n": sub.groupby("lot").size().reindex(names).values,
                         "dominant": sub.groupby("lot")["ftype"].apply(dom).reindex(names).values,
                         "product": np.searchsorted(qs, dsz.values).astype(int)})
    lots["y"] = (lots["nbad"] >= WMLOT_K).astype(int); lots["tool"] = "T"; lots["subset_same"] = True
    lots["time"] = pd.NaT; lots["month"] = "NA"
    fcols = [f"MAP:{j:02d}" for j in range(mf.shape[1])]
    W = pd.DataFrame(mf, columns=fcols)
    W["li"] = sub["lot"].map(li).values; W["slot"] = sub["slot"].values; W["chamber"] = "C0"; W["pm"] = 0.0
    W["wbad"] = sub["wbad"].values; W["map_i"] = sub["map_i"].values
    W["key_MAP_0"] = W["li"].map(dict(enumerate(lots["product"].astype(str)))); W["key_MAP_1"] = "all"
    notes = {"lot": int(len(lots)), "abnormal_(>=%d_patterned)" % WMLOT_K: int(lots.y.sum()), "wafer": int(len(W)),
             "patterned_wafers": int(W.wbad.sum()), "mixed_lots": int(((lots.nbad > 0) & (lots.nbad < lots.n)).sum()),
             "dominant_pattern_abnormal_lots": lots.loc[lots.y == 1, "dominant"].value_counts().to_dict()}
    d = LotData("wmlot", lots, W, {"MAP": fcols}, {"MAP": ["key_MAP_0", "key_MAP_1"]}, [], ["defect-pattern lot"],
                maps=P, mapfeat=mf, notes=notes)
    d.ftype_w = sub["ftype"].values                     # per-wafer pattern label (in map_i order), for pretraining the map encoder
    return d


def exp_wmlot(args, outdir):
    meta, maps = get_wm811k(args)
    if meta is None:
        log("WM-811K not found; part wmlot skipped"); return None
    data = make_wmlot(meta, maps)
    log(f"[wmlot] {json.dumps(data.notes, default=str)}")
    if data.ybin.sum() < 20 or (data.ybin == 0).sum() < 50:
        log("[wmlot] too few abnormal/normal lots; skipped"); return None
    MIX = (None, None)
    if args.mixedwm38 and os.path.exists(args.mixedwm38) and not args.demo:
        try:
            MIX = load_mixedwm38(args.mixedwm38)
        except Exception as ex:
            log(f"   MixedWM38 could not be read ({type(ex).__name__}: {ex})")
    R = {"seeds": {}, "notes": data.notes, "version": VERSION,
         "definition": {"WMLOT_K": WMLOT_K, "WMLOT_MIN_W": WMLOT_MIN_W, "split": "70/15/15 per lot"}}
    methods = CLASSIC + NEURAL; yb = data.ybin; L = len(yb)
    for si, seed in enumerate(args.seeds):
        rng = np.random.default_rng(1000 + seed); perm = rng.permutation(L)
        tr, cal, te = perm[:int(.70 * L)], perm[int(.70 * L):int(.85 * L)], perm[int(.85 * L):]
        fit, va = train_test_split(tr, test_size=0.10, stratify=yb[tr], random_state=seed)
        # map-encoder pretraining: wafer pattern labels of TRAINING lots only
        trw = np.isin(data.W["li"].values, tr); mi = data.W["map_i"].values[trw]
        mi = np.random.default_rng(seed).permutation(mi)[:30000]; ft = data.ftype_w[mi]
        cls = sorted(set(ft)); data.pre_maps = data.maps[mi]; data.pre_y = np.array([cls.index(f) for f in ft])
        data.mix_maps, data.mix_y = MIX
        opts = dict(epochs=args.epochs, pre_epochs=args.pre_epochs, max_attr=args.max_attr, menc_epochs=args.menc_epochs)
        log(f"[wmlot] seed {seed}: {len(fit)} train / {len(cal)} calibration / {len(te)} test; training {len(methods)} methods ...")
        out = run_fold(data, fit, va, cal, te, {}, methods, seed=seed, opts=opts)
        ev = eval_single(data, out)
        cause = data.lots["dominant"].values; tab = [i for i in te if yb[i] == 1]
        for m_, zkey, fkey in (("Ours", "z", "z_fit_ab"), ("B3-L1", "feat", "feat_fit_ab"), ("Ours-S", "feat", "feat_fit_ab")):
            r_ = out.get(m_, {})
            if zkey in r_ and fkey in r_:
                pos = {i: k for k, i in enumerate(out["ev"])}
                Zt = np.asarray(r_[zkey])[[pos[i] for i in tab]]
                h_ = retrieval_hits5({"fit": np.asarray(r_[fkey]), "te": Zt}, np.asarray(r_["fit_ab"]), np.asarray(tab), cause)
                ev.setdefault(m_, {})["Retrieval hit@5"] = float(np.mean(list(h_.values()))) if h_ else np.nan
                ev[m_]["_retrieval_hits"] = h_
        R["seeds"][seed] = ev
        log(f"[wmlot] seed {seed} AUROC: " + ", ".join(f"{m} {ev[m]['AUROC']:.3f}" for m in ev if "AUROC" in ev[m]))
        jdump(R, os.path.join(outdir, "raw", "wmlot.json"))
    return R


# ------------------------------------------------------------ B. production lines
def subset_data(data, mask):
    keep = np.where(mask)[0]; mp = {o: n for n, o in enumerate(keep)}
    lots = data.lots.iloc[keep].reset_index(drop=True)
    W = data.W[data.W["li"].isin(keep)].copy(); W["li"] = W["li"].map(mp)
    W = W.drop(columns=["ch_id", "ch_loc"])
    d = LotData(data.name, lots, W, data.groups, data.keys, getattr(data, "ctx_all", data.ctx_cols), data.types, maps=data.maps,
                route=None if data.route is None else {k: v[keep] for k, v in data.route.items()}, mapfeat=data.mapfeat, notes=data.notes)
    d.ctx_cols = list(data.ctx_cols)
    return d


def with_labels(data, ynew):
    d = subset_data(data, np.ones(len(data.lots), bool)); d.lots["y"] = ynew; return d


def cv_production(data, args, methods_classic, methods_deep, R=None, R_deep=None, cfg=None, opts=None, tag=""):
    R = args.repeat if R is None else R; R_deep = args.repeat_deep if R_deep is None else R_deep
    yb = data.ybin; L = len(yb); nb = int(yb.sum())
    if nb < 3 or (L - nb) < 10:
        log(f"   {data.name}{tag}: too few abnormal/normal lots ({nb}/{L - nb})"); return None
    reps = []
    for r in range(R):
        meth = methods_classic + (methods_deep if r < R_deep else [])
        skf = StratifiedKFold(min(5, nb), shuffle=True, random_state=1000 + r)
        st = defaultdict(lambda: defaultdict(dict))
        for k, (tr, te) in enumerate(skf.split(np.zeros(L), yb)):
            fit, va, cal = split_train(tr, yb, seed=r * 100 + k)
            log(f"   {data.name}{tag} repetition {r + 1}/{R} fold {k + 1}: {len(fit)} train / {len(te)} test, {len(meth)} methods")
            o = run_fold(data, fit, va, cal, te, cfg or {}, meth, seed=r * 100 + k, opts=dict(opts or {}, epochs=args.epochs, pre_epochs=args.pre_epochs_prod))
            for m in meth:
                rr = o.get(m, {})
                if "score" not in rr:
                    continue
                s = np.asarray(rr["score"]); sc = s[:len(cal)][yb[cal] == 0]; stt = s[len(cal):]
                for e_ in (0.01, 0.05):
                    tau = conformal_tau(sc, e_)
                    for i, v in zip(te, stt):
                        st[m][f"alarm{e_}"][i] = bool(v > tau)
                    nt = yb[te] == 0
                    st[m][f"farfold{e_}"][k] = float((stt[nt] > tau).mean()) if nt.any() else np.nan
                st[m]["ncal"][k] = int(len(sc))
                for i, v, pv in zip(te, stt, conformal_pvalues(sc, stt)):
                    st[m]["score"][i] = float(v); st[m]["pval"][i] = float(pv)
                for key in ("w_attr", "a_s_att", "phi_c", "ch_rank"):
                    if key in rr:
                        st[m][key].update(rr[key])
                if "gates" in rr:
                    for i, g in zip(te, np.asarray(rr["gates"])[len(cal):]):
                        st[m]["gates"][i] = g
                if "coef" in rr:
                    st[m]["coef"][k] = rr["coef"]
        reps.append({m: {k: dict(v) for k, v in d.items()} for m, d in st.items()})
    return reps


def summarize_cv(data, reps):
    yb = data.ybin; T = truth_matrix(data); S = {}
    if not reps:
        return S
    allm = sorted({m for rp in reps for m in rp})
    for m in allm:
        au, pr, rc, aup = [], [], [], []
        for rp in reps:
            if m not in rp:
                continue
            idx = np.array(sorted(rp[m]["score"])); s = np.array([rp[m]["score"][i] for i in idx])
            au.append(auroc(yb[idx], s)); pr.append(prauc(yb[idx], s)); rc.append(recall_at_fpr(yb[idx], s))
            if rp[m].get("pval"):
                aup.append(auroc(yb[idx], -np.array([rp[m]["pval"][i] for i in idx])))
        rp0 = next(rp for rp in reps if m in rp); idx = np.array(sorted(rp0[m]["score"])); s0 = np.array([rp0[m]["score"][i] for i in idx])
        ci = [float(np.percentile(au, 2.5)), float(np.percentile(au, 97.5))] if len(au) >= 3 else boot_ci(yb[idx], s0)
        # NOTE: "CI" = 2.5th/97.5th percentile of the AUROC over CV repetitions (variation of the partition), NOT a
        # confidence interval of the AUROC; use the DeLong interval of delong_table for uncertainty.
        e = {"AUROC": float(np.nanmean(au)), "CI": ci, "PR-AUC": float(np.nanmean(pr)), "R@5%": float(np.nanmean(rc)), "n_rep": len(au),
             "abnormal": int(yb[idx].sum()), "normal": int((yb[idx] == 0).sum()),
             "AUROC_pval": float(np.nanmean(aup)) if aup else np.nan}
        for e_ in (0.01, 0.05):
            # FAR and recall averaged over ALL CV repetitions (formerly the first one only); per-fold spread is stored
            fr, rc_ = [], []
            for rp in reps:
                if m not in rp or f"alarm{e_}" not in rp[m]:
                    continue
                ii = np.array(sorted(rp[m][f"alarm{e_}"])); al = np.array([rp[m][f"alarm{e_}"][i] for i in ii])
                fr.append(float(al[yb[ii] == 0].mean())); rc_.append(float(al[yb[ii] == 1].mean()))
            e[f"FAR@{e_}"] = float(np.mean(fr)); e[f"recall@{e_}"] = float(np.mean(rc_)); e[f"FAR_reps@{e_}"] = fr
            e[f"FAR_folds@{e_}"] = [v for rp in reps if m in rp for v in rp[m].get(f"farfold{e_}", {}).values()]
            if len(fr) >= 3:
                e[f"FAR_CI@{e_}"] = [float(np.percentile(fr, 2.5)), float(np.percentile(fr, 97.5))]
            else:
                al = np.array([rp0[m][f"alarm{e_}"][i] for i in idx])
                e[f"FAR_CI@{e_}"] = boot_ci(1 - yb[idx], al.astype(float), fn=lambda y_, a_: float(a_[y_ == 1].mean()))
        if "ncal" in rp0[m]:
            e["n_cal_median"] = float(np.median(list(rp0[m]["ncal"].values())))
        if "w_attr" in rp0[m]:
            e["Wafer attr. AUROC"], e["n_attr_lot"] = wafer_attr_auc(rp0[m]["w_attr"], T, [i for i in idx if yb[i] == 1])
        if "a_s_att" in rp0[m]:
            e["Wafer attr. AUROC (attention)"], _ = wafer_attr_auc(rp0[m]["a_s_att"], T, [i for i in idx if yb[i] == 1])
        if "gates" in rp0[m]:
            e["gates_mean"] = np.mean([rp0[m]["gates"][i] for i in idx if i in rp0[m]["gates"]], 0).tolist()
        e["_idx"] = idx.tolist(); e["_s0"] = s0.tolist()
        S[m] = e
    return S


def delong_vs_best(data, S, reps, target="Ours"):
    """Paired DeLong test: target method vs strongest baseline (not Ours/Ours-S; first repetition that contains both)."""
    if not reps or target not in S:
        return None
    base = {m: v["AUROC"] for m, v in S.items() if m not in ("Ours", "Ours-S")}
    if not base:
        return None
    bm = max(base, key=base.get); rp0 = next((r for r in reps if target in r and bm in r), None)
    if rp0 is None:
        return None
    idx = np.array(sorted(set(rp0[target]["score"]) & set(rp0[bm]["score"])))
    a1, a2, z, p = delong_paired(data.ybin[idx], [rp0[target]["score"][i] for i in idx], [rp0[bm]["score"][i] for i in idx])
    return {"strongest_baseline": bm, "AUROC_ours": a1, "AUROC_base": a2, "z": z, "p": p}


def within_tool(data, S, B=2000):
    """Within-tool AUROC (Eq. 14: tools with >= 5 normal and >= 2 abnormal lots, weighted by abnormal lots) and the
    pooled AUROC of the same CV repetition, for every method, with a stratified bootstrap test of the drop.
    'within_all_mixed' is the sensitivity version that counts every tool with both classes."""
    out = {}; tl_all = data.lots["tool"].astype(str).values
    for m_, e_ in S.items():
        idx_ = np.array(e_["_idx"]); s_ = np.array(e_["_s0"]); yb_ = data.ybin[idx_]; tl_ = tl_all[idx_]
        w, n_ab = within_tool_auroc(yb_, s_, tl_)
        w_all, n_all = within_tool_auroc(yb_, s_, tl_, min_normal=1, min_abnormal=1)
        r = {"within": w, "pooled": e_["AUROC"], "pooled_rep1": auroc(yb_, s_), "abnormal_in_tools_used": n_ab,
             "within_all_mixed": w_all, "abnormal_in_mixed_tools": n_all}
        if np.isfinite(w):
            r.update(within_drop_bootstrap(yb_, s_, tl_, B=B))
        out[m_] = r
    return out


def delong_table(data, S, reps):
    """Appendix Table A2: AUROC with DeLong CI of every detector on the first CV repetition and the paired DeLong test
    against the strongest baseline (highest mean AUROC over repetitions), Holm-adjusted within the data set."""
    if not reps or not S:
        return None
    base = {m: v["AUROC"] for m, v in S.items() if m not in ("Ours", "Ours-S") and np.isfinite(v["AUROC"])}
    if not base:
        return None
    bm = max(base, key=base.get); rp0 = next((r for r in reps if bm in r), None)
    if rp0 is None:
        return None
    rows, ps = {}, []
    for m in S:
        rr = next((r for r in reps if m in r and bm in r), None)
        if rr is None:
            continue
        idx = np.array(sorted(set(rr[m]["score"]) & set(rr[bm]["score"]))); y = data.ybin[idx]
        s1 = [rr[m]["score"][i] for i in idx]; s2 = [rr[bm]["score"][i] for i in idx]
        a, lo, hi = auroc_ci_delong(y, s1); e = {"AUROC_rep1": a, "lo": lo, "hi": hi}
        if m != bm:
            _, _, z, p = delong_paired(y, s1, s2); e.update(z=z, p=p); ps.append((m, p))
        rows[m] = e
    for (m, _), ph in zip(ps, holm([p for _, p in ps])):
        rows[m]["p_holm"] = float(ph)
    return {"strongest_baseline": bm, "rows": rows}


def exp_production(data, args, outdir):
    log(f"[{data.name}] {json.dumps(data.notes, default=str)}")
    out = {"notes": data.notes, "version": VERSION}
    keep = CTX_PROD.get(data.name, [])
    data.ctx_cols = [c for c in data.ctx_all if c in keep]
    out["context_used"] = data.ctx_cols
    log(f"   context as model input: {data.ctx_cols or 'none'} (context-only tests still use {data.ctx_all})")
    deep = [m for m in NEURAL if HAS_TORCH]
    reps = cv_production(data, args, ["B1", "B2", "B3-L1", "B3-RF", "B4", "B5", "Ours-S"], deep)
    _LAST_REPS.extend(reps or [])
    out["cv"] = summarize_cv(data, reps)
    if out["cv"]:
        # alarms per tool per week (Ours, eps 5 %) if lot times are available
        t = pd.to_datetime(data.lots["time"]) if "time" in data.lots else None
        if t is not None and t.notna().mean() >= .8 and "Ours" in out["cv"]:
            rp0 = next(r for r in reps if "Ours" in r); al = sum(rp0["Ours"]["alarm0.05"].values())
            weeks = max(1.0, (t.max() - t.min()).days / 7); out["alarms_per_tool_week"] = al / (data.lots["tool"].nunique() * weeks)
        out["delong"] = delong_vs_best(data, out["cv"], reps)
        out["delong_S"] = delong_vs_best(data, out["cv"], reps, target="Ours-S")
        out["within_tool"] = within_tool(data, out["cv"])        # within-tool AUROC, all methods
        out["delong_table"] = delong_table(data, out["cv"], reps)
        nb_all, nn_all = int(data.ybin.sum()), int((data.ybin == 0).sum())
        out["power"] = {"n_abnormal": nb_all, "n_normal": nn_all, "min_detectable_difference": min_detectable_difference(nb_all, nn_all),
                        "assumption": "Hanley-McNeil SE at AUROC 0.85, correlation 0.5, alpha 0.05 two-sided, power 0.8"}
        # per-lot labels, tools and months (confidential JSON; needed to rebuild every table from the stored scores)
        out["lots"] = {"y": data.ybin.tolist(), "tool": data.lots["tool"].astype(str).tolist(),
                       "month": data.lots["month"].astype(str).tolist() if "month" in data.lots else None}
        # label permutations (B3-L1, one 5-fold CV per permutation)
        rng = np.random.default_rng(7); nul = []
        for k in range(args.null):
            yp = rng.permutation(data.y.copy())
            r_ = cv_production(with_labels(data, yp), args, ["B3-L1"], [], R=1, R_deep=0, tag=f" null{k + 1}")
            s_ = summarize_cv(with_labels(data, yp), r_); nul.append(s_.get("B3-L1", {}).get("AUROC", np.nan))
        if nul:
            obs = out["cv"].get("B3-L1", {}).get("AUROC", np.nan)
            out["null_B3L1"] = {"mean": float(np.nanmean(nul)), "p95": float(np.nanpercentile(nul, 95)), "p": float((1 + np.sum(np.array(nul) >= obs)) / (len(nul) + 1))}
        # v20: permutation WITHIN tool. The global permutation also destroys the tool-label association, so a model that
        # only recognizes tools passes it; permuting within tool keeps that association and tests the signal beyond tool.
        tl_ = data.lots["tool"].astype(str).values; rng = np.random.default_rng(8); nul_w = []
        for k in range(args.null):
            yp = data.y.copy()
            for t_ in np.unique(tl_):
                ii = np.where(tl_ == t_)[0]; yp[ii] = rng.permutation(yp[ii])
            r_ = cv_production(with_labels(data, yp), args, ["B3-L1"], [], R=1, R_deep=0, tag=f" null-within-tool{k + 1}")
            s_ = summarize_cv(with_labels(data, yp), r_); nul_w.append(s_.get("B3-L1", {}).get("AUROC", np.nan))
        if nul_w:
            obs = out["cv"].get("B3-L1", {}).get("AUROC", np.nan)
            out["null_B3L1_within_tool"] = {"mean": float(np.nanmean(nul_w)), "p95": float(np.nanpercentile(nul_w, 95)),
                                            "p": float((1 + np.sum(np.array(nul_w) >= obs)) / (len(nul_w) + 1)),
                                            "note": "labels permuted within each tool; observed = mean AUROC over CV repetitions"}
    # subset of coaters with both classes ("same coaters", main Plant B result)
    if "subset_same" in data.lots and data.lots["subset_same"].sum() < len(data.lots):
        sd = subset_data(data, data.lots["subset_same"].values)
        reps_s = cv_production(sd, args, ["B1", "B2", "B3-L1", "B3-RF", "B4", "B5", "Ours-S"], deep, tag=" same")
        out["same"] = summarize_cv(sd, reps_s)
        out["delong_same"] = delong_vs_best(sd, out["same"], reps_s)
        out["delong_same_S"] = delong_vs_best(sd, out["same"], reps_s, target="Ours-S")
        out["within_tool_same"] = within_tool(sd, out["same"])
        out["delong_table_same"] = delong_table(sd, out["same"], reps_s)
        nb_s, nn_s = int(sd.ybin.sum()), int((sd.ybin == 0).sum())
        out["power_same"] = {"n_abnormal": nb_s, "n_normal": nn_s, "min_detectable_difference": min_detectable_difference(nb_s, nn_s)}
    # unseen tool
    out["loto"] = {}
    for tl, g in data.lots.groupby("tool"):
        nb_, ng_ = int(g["y"].gt(0).sum()), int((g["y"] == 0).sum())
        if nb_ < 2 or ng_ < 5:
            continue
        te = g.index.values; tr = np.setdiff1d(np.arange(len(data.lots)), te)
        if data.ybin[tr].sum() < 2:
            continue
        fit, va, cal = split_train(tr, data.ybin, seed=11)
        o = run_fold(data, fit, va, cal, te, {}, ["B2", "B3-L1", "B3-RF", "B4", "B9", "Ours-S"] + (["Ours"] if HAS_TORCH else []), seed=11,
                     opts=dict(epochs=args.epochs, pre_epochs=args.pre_epochs_prod, attr=False))
        out["loto"][str(tl)] = {"bad": nb_, "good": ng_, **{m: auroc(data.ybin[te], np.asarray(o[m]["score"])[len(cal):]) for m in o if isinstance(o[m], dict) and "score" in o[m]}}
        out.setdefault("loto_stats", {})[str(tl)] = {m: auroc_hanley(v, nb_, ng_) for m, v in out["loto"][str(tl)].items() if m not in ("bad", "good")}
    # shortcut / leakage tests
    if HAS_TORCH and not args.skip_shortcut:
        sh = {}
        cc = data.ctx_cols; data.ctx_cols = list(data.ctx_all)
        ctx = cv_production(data, args, ["CTX"], [], R=1, R_deep=0, tag=" context")
        sh["Context-only model"] = summarize_cv(data, ctx).get("CTX", {}).get("AUROC", np.nan)
        for c_ in data.ctx_all:
            data.ctx_cols = [c_]
            r1 = cv_production(data, args, ["CTX"], [], R=1, R_deep=0, tag=f" context {c_}")
            sh[f"Context-only: {c_}"] = summarize_cv(data, r1).get("CTX", {}).get("AUROC", np.nan)
        data.ctx_cols = cc
        rng = np.random.default_rng(3)
        # within-tool AUROC: a large drop means the score mostly separates TOOLS, not abnormal lots within a tool
        if out.get("within_tool"):
            wt = out["within_tool"].get("Ours")
            sh["Within-tool AUROC"] = (wt["within"] - wt["pooled_rep1"]) if wt and np.isfinite(wt["within"]) else np.nan
        mon = data.lots["month"].astype(str).values; yp = data.y.copy()
        for mth in np.unique(mon):
            ii = np.where(mon == mth)[0]; yp[ii] = rng.permutation(yp[ii])
        ls = cv_production(with_labels(data, yp), args, [], ["Ours"], R=1, R_deep=1, tag=" labels shuffled", opts={"attr": False})
        sh["Labels shuffled within month"] = summarize_cv(with_labels(data, yp), ls).get("Ours", {}).get("AUROC", np.nan)
        dl = subset_data(data, np.ones(len(data.lots), bool))
        flip = rng.random(len(dl.lots)) < .02          # strong leak (AUROC of the leak alone ~0.98)
        dl.lots["leak"] = np.where(flip, 1 - dl.ybin, dl.ybin).astype(str); dl.ctx_cols = dl.ctx_cols + ["leak"]
        dl.ctx_vocab["leak"] = {"0": 1, "1": 2}
        lk = cv_production(dl, args, [], ["Ours"], R=1, R_deep=1, tag=" deliberate leak", opts={"attr": False})
        sh["Deliberate leak"] = summarize_cv(dl, lk).get("Ours", {}).get("AUROC", np.nan)
        if "Ours" in out.get("cv", {}):
            e = out["cv"]["Ours"]; idx = np.array(e["_idx"]); s0 = np.array(e["_s0"]); yb = data.ybin[idx]
            key = data.lots["tool"].astype(str).values + "|" + mon
            okk = set(key[np.where(data.ybin == 1)[0]])
            msk = (yb == 1) | np.isin(key[idx], list(okk))
            # v20: compare with the pooled AUROC of the SAME repetition (s0), not with the mean over repetitions
            sh["Normal lots matched by tool and month"] = auroc(yb[msk], s0[msk]) - auroc(yb, s0) if msk.sum() > 5 else np.nan
        out["shortcut"] = sh
        n1_, n0_ = int(data.ybin.sum()), int((data.ybin == 0).sum())
        out["shortcut_stats"] = {k: auroc_hanley(v, n1_, n0_) for k, v in sh.items()
                                 if k not in ("Within-tool AUROC", "Normal lots matched by tool and month")}
    # ablations (production)
    if HAS_TORCH and args.ablation_prod > 0:
        out["ablation"] = {}
        base = out["cv"].get("Ours", {}).get("AUROC", np.nan) if out.get("cv") else np.nan
        for nm, ch in ABLATIONS.items():
            if nm.startswith("Cartesian") or (nm == "w/o route branch" and data.route is None):
                out["ablation"][nm] = None; continue
            rr = cv_production(data, args, [], ["Ours"], R=args.ablation_prod, R_deep=args.ablation_prod, cfg=ch, opts={"attr": False}, tag=f" ablation {nm[:25]}")
            a_ = summarize_cv(data, rr).get("Ours", {}).get("AUROC", np.nan); out["ablation"][nm] = a_ - base
    jdump(out, os.path.join(outdir, "raw", f"{data.name}.json"))
    return out


# ============================================================ report: tables, figures, hypotheses, paper text
def md_table(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |" for r in df.values])


def mean_sd(vals, d=3, pm=True):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "—"
    return f"{np.mean(v):.{d}f} ± {np.std(v):.{d}f}" if pm and len(v) > 1 else f"{np.mean(v):.{d}f}"


def make_report(outdir, datasets):
    global ETAS
    os.makedirs(outdir, exist_ok=True)
    rd = lambda n: json.load(open(os.path.join(outdir, "raw", n), encoding="utf-8")) if os.path.exists(os.path.join(outdir, "raw", n)) else None
    LB, PB, DS, FA = rd("lotbench.json"), rd("plantb.json"), rd("diesaw.json"), rd("faba.json")
    if LB and LB.get("etas"):
        ETAS = [float(e) for e in LB["etas"]]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
    BLUE, RED, GREY, ORANGE, GREEN = "#0070C0", "#C00000", "#7F7F7F", "#ED7D31", "#548235"
    md = [f"# REPORT OF THE CORE EXPERIMENTS (created {time.strftime('%Y-%m-%d %H:%M')}, core_experiments.py {VERSION})"]
    if DEMO:
        md += ["", "> **DEMO / QUICK TEST — THESE NUMBERS MUST NOT BE USED IN THE PAPER.** Run again without --quick-test / --demo."]
    if LB and LB.get("version") not in ("v17", "v18", "v19", "v20"):
        md.append("> WARNING: raw/lotbench.json was produced by an old version (route/chamber bug); run --parts lotbench again.")
    for nm, D in (("plantb", PB), ("diesaw", DS), ("faba", FA)):
        if D and "within_tool" not in D:
            md.append(f"> WARNING: raw/{nm}.json was produced before the conformal-score fix; run --parts {nm} again.")
    H = {}
    order = ["B1", "B2", "B3-L1", "B3-RF", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "Ours", "Ours-S"]
    # ---------------- LotBench detection and per-scenario tables
    if LB:
        seeds = list(LB["seeds"].values())
        def col(m, k):
            return [s.get(m, {}).get(k, np.nan) for s in seeds]
        rows = []
        for m in order:
            if not any(m in s for s in seeds):
                continue
            rows.append([NAMES[m]] + [mean_sd(col(m, k)) if k in ("AUROC", "PR-AUC", "R@5%") else mean_sd(col(m, k), 2, pm=(m == "Ours"))
                                      for k in ("AUROC", "PR-AUC", "R@5%", "Macro-F1", "Wafer attr. AUROC", "Chamber hit@1", "Step hit@1")])
        T6 = pd.DataFrame(rows, columns=["Method", "AUROC", "PR-AUC", "R@5%", "Macro-F1", "Wafer attr. AUROC", "Chamber hit@1", "Step hit@1"])
        T6.to_csv(os.path.join(outdir, "table_lotbench_detection.csv"), index=False)
        md += ["", f"## LotBench-811K detection (mean ± sd over {len(seeds)} seeds; eta = {ETAS})", md_table(T6)]
        rows7 = []
        for m in ["B1", "B2", "B3-L1", "B3-RF", "B4", "B8", "B9", "B10", "Ours", "Ours-S"]:
            if not any(m in s for s in seeds):
                continue
            ps = [np.nanmean([s.get(m, {}).get("per_scen", {}).get(k, np.nan) for s in seeds]) for k in SCEN]
            rows7.append([m] + [fmt(v, 3) for v in ps] + [fmt(100 * np.nanmean(col(m, "FAR_N_rot3")), 1)])
        T7 = pd.DataFrame(rows7, columns=["Method", "F1 chamber", "F2 post-PM", "F3 drift", "F4 upstream", "F5 material", "F6 intermittent", "FAR on N (%)"])
        T7.to_csv(os.path.join(outdir, "table_lotbench_per_scenario.csv"), index=False)
        md += ["", "## LotBench-811K per scenario", md_table(T7)]
        # AUROC per eta level (all scenarios)
        rowsE = []
        for m in order:
            if not any(m in s for s in seeds):
                continue
            rowsE.append([NAMES[m]] + [fmt(np.nanmean([s.get(m, {}).get("eta_all", {}).get(f"{e}", np.nan) for s in seeds])) for e in ETAS])
        TE_ = pd.DataFrame(rowsE, columns=["Method"] + [f"η = {e}" for e in ETAS])
        TE_.to_csv(os.path.join(outdir, "table_lotbench_per_eta.csv"), index=False)
        md += ["", "## AUROC per injection strength η (all scenarios F1–F6 pooled vs normal lots)", md_table(TE_)]
        ours = np.nanmean(col("Ours", "AUROC"))
        bases = {m: np.nanmean(col(m, "AUROC")) for m in order if m not in ("Ours", "Ours-S") and any(m in s for s in seeds)}
        oS = np.nanmean(col("Ours-S", "AUROC"))
        H["H1_OursS"] = {"Ours-S": oS, "diff_vs_strongest": oS - max(bases.values()) if bases else np.nan,
                         "chamber_hit1_F1": np.nanmean(col("Ours-S", "Chamber hit@1 F1")), "step_hit1": np.nanmean(col("Ours-S", "Step hit@1")),
                         "wafer_attr": np.nanmean(col("Ours-S", "Wafer attr. AUROC"))}
        bm = max(bases, key=lambda k: np.nan_to_num(bases[k], nan=-1)) if bases else None
        H["H1"] = {"Ours": ours, "strongest": bm, "base": bases.get(bm), "diff": ours - bases.get(bm, np.nan), "supported": bool(ours - bases.get(bm, np.nan) >= .05)}
        h5 = np.nanmean(col("Ours", "Chamber hit@1 F1")); H["H5"] = {"hit1_F1": h5, "supported": bool(h5 >= .8)}
        H["LotBench_extra"] = {"slot_pattern_macroF1": np.nanmean(col("Ours", "Slot-pattern macro-F1")),
                               "FAR_eps5": np.nanmean(col("Ours", "realized_FAR")), "recall_eps5": np.nanmean(col("Ours", "recall_eps")),
                               "retrieval_hit5_ours": np.nanmean(col("Ours", "Retrieval hit@5")), "retrieval_hit5_B3": np.nanmean(col("B3-L1", "Retrieval hit@5")),
                               "wafer_attr_attention": np.nanmean(col("Ours", "Wafer attr. AUROC (attention)")),
                               "step_hit1_ours": np.nanmean(col("Ours", "Step hit@1")), "step_hit1_B10": np.nanmean(col("B10", "Step hit@1")),
                               "n_param": seeds[0].get("Ours", {}).get("n_param"), "ms_per_lot": np.nanmean(col("Ours", "ms_per_lot"))}
        # LotBench ROC and eta figure
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), dpi=200, gridspec_kw={"width_ratios": [1, 1.6]})
        roc = LB.get("roc", {})
        for m, cl in (("Ours", RED), (bm, BLUE), ("B2", GREY), ("B4", ORANGE)):
            if m in roc:
                f_, t_, _ = roc_curve(roc[m]["y"], roc[m]["s"]); ax[0].plot(f_, t_, c=cl, lw=1.5, label=f"{NAMES[m]} ({auroc(roc[m]['y'], roc[m]['s']):.3f})")
        ax[0].plot([0, 1], [0, 1], "k:", lw=.7); ax[0].set_xlabel("false-alarm rate"); ax[0].set_ylabel("recall"); ax[0].legend(fontsize=6, frameon=False, loc="lower right")
        ax[0].set_title("(a) ROC, test split, first seed", fontsize=9)
        cols_ = plt.cm.tab10(np.arange(6))
        for k, sc in enumerate(SCEN):
            v = [np.nanmean([s.get("Ours", {}).get("per_eta", {}).get(f"{sc}_{e}", np.nan) for s in seeds]) for e in ETAS]
            ax[1].plot(ETAS, v, "-o", c=cols_[k], ms=3, label=sc)
            bsc = {m: np.nanmean([s.get(m, {}).get("per_scen", {}).get(sc, np.nan) for s in seeds]) for m in bases}
            bb = max(bsc, key=lambda q: np.nan_to_num(bsc[q], nan=-1))
            vb = [np.nanmean([s.get(bb, {}).get("per_eta", {}).get(f"{sc}_{e}", np.nan) for s in seeds]) for e in ETAS]
            ax[1].plot(ETAS, vb, ":", c=cols_[k], lw=1)
        ax[1].set_xlabel("injection strength η"); ax[1].set_ylabel("AUROC"); ax[1].legend(fontsize=6, ncol=3, frameon=False)
        ax[1].set_title("(b) AUROC vs η (solid: Ours; dotted: strongest baseline per scenario)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_lotbench_roc_eta.png")); plt.close(fig)
    # ---------------- WM-811K lot-task table
    WL = rd("wmlot.json")
    if WL and WL.get("seeds"):
        sw = list(WL["seeds"].values())
        rowsW = []
        for m in order:
            if not any(m in s for s in sw):
                continue
            c_ = lambda k: [s.get(m, {}).get(k, np.nan) for s in sw]
            rowsW.append([NAMES[m], mean_sd(c_("AUROC")), mean_sd(c_("PR-AUC")), mean_sd(c_("R@5%")), mean_sd(c_("Wafer attr. AUROC"), 2),
                          mean_sd(c_("Retrieval hit@5"), 2), fmt(100 * np.nanmean(c_("realized_FAR")), 1)])
        TW = pd.DataFrame(rowsW, columns=["Method", "AUROC", "PR-AUC", "R@5%", "Wafer attr. AUROC", "Retrieval hit@5", "FAR at ε = 5% (%)"])
        TW.to_csv(os.path.join(outdir, "table_wmlot_detection.csv"), index=False)
        md += ["", f"## WM-811K lot task with real defects (mean ± sd over {len(sw)} split seeds; definition: {WL.get('definition')})", md_table(TW),
               f"- data: {WL.get('notes')}"]
        au = {m: np.nanmean([s.get(m, {}).get("AUROC", np.nan) for s in sw]) for m in order if any(m in s for s in sw)}
        bw = {m: v for m, v in au.items() if m not in ("Ours", "Ours-S") and np.isfinite(v)}
        if bw:
            bm_ = max(bw, key=bw.get)
            H["WMLOT"] = {"strongest_baseline": bm_, "AUROC_base": bw[bm_], "Ours": au.get("Ours"), "Ours-S": au.get("Ours-S"),
                          "diff_Ours": (au.get("Ours", np.nan) - bw[bm_]), "diff_Ours-S": (au.get("Ours-S", np.nan) - bw[bm_])}
    # ---------------- production table (+ supplement B5-B8)
    def cv(D, m, k="AUROC", key="cv"):
        return (D or {}).get(key, {}).get(m, {}).get(k, np.nan)
    def ci_txt(D, m, key="cv"):
        e = (D or {}).get(key, {}).get(m)
        if not e:
            return "TBD"
        return f"{e['AUROC']:.3f} [{e['CI'][0]:.2f}, {e['CI'][1]:.2f}]"
    def loto(D, m):
        if not D or not D.get("loto"):
            return "—"
        return " / ".join(fmt(v.get(m, np.nan)) for v in D["loto"].values())
    rows8 = []
    for m in ["B1", "B2", "B3-L1", "B3-RF", "B4", "B9", "Ours", "Ours-S"]:
        rows8.append([NAMES[m], ci_txt(PB, m, "same"), ci_txt(PB, m), ci_txt(DS, m), loto(PB, m), ci_txt(FA, m),
                      f"{fmt(cv(PB, m, 'R@5%', 'same'), 2)} / {fmt(cv(PB, m, 'R@5%'), 2)}"])
    T8 = pd.DataFrame(rows8, columns=["Method", "Plant B coat/develop, same coaters (main)", "Plant B coat/develop, all tools (shortcut-prone)",
                                      "Wafer saw (≥ 3 chipped)", "Plant B unseen tool (" + " / ".join((PB or {}).get("loto", {}).keys()) + ")",
                                      "Fab A excursions", "R@5% (same coaters / all tools)"])
    T8.to_csv(os.path.join(outdir, "table_production_detection.csv"), index=False)
    md += ["", "## Production lines (AUROC [percentile interval over CV repetitions])", md_table(T8)]
    for nm, D in (("Plant B coat/develop", PB),):
        if D and D.get("notes", {}).get("bad_per_tool"):
            md.append(f"- {nm}, abnormal lots per tool: {D['notes']['bad_per_tool']} of {D['notes'].get('tool')}")
    sup = [[NAMES[m], ci_txt(PB, m, "same"), ci_txt(DS, m), ci_txt(FA, m)] for m in ["B5", "B6", "B7", "B8"]]
    TS_ = pd.DataFrame(sup, columns=["Method", "Plant B same coaters", "Wafer saw", "Fab A"])
    TS_.to_csv(os.path.join(outdir, "table_supplement_B5_B8.csv"), index=False)
    md += ["", "## Supplementary table B5–B8", md_table(TS_)]
    DLs = []
    for nm, D, k in (("Plant B coat/develop (same coaters)", PB, "delong_same"), ("Plant B wafer saw", DS, "delong"), ("Fab A", FA, "delong")):
        d_ = (D or {}).get(k)
        if d_:
            DLs.append((nm, d_))
    if DLs:
        hp = holm([d_["p"] for _, d_ in DLs]); H["H2"] = {n: {**d_, "p_holm": float(p)} for (n, d_), p in zip(DLs, hp)}
        H["H2"]["supported"] = bool(all(v["p_holm"] < .05 and v["AUROC_ours"] > v["AUROC_base"] for k, v in H["H2"].items() if isinstance(v, dict)))
    for nm, D in (("Plant B coat/develop", PB), ("Wafer saw", DS), ("Fab A", FA)):
        if D and D.get("null_B3L1"):
            md.append(f"- Label permutation {nm} (B3-L1): random AUROC {D['null_B3L1']['mean']:.2f} (p95 {D['null_B3L1']['p95']:.2f}), p = {D['null_B3L1']['p']:.3f}")
    # ---------------- ablation table
    rows9 = []
    lbo = None
    if LB:
        seeds = list(LB["seeds"].values()); n_ab = len(next(iter(LB["ablation"].values()), []))
        lbo = np.nanmean([s.get("Ours", {}).get("AUROC", np.nan) for s in seeds[:max(1, n_ab)]])
    rows9.append(["Full model", "reference", fmt(lbo), fmt(cv(PB, "Ours")), fmt(cv(DS, "Ours")), fmt(cv(FA, "Ours"))])
    exp_txt = {"w/o chamber-relative normalization (2)": "large drop on production", "w/o peer contrast (3)": "drop on F1 / Plant B",
               "Flat MIL instead of hierarchy (7)–(8)": "large drop", "w/o slot and chamber embeddings": "drop on F3 / F6",
               "Cartesian CNN instead of polar encoder (4)–(5)": "drop on rotated defects", "w/o slot Transformer (zslot)": "drop on F3 / trends",
               "w/o route branch": "drop on F2 / F4 / Fab A", "w/o context branch": "drop on F5, small elsewhere",
               "Concatenation instead of gated fusion": "small drop", "w/o pretraining (Stage 1)": "drop, largest on production",
               "w/o ℒord (within-chamber shuffle)": "drop on F3", "Cross-entropy instead of focal loss": "small drop in PR-AUC"}
    drops = {}
    for nm in ABLATIONS:
        lb = (np.nanmean(LB["ablation"][nm]) - lbo) if LB and nm in LB.get("ablation", {}) else np.nan
        vals = [lb] + [((D or {}).get("ablation", {}) or {}).get(nm) for D in (PB, DS, FA)]
        rows9.append([nm, exp_txt[nm]] + [("—" if v is None else f"{v:+.3f}" if np.isfinite(v) else "—") for v in vals])
        drops[nm] = [v for v in vals[1:] if v is not None and np.isfinite(v)]
    T9 = pd.DataFrame(rows9, columns=["Variant", "Expectation", "LotBench-811K", "Plant B coat/develop", "Wafer saw", "Fab A"])
    T9.to_csv(os.path.join(outdir, "table_ablation.csv"), index=False); md += ["", "## Ablations (ΔAUROC)", md_table(T9)]
    prod_mean = {k: np.mean(v) for k, v in drops.items() if v}
    if prod_mean:
        worst = sorted(prod_mean, key=prod_mean.get)[:2]
        H["H3"] = {"two_largest_drops_production": worst, "supported": bool(any(w.startswith(("w/o chamber-relative", "Flat MIL")) for w in worst))}
    # ---------------- localization and attribution table
    def att(D, m, k):
        return fmt((D or {}).get("cv", {}).get(m, {}).get(k, np.nan), 2)
    def natt(D, m):
        n = (D or {}).get("cv", {}).get(m, {}).get("n_attr_lot")
        return f" (n = {n} lot)" if n else ""
    rows10 = [["Chamber hit@3 vs maintenance/root-cause log", "— (no log)", "— (no log)", "— (no log)", "—"],
              ["Wafer attribution vs per-wafer outcome (occlusion)", "— (lot labels)", att(DS, "Ours", "Wafer attr. AUROC") + natt(DS, "Ours"),
               att(FA, "Ours", "Wafer attr. AUROC") + natt(FA, "Ours"),
               f"B9: {att(DS, 'B9', 'Wafer attr. AUROC')} / {att(FA, 'B9', 'Wafer attr. AUROC')}; B4: {att(DS, 'B4', 'Wafer attr. AUROC')} / {att(FA, 'B4', 'Wafer attr. AUROC')}"],
              ["Wafer attribution with attention weights instead of occlusion", "—", att(DS, "Ours", "Wafer attr. AUROC (attention)"), att(FA, "Ours", "Wafer attr. AUROC (attention)"), "—"],
              ["Engineer usefulness rating", "— (not collected)", "—", "—", "—"],
              ["Similar-lot retrieval: same cause in top-5 (LotBench-811K)", "—", "—", "—",
               f"Ours {fmt(H.get('LotBench_extra', {}).get('retrieval_hit5_ours', np.nan), 2)}; B3 {fmt(H.get('LotBench_extra', {}).get('retrieval_hit5_B3', np.nan), 2)}"]]
    T10 = pd.DataFrame(rows10, columns=["Measure", "Plant B coat/develop", "Plant B wafer saw", "Fab A", "Reference"])
    T10.to_csv(os.path.join(outdir, "table_localization.csv"), index=False); md += ["", "## Localization and attribution", md_table(T10)]
    # ---------------- calibration table
    rows11 = []; fars = []
    for nm, D, key in (("Plant B coat/develop (same coaters)", PB, "same"), ("Plant B wafer saw", DS, "cv"), ("Fab A", FA, "cv")):
        e = (D or {}).get(key, {}).get("Ours")
        if not e:
            continue
        nc = e.get("n_cal_median") or 0
        for eps in (0.05, 0.01):
            note = "" if nc and (nc + 1) * eps >= 1 else " (not attainable: ε < 1/(n_cal+1))"
            rows11.append([nm, f"{int(eps * 100)}%{note}", f"{100 * e[f'FAR@{eps}']:.1f}%", fmt(e[f"recall@{eps}"], 2),
                           fmt(D.get("alarms_per_tool_week", np.nan), 2) if eps == .05 and key == "cv" else "—", f"{nc:.0f}"])
            if not note:
                fars.append((eps, e[f"FAR@{eps}"]))
    if LB:
        rows11.append(["LotBench-811K", "5%", f"{100 * H['LotBench_extra']['FAR_eps5']:.1f}%", fmt(H["LotBench_extra"]["recall_eps5"], 2), "—", "≈ 2800"])
        fars.append((.05, H["LotBench_extra"]["FAR_eps5"]))
    T11 = pd.DataFrame(rows11, columns=["Data", "ε", "Realized FAR", "Recall", "Alarms / tool / week", "n calibration lots (median)"])
    T11.to_csv(os.path.join(outdir, "table_calibration.csv"), index=False); md += ["", "## Calibration", md_table(T11)]
    if fars:
        H["H6"] = {"max_abs_dev_pp": float(max(abs(f - e) for e, f in fars) * 100), "supported": bool(max(abs(f - e) for e, f in fars) <= .015)}
    # ---------------- shortcut and confounding tests table
    def sh(D, k):
        v = (D or {}).get("shortcut", {}).get(k, np.nan)
        return "—" if v is None or not np.isfinite(v) else (f"Δ {v:+.3f}" if k in ("Within-tool AUROC", "Normal lots matched by tool and month") else f"{v:.2f}")
    keys = [("Context-only model", "shortcut through product / recipe / time", "≤ 0.65"),
            ("Within-tool AUROC", "confounding by tool identity", "Δ ≤ 0.03"),
            ("Labels shuffled within month", "null distribution", "≈ 0.50"), ("Deliberate leak", "sensitivity of the protocol", "large increase"),
            ("Normal lots matched by tool and month", "confounding by tool / time", "Δ ≤ 0.03")]
    T12 = pd.DataFrame([[k, p, sh(PB, k), sh(DS, k), sh(FA, k), e] for k, p, e in keys],
                       columns=["Test", "Purpose", "Plant B coat/develop", "Plant B wafer saw", "Fab A", "Expectation"])
    T12.to_csv(os.path.join(outdir, "table_shortcut_tests.csv"), index=False); md += ["", "## Shortcut and confounding tests", md_table(T12)]
    for nm, D in (("Plant B coat/develop", PB), ("Plant B wafer saw", DS), ("Fab A", FA)):
        ex = {k: v for k, v in ((D or {}).get("shortcut", {}) or {}).items() if k.startswith("Context-only: ")}
        if ex:
            md.append(f"- {nm}, context-only per variable: " + "; ".join(f"{k[14:]} {v:.2f}" for k, v in ex.items()))
        if D and D.get("context_used") is not None:
            md.append(f"- {nm}: context used by the models = {D['context_used'] or 'none'}")
    md += ["", "## Within-tool AUROC (AUROC within tools, weighted by abnormal lots; pooled = ordinary AUROC)"]
    for nm, D in (("Plant B coat/develop", PB), ("Plant B wafer saw", DS), ("Fab A", FA)):
        wt = (D or {}).get("within_tool")
        if wt:
            md.append(f"- {nm}: " + "; ".join(f"{m} {fmt(v['within'])} (pooled {fmt(v['pooled'])})" for m, v in wt.items()))
    co = [(D or {}).get("shortcut", {}).get("Context-only model", np.nan) for D in (PB, DS, FA)]
    co = [v for v in co if v is not None and np.isfinite(v)]
    if co:
        H["H4"] = {"context_only_max": max(co), "supported": bool(max(co) <= .65)}
    # ---------------- gate figure: LotBench per scenario; production only if LotBench is absent
    G = {}
    if LB:
        gs = LB["seeds"][next(iter(LB["seeds"]))].get("Ours", {}).get("gates_scen", {})
        for k, v in gs.items():
            G[f"LB {k}"] = v
    else:
        for nm, D in (("Plant B\ncoat/develop", PB), ("Plant B\nwafer saw", DS), ("Fab A", FA)):
            g = (D or {}).get("cv", {}).get("Ours", {}).get("gates_mean")
            if g:
                G[nm] = g
    if G:
        fig, ax = plt.subplots(figsize=(8, 3), dpi=200); x = np.arange(len(G)); bot = np.zeros(len(G))
        lab_ = ["wafer πw", "route πr", "context πc"]
        G3 = [g if len(g) == 3 else ([g[0], 0.0, g[1]] if len(g) == 2 else [g[0], 0.0, 0.0]) for g in G.values()]
        for j in range(3):
            v = np.array([g[j] for g in G3])
            ax.bar(x, v, bottom=bot, color=[BLUE, ORANGE, GREEN][j], label=lab_[j], width=.65); bot += v
        ax.set_xticks(x); ax.set_xticklabels(list(G), fontsize=7); ax.set_ylabel("mean gate weight"); ax.legend(fontsize=7, frameon=False, ncol=3)
        ax.set_title("Gate weights per LotBench-811K scenario", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_gates.png")); plt.close(fig)
        md += ["", "## Gate weights", "```", json.dumps({k.replace(chr(10), ' '): [round(x, 3) for x in v] for k, v in G.items()}, ensure_ascii=False), "```"]
    # ---------------- attribution figure (two lots; data with per-wafer labels first)
    cases = []
    for nm, D in (("Plant B wafer saw", DS), ("Fab A", FA), ("Plant B coat/develop", PB)):
        ex = (D or {}).get("example")
        if ex:
            cases.append((nm, ex))
    if cases:
        fig, axs = plt.subplots(1, len(cases[:2]), figsize=(10, 3), dpi=200, squeeze=False)
        for ax, (nm, ex) in zip(axs[0], cases[:2]):
            ph = np.array(ex["phi"], float); ch = np.array(ex["ch"]); tr_ = np.array(ex["truth"], float)
            cc = plt.cm.Set2(pd.factorize(ch)[0] % 8)
            bars = ax.bar(np.arange(1, 26), np.nan_to_num(ph), color=cc)
            for b_, t_ in zip(bars, tr_):
                if t_ == 1:
                    b_.set_edgecolor(RED); b_.set_linewidth(1.5)
            ax.set_xlabel("slot"); ax.set_ylabel("occlusion φs"); ax.set_title(f"{nm}: lot {ex['lot']} (colour = chamber; red = affected wafer)", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_attribution_example.png")); plt.close(fig)
    # ---------------- calibration figure
    rowsF = [(nm, D[key]["Ours"]) for nm, D, key in (("Plant B (same coaters)", PB, "same"), ("Wafer saw", DS, "cv"), ("Fab A", FA, "cv"))
             if D and "Ours" in D.get(key, {})]
    if rowsF:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), dpi=200)
        for k, (nm, e) in enumerate(rowsF):
            for eps, mk in ((0.01, "o"), (0.05, "s")):
                v = e[f"FAR@{eps}"]; ci = e[f"FAR_CI@{eps}"]
                ax[0].errorbar(100 * eps + (k - 1) * .12, 100 * v, yerr=[[100 * max(0, v - ci[0])], [100 * max(0, ci[1] - v)]], fmt=mk, c=[BLUE, ORANGE, GREEN][k % 3],
                               label=nm if eps == .05 else None, capsize=2)
        ax[0].plot([0, 6], [0, 6], "k:", lw=.7); ax[0].set_xlabel("target ε (%)"); ax[0].set_ylabel("realized false-alarm rate (%)"); ax[0].legend(fontsize=7, frameon=False)
        ax[0].set_title("(a) realized vs target false-alarm rate", fontsize=9)
        ts = (datasets or {}).get("diesaw")
        if ts is not None and DS and "Ours" in DS.get("cv", {}):
            e = DS["cv"]["Ours"]; idx = np.array(e["_idx"]); s = np.array(e["_s0"]); t = pd.to_datetime(ts.lots["time"]).values[idx]
            ok = ~pd.isna(t); yb = ts.ybin[idx]
            ax[1].scatter(t[ok][yb[ok] == 0], s[ok][yb[ok] == 0], s=4, c=GREY, label="normal")
            ax[1].scatter(t[ok][yb[ok] == 1], s[ok][yb[ok] == 1], s=10, c=RED, label="abnormal")
            ax[1].axhline(np.quantile(s[yb == 0], .95), c="k", ls="--", lw=.8, label="≈ threshold ε = 5%")
            ax[1].legend(fontsize=7, frameon=False); ax[1].set_ylabel("abnormality score s(q)")
            ax[1].set_title("(b) scores over time (wafer saw, cross-validated)", fontsize=9)
            ax[1].tick_params(axis="x", rotation=30, labelsize=7)
        else:
            ax[1].axis("off"); ax[1].text(.5, .5, "run diesaw + report in one process\nfor panel (b)", ha="center", va="center", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_calibration.png")); plt.close(fig)
    # ---------------- summary and text
    jdump(H, os.path.join(outdir, "hypotheses.json"))
    md += ["", "## Hypotheses (pre-specified expectations)", "```", json.dumps(H, indent=1, default=str), "```"]
    md += ["", "## Ready-to-paste sentences (check the numbers first)"]
    if LB and "H1" in H:
        md.append(f"- RQ1: On LotBench-811K the proposed encoder reaches AUROC {H['H1']['Ours']:.3f} against {H['H1']['base']:.3f} for the strongest baseline "
                  f"({NAMES.get(H['H1']['strongest'], H['H1']['strongest'])}); difference {H['H1']['diff']:+.3f}. "
                  f"Chamber hit@1 on F1 lots is {H['H5']['hit1_F1']:.2f}.")
    if PB and "Ours" in PB.get("same", {}):
        e = PB["same"]["Ours"]; dl = PB.get("delong_same") or {}
        md.append(f"- RQ2 Plant B same coaters (main Plant B result): Ours AUROC {e['AUROC']:.3f} [{e['CI'][0]:.2f}, {e['CI'][1]:.2f}], "
                  f"{e['abnormal']} abnormal / {e['normal']} normal lots; {NAMES.get(dl.get('strongest_baseline'), '')} "
                  f"{dl.get('AUROC_base', np.nan):.3f} (paired DeLong p = {dl.get('p', np.nan):.3g}).")
    for nm, D in (("the wafer-saw line", DS), ("Fab A", FA)):
        if D and "Ours" in D.get("cv", {}):
            dl = D.get("delong") or {}
            md.append(f"- RQ2 {nm}: Ours AUROC {D['cv']['Ours']['AUROC']:.3f} [{D['cv']['Ours']['CI'][0]:.2f}, {D['cv']['Ours']['CI'][1]:.2f}] vs "
                      f"{NAMES.get(dl.get('strongest_baseline'), '')} {dl.get('AUROC_base', np.nan):.3f} (paired DeLong p = {dl.get('p', np.nan):.3g}).")
    for nm, D, k in (("Plant B same coaters", PB, "delong_same_S"), ("Plant B all tools", PB, "delong_S"), ("Wafer saw", DS, "delong_S"), ("Fab A", FA, "delong_S")):
        dl = (D or {}).get(k)
        if dl:
            md.append(f"- Ours-S {nm}: AUROC {dl['AUROC_ours']:.3f} vs {NAMES.get(dl['strongest_baseline'])} {dl['AUROC_base']:.3f} "
                      f"(paired DeLong p = {dl['p']:.3g}, first repetition)")
    # ---------------- statistics reported in the paper (paired tests, intervals, power)
    md += ["", "## Statistics reported in the paper"]
    if LB:
        seeds_ = list(LB["seeds"].values())
        def paired_hits(m1, m2, key):
            a_, b_ = [], []
            for sd_ in seeds_:
                h1_, h2_ = sd_.get(m1, {}).get(key) or {}, sd_.get(m2, {}).get(key) or {}
                for i in sorted(set(h1_) & set(h2_)):
                    a_.append(h1_[i]); b_.append(h2_[i])
            return a_, b_
        a_, b_ = paired_hits("Ours", "B1", "_chamber_hits_F1")
        if a_:
            r1 = [s_.get("Ours", {}).get("Chamber hit@1 F1", np.nan) for s_ in seeds_]; r2 = [s_.get("B1", {}).get("Chamber hit@1 F1", np.nan) for s_ in seeds_]
            tt = stats.ttest_rel(r1, r2) if len(seeds_) >= 2 else None
            H["stats_chamber_hit_F1_SALE_vs_B1"] = {"mcnemar": mcnemar_exact(a_, b_), "diff_mean_over_seeds": float(np.nanmean(np.array(r1) - np.array(r2))),
                                                   "paired_t_over_seeds_p": float(tt.pvalue) if tt is not None else np.nan, "n_seeds": len(seeds_)}
        a_, b_ = paired_hits("Ours", "B3-L1", "_retrieval_hits")
        if a_:
            H["stats_retrieval_SALE_vs_B3"] = mcnemar_exact(a_, b_)
        wb = [s_.get("Ours", {}).get("Wafer attr. bootstrap") for s_ in seeds_ if s_.get("Ours", {}).get("Wafer attr. bootstrap")]
        if wb:
            H["stats_wafer_attribution_SALE"] = wb
    rowsS = []
    for nm, D in (("Plant B coat/develop (all tools)", PB), ("Plant B wafer saw", DS), ("Fab A PECVD", FA)):
        if not D:
            continue
        for k, e in (D.get("shortcut_stats") or {}).items():
            rowsS.append([nm, k, fmt(e["AUROC"]), f"[{fmt(e['lo'], 2)}, {fmt(e['hi'], 2)}]", fmt(e["p_vs_0.5"])])
        for m, e in (D.get("within_tool") or {}).items():
            if np.isfinite(e.get("within", np.nan)):
                rowsS.append([nm, f"within-tool drop {m}", f"{fmt(e['pooled_rep1'])} -> {fmt(e['within'])}",
                              f"[{fmt(e.get('drop_lo'), 3)}, {fmt(e.get('drop_hi'), 3)}]", fmt(e.get("drop_p"))])
        if D.get("power"):
            rowsS.append([nm, "minimal detectable paired AUROC difference", fmt(D["power"]["min_detectable_difference"]), "", ""])
        if D.get("power_same"):
            rowsS.append([nm.replace("all tools", "same coaters"), "minimal detectable paired AUROC difference", fmt(D["power_same"]["min_detectable_difference"]), "", ""])
    if rowsS:
        TSt = pd.DataFrame(rowsS, columns=["Data", "Statistic", "Value", "95% interval", "p"])
        TSt.to_csv(os.path.join(outdir, "table_statistics_production.csv"), index=False); md.append(md_table(TSt))
    rowsD = []
    for nm, D, key in (("Plant B coat/develop, same coaters", PB, "delong_table_same"), ("Plant B wafer saw", DS, "delong_table"),
                       ("Fab A PECVD", FA, "delong_table"), ("Plant B coat/develop, all tools", PB, "delong_table")):
        T_ = (D or {}).get(key)
        if T_:
            for m, e in T_["rows"].items():
                rowsD.append([nm, NAMES.get(m, m), f"{fmt(e['AUROC_rep1'])} [{fmt(e['lo'], 2)}, {fmt(e['hi'], 2)}]",
                              "reference" if m == T_["strongest_baseline"] else f"{fmt(e.get('z'), 2)}; {fmt(e.get('p_holm'))}"])
    if rowsD:
        TD = pd.DataFrame(rowsD, columns=["Data", "Method", "AUROC [95% CI], repetition 1", "z; Holm p vs strongest baseline"])
        TD.to_csv(os.path.join(outdir, "table_delong_production.csv"), index=False)
        md += ["", "## Paired DeLong tests on the first CV repetition (Appendix Table A2)", md_table(TD)]
    jdump(H, os.path.join(outdir, "hypotheses.json"))
    md += ["", "## Data notes", "```", json.dumps({k: (v or {}).get("notes") for k, v in (("plantb", PB), ("diesaw", DS), ("faba", FA))}, indent=1, default=str), "```"]
    if DEMO:
        md.append("\n**DEMO — do not use in the paper.**")
    open(os.path.join(outdir, "REPORT_core.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    log(f"Report: {os.path.join(outdir, 'REPORT_core.md')}")


def add_example(data, res):
    """Example lot for the attribution figure: a reasonably large abnormal lot with mixed per-wafer outcomes (if available)
    and the most concentrated attribution."""
    try:
        rp = res["_reps"][0]["Ours"]
    except Exception:
        return
    nw = data.W.groupby("li").size(); min_w = min(15.0, float(nw.quantile(.75)))
    best, bv = None, -1
    for i, phi in rp.get("w_attr", {}).items():
        ph = np.nan_to_num(np.asarray(phi, float))
        if ph.max() <= 0:
            continue
        Wi = data.W[data.W["li"] == i]; wb = Wi["wbad"].values.astype(float)
        if len(Wi) < min_w:
            continue
        if np.isfinite(wb).any() and not (0 < np.nansum(wb) < np.isfinite(wb).sum()):
            continue
        v = ph.max() / (np.abs(ph).sum() + 1e-9)
        if v > bv:
            best, bv = i, v
    if best is None:
        return
    W = data.W[data.W["li"] == best]; ch = np.array([""] * S_MAX, dtype=object); tr = np.full(S_MAX, np.nan)
    ch[W["slot"].values - 1] = W["chamber"].astype(str).values; tr[W["slot"].values - 1] = W["wbad"].values
    res["example"] = {"lot": data.lots.loc[best, "alias"], "phi": np.asarray(rp["w_attr"][best], float).tolist(), "ch": ch.tolist(), "truth": tr.tolist()}


# ============================================================ main
def main():
    global DEMO
    ap = argparse.ArgumentParser(description="All experiments of the JIM paper (core)")
    ap.add_argument("--folder", default=PARTNER_DATA_DIR); ap.add_argument("--label", default=LABEL_FINAL)
    ap.add_argument("--wm811k", default=WM811K_PKL); ap.add_argument("--mixedwm38", default=MIXEDWM38_NPZ)
    ap.add_argument("--public-data", default=PUBLIC_DATA_DIR)
    ap.add_argument("--out", default=OUT); ap.add_argument("--diesaw-zip", default=None)
    ap.add_argument("--parts", default="lotbench,wmlot,plantb,diesaw,faba,report")
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--n-lots", type=int, default=20000)
    ap.add_argument("--eta", default=None, help="LotBench injection strengths, e.g. 0.05,0.1,0.2,0.3 (default: ETA_LEVELS)")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--pre-epochs", type=int, default=100)
    ap.add_argument("--pre-epochs-prod", type=int, default=100); ap.add_argument("--menc-epochs", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=20, help="CV repetitions (5-fold) for classical methods")
    ap.add_argument("--repeat-deep", type=int, default=5, help="CV repetitions for neural models (including SALE)")
    ap.add_argument("--null", type=int, default=200)
    ap.add_argument("--ablation-seeds", type=int, default=0, help="number of first LotBench seeds that get the ablations (0 = none)")
    ap.add_argument("--ablation-prod", type=int, default=0, help="CV repetitions for production ablations (0 = none)")
    ap.add_argument("--max-attr", type=int, default=400)
    ap.add_argument("--k-chipped", type=int, default=3); ap.add_argument("--skip-shortcut", action="store_true")
    ap.add_argument("--demo", action="store_true"); ap.add_argument("--demo-lots", type=int, default=600)
    ap.add_argument("--quick-test", action="store_true", help="check the code in a few minutes (DEMO results)")
    a = ap.parse_args()
    if a.eta:
        ETAS[:] = [float(x) for x in a.eta.split(",")]
    if a.quick_test:
        a.demo = True; a.seeds = "0"; a.n_lots = 150; a.demo_lots = 150; a.epochs = 2; a.pre_epochs = 1; a.pre_epochs_prod = 1
        a.repeat = 2; a.repeat_deep = 1; a.null = 3; a.ablation_seeds = 1; a.ablation_prod = 0; a.max_attr = 10; a.menc_epochs = 1
    DEMO = a.demo or a.quick_test
    if not a.wm811k or not os.path.exists(a.wm811k):
        a.wm811k = find_public(a.public_data, ["LSWMD.pkl"]) or a.wm811k
    if not a.mixedwm38 or not os.path.exists(a.mixedwm38):
        a.mixedwm38 = find_public(a.public_data, ["MixedWM38.npz", "Wafer_Map_Datasets.npz"]) or a.mixedwm38
    a.seeds = [int(s) for s in a.seeds.split(",")]
    out = a.out + ("_DEMO" if DEMO else ""); os.makedirs(os.path.join(out, "raw"), exist_ok=True)
    log(f"core_experiments.py {VERSION}; torch: {HAS_TORCH} ({DEV}); output: {out}")
    log(f"WM-811K: {a.wm811k or '-'}; MixedWM38: {a.mixedwm38 or '-'}")
    if HAS_TORCH and DEV == "cpu" and "lotbench" in a.parts and not DEMO:
        log("WARNING: CPU build of PyTorch. The full LotBench run will be very slow; install the CUDA build (see top of file).")
    if not HAS_TORCH:
        log("WARNING: PyTorch is not installed -> SALE and B6-B9 are skipped.")
    parts = {p.strip() for p in a.parts.split(",")}; t0 = time.time(); datasets = {}
    if "lotbench" in parts:
        exp_lotbench(a, out)
    if "wmlot" in parts:
        exp_wmlot(a, out)
    for nm, loader in (("plantb", lambda: load_plantb(a.folder, a.label) if os.path.exists(a.label) else None),
                       ("diesaw", lambda: load_diesaw(a.folder, a.diesaw_zip, k_bad=a.k_chipped)),
                       ("faba", lambda: load_faba(a.folder))):
        if nm not in parts:
            continue
        try:
            d = loader()
        except SystemExit as e:
            log(f"[{nm}] skipped: {e}"); d = None
        except Exception as e:
            log(f"[{nm}] could not be read: {type(e).__name__}: {e}"); d = None
        if d is None:
            log(f"[{nm}] data not found, skipped"); continue
        datasets[nm] = d
        exp_production_wrap(d, a, out)
    if "report" in parts:
        if "diesaw" not in datasets and os.path.exists(os.path.join(out, "raw", "diesaw.json")):
            try:
                datasets["diesaw"] = load_diesaw(a.folder, a.diesaw_zip, k_bad=a.k_chipped)     # for the time panel of the calibration figure
            except Exception:
                pass
        make_report(out, datasets)
    logname = "log_" + "_".join(sorted(parts)) + ".txt"
    open(os.path.join(out, logname), "w", encoding="utf-8").write("\n".join(LOG))
    log(f"DONE in {(time.time() - t0) / 60:.1f} min. Output folder: {out} (no raw data).")


def exp_production_wrap(d, a, out):
    """Run exp_production and store an attribution example for the attribution figure."""
    _LAST_REPS.clear()
    res = exp_production(d, a, out)
    res["_reps"] = list(_LAST_REPS)
    add_example(d, res)
    res.pop("_reps", None)
    jdump(res, os.path.join(out, "raw", f"{d.name}.json"))
    return res


_LAST_REPS = []


if __name__ == "__main__":
    main()
