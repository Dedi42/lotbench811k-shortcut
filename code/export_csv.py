#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
export_csv.py - complete CSV package (input, process, output) for the lot-level abnormality study.

Layout (root = --package, default ./csv_package):
  public/                              <- the ONLY folder that may be uploaded (Zenodo/GitHub)
    00_config/<run>/                   frozen configuration and SHA-256 fingerprints of the scripts
    01_input/<dataset>/                data as given to the detectors (LotBench-811K per seed, WM-811K lot task)
    02_process/<dataset>/              roles per seed, wafer features after normalization (Eq. 2), lot features,
                                       learning-curve subsamples
    03_output/<run>/<dataset>/         score of every detector for every lot and seed, per-seed metrics, summaries
    03_output/<run>/core_tables/       tables of core_experiments.py that concern public data only
    03_output/paper_tables/            Tables 6-8 of the paper (manuscript v41 numbering) and sensitivity tables,
                                       rebuilt from the scores
    FILES.csv                          one line per file with a description
  confidential/                        <- NEVER upload
    for_editor/...                     production lines with anonymized identifiers (on request of the editor,
                                       after the partners' approval)
    alias_keys_INTERNAL_ONLY/          original -> alias keys (stay in the lab, not for the editor)

Rule: everything that is not explicitly public goes to confidential (default deny).
Detector names in the CSV files follow the paper: Ours -> SALE, Ours-S -> SALE-S.
Results of earlier (Indonesian) versions of the scripts are read as well; their keys are translated on loading.

Stand-alone use (no models are run):
  python export_csv.py --json-run results_core --json-run results_extended --json-run results_revision
  python export_csv.py --input-public          # inputs and process files of LotBench-811K (seeds 10-19) and WM-811K
  python export_csv.py --input-production      # inputs and CV folds of the production lines (anonymized, confidential)
revision_experiments.py calls this module automatically.
"""
import argparse, glob, json, os, re, shutil, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GZ = {"method": "gzip", "mtime": 0}                         # mtime 0 -> identical .gz files across runs (stable hashes)
DET = {"Ours": "SALE", "Ours-S": "SALE-S"}

# JSON stems -> public data set (everything else is confidential)
PUBLIC_STEMS = {"lotbench": "lotbench", "lotbench_scores": "lotbench", "wmlot": "wmlot", "wmlot_scores": "wmlot",
                "bench_lotbench": "lotbench", "bench_lotbench_scores": "lotbench", "bench_wmlot": "wmlot",
                "bench_wmlot_scores": "wmlot", "ablation": "ablation", "ablation_multi": "ablation", "learning_curve": "learning_curve"}
# production stems -> alias name (used in confidential/for_editor) and alias key store
PROD_NAME = {"plantb": "plantB_coat_develop", "samecoater": "plantB_coat_develop_same_coaters", "same_coaters": "plantB_coat_develop_same_coaters",
             "diesaw": "plantB_wafer_saw", "faba": "fabA_pecvd", "demo": "synthetic_demo"}
PROD_STORE = {"plantb": "plantB_coat_develop", "samecoater": "plantB_coat_develop", "same_coaters": "plantB_coat_develop",
              "diesaw": "plantB_wafer_saw", "faba": "fabA_pecvd", "demo": "synthetic_demo"}
# paper table files (manuscript v41): Table 6 = detection on both public data sets, Table 7 = per scenario,
# Table 8 = MCE design vs confirmatory seeds, Table S3 = ablations (Online Resource 1)
PAPER_FILES = {"det_lotbench": ("table06a_lotbench_detection", "Table 6 (left, LotBench-811K)"),
               "det_wmlot": ("table06b_wmlot_detection", "Table 6 (right, WM-811K lot task)"),
               "scen": ("table07_lotbench_per_scenario", "Table 7"),
               "conf": ("table08_mce_design_vs_confirmatory", "Table 8"),
               "abl": ("tableS3_ablation_seed10", "Online Resource 1, Table S3 (seed 10 only)"),
               "sens": ("sensitivity_lotbench_without_F5_F4", "sensitivity analysis (not in manuscript v41): LotBench-811K without F5 and without F4+F5")}
# file names and keys of earlier (Indonesian) versions of the scripts
LEGACY_STEMS = {"ablasi": "ablation", "ablasi_multi": "ablation_multi", "kurva": "learning_curve", "samecoater": "same_coaters"}
LEGACY_FILES = {"PRAREGISTRASI.json": "PREREGISTRATION.json", "PRAREGISTRASI_REVISI.json": "PREREGISTRATION_REVISION.json",
                "LAPORAN.md": "REPORT.md", "LAPORAN_REVISI.md": "REPORT_REVISION.md", "ISI_JIM.md": "REPORT_core.md",
                "log_lanjut.txt": "log_extended.txt", "log_revisi.txt": "log_revision.txt"}
LEGACY_KEYS = {"versi": "version", "catatan": "note", "definisi": "definition", "fitur": "features", "konfigurasi": "configuration",
               "waktu": "time", "bagian": "parts", "uji_cepat": "quick_test", "ulangan": "repeats", "latih": "train", "uji": "test",
               "abnormal_uji": "abnormal_test", "normal_uji": "normal_test", "bulan_uji": "test_months", "bulan": "monthly",
               "metrik_utama": "primary_metric", "aturan_klaim": "claim_rule", "frac_kalibrasi_normal": "frac_calibration_normal",
               "frac_latih": "frac_train", "bases_produksi": "bases_production", "batch_besar": "large_batch",
               "rasio_unlabeled": "unlabeled_ratio", "min_abnormal_latih": "min_abnormal_train", "min_normal_latih": "min_normal_train",
               "ablasi_seeds": "ablation_seeds", "sha256_skrip": "sha256_script", "sha256_jalankan_jim": "sha256_core",
               "chamber_utama": "main_chamber", "jenis_chamber": "chamber_types", "lot_dengan_tanggal": "lots_with_date",
               "wafer_dibuang_slot": "wafers_dropped_slot", "ambang_chipping": "chipping_threshold", "indikator_dibuang": "indicators_dropped",
               "lot_campuran": "mixed_lots", "ekskursi": "excursions", "wafer_berpola": "patterned_wafers",
               "pola_dominan_lot_abnormal": "dominant_pattern_abnormal_lots",
               "Flat MIL instead of hierarchy (5)–(6)": "Flat MIL instead of hierarchy (7)–(8)",
               "Cartesian CNN instead of polar encoder (4a)–(4b)": "Cartesian CNN instead of polar encoder (4)–(5)"}
SKIP_ATTR = {"maps", "pre_maps", "pre_y", "mix_maps", "mix_y", "ftype_w"}
ID_HINT = ("lot", "tool", "chamber", "ch_", "eqp", "equip", "recipe", "product", "device", "blade", "batch", "wafer_id",
           "carrier", "foup", "operator", "host", "material")
TIME_HINT = ("time", "date")
SAFE_COLS = {"li", "lot_index", "wafer_row", "slot", "y", "ybin", "split", "scenario", "subset_same", "s", "pos", "order"}
PREFIX = {"tool": "T", "lot": "L", "chamber": "C", "month": "M", "feature_name": "F"}


def det(m):
    return DET.get(m, m)


def normalize_legacy(obj):
    """Translate keys of earlier result files (recursively); values are kept."""
    if isinstance(obj, dict):
        return {LEGACY_KEYS.get(k, k) if isinstance(k, str) else k: normalize_legacy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_legacy(v) for v in obj]
    return obj


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return normalize_legacy(json.load(f))


# ============================================================ package and file register
class Package:
    def __init__(self, root):
        self.root = os.path.abspath(root); self.notes = {}

    def pub(self, *p):
        return os.path.join(self.root, "public", *p)

    def conf(self, *p):
        return os.path.join(self.root, "confidential", "for_editor", *p)

    def keydir(self, store):
        return os.path.join(self.root, "confidential", "alias_keys_INTERNAL_ONLY", store)

    def write(self, df, path, desc, index=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=index, float_format="%.8g", encoding="utf-8", compression=GZ if path.endswith(".gz") else None)
        self.note(path, desc, len(df))

    def write_text(self, text, path, desc):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self.note(path, desc, None)

    def note(self, path, desc, rows):
        path = os.path.abspath(path)
        for area in (os.path.join(self.root, "public"), os.path.join(self.root, "confidential", "for_editor")):
            if path.startswith(area + os.sep):
                self.notes.setdefault(area, {})[os.path.relpath(path, area).replace(os.sep, "/")] = (desc, rows)

    def flush(self):
        for area, d in self.notes.items():
            p = os.path.join(area, "FILES.csv")
            old = pd.read_csv(p, dtype=str, keep_default_na=False) if os.path.exists(p) else pd.DataFrame(columns=["file", "rows", "description"])
            new = pd.DataFrame([{"file": k, "rows": "" if r is None else str(r), "description": v} for k, (v, r) in d.items()])
            out = pd.concat([old[~old["file"].isin(new["file"])], new]).sort_values("file")
            out = out[[os.path.exists(os.path.join(area, f)) for f in out["file"]]]
            out.to_csv(p, index=False, encoding="utf-8")
        self.notes = {}


# ============================================================ aliases (anonymization of production data)
class Alias:
    """Original -> alias keys that stay consistent across runs (stored in alias_keys_INTERNAL_ONLY/<store>/<column>.csv)."""
    def __init__(self, pkg, store):
        self.dir = pkg.keydir(store); self.maps = {}; self.features = set()

    def _load(self, col):
        if col not in self.maps:
            p = os.path.join(self.dir, f"{_safe(col)}.csv")
            self.maps[col] = dict(pd.read_csv(p, dtype=str, keep_default_na=False).values) if os.path.exists(p) else {}
        return self.maps[col]

    def _save(self, col):
        os.makedirs(self.dir, exist_ok=True)
        pd.DataFrame(sorted(self.maps[col].items(), key=lambda kv: kv[1]), columns=["original", "alias"]).to_csv(
            os.path.join(self.dir, f"{_safe(col)}.csv"), index=False, encoding="utf-8")

    def map(self, col, values, prefix=None):
        m = self._load(col); vals = pd.Series(list(values), dtype=object).astype(str)
        na = {"nan", "None", "NaT", "NA", "", "<NA>"}
        new = sorted(set(vals) - set(m) - na)
        if new:
            p = prefix or PREFIX.get(col) or (re.sub(r"[^A-Za-z]", "", str(col))[:3].upper() or "X") + "_"
            for i, v in enumerate(new, len(m) + 1):
                m[v] = f"{p}{i:04d}"
            self._save(col)
        return vals.map(lambda v: m.get(v, "" if v in na else v))

    def feature(self, c):
        return self.map("feature_name", [c], "F").iloc[0]

    def all_pairs(self):
        pairs = {}
        for p in glob.glob(os.path.join(self.dir, "*.csv")):
            for o, a in pd.read_csv(p, dtype=str, keep_default_na=False).values:
                if len(o) >= 3 and not o.isdigit():
                    pairs[o] = a
        return pairs

    def scrub(self, text):
        pairs = self.all_pairs()
        for o in sorted(pairs, key=len, reverse=True):
            text = text.replace(o, pairs[o])
        return text


def _safe(c):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(c))


def scrub_all(pkg, text):
    """Replace every known original (all production lines) by its alias."""
    for store in sorted(set(PROD_STORE.values())):
        text = Alias(pkg, store).scrub(text)
    return text


def anonymize_df(df, alias):
    out = {}; t0 = None
    for c in df.columns:
        s = df[c]; lc = str(c).lower()
        if c in alias.features:
            out[alias.feature(c)] = s; continue
        if c in SAFE_COLS:
            out[c] = s
        elif lc == "month":
            out[c] = alias.map("month", s, "M")
        elif pd.api.types.is_datetime64_any_dtype(s) or any(h in lc for h in TIME_HINT):
            t = pd.to_datetime(s, errors="coerce")
            if t.notna().mean() > .5:
                t0 = t.min() if t0 is None else min(t0, t.min())
                out[f"{c}_days_from_start"] = t
            else:
                out[c] = alias.map(str(c), s)
        elif any(h in lc for h in ID_HINT) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            out[c] = alias.map(lc if lc in PREFIX else str(c), s)
        else:
            out[c] = s
    for k, v in out.items():
        if k.endswith("_days_from_start"):
            out[k] = ((v - t0).dt.total_seconds() / 86400).round(4)
    return pd.DataFrame(out)


# ============================================================ input (data-set tables)
def dataset_tables(d):
    """Lot table, wafer table and other tabular attributes of a LotData object (3-D wafer maps are not exported;
    WM-811K maps are referenced by their row in LSWMD.pkl, column map_i)."""
    lots = d.lots.copy(); W = d.W.copy() if isinstance(getattr(d, "W", None), pd.DataFrame) else None
    nL = len(lots); nW = len(W) if W is not None else -1; tabs = {}; meta = {}
    for k, v in vars(d).items():
        if k in ("lots", "W") or k.startswith("_") or k in SKIP_ATTR:
            continue
        if isinstance(v, pd.DataFrame):
            tabs[k] = v
        elif isinstance(v, np.ndarray) and v.ndim == 1 and len(v) in (nL, nW):
            if len(v) == nL and k not in lots:
                lots[k] = v
            elif W is not None and len(v) == nW and k not in W:
                W[k] = v
        elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] in (nL, nW) and v.shape[1] <= 1000:
            tabs[k] = pd.DataFrame(v, columns=[f"{k}_{j:03d}" for j in range(v.shape[1])]).assign(
                **{"li" if v.shape[0] == nL else "wafer_row": np.arange(v.shape[0])})
        elif isinstance(v, dict) and k == "route":
            for rk, rv in v.items():
                rv = np.asarray(rv)
                if rv.ndim >= 2 and rv.shape[0] == nL:
                    tabs[f"route_{rk}"] = pd.DataFrame(rv.reshape(nL, -1), columns=[f"{rk}_{j:02d}" for j in range(int(np.prod(rv.shape[1:])))]).assign(li=np.arange(nL))
        elif isinstance(v, (str, int, float, bool, list, dict, tuple)) or v is None:
            try:
                meta[k] = json.dumps(v, default=str, ensure_ascii=False)
            except Exception:
                meta[k] = str(v)
    return lots, W, tabs, meta


def export_dataset(pkg, d, name, public, prefix="", stem=None, desc=""):
    lots, W, tabs, meta = dataset_tables(d)
    if public:
        base = pkg.pub("01_input", name)
        pkg.write(lots.reset_index().rename(columns={"index": "lot_index"}), os.path.join(base, f"{prefix}lots.csv.gz"),
                  f"{desc}: one row per lot (label y, simulated tool/chamber, scenario, split, WM-811K host lot)")
        if W is not None:
            pkg.write(W, os.path.join(base, f"{prefix}wafers.csv.gz"),
                      f"{desc}: one row per wafer (li = lot_index, slot, chamber, FDC features; map_i = row of LSWMD.pkl)")
        for k, t in tabs.items():
            pkg.write(t, os.path.join(base, f"{prefix}{k}.csv.gz"), f"{desc}: attribute '{k}'")
        pkg.write(pd.DataFrame(sorted(meta.items()), columns=["key", "value_json"]), os.path.join(base, f"{prefix}metadata.csv"),
                  f"{desc}: metadata of the data set (feature groups, notes)")
        return
    al = Alias(pkg, PROD_STORE[stem]); al.features = {c for g in getattr(d, "groups", {}).values() for c in (g if isinstance(g, (list, tuple)) else [])}
    base = pkg.conf("01_input", PROD_NAME[stem])
    pkg.write(anonymize_df(lots.reset_index().rename(columns={"index": "lot_index"}), al), os.path.join(base, "lots.csv.gz"),
              "one row per lot; lot/tool/chamber identifiers and months anonymized, time = days since the first lot")
    if W is not None:
        pkg.write(anonymize_df(W, al), os.path.join(base, "wafers.csv.gz"), "one row per wafer; FDC sensor names anonymized (F0001...)")
    for k, t in tabs.items():
        pkg.write(anonymize_df(t, al), os.path.join(base, f"{_safe(k)}.csv.gz"), f"attribute '{k}' (anonymized)")
    meta = {k: al.scrub(v) for k, v in meta.items()}
    for g, cols in getattr(d, "groups", {}).items():
        if isinstance(cols, (list, tuple)):
            meta[f"groups.{g}"] = json.dumps([al.feature(c) for c in cols])
    pkg.write(pd.DataFrame(sorted(meta.items()), columns=["key", "value_json"]), os.path.join(base, "metadata.csv"), "metadata (anonymized)")


# ============================================================ process (roles, features, CV folds)
def export_bench_input(pkg, J, d, kind, seed, roles):
    """Input and process files of one benchmark seed: data tables, roles, normalized wafer features, lot features."""
    from sklearn.preprocessing import StandardScaler
    if kind == "lotbench":
        if not os.path.exists(pkg.pub("01_input", kind, f"seed{seed}_lots.csv.gz")):
            export_dataset(pkg, d, kind, True, f"seed{seed}_", desc=f"LotBench-811K seed {seed}")
    elif not os.path.exists(pkg.pub("01_input", kind, "lots.csv.gz")):
        export_dataset(pkg, d, kind, True, "", desc="WM-811K lot task")
    sp = pkg.pub("02_process", kind, f"splits_seed{seed}.csv")
    if not os.path.exists(sp):
        rows = [(int(i), r) for r, idx in roles.items() for i in idx]
        pkg.write(pd.DataFrame(rows, columns=["lot_index", "role"]).sort_values("lot_index"), sp,
                  f"seed {seed}: role of every lot (fit = trains the base detectors, blend = trains the MCE meta-learner / "
                  f"validation in the ablations, cal = conformal calibration, test)")
    fp = pkg.pub("02_process", kind, f"lot_features_seed{seed}.csv.gz")
    if os.path.exists(fp):
        return
    yb = d.ybin; fit = np.asarray(roles["fit"])
    Xn = J.crn(d, fit[yb[fit] == 0]); Xp = J.peer_contrast(d, Xn); A = J.lot_arrays(d, Xn, Xp)
    S, sn = J.lot_summary(d, Xn); Sstd = np.clip(StandardScaler().fit(S[fit]).transform(S), -10, 10)
    Z, OH, aw, s_oc = J.struct_features(d, A, fit)
    sn = [str(x) for x in sn] if isinstance(sn, (list, tuple, np.ndarray)) and len(sn) == S.shape[1] else [f"summary_{j:03d}" for j in range(S.shape[1])]
    cols = [f"B3:{c}" for c in sn] + [f"SALE-S:struct_{j:03d}" for j in range(Z.shape[1])] + [f"onehot_{j:03d}" for j in range(OH.shape[1])]
    F = pd.DataFrame(np.hstack([Sstd, Z, OH]), columns=cols); F.insert(0, "lot_index", np.arange(len(F)))
    F["SALE-S:oneclass_term"] = np.asarray(s_oc)
    pkg.write(F, fp, f"seed {seed}: lot features (standardized B3 summary, per-tool SALE-S structure statistics, one-hot route/context), referenced to normal fit lots")
    if isinstance(Xn, np.ndarray) and Xn.ndim == 2:
        pkg.write(pd.DataFrame(Xn, columns=[f"x_tilde_{j:03d}" for j in range(Xn.shape[1])]).assign(wafer_row=np.arange(len(Xn))),
                  pkg.pub("02_process", kind, f"wafer_features_normalized_seed{seed}.csv.gz"),
                  f"seed {seed}: wafer features after chamber-relative normalization (Eq. 2); rows in the order of wafers.csv")


def export_folds(pkg, rows, stem, protocol, desc):
    df = pd.DataFrame(rows, columns=["repetition", "fold", "lot_index", "role"])
    path = pkg.conf("02_process", PROD_NAME[stem], f"cv_folds_{protocol}.csv.gz") if stem in PROD_NAME else pkg.pub("02_process", stem, f"cv_folds_{protocol}.csv.gz")
    pkg.write(df, path, desc)


def prod_fold_rows(J, d, R, seed_base, folds=5, cal_frac=0.15, subset=None):
    """Re-create the CV folds exactly. core_experiments.cv_production: seed_base 1000 (20 repetitions);
    extended_experiments.cv_protocol: seed_base 5000 (10 repetitions). split_train uses seed r*100+k in both."""
    from sklearn.model_selection import StratifiedKFold
    yb = d.ybin; idx = np.arange(len(yb)) if subset is None else np.asarray(subset); nb = int(yb[idx].sum()); rows = []
    for r in range(R):
        skf = StratifiedKFold(min(folds, nb), shuffle=True, random_state=seed_base + r)
        for k, (a, b) in enumerate(skf.split(np.zeros(len(idx)), yb[idx])):
            tr, te = idx[a], idx[b]
            fit, va, cal = J.split_train(tr, yb, seed=r * 100 + k, cal_frac=cal_frac)
            rows += [(r + 1, k + 1, int(i), role) for role, ii in (("fit", fit), ("validation", va), ("cal", cal), ("test", te)) for i in ii]
    return rows


# ============================================================ output from JSON
def _flat(obj, prefix=""):
    rows = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            rows += _flat(v, f"{prefix}/{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            rows += _flat(v, f"{prefix}[{i}]")
    else:
        rows.append((prefix, obj))
    return rows


def _scalar(e):
    return {k: v for k, v in e.items() if not isinstance(v, (dict, list)) and not str(k).startswith("_")}


def _metrics(per_seed):
    rows, scen = [], []
    for s, ms in per_seed.items():
        for m, e in ms.items():
            if not isinstance(e, dict):
                continue
            rows.append({"seed": int(s), "detector": det(m), **_scalar(e)})
            if isinstance(e.get("per_scen"), dict):
                scen.append({"seed": int(s), "detector": det(m), **e["per_scen"]})
    return pd.DataFrame(rows), pd.DataFrame(scen)


def _paired(v, ref, ratio=None):
    from scipy import stats
    v = np.asarray(v, float); ref = np.asarray(ref, float); d = v - ref; n = len(d)
    r = {"n_seeds": n, "mean": v.mean(), "ref_mean": ref.mean(), "diff": d.mean(), "diff_min": d.min(), "diff_max": d.max(), "n_positive": int((d > 0).sum())}
    if n >= 2 and d.std(ddof=1) > 0:
        tt = stats.ttest_rel(v, ref); r.update(t=tt.statistic, df=n - 1, p_t=tt.pvalue)
        if ratio is not None and np.isfinite(ratio):           # corrected resampled t test (Nadeau & Bengio, 2003)
            tc = d.mean() / np.sqrt((1.0 / n + ratio) * d.var(ddof=1)); r.update(t_nadeau_bengio=tc, p_nadeau_bengio=2 * stats.t.sf(abs(tc), n - 1))
    if n >= 6:
        try:
            r["p_wilcoxon"] = stats.wilcoxon(v, ref).pvalue
        except ValueError:
            pass
    return r


def _bench_tables(pkg, B, base, kind, run, paper):
    """Benchmark results of extended_experiments.py / revision_experiments.py (key 'per_seed')."""
    m, sc = _metrics(B["per_seed"])
    pkg.write(m, os.path.join(base, "metrics_per_seed.csv"), f"{kind}: AUROC, PR-AUC, recall and FAR at ε = 5%, exact binomial test of the FAR, DeLong vs B3-RF per seed and detector")
    if len(sc):
        pkg.write(sc, os.path.join(base, "auroc_per_scenario_per_seed.csv"), f"{kind}: AUROC per scenario F1-F6 per seed")
    if isinstance(B.get("summary"), dict) and any(isinstance(v, dict) and "AUROC_mean" in v for v in B["summary"].values()):
        S = pd.DataFrame([{"detector": det(k), **_scalar(v)} for k, v in B["summary"].items()])
        pkg.write(S, os.path.join(base, "summary_over_seeds.csv"), f"{kind}: summary over seeds (mean, sd, difference to B3-RF, t test, Wilcoxon, Holm)")
    w = B.get("mce_weights") or {s: W.get("MCE") for s, W in B.get("weights", {}).items()}
    rows = [{"seed": int(s), "ensemble": "MCE", "base_detector": det(k), "weight": v} for s, ww in (w or {}).items() if ww for k, v in ww.items()]
    for s, W in B.get("weights", {}).items():
        rows += [{"seed": int(s), "ensemble": e, "base_detector": det(k), "weight": v} for e, ww in W.items() if e != "MCE" and ww for k, v in ww.items()]
    if rows:
        pkg.write(pd.DataFrame(rows), os.path.join(base, "meta_learner_weights.csv"), f"{kind}: meta-learner weights per seed")
    if B.get("vs"):
        pkg.write(pd.DataFrame([{"seed": int(s), "model": det(k), **{kk: (det(vv) if kk == "vs" else vv) for kk, vv in v.items()}} for s, V in B["vs"].items() for k, v in V.items()]),
                  os.path.join(base, "variants_vs_reference_per_seed.csv"), f"{kind}: MCE-lite / MCE without SALE vs MCE, SALE-lite vs SALE (DeLong, Holm within seed)")
    if B.get("reproduction"):
        pkg.write(pd.DataFrame([{"seed": int(s), "detector": det(k), **v} for s, V in B["reproduction"].items() for k, v in V.items()]),
                  os.path.join(base, "reproduction_check.csv"), f"{kind}: AUROC of the earlier run vs the rerun")
    R = B.get("summary") if isinstance(B.get("summary"), dict) and ("all" in B["summary"] or "confirmatory" in B["summary"]) else None
    if R:
        rows = []
        for label in ("all", "confirmatory"):
            for k, v in (R.get(label) or {}).items():
                mdl, ref = (k.split("|vs ")[0], k.split("|vs ")[1]) if "|vs " in k else (k, v.get("vs", ""))
                rows.append({"seeds": "10-19" if label == "all" else "13-19", "model": det(mdl), "reference": det(ref), **_scalar(v)})
        if rows:
            pkg.write(pd.DataFrame(rows), os.path.join(base, "variants_summary_over_seeds.csv"), f"{kind}: summary over seeds of the variants (difference, CI, t test, Wilcoxon)")
        if "per_scenario" in R:
            pkg.write(pd.DataFrame([{"detector": det(k), **v} for k, v in R["per_scenario"].items()]), os.path.join(base, "variants_auroc_per_scenario.csv"), f"{kind}: AUROC per scenario (mean over seeds)")
    if paper:
        _paper_bench(pkg, B, kind, run)


def _bench_core(pkg, B, base, kind):
    """Benchmark results of core_experiments.py (key 'seeds'): metrics, localization, retrieval and ablations."""
    m, sc = _metrics(B["seeds"])
    pkg.write(m, os.path.join(base, "core_metrics_per_seed.csv"), f"{kind} (core run): detection, calibration, localization and retrieval metrics per seed")
    if len(sc):
        pkg.write(sc, os.path.join(base, "core_auroc_per_scenario_per_seed.csv"), f"{kind} (core run): AUROC per scenario per seed")
    rows = []
    for s, ms in B["seeds"].items():
        for mth, e in ms.items():
            for key, what in (("_chamber_hits_F1", "chamber_hit_F1"), ("_retrieval_hits", "retrieval_hit5")):
                for lot, hit in (e.get(key) or {}).items():
                    rows.append({"seed": int(s), "detector": det(mth), "measure": what, "lot_index": int(lot), "hit": bool(hit)})
    if rows:
        pkg.write(pd.DataFrame(rows), os.path.join(base, "core_per_lot_hits.csv"), f"{kind} (core run): per-lot chamber hit@1 (F1) and retrieval hit@5, for the McNemar tests")
    if B.get("ablation"):
        pkg.write(pd.DataFrame([{"variant": k, "seed_position": i, "AUROC": v} for k, vs in B["ablation"].items() for i, v in enumerate(vs)]),
                  os.path.join(base, "core_ablation.csv"), f"{kind} (core run): AUROC of the SALE ablations")


def _paper_bench(pkg, B, kind, run):
    """Table 6 (detection, both data sets), Table 7 (per scenario), Table 8 (MCE design vs confirmatory seeds) and the
    LotBench sensitivity table without F5 / F4+F5, rebuilt from the source run (manuscript v41 numbering)."""
    P = B["per_seed"]; S = B.get("summary", {}); rows = []; ratio = B.get("test_train_ratio")
    for m, e in S.items():
        pr = [P[s][m].get("PR-AUC") for s in P if m in P[s]]; rc = [P[s][m].get("recall5") for s in P if m in P[s]]; fa = [P[s][m].get("FAR5") for s in P if m in P[s]]
        rows.append({"detector": det(m), "AUROC_mean": e.get("AUROC_mean"), "AUROC_sd": e.get("AUROC_sd"), "PR_AUC_mean": np.nanmean(pr),
                     "recall_at_eps5_mean": np.nanmean(rc), "FAR_at_eps5_pct_mean": 100 * np.nanmean(fa), "delta_vs_B3RF": e.get("diff"),
                     "delta_CI_low": e.get("lo"), "delta_CI_high": e.get("hi"), "t": e.get("t"), "df": e.get("df"), "p_t_holm": e.get("p_t_holm"),
                     "t_nadeau_bengio": e.get("t_nb"), "p_nadeau_bengio_holm": e.get("p_nb_holm"),
                     "p_wilcoxon": e.get("p_wilcoxon"), "seeds_DeLong_better": e.get("seeds_better"), "seeds_DeLong_worse": e.get("seeds_worse")})
    fn, lab_ = PAPER_FILES["det_" + kind]
    pkg.write(pd.DataFrame(rows), pkg.pub("03_output", "paper_tables", f"{fn}.csv"), f"{lab_} of the paper (source: run {run})")
    if kind == "lotbench":
        sc = [{"detector": det(m), **e["per_scen"]} for m, e in S.items() if isinstance(e.get("per_scen"), dict)]
        if sc:
            fn, lab_ = PAPER_FILES["scen"]
            pkg.write(pd.DataFrame(sc), pkg.pub("03_output", "paper_tables", f"{fn}.csv"), f"{lab_} of the paper (source: run {run})")
        se = [{"detector": det(m), "AUROC_all": e.get("AUROC_mean"),
               **{k: e.get(k) for k in ("AUROC_excl_F5_mean", "AUROC_excl_F5_diff", "AUROC_excl_F5_t_nb", "AUROC_excl_F5_p_nb_holm",
                                        "AUROC_excl_F4F5_mean", "AUROC_excl_F4F5_diff", "AUROC_excl_F4F5_t_nb", "AUROC_excl_F4F5_p_nb_holm")}}
              for m, e in S.items() if "AUROC_excl_F5_mean" in e]
        if se:
            fn, lab_ = PAPER_FILES["sens"]
            pkg.write(pd.DataFrame(se), pkg.pub("03_output", "paper_tables", f"{fn}.csv"), f"{lab_} (source: run {run}; diff and corrected t vs B3-RF, Holm)")
    rows = []
    for lab, ss in (("10-12 (design)", [10, 11, 12]), ("13-19 (confirmatory)", list(range(13, 20)))):
        sel = [s for s in P if int(s) in ss and "MCE" in P[s] and "B3-RF" in P[s]]
        if len(sel) >= 2:
            r = _paired([P[s]["MCE"]["AUROC"] for s in sel], [P[s]["B3-RF"]["AUROC"] for s in sel], ratio)
            r["seeds_DeLong_significant"] = sum(1 for s in sel if P[s]["MCE"].get("p_holm", 1) < .05)
            rows.append({"data": kind, "seeds": lab, **r})
    if rows:
        fn, lab_ = PAPER_FILES["conf"]
        p = pkg.pub("03_output", "paper_tables", f"{fn}.csv")
        old = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
        new = pd.concat([old[old["data"] != kind] if len(old) and "data" in old else old, pd.DataFrame(rows)])
        pkg.write(new, p, f"{lab_} of the paper (source: run {run})")


def _scores_extended(pkg, raw, base, kind):
    for s, r in raw.items():
        df = pd.DataFrame({"lot_index": r["te"], "label": r["y_te"]})
        if r.get("scen_te") is not None:
            df["scenario"] = r["scen_te"]
        for m, v in r["scores_te"].items():
            df[f"score_{det(m)}"] = v
        pkg.write(df, os.path.join(base, f"test_scores_seed{s}.csv.gz"), f"{kind} seed {s}: score of every detector for every test lot (conformal threshold from the cal lots)")


def _scores_revision(pkg, raw, base, kind):
    for s, r in raw.items():
        nb, nc = r["n_blend"], r["n_cal"]; n = len(r["E"])
        df = pd.DataFrame({"lot_index": r["E"], "role": ["blend"] * nb + ["cal"] * nc + ["test"] * (n - nb - nc), "label": r["y_E"]})
        if r.get("scen_E") is not None:
            df["scenario"] = r["scen_E"]
        for m, v in r["scores_E"].items():
            df[f"score_{det(m)}"] = v
        pkg.write(df, os.path.join(base, f"scores_seed{s}.csv.gz"), f"{kind} seed {s}: score of every detector for the blend, cal and test lots")


def _prod(pkg, stem, R, run, publish_agg):
    """Production results of extended_experiments.py / revision_experiments.py and of core_experiments.py."""
    al = Alias(pkg, PROD_STORE[stem]); name = PROD_NAME[stem]; base = pkg.conf("03_output", run, name)
    agg = lambda fn: pkg.pub("03_output", run, name + "_aggregated", fn)
    pl = R.get("per_lot_rep1")
    if pl:
        y = np.asarray(pl["y"]); idx = np.asarray(pl["idx"], int)
        df = pd.DataFrame({"lot_index": idx, "label": y[idx]})
        if pl.get("tool"):
            df["tool"] = al.map("tool", np.asarray(pl["tool"], object)[idx]).values
        if pl.get("month"):
            df["month"] = al.map("month", np.asarray(pl["month"], object)[idx], "M").values
        for m, sc in pl["scores"].items():
            df[f"score_{det(m)}"] = [sc.get(str(i), sc.get(i)) for i in idx]
        pkg.write(df, os.path.join(base, "oof_scores_repetition1.csv"), "out-of-fold score of every lot in the first CV repetition (tool and month anonymized)")
    lots = R.get("lots")                       # core run: per-lot labels/tools, scores in cv[m]["_idx"/"_s0"]
    for key, sub in (("cv", ""), ("same", "_same_coaters")):
        cv = R.get(key)
        if not cv:
            continue
        if lots and key == "cv":
            y = np.asarray(lots["y"]); df = None
            for m, e in cv.items():
                if "_idx" in e:
                    d_ = pd.DataFrame({"lot_index": e["_idx"], f"score_{det(m)}": e["_s0"]})
                    df = d_ if df is None else df.merge(d_, on="lot_index", how="outer")
            if df is not None:
                df.insert(1, "label", y[df["lot_index"].values])
                df.insert(2, "tool", al.map("tool", np.asarray(lots["tool"], object)[df["lot_index"].values]).values)
                if lots.get("month"):
                    df.insert(3, "month", al.map("month", np.asarray(lots["month"], object)[df["lot_index"].values], "M").values)
                pkg.write(df.sort_values("lot_index"), os.path.join(base, "core_oof_scores_repetition1.csv"), "core run: out-of-fold score of every lot in the first CV repetition")
        rows = []
        for m, e in cv.items():
            r = {"detector": det(m), **_scalar(e)}
            for k in ("AUROC_pct", "within_pct", "CI"):
                if isinstance(e.get(k), list) and len(e[k]) == 2:
                    r[k + "_low"], r[k + "_high"] = e[k]
            if isinstance(e.get("FAR5_reps"), list) and e["FAR5_reps"]:
                r["FAR5_min"], r["FAR5_max"] = min(e["FAR5_reps"]), max(e["FAR5_reps"])
            for eps in ("0.05", "0.01"):
                fr = e.get(f"FAR_reps@{eps}")
                if isinstance(fr, list) and fr:
                    r[f"FAR_min@{eps}"], r[f"FAR_max@{eps}"] = min(fr), max(fr)
            rows.append(r)
        S = pd.DataFrame(rows); pkg.write(S, os.path.join(base, f"summary_cv{sub}.csv"), f"CV summary per detector{sub.replace('_', ' ')} (method level only)")
        if publish_agg:
            pkg.write(S, agg(f"summary_cv{sub}.csv"), f"{name}: CV summary, method level only (aggregates reported in the paper; no lot data)")
    for key in ("within_tool", "within_tool_same"):
        if R.get(key):
            S = pd.DataFrame([{"detector": det(m), **_scalar(v)} for m, v in R[key].items()])
            pkg.write(S, os.path.join(base, f"{key}.csv"), "within-tool AUROC (Eq. 14), sensitivity version and bootstrap test of the drop")
            if publish_agg:
                pkg.write(S, agg(f"{key}.csv"), f"{name}: within-tool AUROC, method level only")
    for key in ("delong_table", "delong_table_same"):
        T = R.get(key)
        if T:
            S = pd.DataFrame([{"detector": det(m), "strongest_baseline": det(T["strongest_baseline"]), **v} for m, v in T["rows"].items()])
            pkg.write(S, os.path.join(base, f"{key}.csv"), "Appendix Table A2: AUROC with DeLong CI and paired DeLong test vs the strongest baseline")
            if publish_agg:
                pkg.write(S, agg(f"{key}.csv"), f"{name}: paired DeLong tests, method level only")
    if R.get("shortcut_stats") or R.get("shortcut"):
        rows = [{"test": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in (R.get("shortcut_stats") or {}).items()]
        rows += [{"test": k, "value": v} for k, v in (R.get("shortcut") or {}).items() if k not in (R.get("shortcut_stats") or {})]
        S = pd.DataFrame(rows); pkg.write(S, os.path.join(base, "shortcut_tests.csv"), "shortcut tests with Hanley-McNeil intervals and tests against 0.5")
        if publish_agg:
            pkg.write(S, agg("shortcut_tests.csv"), f"{name}: shortcut tests, method level only")
    for key in ("power", "power_same"):
        if R.get(key):
            pkg.write(pd.DataFrame([R[key]]), os.path.join(base, f"{key}.csv"), "minimal detectable paired AUROC difference")
            if publish_agg:
                pkg.write(pd.DataFrame([R[key]]), agg(f"{key}.csv"), f"{name}: minimal detectable paired AUROC difference")
    if isinstance(R.get("loto"), dict) and R["loto"]:
        rows = []
        for t, e in R["loto"].items():
            st_ = e.get("_stats") or (R.get("loto_stats") or {}).get(t, {})
            for m, v in e.items():
                if m in ("bad", "good", "_stats"):
                    continue
                s_ = st_.get(m, {})
                rows.append({"held_out_tool": al.map("tool", [t]).iloc[0], "abnormal": e.get("bad"), "normal": e.get("good"), "detector": det(m),
                             "AUROC": v, "lo": s_.get("lo"), "hi": s_.get("hi"), "p_vs_0.5": s_.get("p_vs_0.5")})
        pkg.write(pd.DataFrame(rows), os.path.join(base, "leave_one_tool_out.csv"), "leave-one-tool-out AUROC with Hanley-McNeil intervals (tool anonymized)")
    for key, fname in (("time", "time_split.csv"), ("month", "monthly_forward_chaining.csv")):
        T = R.get(key)
        if isinstance(T, dict) and "note" not in T:
            rows = [{"detector": det(m), **_scalar(v)} for m, v in T.items() if isinstance(v, dict) and m != "per_lot"]
            S = pd.DataFrame(rows); pkg.write(S, os.path.join(base, fname), f"time-ordered evaluation ({key}), method level, Hanley-McNeil intervals")
            if publish_agg:
                pkg.write(S, agg(fname), f"{name}: time-ordered evaluation, method level only")
            if key == "month" and isinstance(T.get("per_lot"), dict):
                pl2 = T["per_lot"]; df = pd.DataFrame({"lot_index": pl2["idx"]})
                for m, v in pl2["scores"].items():
                    df[f"score_{det(m)}"] = v
                pkg.write(df, os.path.join(base, "monthly_forward_chaining_scores.csv"), "per-lot scores of the monthly forward chaining")
        elif isinstance(T, dict):
            pkg.write(pd.DataFrame([{"note": al.scrub(str(T["note"]))}]), os.path.join(base, fname), "time-ordered evaluation not possible; reason")
    for key in ("mce_w", "hyb_w"):
        if R.get(key):
            pkg.write(pd.DataFrame([{"base_detector": det(k), "mean_weight": v} for k, v in R[key].items()]), os.path.join(base, f"{key}.csv"),
                      f"meta-learner weights, mean over folds ({key})")
    done = {"per_lot_rep1", "lots", "cv", "same", "within_tool", "within_tool_same", "delong_table", "delong_table_same", "shortcut",
            "shortcut_stats", "power", "power_same", "loto", "loto_stats", "time", "month", "mce_w", "hyb_w"}
    rest = {k: v for k, v in R.items() if k not in done}
    if rest:
        pkg.write(pd.DataFrame([(al.scrub(k), al.scrub(str(v))) for k, v in _flat(rest)], columns=["key", "value"]),
                  os.path.join(base, "other_fields.csv"), "other fields of the result JSON (notes, DeLong, permutations, ablations), anonymized")


def export_run(pkg, run_dir, publish_prod_agg=False, paper_source=None, extra_public=()):
    """Convert every result JSON of one run into CSV files. paper_source: True/False/None
    (None = automatic: an extended run that is not a quick test is the source of the paper tables)."""
    run = os.path.basename(os.path.normpath(run_dir)); raw = os.path.join(run_dir, "raw")
    pre = None
    pre_files = sorted(set(glob.glob(os.path.join(run_dir, "PREREGISTRATION*.json")) + glob.glob(os.path.join(run_dir, "PRAREGISTRASI*.json"))))
    for p in pre_files:
        fn = os.path.basename(p)
        if "quicktest" in fn:
            continue
        pre = read_json(p); fn_ = LEGACY_FILES.get(fn, fn)
        pkg.write(pd.DataFrame(_flat(pre), columns=["key", "value"]), pkg.pub("00_config", run, fn_.replace(".json", ".csv")),
                  f"frozen configuration and SHA-256 of the scripts of run {run} (written once, before the data were read)")
    if pre is None and os.path.exists(os.path.join(run_dir, "PREREGISTRATION_quicktest.json")):
        pre = read_json(os.path.join(run_dir, "PREREGISTRATION_quicktest.json"))
    version = str((pre or {}).get("configuration", {}).get("version", ""))
    paper = paper_source if paper_source is not None else bool(pre and not pre.get("quick_test") and version.startswith(("extended", "lanjut")))
    for fp in sorted(glob.glob(os.path.join(raw, "*.json"))):
        stem = os.path.splitext(os.path.basename(fp))[0]; stem = LEGACY_STEMS.get(stem, stem)
        try:
            R = read_json(fp)
        except Exception as e:
            print(f"[export] {fp}: could not be read ({e})"); continue
        if stem in PUBLIC_STEMS or stem in extra_public:
            ds = PUBLIC_STEMS.get(stem, stem); base = pkg.pub("03_output", run, ds)
            if stem in ("lotbench", "wmlot", "bench_lotbench", "bench_wmlot") and "per_seed" in R:
                _bench_tables(pkg, R, base, ds, run, paper and stem in ("lotbench", "wmlot"))
            elif stem in ("lotbench", "wmlot") and "seeds" in R:
                _bench_core(pkg, R, base, ds)
            elif stem in ("lotbench_scores", "wmlot_scores"):
                _scores_extended(pkg, R, base, ds)
            elif stem in ("bench_lotbench_scores", "bench_wmlot_scores"):
                _scores_revision(pkg, R, base, ds)
            elif stem == "ablation":
                df = pd.DataFrame([{"seed": int(s), "variant": v, **r} for s, rows in R.items() for v, r in rows.items()])
                pkg.write(df, os.path.join(base, "ablation_seed10.csv" if set(R) == {"10"} else "ablation.csv"), "SALE ablations: AUROC of the variant, difference to the full model, DeLong z, Holm p")
                if paper:
                    fn, lab_ = PAPER_FILES["abl"]
                    pkg.write(df, pkg.pub("03_output", "paper_tables", f"{fn}.csv"), f"{lab_} (source: run {run})")
            elif stem == "ablation_multi":
                pkg.write(pd.DataFrame([{"seed": int(s), "variant": v, **r} for s, S in R["per_seed"].items() for v, r in S["rows"].items()]),
                          os.path.join(base, "ablation_per_seed.csv"), "SALE ablations per seed (including SALE-lite)")
                if R.get("summary"):
                    pkg.write(pd.DataFrame([{"variant": v, **{k: (json.dumps(x) if isinstance(x, list) else x) for k, x in r.items()}} for v, r in R["summary"].items()]),
                              os.path.join(base, "ablation_summary_over_seeds.csv"), "SALE ablations: mean difference over seeds, t test, number of significant seeds")
            elif stem == "learning_curve":
                rows = []
                for r in R.get("runs", {}).values():
                    for arm in ("prod", "sale"):
                        for m, e in (r.get(arm) or {}).items():
                            if isinstance(e, dict):
                                rows.append({"seed": r["seed"], "train_fraction": r.get("fraction"), "n_train_lots": r.get("n_train"), "n_abnormal_train": r.get("n_abnormal"),
                                             "arm": "production_style_MCE" if arm == "prod" else "benchmark_MCE_with_SALE", "detector": det(m), **_scalar(e)})
                pkg.write(pd.DataFrame(rows), os.path.join(base, "learning_curve_runs.csv"), "learning curve: metrics per seed, training fraction and detector")
                rows = [{"train_fraction": float(f), "n_train_lots": e["n_train"], "n_abnormal_train": e["n_abnormal"], "n_seeds": e["n_seeds"],
                         "comparison": {"prod": "MCE-prod vs B3-RF", "sale": "MCE (with SALE) vs B3-RF", "sale_SALE": "SALE vs B3-RF"}[arm], **_scalar(v)}
                        for f, e in R.get("summary", {}).items() for arm, v in e.items() if isinstance(v, dict)]
                pkg.write(pd.DataFrame(rows), os.path.join(base, "learning_curve_summary.csv"), "learning curve: summary over seeds per fraction")
            else:
                pkg.write(pd.DataFrame(_flat(R), columns=["key", "value"]), os.path.join(base, f"{stem}_flat.csv.gz"), f"full content of {stem}.json (long format)")
        elif stem in PROD_NAME:
            _prod(pkg, stem, R, run, publish_prod_agg)
        else:   # default deny
            pkg.write(pd.DataFrame([(scrub_all(pkg, str(k)), scrub_all(pkg, str(v))) for k, v in _flat(R)], columns=["key", "value"]),
                      pkg.conf("03_output", run, f"{stem}_flat.csv.gz"), f"full content of {stem}.json (not listed as public -> confidential)")
    # tables written by core_experiments.py: public only if they concern public data only
    for fp in sorted(glob.glob(os.path.join(run_dir, "table_*.csv"))):
        fn = os.path.basename(fp); df = pd.read_csv(fp)
        if fn.startswith(("table_lotbench", "table_wmlot")):
            pkg.write(df, pkg.pub("03_output", run, "core_tables", fn), f"core run table {fn} (public data only)")
        else:
            pkg.write(df, pkg.conf("03_output", run, "core_tables", fn), f"core run table {fn} (contains production results)")
            if publish_prod_agg:
                pkg.write(df, pkg.pub("03_output", run, "core_tables_production_aggregated", fn), f"core run table {fn}: method-level aggregates reported in the paper")
    for fn in ("REPORT.md", "REPORT_REVISION.md", "REPORT_core.md", "LAPORAN.md", "LAPORAN_REVISI.md", "ISI_JIM.md", "hypotheses.json",
               "log_extended.txt", "log_revision.txt", "log_lanjut.txt", "log_revisi.txt"):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                pkg.write_text(scrub_all(pkg, f.read()), pkg.conf("03_output", run, LEGACY_FILES.get(fn, fn)), f"report/log of run {run} (anonymized)")
    for fp in sorted(glob.glob(os.path.join(run_dir, "fig_*.png"))):
        dst = pkg.conf("03_output", run, "figures", os.path.basename(fp)); os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(fp, dst); pkg.note(dst, "figure of the core run (may show production results)", None)
    print(f"[export] run {run} done")


# ============================================================ CLI
def main():
    sys.path.insert(0, HERE)
    try:
        import local_config as LC
    except ImportError:
        LC = None
    setting = lambda n, d: getattr(LC, n, d) if LC is not None else d
    ap = argparse.ArgumentParser(description="Export the CSV package (input, process, output)")
    ap.add_argument("--package", default=setting("CSV_PACKAGE", "csv_package"))
    ap.add_argument("--json-run", action="append", default=[], help="run folder (with raw/*.json); may be repeated")
    ap.add_argument("--paper-table-source", default=None, help="name of the run folder that is the source of the paper tables (default: automatic = the extended run)")
    ap.add_argument("--extra-public", action="append", default=[], help="extra JSON stem that certainly contains public data only")
    ap.add_argument("--publish-production-aggregates", action="store_true", help="also copy METHOD-LEVEL summaries of the production lines to public (they are reported in the paper)")
    ap.add_argument("--input-public", action="store_true"); ap.add_argument("--input-production", action="store_true")
    ap.add_argument("--folder", default=None); ap.add_argument("--label", default=None)
    a = ap.parse_args(); pkg = Package(a.package)
    if a.input_public or a.input_production:
        import revision_experiments as JR
        JR.init(); J, JL = JR.J, JR.JL
        if a.input_public:
            from sklearn.model_selection import train_test_split
            ns = argparse.Namespace(quick_test=False); meta, maps, MIX = JR.load_public(ns)
            if meta is None:
                sys.exit("WM-811K not found")
            wmd = None
            for kind in ("lotbench", "wmlot"):
                if kind == "wmlot":
                    wmd = J.make_wmlot(meta, maps)
                for seed in JR.CFG["bench_seeds"]:
                    d, tr, cal, te, _ = JR.prep_bench(kind, seed, meta, maps, MIX, ns, wmd)
                    fit, blend = train_test_split(tr, test_size=0.10, stratify=d.ybin[tr], random_state=seed)
                    export_bench_input(pkg, J, d, kind, seed, {"fit": fit, "blend": blend, "cal": cal, "test": te}); pkg.flush()
                    print(f"[export] input {kind} seed {seed}")
        if a.input_production:
            ns = argparse.Namespace(folder=a.folder or J.PARTNER_DATA_DIR, label=a.label or J.LABEL_FINAL)
            for stem in ("plantb", "diesaw", "faba"):
                d = JL.load_prod(stem, ns)
                if d is None:
                    print(f"[export] {stem}: data not found"); continue
                export_dataset(pkg, d, PROD_NAME[stem], False, stem=stem)
                export_folds(pkg, prod_fold_rows(J, d, 20, 1000), stem, "core", "CV folds of core_experiments.py (20 repetitions x 5 folds): fit, validation, cal, test")
                export_folds(pkg, prod_fold_rows(J, d, JL.CFG["cv"]["repeats"], 5000, cal_frac=JL.CFG["cv"]["frac_calibration_normal"]), stem, "extended",
                             "CV folds of extended_experiments.py (10 repetitions x 5 folds): fit, validation, cal, test")
                if stem == "plantb":
                    sub, _ = JR.same_coater_idx(d)
                    export_folds(pkg, prod_fold_rows(J, d, JR.CFG["samecoater"]["repeats"], 5000, subset=sub), "samecoater", "revision",
                                 "CV folds of the same-coaters subset (revision_experiments.py)")
                pkg.flush(); print(f"[export] production input {stem} (confidential)")
    for r in a.json_run:
        export_run(pkg, r, a.publish_production_aggregates,
                   None if a.paper_table_source is None else os.path.basename(os.path.normpath(r)) == a.paper_table_source, set(a.extra_public))
    pkg.flush()
    print(f"Package: {pkg.root}\n  public/       -> may be uploaded (check with build_release.py first)\n  confidential/ -> NEVER upload")


if __name__ == "__main__":
    main()
