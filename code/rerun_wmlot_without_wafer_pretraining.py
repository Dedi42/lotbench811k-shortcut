#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rerun_wmlot_without_wafer_pretraining.py - WM-811K lot task: SALE and SALE-lite WITHOUT map-encoder pretraining.

Why: in the paper run, the map encoder of SALE and SALE-lite was pretrained on the wafer pattern labels (failureType)
of the training lots and on MixedWM38. On the WM-811K lot task the lot label is defined from these wafer labels, so
SALE / SALE-lite received wafer-level supervision that no other detector had. This script repeats exactly the same
ten seeds and splits with the map-encoder pretraining switched off (opts pre_menc=False), so SALE and SALE-lite see
lot labels only, like every other detector. Everything else is unchanged (Stage-1 pretraining of SALE on normal
training lots, which uses no labels, stays as in the paper; SALE-lite drops it as before).

The stacked ensembles that contain SALE or SALE-lite (MCE, MCE-lite) are re-fitted on the blending lots with the new
scores; the scores of all other detectors are taken unchanged from the released score files of the paper run.

Run (GPU recommended, about 3-5 h for ten seeds; resumable, finished seeds are skipped):
    python rerun_wmlot_without_wafer_pretraining.py
    python rerun_wmlot_without_wafer_pretraining.py --released <.../03_output/results_revision_v20> --out results_wmlot_nopretrain
    python rerun_wmlot_without_wafer_pretraining.py --quick-test          # a few minutes on synthetic data (pipeline check)
Then:
    python ../reproduce/compare_wmlot_pretraining.py --nopretrain results_wmlot_nopretrain
"""
import argparse, glob, hashlib, json, os, sys, time
from types import SimpleNamespace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import revision_experiments as RV                     # noqa: E402

SEEDS = list(range(10, 20))
SWAP = {"Ours": "SALE", "SALE-lite": "SALE-lite"}     # internal name -> column name in the score files


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def default_released():
    pkg = None
    try:
        import local_config as LC
        pkg = getattr(LC, "CSV_PACKAGE", None)
    except Exception:                                  # noqa: BLE001
        pass
    for c in ([os.path.join(pkg, "public", "03_output", "results_revision_v20")] if pkg else []) + \
             [os.path.join(HERE, "..", "results", "public", "results_revision_v20")]:
        if os.path.isdir(os.path.join(c, "wmlot")):
            return c
    return os.path.join(HERE, "..", "results", "public", "results_revision_v20")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--released", default=None, help="folder results_revision_v20 with wmlot/scores_seed*.csv.gz of the paper run")
    ap.add_argument("--out", default="results_wmlot_nopretrain")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--quick-test", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    RV.init(); J, JL, EK = RV.J, RV.JL, RV.EK; RV.PKG = None
    if not J.HAS_TORCH and not a.quick_test:
        sys.exit("STOP: PyTorch is not installed; SALE cannot be trained.")
    released = a.released or default_released()
    out = a.out + ("_DEMO" if a.quick_test else ""); os.makedirs(os.path.join(out, "wmlot"), exist_ok=True)
    seeds = [0, 1] if a.quick_test else [int(s) for s in a.seeds.split(",")]
    ra = SimpleNamespace(quick_test=a.quick_test)
    opts = dict(RV.bench_opts(ra), pre_menc=False)
    cfg = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "purpose": "WM-811K lot task, SALE and SALE-lite without map-encoder pretraining",
           "opts": opts, "seeds": seeds, "released_scores": os.path.abspath(released) if not a.quick_test else None,
           "sha256": {f: sha(os.path.join(HERE, f)) for f in ("core_experiments.py", "extended_experiments.py", "revision_experiments.py",
                                                            os.path.basename(__file__))}}
    J.jdump(cfg, os.path.join(out, "CONFIGURATION_nopretrain.json"))
    meta, maps, MIX = RV.load_public(ra)
    if meta is None:
        sys.exit("WM-811K (LSWMD.pkl) not found: set PUBLIC_DATA_DIR in local_config.py.")
    wmd = J.make_wmlot(meta, maps)
    lite, used = RV.lite_changes(J.ABLATIONS)
    bases = [b for b in JL.CFG["MCE"]["bases_benchmark"]]
    summary = {}
    for seed in seeds:
        dst = os.path.join(out, "wmlot", f"scores_seed{seed}.csv.gz")
        if os.path.exists(dst):
            print(f"[seed {seed}] already done; skipped", flush=True); continue
        t0 = time.time()
        d, tr, cal, te, _ = RV.prep_bench("wmlot", seed, meta, maps, MIX, ra, wmd); yb = d.ybin
        d.pre_maps = d.pre_y = None; d.mix_maps = d.mix_y = None          # no wafer-level labels anywhere
        fit, blend = train_test_split(tr, test_size=0.10, stratify=yb[tr], random_state=seed)
        E = np.r_[blend, cal, te]; nb, nbc = len(blend), len(blend) + len(cal)
        new = {}
        for nm, ch in (("Ours", {}), ("SALE-lite", lite)):
            r = J.run_fold(d, fit, np.array([], int), np.r_[blend, cal], te, ch, ["Ours"], seed=seed, opts=opts)
            new[nm] = RV.score_of(r)
            if new[nm] is None or len(new[nm]) != len(E):
                sys.exit(f"[seed {seed}] {nm}: no score returned")
        if a.quick_test:
            df = pd.DataFrame({"lot_index": E, "role": ["blend"] * nb + ["cal"] * len(cal) + ["test"] * len(te), "label": yb[E]})
            for b in bases:
                if b not in SWAP:
                    df["score_" + EK.det(b)] = np.random.default_rng(seed).normal(size=len(E)) + yb[E]
        else:
            src = os.path.join(released, "wmlot", f"scores_seed{seed}.csv.gz")
            df = pd.read_csv(src)
            if not np.array_equal(df["lot_index"].values, E) or not np.array_equal(df["label"].values, yb[E]):
                sys.exit(f"[seed {seed}] the released score file does not match the splits of this run (lot order or labels differ)")
        df["score_SALE"] = new["Ours"]; df["score_SALE-lite"] = new["SALE-lite"]
        cols = {b: "score_" + EK.det(b) for b in bases}
        have = [b for b in bases if cols[b] in df or b == "Ours"]
        M = np.column_stack([df[cols[b]].values for b in have])
        df["score_MCE"], w = JL.meta_combine(M[:nb], yb[blend], M)
        lb = ["SALE-lite" if b == "Ours" else b for b in have]
        ML = np.column_stack([df["score_SALE-lite"].values if b == "SALE-lite" else df[cols[b]].values for b in lb])
        df["score_MCE-lite"], wl = JL.meta_combine(ML[:nb], yb[blend], ML)
        df.to_csv(dst, index=False, compression="gzip")
        t = df["role"].values == "test"
        auc = {m: float(roc_auc_score(df["label"][t], df["score_" + m][t])) for m in ("B3-RF", "SALE", "SALE-lite", "MCE", "MCE-lite") if "score_" + m in df}
        summary[str(seed)] = {"AUROC": auc, "weights_MCE": dict(zip(have, np.round(w, 3).tolist())), "minutes": (time.time() - t0) / 60}
        J.jdump(summary, os.path.join(out, "summary_nopretrain.json"))
        print(f"[seed {seed}] ({(time.time() - t0) / 60:.0f} min) " + ", ".join(f"{m} {v:.3f}" for m, v in auc.items()), flush=True)
    print(f"done: {out}" + ("  (QUICK TEST: synthetic data, numbers meaningless)" if a.quick_test else ""))


if __name__ == "__main__":
    main()
