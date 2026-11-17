#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
inspect_public_data.py - inspect the STRUCTURE of the public data sets in a folder (read-only; changes nothing).
Purpose: find out which data sets have a lot / tool / wafer structure usable for lot-level tasks.
Run:     python inspect_public_data.py [folder]
Output:  inspect_public_data.txt (next to this script)
"""
import os, glob, sys, io, zipfile
import numpy as np
import pandas as pd

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "data"           # folder with the public data sets
OUTFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inspect_public_data.txt")
out = open(OUTFILE, "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.write(s + "\n")


def mb(f):
    try:
        return f"{os.path.getsize(f) / 1e6:.1f} MB"
    except Exception:
        return "?"


def head_text(f, n=4):
    try:
        with open(f, "r", encoding="utf-8", errors="replace") as h:
            for _ in range(n):
                line = h.readline()
                if not line:
                    break
                p("      |", line[:250].rstrip())
    except Exception as e:
        p("      failed to read text:", e)


def show_table(T, name):
    p(f"   {name}: {T.shape[0]} rows (read), {T.shape[1]} columns")
    p(f"   columns: {list(T.columns)[:60]}")
    p("   first 3 rows:")
    p("   " + T.head(3).to_string()[:2000].replace("\n", "\n   "))
    for col in T.columns:
        cl = str(col).lower()
        if any(k in cl for k in ("tool", "lot", "run", "stage", "recipe", "wafer", "slot", "chamber", "label", "fault", "class")):
            try:
                p(f"   - {col}: {T[col].nunique()} unique values; examples {list(T[col].dropna().unique()[:6])}")
            except Exception:
                pass
    lot = next((c for c in T.columns if str(c).lower() == "lot"), None)
    run = next((c for c in T.columns if str(c).lower() in ("runnum", "run", "wafer", "waferid", "slot")), None)
    if lot is not None and run is not None:
        g = T.groupby(lot)[run].nunique()
        p(f"   unique {run} per {lot}: median {g.median()}, min {g.min()}, max {g.max()} (over {len(g)} {lot})")


if not os.path.isdir(DATA_DIR):
    p(f"Folder not found: {DATA_DIR}. Pass the folder as an argument.")
    out.close(); sys.exit(1)

p(f"Folder data: {DATA_DIR}\n")
p("=== 1. FOLDER CONTENTS (up to 3 levels) ===")
for root, dirs, files in os.walk(DATA_DIR):
    depth = root[len(DATA_DIR):].count(os.sep)
    if depth > 3:
        continue
    p(f"  {'  ' * depth}{root}  -> {len(dirs)} folder, {len(files)} file")
    for f in sorted(files)[:12]:
        p(f"  {'  ' * depth}    {f}  ({mb(os.path.join(root, f))})")
    if len(files) > 12:
        p(f"  {'  ' * depth}    ... and {len(files) - 12} more files")

p("\n=== 2. FILE .npz (MixedWM38 / Wafer_Map_Datasets) ===")
npzs = glob.glob(os.path.join(DATA_DIR, "**", "*.npz"), recursive=True)
sig = {}
for f in npzs:
    p(f"\n-- {f} ({mb(f)})")
    try:
        z = np.load(f, allow_pickle=True)
        for k in z.keys():
            a = z[k]
            p(f"   {k}: shape {a.shape}, dtype {a.dtype}")
            if a.ndim >= 2 and a.shape[0] > 0:
                p(f"      unique values (first example): {np.unique(a[0])[:10]}")
        sig[f] = tuple((k, z[k].shape) for k in z.keys())
    except Exception as e:
        p("   failed:", e)
if len(sig) >= 2:
    vals = list(sig.values())
    p("\n   All .npz files have the same shape -> probably the same file under different names" if all(v == vals[0] for v in vals)
      else "\n   .npz shapes differ -> different data sets")

p("\n=== 3. LSWMD.pkl (WM-811K) ===")
for f in glob.glob(os.path.join(DATA_DIR, "**", "LSWMD.pkl"), recursive=True):
    p(f"-- {f} ({mb(f)})")
    try:
        sys.modules.setdefault("pandas.indexes", pd.core.indexes)
        df = pd.read_pickle(f)
        p(f"   {len(df)} wafers; columns {list(df.columns)}")
        n = df.groupby("lotName").size()
        p(f"   {len(n)} lots; wafers per lot: median {n.median()}, lots with >= 20 wafers: {(n >= 20).sum()}")
        def ft(v):
            try:
                v = np.asarray(v).ravel(); return str(v[0]) if len(v) else None
            except Exception:
                return None
        lab = df["failureType"].map(ft)
        p(f"   labelled wafers: {lab.notna().sum()}; distribution: {lab.value_counts().to_dict()}")
        full = lab.notna().groupby(df["lotName"]).all()
        p(f"   lots whose wafers are ALL labelled: {int(full.sum())}; of these with >= 20 wafers: {int((full & (n >= 20)).sum())}")
    except Exception as e:
        p("   failed:", e)

p("\n=== 4. 'Wafer' data (files or folders named Wafer*) ===")
cand = [c for c in glob.glob(os.path.join(DATA_DIR, "**", "*"), recursive=True)
        if os.path.basename(c).lower().startswith("wafer") and not c.lower().endswith(".npz")]
for c in cand[:10]:
    if os.path.isdir(c):
        fs = sorted(glob.glob(os.path.join(c, "**", "*"), recursive=True))
        p(f"-- folder {c}: {len(fs)} items; examples {[os.path.relpath(x, c) for x in fs[:15]]}")
        for g in [x for x in fs if os.path.isfile(x)][:4]:
            p(f"   {os.path.relpath(g, c)} ({mb(g)}), first rows:")
            head_text(g)
    elif os.path.isfile(c):
        p(f"-- file {c} ({mb(c)}), first rows:")
        head_text(c)

p("\n=== 5. PHM Data Challenge 2018 ===")
phm = [d for d in glob.glob(os.path.join(DATA_DIR, "**", "*"), recursive=True) if "phm" in os.path.basename(d).lower()]
for d in phm[:5]:
    if os.path.isfile(d) and d.lower().endswith(".zip"):
        p(f"-- zip {d} ({mb(d)})")
        try:
            z = zipfile.ZipFile(d); names = z.namelist()
            p(f"   {len(names)} entries; examples {names[:15]}")
            for nm in [x for x in names if x.lower().endswith(".csv")][:2] + [x for x in names if "fault" in x.lower() and x.lower().endswith(".csv")][:2]:
                show_table(pd.read_csv(io.BytesIO(z.read(nm)), nrows=200000), nm)
        except Exception as e:
            p("   failed:", e)
        continue
    if not os.path.isdir(d):
        continue
    csvs = sorted(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))
    p(f"-- folder {d}: {len(csvs)} file csv")
    p(f"   examples: {[os.path.relpath(c, d) for c in csvs[:15]]}")
    pick = csvs[:2] + [c for c in csvs if "fault" in c.lower()][:3]
    for c in dict.fromkeys(pick):
        p(f"\n   >> {os.path.relpath(c, d)} ({mb(c)})")
        try:
            show_table(pd.read_csv(c, nrows=200000), os.path.basename(c))
        except Exception as e:
            p("   failed to read:", e)

out.close()
print(f"\nDone. Output: {OUTFILE}")
