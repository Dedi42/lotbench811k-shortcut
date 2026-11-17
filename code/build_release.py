#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
build_release.py - assemble the upload folder (code + public CSV files) and check that no company data are included.

Copied:       the public scripts -> release/code/ (production_loaders_stub.py is copied as production_loaders.py),
              csv_package/public -> release/data/
NEVER copied: csv_package/confidential, production_loaders.py (the real loaders), local_config.py.
Checks (every text file, including .csv.gz):
  - every original value in confidential/alias_keys_INTERNAL_ONLY (lot, tool, chamber, ... identifiers)
  - strings from --secret-list (one per line: company names, server names, internal IDs, ...)
  - absolute Windows/UNC paths, e-mail addresses, IP addresses
If anything is found, the list is written OUTSIDE the release folder (<out>_CHECK.txt, because it contains the secret
strings), the release folder is marked NOT_SAFE.txt and the script exits with code 1.
Text replacements in the copied code only (never in the originals): --replace "<old>=><new>" (may be repeated).
Note: replacements change the SHA-256 of the copied scripts, so they no longer match the pre-registration files;
prefer editing local_config.py instead.

Example:
  python build_release.py --package csv_package --out release --secret-list secret_list.txt --zip
"""
import argparse, glob, gzip, hashlib, os, re, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_FILES = ["core_experiments.py", "extended_experiments.py", "revision_experiments.py", "export_csv.py", "build_release.py",
              "inspect_public_data.py", "local_config_example.py", "run_all.py"]
STUB = ("production_loaders_stub.py", "production_loaders.py")      # (source, name in the release)
NEVER = {"production_loaders.py", "local_config.py"}
KEY_COLS = ("lot", "tool", "chamber", "ch_", "eqp", "equip", "recipe", "product", "device", "blade", "batch", "carrier", "foup", "operator", "host", "material")
PATTERNS = {"Windows path": r"\b[A-Za-z]:\\[^\s'\"]+", "UNC path": r"\\\\[A-Za-z0-9_.-]+\\[^\s'\"]+",
            "e-mail": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "IP address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b"}
TEXT_EXT = (".py", ".md", ".txt", ".csv", ".json", ".gz", ".cfg", ".toml", ".yml", ".yaml")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def blocklist(package, extra_file):
    words = set()
    for p in glob.glob(os.path.join(package, "confidential", "alias_keys_INTERNAL_ONLY", "*", "*.csv")):
        if os.path.basename(os.path.dirname(p)) == "synthetic_demo":
            continue                       # quick-test data are synthetic (built from public WM-811K lot names): nothing secret
        if not any(k in os.path.basename(p).lower() for k in KEY_COLS):
            continue                       # sensor names and months are often ordinary words -> not blocked automatically
        with open(p, encoding="utf-8") as f:
            next(f, None)
            for line in f:
                o = line.rsplit(",", 1)[0].strip().strip('"')
                if len(o) >= 4 and not o.isdigit():
                    words.add(o)
    if extra_file:
        with open(extra_file, encoding="utf-8") as f:
            words |= {w.strip() for w in f if w.strip() and not w.startswith("#")}
    return sorted(words, key=len, reverse=True)


def scan(root, words, allow_paths=()):
    rx_w = re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE) if words else None
    rx_p = {k: re.compile(v) for k, v in PATTERNS.items()}; hits = []
    for dp, _, fs in os.walk(root):
        for fn in fs:
            p = os.path.join(dp, fn)
            if not fn.lower().endswith(TEXT_EXT) and fn != "LICENSE":
                hits.append((p, 0, "unknown file type (check manually)", fn)); continue
            op = gzip.open if fn.endswith(".gz") else open
            try:
                with op(p, "rt", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if rx_w:
                            m = rx_w.search(line)
                            if m:
                                hits.append((p, i, "secret string", m.group(0)))
                        for k, r in rx_p.items():
                            m = r.search(line)
                            if m and not any(a in m.group(0) for a in allow_paths) and not (k == "IP address" and fn.endswith((".csv", ".gz"))):
                                hits.append((p, i, k, m.group(0)[:80]))
            except OSError as e:
                hits.append((p, 0, "unreadable", str(e)))
    return hits


def requirements():
    from importlib import metadata
    out = []
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "torch"):
        try:
            out.append(f"{pkg}=={metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            pass
    return "\n".join(out) + "\n"


README = """# From wafers to lots: code and public data

Code and CSV files for *From wafers to lots: complementary detectors, shortcuts and calibrated alarms in lot-level
abnormality detection for semiconductor manufacturing* (Surahman & Hsieh, submitted to the Journal of Intelligent Manufacturing).
DOI of this archive: [to be added after the Zenodo upload]. Licence: MIT for the code (LICENSE), CC BY 4.0 for the CSV files.

## Contents
- `code/core_experiments.py` – LotBench-811K generator, WM-811K lot task, all detectors (B1–B10, SALE, SALE-S), production
  protocols, shortcut tests, calibration, attribution and the statistics reported in the paper.
- `code/extended_experiments.py` – tool-invariant variants, the mechanism-complementary ensemble (MCE), ten-seed benchmarks
  (paired t test and corrected resampled t test of Nadeau & Bengio), LotBench sensitivity without F5/F4 and a context-only
  check, SALE ablations, leave-one-tool-out and time-ordered evaluation.
- `code/revision_experiments.py` – SALE-lite, MCE without SALE, ablations over three seeds, learning curve, same-coaters MCE.
- `code/export_csv.py` – writes every input, process and output as CSV; `code/build_release.py` – builds and checks this archive.
- `code/production_loaders.py` – a stub: the loaders of the confidential production data are not released (see below).
- `code/local_config_example.py` – copy to `local_config.py` and set the data folders.
- `data/` – CSV files of the **public data only** (LotBench-811K and the WM-811K lot task):
  - `00_config/` frozen configurations and SHA-256 fingerprints of the scripts of each run;
  - `01_input/` lots and wafers as given to the detectors (`map_i` indexes rows of `LSWMD.pkl`);
  - `02_process/` role of every lot per seed (fit, blend, cal, test), wafer features after chamber-relative
    normalization (Eq. 2), lot-level features, learning-curve subsamples;
  - `03_output/` score of every detector for every lot and seed, per-seed metrics, summaries;
    `03_output/paper_tables/` rebuilds Table 6 (`table06a`, `table06b`), Table 7, Table 8 and the sensitivity table
    from these scores;
  - `FILES.csv` describes every file.
- `MANIFEST.csv` – size and SHA-256 of every file; `requirements.txt` – package versions used.

Inside the code SALE is called `Ours` and SALE-S `Ours-S`; the CSV files use the names of the paper.
Fab A is called `faba` in the code.

## Not included
The production data of the three lines (Plant B coat/develop, Plant B wafer saw, Fab A PECVD) and the loaders of their file
formats are confidential under agreements with the partners. Anonymized lot-level CSV files with the same structure as
`data/` can be made available to the editor and reviewers on request, subject to the partners' approval. Only aggregated
results of these lines appear in the paper.

## Reproducing
1. Download WM-811K (`LSWMD.pkl`, Kaggle: qingyi/wm811k-wafer-map) and, optionally, MixedWM38; put them in `data/` or set
   `PUBLIC_DATA_DIR` in `local_config.py`.
2. `pip install -r requirements.txt` (PyTorch with CUDA is recommended for SALE).
3. Benchmarks: `python code/extended_experiments.py --parts lotbench,wmlot` and
   `python code/revision_experiments.py --parts bench,curve,ablation`; localization and attribution:
   `python code/core_experiments.py --parts lotbench --seeds 10,11,12`.
4. CSV files: `python code/export_csv.py --input-public --json-run results_extended --json-run results_revision`.
Everything in the right order: `python code/run_all.py` (quick check on small data: `python code/run_all.py --quick-test`).
"""


LICENSE = """MIT License

Copyright (c) 2026 Dedi Surahman and Yu-Ming Hsieh

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The CSV files in data/ are licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
"""


def main():
    ap = argparse.ArgumentParser(description="Assemble and check the upload folder")
    ap.add_argument("--code", default=HERE); ap.add_argument("--files", nargs="*", default=CODE_FILES)
    ap.add_argument("--package", default=os.path.join(HERE, "csv_package")); ap.add_argument("--out", default=os.path.join(HERE, "release"))
    ap.add_argument("--secret-list", default=None); ap.add_argument("--replace", action="append", default=[])
    ap.add_argument("--allow-path", action="append", default=[], help="path fragment that may appear (e.g. ./data_local)")
    ap.add_argument("--overwrite", action="store_true"); ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    bad = [f for f in a.files if os.path.basename(f) in NEVER]
    if bad:
        sys.exit(f"Refused: {bad} must never be released (confidential loaders / local settings).")
    pub = os.path.join(a.package, "public")
    if not os.path.isdir(pub):
        sys.exit(f"{pub} does not exist; run export_csv.py first.")
    if os.path.exists(a.out):
        if not a.overwrite:
            sys.exit(f"{a.out} already exists; use --overwrite to rebuild it.")
        shutil.rmtree(a.out)
    os.makedirs(os.path.join(a.out, "code"))
    for src_name, dst_name in [(f, f) for f in a.files] + [STUB]:
        src = os.path.join(a.code, src_name)
        if not os.path.exists(src):
            print(f"WARNING: {src} not found; not copied"); continue
        with open(src, encoding="utf-8") as f:
            txt = f.read()
        for g in a.replace:
            old, new = g.split("=>", 1); txt = txt.replace(old, new)
        with open(os.path.join(a.out, "code", dst_name), "w", encoding="utf-8") as f:
            f.write(txt)
    shutil.copytree(pub, os.path.join(a.out, "data"))
    with open(os.path.join(a.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(README)
    with open(os.path.join(a.out, "LICENSE"), "w", encoding="utf-8") as f:
        f.write(LICENSE)
    with open(os.path.join(a.out, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(requirements())
    hits = scan(a.out, blocklist(a.package, a.secret_list), tuple(a.allow_path))
    rows = ["file,bytes,sha256"]
    for dp, _, fs in os.walk(a.out):
        for fn in sorted(fs):
            p = os.path.join(dp, fn); rows.append(f"{os.path.relpath(p, a.out).replace(os.sep, '/')},{os.path.getsize(p)},{sha256(p)}")
    with open(os.path.join(a.out, "MANIFEST.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    rep = a.out.rstrip("\\/") + "_CHECK.txt"
    with open(rep, "w", encoding="utf-8") as f:
        f.write(f"Check {time.strftime('%Y-%m-%d %H:%M')}: {len(hits)} findings\n")
        for p, i, k, m in hits:
            f.write(f"{os.path.relpath(p, a.out)}:{i}: {k}: {m}\n")
    if hits:
        with open(os.path.join(a.out, "NOT_SAFE.txt"), "w", encoding="utf-8") as f:
            f.write("This folder must NOT be uploaded yet. See the check report outside this folder.\n")
        print(f"NOT SAFE: {len(hits)} findings. List: {rep}")
        for p, i, k, m in hits[:20]:
            print(f"  {os.path.relpath(p, a.out)}:{i}: {k}")
        sys.exit(1)
    if a.zip:
        shutil.make_archive(a.out, "zip", a.out); print(f"Archive: {a.out}.zip")
    print(f"SAFE: no findings. Upload folder: {a.out} ({len(rows) - 1} files)")


if __name__ == "__main__":
    main()
