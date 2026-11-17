#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py - ONE command for the whole pipeline of the paper. Nothing has to be prepared by hand:

    python run_all.py --quick-test      # first: 20-40 min on small/demo data (must end with QUICK TEST PASSED)
    python run_all.py                   # then: the full run (about 3-4 days with a CUDA GPU), resumable

On every start the script
  1. creates local_config.py (paths) and secret_list.txt if they do not exist yet,
  2. installs missing Python packages (numpy, pandas, scipy, scikit-learn, matplotlib, openpyxl, xlrd, PyTorch with CUDA),
  3. checks the GPU and the data files and prints what was found,
  4. runs the steps in order; a finished step is skipped on the next start, a failed step stops the run.

More options:
    python run_all.py --list              # steps and commands
    python run_all.py --only revision     # selected steps (comma separated)
    python run_all.py --from export       # start at a step
    python run_all.py --redo core_report  # repeat a finished step
    python run_all.py --setup-only        # only steps 1-3
All output also goes to run_all_log.txt.
"""
import argparse, glob, importlib, importlib.util, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# ============================================================ 1. local files (created once, never uploaded)
LOCAL_CONFIG = r'''# -*- coding: utf-8 -*-
# local_config.py - created automatically by run_all.py. NEVER upload this file. Edit the paths to your data.
PUBLIC_DATA_DIR = "data"            # LSWMD.pkl and MixedWM38.npz (searched recursively)
WM811K_PKL = ""
MIXEDWM38_NPZ = ""
PARTNER_DATA_DIR = ""               # partner exports (production lines, confidential)
PLANTB_LABEL_CSV = ""               # Plant B lot labels (lot_id,status), confidential
LOADER_DIR = ""                     # folder of production_loaders.py if it is not next to the scripts
OUT_CORE = "results_core_v20"        # NEW folders for code v20: never mix with results of older code versions
OUT_EXTENDED = "results_extended_v20"
OUT_REVISION = "results_revision_v20"
CSV_PACKAGE = "csv_package_v20"
'''
SECRET_LIST = """# One string per line. build_release.py refuses the release if one of them appears in the code or the public CSV files.
# Production lot/tool/chamber IDs are added automatically from csv_package/confidential/alias_keys_INTERNAL_ONLY.
# Add the short names of ALL partner companies, names of company folders, servers, label files, products or recipes here.
"""


def secret_entries():
    p = os.path.join(HERE, "secret_list.txt")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [w.strip() for w in f if w.strip() and not w.strip().startswith("#")]


def ensure_local_files(say):
    p = os.path.join(HERE, "local_config.py")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            f.write(LOCAL_CONFIG)
        say(f"[setup] created {p} (edit it if your paths differ)")
    p = os.path.join(HERE, "secret_list.txt")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            f.write(SECRET_LIST)
        say(f"[setup] created {p}")


# ============================================================ 2. packages
PACKAGES = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn", "matplotlib": "matplotlib",
            "openpyxl": "openpyxl", "xlrd": "xlrd"}


def pip(args, say):
    say("[setup] pip install " + " ".join(args))
    return subprocess.call([PY, "-m", "pip", "install", "--disable-pip-version-check"] + args) == 0


def ensure_packages(say):
    missing = [p for m, p in PACKAGES.items() if importlib.util.find_spec(m) is None]
    if missing and not pip(missing, say):
        sys.exit("Package installation failed; check the internet connection and run again.")
    if importlib.util.find_spec("torch") is None:
        ok = pip(["torch", "--index-url", "https://download.pytorch.org/whl/cu121"], say) or pip(["torch"], say)
        if not ok:
            say("[setup] WARNING: PyTorch could not be installed; SALE and the neural baselines will be skipped.")
    importlib.invalidate_caches()


def gpu_status():
    code = "import torch;print(torch.cuda.is_available());print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    try:
        out = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=120).stdout.split("\n")
        if not out[0].strip():
            return None, ""
        return out[0].strip() == "True", out[1].strip() if len(out) > 1 else ""
    except Exception:
        return None, ""


# ============================================================ 3. checks
def preflight(LC, quick, say):
    S = lambda n, d: getattr(LC, n, d)
    pub = S("PUBLIC_DATA_DIR", "data")
    if not os.path.isdir(pub) and os.path.isdir(os.path.join(HERE, "data")):
        pub = os.path.join(HERE, "data")
    wm = S("WM811K_PKL", "") or next(iter(glob.glob(os.path.join(pub, "**", "LSWMD.pkl"), recursive=True)), "")
    mix = S("MIXEDWM38_NPZ", "") or next(iter(glob.glob(os.path.join(pub, "**", "MixedWM38.npz"), recursive=True) +
                                               glob.glob(os.path.join(pub, "**", "Wafer_Map_Datasets.npz"), recursive=True)), "")
    part, lab = S("PARTNER_DATA_DIR", ""), S("PLANTB_LABEL_CSV", "")
    loaders = os.path.exists(os.path.join(HERE, "production_loaders.py"))
    cuda, name = gpu_status()
    say("[check] WM-811K (LSWMD.pkl):   " + (wm or "NOT FOUND"))
    say("[check] MixedWM38:             " + (mix or "not found (optional)"))
    say("[check] partner data folder:   " + (part if part and os.path.isdir(part) else f"NOT FOUND ({part})"))
    say("[check] Plant B labels:        " + (lab if lab and os.path.exists(lab) else f"NOT FOUND ({lab})"))
    say("[check] production_loaders.py: " + ("found" if loaders else "NOT FOUND"))
    say("[check] PyTorch / GPU:         " + ({True: f"CUDA available ({name})", False: "CPU only", None: "PyTorch missing"}[cuda]))
    if quick:
        return
    problems = []
    if not wm:
        problems.append("LSWMD.pkl not found: put it in PUBLIC_DATA_DIR (local_config.py)")
    if not (part and os.path.isdir(part)) or not (lab and os.path.exists(lab)) or not loaders:
        problems.append("production data, labels or production_loaders.py missing: the production steps would be skipped")
    if cuda is not True:
        problems.append("no CUDA GPU: SALE would be extremely slow or skipped")
    if len(secret_entries()) < 1:
        problems.append("secret_list.txt has no entries: add the partner short names and internal folder names (see CARA_RUN.md)")
    for n in ("OUT_CORE", "OUT_EXTENDED", "OUT_REVISION"):
        o = S(n, "")
        if o and os.path.isdir(os.path.join(o, "raw")) and glob.glob(os.path.join(o, "raw", "*.json")) and not os.path.exists(os.path.join(HERE, "run_all_status")):
            problems.append(f"{n} = {o} already contains results but this code folder has not run yet: use a NEW, empty output folder")
    if problems:
        say("[check] PROBLEMS:\n  - " + "\n  - ".join(problems))
        sys.exit("Fix the problems above (edit local_config.py) and run again, or add --ignore-checks to continue anyway.")


# ============================================================ 4. steps
def steps(quick, LC):
    S = lambda n, d: getattr(LC, n, d)
    oc, oe, orv, pk = S("OUT_CORE", "results_core"), S("OUT_EXTENDED", "results_extended"), S("OUT_REVISION", "results_revision"), S("CSV_PACKAGE", "csv_package")
    if quick:
        d = "_DEMO"
        return [
            ("core", "core: LotBench, WM-811K and report on fake data", [PY, "core_experiments.py", "--quick-test", "--parts", "lotbench,wmlot,report"]),
            ("extended", "extended: synthetic production line and small benchmarks", [PY, "extended_experiments.py", "--quick-test"]),
            ("revision", "revision: all four parts on small data", [PY, "revision_experiments.py", "--quick-test"]),
            ("export", "CSV export of the demo runs", [PY, "export_csv.py", "--package", pk + d, "--json-run", oc + d, "--json-run", oe + d, "--json-run", orv + d]),
            ("release", "release folder + leak check on the demo package",
             [PY, "build_release.py", "--package", pk + d, "--out", os.path.join(os.path.dirname(os.path.abspath(pk)), "release_DEMO"),
              "--overwrite", "--secret-list", os.path.join(HERE, "secret_list.txt")]),
        ]
    secret = os.path.join(HERE, "secret_list.txt")
    return [
        ("core_lotbench", "core: LotBench-811K seeds 10-12 (localization, attribution, retrieval) ~5-6 h",
         [PY, "core_experiments.py", "--parts", "lotbench", "--seeds", "10,11,12"]),
        ("core_wmlot", "core: WM-811K lot task seeds 10-12 (attribution, retrieval) ~1 h",
         [PY, "core_experiments.py", "--parts", "wmlot", "--seeds", "10,11,12"]),
        ("core_production", "core: three production lines (CV, same coaters, LOTO, shortcut tests, permutations) ~6-12 h",
         [PY, "core_experiments.py", "--parts", "plantb,diesaw,faba"]),
        ("core_report", "core: tables, figures, statistics, REPORT_core.md ~10 min", [PY, "core_experiments.py", "--parts", "report"]),
        ("extended_production", "extended: production lines (variants, MCE, LOTO, time split, monthly) ~3-6 h",
         [PY, "extended_experiments.py", "--parts", "plantb,diesaw,faba"]),
        ("extended_benchmarks", "extended: LotBench-811K and WM-811K, 10 seeds (Tables 7-10) ~18 h",
         [PY, "extended_experiments.py", "--parts", "lotbench,wmlot"]),
        ("revision", "revision: same coaters, learning curve, SALE-lite/MCE variants, ablations on 3 seeds ~45-50 h",
         [PY, "revision_experiments.py", "--parts", "samecoater,curve,bench,ablation", "--previous", oe]),
        ("export", "CSV package: inputs, process files and outputs of all runs ~1-2 h",
         [PY, "export_csv.py", "--package", pk, "--input-public", "--input-production", "--json-run", oc, "--json-run", oe, "--json-run", orv]),
        ("release", "release folder + leak check + zip",
         [PY, "build_release.py", "--package", pk, "--out", os.path.join(os.path.dirname(os.path.abspath(pk)), "release_v20"),
          "--overwrite", "--zip", "--secret-list", secret]),
    ]


def main():
    ap = argparse.ArgumentParser(description="Run the whole pipeline (setup included)")
    ap.add_argument("--quick-test", action="store_true"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None); ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--redo", default=None, help="steps to run again even if finished (comma separated)")
    ap.add_argument("--setup-only", action="store_true"); ap.add_argument("--ignore-checks", action="store_true")
    a = ap.parse_args()
    tag = "_DEMO" if a.quick_test else ""
    logf = open(os.path.join(HERE, f"run_all_log{tag}.txt"), "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    say(f"\n##### run_all.py started {time.strftime('%Y-%m-%d %H:%M')} ({'quick test' if a.quick_test else 'full run'}); Python {sys.version.split()[0]} at {PY}")
    ensure_local_files(say)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    LC = importlib.import_module("local_config")
    st = steps(a.quick_test, LC); names = [s[0] for s in st]
    if a.list:
        for n, desc, cmd in st:
            print(f"{n:22s} {desc}\n{'':22s} {' '.join(cmd[1:])}")
        return
    ensure_packages(say)
    preflight(LC, a.quick_test or a.ignore_checks, say)
    if a.setup_only:
        say("Setup finished."); return
    sel = [n for n in (a.only.split(",") if a.only else names) if n in names]
    if a.start:
        if a.start not in names:
            sys.exit(f"unknown step {a.start}; steps: {names}")
        sel = [n for n in sel if names.index(n) >= names.index(a.start)]
    redo = set((a.redo or "").split(","))
    status = os.path.join(HERE, "run_all_status" + tag); os.makedirs(status, exist_ok=True)
    t_all = time.time()
    for n, desc, cmd in st:
        if n not in sel:
            continue
        mark = os.path.join(status, n + ".done")
        if os.path.exists(mark) and n not in redo:
            say(f"=== {n}: already finished (use --redo {n} to repeat)"); continue
        say(f"\n=== {n}: {desc}\n    {' '.join(cmd[1:])}   [{time.strftime('%Y-%m-%d %H:%M')}]")
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                             env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"))
        for line in p.stdout:
            print(line, end="", flush=True); logf.write(line)
        rc = p.wait()
        if rc != 0:
            say(f"=== {n}: FAILED (exit code {rc}) after {(time.time() - t0) / 60:.0f} min. Send run_all_log{tag}.txt, or fix the "
                f"problem and run the same command again (finished steps are skipped).")
            sys.exit(rc)
        with open(mark, "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M"))
        say(f"=== {n}: done in {(time.time() - t0) / 60:.0f} min")
    if a.quick_test:
        say(f"\nQUICK TEST PASSED in {(time.time() - t_all) / 60:.0f} min. Now run:  python run_all.py")
    else:
        say(f"\nALL STEPS DONE in {(time.time() - t_all) / 3600:.1f} h. Send raw/ and REPORT*.md of {getattr(LC, 'OUT_CORE', '')}, "
            f"{getattr(LC, 'OUT_EXTENDED', '')}, {getattr(LC, 'OUT_REVISION', '')} for the manuscript update; upload only release.zip.")


if __name__ == "__main__":
    main()
