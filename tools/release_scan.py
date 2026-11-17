#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
release_scan.py - scan a release folder for confidential content before upload.

Checks every file (text, .csv.gz, .json, .py, .md, ...):
  * strings from a PRIVATE word list (--secret-list; partner names, internal folders, tool/lot ID prefixes ...),
    kept outside the release and never printed: hits are reported only as file, line and list index;
  * internal identifier patterns given privately with --id-regex (no default, so the release does not reveal them);
  * absolute Windows / UNC / Unix home paths, e-mail addresses (except --allow-email), IP addresses;
  * files that must never be released (local_config.py, secret_list.txt, run logs, confidential folders,
    real production loaders recognised by their content);
  * binary or unknown file types (listed for manual review).
Exit code 0 = no findings.
Usage: python tools/release_scan.py <release_dir> --secret-list <private_list.txt> --report <file outside release>
"""
import argparse, gzip, os, re, sys

NEVER_NAMES = {"local_config.py", "secret_list.txt"}
NEVER_PARTS = ("confidential", "alias_keys", "INTERNAL_ONLY", "run_all_log", "run_all_status")
TEXT = (".py", ".md", ".txt", ".csv", ".json", ".cff", ".cfg", ".toml", ".yml", ".yaml", ".gz", ".gitignore", ".sh", ".bat")
OK_BINARY = (".png", ".pdf", ".tif", ".tiff", ".jpg")
PATTERNS = {"Windows path": r"\b[A-Za-z]:\\[^\s'\"]+", "UNC path": r"\\\\[A-Za-z0-9_.-]+\\[^\s'\"]+",
            "Unix home or temp path": r"(/home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/|/mnt/[A-Za-z0-9_.-]+/)",
            "e-mail": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "IP address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--secret-list", default=None); ap.add_argument("--report", default=None)
    ap.add_argument("--id-regex", action="append", default=[], help="private regular expression for internal IDs (repeatable)"); ap.add_argument("--allow-email", action="append", default=[])
    a = ap.parse_args()
    words = []
    if a.secret_list:
        words = [w.strip() for w in open(a.secret_list, encoding="utf-8") if w.strip() and not w.startswith("#")]
    rx_w = [re.compile(re.escape(w), re.I) for w in words]
    rx = {k: re.compile(v) for k, v in PATTERNS.items()}
    for n_, r_ in enumerate(a.id_regex):
        rx[f"private ID pattern #{n_ + 1}"] = re.compile(r_)
    hits, n_files, n_lines, review = [], 0, 0, []
    for dp, dn, fs in os.walk(a.root):
        for f in fs:
            p = os.path.join(dp, f); rel = os.path.relpath(p, a.root); n_files += 1
            if f in NEVER_NAMES or any(s in rel for s in NEVER_PARTS):
                hits.append((rel, 0, "file must never be released")); continue
            low = f.lower()
            if low.endswith(OK_BINARY):
                review.append(rel); continue
            if not (low.endswith(TEXT) or f in ("LICENSE", ".zenodo.json", ".gitignore")):
                hits.append((rel, 0, "unknown file type: review manually")); continue
            op = gzip.open if low.endswith(".gz") else open
            with op(p, "rt", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    n_lines += 1
                    for j, r in enumerate(rx_w):
                        if r.search(line):
                            hits.append((rel, i, f"private word-list entry #{j + 1}"))
                    for k, r in rx.items():
                        m = r.search(line)
                        if not m:
                            continue
                        if k == "e-mail" and any(e in m.group(0) for e in a.allow_email):
                            continue
                        if k == "IP address" and low.endswith((".csv", ".gz")):
                            continue                          # numeric columns can look like IP addresses
                        hits.append((rel, i, k))
            if f == "production_loaders.py":
                txt = open(p, encoding="utf-8", errors="replace").read()
                if "def load_plantb" in txt and "_MSG" not in txt:
                    hits.append((rel, 0, "looks like the REAL production loaders, not the stub"))
    lines = [f"Scanned {n_files} files, {n_lines} text lines; {len(words)} private word-list entries; {len(a.id_regex)} private ID patterns",
             f"Findings: {len(hits)}"] + [f"  {r}:{i}  {k}" for r, i, k in hits] + \
            [f"Image/PDF files for visual review ({len(review)}):"] + [f"  {r}" for r in review]
    rep = "\n".join(lines); print(rep)
    if a.report:
        open(a.report, "w", encoding="utf-8").write(rep + "\n")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
