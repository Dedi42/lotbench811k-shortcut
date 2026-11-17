# Confidentiality of the production data

The paper evaluates three production lines (Plant B coat/develop, Plant B wafer saw, Fab A PECVD). Their data are
confidential under agreements with the manufacturing partners. This record therefore releases **only public-data
material and code**.

## Not included

| Excluded | Why |
|---|---|
| Raw production exports, labels and per-lot production scores | confidential partner data |
| Real readers of the production files (`production_loaders.py` of the authors) | reveal file formats and system names; replaced by a documented stub with the same interface |
| Tool, chamber, lot, recipe, product and device identifiers, alias keys | confidential; the paper uses aliases (Plant B, Fab A, tool 1–4) |
| Raw result files of the production runs (`raw/*.json`, reports) | contain identifiers |
| `local_config.py`, `secret_list.txt`, run logs and status folders | local paths and the private word list |
| WM-811K and MixedWM38 raw files | redistributed only by their providers |

Production results appear only in aggregated form in the paper (AUROC, recall and false-alarm rates per line).

## Included

* experiment code with anonymised names only (`code/`);
* outputs of the paper run on public data (`results/public/`): the score of every LotBench-811K and WM-811K test,
  calibration and blending lot for every detector and seed, learning-curve and ablation runs, paper tables;
* reproduction scripts and figures (`reproduce/`).

## How the release was checked

`tools/release_scan.py` was run on the complete folder before upload, with the authors' private word list (partner
names, internal folder and server names, label-file names, identifier prefixes; kept outside the release and never
printed) and the private identifier patterns (passed on the command line, not stored in the release). It checks every file, including compressed score files, for the private strings, internal equipment-ID
patterns, absolute Windows/UNC/home paths, e-mail and IP addresses, files that must never be released and files of
unknown type.

Result for version 1.0.0: **176 files, 355,308 text lines, 0 findings**; the figure files were reviewed visually (anonymised line names only).

To check a modified release yourself:

```bash
python tools/release_scan.py <release_folder> --secret-list <private_list_outside_release.txt> --id-regex '<private ID pattern>' --report <report_outside_release.txt>
```

The authors' own pipeline adds a second check: `code/build_release.py` blocks every original value in the
confidential alias-key tables of a run, in addition to the private word list.
