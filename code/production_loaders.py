#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
production_loaders.py (public stub) - the production data of the paper are confidential.

The three production lines (Plant B coat/develop, Plant B wafer saw, Fab A PECVD) cannot be released under the
agreements with the partners, and neither can the loaders that read their file formats. This stub keeps the interface
so that core_experiments.py runs on the public data and shows how a new data source can be connected.

A loader returns a dict with the arguments of core_experiments.LotData:
    name      : str, data-set name ("plantb", "diesaw", "faba", or your own)
    lots      : DataFrame, one row per lot. Required columns: lot (alias), y (0 = normal, 1..D = abnormal type),
                tool, time (datetime or NaT), month ("YYYY-MM" or "NA"), subset_same (bool), plus the context columns
    W         : DataFrame, one row per wafer. Required columns: li (row index in lots), slot (1..25), chamber,
                pm (maintenance counter, 0 if unknown), wbad (per-wafer outcome 0/1 or NaN), the feature columns,
                and the normalization keys listed in `keys`
    groups    : dict group -> list of feature columns (e.g. {"FDC": [...]})
    keys      : dict group -> list of key columns from fine to coarse (e.g. tool|chamber|recipe, then tool|chamber)
    ctx_cols  : list of context columns in lots (used only by the context-only shortcut test unless enabled)
    types     : list of abnormal type names (len = D)
    notes     : dict of counts for the report (no identifiers)
Return None if the data are not available.
"""

_MSG = ("The production data and their loaders are confidential and not part of the public release. "
        "Only aggregated results are reported; access to the data requires the written approval of the partners.")


def load_plantb(folder, label_csv, coater_same=None, min_cov=0.2):
    print("[plantb] " + _MSG)
    return None


def load_diesaw(folder, zip_path=None, k_bad=3, pct=95):
    print("[diesaw] " + _MSG)
    return None


def load_faba(folder, zthr=2.5):
    print("[faba] " + _MSG)
    return None
