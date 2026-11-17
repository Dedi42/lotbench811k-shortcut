#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
far_theory.py - finite-sample theory for the realized false-alarm rate of split-conformal alarms (Section 5.5, S10).

Given one calibration on n normal lots and k = ceil((n+1)(1-eps)), the false-alarm probability of the threshold
follows Beta(n+1-k, k) (Vovk, 2012); the realized rate on m normal test lots adds binomial noise. The script simulates
this Beta-binomial model for the calibration and test sizes of the paper and reports
  * the 90% band of one repetition (production: five calibrations of a cross-validation repetition, pooled),
  * the predicted range of SALE for its number of repetitions (5th percentile of the minimum, 95th of the maximum).
Production calibration sizes are the medians reported in the paper; no production data are needed.
Usage: python far_theory.py [--sims 200000]
"""
import argparse, json, os
from math import ceil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# name: (n calibration lots, m normal test lots per calibration, calibrations per repetition, repetitions of SALE)
SETS = {"same": (26, 43, 5, 5), "saw": (39, 66, 5, 5), "faba": (44, 74, 5, 5), "lotbench": (2753, 2753, 1, 10), "wmlot": (518, 518, 1, 10)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sims", type=int, default=200000); ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); out = {}
    for key, (n, m, folds, r) in SETS.items():
        k = ceil((n + 1) * 0.95)
        p = rng.beta(n + 1 - k, k, size=(a.sims, r, folds))
        far = rng.binomial(m, p).sum(2) / (m * folds)
        out[key] = {"n": n, "m": m, "expected": 1 - k / (n + 1), "rep_band": np.percentile(far[:, 0], [5, 95]).tolist(),
                    "sale_pred": [float(np.percentile(far.min(1), 5)), float(np.percentile(far.max(1), 95))]}
        print(f"{key:9s} n={n:5d} expected {100 * out[key]['expected']:.1f}%  90% band of one repetition "
              f"{100 * out[key]['rep_band'][0]:.1f}-{100 * out[key]['rep_band'][1]:.1f}%  predicted SALE range "
              f"{100 * out[key]['sale_pred'][0]:.1f}-{100 * out[key]['sale_pred'][1]:.1f}%")
    os.makedirs(os.path.join(HERE, "reproduced"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "reproduced", "far_theory.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
