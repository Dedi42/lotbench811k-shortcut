#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fig6.py - draw Fig. 6 of the paper (two pitfalls on the production lines).
The production data are confidential, so this figure is drawn from the aggregated values reported in the paper
(Table 5, Online Resource 1, Table S9): pooled and within-tool AUROC on Plant B (repetition 1) and the AUROC of the
lot-summary forest and the ensemble under three validation schemes on the wafer saw and Fab A, with 95% intervals.
Output: reproduce/figures/Fig6.png and Fig6.pdf
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
fig = plt.figure(figsize=(W, 3.05))
Y0, H = 0.15, 0.70
axA = fig.add_axes([0.215, Y0, 0.345, H]); axB = fig.add_axes([0.695, Y0, 0.29, H])
for x, l in ((0.0, "a"), (0.605, "b")):
    fig.text(x, 0.985, l, fontsize=11.5, fontweight="bold", ha="left", va="top")
SUP, REF = RED, TEAL
rows = [("Lot-summary forest", 0.926, 0.627, "sup", True), ("SALE", 0.906, 0.587, "sup", True), ("Structured SALE", 0.881, 0.553, "sup", True),
        ("Wafer classifier", 0.860, 0.552, "sup", True), ("Isolation forest", 0.893, 0.670, "ref", True), ("Deep one-class", 0.946, 0.762, "ref", True),
        (r"Hotelling $T^2$", 0.818, 0.746, "ref", False)]
ax = axA
ax.axvspan(0.45, 0.5, color="#f1f2f4", zorder=0, lw=0)
ax.axvline(0.5, color=MUTED, lw=0.8, ls=":"); ax.text(0.475, 3.4, "chance", fontsize=6.8, color=INK2, ha="center", va="center", rotation=90)
ys = [6.4, 5.4, 4.4, 3.4, 1.9, 0.9, -0.1]
for (n, p, w, grp, sig), yy in zip(rows, ys):
    col = SUP if grp == "sup" else REF
    ax.plot([w, p], [yy, yy], color=col, alpha=0.22, lw=5, solid_capstyle="round", zorder=1)
    ax.scatter(p, yy, s=40, facecolor="white", edgecolor=INK, linewidth=1.4, zorder=3)
    ax.scatter(w, yy, s=40, color=col if sig else MUTED, zorder=3)
    d = p - w
    ax.text((p + w) / 2, yy + 0.30, f"−{d:.2f}" + ("" if sig else "  n.s."), fontsize=7.8, ha="center", va="bottom",
            color=(col if sig else INK2), fontweight="bold" if sig else "normal")
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows]); ax.tick_params(axis="y", length=0)
for t, r in zip(ax.get_yticklabels(), rows): pass
ax.set_ylim(-1.05, 7.25); ax.set_xlim(0.45, 1.0); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlabel("AUROC, Plant B coat/develop (repetition 1)")
# group brackets
for y0, y1, lab, col in ((3.05, 6.75, "supervised", SUP), (-0.45, 2.25, "referenced to\nnormal lots", REF)):
    xb = -0.50
    ax.plot([xb, xb], [y0, y1], color=col, lw=1.6, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.text(xb - 0.03, (y0 + y1) / 2, lab, transform=ax.get_yaxis_transform(), rotation=90, fontsize=7.4, color=col, ha="right", va="center",
            fontweight="bold", linespacing=0.95, clip_on=False)
ax.scatter([], [], s=34, facecolor="white", edgecolor=INK, linewidth=1.3, label="all tools (pooled)")
ax.scatter([], [], s=34, color=INK2, label="within tool (Eq. 4)")
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=7.2, handletextpad=0.2, columnspacing=1.0, borderaxespad=0.1)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)

# ---- b: validation schemes
ax = axB
prot = ["random\nCV", "time\nsplit", "month by\nmonth"]
lines = [("Wafer saw", "forest", BLUE, "s", "--", [(0.894, None, None), (0.944, .70, .99), (0.935, .78, .98)]),
         ("Wafer saw", "ensemble", RED, "s", "--", [(0.858, None, None), (0.834, .60, .94), (0.920, .76, .98)]),
         ("Fab A", "forest", BLUE, "o", "-", [(0.791, None, None), (0.447, .21, .71), (0.536, .33, .73)]),
         ("Fab A", "ensemble", RED, "o", "-", [(0.752, None, None), (0.555, .27, .81), (0.523, .32, .71)])]
ax.axhspan(0.15, 0.5, color="#f1f2f4", zorder=0, lw=0)
ax.axhline(0.5, color=MUTED, lw=0.8, ls=":"); ax.text(-0.42, 0.515, "chance", fontsize=6.8, color=INK2, ha="left", va="bottom")
for j, (line, det, col, mk, ls, data) in enumerate(lines):
    x = np.arange(3) + (-0.09 + 0.06 * j)
    for xi, (v, lo, hi) in zip(x, data):
        if lo is not None: ax.plot([xi, xi], [lo, hi], color=col, lw=0.8, alpha=0.35)
    ax.plot(x, [d[0] for d in data], color=col, lw=1.7 if det == "forest" else 1.2, ls=ls, marker=mk, ms=5.5, mfc="white" if mk == "s" else col,
            mec=col, mew=1.4, alpha=1 if det == "forest" else 0.75, label=f"{line}, {det}", zorder=3)
# key values (forest)
ax.text(-0.15, 0.791, "0.79", fontsize=7.2, color=BLUE, ha="right", va="center", fontweight="bold")
ax.text(1.0, 0.425, "0.45", fontsize=7.2, color=BLUE, ha="left", va="top", fontweight="bold")
ax.text(1.50, 0.30, "Fab A falls\nto chance\n(−0.34)", fontsize=7.2, color=RED, ha="center", va="center", fontweight="bold", linespacing=1.0)
ax.text(1.45, 1.005, "wafer saw holds (0.89 → 0.94)", fontsize=7.0, color=BLUE, ha="center", va="bottom")
ax.set_xticks(range(3)); ax.set_xticklabels(prot, linespacing=0.95); ax.set_xlim(-0.45, 2.4); ax.set_ylim(0.15, 1.05); ax.set_ylabel("AUROC")
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.tick_params(axis="x", length=0)
ax.legend(loc="lower left", bbox_to_anchor=(-0.22, 1.0), ncol=2, frameon=False, fontsize=6.8, handlelength=2.0, columnspacing=0.7, handletextpad=0.3, borderaxespad=0.1)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
save(fig, "Fig6")
print("Fig. 6 written to", D)
