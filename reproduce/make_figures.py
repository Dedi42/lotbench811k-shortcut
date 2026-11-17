#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py - redraw Figs. 4, 5 and 7 of the paper from the reproduced public results.
Run reproduce_public_results.py and far_theory.py first. Figures are written to reproduce/figures/ (PNG 600 dpi + PDF).
Fig. 7 shows the observed false-alarm rates of SALE on the production lines as reported in the paper (aggregated values;
the production data are confidential); everything else is recomputed from the released files.
"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
from scipy import stats as st
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "reproduced", "public_results.json")))
FT = json.load(open(os.path.join(HERE, "reproduced", "far_theory.json")))
RES = os.path.join(HERE, "..", "results", "public")
# observed realized FAR of SALE on the production lines (paper, Table 4): mean, min, max
PROD_OBS = {"same": (0.034, 0.023, 0.047), "saw": (0.060, 0.030, 0.110), "faba": (0.040, 0.016, 0.068)}
for k in ("lotbench", "wmlot"):
    f = S[k]["recall"]["SALE"]["far_per_seed"]; FT[k]["sale_mean"] = float(np.mean(f)); FT[k]["sale_obs"] = [min(f), max(f)]
for k, (mu, lo, hi) in PROD_OBS.items():
    FT[k]["sale_mean"] = mu; FT[k]["sale_obs"] = [lo, hi]

# ---------------------------------------------------------------- Fig. 4
COL = {"MCE": RED, "B3-RF": BLUE, "SALE": ORANGE, "SALE-lite": GREEN, "B9": TEAL, "B1": PURPLE}
NM = {"MCE": "Ensemble", "B3-RF": "Lot-summary forest", "SALE": "SALE", "SALE-lite": "SALE-lite", "B9": "Attention MIL", "B1": r"Hotelling $T^2$"}
fig = plt.figure(figsize=(W, 5.75))
# fixed grid: left column x0 = 0.085, right column x0 = 0.585; top row y0 = 0.585, bottom row y0 = 0.055
axA = fig.add_axes([0.085, 0.585, 0.385, 0.365]); axB = fig.add_axes([0.585, 0.585, 0.385, 0.365])
axC = fig.add_axes([0.0, 0.03, 0.47, 0.41]); axD = fig.add_axes([0.735, 0.075, 0.235, 0.335])
LX, RX, TY, BY = 0.0, 0.50, 0.985, 0.475          # panel-letter anchors (figure coords)
for x, y, l in ((LX, TY, "a"), (RX, TY, "b"), (LX, BY, "c"), (RX, BY, "d")):
    fig.text(x, y, l, fontsize=11.5, fontweight="bold", ha="left", va="top")

def roc(ax, kind, title, key, ref):
    R = S[kind]["roc"]; f = np.array(R["fpr"])
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=":")
    ax.axvline(0.05, color=INK2, lw=0.8, ls="--", zorder=1)
    order = sorted(["B1", "B9", "SALE", "SALE-lite", "B3-RF", "MCE"], key=lambda m: -S[kind]["all"][m]["mean"])
    for m in order:
        a = S[kind]["all"][m]["mean"]; main = m in (key, ref)
        ax.plot(f, R[m], color=COL[m], lw=2.3 if m == key else (1.7 if m == ref else 0.95), alpha=1 if main else 0.55,
                zorder=5 if m == key else (4 if m == ref else 2), label=f"{NM[m]}  {a:.3f}")
    rk, rr = S[kind]["recall"][key]["recall"], S[kind]["recall"][ref]["recall"]
    for r, m in ((rk, key), (rr, ref)):
        ax.plot([0, 0.05], [r, r], color=COL[m], lw=0.8, ls=(0, (2, 1.5)), zorder=3)
        ax.scatter(0.05, r, s=40, facecolor="white", edgecolor=COL[m], linewidth=1.7, zorder=6)
    # value ticks on the y axis
    ax.set_yticks([0, 0.2, 0.4, 0.8, 1.0] if rk > 0.7 else [0, 0.2, 0.4, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.8", "1.0"])
    gap = max(0.0, 0.06 - abs(rk - rr)) / 2                   # keep the two value labels apart when the rates are close
    for r, m, sgn in ((rk, key, 1), (rr, ref, -1)):
        ax.text(-0.018, r + sgn * gap, f"{r:.2f}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8, color=COL[m], fontweight="bold")
    leg = ax.legend(loc="lower right", frameon=False, fontsize=7.3, handlelength=1.5, title="AUROC", title_fontsize=7.6, borderaxespad=0.1, labelspacing=0.3)
    for t, m in zip(leg.get_texts(), order):
        if m in (key, ref): t.set_fontweight("bold"); t.set_color(COL[m])
        else: t.set_color(INK2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.0); ax.set_xticks([0, .2, .4, .6, .8, 1]); ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_xlabel("False-alarm rate"); ax.set_ylabel("Abnormal lots held", labelpad=14)
    ax.text(0, 1.035, title, transform=ax.transAxes, fontsize=9.5, ha="left", va="bottom")
    ax.text(1, 1.035, rf"+{(rk - rr) * 100:.1f} pts held at $\varepsilon$ = 5%" + ("" if S[kind]["all"][key].get("p_nb_holm", 1) < 0.05 else " (n.s.)"), transform=ax.transAxes, fontsize=8.2,
            ha="right", va="bottom", color=COL[key], fontweight="bold")
roc(axA, "lotbench", "LotBench-811K", "MCE", "B3-RF")
roc(axB, "wmlot", "WM-811K lot task", "MCE", "B3-RF")

# ---- c: hold/release decisions at eps = 5 %
axC.set_xlim(0, 48); axC.set_ylim(0, 30); axC.axis("off")
DEC = S["lotbench"]["decisions"]
grids = tuple((nm, col, (DEC[m]["release_normal"], DEC[m]["hold_normal"], DEC[m]["release_abnormal"], DEC[m]["hold_abnormal"])) for nm, col, m in (("Lot-summary forest", BLUE, "B3-RF"), ("Ensemble", RED, "MCE")))
for g, (nm, col, (tn, fp, fn, tp)) in enumerate(grids):
    x0 = 10.5 + g * 18.5
    T(axC, x0 + 8.0, 27.2, nm, 9, color=col, fontweight="bold")
    cells = [(0, 1, tn, tn / (tn + fp), PALE["blue"], False), (1, 1, fp, fp / (tn + fp), PALE["red"], False),
             (0, 0, fn, fn / (fn + tp), PALE["red"], False), (1, 0, tp, tp / (fn + tp), PALE["blue"], True)]
    for cx_, cy_, n, p, fc, key in cells:
        hl = key and g == 1
        axC.add_patch(Rectangle((x0 + cx_ * 8.2, 8.6 + cy_ * 8.4), 7.8, 8.0, fc=fc, ec=RED if hl else "none", lw=1.6 if hl else 0))
        T(axC, x0 + cx_ * 8.2 + 3.9, 8.6 + cy_ * 8.4 + 5.0, f"{n:,}", 8.8, fontweight="bold", color=RED if hl else INK)
        T(axC, x0 + cx_ * 8.2 + 3.9, 8.6 + cy_ * 8.4 + 2.4, f"{p:.0%}", 8.2, color=RED if hl else INK2, fontweight="bold" if hl else "normal")
    T(axC, x0 + 3.9, 7.2, "release", 8, color=INK2); T(axC, x0 + 12.1, 7.2, "hold", 8, color=INK2)
carrier(axC, 2.6, 19.9, 3.6, 4.6, n=4, cols=(CA,)); T(axC, 4.4, 18.3, "normal", 7.8, color=INK2)
carrier(axC, 2.6, 11.5, 3.6, 4.6, n=4, cols=("#e9b8b5",)); T(axC, 4.4, 9.9, "abnormal", 7.8, color=INK2)
T(axC, 29.0, 3.0, f"Ensemble: +{DEC['MCE']['hold_abnormal'] - DEC['B3-RF']['hold_abnormal']} abnormal lots held,  {DEC['B3-RF']['hold_normal'] - DEC['MCE']['hold_normal']} fewer false alarms", 8.2, color=RED, fontweight="bold")

# ---- d: share held per fault mechanism
names = ["faulty chamber", "post-maintenance", "within-lot drift", "upstream tool", "material batch", "intermittent"]
pb = S["lotbench"]["recall"]["B3-RF"]["per"]; pm = S["lotbench"]["recall"]["MCE"]["per"]
y = np.arange(6)[::-1]
vb = [pb[f"F{i+1}"] for i in range(6)]; vm = [pm[f"F{i+1}"] for i in range(6)]
axD.barh(y + 0.19, vb, height=0.36, color=BLUE, label="Lot-summary forest")
axD.barh(y - 0.19, vm, height=0.36, color=RED, label="Ensemble")
for yy, a, b in zip(y, vb, vm):
    T(axD, a + 0.02, yy + 0.19, f"{a:.2f}", 7.2, color=BLUE, ha="left")
    T(axD, b + 0.02, yy - 0.19, f"{b:.2f}", 7.2, color=RED, ha="left", fontweight="bold")
    d = b - a; big = d >= 0.05
    T(axD, 1.33, yy, f"+{d:.2f}" if big else "±0", 7.8, color=RED if big else INK2, ha="center", fontweight="bold" if big else "normal")
T(axD, 1.33, 5.75, "Δ", 8.2, color=INK, ha="center", fontweight="bold")
axD.set_yticks(y); axD.set_yticklabels(names); axD.set_xlim(0, 1.45); axD.set_xticks([0, .5, 1]); axD.set_xticklabels(["0", "0.5", "1.0"])
axD.set_ylim(-0.6, 5.95); axD.spines["bottom"].set_bounds(0, 1.0)
axD.set_xlabel(r"Abnormal lots held at $\varepsilon = 5\%$", x=0.36); axD.tick_params(axis="y", length=0)
axD.legend(loc="lower left", bbox_to_anchor=(-0.62, 1.03), ncol=2, frameon=False, fontsize=7.6, handlelength=1.0, columnspacing=1.0, borderaxespad=0)
save(fig, "Fig4")

# ---------------------------------------------------------------- Fig. 5
lc = pd.read_csv(os.path.join(RES, "results_revision_v20", "learning_curve", "learning_curve_runs.csv"))
diffs = [r["diff"] for r in S["learning_curve"]]
pv = [r["p_paired_t"] for r in S["learning_curve"]]
fig = plt.figure(figsize=(W, 2.75))
Y0, H = 0.17, 0.72
axA = fig.add_axes([0.075, Y0, 0.235, H]); axB = fig.add_axes([0.405, Y0, 0.235, H]); axC = fig.add_axes([0.80, Y0, 0.19, H])
for x, l in ((0.0, "a"), (0.333, "b"), (0.666, "c")):
    fig.text(x, 0.985, l, fontsize=11.5, fontweight="bold", ha="left", va="top")
def curves(ax, met, ylim, lab, top=True):
    ax.axvspan(5, 19, color=PALE["red"], zorder=0, lw=0)
    ax.text(9.7, ylim[0] + (ylim[1] - ylim[0]) * (0.86 if top else 0.04), "production\nfolds", fontsize=7, color=RED, ha="center", va="top" if top else "bottom", linespacing=0.95)
    ax.plot([50, 50], [ylim[0], ylim[0] + (ylim[1] - ylim[0]) * (0.93 if top else 0.66)], color=INK2, lw=0.9, ls="--", zorder=1)
    out = {}
    for det, col, nm in (("B3-RF", BLUE, "Lot-summary forest"), ("MCE-prod", RED, "Ensemble")):
        g = lc[lc.detector == det].groupby("train_fraction"); n = g["n_abnormal_train"].mean(); m = g[met].mean(); s = g[met].std(ddof=1); kk = g.size()
        h = st.t.ppf(.975, kk - 1) * s / np.sqrt(kk)
        ax.fill_between(n, m - h, m + h, color=col, alpha=0.13, lw=0)
        ax.plot(n, m, color=col, lw=1.8, marker="o", ms=3.8, label=nm, zorder=3); out[det] = (n.values, m.values)
    ax.set_xscale("log"); ax.set_xticks([5, 20, 50, 200, 1000]); ax.set_xticklabels(["5", "20", "50", "200", "1000"]); ax.minorticks_off()
    ax.set_xlim(3.5, 1700); ax.set_ylim(*ylim); ax.set_ylabel(lab); ax.set_xlabel("Abnormal training lots")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    return out
o = curves(axA, "AUROC", (0.60, 1.0), "AUROC", top=True)
axA.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
n = o["B3-RF"][0]
for xi, yv, txt, ha in ((19.5, 0.615, "Δ −0.003\n(n.s.)", "right"), (45, 0.800, "Δ +0.024*", "right"),
                        (221, 0.902, "+0.028*", "center"), (1108, 0.928, "+0.015*", "center")):
    axA.text(xi, yv, txt, fontsize=6.9, color=INK2 if "n.s." in txt else RED, fontweight="normal" if "n.s." in txt else "bold",
             ha=ha, va="bottom", linespacing=1.0)
axA.text(54, 0.975, "higher on\nevery seed", fontsize=6.6, color=INK2, fontweight="bold", ha="left", va="center", linespacing=1.0)

o = curves(axB, "recall5", (0.0, 0.8), r"Abnormal lots held at $\varepsilon = 5\%$")
nb, mb = o["B3-RF"]; nm_, mm = o["MCE-prod"]
axB.text(nm_[0] * 0.86, mm[0], f"{mm[0]:.2f}", fontsize=7.4, color=RED, fontweight="bold", ha="right", va="center")
axB.text(nb[0] * 0.86, mb[0], f"{mb[0]:.2f}", fontsize=7.4, color=BLUE, fontweight="bold", ha="right", va="center")
axB.text(nm_[5], mm[5] + 0.035, f"{mm[5]:.2f}", fontsize=7.4, color=RED, fontweight="bold", ha="center", va="bottom")
axB.text(nb[5], mb[5] - 0.04, f"{mb[5]:.2f}", fontsize=7.4, color=BLUE, fontweight="bold", ha="center", va="top")
axA.legend(loc="lower left", bbox_to_anchor=(-0.03, 1.0), ncol=2, frameon=False, fontsize=6.9, handlelength=1.0, handletextpad=0.4, columnspacing=0.8, borderaxespad=0.1)
axB.text(50 * 1.08, 0.75, "higher on\nevery seed", fontsize=6.6, color=INK2, fontweight="bold", ha="left", va="center", linespacing=1.0)
# c: delta to forest
ax = axC
items = [("MCE", "Ensemble"), ("MCE-without-SALE", "without SALE"), ("SALE-lite", "SALE-lite"), ("SALE", "SALE")]
for kind, col, mk, off, lab in (("lotbench", INK, "s", 0.17, "LotBench-811K"), ("wmlot", PURPLE, "o", -0.17, "WM-811K lot task")):
    for i, (m, nmx) in enumerate(items):
        r = S[kind]["all"][m]; yy = len(items) - 1 - i + off; sig = (r["p_nb_holm"] if m in ("MCE", "SALE") else r["p_nb"]) < 0.05   # Holm in the detector family, unadjusted for second-run variants
        ax.plot([r["lo_nb"], r["hi_nb"]], [yy, yy], color=col, lw=1.2)
        ax.scatter(r["diff"], yy, s=26, marker=mk, color=col if mk == "o" else "white", edgecolor=col, linewidth=1.3, zorder=3, label=lab if i == 0 else None)
        ax.text(0.1, yy, f"{r['diff']:+.3f}".replace("-", "−"), fontsize=7, color=col, fontweight="bold" if sig else "normal", ha="right", va="center",
                transform=ax.get_yaxis_transform() if False else ax.transData)
ax.axvline(0, color=BLUE, lw=1.0); ax.text(0.003, 3.7, "forest", fontsize=7.2, color=BLUE, ha="left", va="center")
ax.set_yticks(range(4)); ax.set_yticklabels([x for _, x in items[::-1]]); ax.set_ylim(-0.55, 4.0)
ax.set_xlim(-0.095, 0.1); ax.set_xticks([-0.05, 0, 0.05]); ax.set_xticklabels(["−0.05", "0", "+0.05"])
ax.spines["bottom"].set_bounds(-0.095, 0.05)
ax.set_xlabel("ΔAUROC vs forest", x=0.37); ax.tick_params(axis="y", length=0); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=6.9, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1)
save(fig, "Fig5")

# ---------------------------------------------------------------- Fig. 7
fig = plt.figure(figsize=(W * 0.82, 2.55))
ax = fig.add_axes([0.26, 0.17, 0.53, 0.66])
rows = [("LotBench-811K", "lotbench", "≈ 2,750", PURPLE), ("WM-811K lot task", "wmlot", "≈ 520", PURPLE),
        ("Plant B same coaters", "same", "26", BLUE), ("Plant B wafer saw", "saw", "39", BLUE), ("Fab A PECVD", "faba", "44", BLUE)]
ys = [5.0, 4.0, 2.3, 1.3, 0.3]
ax.axvline(5.0, color=INK, lw=1.0, zorder=1)
ax.text(5.0, 5.75, r"target $\varepsilon$ = 5%", fontsize=7.4, color=INK, ha="center", va="bottom")
for (n, k, ncal, col), yy in zip(rows, ys):
    r = FT[k]; lo, hi = [100 * v for v in r["sale_pred"]]; olo, ohi = [100 * v for v in r["sale_obs"]]; m = 100 * r["sale_mean"]
    ax.add_patch(Rectangle((lo, yy - 0.28), hi - lo, 0.56, fc="#e3e6eb", ec="none", zorder=0))
    ax.plot([olo, ohi], [yy, yy], color=col, lw=3.2, solid_capstyle="round", zorder=2)
    ax.scatter(m, yy, s=48, color="white", edgecolor=INK, linewidth=1.5, zorder=3)
    ax.text(12.9, yy, f"{m:.1f}%", fontsize=7.8, color=INK, ha="right", va="center", fontweight="bold", clip_on=False)
    ax.text(13.1, yy, f"({olo:.1f}–{ohi:.1f})", fontsize=7.2, color=col, ha="left", va="center", clip_on=False)
    ax.text(-0.35, yy + 0.13, n, fontsize=8.0, color=INK, ha="right", va="center", clip_on=False)
    ax.text(-0.35, yy - 0.23, (f"n {ncal}" if ncal.startswith("≈") else f"n = {ncal}") + " calibration lots", fontsize=6.6, color=INK2, ha="right", va="center", clip_on=False)
ax.text(12.9, 5.75, "mean", fontsize=7.0, color=INK2, ha="right", va="bottom", clip_on=False)
ax.text(13.1, 5.75, "(range)", fontsize=7.0, color=INK2, ha="left", va="bottom", clip_on=False)
for y0, y1, lab, col in ((3.6, 5.4, "public", PURPLE), (-0.1, 2.7, "production", BLUE)):
    ax.plot([-5.9, -5.9], [y0, y1], color=col, lw=1.6, clip_on=False)
    ax.text(-6.15, (y0 + y1) / 2, lab, fontsize=7.4, color=col, rotation=90, ha="right", va="center", fontweight="bold", clip_on=False)
ax.set_yticks([]); ax.set_ylim(-0.45, 5.7); ax.set_xlim(0, 12)
ax.set_xticks([0, 2.5, 5, 7.5, 10]); ax.set_xticklabels(["0", "2.5", "5", "7.5", "10"])
ax.set_xlabel("Realized false-alarm rate of SALE (%)"); ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
# legend
lg = fig.add_axes([0.10, 0.905, 0.88, 0.075]); lg.axis("off"); lg.set_xlim(0, 10); lg.set_ylim(0, 1)
lg.add_patch(Rectangle((0.0, 0.22), 0.45, 0.56, fc="#e3e6eb", ec="none")); lg.text(0.58, 0.5, "range predicted by theory (Beta-binomial)", fontsize=7.2, color=INK2, va="center")
lg.plot([5.6, 6.1], [0.5, 0.5], color=BLUE, lw=3.2, solid_capstyle="round"); lg.text(6.25, 0.5, "observed range", fontsize=7.2, color=INK2, va="center")
lg.scatter([8.05], [0.5], s=36, color="white", edgecolor=INK, linewidth=1.3); lg.text(8.25, 0.5, "observed mean", fontsize=7.2, color=INK2, va="center")
save(fig, "Fig7")
print("figures written to", D)
