# Plotting helpers and style of the paper figures (serif font, colours, small pictograms)
import numpy as np, matplotlib, matplotlib.font_manager
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch, Ellipse, Wedge, Arc
plt.rcParams.update({
    "font.family": next((f for f in ("TeX Gyre Termes", "Times New Roman", "Liberation Serif", "DejaVu Serif") if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}), "serif"), "mathtext.fontset": "stix", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.8, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#222222",
    "savefig.dpi": 600, "pdf.fonttype": 42, "figure.dpi": 150, "lines.solid_capstyle": "round"})
INK, INK2, MUTED, GRID = "#1d1d1d", "#4a4a4a", "#8c8c8c", "#e2e2e2"
BLUE, RED, ORANGE, GREEN, PURPLE, TEAL, GOLD = "#2c5d8f", "#b3302e", "#c8691c", "#3b8a3e", "#5e4b9c", "#2a8f84", "#b8860b"
CA, CB = "#a9c4e0", "#f2c9a0"            # chamber A / B wafer colours
PALE = {"blue": "#e8eef6", "red": "#f8e6e5", "green": "#e7f1e6", "gold": "#fbf3dc", "grey": "#f2f2f0", "purple": "#ece8f5"}
W = 6.3
import os
D = os.environ.get("FIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")) + os.sep
os.makedirs(D, exist_ok=True)

def canvas(h_in, xmax=100):
    fig = plt.figure(figsize=(W, h_in)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, xmax); ax.set_ylim(0, xmax * h_in / W); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax
def save(fig, name):
    fig.savefig(D + name + ".png", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(D + name + ".pdf", bbox_inches="tight", pad_inches=0.04, facecolor="white")
def panel(ax, letter, x=-0.16, y=1.03):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")
def T(ax, x, y, s, size=8.5, **k):
    k.setdefault("ha", "center"); k.setdefault("va", "center"); ax.text(x, y, s, fontsize=size, color=k.pop("color", INK), **k)
def rbox(ax, x, y, w, h, fc="white", ec=INK2, lw=1.0, r=1.0, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.3, ls="-", head=4.5, z=3, conn=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=f"-|>,head_length={head*0.8},head_width={head*0.5}", color=color, lw=lw,
                                 ls=ls, shrinkA=0, shrinkB=0, zorder=z, connectionstyle=conn or "arc3"))

def wafer(ax, cx, cy, r, pattern="none", seed=0, n=15, bad=RED, good="#dce6f0", edge=INK2, lw=0.8, z=3):
    """Die-grid wafer map with a WM-811K-like defect pattern."""
    rng = np.random.default_rng(seed); s = 2 * r / n
    for i in range(n):
        for j in range(n):
            x = cx - r + (i + 0.5) * s; y = cy - r + (j + 0.5) * s
            dx, dy = x - cx, y - cy; d = np.hypot(dx, dy) / r
            if d > 0.93: continue
            b = False
            if pattern == "center": b = d < 0.38 and rng.random() < 0.85
            elif pattern == "edge": b = d > 0.72 and rng.random() < 0.8
            elif pattern == "donut": b = 0.35 < d < 0.62 and rng.random() < 0.8
            elif pattern == "scratch": b = abs((dy) - 0.9 * (dx) + 0.1 * r) < 0.7 * s and d < 0.85
            elif pattern == "loc": b = np.hypot(dx - 0.42 * r, dy + 0.35 * r) < 0.33 * r and rng.random() < 0.85
            elif pattern == "random": b = rng.random() < 0.1
            ax.add_patch(Rectangle((x - s * 0.44, y - s * 0.44), s * 0.88, s * 0.88, fc=bad if b else good, ec="none", zorder=z))
    ax.add_patch(Circle((cx, cy), r, fc="none", ec=edge, lw=lw, zorder=z + 1))
    ax.add_patch(Wedge((cx, cy - r), r * 0.12, 0, 180, fc="white", ec=edge, lw=lw * 0.8, zorder=z + 2))   # notch

def carrier(ax, x, y, w, h, n=8, cols=(CA, CB), lock=None, z=3):
    """Open wafer carrier (FOUP) seen from the side with n wafers."""
    gap = h / (n + 0.6)
    for k in range(n):
        yy = y + gap * (k + 0.8)
        ax.add_patch(Ellipse((x + w / 2, yy), w * 0.84, gap * 0.95, fc=cols[k % len(cols)], ec=INK, lw=0.6, zorder=z))
    ax.plot([x, x, x + w, x + w], [y + h, y, y, y + h], color=INK2, lw=1.4, zorder=z + 1, solid_capstyle="butt")
    if lock is not None:
        lx, ly = x + w + 0.2, y + h * 0.55
        ax.add_patch(Arc((lx + 1.1, ly + 1.2), 1.6, 2.2, theta1=0, theta2=180, color=lock, lw=1.3, zorder=z + 2))
        ax.add_patch(Rectangle((lx, ly), 2.2, 1.6, fc=lock, ec=lock, zorder=z + 2))

def tool(ax, x, y, w, h, chambers=(CA, CB), labels=("A", "B"), hot=None, z=2, small=False):
    """Cluster tool: load port, transfer chamber with robot arm and process chambers."""
    rbox(ax, x, y, w, h, fc=PALE["grey"], ec=INK2, lw=1.0, r=1.2, z=z)
    cx, cy = x + w / 2, y + h * 0.42; rt = min(w, h) * 0.17
    ax.add_patch(Circle((cx, cy), rt, fc="white", ec=INK2, lw=0.9, zorder=z + 1))
    ax.plot([cx, cx + rt * 0.75], [cy, cy + rt * 0.45], color=INK2, lw=1.4, zorder=z + 2)
    ax.add_patch(Circle((cx, cy), rt * 0.18, fc=INK2, ec="none", zorder=z + 3))
    rc = min(w, h) * 0.2; pos = [(cx - w * 0.27, cy + h * 0.3), (cx + w * 0.27, cy + h * 0.3)]
    for k, ((px, py), col) in enumerate(zip(pos, chambers)):
        ec = RED if hot == k else INK2
        ax.plot([cx, px], [cy, py], color=INK2, lw=0.8, zorder=z)
        ax.add_patch(Circle((px, py), rc, fc="white", ec=ec, lw=1.6 if hot == k else 0.9, zorder=z + 1))
        ax.add_patch(Circle((px, py), rc * 0.62, fc=col, ec=INK, lw=0.5, zorder=z + 2))
        if not small and labels: T(ax, px, py, labels[k], 7.5, color=INK, zorder=z + 3, fontweight="bold")
    ax.add_patch(Rectangle((cx - w * 0.16, y + h * 0.04), w * 0.32, h * 0.12, fc="white", ec=INK2, lw=0.8, zorder=z + 1))
    ax.plot([cx, cx], [y + h * 0.16, cy - rt], color=INK2, lw=0.8, zorder=z)

def trace(ax, x, y, w, h, seed=1, spike=None, z=3, frame=True):
    if frame: rbox(ax, x, y, w, h, fc="white", ec=INK2, lw=0.8, r=0.5, z=z)
    t = np.linspace(0, 1, 120); rng = np.random.default_rng(seed)
    v = 0.5 + 0.25 * np.sin(t * 18) * np.exp(-t * 1.5) + 0.05 * rng.standard_normal(120)
    if spike is not None: v[int(spike * 120):int(spike * 120) + 6] += 0.35
    ax.plot(x + 0.06 * w + t * 0.88 * w, y + h * (0.12 + 0.76 * np.clip(v, 0, 1)), color=BLUE, lw=0.8, zorder=z + 1)

def route(ax, x, y, w, n=5, hot=None, r=0.9, col=GREEN, z=3):
    xs = np.linspace(x + r, x + w - r, n)
    ax.plot(xs, [y] * n, color=col, lw=1.0, zorder=z)
    for k, xx in enumerate(xs):
        ax.add_patch(Circle((xx, y), r, fc=RED if hot == k else "white", ec=RED if hot == k else col, lw=1.0, zorder=z + 1))

def tag(ax, x, y, w, h, text, size=7.5, fc="white", ec=INK2):
    ax.add_patch(Polygon([[x, y], [x + w - h * 0.45, y], [x + w, y + h / 2], [x + w - h * 0.45, y + h], [x, y + h]], closed=True, fc=fc, ec=ec, lw=0.8, zorder=3))
    ax.add_patch(Circle((x + w - h * 0.5, y + h / 2), h * 0.12, fc="white", ec=ec, lw=0.6, zorder=4))
    T(ax, x + (w - h * 0.45) / 2, y + h / 2, text, size, zorder=5)

def slots(ax, x, y, w, h, prof, z=3, col=RED, n=25, base="#e6e6e3"):
    sw = w / n
    for s in range(n):
        v = prof(s)
        ax.add_patch(Rectangle((x + s * sw, y), sw * 0.8, h, fc=col if v > 0 else base, alpha=(0.25 + 0.75 * v) if v > 0 else 1, ec="none", zorder=z))
