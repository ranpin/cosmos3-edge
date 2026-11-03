#!/usr/bin/env python3
"""Shared figure style for all Alpamayo-Edge figures (CVPR-style typography).

Every figure in docs/figures/ imports this so fonts, sizes, spines, grid and
legends are identical across the whole document set.

Conventions (conference-paper style):
  * Serif type throughout: Times New Roman (Latin) + Songti SC (CJK fallback).
  * One type scale — no per-figure ad-hoc font sizes (see the SIZES block).
  * Titles are short and neutral (quantity + measurement conditions); results
    and interpretation belong in the figure caption, not inside the image.
  * Top/right spines removed, light dotted y-grid only, frameless legend.
  * 300 dpi output (print-grade).

Dual theme (2026-11-03): the report site has light/dark themes; every figure
renders in two variants — light (docs/figures/, the canonical files; a dark
run must NEVER overwrite them) and dark (docs/figures/dark/, background
#1a1a2e = the site's dark figure-card --bg2, palette brightened). Scripts
wrap hardcoded hex colors in C() for theme-aware mapping; text that sits ON
saturated fills uses ONACC (white in light, near-black in dark because dark
accents are bright). Render dark variant:
    FIG_THEME=dark .venv/bin/python eval/<script>.py

Import before creating any figure:
    from paper_style import apply, C, C4, C8, CF, ANNOT, bar_kw, out_dir, ...
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# ---- theme ---------------------------------------------------------------
THEME = os.environ.get("FIG_THEME", "light")
DARK = THEME == "dark"

# ---- type scale (points) --------------------------------------------------
BASE = 9.0       # body / tick text
TITLE = 9.5      # axes + figure title
LABEL = 9.0      # axis labels
TICK = 8.0       # tick labels
LEGEND = 8.0     # legend entries
ANNOT = 7.5      # in-plot data labels (bar values, baseline callouts)

# ---- palette (kept from the site's brand colors) -------------------------
if DARK:
    # brightened for the dark canvas; hues match the site's dark theme accents
    C4, C8, CF = "#60a5fa", "#fbbf24", "#34d399"   # INT4 blue, INT8 amber, FP16 green
    RULE = "#f87171"                                # memory-limit rule
    INK = "#e0e0e0"                                 # text / edges
    FACE = "#1a1a2e"                                # = site dark --bg2 (figure card)
    EDGE = "#8a8a9e"                                # axes spines
    TICKC = "#9a9aae"                               # tick labels
    GRID = "#3a3a50"                                # dotted y-grid
    BAREDGE = "#d0d0dc"                             # bar outlines
    ONACC = "#14141f"                               # text ON saturated fills
else:
    C4, C8, CF = "#2563eb", "#f59e0b", "#16a34a"   # INT4 blue, INT8 amber, FP16 green
    RULE = "#e74c3c"                                # memory-limit rule
    INK = "#1a1a2e"                                 # text / edges
    FACE = "white"
    EDGE = "#444444"
    TICKC = "#444444"
    GRID = "#b0b0b0"
    BAREDGE = "#333333"
    ONACC = "#ffffff"

# ---- light→dark map for hardcoded hex colors inside scripts --------------
# Keys/values lowercase; 3-digit hexes are expanded before lookup. Anything
# unmapped passes through unchanged (C() below).
_DARK_MAP = {
    # brand accents → brightened
    "#2563eb": "#60a5fa", "#1d4ed8": "#3b82f6", "#3b82f6": "#60a5fa",
    "#f59e0b": "#fbbf24", "#d97706": "#fbbf24", "#b45309": "#f59e0b",
    "#c2410c": "#fb923c", "#a16207": "#eab308",
    "#16a34a": "#34d399", "#22c55e": "#4ade80",
    "#e74c3c": "#f87171",
    "#7c3aed": "#a78bfa", "#8b5cf6": "#a78bfa", "#5c6bc0": "#8c9eff",
    # pale tinted fills → dark tinted fills (same hue family)
    "#dbeafe": "#1e3a5f", "#f2f6ff": "#1c2340", "#fbfcfe": "#1e1e32",
    "#fef3c7": "#3a2a06", "#fff7ed": "#33240c",
    "#dcfce7": "#06301f", "#f1fbf4": "#0a2419",
    "#f3e8ff": "#2e1065", "#ede9fe": "#2a1a5e", "#e8eaf6": "#232347",
    "#f8fafc": "#232338", "#f1f5f9": "#222236", "#eef2f7": "#232338",
    "#e9ecef": "#242438",
    # white BOX FILLS → dark neutral (text-on-fill must use ONACC, not C())
    "#ffffff": "#26263c",
    # ink / gray text → lightened
    "#1a1a2e": "#e0e0e0",
    "#333333": "#d0d0dc",
    "#444444": "#b8b8c8",
    "#555555": "#a8a8b8",
    "#666666": "#9a9aaa",
    "#888888": "#9090a0", "#999999": "#9090a0",
    "#b0b0b0": "#4a4a60",
    # slate family (annotations / borders)
    "#334155": "#cbd5e1", "#475569": "#adb5bd", "#64748b": "#94a3b8",
    "#94a3b8": "#8494ab",
    "#cbd5e1": "#4a5568", "#c7d2df": "#3e4c63", "#adb5bd": "#5c6675",
    "#6c757d": "#868e96", "#868e96": "#7a828c",
    "#777777": "#9090a0",
    "#000000": "#e8e8f0",
}

_NAMED = {"white": "#ffffff", "black": "#000000"}


def C(color):
    """Theme-aware color: identity in light mode; dark mode maps known hexes
    (and the named white/black) via _DARK_MAP (3-digit expanded first),
    unknown colors pass through."""
    if not DARK or not isinstance(color, str):
        return color
    h = _NAMED.get(color.strip().lower(), color.strip().lower())
    if not h.startswith("#"):
        return color
    if len(h) == 4:
        h = "#" + "".join(ch * 2 for ch in h[1:])
    return _DARK_MAP.get(h, color)


def out_dir(here):
    """Figure output dir: docs/figures (light) or docs/figures/dark (dark).
    `here` = the script's directory at any depth under the repo (walks up to
    the dir containing docs/figures). Creates the dark dir on demand."""
    d = os.path.abspath(here)
    while not os.path.isdir(os.path.join(d, "docs", "figures")):
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("docs/figures not found above %r" % here)
        d = parent
    out = os.path.join(d, "docs", "figures")
    if DARK:
        out = os.path.join(out, "dark")
        os.makedirs(out, exist_ok=True)
    return out


# ---- canvas widths -------------------------------------------------------
# The site renders half-slot figures at ~440 px and full-width ones at ~870 px.
# Keeping width proportional to the display slot makes the *on-screen* type
# size identical across every figure (~84 px/inch in both cases).
WIDTH_HALF = 5.2
WIDTH_FULL = 10.4


def _serif_stack():
    """Latin serif first, CJK serif after it as the fallback face.

    Must be assigned to ``font.family`` as an explicit list: matplotlib only
    walks a *family list* when a glyph is missing. Pointing ``font.family`` at
    "serif" and listing the faces in ``font.serif`` resolves to the first
    available face only, which renders CJK as tofu boxes.
    """
    have = {f.name for f in _fm.fontManager.ttflist}
    latin = [n for n in ("Times New Roman", "STIX Two Text", "DejaVu Serif") if n in have]
    cjk = [n for n in ("Songti SC", "STSong", "Arial Unicode MS") if n in have]
    return (latin + cjk) or ["DejaVu Serif"]


def apply():
    """Install the shared rcParams. Call once, before building figures."""
    plt.rcParams.update({
        # type — explicit family list so CJK falls back to a serif CJK face
        "font.family": _serif_stack(),
        "font.serif": _serif_stack(),
        "mathtext.fontset": "stix",
        "font.size": BASE,
        "axes.titlesize": TITLE,
        "axes.titleweight": "normal",   # papers don't bold figure titles
        "axes.labelsize": LABEL,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "legend.fontsize": LEGEND,
        "figure.titlesize": TITLE,
        "axes.unicode_minus": False,
        # frame
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": EDGE,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": TICKC,
        "ytick.color": TICKC,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        # grid: light dotted, y only, behind the data
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        # legend
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.3,
        # output
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.facecolor": FACE,
        "savefig.facecolor": FACE,
        # axes patch follows the canvas (default white would leave a glaring
        # white plot area in dark mode; "white" keeps light output identical)
        "axes.facecolor": FACE,
    })


def bar_kw():
    """Thin bar outline — keeps bars legible in grayscale print."""
    return dict(edgecolor=BAREDGE, linewidth=0.5)


def fig_legend(fig, ax, ncol=3):
    """One shared frameless legend centred below the figure.

    Panels in these figures always plot the same series, so a single figure
    legend (rather than one per axes) avoids colliding with the y-axis ticks
    and keeps every figure's legend in the same place.
    """
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, 0.0), frameon=False)


def label_bars(ax, bars, fmt="{:.1f}", pad=2.0):
    """Uniform in-plot data labels above bars."""
    for r in bars:
        h = r.get_height()
        ax.annotate(fmt.format(h), (r.get_x() + r.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=ANNOT,
                    xytext=(0, pad), textcoords="offset points")
