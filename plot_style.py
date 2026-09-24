"""
plot_style.py
=============

Unified figure style.  Import once at the top of a plotting script:

    from plot_style import apply_style
    apply_style()

Exports
-------
- apply_style()       : global matplotlib rcParams (serif fonts, font
                        sizes, axis and grid conventions).
- Palette dicts       : per-algorithm line colours, ablation bar
                        colours, environment categorical colours,
                        sensitivity-curve colours and markers.
- HEATMAP_CMAP        : diverging colormap for rank and gap heatmaps.
- Helper kwargs       : style_errorbar_kwargs, style_bar_kwargs,
                        style_marker_kwargs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ======================================================================
# 1. Palettes
# ======================================================================

# Per-algorithm line colours used by the summary line plots and
# heatmaps.  TOBYQA is rendered in Prussian blue; the other solvers
# occupy hue-separated, desaturated tones.
LINE_COLORS = {
    "TOBYQA": "#003153",   # Prussian blue (proposed method)
    "NEWUOA": "#A23838",   # deep chestnut red
    "BOBYQA": "#2D5A4C",   # pine green
    "DFO-LS": "#7B9CBB",   # muted blue-grey
    "NMSMAX": "#C5951D",   # antique gold
    "BFGS":   "#5C4033",   # dark umber
}

# Diverging colormap for rank / gap heatmaps. Warm end indicates
# worse performance (high rank, large gap); cool end indicates better.
_HEATMAP_HEX = [
    "#003153",  # deepest blue   (rank 1 / smallest gap)
    "#255E89",
    "#6B9ABF",
    "#E8E3D7",  # neutral cream
    "#E3C270",
    "#D4AA45",
    "#C5951D",  # deepest gold   (high rank / large gap)
]
HEATMAP_CMAP = LinearSegmentedColormap.from_list("paper_diverging", _HEATMAP_HEX, N=256)

# Ablation bar plot (four variants).
ABLATION_COLORS = {
    "Full":         "#003153",  # Prussian blue
    "-DriftModel":  "#7B9CBB",  # muted blue-grey
    "-AffineScale": "#2D5A4C",  # pine green
    "-Drift&AS":    "#C5951D",  # antique gold
}
# Index-addressable list form of the same palette.
ABLATION_COLOR_LIST = [
    "#003153", "#7B9CBB", "#2D5A4C", "#C5951D",
]

# Categorical colours for the four environment types used by the
# ablation grouped plot.
ENV_COLORS = {
    "Static":  "#003153",  # Prussian blue
    "Dynamic": "#C5951D",  # antique gold
    "Tidal":   "#7B9CBB",  # muted blue-grey
    "Coupled": "#2D5A4C",  # pine green
}

# Sensitivity line plots: Static vs Dynamic two-colour contrast.
SENSITIVITY_COLORS = {
    "Static":  "#003153",
    "Dynamic": "#C5951D",
}
SENSITIVITY_MARKERS = {
    "Static":  "o",
    "Dynamic": "s",
}


# ======================================================================
# 2. Global style
# ======================================================================

def apply_style(use_tex: bool = False):
    """
    Configure global matplotlib rcParams.

    Parameters
    ----------
    use_tex : bool, default False
        When True, enables LaTeX rendering (requires a local LaTeX
        installation). When False, mathtext combined with a
        Times-like serif family is used; this keeps mathematical
        expressions readable without depending on an external LaTeX
        toolchain.
    """
    rc = {
        # Fonts
        "font.family":            "serif",
        "font.serif":             ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":              11,
        "mathtext.fontset":       "stix",   # math font close to Times

        # Font-size hierarchy
        "axes.titlesize":         12,
        "axes.labelsize":         11,
        "xtick.labelsize":        10,
        "ytick.labelsize":        10,
        "legend.fontsize":        10,
        "legend.title_fontsize":  11,
        "figure.titlesize":       14,
        "figure.titleweight":     "bold",

        # Axes
        "axes.linewidth":         0.8,
        "axes.edgecolor":         "#333333",
        "axes.spines.top":        False,
        "axes.spines.right":      False,
        "axes.spines.left":       True,
        "axes.spines.bottom":     True,

        # Grid: dashed light-grey horizontal lines on the y-axis only.
        "axes.grid":              True,
        "axes.grid.axis":         "y",
        "grid.color":             "#E0E0E0",
        "grid.linestyle":         "--",
        "grid.linewidth":         0.5,
        "axes.axisbelow":         True,        # grid is drawn below data

        # Legend
        "legend.frameon":         False,

        # Output
        "figure.dpi":             100,
        "savefig.dpi":            150,
        "savefig.bbox":           "tight",
    }
    if use_tex:
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = r"\usepackage{times}\usepackage{amsmath}"
    plt.rcParams.update(rc)


# ======================================================================
# 3. Helper kwargs
# ======================================================================

def style_errorbar_kwargs(color: str = "#333333") -> dict:
    """
    Error-bar kwargs shared by bar and line plots: thin lines,
    small caps, restrained colour.
    """
    return dict(
        fmt="none",
        ecolor=color,
        elinewidth=1.2,
        capsize=3,
        capthick=1.0,
        alpha=0.85,
    )


def style_bar_kwargs() -> dict:
    """Bar-body kwargs: light transparency, thin black edge."""
    return dict(
        alpha=0.85,
        edgecolor="black",
        linewidth=1.0,
    )


def style_marker_kwargs(env: str) -> dict:
    """
    Data-point kwargs for sensitivity line plots.
    Static is rendered as a circle, Dynamic as a square.
    """
    return dict(
        marker=SENSITIVITY_MARKERS.get(env, "o"),
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=0.5,
        markerfacecolor=SENSITIVITY_COLORS.get(env, "#003153"),
    )
