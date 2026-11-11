"""Figure style of the example gallery.

sphinx-gallery calls :func:`reset_style` before every example, so the
examples keep only the code about the models and the figures share one look:
the dark grid of seaborn and Computer Modern text, as in a LaTeX document.
The text goes through matplotlib's own mathtext, which needs no LaTeX
installation, so the gallery builds anywhere.

Every model has one colour in every figure (Okabe-Ito and Tol colours, which
stay apart for colour-blind readers); the examples use these values:

=======================  =========
Model                    Colour
=======================  =========
MixedNB, Naive Bayes     #6E6E6E
AnDE (A1DE)              #0072B2
A2DE                     #56B4E9
WeightedAnDE             #CC79A7
WeightedAnDE, factor     #7E2F8E
AnJE                     #009E73
GaussianAnJE             #44AA99
ALR                      #D55E00
GaussianALR              #882255
TAN                      #E69F00
WeightedTAN              #EE6677
KDB                      #332288
WeightedKDB              #228833
DecisionTreeDiscretizer  #999933
MDLPDiscretizer          #661100
scikit-learn baselines   #1A1A1A
=======================  =========
"""

import matplotlib as mpl

# The rc parameters of ``seaborn.set_theme(style="darkgrid")`` that change
# the look, so that the docs do not need seaborn
_DARKGRID = {
    "axes.facecolor": "#EAEAF2",
    "axes.edgecolor": "white",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.linewidth": 1.25,
    "grid.color": "white",
    "grid.linewidth": 1.0,
    "grid.linestyle": "-",
    "xtick.bottom": False,
    "ytick.left": False,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.labelcolor": "#1A1A1A",
    "text.color": "#1A1A1A",
    "figure.facecolor": "white",
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "white",
    "legend.framealpha": 0.9,
}

_TEXT = {
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    # cmr10 has no minus sign: tick labels go through mathtext instead
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

_LINES = {
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "figure.constrained_layout.use": True,
}


def reset_style(gallery_conf, fname):
    """Set the gallery style; called by sphinx-gallery before each example."""
    mpl.rcdefaults()
    mpl.rcParams.update({**_DARKGRID, **_TEXT, **_LINES})
