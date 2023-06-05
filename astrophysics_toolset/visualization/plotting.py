"""Scale bar from https://gist.github.com/dmeliza/3251476."""
# Adapted from mpl_toolkits.axes_grid1
# LICENSE: Python Software Foundation (http://docs.python.org/license.html)

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.mathtext import MathTextWarning
from matplotlib.offsetbox import AnchoredOffsetbox


class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(
        self,
        transform,
        sizex=0,
        sizey=0,
        labelx=None,
        labely=None,
        loc=4,
        pad=0.1,
        borderpad=0.1,
        sep=2,
        prop=None,
        barcolor="black",
        barwidth=None,
        label_kwa=None,
        **kwargs,
    ):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).

        Parameters
        ----------
        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size
          (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.offsetbox import AuxTransformBox, HPacker, TextArea, VPacker
        from matplotlib.patches import Rectangle

        if label_kwa is None:
            label_kwa = {}

        bars = AuxTransformBox(transform)
        rect_kwa = {"ec": barcolor, "lw": barwidth, "fc": "none"}
        if sizex:
            bars.add_artist(Rectangle((0, 0), sizex, 0, **rect_kwa))
        if sizey:
            bars.add_artist(Rectangle((0, 0), 0, sizey, **rect_kwa))

        vpacker_kwa = {"align": "center", "pad": 0, "sep": sep}
        if sizex and labelx:
            self.xlabel = TextArea(labelx, minimumdescent=False, **label_kwa)
            bars = VPacker(children=[bars, self.xlabel], **vpacker_kwa)
        if sizey and labely:
            self.ylabel = TextArea(labely, **label_kwa)
            bars = HPacker(children=[self.ylabel, bars], **vpacker_kwa)

        AnchoredOffsetbox.__init__(
            self,
            loc,
            pad=pad,
            borderpad=borderpad,
            child=bars,
            prop=prop,
            frameon=False,
            **kwargs,
        )


def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
    """Add scalebars to axes
    Adds a set of scale bars to *ax*, matching the size to the ticks of the
    plot and optionally hiding the x and y axes.

    Parameters
    ----------
    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars

    Returns:
    --------
    sb : scalebar object
    """

    def f(axis):
        locs = axis.get_majorticklocs()
        return len(locs) > 1 and (locs[1] - locs[0])

    if matchx:
        kwargs["sizex"] = f(ax.xaxis)
        kwargs["labelx"] = str(kwargs["sizex"])
    if matchy:
        kwargs["sizey"] = f(ax.yaxis)
        kwargs["labely"] = str(kwargs["sizey"])

    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex:
        ax.xaxis.set_visible(False)
    if hidey:
        ax.yaxis.set_visible(False)
    if hidex and hidey:
        ax.set_frame_on(False)

    return sb


def fix_glyph_errors(ax=None):
    """Fix missing glyph warnings in matplotlib"""
    # From https://stackoverflow.com/a/47850541

    if mpl.rcParams["axes.unicode_minus"] is False:
        warnings.warn(
            "If you have issues with minus sign, set `axes.unicode_minus` "
            "to True in the rcParams.",
            stacklevel=1,
        )
    if mpl.rcParams["text.usetex"]:
        # Everything is handled by LaTeX, so nothing to do.
        return

    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()
    # Force the figure to be drawn
    if tuple(int(_) for _ in mpl.__version__.split(".")) >= (3, 1, 0):
        import logging

        logger = logging.getLogger("matplotlib.mathtext")
        original_level = logger.getEffectiveLevel()
        logger.setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=MathTextWarning)
            fig.canvas.draw()
        logger.setLevel(original_level)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=MathTextWarning)
            fig.canvas.draw()
    # Remove '\mathdefault' from all minor tick labels
    labels = [
        label.get_text().replace(r"\mathdefault", "")
        for label in ax.get_xminorticklabels()
    ]
    ax.set_xticklabels(labels, minor=True)
    labels = [
        label.get_text().replace(r"\mathdefault", "")
        for label in ax.get_yminorticklabels()
    ]
    ax.set_yticklabels(labels, minor=True)


def adjust_lightness(color, amount=0.5):
    """Adjust the lightness of a colour.

    Parameters
    ----------
    color : matplotlib color ("black", "C0", etc.) or RGB(A) value
    amount : float
        Amplitude of the adjustment. If <1, will darken and if >1, will lighten.
    """
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def colorbar(mappable, ax=None):
    """Add a colorbar attached to the mappable, using the `make_axes_locatable` API

    Adapted from https://joseph-long.com/writing/colorbars/
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    try:
        # Save previous current axis
        last_axes = plt.gca()
        ax = mappable.axes if ax is None else ax
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax)
    finally:
        # Restore previous current axis
        plt.sca(last_axes)
    return cbar


def plot_linecolor(x, y, c, *args, **kwargs) -> LineCollection:
    """Plot a line with a position-dependent color.

    Parameters
    ----------
    x, y : array-like
        The data to plot
    c : array-like
        The color of each point
    *args, **kwargs : passed to `plt.plot`
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments)
    lc.set_color(c)
    ax = plt.gca()
    ax.add_collection(lc)
    ax.autoscale_view()

    return lc
