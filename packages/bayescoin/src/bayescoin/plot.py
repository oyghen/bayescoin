__all__ = ["plot"]

from functools import singledispatch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import bayescoin


@singledispatch
def plot(value, *args, **kwargs) -> Axes:
    """Return Axes plotting a Beta density for supported inputs."""
    raise TypeError(f"unsupported type {type(value).__name__!r}")


@plot.register
def _(
    a: int | float,
    b: int | float,
    hdi_level: float = 0.95,
    num_points: int = 4096,
    ax: Axes | None = None,
) -> Axes:
    """Return Axes plotting the Beta(a, b) density with shaded HDI."""
    besh = bayescoin.BetaShape(a, b)
    return plot(besh, hdi_level, num_points, ax)


@plot.register
def _(
    besh: bayescoin.BetaShape,
    hdi_level: float = 0.95,
    num_points: int = 4096,
    ax: Axes | None = None,
) -> Axes:
    """Return Axes plotting the Beta density for a BetaShape object."""
    if ax is None:
        ax = _gca_with_size(figsize=(9, 6))

    # compute Beta density
    x = np.linspace(0.0, 1.0, num_points)
    dist = besh.to_dist()
    y = dist.pdf(x)

    # plot Beta density
    ax.plot(x, y, label=_pretty_beta_string(besh), linewidth=2, zorder=0)

    hdi = besh.hdi(hdi_level)
    if hdi is None:
        # shade entire area under the Beta distribution
        ax.fill_between(x, y, alpha=0.2, zorder=0)
    else:
        # shade HDI area
        mask = (x >= hdi[0]) & (x <= hdi[1])
        hdi_label = f"{100 * hdi_level:g}%-HDI"
        ax.fill_between(x, y, where=mask, label=hdi_label, alpha=0.2, zorder=0)

        # add HDI endpoints
        dark_gray = "#333333"
        ax.plot(hdi, [dist.pdf(v) for v in hdi], color=dark_gray)
        for i, endpoint in enumerate(hdi):
            xcoords = 2 * [endpoint]
            ycoords = [0, dist.pdf(endpoint)]

            # add endpoint line segments
            ax.plot(xcoords, ycoords, color=dark_gray, linestyle="--")

            # add endpoint dots
            ax.scatter(xcoords[1], ycoords[1], color=dark_gray)

            # add endpoint text
            xpos = round(endpoint, 2) - 0.01 if i == 0 else round(endpoint, 2) + 0.01
            ypos = 1.01 * ycoords[1]
            text = f"{endpoint:.4f}"
            ax.text(xpos, ypos, text, ha="right" if i == 0 else "left")

    # add mode to legend if exists
    mode = besh.mode
    if mode is not None:
        ax.scatter(
            besh.mode,
            0,
            marker="v",
            color="darkorange",
            label=f"mode={besh.mode:g}",
            alpha=0.6,
            zorder=1,
        )

    # add mean to legend
    ax.scatter(
        besh.mean,
        0,
        marker="v",
        color="black",
        label=f"mean={besh.mean:g}",
        alpha=0.6,
        zorder=1,
    )

    # hide all spines except the bottom
    for spine_name, spine in ax.spines.items():
        if spine_name != "bottom":
            spine.set_visible(False)

    # set x-axis limits and ticks
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.arange(0, 1.1, 0.1))

    # set color of x-axis
    gray = "#666666"
    ax.spines["bottom"].set_color(gray)
    ax.tick_params(axis="x", colors=gray)

    # disable y ticks and labels
    ax.set_yticks([])

    # make uniform Beta plot prettier by increasing the upper y limit
    if besh.a == 1.0 and besh.b == 1.0:
        _, y_max = ax.set_ylim()
        ax.set_ylim(top=2 * y_max)

    # set legend
    ax.legend(loc="best", frameon=False)

    return ax


def _gca_with_size(figsize: tuple[int, int]) -> Axes:
    """Return current axes with given figsize if none exist."""
    if not plt.get_fignums():
        plt.figure(figsize=figsize)
    return plt.gca()


def _pretty_beta_string(beta_shape: bayescoin.BetaShape) -> str:
    """Return polished Beta(a, b) string."""
    return (
        repr(beta_shape)
        .replace("BetaShape", "Beta")
        .replace("a=", "")
        .replace("b=", "")
    )
