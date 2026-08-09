"""
geo_lod_figures.py
==================
Two things every figure in this repository needs, and no single plot script
should own: how a figure is written to disk, and how a value axis is divided.

Lives next to ``geo_lod_utils.py`` and for the same reason. Every sub-script
already runs with ``ontology/`` on its ``PYTHONPATH``, so EPICA, SISAL, CI and
whatever comes with ELSA can import this without any further plumbing. It was
written for EPICA in S2 and moved here immediately afterwards, because the
CRLF bug it fixes is not an EPICA bug: ``SISAL/plot_sisal_from_csv.py`` writes
its 36 SVG the same way EPICA did, and would keep reporting them as modified
on every Windows checkout.

Deliberately free of rdflib: this module is about drawing, ``geo_lod_utils``
is about triples, and neither should drag the other\'s dependencies into a
script that only needs one of them.

Writing
-------
``save_figure`` writes the SVG through a file handle opened in binary mode.
Matplotlib otherwise opens the target in text mode, and on Windows Python then
translates every newline to CRLF. With ``.gitattributes`` storing LF, the
working copy differs from its own committed form on every single figure - the
same trap the log files fell into, one directory further along. Reading the
bytes back and comparing is the only way to notice, so the fix belongs in one
place rather than at every call site.

Dividing an axis
----------------
``nice_ticks`` derives ticks from the data instead of from a hand-written list.
The hand-written lists were not merely inconvenient: the d18O list ended at
1.0 while the record reaches 1.457, so six per cent of the measurements were
drawn outside the axis and clipped away - in every d18O figure, including the
published one. Ticks computed from the range cannot do that, because they are
built to bracket it.

Steps are chosen from 1, 2, 2.5 and 5 times a power of ten, so a label reads
as 0, 0.5, 1 or 200, 400, 600 rather than 0.37, 0.74. Decimals follow the
step: an integer step prints integers.
"""

from __future__ import annotations

import math

import numpy as np

#: Preferred tick steps, per decade. 2.5 is included because it is what makes
#: a range like 0-1.5 come out as 0, 0.25, 0.5 ... rather than 0, 0.4, 0.8.
NICE_STEPS = (1.0, 2.0, 2.5, 5.0, 10.0)


def nice_step(span: float, target: int) -> float:
    """Round tick step for *span* aiming at roughly *target* intervals."""
    if span <= 0:
        return 1.0
    raw = span / max(target, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    for step in NICE_STEPS:
        if step * magnitude >= raw:
            return step * magnitude
    return 10.0 * magnitude


def nice_ticks(
    vmin: float, vmax: float, target: int = 6, pad_fraction: float = 0.04
) -> tuple[list[float], tuple[float, float], int]:
    """Ticks, axis limits and decimal places for a value range.

    The returned limits always contain ``[vmin, vmax]``: they run from the
    tick below the minimum to the tick above the maximum, plus a small margin
    so the curve does not touch the frame. Nothing can be clipped.
    """
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5

    step = nice_step(vmax - vmin, target)
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    # floor/ceil can land exactly on the data point; one more step keeps the
    # extreme value off the frame.
    if math.isclose(lo, vmin, rel_tol=1e-12, abs_tol=step * 1e-9):
        lo -= step
    if math.isclose(hi, vmax, rel_tol=1e-12, abs_tol=step * 1e-9):
        hi += step

    n = int(round((hi - lo) / step))
    ticks = [lo + i * step for i in range(n + 1)]

    margin = (hi - lo) * pad_fraction
    decimals = max(0, -math.floor(math.log10(step) + 1e-9))
    # 2.5-type steps need one decimal more than their magnitude suggests.
    if not math.isclose(step * 10**decimals, round(step * 10**decimals)):
        decimals += 1

    return ticks, (lo - margin, hi + margin), int(decimals)


def log_ticks(
    vmin: float, vmax: float, margin: float = 1.35
) -> tuple[list[float], tuple[float, float]]:
    """Decade ticks for a logarithmic axis.

    The ticks sit on full decades, the limits do not: rounding the limits out
    to decades as well would leave most of a dust axis empty, because 2.7 to
    1525 would become 1 to 10000. The limits follow the data with a factor of
    margin on each side, and the decades that fall inside get a label.
    """
    lo, hi = max(vmin, 1e-12) / margin, max(vmax, 1e-12) * margin
    ticks, t = [], 10.0 ** math.floor(math.log10(lo))
    while t <= hi:
        if t >= lo:
            ticks.append(t)
        t *= 10.0
    return ticks, (lo, hi)


def format_tick(value: float, decimals: int) -> str:
    """Tick label with *decimals* places, without a stray minus zero."""
    if abs(value) < 10 ** (-decimals) / 2:
        value = 0.0
    return f"{value:.{decimals}f}"


def save_figure(fig, base_path: str, dpi: int = 100, verbose: bool = True) -> None:
    """Write *fig* as .svg and .jpg next to each other.

    The SVG goes through a binary handle so that the file holds the bytes
    matplotlib produced, with LF endings, on every platform. ``Date: None``
    drops the timestamp matplotlib would otherwise put in the metadata; the
    deterministic element ids come from ``plt.rcParams["svg.hashsalt"]``,
    which the calling script sets.
    """
    with open(base_path + ".svg", "wb") as fh:
        fig.savefig(fh, format="svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(base_path + ".jpg", format="jpg", dpi=dpi, bbox_inches="tight")
    if verbose:
        print(f"  ✓ Saved: {base_path}.svg / .jpg")


def runs(n: int, breaks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Index ranges of the uninterrupted runs of a series of length *n*."""
    starts = [0] + [after for _, after in breaks]
    ends = [before + 1 for before, _ in breaks] + [n]
    return list(zip(starts, ends))


def smooth_by_run(
    values: np.ndarray,
    breaks: list[tuple[int, int]],
    kind: str,
    window: int,
    polyorder: int = 2,
) -> np.ndarray:
    """Smooth each uninterrupted run on its own.

    A centred filter run over the whole series reaches across a break and mixes
    values from either side of it: with a window of 11 points and a break of
    178 ka, the smoothed value at 214 ka is partly made of measurements from
    392 ka. That is the same mistake as interpolating a boundary depth across
    the gap, one step further along, and it is not visible in the result - the
    curve stays smooth and plausible.

    Runs shorter than the window are smoothed with the largest window that fits
    (odd, at least ``polyorder + 2`` for Savitzky-Golay); a run too short even
    for that is returned unchanged rather than dropped.
    """
    out = np.array(values, dtype=float)
    for start, end in runs(len(values), breaks):
        chunk = out[start:end]
        length = len(chunk)
        if length < 3:
            continue
        if kind == "median":
            import pandas as pd

            out[start:end] = (
                pd.Series(chunk)
                .rolling(window=min(window, length), center=True, min_periods=1)
                .median()
                .to_numpy()
            )
        else:
            from scipy.signal import savgol_filter

            size = min(window, length if length % 2 else length - 1)
            if size < polyorder + 2:
                continue
            out[start:end] = savgol_filter(
                chunk, window_length=size, polyorder=polyorder
            )
    return out


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    import pandas as pd

    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )


# ---------------------------------------------------------------------------
# Breaks in a record
# ---------------------------------------------------------------------------
# Taken over from wdttest-sisal (py/main.py, draw_profile) so that a break looks
# the same in both repository families: the run is drawn solid, the segment
# across the break dashed, and the last sample before and the first after it are
# ringed. A reader who has seen one figure can then read the other.
#
# The caveat travels with the code, because it is the part that matters: a
# dashed segment states "no samples here", NOT "no growth here" and not "no gas
# trapped here". The threshold is our setting, not a property of the archive;
# changing it changes which breaks are marked, not the data.
#
# The value differs between the families because the records do. wdttest-sisal
# uses 5 kyr for speleothems; at that threshold an EPICA dust record would show
# fourteen breaks that are merely sparse sampling. EPICA uses 15 ka, which is
# also the widest spacing across which geo-lod will interpolate a stage boundary
# into depth - so a stretch that is a break in a figure is exactly a stretch the
# graph refuses to interpolate across.

GAP_DASH_PATTERN = (0, (4, 3))
GAP_MARKER_SIZE = 9
GAP_MARKER_EDGE_WIDTH = 1.6


def find_breaks(positions: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Index pairs (before, after) bounding each break.

    *positions* must be sorted ascending. A break is a spacing wider than
    *threshold*.
    """
    return [
        (i, i + 1)
        for i in range(len(positions) - 1)
        if positions[i + 1] - positions[i] > threshold
    ]


def draw_profile(
    ax,
    values: np.ndarray,
    positions: np.ndarray,
    breaks: list[tuple[int, int]],
    colour: str = "black",
    linewidth: float = 1.0,
    vertical: bool = True,
    marker_size: float = GAP_MARKER_SIZE,
    zorder: int = 2,
) -> None:
    """Draw one profile solid, dashed across breaks, with the break ends ringed.

    With no breaks this is a single solid line. *vertical* says which axis the
    positions belong to: True puts them on y (the portrait figures of this
    repository), False on x (the stacked plates).
    """

    def line(v, p, **kwargs):
        if vertical:
            ax.plot(v, p, **kwargs)
        else:
            ax.plot(p, v, **kwargs)

    starts = [0] + [after for _, after in breaks]
    ends = [before for before, _ in breaks] + [len(positions) - 1]
    for start, end in zip(starts, ends):
        line(
            values[start:end + 1],
            positions[start:end + 1],
            linewidth=linewidth,
            color=colour,
            zorder=zorder,
        )

    # Dashed across each break: interpolated by the eye, not measured.
    for before, after in breaks:
        line(
            values[[before, after]],
            positions[[before, after]],
            linewidth=linewidth,
            color=colour,
            linestyle=GAP_DASH_PATTERN,
            zorder=zorder,
        )

    if breaks:
        edges = sorted({i for pair in breaks for i in pair})
        line(
            values[edges],
            positions[edges],
            linestyle="none",
            marker="o",
            markersize=marker_size,
            markerfacecolor="white",
            markeredgecolor=colour,
            markeredgewidth=GAP_MARKER_EDGE_WIDTH,
            zorder=zorder + 1,
        )
