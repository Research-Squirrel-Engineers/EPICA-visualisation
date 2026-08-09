"""
epica_style.py
==============
Two things every EPICA figure needs and neither the plot script nor the plate
module should own alone: how a figure is written to disk, and how a value axis
is divided.

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


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    import pandas as pd

    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
