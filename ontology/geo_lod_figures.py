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
import os

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


from geo_lod_release import GEO_LOD_RELEASE

# What matplotlib would otherwise write here is its own version number, and
# that is a property of the machine the run happened on, not of the figure.
# A2 already rules out a clock in the output; a version stamp is the same
# thing one step removed - it turned 40 unchanged figures into a diff when
# matplotlib went from 3.9.2 to 3.9.4. The release date changes when geo-lod
# decides it does.
SVG_CREATOR = f"geo-lod, release {GEO_LOD_RELEASE}, https://w3id.org/geo-lod/"


# --------------------------------------------------------------------------
# Raster quality
# --------------------------------------------------------------------------
# Two settings, and they are deliberately not constants: an everyday run wants
# small files it can look at, a release wants rasters a journal will print.
# Both come from the environment because main.py starts every drawing script
# as its own process - a flag parsed in main.py has to reach them somehow, and
# an environment variable is the one channel that needs no argparse in six
# scripts. A script run by hand therefore draws draft quality unless told
# otherwise, which is the right default for the case where somebody is
# iterating on one figure.

#: Draft: what a development run writes. No pixel floor - the figure is what
#: it was laid out as.
DRAFT_DPI = 100

#: Print: 300 dpi, and at least this many pixels on the shorter side of the
#: cropped image. Dots per inch alone says nothing about the size of an image,
#: and a four-inch panel at 300 dpi is 1200 px and unusable on a poster.
PRINT_DPI = 300
PRINT_MIN_PIXELS = 3000

ENV_DPI = "GEO_LOD_RASTER_DPI"
ENV_MIN_PIXELS = "GEO_LOD_RASTER_MIN_PX"


def raster_quality() -> tuple[int, int]:
    """(dpi, minimum pixels on the shorter side), from the environment.

    An unreadable value is worth a word rather than a crash: a mistyped
    variable should not stop a run that is otherwise fine, but it must not
    quietly produce draft rasters in a release either.
    """
    dpi, min_pixels = DRAFT_DPI, 0
    raw = os.environ.get(ENV_DPI, "").strip().lower()
    if raw in ("print", "release"):
        dpi, min_pixels = PRINT_DPI, PRINT_MIN_PIXELS
    elif raw in ("draft", "dev", ""):
        pass
    else:
        try:
            dpi = int(float(raw))
            # From 300 upwards the intent is a printable raster, so the pixel
            # floor comes with it; below that the caller wants a small file
            # and a floor would defeat the request.
            min_pixels = PRINT_MIN_PIXELS if dpi >= PRINT_DPI else 0
        except ValueError:
            print(f"  ⚠  {ENV_DPI}={raw!r} is not a number, 'draft' or "
                  f"'print' - drawing at {DRAFT_DPI} dpi")
    raw_pixels = os.environ.get(ENV_MIN_PIXELS, "").strip()
    if raw_pixels:
        try:
            min_pixels = int(float(raw_pixels))
        except ValueError:
            print(f"  ⚠  {ENV_MIN_PIXELS}={raw_pixels!r} is not a number - "
                  f"keeping {min_pixels}")
    return dpi, min_pixels


def raster_size_inches(fig) -> tuple[float, float]:
    """What the saved image will measure, not what the figure was laid out as.

    Everything here is written with ``bbox_inches="tight"``, which crops the
    margins away - a 10 by 9 inch figure came out as 8.9 by 9.8 after
    cropping, and a pixel floor computed from the uncropped size misses it.
    Falls back to the figure size where no renderer is available yet.
    """
    try:
        fig.canvas.draw()
        box = fig.get_tightbbox(fig.canvas.get_renderer())
        if box.width > 0 and box.height > 0:
            return float(box.width), float(box.height)
    except Exception:
        pass
    width, height = fig.get_size_inches()
    return float(width), float(height)


def raster_dpi(fig) -> int:
    """The dpi at which *fig* satisfies whatever quality was asked for.

    Where a pixel floor applies, whichever of the two binds is the one that
    counts: a wide, short figure is driven by the floor on its short side, a
    large one by the dots per inch.
    """
    dpi, min_pixels = raster_quality()
    shorter_inches = min(raster_size_inches(fig))
    if shorter_inches <= 0 or min_pixels <= 0:
        return dpi
    return int(max(dpi, math.ceil(min_pixels / shorter_inches)))


def save_figure(fig, base_path: str, dpi: int | None = None,
                verbose: bool = True) -> None:
    """Write *fig* as .svg and .jpg next to each other.

    The SVG goes through a binary handle so that the file holds the bytes
    matplotlib produced, with LF endings, on every platform. ``Date: None``
    drops the timestamp matplotlib would otherwise put in the metadata,
    ``Creator`` replaces the matplotlib version string with a fixed one, and
    the deterministic element ids come from ``plt.rcParams["svg.hashsalt"]``,
    which the calling script sets.

    *dpi* is worked out from the figure and the requested quality unless a
    caller insists on a value. Passing one is almost always a mistake: it was
    the per-script ``DPI = 100`` that made every raster in this repository too
    small to print, and it made the setting unreachable from outside.
    """
    with open(base_path + ".svg", "wb") as fh:
        fig.savefig(fh, format="svg", bbox_inches="tight",
                    metadata={"Date": None, "Creator": SVG_CREATOR})
    effective = dpi if dpi is not None else raster_dpi(fig)
    fig.savefig(base_path + ".jpg", format="jpg", dpi=effective,
                bbox_inches="tight")
    if verbose:
        width, height = raster_size_inches(fig)
        print(f"  ✓ Saved: {base_path}.svg / .jpg "
              f"({round(width * effective)}×{round(height * effective)} px, "
              f"{effective} dpi)")


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


#: Absolute floor and relative factor of ``find_breaks_relative``. See there
#: for why a speleothem needs both where an ice core needs neither.
RELATIVE_BREAK_FLOOR_KYR = 2.0
RELATIVE_BREAK_FACTOR = 10.0
RELATIVE_BREAK_WINDOW = 25


def find_breaks_relative(
    positions: np.ndarray,
    floor: float = RELATIVE_BREAK_FLOOR_KYR,
    factor: float = RELATIVE_BREAK_FACTOR,
    window: int = RELATIVE_BREAK_WINDOW,
) -> list[tuple[int, int]]:
    """Index pairs bounding each break, judged against the record's own pace.

    A spacing counts as a break when it is both wider than *floor* and more
    than *factor* times the median spacing of its *window* neighbours on
    either side. *positions* must be sorted ascending.

    Why not the fixed threshold of ``find_breaks``. Speleothem sampling
    density varies by three orders of magnitude between the records in this
    repository - SPA133 samples every 1.4 years, BG67 every 1060 - and a
    single number cannot mean the same thing in both. Two records show what
    goes wrong. SB61 is sampled at a steady 1.6 to 1.85 kyr between 260 and
    318 ka; at a 1 kyr threshold that regular stretch dissolves into thirty
    dashed segments. SPA121_2021 is sampled at a steady 5.09, 5.21, 5.11,
    5.13, 4.98 kyr between 164 and 191 ka; at a 5 kyr threshold four of those
    five equal steps are dashed and the fifth is not, which is visibly
    arbitrary and impossible to defend in a caption.

    Each term does its own work. The floor keeps a finely sampled record from
    reporting breaks that are merely a few centuries wide: Corchia's CC-1_2018
    has a 0.45 kyr spacing that is 32 times its median, and 450 years is not a
    gap in a 12 kyr record. The factor keeps a uniformly coarse record from
    reporting its own resolution as a break: BG67 samples every 1.06 kyr and
    twice reaches 2.0, which clears the floor but is the record behaving
    normally. Over the six sites the two together mark fifteen breaks, and
    every one of them is a stretch where the neighbouring spacing is 33 to 196
    times smaller.

    The caveat is the same as for ``find_breaks`` and travels with it: a
    dashed segment states "no samples here", not "no growth here". SISAL does
    record growth interruptions - the ``hiatus`` and ``gap`` tables, in the
    graph since S3c.3 as ``geolod:GrowthHiatus`` and ``geolod:RecordGap`` -
    but those rows carry no chronology and therefore never reach a figure.
    """
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 3:
        return []

    spacing = np.diff(positions)
    breaks = []
    for i, width in enumerate(spacing):
        if width <= floor:
            continue
        lo = max(0, i - window)
        hi = min(len(spacing), i + window + 1)
        local = float(np.median(spacing[lo:hi]))
        if local > 0 and width > factor * local:
            breaks.append((i, i + 1))
    return breaks


# ---------------------------------------------------------------------------
# Telling several records apart in one figure
# ---------------------------------------------------------------------------
# A SISAL site holds up to eighteen speleothems, and they overlap in age.
# Drawing them as one series sorted by age produces a curve that jumps between
# specimens; drawing them as one colour produces a thicket. Colours come from
# matplotlib's qualitative maps in a fixed order, so the same speleothem keeps
# the same colour across the three smoothing variants of a site.

def series_colours(n: int) -> list[str]:
    """*n* distinguishable colours, in a fixed order.

    tab10 up to ten series. Beyond that tab20, but with its ten saturated
    shades taken first and the ten pale ones only afterwards: tab20 alternates
    dark and light shades of one hue, and a pale line is hard to follow across
    the pastel MIS bands these figures carry behind the curves. Taken in the
    map's own order, Sanbao's second-longest record came out in the palest
    blue available.
    """
    from matplotlib.colors import to_hex
    import matplotlib.pyplot as plt

    if n <= 10:
        cmap = plt.get_cmap("tab10")
        return [to_hex(cmap(i)) for i in range(n)]

    cmap = plt.get_cmap("tab20")
    order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    return [to_hex(cmap(order[i % 20])) for i in range(n)]


#: Candidate legend anchors, in the order a tie is broken. The three right
#: hand positions are missing on purpose: in the portrait figures of this
#: repository the right edge carries the MIS labels, and a legend there
#: collides with text rather than with a curve, which is worse - the curve
#: shows through, the label does not.
LEGEND_ANCHORS = (
    ("upper left", 0.0, 1.0),
    ("lower left", 0.0, 0.0),
    ("upper center", 0.5, 1.0),
    ("lower center", 0.5, 0.0),
    ("center left", 0.0, 0.5),
    ("center", 0.5, 0.5),
)

#: Returned instead of an anchor when every anchor would bury a series. The
#: caller is expected to put the legend outside the axes.
LEGEND_OUTSIDE = "outside"

#: A series counts as buried above this share of its points. Ten per cent is
#: well above what a legend covers when it sits in genuinely empty space -
#: every uncrowded site in this repository lands on zero - and well below the
#: 27 per cent that is the best any interior anchor manages at Sanbao.
LEGEND_MAX_COVERED = 0.10


def best_legend_loc(
    points: list[tuple[np.ndarray, np.ndarray]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    width: float = 0.42,
    height: float = 0.10,
    max_covered: float = LEGEND_MAX_COVERED,
) -> str:
    """The anchor that buries the least of any one series.

    *points* is a list of (values, positions) pairs in data coordinates, one
    per series; *xlim* and *ylim* are the axis limits in the order they were
    set, so an inverted age axis works unchanged. *width* and *height* are the
    footprint of the legend as a fraction of the axes - measure it, do not
    guess: Sanbao's three columns of speleothem names are 0.59 wide, and an
    assumed 0.42 declared a corner empty that a record ran straight through.

    Scored per series and not over all points. Sanbao has eighteen
    speleothems, and the corner with the fewest points under it hides 140 of
    SB-58's 155 - ninety per cent of the only record covering MIS 11 and 12 -
    while amounting to two per cent of the site. A legend that hides a little
    of each of eighteen long records costs nothing; one that hides a whole
    short record costs the record.

    Returns ``LEGEND_OUTSIDE`` when even the best anchor buries more than
    *max_covered* of some series. At a site as densely covered as Sanbao there
    is no free space inside the axes, and pretending otherwise only picks the
    least bad place to lose a speleothem.

    Why not matplotlib's ``loc="best"``. It does a similar search, but at draw
    time, over whatever artists are on the axes, and by area rather than by
    series. The smoothed variants carry twice as many lines as the unsmoothed
    one, so the three figures of a record could land on different corners, and
    three figures whose legend jumps about read as three unrelated figures.
    """
    x0, x1 = float(xlim[0]), float(xlim[1])
    y0, y1 = float(ylim[0]), float(ylim[1])
    if x1 == x0 or y1 == y0 or not points:
        return LEGEND_ANCHORS[0][0]

    fractions = [
        ((np.asarray(v, dtype=float) - x0) / (x1 - x0),
         (np.asarray(p, dtype=float) - y0) / (y1 - y0))
        for v, p in points
    ]

    best, best_score = LEGEND_ANCHORS[0][0], None
    for name, ax_x, ax_y in LEGEND_ANCHORS:
        left = min(max(ax_x - width / 2 if ax_x == 0.5 else ax_x, 0.0),
                   max(1.0 - width, 0.0))
        bottom = min(max(ax_y - height / 2 if ax_y == 0.5 else
                         (ax_y - height if ax_y == 1.0 else ax_y), 0.0),
                     max(1.0 - height, 0.0))
        worst, total = 0.0, 0.0
        for fx, fy in fractions:
            covered = np.count_nonzero(
                (fx >= left) & (fx <= left + width)
                & (fy >= bottom) & (fy <= bottom + height)
            )
            worst = max(worst, covered / len(fx))
            total += covered
        score = (worst, total)
        if best_score is None or score < best_score:
            best, best_score = name, score
        if score == (0.0, 0.0):
            break

    if best_score is not None and best_score[0] > max_covered:
        return LEGEND_OUTSIDE
    return best


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
