"""
epica_plates.py
===============
The multi-panel plates of S2: the comparisons a single-proxy figure cannot
make. The twelve single figures stay as they are; these come alongside.

Three families, all over the full range 0-806 ka BP:

  plate_columns_<variant>   five proxies side by side, one shared vertical age
                            axis - the extension of the existing figure style
  plate_rows_<variant>      five proxies stacked, one shared horizontal age
                            axis - the arrangement the ice-core literature uses
  plate_boundary_depths     the same stage boundaries in the depth axis of each
                            record, which is where the four age models pull
                            apart

``<variant>`` is ``unsmoothed``, ``smooth11`` (rolling median) or
``savgol11p2``, written as three separate sets rather than as one plate with
the raw curve behind the smoothed one.

Two things the plates have to be honest about
---------------------------------------------
The five records do not cover the same range. CH4, deuterium and dust reach
the present, the two gas records begin only at 102 and 106 ka; CH4 ends at
649 ka and additionally has no data between 214 and 392 ka. On a shared axis
that leaves visible gaps, and the plate says so rather than closing them:
stages with no measurement in a given record get a hatched band instead of a
filled one, and the CH4 gap is bridged by a dashed line so the eye does not
read a flat interval as a measured one.

Determinism follows the rest of the pipeline: the hash salt is set by the
calling script, ``metadata={"Date": None}`` drops the SVG timestamp, and no
clock is read anywhere.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy.signal import savgol_filter

import epica_data as ed
import epica_style as st

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")

# --- shared with the single figures -----------------------------------------
ROLLING_WINDOW = 11
SG_WINDOW = 11
SG_POLYORDER = 2

LINE_COLOR = "black"
LINE_WIDTH = 0.9
LINE_COLOR_FADED = "#aaaaaa"
GRID_COLOR = "#cccccc"
DPI = 100

MIS_COLOR_WARM = "#fddbc7"
MIS_COLOR_COLD = "#d6e8f7"
MIS_LABEL_WARM = "#8b1a00"
MIS_LABEL_COLD = "#003f6b"

AGE_MAJOR = 100
AGE_MINOR = 20
AGE_MIN, AGE_MAX = 0.0, 810.0

# Axis labels. "Age [ka]" without BP or b2k, per the decision of 2026-08-08:
# the reference point belongs on the chronology node in the graph, not in
# every caption, and writing BP on a b2k scale is the mistake that decision
# exists to prevent.
AGE_LABEL = "Age [ka]"

# One colour per chronology for the boundary plate. Gas-age scales warm, ice-age
# scales cool - the point of that plate is that the two groups separate.
TRS_COLORS = {
    "EDC2-gas": "#b2182b",
    "AICC2023-gas": "#ef8a62",
    "EDC2-ice": "#2166ac",
    "EDC3-ice": "#4393c3",
    "AICC2023-ice": "#92c5de",
}

VARIANTS = ("unsmoothed", f"smooth{ROLLING_WINDOW}", f"savgol{SG_WINDOW}p{SG_POLYORDER}")

# The gap in the CH4 record, taken from the data rather than written down: any
# interval between consecutive measurements longer than this counts as a gap
# and is drawn dashed instead of solid.
GAP_THRESHOLD_KA = 15.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def smooth_values(values: np.ndarray, variant: str) -> np.ndarray:
    if variant == "unsmoothed":
        return values
    if variant.startswith("smooth"):
        return (
            pd.Series(values)
            .rolling(window=ROLLING_WINDOW, center=True, min_periods=1)
            .median()
            .to_numpy()
        )
    return savgol_filter(
        values, window_length=min(SG_WINDOW, len(values)), polyorder=SG_POLYORDER
    )


def segments(ages: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges of continuous stretches, split at gaps."""
    cuts = [0]
    for i in range(len(ages) - 1):
        if ages[i + 1] - ages[i] > GAP_THRESHOLD_KA:
            cuts.append(i + 1)
    cuts.append(len(ages))
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def draw_bands(ax, stages, covered, horizontal: bool):
    """MIS bands, hatched where the record has no measurement."""
    import matplotlib.transforms as transforms

    if horizontal:
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        span = ax.axvspan
    else:
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        span = ax.axhspan

    # Grenzen aus der Achse, nicht aus den Modulkonstanten: die Collage setzt
    # je Panel eigene Grenzen, und ein Band ausserhalb davon würde sein Label
    # neben die Abbildung schreiben.
    if horizontal:
        lo, hi = sorted(ax.get_xlim())
    else:
        lo, hi = sorted(ax.get_ylim())

    ages = sorted(covered)
    for st in stages:
        visible_lo, visible_hi = max(st["end"], lo), min(st["begin"], hi)
        if visible_lo >= visible_hi:
            continue
        warm = st["mode"] == "warm"
        color = MIS_COLOR_WARM if warm else MIS_COLOR_COLD
        label_color = MIS_LABEL_WARM if warm else MIS_LABEL_COLD
        has_data = any(st["end"] <= a < st["begin"] for a in ages)

        if has_data:
            span(st["end"], st["begin"], facecolor=color, zorder=0)
        else:
            span(
                st["end"],
                st["begin"],
                facecolor=color,
                alpha=0.35,
                edgecolor=label_color,
                linestyle=(0, (4, 3)),
                linewidth=1.0,
                zorder=0,
            )
        middle = (visible_lo + visible_hi) / 2.0
        if horizontal:
            ax.text(
                middle, 0.97, st["label"].replace("MIS ", ""), transform=trans,
                ha="center", va="top", fontsize=8, fontweight="bold",
                color=label_color, zorder=3,
            )
        else:
            ax.text(
                0.985, middle, st["label"].replace("MIS ", ""), transform=trans,
                ha="right", va="center", fontsize=8, fontweight="bold",
                color=label_color, zorder=3,
            )


def save(fig, name: str) -> None:
    # Über epica_style, damit das SVG auch auf Windows mit LF herauskommt.
    st.save_figure(fig, os.path.join(OUTPUT_DIR, name), dpi=DPI)
    plt.close(fig)


def axis_label(dataset_id: str) -> str:
    meta = ed.DATASETS[dataset_id]
    return f"{meta['label']} [{meta['unit_label']}]"


def panel_title(dataset_id: str) -> str:
    """Proxy name with the chronology it sits on.

    The chronology belongs in the panel title, not in the caption. Without it
    a reader comparing two panels has no way to know that the horizontal
    positions were produced by different age models.
    """
    meta = ed.DATASETS[dataset_id]
    return f"{meta['label']}  ({meta['trs']})"


# ---------------------------------------------------------------------------
# Plate A: five columns, shared vertical age axis
# ---------------------------------------------------------------------------


def plate_columns(frames: dict, stages: list[dict], variant: str) -> None:
    fig, axes = plt.subplots(
        1, len(ed.DATASET_ORDER), figsize=(22, 26), sharey=True
    )

    for ax, dataset_id in zip(axes, ed.DATASET_ORDER):
        df = frames[dataset_id]
        meta = ed.DATASETS[dataset_id]
        ages = df["age_ka"].to_numpy()
        values = smooth_values(df["value"].to_numpy(), variant)

        # Grenzen vor den Bändern setzen: draw_bands liest sie von der Achse.
        ax.set_ylim(AGE_MAX, AGE_MIN)
        draw_bands(ax, stages, ages, horizontal=False)

        for lo, hi in segments(ages):
            ax.plot(values[lo:hi], ages[lo:hi], color=LINE_COLOR,
                    linewidth=LINE_WIDTH, zorder=3)
        # Dashed bridge across every gap, so a gap reads as a gap.
        segs = segments(ages)
        for (_, hi), (lo, _) in zip(segs, segs[1:]):
            ax.plot([values[hi - 1], values[lo]], [ages[hi - 1], ages[lo]],
                    color=LINE_COLOR, linewidth=LINE_WIDTH,
                    linestyle=(0, (5, 4)), zorder=3)

        if dataset_id == "dust":
            ax.set_xscale("log")
        ax.set_xlabel(axis_label(dataset_id), fontsize=13, labelpad=8)
        ax.set_title(panel_title(dataset_id), fontsize=14, pad=12)
        ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, zorder=1)
        ax.tick_params(labelsize=11)

    axes[0].set_ylabel(AGE_LABEL, fontsize=14, labelpad=10)
    axes[0].set_ylim(AGE_MAX, AGE_MIN)
    axes[0].yaxis.set_major_locator(MultipleLocator(AGE_MAJOR))
    axes[0].yaxis.set_minor_locator(MultipleLocator(AGE_MINOR))

    fig.suptitle(
        "EPICA Dome C - five proxy records on a shared age axis\n"
        "Each record on its own chronology; hatched stages carry no "
        "measurement in that record",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, f"plate_columns_{variant}")


# ---------------------------------------------------------------------------
# Plate B: five rows, shared horizontal age axis
# ---------------------------------------------------------------------------


def plate_rows(frames: dict, stages: list[dict], variant: str) -> None:
    fig, axes = plt.subplots(
        len(ed.DATASET_ORDER), 1, figsize=(18, 20), sharex=True
    )

    for ax, dataset_id in zip(axes, ed.DATASET_ORDER):
        df = frames[dataset_id]
        ages = df["age_ka"].to_numpy()
        values = smooth_values(df["value"].to_numpy(), variant)

        ax.set_xlim(AGE_MIN, AGE_MAX)
        draw_bands(ax, stages, ages, horizontal=True)

        for lo, hi in segments(ages):
            ax.plot(ages[lo:hi], values[lo:hi], color=LINE_COLOR,
                    linewidth=LINE_WIDTH, zorder=3)
        segs = segments(ages)
        for (_, hi), (lo, _) in zip(segs, segs[1:]):
            ax.plot([ages[hi - 1], ages[lo]], [values[hi - 1], values[lo]],
                    color=LINE_COLOR, linewidth=LINE_WIDTH,
                    linestyle=(0, (5, 4)), zorder=3)

        if dataset_id == "dust":
            ax.set_yscale("log")
        ax.set_ylabel(axis_label(dataset_id), fontsize=12, labelpad=8)
        ax.set_title(panel_title(dataset_id), fontsize=13, loc="left", pad=6)
        ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6, zorder=1)
        ax.tick_params(labelsize=11)

    axes[-1].set_xlabel(AGE_LABEL, fontsize=14, labelpad=10)
    axes[-1].set_xlim(AGE_MIN, AGE_MAX)
    axes[-1].xaxis.set_major_locator(MultipleLocator(AGE_MAJOR))
    axes[-1].xaxis.set_minor_locator(MultipleLocator(AGE_MINOR))

    fig.suptitle(
        "EPICA Dome C - five proxy records on a shared age axis\n"
        "Each record on its own chronology; hatched stages carry no "
        "measurement in that record",
        fontsize=16,
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, f"plate_rows_{variant}")


# ---------------------------------------------------------------------------
# Plate C: the same stage boundaries in the depth axis of each record
# ---------------------------------------------------------------------------


def plate_boundary_depths(frames: dict, stages: list[dict]) -> None:
    """Where the age models pull apart, in metres.

    Every point is a stage boundary published as an age, carried into the
    depth axis of one record by linear interpolation - the same values the
    graph carries as geolod:MISBoundaryDepth. Nothing is extrapolated, so each
    curve stops where its record stops.

    Two panels, because one cannot do it. On the left the depth-age
    relationship itself, which is the context: 3200 m of core over 800 ka. On
    that axis the disagreement between the models is invisible, and plotting
    only that would suggest the five records agree. The right panel therefore
    shows each record's departure from the mean depth of the same boundary
    across all records, on an axis of tens of metres.

    That is where the point lies. The beginning of MIS 5 sits at 1734 m on
    AICC2023-ice and at 1782 m on EDC2-gas - 48 m apart in the same core, from
    one published boundary age. Below about 400 ka the spread narrows to a few
    metres, so the disagreement is not uniform and the plate should not be
    read as a constant offset between the models.
    """
    # Departure is measured against the mean over the records that cover a
    # given boundary, not against a chosen reference record: there is no
    # ground truth here, only models that disagree.
    depths: dict[str, dict[str, float]] = {}
    for dataset_id in ed.DATASET_ORDER:
        df = frames[dataset_id]
        depths[dataset_id] = {}
        for st in stages:
            d = ed.interpolate_depth(df, st["begin"])
            if d is not None:
                depths[dataset_id][st["stage"]] = d

    mean_depth = {}
    for st in stages:
        vals = [depths[k][st["stage"]] for k in ed.DATASET_ORDER
                if st["stage"] in depths[k]]
        if vals:
            mean_depth[st["stage"]] = sum(vals) / len(vals)

    fig, (ax, ax_dev) = plt.subplots(
        1, 2, figsize=(21, 11), gridspec_kw={"width_ratios": [1, 1.15]}
    )

    for dataset_id in ed.DATASET_ORDER:
        meta = ed.DATASETS[dataset_id]
        color = TRS_COLORS[meta["trs"]]
        label = f"{meta['label']} ({meta['trs']})"

        # NaN where this record has no depth for a boundary, so the line
        # breaks instead of running straight across the CH4 data gap - a
        # connecting segment there would look like an interpolation we have
        # just refused to make.
        xs, ys, dev = [], [], []
        for st in stages:
            if st["stage"] not in mean_depth:
                continue
            xs.append(st["begin"])
            d = depths[dataset_id].get(st["stage"])
            ys.append(d if d is not None else float("nan"))
            dev.append(
                d - mean_depth[st["stage"]] if d is not None else float("nan")
            )

        ax.plot(xs, ys, "o-", color=color, linewidth=1.6, markersize=5,
                label=label, zorder=3)
        ax_dev.plot(xs, dev, "o-", color=color, linewidth=1.8, markersize=6,
                    label=label, zorder=3)

    # Stage numbers along the top of the departure panel, so a reader can name
    # the boundary a point belongs to without counting.
    for st in stages:
        if st["stage"] not in mean_depth or st["begin"] > AGE_MAX:
            continue
        ax_dev.annotate(
            st["label"].replace("MIS ", ""),
            xy=(st["begin"], 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center", va="top", fontsize=9, color="#555555",
        )

    ax.set_xlim(AGE_MIN, AGE_MAX)
    ax.invert_yaxis()
    ax.set_xlabel(AGE_LABEL, fontsize=13, labelpad=8)
    ax.set_ylabel("Depth [m]", fontsize=13, labelpad=8)
    ax.set_title("Depth of each boundary", fontsize=14, pad=10)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=1)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95)

    ax_dev.set_xlim(AGE_MIN, AGE_MAX)
    ax_dev.axhline(0, color="#666666", linewidth=1.0, zorder=2)
    ax_dev.set_xlabel(AGE_LABEL, fontsize=13, labelpad=8)
    ax_dev.set_ylabel("Departure from the mean depth of that boundary [m]",
                      fontsize=13, labelpad=8)
    ax_dev.set_title(
        "Departure between the records, same boundary age",
        fontsize=14, pad=22,
    )
    ax_dev.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=1)
    ax_dev.tick_params(labelsize=11)

    fig.suptitle(
        "EPICA Dome C - the same Marine Isotope Stage boundaries in the depth "
        "axis of each record\n"
        "Boundary ages after Railsback et al. (2015), carried into depth by "
        "linear interpolation in each record's own chronology",
        fontsize=15,
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "plate_boundary_depths")


# ---------------------------------------------------------------------------
# The paper collage
# ---------------------------------------------------------------------------


def collage(frames: dict, stages: list[dict], name: str,
            entries: list[tuple[str, str]], title: str) -> None:
    """Panels side by side, raw curve behind the smoothed one.

    This is the figure the paper uses to show what the pipeline produces, so
    it deliberately shows both the measured series and the smoothed one in the
    same panel: the point being made is that the graph carries both, not that
    one of them is prettier. The footnote names the filter, because a smoothed
    curve without its window size is not reproducible.

    *entries* is a list of (dataset_id, axis) with axis one of "age_ka" or
    "depth_m" - which panels appear, and in which order, is the caller's
    decision rather than something this function guesses.
    """
    fig, axes = plt.subplots(1, len(entries), figsize=(4.6 * len(entries), 20))
    if len(entries) == 1:
        axes = [axes]

    for ax, (dataset_id, axis_key) in zip(axes, entries):
        meta = ed.DATASETS[dataset_id]
        df = frames[dataset_id]
        y = df[axis_key].to_numpy()
        raw = df["value"].to_numpy()
        smooth = st.rolling_median(raw, ROLLING_WINDOW)

        ax.set_ylim(float(y.max()), float(y.min()))
        ax.margins(y=0)

        if axis_key == "age_ka":
            draw_bands(ax, stages, y, horizontal=False)
            ylabel = AGE_LABEL
        else:
            ylabel = "Depth [m]"

        ages = df["age_ka"].to_numpy()
        for lo, hi in segments(ages):
            ax.plot(raw[lo:hi], y[lo:hi], linewidth=0.8,
                    color=LINE_COLOR_FADED, zorder=2)
            ax.plot(smooth[lo:hi], y[lo:hi], linewidth=1.4,
                    color=LINE_COLOR, zorder=3)
        segs = segments(ages)
        for (_, hi), (lo, _) in zip(segs, segs[1:]):
            ax.plot([smooth[hi - 1], smooth[lo]], [y[hi - 1], y[lo]],
                    linewidth=1.0, color=LINE_COLOR, linestyle=(0, (5, 4)),
                    zorder=3)

        if dataset_id == "dust":
            ticks, (x0, x1) = st.log_ticks(float(raw.min()), float(raw.max()))
            ax.set_xscale("log")
            ax.set_xticks(ticks)
            ax.set_xlim(x0, x1)
        else:
            ticks, (x0, x1), decimals = st.nice_ticks(
                float(raw.min()), float(raw.max()), target=4
            )
            ax.set_xticks(ticks)
            ax.set_xlim(x0, x1)
            ax.set_xticklabels([st.format_tick(t, decimals) for t in ticks])

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel(f"{meta['label']} [{meta['unit_label']}]", fontsize=13,
                      labelpad=8)
        ax.set_ylabel(ylabel, fontsize=13, labelpad=8, fontweight="bold")
        ax.set_title(f"EPICA - {meta['label']}\n({meta['trs']})",
                     fontsize=14, fontweight="bold", pad=10)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
        ax.tick_params(labelsize=11)
        ax.annotate(
            f"Rolling median filter  |  window = {ROLLING_WINDOW} pts",
            xy=(0.5, -0.012), xycoords="axes fraction", ha="center", va="top",
            fontsize=9, fontstyle="italic", color="#777777",
        )

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_all() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stages = ed.read_mis_stages()
    frames = {k: df for k, df in ed.load_all()}

    print("\n" + "─" * 60)
    print("Generating plates …")
    print("─" * 60)

    for variant in VARIANTS:
        plate_columns(frames, stages, variant)
        plate_rows(frames, stages, variant)
    plate_boundary_depths(frames, stages)

    # Die Collage des Beitrags, in zwei Zuschnitten. Der alte Vierer bleibt,
    # damit die bestehende Abbildung ersetzbar ist, ohne dass sich der Text
    # ändern muss; der Fünfer zeigt dasselbe Prinzip - Rohwerte und Glättung
    # nebeneinander - über alle Datensätze, die seit S2 im Graphen stehen.
    collage(
        frames, stages, "fig02_pipeline_outputs",
        [("d18o", "age_ka"), ("d18o", "depth_m"),
         ("ch4", "age_ka"), ("ch4", "depth_m")],
        "EPICA Dome C - pipeline outputs: measured series and rolling median, "
        "on the age and the depth axis",
    )
    collage(
        frames, stages, "fig02_pipeline_outputs_five",
        [(k, "age_ka") for k in ed.DATASET_ORDER],
        "EPICA Dome C - pipeline outputs: all five records, measured series "
        "and rolling median, each on its own chronology",
    )


if __name__ == "__main__":
    plt.rcParams["svg.hashsalt"] = "geo-lod"
    build_all()
