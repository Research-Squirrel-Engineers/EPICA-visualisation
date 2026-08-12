"""
sisal_plates.py
===============
Die Tafeln des SISAL-Strangs, wie ``EPICA/epica_plates.py`` fuer EPICA. Beide
werden aus dem Plotskript des Strangs aufgerufen und tragen ihre Unterschriften
in dieselbe ``captions.yaml`` ein - zwei Sammler wuerden sich beim Schreiben
gegenseitig ueberschreiben.

Zwei Tafeln, und nicht dieselben wie bei EPICA.

``plate_coverage``
    Ein Balken je Speleothem, Alter auf der Achse, nach Site gruppiert. Es hat
    bei EPICA kein Gegenstueck, weil ein Bohrkern eine Reihe ist und eine
    Hoehle mehrere. Die Tafel beantwortet in einem Bild, warum die sechs Sites
    nicht sechs Kurven sind: bei Sanbao liegen zwischen 0,4 und 13,6 ka sechs
    Speleotheme uebereinander, und zwischen 388 und 425 ka keines.

``plate_pipeline_outputs``
    Sechs Felder, ein Site je Feld, delta18O, Rohwerte blass hinter dem
    gleitenden Median. Das Gegenstueck zu ``plate_pipeline_outputs_five``.

Ohne Gegenstueck bleibt ``plate_boundary_depths``: die Tiefe je Probe steht
seit S3c.2 als ``geolod:atDepth_mm`` im Graphen, aber nicht im flachen
Ausschnitt unter ``data/derived/sisal/sites/``, aus dem die Abbildungen
entstehen.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "ontology"))

import geo_lod_figures as gf  # noqa: E402
import geo_lod_mis as gm  # noqa: E402

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")

MIS_COLOR_WARM = "#fddbc7"
MIS_COLOR_COLD = "#d6e8f7"
MIS_LABEL_COLOR_WARM = "#8b1a00"
MIS_LABEL_COLOR_COLD = "#003f6b"

LINE_COLOR_FADED = "#aaaaaa"
FIGURE_LICENSE = "CC BY 4.0, Florian Thiery"
SISAL_DOI = "https://doi.org/10.5194/essd-16-1933-2024"
ROLLING_WINDOW = 11

DPI = 100


def draw_bands(ax, stages, lo, hi, horizontal: bool, covered=None):
    """MIS bands over the visible age range, on x or on y."""
    ages = sorted(covered) if covered is not None else None
    if horizontal:
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        span = ax.axvspan
    else:
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        span = ax.axhspan

    for st in stages:
        top, bot = st["end"], st["begin"]
        if max(top, lo) >= min(bot, hi):
            continue
        warm = st["mode"] == "warm"
        colour = MIS_COLOR_WARM if warm else MIS_COLOR_COLD
        label_colour = MIS_LABEL_COLOR_WARM if warm else MIS_LABEL_COLOR_COLD

        has_data = True if ages is None else any(top <= a < bot for a in ages)
        if has_data:
            span(top, bot, facecolor=colour, alpha=1.0, zorder=0)
        else:
            span(top, bot, facecolor=colour, alpha=0.35,
                 edgecolor=label_colour, linestyle=(0, (4, 3)),
                 linewidth=1.0, zorder=0)

        middle = (max(top, lo) + min(bot, hi)) / 2.0
        if horizontal:
            ax.text(middle, 1.005, st["label"], transform=trans, ha="center",
                    va="bottom", fontsize=9, fontweight="bold",
                    color=label_colour, rotation=90, zorder=2)
        else:
            ax.text(0.99, middle, st["label"], transform=trans, ha="right",
                    va="center", fontsize=10, fontweight="bold",
                    color=label_colour, zorder=2)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def plate_coverage(sites, stages, captions) -> None:
    """One bar per speleothem, split at its breaks.

    The bar is drawn segment by segment rather than end to end, so a break in
    sampling is a hole in the bar and not a thinner line: the reader sees the
    same interruption the curve shows, at the same age.
    """
    rows = []
    for site in sites:
        # delta18O carries every speleothem of a site; delta13C does not.
        for name, colour, _, ages, breaks in site["series"]["d18o"]:
            rows.append((site["site_name"], name, colour, ages, breaks))

    height = 2.6 + 0.30 * len(rows)
    fig = plt.figure(figsize=(16, height), dpi=DPI)
    ax = fig.add_subplot(111)

    age_max = max(float(a.max()) for _, _, _, a, _ in rows)
    ax.set_xlim(-8, age_max * 1.02)
    ax.set_ylim(len(rows) - 0.5, -0.5)

    draw_bands(ax, stages, 0.0, age_max * 1.02, horizontal=True)

    labels = []
    previous_site = None
    for index, (site_name, name, colour, ages, breaks) in enumerate(rows):
        starts = [0] + [after for _, after in breaks]
        ends = [before for before, _ in breaks] + [len(ages) - 1]
        for start, end in zip(starts, ends):
            ax.plot([ages[start], ages[end]], [index, index],
                    color=colour, linewidth=7, solid_capstyle="butt",
                    zorder=3)
        if previous_site is not None and site_name != previous_site:
            ax.axhline(index - 0.5, color="#666666", linewidth=0.8, zorder=4)
        previous_site = site_name
        labels.append(name)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(axis="y", length=0)

    ticks, (lo, hi), decimals = gf.nice_ticks(0.0, age_max, target=10)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda val, pos: gf.format_tick(val, decimals))
    )
    ax.set_xlabel("Age [ka]", fontsize=14, labelpad=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="x", which="major", color="#bbbbbb", linewidth=0.8, zorder=1)

    # Site names down the left, once per group, flat and in their own strip.
    # Turned upright they take no width, but Botuvera and Piani Eterni have
    # two rows each and "Piani Eterni karst system" set vertically is longer
    # than two rows are high - the labels of the four small sites at the
    # bottom then run into one another. Flat they need width, and width is the
    # thing this figure has: it is sixteen inches across.
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    start = 0
    for index in range(len(rows) + 1):
        ends_here = index == len(rows) or rows[index][0] != rows[start][0]
        if ends_here:
            ax.text(-0.30, (start + index - 1) / 2.0, rows[start][0],
                    transform=trans, ha="left", va="center", fontsize=11,
                    fontweight="bold")
            start = index

    # The stage labels stand upright above the axes, so the title needs room
    # above them rather than the usual few points.
    ax.set_title(
        f"SISAL – age coverage of the {len(rows)} speleothems",
        fontsize=18, fontweight="bold", pad=46,
    )
    fig.subplots_adjust(left=0.30, right=0.99, top=0.905, bottom=0.07)

    name = "plate_coverage"
    gf.save_figure(fig, os.path.join(OUTPUT_DIR, name), dpi=DPI)
    plt.close(fig)

    with_breaks = sum(1 for _, _, _, _, b in rows if b)
    captions.add(
        name,
        caption=(
            f"Age coverage of the {len(rows)} speleothems of the six SISAL "
            f"sites in this repository, one bar per speleothem, grouped by "
            f"site and coloured as in the site figures. A bar is interrupted "
            f"where the record is: {with_breaks} of them carry a break in "
            f"sampling. Overlapping bars within a site are why the samples of "
            f"a site are not drawn as one curve - at Sanbao six speleothems "
            f"cover the interval between 0.4 and 13.6 ka. Marine Isotope "
            f"Stage bands follow Railsback et al. (2015)."
        ),
        license=FIGURE_LICENSE,
        sources=[SISAL_DOI],
    )
    print(f"    → {name}")


# ---------------------------------------------------------------------------
# Pipeline outputs
# ---------------------------------------------------------------------------


def plate_pipeline_outputs(sites, stages, captions) -> None:
    """One panel per site, measured series behind a rolling median."""
    fig, axes = plt.subplots(
        1, len(sites), figsize=(4.2 * len(sites), 15), dpi=DPI
    )
    axes = np.atleast_1d(axes)

    for ax, site in zip(axes, sites):
        series = site["series"]["d18o"]
        all_ages = np.concatenate([a for _, _, _, a, _ in series])
        all_values = np.concatenate([v for _, _, v, _, _ in series])
        ax.set_ylim(float(all_ages.max()), float(all_ages.min()))
        ax.margins(y=0)
        draw_bands(ax, stages, float(all_ages.min()), float(all_ages.max()),
                   horizontal=False, covered=all_ages)

        for _, colour, values, ages, breaks in series:
            gf.draw_profile(ax, values, ages, breaks, colour=LINE_COLOR_FADED,
                            linewidth=0.8, marker_size=5, zorder=2)
            smooth = gf.smooth_by_run(values, breaks, "median", ROLLING_WINDOW)
            gf.draw_profile(ax, smooth, ages, breaks, colour=colour,
                            linewidth=1.2, marker_size=5, zorder=3)

        ticks, (lo, hi), decimals = gf.nice_ticks(
            float(all_values.min()), float(all_values.max()), target=4
        )
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda val, pos: gf.format_tick(val, decimals))
        )
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.grid(axis="y", which="major", color="#cccccc", linewidth=0.8)
        ax.tick_params(labelsize=11)
        ax.set_title(
            f"{site['site_name']}\n{len(series)} speleothems",
            fontsize=13, fontweight="bold", pad=10,
        )

    axes[0].set_ylabel("Age [ka]", fontsize=15, fontweight="bold", labelpad=10)
    fig.suptitle(
        "SISAL – pipeline outputs of the speleothem strand, δ¹⁸O",
        fontsize=19, fontweight="bold", y=0.985,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.03,
                        wspace=0.28)

    name = "plate_pipeline_outputs"
    gf.save_figure(fig, os.path.join(OUTPUT_DIR, name), dpi=DPI)
    plt.close(fig)

    panels = ", ".join(site["site_name"] for site in sites)
    captions.add(
        name,
        caption=(
            f"δ¹⁸O of the six SISAL sites of this repository, one panel per "
            f"site: {panels}. Each panel shows the measured series in grey "
            f"behind a rolling median over {ROLLING_WINDOW} points, one line "
            f"per speleothem, coloured as in the site figures. The panels do "
            f"not share an age axis: each is scaled to its own record, and "
            f"the age range is stated on the axis. Marine Isotope Stage bands "
            f"follow Railsback et al. (2015); an outlined band marks a stage "
            f"without a measurement in that site."
        ),
        license=FIGURE_LICENSE,
        sources=[SISAL_DOI],
    )
    print(f"    → {name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_all(sites, captions) -> int:
    """Draws both plates, returns how many were written."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stages = gm.read_mis_stages()

    print("\n" + "─" * 60)
    print("Generating plates …")
    print("─" * 60)

    plate_coverage(sites, stages, captions)
    plate_pipeline_outputs(sites, stages, captions)
    return 2
