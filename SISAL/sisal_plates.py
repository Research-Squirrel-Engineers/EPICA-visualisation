"""
sisal_plates.py
===============
Die Tafeln des SISAL-Strangs, wie ``EPICA/epica_plates.py`` fuer EPICA. Beide
werden aus dem Plotskript des Strangs aufgerufen und tragen ihre Unterschriften
in dieselbe ``captions.yaml`` ein - zwei Sammler wuerden sich beim Schreiben
gegenseitig ueberschreiben.

Sechs Tafeln seit S3c.6, zwei je Klimasystem, und nicht dieselben wie bei
EPICA. Die Gruppierung ist der Punkt: zwoelf Sites nebeneinander sind zwoelf
Bilder, vier Sites eines Monsunsystems auf einer gemeinsamen Altersachse sind
eine Aussage.

``plate_coverage_<cluster>``
    Ein Balken je Speleothem, Alter auf der Achse, nach Site gruppiert. Es hat
    bei EPICA kein Gegenstueck, weil ein Bohrkern eine Reihe ist und eine
    Hoehle mehrere. Die Tafel beantwortet in einem Bild, warum die Sites nicht
    je eine Kurve sind: bei Sanbao liegen zwischen 0,4 und 13,6 ka sechs
    Speleotheme uebereinander, und zwischen 388 und 425 ka keines.

``plate_cluster_<cluster>``
    Vier Felder, ein Site je Feld, delta18O, Rohwerte blass hinter dem
    gleitenden Median, alle vier auf einer Altersachse. Ersetzt seit S3c.6
    ``plate_pipeline_outputs``, das sechs unverbundene Spalten mit je eigener
    Achse zeigte.

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

#: Screen dpi of the figure while it is being laid out. It is not the
#: resolution of the JPG - that comes from geo_lod_figures, which reads
#: it from the environment so that main.py --dpi can reach every
#: drawing script.
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
# Clusters
# ---------------------------------------------------------------------------

#: The three climate systems, in the order their plates are drawn. The key is
#: what ``SITES`` in the plot script carries, the label is what a title says,
#: and the phrase is how a caption names the cluster - "the European cluster",
#: not "the Europe cluster". A site whose key is not listed here draws no
#: plate and is reported.
CLUSTERS = (
    ("europe", "Europe", "European"),
    ("east_asian_monsoon", "East Asian monsoon", "East Asian monsoon"),
    ("south_american_monsoon", "South American monsoon",
     "South American monsoon"),
)

#: The measure the plates carry. Not both: Sanbao, Dongge and Xiaobailong
#: hold no delta13C in SISAL, so a delta13C plate of the East Asian cluster
#: would be one panel wide.
PLATE_MEASURE = "d18o"


def series_of(site):
    """The drawn series of a site: (name, colour, values, ages, breaks)."""
    return site["series"][PLATE_MEASURE]


def reach(site) -> float:
    """Oldest age the site's drawn series reaches, in ka."""
    return max(float(s[3].max()) for s in series_of(site))


def by_reach(sites) -> list:
    """Sites ordered by reach, the one going furthest back first.

    The same order the legend of a single figure uses for its speleothems, so
    that the plate reads in the direction the panels themselves do.
    """
    return sorted(sites, key=reach, reverse=True)


def age_span(sites) -> tuple[float, float]:
    """Youngest and oldest age over all drawn series of *sites*, in ka."""
    lo = min(float(s[3].min()) for site in sites for s in series_of(site))
    hi = max(float(s[3].max()) for site in sites for s in series_of(site))
    return lo, hi


def entity_count(sites) -> int:
    return sum(len(series_of(site)) for site in sites)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def plate_coverage(key, label, phrase, sites, stages, captions) -> None:
    """One bar per speleothem of one cluster, split at its breaks.

    The bar is drawn segment by segment rather than end to end, so a break in
    sampling is a hole in the bar and not a thinner line: the reader sees the
    same interruption the curve shows, at the same age.

    One plate per cluster and not one for all twelve sites: 72 bars at the
    spacing a name needs are two feet of figure, and the age axis of such a
    plate would be Sanbao's 465 ka for every row, including the eleven of
    Heshang and Huagapo that end before 10 ka.
    """
    rows = []
    for site in by_reach(sites):
        # delta18O carries every speleothem of a site; delta13C does not.
        for name, colour, _, ages, breaks in series_of(site):
            rows.append((site["site_name"], name, colour, ages, breaks))

    height = 2.6 + 0.30 * len(rows)
    fig = plt.figure(figsize=(16, height), dpi=DPI)
    ax = fig.add_subplot(111)

    age_max = max(float(a.max()) for _, _, _, a, _ in rows)
    ax.set_xlim(-age_max * 0.02, age_max * 1.02)
    ax.set_ylim(len(rows) - 0.5, -0.5)

    draw_bands(ax, stages, 0.0, age_max * 1.02, horizontal=True)

    # A segment shorter than this is drawn as a dot. On the East Asian axis,
    # which runs to 465 ka, HS4_2013 covers 171 years and DA_2009 332: as a
    # bar both are a fraction of a pixel wide, and the row reads as if the
    # speleothem had no record at all. The dot says the record is there and
    # shorter than the axis can show, which is the truth about it.
    min_visible = age_max * 0.004
    dots = 0

    labels = []
    previous_site = None
    for index, (site_name, name, colour, ages, breaks) in enumerate(rows):
        starts = [0] + [after for _, after in breaks]
        ends = [before for before, _ in breaks] + [len(ages) - 1]
        for start, end in zip(starts, ends):
            if ages[end] - ages[start] < min_visible:
                ax.plot((ages[start] + ages[end]) / 2.0, index, marker="o",
                        markersize=5, color=colour, zorder=3)
                dots += 1
                continue
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
    # than two rows are high - the labels of the small sites then run into one
    # another. Flat they need width, and width is the thing this figure has:
    # it is sixteen inches across.
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
        f"SISAL – age coverage, {label}",
        fontsize=18, fontweight="bold", pad=46,
    )
    fig.subplots_adjust(left=0.30, right=0.99, top=1 - 0.95 / height,
                        bottom=0.65 / height)

    name = f"plate_coverage_{key}"
    gf.save_figure(fig, os.path.join(OUTPUT_DIR, name))
    plt.close(fig)

    with_breaks = sum(1 for _, _, _, _, b in rows if b)
    site_names = ", ".join(site["site_name"] for site in by_reach(sites))
    dot_note = ""
    if dots:
        dot_note = (
            f" {dots} stretch{'es' if dots != 1 else ''} shorter than "
            f"{gf.format_tick(min_visible, 1)} ka {'are' if dots != 1 else 'is'} "
            f"drawn as a dot rather than a bar, being narrower than the axis "
            f"can resolve."
        )
    captions.add(
        name,
        caption=(
            f"Age coverage of the {len(rows)} speleothems of the "
            f"{phrase} cluster, one bar per speleothem, grouped by "
            f"site and coloured as in the site figures. The sites are "
            f"{site_names}, ordered by how far back their record reaches. A "
            f"bar is interrupted where the record is: {with_breaks} of them "
            f"carry a break in sampling.{dot_note} Overlapping bars within a "
            f"site are "
            f"why the samples of a site are not drawn as one curve. Marine "
            f"Isotope Stage bands follow Railsback et al. (2015)."
        ),
        license=FIGURE_LICENSE,
        sources=[SISAL_DOI],
    )
    print(f"    → {name}")


# ---------------------------------------------------------------------------
# Cluster panels
# ---------------------------------------------------------------------------


def plate_cluster(key, label, phrase, sites, stages, captions) -> None:
    """One panel per site of one climate system, on one shared age axis.

    The shared axis is what makes the plate a cluster rather than four
    pictures side by side: a swing at 130 ka in Sanbao and one in Dongge sit
    at the same height, and a record that does not reach that far shows it by
    stopping. It costs the short records their detail - Heshang ends at 9.5 ka
    and fills two per cent of an axis that runs to 465 ka - and that is the
    trade this plate makes. The single figures keep each record on its own
    scale.

    The value axis is not shared: delta18O of a European cave and of a
    monsoon cave differ by several per mil, and a common scale would flatten
    both. Each panel is scaled to its own record, as in the single figures.
    """
    sites = by_reach(sites)
    fig, axes = plt.subplots(
        1, len(sites), figsize=(4.2 * len(sites), 15), dpi=DPI, sharey=True
    )
    axes = np.atleast_1d(axes)

    age_lo, age_hi = age_span(sites)
    axes[0].set_ylim(age_hi, age_lo)
    ticks, _, decimals = gf.nice_ticks(age_lo, age_hi, target=8)
    step = ticks[1] - ticks[0] if len(ticks) > 1 else 50.0

    for ax, site in zip(axes, sites):
        series = series_of(site)
        all_ages = np.concatenate([a for _, _, _, a, _ in series])
        all_values = np.concatenate([v for _, _, v, _, _ in series])
        ax.margins(y=0)
        # Bands over the whole cluster range, filled only where this site has
        # a measurement: the stretch a record does not reach is then outlined
        # rather than blank, and the panels stay comparable.
        draw_bands(ax, stages, age_lo, age_hi, horizontal=False,
                   covered=all_ages)

        for _, colour, values, ages, breaks in series:
            gf.draw_profile(ax, values, ages, breaks, colour=LINE_COLOR_FADED,
                            linewidth=0.8, marker_size=5, zorder=2)
            smooth = gf.smooth_by_run(values, breaks, "median", ROLLING_WINDOW)
            gf.draw_profile(ax, smooth, ages, breaks, colour=colour,
                            linewidth=1.2, marker_size=5, zorder=3)

        vticks, (lo, hi), vdecimals = gf.nice_ticks(
            float(all_values.min()), float(all_values.max()), target=4
        )
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.xaxis.set_major_locator(FixedLocator(vticks))
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda val, pos, d=vdecimals: gf.format_tick(val, d))
        )
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_minor_locator(MultipleLocator(step / 5.0))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda val, pos: gf.format_tick(val, decimals))
        )
        ax.grid(axis="y", which="major", color="#cccccc", linewidth=0.8)
        ax.tick_params(labelsize=11)
        ax.set_title(
            f"{site['site_name']}\n{len(series)} speleothems, to "
            f"{gf.format_tick(reach(site), 1)} ka",
            fontsize=13, fontweight="bold", pad=10,
        )

    axes[0].set_ylabel("Age [ka]", fontsize=15, fontweight="bold", labelpad=10)
    fig.suptitle(
        f"SISAL – {label}, δ¹⁸O",
        fontsize=19, fontweight="bold", y=0.985,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.03,
                        wspace=0.18)

    name = f"plate_cluster_{key}"
    gf.save_figure(fig, os.path.join(OUTPUT_DIR, name))
    plt.close(fig)

    panels = ", ".join(site["site_name"] for site in sites)
    captions.add(
        name,
        caption=(
            f"δ¹⁸O of the {len(sites)} SISAL sites of the {phrase} "
            f"cluster, one panel per site: {panels}. Each panel shows the "
            f"measured series in grey behind a rolling median over "
            f"{ROLLING_WINDOW} points, one line per speleothem, coloured as "
            f"in the site figures; {entity_count(sites)} speleothems in all. "
            f"The panels share one age axis over the whole cluster, "
            f"{gf.format_tick(age_lo, 1)} to {gf.format_tick(age_hi, 1)} ka "
            f"BP, so that features can be compared between sites; the value "
            f"axis is scaled per panel. Marine Isotope Stage bands follow "
            f"Railsback et al. (2015); an outlined band marks a stage without "
            f"a measurement in that site."
        ),
        license=FIGURE_LICENSE,
        sources=[SISAL_DOI],
    )
    print(f"    → {name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_all(sites, captions) -> int:
    """Draws two plates per cluster, returns how many were written."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stages = gm.read_mis_stages()

    print("\n" + "─" * 60)
    print("Generating plates …")
    print("─" * 60)

    written = 0
    for key, label, phrase in CLUSTERS:
        members = [s for s in sites if s.get("cluster") == key]
        if not members:
            print(f"  ⚠  No sites in cluster {key} – skipping its plates.")
            continue
        print(f"\n  {label}: {len(members)} sites, "
              f"{entity_count(members)} speleothems")
        plate_coverage(key, label, phrase, members, stages, captions)
        plate_cluster(key, label, phrase, members, stages, captions)
        written += 2

    known = {key for key, _, _ in CLUSTERS}
    for site in sites:
        if site.get("cluster") not in known:
            print(f"  ⚠  {site['site_name']}: unknown cluster "
                  f"{site.get('cluster')!r} – in no plate.")
    return written
