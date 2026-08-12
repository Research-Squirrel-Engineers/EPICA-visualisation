# Datei: plot_sisal_from_csv.py
# Abbildungen des SISAL-Speleothem-Strangs (d18O, d13C gegen Alter).
#
# Reines Abbildungsskript. Der RDF-Export ist mit S3c.2 nach SISAL/sisal_rdf.py
# gewandert; mit S3c.4 liest auch dieses Skript aus dem geprueften Ausschnitt
# unter data/derived/sisal/sites/ statt aus den v_data_*.csv. Die alten CSV
# trugen den Dezimaltrenner-Fehler und einen engeren Stand: Botuvera 907 statt
# 920 Zeilen, Sanbao 5832 statt 6085, Buraca Gloriosa 1137 statt 1178.
#
# Drei Dinge, die sich mit S3c.4 geaendert haben:
#
#   1. Eine Linie je Entitaet, nicht je Site. Vorher wurden alle Proben einer
#      Site nach Alter sortiert und zu einer Kurve verbunden. Sanbao hat
#      achtzehn Speleotheme mit ueberlappenden Altersbereichen - zwischen 0,4
#      und 13,6 ka liegen sechs uebereinander -, und die Linie sprang zwischen
#      ihnen hin und her. Was wie ein verrauschtes Signal aussah, war die
#      Sortierung.
#   2. MIS-Baender aus dist/mis_stages.csv statt aus einer Liste im Code. Die
#      Liste war LR04 bis MIS 12; sie endete bei 533 ka und fuehrte MIS 3 als
#      Interstadial, waehrend das Leitschema Railsback es als warm fuehrt.
#   3. Achsen aus geo_lod_figures.nice_ticks statt aus handgesetzten Ticks.

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
ONTOLOGY_DIR = os.path.join(REPO_DIR, "ontology")
sys.path.insert(0, ONTOLOGY_DIR)

import geo_lod_figures as gf  # noqa: E402
import geo_lod_mis as gm  # noqa: E402
from geo_lod_captions import CaptionFile  # noqa: E402

sys.path.insert(0, SCRIPT_DIR)
import sisal_plates  # noqa: E402

# Byte-identical SVG across runs: the salt fixes the ids matplotlib gives clip
# paths, save_figure drops the date from the metadata.
plt.rcParams["svg.hashsalt"] = "geo-lod"


class Tee:
    """Writes simultaneously to stdout and a file."""

    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8", newline="\n")
        self.stdout = sys.stdout
        # The report holds box-drawing rules and delta signs, and the console
        # is not always what receives them: redirect the run to a file or to
        # nul and Python falls back to the locale encoding, cp1252 on Windows,
        # which has no U+2500. The run then dies on its first section header,
        # before a single figure is written - and a byte-stability check that
        # redirects output would compare a file against itself.
        try:
            self.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
        sys.stdout = self

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


SITES_DIR = os.path.join(REPO_DIR, "data", "derived", "sisal", "sites")
ENTITY_CSV = os.path.join(
    REPO_DIR, "data", "derived", "sisal", "tables", "entity.csv"
)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")
REPORT_DIR = os.path.join(SCRIPT_DIR, "report")

# ──────────────────────────────────────────────
# Shared plot settings
# ──────────────────────────────────────────────
FIGURE_SIZE = (10, 20)
DPI = 100
LINE_WIDTH = 1.0
LINE_WIDTH_SMOOTH = 1.6
RAW_ALPHA = 0.35  # measured series behind a smoothed one
GRID_COLOR = "#cccccc"
GRID_WIDTH = 1

AGE_MAJOR_TICK_INTERVAL = 20  # ka
AGE_MINOR_TICK_INTERVAL = 5

FONT_SIZE_LABEL = 26
FONT_SIZE_TICK = 22
TITLE_FONTSIZE = 26
FONT_SIZE_MIS = 14
FONT_SIZE_LEGEND = 13
LABEL_PAD = 12

ROLLING_WINDOW = 11
SG_WINDOW = 11
SG_POLYORDER = 2

MIS_COLOR_WARM = "#fddbc7"
MIS_COLOR_COLD = "#d6e8f7"
MIS_LABEL_COLOR_WARM = "#8b1a00"
MIS_LABEL_COLOR_COLD = "#003f6b"

FIGURE_LICENSE = "CC BY 4.0, Florian Thiery"
SISAL_DOI = "https://doi.org/10.5194/essd-16-1933-2024"

MEASURES = {
    "d18o": {
        "column": "d18o",
        "label": r"$\boldsymbol{\delta}^{\mathbf{18}}\mathbf{O}\ \mathbf{[‰]}$",
        "prose": "\u03b4\u00b9\u2078O",
    },
    "d13c": {
        "column": "d13c",
        "label": r"$\boldsymbol{\delta}^{\mathbf{13}}\mathbf{C}\ \mathbf{[‰]}$",
        "prose": "\u03b4\u00b9\u00b3C",
    },
}

VARIANTS = [
    ("unsmoothed", {}),
    (f"smooth{ROLLING_WINDOW}", {"rolling_window": ROLLING_WINDOW}),
    (f"savgol{SG_WINDOW}p{SG_POLYORDER}", {"use_savgol": True}),
]

VARIANT_PROSE = {
    "unsmoothed": "The measured series is shown unsmoothed.",
    f"smooth{ROLLING_WINDOW}": (
        f"A rolling median over {ROLLING_WINDOW} points is drawn over the "
        f"measured series."
    ),
    f"savgol{SG_WINDOW}p{SG_POLYORDER}": (
        f"A Savitzky-Golay filter over {SG_WINDOW} points at polynomial order "
        f"{SG_POLYORDER} is drawn over the measured series."
    ),
}

CAPTIONS = CaptionFile(
    os.path.join(SCRIPT_DIR, "captions.yaml"),
    header=(
        "captions.yaml - one entry per generated figure of the SISAL strand.\n"
        "\n"
        "Written by SISAL/plot_sisal_from_csv.py.\n"
        "Edit 'caption' freely: an entry whose caption differs from 'generated'\n"
        "is treated as hand-written and kept on the next run, while 'generated'\n"
        "is refreshed so the diff shows what the code would say now."
    ),
)


# ──────────────────────────────────────────────
# Input
# ──────────────────────────────────────────────
#: Which cuts are drawn, and under which name. The slug keeps the site_id in
#: front, as the published figures do; the file name of the cut is not the
#: figure name, because sites/spannagel_all.csv would give "spannagel_all".
SITES = [
    {"file": "spannagel_all.csv", "slug": "58_spannagel"},
    {"file": "sanbao.csv", "slug": "140_sanbao"},
    {"file": "botuvera.csv", "slug": "144_botuvera"},
    {"file": "corchia.csv", "slug": "145_corchia"},
    {"file": "piani_eterni_karst_system.csv", "slug": "202_pianieterni"},
    {"file": "buraca_gloriosa.csv", "slug": "275_buracagloriosa"},
]


def load_site(filename: str) -> pd.DataFrame:
    """One site of the cut, ages in ka BP, sorted per entity.

    Sorted by entity first and age second, and never by age alone: the age
    order across entities is what the old script drew, and it is not a series.
    """
    path = os.path.join(SITES_DIR, filename)
    df = pd.read_csv(path)
    for column in ("age_bp", "d18o", "d13c"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["age_ka"] = df["age_bp"] / 1000.0
    df = df.dropna(subset=["age_ka"])
    return df.sort_values(["entity_id", "age_ka"]).reset_index(drop=True)


def legend_columns(count: int) -> int:
    """Columns of the legend. Eighteen names in one column fill the figure."""
    return 1 if count <= 6 else 2 if count <= 12 else 3


def entity_order(df: pd.DataFrame) -> list[int]:
    """Entities of a site, the one reaching furthest back first.

    Ordered by age rather than by id, so the legend reads in the same
    direction as the axis and the colours do not jump about.
    """
    reach = df.groupby("entity_id")["age_ka"].max()
    return list(reach.sort_values(ascending=False).index)


# ──────────────────────────────────────────────
# MIS bands
# ──────────────────────────────────────────────


def draw_mis_bands(ax, y_min_ka, y_max_ka, covered=None):
    """Stage bands on the age axis, from dist/mis_stages.csv.

    A stage with no measurement anywhere in the site is outlined instead of
    filled, the same convention as EPICA: the band then reads as a stretch the
    record does not cover rather than as one it covers flatly.
    """
    stages = gm.read_mis_stages()
    y_lo, y_hi = min(y_min_ka, y_max_ka), max(y_min_ka, y_max_ka)
    ages = sorted(covered) if covered is not None else None
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    for st in stages:
        top, bot = st["end"], st["begin"]
        visible_top, visible_bot = max(top, y_lo), min(bot, y_hi)
        if visible_top >= visible_bot:
            continue

        warm = st["mode"] == "warm"
        color = MIS_COLOR_WARM if warm else MIS_COLOR_COLD
        label_color = MIS_LABEL_COLOR_WARM if warm else MIS_LABEL_COLOR_COLD

        has_data = True
        if ages is not None:
            has_data = any(top <= a < bot for a in ages)

        if has_data:
            ax.axhspan(top, bot, facecolor=color, alpha=1.0, zorder=0)
        else:
            ax.axhspan(
                top, bot, facecolor=color, alpha=0.35, edgecolor=label_color,
                linestyle=(0, (4, 3)), linewidth=1.2, zorder=0,
            )

        middle = (visible_top + visible_bot) / 2.0
        ax.text(
            0.99, middle, st["label"], transform=trans, ha="right",
            va="center", fontsize=FONT_SIZE_MIS, fontweight="bold",
            color=label_color, zorder=2,
        )


# ──────────────────────────────────────────────
# One figure
# ──────────────────────────────────────────────


def create_plot(
    series,
    xlabel,
    ylabel,
    title_text,
    output_filename,
    legend_loc,
    rolling_window=None,
    use_savgol=False,
):
    """One figure, one line per entity.

    *series* is a list of (name, colour, values, ages, breaks), each already
    sorted by age within the entity.
    """
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    ax = fig.add_subplot(111)

    all_ages = np.concatenate([ages for _, _, _, ages, _ in series])
    all_values = np.concatenate([values for _, _, values, _, _ in series])
    y_min, y_max = float(all_ages.min()), float(all_ages.max())
    ax.set_ylim(y_max, y_min)  # oldest at the bottom
    ax.margins(y=0)

    draw_mis_bands(ax, y_min_ka=y_min, y_max_ka=y_max, covered=all_ages)

    # Breaks are found per entity and never across the site: the wide age
    # spans between two speleothems are not gaps in a record, they are two
    # records. Drawing anything across them - dashed or otherwise - would
    # assert a continuity that does not exist.
    for index, (name, colour, values, ages, breaks) in enumerate(series):
        if use_savgol or rolling_window is not None:
            # Measured series behind, faded but in the same colour, so it is
            # readable which specimen a smoothed curve belongs to. It carries
            # the rings too: the ring marks the last measured sample before a
            # break, which is a statement about the series, not the filter.
            before = len(ax.lines)
            gf.draw_profile(
                ax, values, ages, breaks, colour=colour,
                linewidth=LINE_WIDTH, zorder=2 + 2 * index,
            )
            # Counted, not calculated: draw_profile lays down one line per run,
            # one per break and one marker line only if there is a break at
            # all, and an arithmetic guess at that number left part of the
            # measured series at full strength.
            for artist in ax.lines[before:]:
                artist.set_alpha(RAW_ALPHA)
            # Smoothed run by run, so a centred window cannot mix values from
            # either side of a break.
            if use_savgol:
                smooth = gf.smooth_by_run(
                    values, breaks, "savgol", SG_WINDOW, SG_POLYORDER
                )
            else:
                smooth = gf.smooth_by_run(
                    values, breaks, "median", rolling_window
                )
            gf.draw_profile(
                ax, smooth, ages, breaks, colour=colour,
                linewidth=LINE_WIDTH_SMOOTH, zorder=3 + 2 * index,
            )
        else:
            gf.draw_profile(
                ax, values, ages, breaks, colour=colour,
                linewidth=LINE_WIDTH, zorder=2 + 2 * index,
            )
        ax.plot([], [], color=colour, linewidth=2.0, label=name)

    ax.yaxis.set_major_locator(MultipleLocator(AGE_MAJOR_TICK_INTERVAL))
    ax.yaxis.set_minor_locator(MultipleLocator(AGE_MINOR_TICK_INTERVAL))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:.0f}"))
    ax.grid(axis="y", which="major", color=GRID_COLOR, linewidth=GRID_WIDTH)
    ax.tick_params(axis="y", which="minor", length=4, width=0.8)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Ticks from the data, over the range of all entities of the site, so the
    # three smoothing variants of one measure share a scale. The hand-written
    # lists this replaces were per site and could clip: the d13c list of
    # Buraca Gloriosa ran to -10 while the record reaches -11.76.
    ticks, (lo, hi), decimals = gf.nice_ticks(
        float(all_values.min()), float(all_values.max())
    )
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda val, pos: gf.format_tick(val, decimals))
    )

    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL, labelpad=LABEL_PAD)
    ax.set_ylabel(
        ylabel, fontsize=FONT_SIZE_LABEL, labelpad=LABEL_PAD, fontweight="bold"
    )

    if use_savgol:
        subtitle = (
            f"Savitzky-Golay filter  |  window = {SG_WINDOW} pts  |  "
            f"polyorder = {SG_POLYORDER}"
        )
    elif rolling_window is not None:
        subtitle = f"Rolling median filter  |  window = {rolling_window} pts"
    else:
        subtitle = "unsmoothed"

    ax.set_title(title_text, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=8)
    ax.annotate(
        subtitle, xy=(0.5, -0.01), xycoords="axes fraction", ha="center",
        va="top", fontsize=TITLE_FONTSIZE * 0.55, fontstyle="italic",
        color="#777777",
    )

    # The legend is drawn once to be measured and then placed. Its footprint
    # cannot be guessed from the number of entries: Sanbao's three columns of
    # speleothem names cover 60 per cent of the axis width, and an assumed
    # 42 per cent let the search declare the lower left corner empty while
    # SB-58 ran right through it.
    #
    # *legend_loc* is None for the first variant of a measure and the answer
    # is handed back to the caller, so all three variants of one record put
    # their legend in the same place.
    columns = legend_columns(len(series))
    style = dict(
        fontsize=FONT_SIZE_LEGEND, ncol=columns, framealpha=0.9,
        edgecolor="#999999", borderpad=0.6, labelspacing=0.4,
        columnspacing=1.2, handlelength=1.6,
    )
    legend = ax.legend(loc="upper left", **style)
    if legend_loc is None:
        fig.canvas.draw()
        box = legend.get_window_extent().transformed(ax.transAxes.inverted())
        legend_loc = gf.best_legend_loc(
            [(v, p) for _, _, v, p, _ in series],
            (lo, hi),
            (y_max, y_min),
            width=min(box.width + 0.03, 0.95),
            height=min(box.height + 0.03, 0.95),
        )
    legend.remove()
    if legend_loc == gf.LEGEND_OUTSIDE:
        # No room inside: below the axes, under the smoothing subtitle. Laid
        # out flat in three rows rather than as a tall block, so it takes
        # height from the margin and not from the record.
        style["ncol"] = max(1, -(-len(series) // 3))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.022), **style)
    else:
        ax.legend(loc=legend_loc, **style)

    ax.tick_params(axis="x", labelsize=FONT_SIZE_TICK)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)

    # Over geo_lod_figures, not plt.savefig: matplotlib otherwise opens the
    # target in text mode and Python writes CRLF on Windows, while
    # .gitattributes stores LF.
    gf.save_figure(fig, output_filename, dpi=DPI)
    plt.close(fig)
    return legend_loc


# ──────────────────────────────────────────────
# One site
# ──────────────────────────────────────────────


def caption_for(site_name, measure, variant, series):
    """The facts the drawing code knows and prose does not keep up with."""
    info = MEASURES[measure]
    points = sum(len(values) for _, _, values, _, _ in series)
    ages = np.concatenate([a for _, _, _, a, _ in series])
    names = ", ".join(name for name, _, _, _, _ in series)
    breaks_total = sum(len(b) for _, _, _, _, b in series)

    gaps = ""
    if breaks_total:
        gaps = (
            f" {breaks_total} break{'s' if breaks_total != 1 else ''} in "
            f"sampling {'are' if breaks_total != 1 else 'is'} drawn dashed, "
            f"with the last sample before and the first after it ringed."
        )

    return (
        f"{info['prose']} of {site_name} from the SISAL v3 database, "
        f"{points} measurements on {len(series)} speleothem"
        f"{'s' if len(series) != 1 else ''} ({names}), ages from "
        # format_tick and not an f-string: Sanbao's SB-43 reaches -0.01 ka, a
        # post-1950 extrapolation the loader already flags, and plain
        # formatting prints that as "-0.0".
        f"{gf.format_tick(float(ages.min()), 1)} to "
        f"{gf.format_tick(float(ages.max()), 1)} ka BP on the linear "
        f"interpolation age model. Each speleothem is drawn as its own line; "
        f"together they are not one record. {VARIANT_PROSE[variant]}{gaps} "
        f"Marine Isotope Stage bands follow Railsback et al. (2015); an "
        f"outlined band marks a stage without a measurement in this site."
    )


def collect_site(cfg, entity_names) -> dict:
    """Everything the figures and the plates need from one site, read once.

    The plates draw the same curves as the single figures, and reading the cut
    twice would let the two drift apart at the first change to the loader.
    """
    df = load_site(cfg["file"])
    order = entity_order(df)
    colours = gf.series_colours(len(order))

    by_measure = {}
    for measure, info in MEASURES.items():
        series = []
        for colour, eid in zip(colours, order):
            g = df[(df.entity_id == eid) & df[info["column"]].notna()]
            if g.empty:
                continue
            ages = g["age_ka"].to_numpy(dtype=float)
            values = g[info["column"]].to_numpy(dtype=float)
            breaks = gf.find_breaks_relative(ages)
            series.append(
                (entity_names.get(eid, str(eid)), colour, values, ages, breaks)
            )
        by_measure[measure] = series

    return {
        "slug": cfg["slug"],
        "site_name": df["site_name"].iloc[0],
        "site_id": int(df["site_id"].iloc[0]),
        "rows": len(df),
        "entities": len(order),
        "age_min": float(df["age_ka"].min()),
        "age_max": float(df["age_ka"].max()),
        "series": by_measure,
    }


def plot_site(site):
    site_name = site["site_name"]

    print(f"\n{'─' * 60}")
    print(f"Loading: {site_name}")
    print("─" * 60)
    print(f"  Loaded: {site_name} (site_id {site['site_id']})")
    print(f"  Data points: {site['rows']}, entities: {site['entities']}")
    print(f"  Age: {site['age_min']:.1f} – {site['age_max']:.1f} ka BP")

    made = 0
    for measure, info in MEASURES.items():
        series = site["series"][measure]
        if not series:
            print(f"\n  ⚠  No {info['prose']} for {site_name} – skipping.")
            continue

        # Decided by the first figure of this measure and reused by the other
        # two, so the legend does not move between the variants of one record.
        legend_loc = None

        carried = sum(len(v) for _, _, v, _, _ in series)
        breaks_total = sum(len(b) for _, _, _, _, b in series)
        print(
            f"\n  {info['prose']}: {carried} measurements on {len(series)} of "
            f"{site['entities']} speleothems, "
            f"{breaks_total} break{'s' if breaks_total != 1 else ''}"
        )
        for variant, kwargs in VARIANTS:
            key = f"{site['slug']}_{measure}_age_{variant}"
            print(f"    → {key}")
            legend_loc = create_plot(
                series=series,
                xlabel=info["label"],
                ylabel="Age [ka]",
                title_text=f"SISAL – {site_name}",
                output_filename=os.path.join(OUTPUT_DIR, key),
                legend_loc=legend_loc,
                **kwargs,
            )
            if variant == VARIANTS[0][0]:
                print(f"      legend: {legend_loc}")
            CAPTIONS.add(
                key,
                caption=caption_for(site_name, measure, variant, series),
                license=FIGURE_LICENSE,
                sources=[SISAL_DOI],
            )
            made += 1
    return made


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    report_path = os.path.join(REPORT_DIR, "report.txt")
    tee = Tee(report_path)

    print("=" * 60)
    print("SISAL Speleothem – Plot Generator")
    print("=" * 60)

    # Names for the legend. entity_name is in the flat cut as well, but the
    # entity table is where an entity is described, and the flat cut is meant
    # to stay a convenience, not a second source.
    entity = pd.read_csv(ENTITY_CSV)
    entity_names = dict(zip(entity.entity_id, entity.entity_name))

    sites = []
    for cfg in SITES:
        if not os.path.exists(os.path.join(SITES_DIR, cfg["file"])):
            print(f"  ⚠  Missing cut: {cfg['file']} – skipping.")
            continue
        sites.append(collect_site(cfg, entity_names))

    total = sum(plot_site(site) for site in sites)

    # The plates come last and share the caption file: two collectors would
    # overwrite each other, and the file has to be complete after one run.
    total += sisal_plates.build_all(sites, CAPTIONS)

    print("\n" + "=" * 60)
    print(f"Done! Plots saved to '{OUTPUT_DIR}/'")
    print(f"Total: {total} plots")
    print("=" * 60)
    CAPTIONS.write()
    print(f"Report saved: {report_path}")
    tee.close()


if __name__ == "__main__":
    main()
