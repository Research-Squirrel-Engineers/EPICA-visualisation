# Datei: plot_sisal_from_csv.py
# Visualisierung von SISAL-Speleothem-Daten (d18O, d13C) analog zum EPICA-Script
# Basiert auf: plot_epica_from_tab.py
#
# Reines Abbildungsskript. Der RDF-Export ist mit S3c.2 nach SISAL/sisal_rdf.py
# gewandert, das aus dem geprueften Ausschnitt in data/derived/sisal/ schreibt
# statt aus den v_*-CSV. Die Cave-Sites, die hier frueher aus v_sites_all.csv
# entstanden, kommen jetzt aus der Datenbank; die archaeologische Anreicherung
# liegt in data/curated/sisal_site_annotations.csv.
#
# Offen und fuer S3c.4 vorgemerkt: die Eingabe dieses Skripts sind weiterhin die
# v_data_*.csv mit dem Dezimaltrenner-Fehler.

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, FixedLocator
import matplotlib.transforms as transforms

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ontology"),
)
import geo_lod_figures as gf  # noqa: E402
from scipy.signal import savgol_filter

# Byte-identical SVG across runs. Two things vary otherwise: the <dc:date>
# in the SVG metadata, and the random ids matplotlib gives clip paths. The
# salt fixes the ids, metadata={"Date": None} at save time drops the date.
plt.rcParams["svg.hashsalt"] = "geo-lod"


class Tee:
    """Writes simultaneously to stdout and a file."""

    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8", newline="\n")
        self.stdout = sys.stdout
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


# Set working directory to the script's folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create output directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")
REPORT_DIR = os.path.join(SCRIPT_DIR, "report")
ONTOLOGY_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "ontology")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ONTOLOGY_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Shared plot settings
# ──────────────────────────────────────────────
FIGURE_SIZE = (10, 20)
DPI = 100
LINE_COLOR = "black"
LINE_WIDTH = 1
GRID_COLOR = "#cccccc"
GRID_WIDTH = 1

# Y-axis ticks (age in ka BP)
AGE_MAJOR_TICK_INTERVAL = 20  # major tick every 20 ka
AGE_MINOR_TICK_INTERVAL = 5  # minor tick every 5 ka

FONT_SIZE_LABEL = 26
FONT_SIZE_TICK = 22
TITLE_FONTSIZE = 26
FONT_SIZE_MIS = 14

# Smoothing
ROLLING_WINDOW = 11
SG_WINDOW = 11
SG_POLYORDER = 2
LINE_COLOR_FADED = "#aaaaaa"
LINE_WIDTH_SMOOTH = 1.5
LABEL_PAD = 12

# ──────────────────────────────────────────────
# MIS intervals (boundaries in ka BP, LR04)
# ──────────────────────────────────────────────
MIS_COLOR_WARM = "#fddbc7"
MIS_COLOR_INTERSTADIAL = "#fef0e6"
MIS_COLOR_COLD = "#d6e8f7"

MIS_INTERVALS = [
    (0, 14, "MIS 1", "warm"),
    (14, 29, "MIS 2", "cold"),
    (29, 57, "MIS 3", "inter"),
    (57, 71, "MIS 4", "cold"),
    (71, 130, "MIS 5", "warm"),
    (130, 191, "MIS 6", "cold"),
    (191, 243, "MIS 7", "warm"),
    (243, 300, "MIS 8", "cold"),
    (300, 337, "MIS 9", "warm"),
    (337, 374, "MIS 10", "cold"),
    (374, 424, "MIS 11", "warm"),
    (424, 533, "MIS 12", "cold"),
]

# ──────────────────────────────────────────────
# Load SISAL CSV
# ──────────────────────────────────────────────


def load_sisal_csv(filepath):
    """
    Liest eine SISAL-CSV-Datei ein.
    Erwartet Spalten: site_id, site_name, entity_id, entity_name,
                      sample_id, age_bp, d18o_permille, d13c_permille
    Returns age in ka BP (age_bp / 1000).
    """
    df = pd.read_csv(filepath)

    df["age_bp"] = pd.to_numeric(df["age_bp"], errors="coerce")
    df["d18o_permille"] = pd.to_numeric(df["d18o_permille"], errors="coerce")
    df["d13c_permille"] = pd.to_numeric(df["d13c_permille"], errors="coerce")

    df["age_ka"] = df["age_bp"] / 1000.0  # in ka BP umrechnen

    df = df.dropna(subset=["age_ka"]).sort_values("age_ka").reset_index(drop=True)

    site_name = df["site_name"].iloc[0]
    entity_ids = df["entity_id"].nunique()
    print(f"  Loaded: {site_name}")
    print(f"  Data points: {len(df)}, entities: {entity_ids}")
    print(f"  Age: {df['age_ka'].min():.1f} – {df['age_ka'].max():.1f} ka BP")
    if df["d18o_permille"].notna().any():
        print(
            f"  d18O: {df['d18o_permille'].min():.2f} – {df['d18o_permille'].max():.2f} ‰"
        )
    if df["d13c_permille"].notna().any():
        print(
            f"  d13C: {df['d13c_permille'].min():.2f} – {df['d13c_permille'].max():.2f} ‰"
        )

    return df


# ──────────────────────────────────────────────
# MIS bands
# ──────────────────────────────────────────────


def draw_mis_bands(ax, y_min_ka, y_max_ka):
    mis_trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    type_config = {
        "warm": (MIS_COLOR_WARM, "#8b1a00", False),
        "inter": (MIS_COLOR_INTERSTADIAL, "#8b1a00", False),
        "cold": (MIS_COLOR_COLD, "#003f6b", False),
    }

    for age_top, age_bot, label, mis_type in MIS_INTERVALS:
        y_lo = min(y_min_ka, y_max_ka)
        y_hi = max(y_min_ka, y_max_ka)
        visible_top = max(age_top, y_lo)
        visible_bot = min(age_bot, y_hi)
        if visible_top >= visible_bot:
            continue

        color, label_color, _ = type_config.get(
            mis_type, (MIS_COLOR_COLD, "#003f6b", False)
        )
        ax.axhspan(age_top, age_bot, facecolor=color, alpha=1.0, zorder=0)

        y_label = (visible_top + visible_bot) / 2.0
        ax.text(
            0.99,
            y_label,
            label,
            transform=mis_trans,
            ha="right",
            va="center",
            fontsize=FONT_SIZE_MIS,
            fontweight="bold",
            color=label_color,
            zorder=2,
        )


# ──────────────────────────────────────────────
# Generic plot function (mirrors EPICA script)
# ──────────────────────────────────────────────


def create_plot(
    x_values,
    y_values,
    xlabel,
    ylabel,
    title_text,
    output_filename,
    y_major_interval,
    y_minor_interval,
    x_ticks=None,
    x_padding=0.05,
    invert_y=True,
    show_mis=False,
    rolling_window=None,
    use_savgol=False,
):
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    ax = fig.add_subplot(111)

    y_min, y_max = y_values.min(), y_values.max()
    if invert_y:
        ax.set_ylim(y_max, y_min)
    else:
        ax.set_ylim(y_min, y_max)
    ax.margins(y=0)

    if show_mis:
        draw_mis_bands(ax, y_min_ka=y_min, y_max_ka=y_max)

    if use_savgol:
        ax.plot(
            x_values, y_values, linewidth=LINE_WIDTH, color=LINE_COLOR_FADED, zorder=2
        )
        smooth = savgol_filter(
            x_values.values, window_length=SG_WINDOW, polyorder=SG_POLYORDER
        )
        ax.plot(
            smooth, y_values, linewidth=LINE_WIDTH_SMOOTH, color=LINE_COLOR, zorder=3
        )
    elif rolling_window is not None:
        ax.plot(
            x_values, y_values, linewidth=LINE_WIDTH, color=LINE_COLOR_FADED, zorder=2
        )
        smooth = (
            pd.Series(x_values.values)
            .rolling(window=rolling_window, center=True, min_periods=1)
            .median()
        )
        ax.plot(
            smooth.values,
            y_values,
            linewidth=LINE_WIDTH_SMOOTH,
            color=LINE_COLOR,
            zorder=3,
        )
    else:
        ax.plot(x_values, y_values, linewidth=LINE_WIDTH, color=LINE_COLOR, zorder=2)

    ax.yaxis.set_major_locator(MultipleLocator(y_major_interval))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor_interval))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:.0f}"))
    ax.grid(axis="y", which="major", color=GRID_COLOR, linewidth=GRID_WIDTH)
    ax.tick_params(axis="y", which="minor", length=4, width=0.8)

    # X-Achse oben
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    x_min, x_max = x_values.min(), x_values.max()
    x_range = x_max - x_min

    if x_ticks is not None:
        ax.xaxis.set_major_locator(FixedLocator(x_ticks))
        t_min, t_max = min(x_ticks), max(x_ticks)
        span = t_max - t_min
        pad = span * 0.05 if span > 0 else 0.5
        ax.set_xlim(t_min - pad, t_max + pad)
    else:
        ax.set_xlim(x_min - x_range * x_padding, x_max + x_range * x_padding)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:.1f}"))

    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL, labelpad=LABEL_PAD)
    ax.set_ylabel(
        ylabel, fontsize=FONT_SIZE_LABEL, labelpad=LABEL_PAD, fontweight="bold"
    )

    # Smoothing subtitle
    if use_savgol:
        subtitle = f"Savitzky-Golay filter  |  window = {SG_WINDOW} pts  |  polyorder = {SG_POLYORDER}"
    elif rolling_window is not None:
        subtitle = f"Rolling median filter  |  window = {rolling_window} pts"
    else:
        subtitle = "unsmoothed"

    ax.set_title(title_text, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=8)
    ax.annotate(
        subtitle,
        xy=(0.5, -0.01),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=TITLE_FONTSIZE * 0.55,
        fontstyle="italic",
        color="#777777",
    )

    ax.tick_params(axis="x", labelsize=FONT_SIZE_TICK)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)

    # Über geo_lod_figures, nicht über plt.savefig: matplotlib öffnet die
    # Zieldatei sonst im Textmodus, und Python schreibt auf Windows CRLF,
    # während .gitattributes LF ablegt - jedes SVG wiche dann dauerhaft von
    # seiner eigenen abgelegten Form ab. Dieselbe Stelle wie in EPICA (S2).
    gf.save_figure(plt.gcf(), output_filename, dpi=DPI)
    plt.close()


# ──────────────────────────────────────────────
# Helper: generate plots for one cave
# ──────────────────────────────────────────────


def generate_cave_plots(df, site_name, site_slug, d18o_ticks=None, d13c_ticks=None):
    """
    Generates up to 6 plots per cave:
      d18O vs Age ka BP  – unsmoothed, rolling median, Savitzky-Golay
      d13C vs Age ka BP  – unsmoothed, rolling median, Savitzky-Golay
    """

    # Saubere Teilsets ohne NaN
    mask_d18o = df["d18o_permille"].notna() & df["age_ka"].notna()
    mask_d13c = df["d13c_permille"].notna() & df["age_ka"].notna()

    df_d18o = df[mask_d18o].copy()
    df_d13c = df[mask_d13c].copy()

    plots = []

    # ── d18O ────────────────────────────────────────────────────────────
    for sm_label, sm_kwargs in [
        ("unsmoothed", {}),
        (f"smooth{ROLLING_WINDOW}", {"rolling_window": ROLLING_WINDOW}),
        (f"savgol{SG_WINDOW}p{SG_POLYORDER}", {"use_savgol": True}),
    ]:
        if df_d18o.empty:
            print(f"  ⚠  No d18O data for {site_name}, skipping.")
            break
        plots.append(
            {
                "x": df_d18o["d18o_permille"],
                "y": df_d18o["age_ka"],
                "xlabel": r"$\boldsymbol{\delta}^{\mathbf{18}}\mathbf{O}\ \mathbf{[‰]}$",
                "ylabel": "Age [ka BP]",
                "title": f"SISAL – {site_name}",
                "filename": os.path.join(
                    OUTPUT_DIR, f"{site_slug}_d18o_age_{sm_label}"
                ),
                "x_ticks": d18o_ticks,
                "show_mis": True,
                **sm_kwargs,
            }
        )

    # ── d13C ────────────────────────────────────────────────────────────
    if d13c_ticks is None or df_d13c.empty:
        print(f"  ⚠  No d13C data for {site_name} – skipping.")
    else:
        for sm_label, sm_kwargs in [
            ("unsmoothed", {}),
            (f"smooth{ROLLING_WINDOW}", {"rolling_window": ROLLING_WINDOW}),
            (f"savgol{SG_WINDOW}p{SG_POLYORDER}", {"use_savgol": True}),
        ]:
            plots.append(
                {
                    "x": df_d13c["d13c_permille"],
                    "y": df_d13c["age_ka"],
                    "xlabel": r"$\boldsymbol{\delta}^{\mathbf{13}}\mathbf{C}\ \mathbf{[‰]}$",
                    "ylabel": "Age [ka BP]",
                    "title": f"SISAL – {site_name}",
                    "filename": os.path.join(
                        OUTPUT_DIR, f"{site_slug}_d13c_age_{sm_label}"
                    ),
                    "x_ticks": d13c_ticks,
                    "show_mis": True,
                    **sm_kwargs,
                }
            )

    print(f"\n  Generating {len(plots)} plots for {site_name} …")
    for cfg in plots:
        print(f"    → {os.path.basename(cfg['filename'])}")
        create_plot(
            x_values=cfg["x"],
            y_values=cfg["y"],
            xlabel=cfg["xlabel"],
            ylabel=cfg["ylabel"],
            title_text=cfg["title"],
            output_filename=cfg["filename"],
            y_major_interval=AGE_MAJOR_TICK_INTERVAL,
            y_minor_interval=AGE_MINOR_TICK_INTERVAL,
            x_ticks=cfg.get("x_ticks"),
            show_mis=cfg.get("show_mis", False),
            rolling_window=cfg.get("rolling_window"),
            use_savgol=cfg.get("use_savgol", False),
        )



def main():
    from datetime import datetime

    report_path = os.path.join(REPORT_DIR, "report.txt")
    tee = Tee(report_path)

    print("=" * 60)
    print("SISAL Speleothem – Plot Generator")
    print("=" * 60)

    # ── Configure SISAL input files ───────────────────────────────────────────
    # Adjust paths if needed (relative to the script directory)
    SISAL_FILES = [
        {
            "path": "v_data_144_botuvera.csv",
            "slug": "144_botuvera",
            "d18o_ticks": [-6, -5, -4, -3, -2, -1],
            "d13c_ticks": [-10, -9, -8, -7, -6, -5, -4, -3],
        },
        {
            "path": "v_data_145_corchia.csv",
            "slug": "145_corchia",
            "d18o_ticks": [-6, -5, -4, -3],
            "d13c_ticks": [-3, -2, -1, 0, 1, 2, 3, 4],
        },
        {
            "path": "v_data_140_sanbao.csv",
            "slug": "140_sanbao",
            "d18o_ticks": [-10, -9, -8, -7, -6, -5, -4],
            "d13c_ticks": None,  # no d13C data in SISAL for Sanbao
        },
        {
            "path": "v_data_275_buracagloriosa.csv",
            "slug": "275_buracagloriosa",
            "d18o_ticks": [-6, -5, -4, -3, -2, -1, 0],
            "d13c_ticks": [-10, -8, -6, -4, -2, 0],
        },
    ]

    total_plots = 0

    for cfg in SISAL_FILES:
        print(f"\n{'─' * 60}")
        print(f"Loading: {cfg['path']}")
        print("─" * 60)

        filepath = cfg["path"]
        if not os.path.isabs(filepath):
            filepath = os.path.join(SCRIPT_DIR, filepath)

        if not os.path.exists(filepath):
            print(f"  ⚠  File not found: {filepath} – skipping.")
            continue

        df = load_sisal_csv(filepath)
        site_name = df["site_name"].iloc[0]

        generate_cave_plots(
            df=df,
            site_name=site_name,
            site_slug=cfg["slug"],
            d18o_ticks=cfg.get("d18o_ticks"),
            d13c_ticks=cfg.get("d13c_ticks"),
        )
        total_plots += 6

    print("\n" + "=" * 60)
    print(f"Done! Plots saved to '{OUTPUT_DIR}/'")
    print(f"Total: {total_plots} plots")
    print("=" * 60)
    print(f"Report saved: {report_path}")
    tee.close()


if __name__ == "__main__":
    main()
