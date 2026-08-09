# Datei: plot_epica_from_tab.py
#
# Draws the EPICA Dome C figures. Nothing else: the RDF that used to be built
# here moved to epica_rdf.py in S2, together with the three records this
# script never covered. Both read the same .tab through epica_data.py, so a
# figure and a triple can no longer disagree about what the file said.
#
# Multi-panel plates over the five records are still to come; this file is
# unchanged in what it draws.
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, FixedLocator
import matplotlib.transforms as transforms
from scipy.signal import savgol_filter

# Byte-identical SVG across runs. Two things vary otherwise: the <dc:date>
# in the SVG metadata, and the random ids matplotlib gives clip paths. The
# salt fixes the ids, metadata={"Date": None} at save time drops the date.
plt.rcParams["svg.hashsalt"] = "geo-lod"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epica_data as ed


class Tee:
    """Schreibt gleichzeitig auf stdout und in eine Datei."""

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


# Arbeitsverzeichnis auf Ordner des Skripts setzen
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Output-Ordner erstellen
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")
RDF_DIR = os.path.join(SCRIPT_DIR, "rdf")
ONTOLOGY_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "ontology")
REPORT_DIR = os.path.join(SCRIPT_DIR, "report")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RDF_DIR, exist_ok=True)
os.makedirs(ONTOLOGY_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Gemeinsame Plot-Einstellungen
# ──────────────────────────────────────────────
FIGURE_SIZE = (10, 30)
DPI = 100
LINE_COLOR = "black"
LINE_WIDTH = 1
GRID_COLOR = "#cccccc"
GRID_WIDTH = 1

# Y-Achsen-Ticks (Tiefe in m)
DEPTH_MAJOR_TICK_INTERVAL = 500  # alle 500 m ein dicker Tick
DEPTH_MINOR_TICK_INTERVAL = 100  # alle 100 m ein kleiner Tick

# Y-Achsen-Ticks (Age in ka BP)
AGE_MAJOR_TICK_INTERVAL = 100  # alle 100 ka ein dicker Tick
AGE_MINOR_TICK_INTERVAL = 20  # alle 20 ka ein kleiner Tick

FONT_SIZE_LABEL = 30
FONT_SIZE_TICK = 26
TITLE_FONTSIZE = 30
FONT_SIZE_MIS = 16

# Smoothing
ROLLING_WINDOW = 11  # Rolling median: window size in data points (~10 ka for CH4)
SG_WINDOW = 11  # Savitzky-Golay: window length (odd, ~10 ka for CH4)
SG_POLYORDER = 2  # Savitzky-Golay: polynomial order (2 = smooth, classic)
LINE_COLOR_FADED = "#aaaaaa"  # original line in smoothed plot
LINE_WIDTH_SMOOTH = 1.5  # smoothed line slightly thicker   # MIS label size
LABEL_PAD = 12

# ──────────────────────────────────────────────
# MIS-Intervalle (Grenzen in ka BP, Quelle: LR04 Lisiecki & Raymo 2005)
# Format: (age_top_ka, age_bottom_ka, label, farbe)
# Warmzeiten (ungerade MIS) = hellblau, Kaltzeiten (gerade MIS) = kein Hintergrund
# ──────────────────────────────────────────────
MIS_COLOR_WARM = "#fddbc7"  # red/orange       – full interglacial
MIS_COLOR_INTERSTADIAL = "#fef0e6"  # pale reddish  – interstadial (MIS 3)
MIS_COLOR_COLD = "#d6e8f7"  # blue             – glacial

# MIS type:
#   "warm"       = full interglacial   → red/orange, solid
#   "inter"      = interstadial        → pale reddish, solid (MIS 3)
#   "cold"       = glacial             → blue, solid
#   "warm_nodata"= interglacial, no CH4 data → red/orange, dashed border
#   "cold_nodata"= glacial, no CH4 data      → blue, dashed border
#
# Grenzen: LR04 (Lisiecki & Raymo 2005), außer:
#   - MIS 13/14-Grenze bei 527 ka (statt LR04 533 ka) → CH4-Minimum in EDC
#   - MIS 14/15-Grenze bei 545 ka (statt LR04 563 ka) → CH4-Anstieg in EDC
#   - MIS 8/9/10 (243–374 ka): keine CH4-Daten in EDC-Tab-Datei → "no_data"
MIS_INTERVALS = [
    (0, 14, "MIS 1", "warm"),
    (14, 29, "MIS 2", "cold"),
    (29, 57, "MIS 3", "inter"),  # Interstadial, kein volles Interglazial
    (57, 71, "MIS 4", "cold"),
    (71, 130, "MIS 5", "warm"),
    (130, 191, "MIS 6", "cold"),
    (191, 243, "MIS 7", "warm"),
    (243, 300, "MIS 8", "cold_nodata"),  # keine EDC CH4-Daten
    (300, 337, "MIS 9", "warm_nodata"),  # keine EDC CH4-Daten
    (337, 374, "MIS 10", "cold_nodata"),  # keine EDC CH4-Daten
    (374, 424, "MIS 11", "warm"),
    (424, 527, "MIS 12", "cold"),  # Grenze bei 527 ka (CH4-Minimum EDC)
    (527, 545, "MIS 13", "warm"),  # Grenze bei 545 ka (CH4-Anstieg EDC)
    (545, 621, "MIS 15", "warm"),  # MIS 14 durch Anpassung entfallen
    (621, 676, "MIS 16", "cold"),
    (676, 712, "MIS 17", "warm"),
    (712, 761, "MIS 18", "cold"),
    (761, 790, "MIS 19", "warm"),
    (790, 814, "MIS 20", "cold"),
]


# ──────────────────────────────────────────────
# Plot function (generic for both axis types)
# ──────────────────────────────────────────────


def draw_mis_bands(ax, y_min_ka, y_max_ka):
    """
    Draws MIS colour bands on the Y-axis (ka BP).

    Types:
      "warm"        → red/orange, solid (full interglacial)
      "inter"       → pale reddish, solid (interstadial, e.g. MIS 3)
      "cold"        → blue, solid (glacial)
      "warm_nodata" → red/orange, dashed border (no measurement data)
      "cold_nodata" → blue, dashed border (no measurement data)
    """
    mis_trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    type_config = {
        "warm": (MIS_COLOR_WARM, "#8b1a00", False),
        "inter": (MIS_COLOR_INTERSTADIAL, "#8b1a00", False),
        "cold": (MIS_COLOR_COLD, "#003f6b", False),
        "warm_nodata": (MIS_COLOR_WARM, "#8b1a00", True),
        "cold_nodata": (MIS_COLOR_COLD, "#003f6b", True),
    }

    for age_top, age_bot, label, mis_type in MIS_INTERVALS:
        y_lo = min(y_min_ka, y_max_ka)
        y_hi = max(y_min_ka, y_max_ka)
        visible_top = max(age_top, y_lo)
        visible_bot = min(age_bot, y_hi)
        if visible_top >= visible_bot:
            continue

        color, label_color, dashed = type_config.get(
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
    gap_line=None,
    rolling_window=None,
    use_savgol=False,
):
    """
    Creates a standardised EPICA plot.

    x_values       : pd.Series  – the measurement quantity shown on the X-axis
    y_values       : pd.Series  – the depth / time shown on the Y-axis
    xlabel         : str        – X-axis label (LaTeX ok)
    ylabel         : str        – Y-axis label
    title_text     : str        – title above the plot
    output_filename: str        – full path without file extension
    y_major_interval: float    – major tick spacing Y
    y_minor_interval: float    – minor tick spacing Y
    x_ticks        : list|None – manual X-tick positions
    x_padding      : float     – relative X padding (if no manual ticks)
    invert_y       : bool      – invert Y-axis (depth increases downward)
    show_mis       : bool      – draw MIS bands and labels (age plots only)
    gap_line       : tuple|None – (x1, y1, x2, y2) dashed line bridging data gaps
    rolling_window : int|None  – window size for rolling median (None = no smoothing)
    use_savgol     : bool      – use Savitzky-Golay filter instead of rolling median
                                 (SG_WINDOW, SG_POLYORDER from config)
                                 If True: original line grey, smoothed line black
    """
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    ax = fig.add_subplot(111)

    # Set Y-axis first (before MIS bands)
    y_min, y_max = y_values.min(), y_values.max()
    if invert_y:
        ax.set_ylim(y_max, y_min)
    else:
        ax.set_ylim(y_min, y_max)
    ax.margins(y=0)

    # MIS bands in background (zorder=0)
    if show_mis:
        draw_mis_bands(ax, y_min_ka=y_min, y_max_ka=y_max)

    if use_savgol:
        # Original grau im Hintergrund
        ax.plot(
            x_values, y_values, linewidth=LINE_WIDTH, color=LINE_COLOR_FADED, zorder=2
        )
        # Savitzky-Golay smoothed in black in foreground
        smooth = savgol_filter(
            x_values.values, window_length=SG_WINDOW, polyorder=SG_POLYORDER
        )
        ax.plot(
            smooth, y_values, linewidth=LINE_WIDTH_SMOOTH, color=LINE_COLOR, zorder=3
        )
    elif rolling_window is not None:
        # Original grau im Hintergrund
        ax.plot(
            x_values, y_values, linewidth=LINE_WIDTH, color=LINE_COLOR_FADED, zorder=2
        )
        # Rolling median smoothed in black in foreground
        import pandas as _pd

        smooth = (
            _pd.Series(x_values.values)
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

    # Dashed connecting line for data gaps
    # gap_line = (x1, y1, x2, y2): connects last point before gap to first point after
    if gap_line is not None:
        x1, y1, x2, y2 = gap_line
        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=LINE_WIDTH,
            color=LINE_COLOR,
            linestyle=(0, (5, 4)),
            dashes=(5, 4),
            zorder=2,
        )

    ax.yaxis.set_major_locator(MultipleLocator(y_major_interval))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor_interval))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{int(val)}"))
    ax.grid(axis="y", which="major", color=GRID_COLOR, linewidth=GRID_WIDTH)
    ax.tick_params(axis="y", which="minor", length=4, width=0.8)

    # X-Achse oben
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # X-Achsen-Grenzen
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

    # Beschriftungen
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

    # Titel oben (fett)
    ax.set_title(
        title_text,
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=8,
    )
    # Untertitel UNTERHALB des X-Achsen-Labels (negative y in figure-koordinaten)
    # Wir nutzen ax.annotate mit xycoords='axes fraction' und negativem y
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

    # Speichern
    jpg_path = output_filename + ".jpg"
    svg_path = output_filename + ".svg"
    plt.savefig(jpg_path, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
    plt.close()

    print(f"  ✓ Saved: {jpg_path}")
    print(f"  ✓ Saved: {svg_path}")



def main():
    report_path = os.path.join(REPORT_DIR, "report.txt")
    tee = Tee(report_path)

    print("=" * 60)
    print("EPICA Dome C – Plot Generator (TAB files, complete)")
    print("=" * 60)

    # ── Daten laden ──────────────────────────────
    # Über epica_data, aus data/raw/epica/. Die Spalten heissen dort für alle
    # fünf Datensätze gleich (depth_m, age_ka, value); die alten Namen bleiben
    # hier stehen, damit die Plot-Konfigurationen unverändert bleiben.
    print("\n[1/2] Loading CH4 TAB file …")
    df_ch4 = ed.load_ch4().rename(columns={"value": "ch4", "age_ka": "age_edc2_ka"})

    print("\n[2/2] Loading d18O TAB file …")
    df_d18o = ed.load_d18o().rename(columns={"value": "d18o"})

    # ── Plot-Konfigurationen ──────────────────────
    # X-Ticks für CH4 (ppbv) und d18O (‰)
    CH4_TICKS = [300, 400, 500, 600, 700, 800, 900]
    D18O_TICKS = [-0.5, 0.0, 0.5, 1.0]

    plots = [
        # ── Nach Tiefe (m) ──────────────────────────
        {
            "x": df_ch4["ch4"],
            "y": df_ch4["depth_m"],
            "xlabel": r"$\mathbf{CH}_{\mathbf{4}}\ \mathbf{[ppbv]}$",
            "ylabel": "Depth [m]",
            "title": "EPICA – CH₄",
            "filename": os.path.join(OUTPUT_DIR, "ch4_vs_depth_full"),
            "y_major": DEPTH_MAJOR_TICK_INTERVAL,
            "y_minor": DEPTH_MINOR_TICK_INTERVAL,
            "x_ticks": CH4_TICKS,
        },
        {
            "x": df_d18o["d18o"],
            "y": df_d18o["depth_m"],
            "xlabel": r"$\boldsymbol{\delta}^{\mathbf{18}}\mathbf{O}\ \mathbf{[‰]}$",
            "ylabel": "Depth [m]",
            "title": "EPICA – δ¹⁸O",
            "filename": os.path.join(OUTPUT_DIR, "d18o_vs_depth_full"),
            "y_major": DEPTH_MAJOR_TICK_INTERVAL,
            "y_minor": DEPTH_MINOR_TICK_INTERVAL,
            "x_ticks": D18O_TICKS,
        },
        # ── Nach Age (ka BP) ─────────────────────────
        {
            "x": df_ch4["ch4"],
            "y": df_ch4["age_edc2_ka"],
            "xlabel": r"$\mathbf{CH}_{\mathbf{4}}\ \mathbf{[ppbv]}$",
            "ylabel": "Age [ka BP]",
            "title": "EPICA – CH₄",
            "filename": os.path.join(OUTPUT_DIR, "ch4_vs_age_ka_full"),
            "y_major": AGE_MAJOR_TICK_INTERVAL,
            "y_minor": AGE_MINOR_TICK_INTERVAL,
            "x_ticks": CH4_TICKS,
            "show_mis": True,
            # Dashed connecting line across data gap MIS 8-10 (243–374 ka)
            # x=CH4 value, y=Age; boundary points taken directly from data
            "gap_line": (505.7, 214.19, 484.9, 391.85),
        },
        {
            "x": df_d18o["d18o"],
            "y": df_d18o["age_ka"],
            "xlabel": r"$\boldsymbol{\delta}^{\mathbf{18}}\mathbf{O}\ \mathbf{[‰]}$",
            "ylabel": "Age [ka BP]",
            "title": "EPICA – δ¹⁸O",
            "filename": os.path.join(OUTPUT_DIR, "d18o_vs_age_ka_full"),
            "y_major": AGE_MAJOR_TICK_INTERVAL,
            "y_minor": AGE_MINOR_TICK_INTERVAL,
            "x_ticks": D18O_TICKS,
            "show_mis": True,
        },
        # ── Smoothed: by depth (m) ──────────────────
        {
            "x": df_ch4["ch4"],
            "y": df_ch4["depth_m"],
            "xlabel": r"$\mathbf{CH}_{\mathbf{4}}\ \mathbf{[ppbv]}$",
            "ylabel": "Depth [m]",
            "title": "EPICA – CH₄",
            "filename": os.path.join(
                OUTPUT_DIR, f"ch4_vs_depth_full_smooth{ROLLING_WINDOW}"
            ),
            "y_major": DEPTH_MAJOR_TICK_INTERVAL,
            "y_minor": DEPTH_MINOR_TICK_INTERVAL,
            "x_ticks": CH4_TICKS,
            "rolling_window": ROLLING_WINDOW,
        },
        {
            "x": df_d18o["d18o"],
            "y": df_d18o["depth_m"],
            "xlabel": r"$\boldsymbol{\delta}^{\mathbf{18}}\mathbf{O}\ \mathbf{[‰]}$",
            "ylabel": "Depth [m]",
            "title": "EPICA – δ¹⁸O",
            "filename": os.path.join(
                OUTPUT_DIR, f"d18o_vs_depth_full_smooth{ROLLING_WINDOW}"
            ),
            "y_major": DEPTH_MAJOR_TICK_INTERVAL,
            "y_minor": DEPTH_MINOR_TICK_INTERVAL,
            "x_ticks": D18O_TICKS,
            "rolling_window": ROLLING_WINDOW,
        },
        # ── Smoothed: by age (ka BP) ─────────────────
        {
            "x": df_ch4["ch4"],
            "y": df_ch4["age_edc2_ka"],
            "xlabel": r"$\mathbf{CH}_{\mathbf{4}}\ \mathbf{[ppbv]}$",
            "ylabel": "Age [ka BP]",
            "title": "EPICA – CH₄",
            "filename": os.path.join(
                OUTPUT_DIR, f"ch4_vs_age_ka_full_smooth{ROLLING_WINDOW}"
            ),
            "y_major": AGE_MAJOR_TICK_INTERVAL,
            "y_minor": AGE_MINOR_TICK_INTERVAL,
            "x_ticks": CH4_TICKS,
            "show_mis": True,
            "gap_line": (505.7, 214.19, 484.9, 391.85),
            "rolling_window": ROLLING_WINDOW,
        },
        {
            "x": df_d18o["d18o"],
            "y": df_d18o["age_ka"],
            "xlabel": r"$\boldsymbol{\delta}^{\mathbf{18}}\mathbf{O}\ \mathbf{[‰]}$",
            "ylabel": "Age [ka BP]",
            "title": "EPICA – δ¹⁸O",
            "filename": os.path.join(
                OUTPUT_DIR, f"d18o_vs_age_ka_full_smooth{ROLLING_WINDOW}"
            ),
            "y_major": AGE_MAJOR_TICK_INTERVAL,
            "y_minor": AGE_MINOR_TICK_INTERVAL,
            "x_ticks": D18O_TICKS,
            "show_mis": True,
            "rolling_window": ROLLING_WINDOW,
        },
        # ── Savitzky-Golay: Nach Tiefe (m) ───────────
        {
            "x": df_ch4["ch4"],
            "y": df_ch4["depth_m"],
            "xlabel": r"$\mathbf{CH}_{\mathbf{4}}\ \mathbf{[ppbv]}$",
            "ylabel": "Depth [m]",
            "title": "EPICA – CH₄",
            "filename": os.path.join(
                OUTPUT_DIR, f"ch4_vs_depth_full_savgol{SG_WINDOW}p{SG_POLYORDER}"
            ),
            "y_major": DEPTH_MAJOR_TICK_INTERVAL,
            "y_minor": DEPTH_MINOR_TICK_INTERVAL,
            "x_ticks": CH4_TICKS,
            "use_savgol": True,
        },
        {
            "x": df_d18o["d18o"],
            "y": df_d18o["depth_m"],
            "xlabel": r"$\boldsymbol{\delta}^{\mathbf{18}}\mathbf{O}\ \mathbf{[‰]}$",
            "ylabel": "Depth [m]",
            "title": "EPICA – δ¹⁸O",
            "filename": os.path.join(
                OUTPUT_DIR, f"d18o_vs_depth_full_savgol{SG_WINDOW}p{SG_POLYORDER}"
            ),
            "y_major": DEPTH_MAJOR_TICK_INTERVAL,
            "y_minor": DEPTH_MINOR_TICK_INTERVAL,
            "x_ticks": D18O_TICKS,
            "use_savgol": True,
        },
        # ── Savitzky-Golay: Nach Age (ka BP) ─────────
        {
            "x": df_ch4["ch4"],
            "y": df_ch4["age_edc2_ka"],
            "xlabel": r"$\mathbf{CH}_{\mathbf{4}}\ \mathbf{[ppbv]}$",
            "ylabel": "Age [ka BP]",
            "title": "EPICA – CH₄",
            "filename": os.path.join(
                OUTPUT_DIR, f"ch4_vs_age_ka_full_savgol{SG_WINDOW}p{SG_POLYORDER}"
            ),
            "y_major": AGE_MAJOR_TICK_INTERVAL,
            "y_minor": AGE_MINOR_TICK_INTERVAL,
            "x_ticks": CH4_TICKS,
            "show_mis": True,
            "gap_line": (505.7, 214.19, 484.9, 391.85),
            "use_savgol": True,
        },
        {
            "x": df_d18o["d18o"],
            "y": df_d18o["age_ka"],
            "xlabel": r"$\boldsymbol{\delta}^{\mathbf{18}}\mathbf{O}\ \mathbf{[‰]}$",
            "ylabel": "Age [ka BP]",
            "title": "EPICA – δ¹⁸O",
            "filename": os.path.join(
                OUTPUT_DIR, f"d18o_vs_age_ka_full_savgol{SG_WINDOW}p{SG_POLYORDER}"
            ),
            "y_major": AGE_MAJOR_TICK_INTERVAL,
            "y_minor": AGE_MINOR_TICK_INTERVAL,
            "x_ticks": D18O_TICKS,
            "show_mis": True,
            "use_savgol": True,
        },
    ]

    print("\n" + "─" * 60)
    print("Generating plots …")
    print("─" * 60)

    for i, cfg in enumerate(plots, 1):
        print(f"\n[{i}/{len(plots)}] {cfg['title']} – Y: {cfg['ylabel']}")
        # Only rows with valid Y values (age can be NaN for individual points)
        mask = cfg["y"].notna() & cfg["x"].notna()
        create_plot(
            x_values=cfg["x"][mask],
            y_values=cfg["y"][mask],
            xlabel=cfg["xlabel"],
            ylabel=cfg["ylabel"],
            title_text=cfg["title"],
            output_filename=cfg["filename"],
            y_major_interval=cfg["y_major"],
            y_minor_interval=cfg["y_minor"],
            x_ticks=cfg.get("x_ticks"),
            show_mis=cfg.get("show_mis", False),
            gap_line=cfg.get("gap_line", None),
            rolling_window=cfg.get("rolling_window", None),
            use_savgol=cfg.get("use_savgol", False),
        )

    print("\n" + "=" * 60)
    print(f"Done! All {len(plots)} plots saved to '{OUTPUT_DIR}/'.")
    print(f"Report saved: {report_path}")
    print("=" * 60)
    tee.close()


if __name__ == "__main__":
    main()
