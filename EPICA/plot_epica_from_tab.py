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
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ontology")
)
import epica_data as ed
import numpy as np
import epica_plates
import geo_lod_figures as st
from geo_lod_captions import CaptionFile


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
# Achsen-Overrides je Datensatz
# ──────────────────────────────────────────────
# Voreinstellung sind Ticks aus den Daten (epica_style.nice_ticks). Wer eine
# Achse von Hand setzen will, trägt sie hier ein - genau dafür ist das Dict
# da, damit niemand wieder eine Tick-Liste in den Plot-Aufruf schreibt und
# damit unbemerkt Messwerte abschneidet.
#
#   "log"      : logarithmische Wertachse
#   "ticks"    : feste Tick-Positionen; die Grenzen umschliessen trotzdem
#                immer die Daten, damit nichts wegfallen kann
#   "target"   : ungefähre Anzahl Intervalle für die automatische Teilung
AXIS_OVERRIDES: dict[str, dict] = {
    # Dust spannt Faktor 560; linear wären neun Zehntel der Achse leer.
    "dust": {"log": True},
}

# ──────────────────────────────────────────────
# MIS-Bänder
# ──────────────────────────────────────────────
# Grenzen und Warm/Kalt kommen aus dist/mis_stages.csv, also aus derselben
# Quelle wie die Zuweisungen im Graphen (Railsback et al. 2015 als Leitschema).
# Vorher stand hier eine eigene Liste nach LR04 mit zwei von Hand an das
# CH4-Signal angepassten Übergängen und ohne MIS 14. Die Abbildungen ändern
# sich dadurch sichtbar; das ist der Beschluss vom 2026-08-09.
MIS_COLOR_WARM = "#fddbc7"
MIS_COLOR_COLD = "#d6e8f7"
MIS_LABEL_COLOR_WARM = "#8b1a00"
MIS_LABEL_COLOR_COLD = "#003f6b"


def draw_mis_bands(ax, y_min_ka, y_max_ka, covered=None, horizontal=False):
    """Zeichnet die MIS-Bänder auf der Altersachse.

    covered  : optionale Folge von Altern der tatsächlichen Messpunkte. Ist sie
               gegeben, bekommen Stadien ohne einen einzigen Messwert eine
               gestrichelte Umrandung statt einer Füllung - so ist eine
               Datenlücke als Lücke erkennbar und nicht als flache Kurve.
               Ersetzt die früher für CH4 hartcodierten "nodata"-Einträge.
    horizontal : True, wenn das Alter auf der X-Achse liegt (Zeilen-Tafeln).
    """
    stages = ed.read_mis_stages()
    y_lo, y_hi = min(y_min_ka, y_max_ka), max(y_min_ka, y_max_ka)
    ages = sorted(covered) if covered is not None else None

    if horizontal:
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    else:
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

        span = ax.axvspan if horizontal else ax.axhspan
        if has_data:
            span(top, bot, facecolor=color, alpha=1.0, zorder=0)
        else:
            span(
                top,
                bot,
                facecolor=color,
                alpha=0.35,
                edgecolor=label_color,
                linestyle=(0, (4, 3)),
                linewidth=1.2,
                zorder=0,
            )

        middle = (visible_top + visible_bot) / 2.0
        if horizontal:
            ax.text(
                middle, 0.985, st["label"], transform=trans, ha="center",
                va="top", fontsize=FONT_SIZE_MIS, fontweight="bold",
                color=label_color, rotation=90, zorder=2,
            )
        else:
            ax.text(
                0.99, middle, st["label"], transform=trans, ha="right",
                va="center", fontsize=FONT_SIZE_MIS, fontweight="bold",
                color=label_color, zorder=2,
            )


#: Ab dieser Lücke zwischen zwei Messungen gilt der Verlauf als unterbrochen.
#: Derselbe Wert wie ed.MAX_INTERPOLATION_GAP_KA, und mit Absicht: eine Strecke,
#: die in der Abbildung als Lücke erscheint, ist genau die Strecke, über die der
#: Generator keine Stadiengrenze interpoliert.
GAP_THRESHOLD_KA = ed.MAX_INTERPOLATION_GAP_KA

FIGURE_LICENSE = "CC BY 4.0, Florian Thiery"

SMOOTHING_TEXT = {
    "": "The measured series is shown unsmoothed.",
    f"_smooth{ROLLING_WINDOW}": (
        f"A rolling median over {ROLLING_WINDOW} points is drawn in black over "
        f"the measured series in grey."
    ),
    f"_savgol{SG_WINDOW}p{SG_POLYORDER}": (
        f"A Savitzky-Golay filter over {SG_WINDOW} points at polynomial order "
        f"{SG_POLYORDER} is drawn in black over the measured series in grey."
    ),
}


def caption_for(meta, df, axis, suffix):
    """Bildunterschrift einer Einzelabbildung, aus dem was gezeichnet wurde."""
    n = len(df)
    if axis["key"] == "age_ka":
        span = (
            f"ages from {df['age_ka'].min():.1f} to {df['age_ka'].max():.1f} ka "
            f"on the {meta['trs']} scale"
        )
        bands = (
            "Marine Isotope Stage bands follow Railsback et al. (2015); a "
            "hatched band marks a stage without a measurement in this record."
        )
    else:
        span = (
            f"depths from {df['depth_m'].min():.1f} to {df['depth_m'].max():.1f} m"
        )
        bands = (
            f"Marine Isotope Stage bands are the boundaries of Railsback et al. "
            f"(2015) carried into depth by linear interpolation on the "
            f"{meta['trs']} scale; a stage whose boundary falls inside a data "
            f"gap carries no band."
        )
    return (
        f"{meta['label']} from the EPICA Dome C ice core, {n} measurements, "
        f"{span}. {SMOOTHING_TEXT[suffix]} {bands}"
    )


def _gap_edge_depth(df, age_ka, side):
    """Tiefe der letzten Messung vor bzw. der ersten nach der Lücke, in die
    *age_ka* fällt. Liefert None, wenn das Alter ausserhalb der Messreihe liegt.

    ``side="older"`` heisst: die gesuchte Kante ist die ältere Grenze eines
    Stadiums, das Band endet also an der letzten Messung *vor* der Lücke.
    """
    pairs = sorted(zip(df["age_ka"].tolist(), df["depth_m"].tolist()))
    for (a0, d0), (a1, d1) in zip(pairs, pairs[1:]):
        if a0 <= age_ka <= a1:
            return d0 if side == "older" else d1
    return None


def mis_depth_bands_for(dataset_id, df):
    """MIS-Bänder in Tiefe für einen Datensatz.

    Jede Stadiengrenze ist als Alter publiziert. Eine Tiefe dafür entsteht nur
    durch Interpolation im Tiefen-Alters-Modell dieses Datensatzes - dasselbe
    Verfahren, das der Generator für `geolod:MISBoundaryDepth` benutzt, und
    aus derselben Funktion, damit Abbildung und Graph nicht auseinanderlaufen
    können.

    Drei Fälle, seit 2026-08-09 unterschieden:

    * Beide Grenzen interpolierbar - das Band steht ausgefüllt.
    * Nur eine Grenze interpolierbar, weil die andere in eine Datenlücke fällt
      - das Band reicht von der bekannten Kante bis an den Rand der Messreihe
      und ist schraffiert. Vorher entfiel es ganz; bei CH4 verschwanden so
      MIS 7 und MIS 11, obwohl von beiden je eine Kante bekannt ist.
    * Keine der beiden - das Stadium liegt vollständig in der Lücke, und es
      gibt keine Tiefe, der es zuzuordnen wäre. Bei CH4 sind das MIS 8 bis 10.

    Dazu ein neutrales Band über die Lücke selbst. Ohne das steht dort nur
    weisse Fläche, und weiss ist in dieser Abbildung sonst nichts.
    """
    bands = []
    top_of_core = float(df["depth_m"].min())
    youngest = float(df["age_ka"].min())

    for st_ in ed.read_mis_stages():
        d_top = ed.interpolate_depth(df, st_["end"]) if st_["end"] > 0 else None
        d_bot = ed.interpolate_depth(df, st_["begin"])

        if d_top is None and st_["end"] <= youngest:
            # Jüngstes Stadium: das Band beginnt am obersten Messpunkt.
            d_top = top_of_core

        partial = False
        if d_top is None and d_bot is not None:
            d_top = _gap_edge_depth(df, st_["end"], "younger")
            partial = True
        elif d_bot is None and d_top is not None:
            d_bot = _gap_edge_depth(df, st_["begin"], "older")
            partial = True

        if d_top is None or d_bot is None or d_bot <= d_top:
            continue
        bands.append((d_top, d_bot, st_["label"], st_["mode"] == "warm", partial))

    return bands


def gap_bands_for(df):
    """Die Datenlücken selbst, als neutrale Bänder in Tiefe."""
    pairs = sorted(zip(df["age_ka"].tolist(), df["depth_m"].tolist()))
    return [
        (d0, d1)
        for (a0, d0), (a1, d1) in zip(pairs, pairs[1:])
        if a1 - a0 > GAP_THRESHOLD_KA
    ]


def draw_mis_depth_bands(ax, bands, gaps=()):
    """Zeichnet die vorberechneten Tiefen-Bänder."""
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    for d_top, d_bot in gaps:
        ax.axhspan(d_top, d_bot, facecolor="#f2f2f2", edgecolor="#999999",
                   linestyle=(0, (4, 3)), linewidth=1.0, zorder=0)
        ax.text(
            0.99, (d_top + d_bot) / 2.0, "no samples", transform=trans,
            ha="right", va="center", fontsize=FONT_SIZE_MIS,
            fontstyle="italic", color="#777777", zorder=2,
        )

    for d_top, d_bot, label, warm, partial in bands:
        color = MIS_COLOR_WARM if warm else MIS_COLOR_COLD
        label_color = MIS_LABEL_COLOR_WARM if warm else MIS_LABEL_COLOR_COLD
        if partial:
            # Schraffiert: das Stadium reicht über den Rand der Messreihe
            # hinaus, wo seine Tiefe unbekannt ist.
            ax.axhspan(d_top, d_bot, facecolor=color, alpha=0.45,
                       edgecolor=label_color, linestyle=(0, (4, 3)),
                       linewidth=1.0, zorder=0)
        else:
            ax.axhspan(d_top, d_bot, facecolor=color, zorder=0)
        ax.text(
            0.99, (d_top + d_bot) / 2.0, label, transform=trans, ha="right",
            va="center", fontsize=FONT_SIZE_MIS, fontweight="bold",
            color=label_color, zorder=2,
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
    x_target=6,
    log_x=False,
    x_padding=0.05,
    invert_y=True,
    show_mis=False,
    mis_depth_bands=None,
    mis_depth_gaps=None,
    age_values=None,
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
    age_values     : pd.Series  – Alter der Messpunkte; nur zum Finden der
                                 Unterbrechungen, auch bei Tiefendarstellungen
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
        # y_values sind hier die Alter - damit weiss die Bänderfunktion,
        # welche Stadien im Datensatz überhaupt Messwerte haben.
        draw_mis_bands(ax, y_min_ka=y_min, y_max_ka=y_max, covered=y_values)
    elif mis_depth_bands:
        draw_mis_depth_bands(ax, mis_depth_bands, mis_depth_gaps or ())

    # Die Kurve, mit Lücken als Lücken. Die Konvention ist aus wdttest-sisal
    # übernommen: durchgezogen innerhalb eines Laufs, gestrichelt über die
    # Unterbrechung, und die letzte Probe davor wie die erste danach geringelt.
    # Ohne sie zieht matplotlib eine gerade Linie über 178 ka ohne Daten, die
    # aussieht wie ein gemessener flacher Verlauf.
    #
    # Die Unterbrechungen werden immer über die Alter gesucht, auch in den
    # Tiefendarstellungen: eine Lücke ist eine Lücke in der Beprobung, und ob
    # sie in Metern gross wirkt, hängt nur an der Kompression des Eises.
    breaks = st.find_breaks(np.asarray(age_values), GAP_THRESHOLD_KA)
    values = np.asarray(x_values, dtype=float)
    positions = np.asarray(y_values, dtype=float)

    if use_savgol or rolling_window is not None:
        # Rohwerte blass im Hintergrund, geglättet darüber.
        # Auch die Rohkurve bekommt Ringe und Strichelung. Der Ring markiert
        # die letzte gemessene Probe vor der Unterbrechung - das ist eine
        # Aussage über die Messreihe, nicht über die Glättung, und sie gilt
        # für beide Kurven.
        st.draw_profile(ax, values, positions, breaks,
                        colour=LINE_COLOR_FADED, linewidth=LINE_WIDTH,
                        zorder=2)
        # Laufweise geglättet, nicht über die Unterbrechung hinweg.
        if use_savgol:
            smooth = st.smooth_by_run(values, breaks, "savgol", SG_WINDOW,
                                      SG_POLYORDER)
        else:
            smooth = st.smooth_by_run(values, breaks, "median", rolling_window)
        st.draw_profile(ax, smooth, positions, breaks,
                        colour=LINE_COLOR, linewidth=LINE_WIDTH_SMOOTH,
                        zorder=3)
    else:
        st.draw_profile(ax, values, positions, breaks,
                        colour=LINE_COLOR, linewidth=LINE_WIDTH, zorder=2)

    ax.yaxis.set_major_locator(MultipleLocator(y_major_interval))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor_interval))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{int(val)}"))
    ax.grid(axis="y", which="major", color=GRID_COLOR, linewidth=GRID_WIDTH)
    ax.tick_params(axis="y", which="minor", length=4, width=0.8)

    # X-Achse oben
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # X-Achse: Ticks und Grenzen aus den Daten, nicht aus einer Liste im Code.
    # Die Grenzen umschliessen den Wertebereich immer - die frühere feste Liste
    # endete bei d18O auf 1.0, wo die Daten bis 1.457 reichen, und schnitt
    # damit 82 von 1378 Messwerten aus der Abbildung.
    x_min, x_max = float(x_values.min()), float(x_values.max())

    if log_x:
        ticks, (lo, hi) = st.log_ticks(x_min, x_max)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda val, pos: f"{val:g}")
        )
    else:
        if x_ticks is not None:
            ticks = list(x_ticks)
            lo = min(min(ticks), x_min)
            hi = max(max(ticks), x_max)
            pad = (hi - lo) * 0.04
            lo, hi = lo - pad, hi + pad
            step = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0
            _, _, decimals = st.nice_ticks(lo, hi)
            if step == int(step):
                decimals = 0
        else:
            ticks, (lo, hi), decimals = st.nice_ticks(x_min, x_max, target=x_target)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda val, pos: st.format_tick(val, decimals))
        )

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

    # Speichern - SVG über ein binär geöffnetes Handle, sonst schreibt Python
    # auf Windows CRLF, während .gitattributes LF ablegt.
    st.save_figure(fig, output_filename, dpi=DPI)
    plt.close(fig)



def main():
    report_path = os.path.join(REPORT_DIR, "report.txt")
    tee = Tee(report_path)

    print("=" * 60)
    print("EPICA Dome C - figure generator (five records)")
    print("=" * 60)

    # Alle fünf Datensätze, beide Achsen, drei Glättungsvarianten. Vorher
    # standen hier zwölf von Hand geschriebene Konfigurationen für CH4 und
    # d18O; die drei übrigen Datensätze kamen in keiner Einzelabbildung vor,
    # obwohl sie seit S2 im Graphen stehen.
    frames = {k: df for k, df in ed.load_all()}

    # Eine Caption-Datei je Strang, gesammelt in epica_plates.CAPTIONS: die
    # Einzelabbildungen tragen ihre Unterschrift dort ein, die Tafeln ebenso,
    # geschrieben wird einmal am Ende. Zwei getrennte Sammler hätten sich beim
    # Schreiben gegenseitig überschrieben.
    captions = epica_plates.CAPTIONS

    variants = [
        ("", {}),
        (f"_smooth{ROLLING_WINDOW}", {"rolling_window": ROLLING_WINDOW}),
        (f"_savgol{SG_WINDOW}p{SG_POLYORDER}", {"use_savgol": True}),
    ]

    n_total = len(ed.DATASET_ORDER) * 2 * len(variants)
    n = 0

    print("\n" + "-" * 60)
    print(f"Generating {n_total} single figures ...")
    print("-" * 60)

    for dataset_id in ed.DATASET_ORDER:
        meta = ed.DATASETS[dataset_id]
        df = frames[dataset_id]
        override = AXIS_OVERRIDES.get(dataset_id, {})
        xlabel = f"{meta['label']} [{meta['unit_label']}]"
        title = f"EPICA - {meta['label']}"

        depth_bands = mis_depth_bands_for(dataset_id, df)
        depth_gaps = gap_bands_for(df)

        axes = [
            {
                "key": "depth",
                "y": df["depth_m"],
                "ylabel": "Depth [m]",
                "y_major": DEPTH_MAJOR_TICK_INTERVAL,
                "y_minor": DEPTH_MINOR_TICK_INTERVAL,
                "show_mis": False,
                "mis_depth_bands": depth_bands,
                "mis_depth_gaps": depth_gaps,
            },
            {
                "key": "age_ka",
                "y": df["age_ka"],
                # Nach A4: "Age [ka]", ohne BP oder b2k. Der Bezugspunkt steht
                # am Chronologieknoten im Graphen.
                "ylabel": "Age [ka]",
                "y_major": AGE_MAJOR_TICK_INTERVAL,
                "y_minor": AGE_MINOR_TICK_INTERVAL,
                "show_mis": True,
                "mis_depth_bands": None,
                "mis_depth_gaps": None,
            },
        ]

        for axis in axes:
            for suffix, opts in variants:
                n += 1
                name = f"{meta['short'].lower()}_vs_{axis['key']}_full{suffix}"
                print(f"\n[{n}/{n_total}] {title} - Y: {axis['ylabel']}")
                create_plot(
                    x_values=df["value"],
                    y_values=axis["y"],
                    xlabel=xlabel,
                    ylabel=axis["ylabel"],
                    title_text=f"{title}  ({meta['trs']})"
                    if axis["key"] == "age_ka"
                    else title,
                    output_filename=os.path.join(OUTPUT_DIR, name),
                    y_major_interval=axis["y_major"],
                    y_minor_interval=axis["y_minor"],
                    x_ticks=override.get("ticks"),
                    x_target=override.get("target", 6),
                    log_x=override.get("log", False),
                    show_mis=axis["show_mis"],
                    mis_depth_bands=axis["mis_depth_bands"],
                    mis_depth_gaps=axis["mis_depth_gaps"],
                    age_values=df["age_ka"],
                    **opts,
                )
                captions.add(
                    name,
                    caption=caption_for(meta, df, axis, suffix),
                    license=FIGURE_LICENSE,
                    sources=[meta["doi"]],
                )

    # Die mehrteiligen Tafeln kommen neben den Einzeldateien, nicht statt
    # ihrer: die Einzelabbildung zeigt eine Kurve gross, die Tafel den
    # Vergleich, den eine Einzelabbildung nicht leisten kann.
    epica_plates.build_all()
    captions.write()

    print("\n" + "=" * 60)
    print(f"Done! {n_total} single figures plus the plates saved to '{OUTPUT_DIR}/'.")
    print(f"Report saved: {report_path}")
    print("=" * 60)
    tee.close()


if __name__ == "__main__":
    main()
