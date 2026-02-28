# Datei: plot_epica_from_tab.py
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, FixedLocator
import matplotlib.transforms as transforms
from scipy.signal import savgol_filter
from datetime import datetime

try:
    from rdflib import Graph, Namespace, URIRef, Literal, BNode
    from rdflib.namespace import RDF, RDFS, OWL, XSD, DCTERMS, PROV

    DCAT = Namespace("http://www.w3.org/ns/dcat#")
    RDF_AVAILABLE = True
except ImportError:
    RDF_AVAILABLE = False
    print("⚠  rdflib not installed – RDF export skipped. (pip install rdflib)")

# geo_lod_utils: shared namespaces, GeoSPARQL helpers, core ontology, Mermaid
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "ontology"))
try:
    from geo_lod_utils import (
        NS as GEO_LOD_NS,
        get_graph,
        wkt_point,
        add_geo_site,
        add_feature_collection,
        write_geo_lod_core,
        write_mermaid as write_geo_lod_mermaid,
    )

    GEO_LOD_UTILS_AVAILABLE = True
except ImportError:
    GEO_LOD_UTILS_AVAILABLE = False
    print("⚠  geo_lod_utils not found – falling back to local namespace definitions.")


class Tee:
    """Schreibt gleichzeitig auf stdout und in eine Datei."""

    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")
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
# TAB-Dateien einlesen
# ──────────────────────────────────────────────


def skip_header_lines(filepath):
    """Returns the number of comment/header lines (everything before the data header line)."""
    with open(filepath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.startswith("/*") or line.startswith(" ") or line.strip() == "":
                continue
            # Erste echte Headerzeile (Spaltenname-Zeile) → danach kommen Daten
            return i
    return 0


def load_ch4_tab(filepath):
    """
    Liest EDC_CH4.tab ein.
    Spalten (Tab-getrennt):
        0: Depth ice/snow [m]
        1: Depth ref [m]
        2: Gas age [ka BP]  (EDC1 timescale)
        3: Gas age [ka BP]  (EDC2 timescale)  ← we use EDC2 (consistent with original CSV)
        4: CH4 [ppbv]
        5: CH4 std dev [±]
    """
    # Skip all lines before the /* ... */ header block
    rows = []
    header_passed = False
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.startswith("*/"):
                header_passed = True
                continue
            if not header_passed:
                continue
            stripped = line.strip()
            if stripped == "":
                continue
            # First line after block = column headers → skip
            if stripped.startswith("Depth ice/snow"):
                continue
            rows.append(stripped.split("\t"))

    df = pd.DataFrame(rows)
    df.columns = [
        "depth_m",
        "depth_ref",
        "age_edc1_ka",
        "age_edc2_ka",
        "ch4",
        "ch4_std",
    ]

    df["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce")
    df["age_edc2_ka"] = pd.to_numeric(df["age_edc2_ka"], errors="coerce")
    df["ch4"] = pd.to_numeric(df["ch4"], errors="coerce")

    df = df.dropna(subset=["depth_m", "ch4"])
    df = df.sort_values("depth_m").reset_index(drop=True)

    print(f"  CH4 loaded: {len(df)} data points")
    print(f"  Depth: {df['depth_m'].min():.1f} – {df['depth_m'].max():.1f} m")
    print(
        f"  Age (EDC2, ka BP): {df['age_edc2_ka'].min():.1f} – {df['age_edc2_ka'].max():.1f}"
    )
    print(f"  CH4: {df['ch4'].min():.1f} – {df['ch4'].max():.1f} ppbv")

    return df[["depth_m", "age_edc2_ka", "ch4"]]


def load_d18o_tab(filepath):
    """
    Liest EPICA_Dome_C_d18O.tab ein.
    Spalten (Tab-getrennt):
        0: Depth ice/snow [m]
        1: Gas age [ka BP]
        2: δ18O-O2 [‰]
    """
    rows = []
    header_passed = False
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.startswith("*/"):
                header_passed = True
                continue
            if not header_passed:
                continue
            stripped = line.strip()
            if stripped == "":
                continue
            if stripped.startswith("Depth ice/snow"):
                continue
            rows.append(stripped.split("\t"))

    df = pd.DataFrame(rows)
    df.columns = ["depth_m", "age_ka", "d18o"]

    df["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce")
    df["age_ka"] = pd.to_numeric(df["age_ka"], errors="coerce")
    df["d18o"] = pd.to_numeric(df["d18o"], errors="coerce")

    df = df.dropna(subset=["depth_m", "d18o"])
    df = df.sort_values("depth_m").reset_index(drop=True)

    print(f"  d18O loaded: {len(df)} data points")
    print(f"  Depth: {df['depth_m'].min():.1f} – {df['depth_m'].max():.1f} m")
    print(f"  Age (ka BP): {df['age_ka'].min():.1f} – {df['age_ka'].max():.1f}")
    print(f"  d18O: {df['d18o'].min():.4f} – {df['d18o'].max():.4f} ‰")

    return df[["depth_m", "age_ka", "d18o"]]


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
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: {jpg_path}")
    print(f"  ✓ Saved: {svg_path}")


# ──────────────────────────────────────────────
# Hauptprogramm
# ──────────────────────────────────────────────

# ============================================================================
# RDF / LINKED DATA EXPORT
# ============================================================================
# Ontologien:
#   SOSA/SSN  – W3C Sensor & Observation (Messdaten)
#   PROV-O    – W3C Provenance (Datenherkunft, PANGAEA-DOI)
#   GeoSPARQL – OGC Geometrie / Standort
#   CIDOC-CRM – ISO 21127, Kulturerbe-Ereignisse (Feldkampagne, Probenahme)
#   CRMsci    – CRM extension for natural-science observations
#   QUDT      – units (ppbv, ‰, m, ka BP)
#   Dublin Core – metadata (title, authors, licence)
# ============================================================================


def build_epica_rdf(df_ch4: pd.DataFrame, df_d18o: pd.DataFrame) -> "Graph":
    """
    Erstellt einen RDF-Graph mit allen EPICA-Daten.

    Modell (je Datenpunkt):
      geolod:Obs_CH4_{i}  a  sosa:Observation, crmsci:S4_Observation ;
          sosa:hasFeatureOfInterest  geolod:EpicaDomeC_IceCore ;
          sosa:observedProperty      geolod:CH4Concentration ;
          sosa:madeBySensor          geolod:GasChromatograph ;
          sosa:resultTime            <age als xsd:decimal, ka BP> ;
          sosa:hasSimpleResult       <CH4-Wert als qudt:PPB> ;
          geolod:atDepth              <Tiefe in m> ;
          geolod:smoothedValue_median <Rolling-Median-Wert> ;
          geolod:smoothedValue_savgol <SG-Wert> ;
          geolod:smoothingWindow      11 ;
          geolod:smoothingPolyorder   2 ;
          prov:wasDerivedFrom        <PANGAEA DOI> ;
          crm:P7_took_place_at       geolod:EpicaDomeC_Site .

    Parameters
    ----------
    df_ch4  : DataFrame mit Spalten depth_m, age_edc2_ka, ch4
    df_d18o : DataFrame mit Spalten depth_m, age_ka, d18o

    Returns
    -------
    rdflib.Graph
    """
    # ── Graph + Namespaces (via geo_lod_utils — single source of truth) ────
    if GEO_LOD_UTILS_AVAILABLE:
        g = get_graph()
    else:
        g = Graph()

    GEOLOD = Namespace("http://w3id.org/geo-lod/")
    SOSA = Namespace("http://www.w3.org/ns/sosa/")
    SSN = Namespace("http://www.w3.org/ns/ssn/")
    GEO = Namespace("http://www.opengis.net/ont/geosparql#")
    SF = Namespace("http://www.opengis.net/ont/sf#")
    QUDT = Namespace("http://qudt.org/schema/qudt/")
    UNIT = Namespace("http://qudt.org/vocab/unit/")
    CRM = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
    CRMSCI = Namespace("http://www.ics.forth.gr/isl/CRMsci/")
    DCT = DCTERMS

    if not GEO_LOD_UTILS_AVAILABLE:
        # Fallback: bind manually if geo_lod_utils is unavailable
        g.bind("geolod", GEOLOD)
        g.bind("sosa", SOSA)
        g.bind("ssn", SSN)
        g.bind("geo", GEO)
        g.bind("sf", SF)
        g.bind("qudt", QUDT)
        g.bind("unit", UNIT)
        g.bind("crm", CRM)
        g.bind("crmsci", CRMSCI)
        g.bind("dct", DCT)
        g.bind("dcat", DCAT)
        g.bind("prov", PROV)
        g.bind("xsd", XSD)

    # ── Metadaten: Datensatz-Beschreibung ─────────────────────────────────
    dataset = GEOLOD["EPICA_DomeC_Dataset"]
    g.add((dataset, RDF.type, DCAT["Dataset"]))
    g.add(
        (
            dataset,
            DCT.title,
            Literal("EPICA Dome C Ice Core – CH₄ and δ¹⁸O Records", lang="en"),
        )
    )
    g.add(
        (
            dataset,
            DCT.description,
            Literal(
                "Methane (CH4) and stable water isotope (δ18O) measurements from the EPICA Dome C ice core, "
                "East Antarctica, covering the last ~800,000 years (ca. 8 glacial cycles).",
                lang="en",
            ),
        )
    )
    g.add(
        (dataset, DCT.license, URIRef("https://creativecommons.org/licenses/by/3.0/"))
    )
    g.add(
        (
            dataset,
            DCT.publisher,
            Literal("PANGAEA – Data Publisher for Earth & Environmental Science"),
        )
    )
    g.add(
        (
            dataset,
            DCT.created,
            Literal(datetime.now().strftime("%Y-%m-%d"), datatype=XSD.date),
        )
    )

    # CH4-Quelle
    src_ch4 = URIRef("https://doi.org/10.1594/PANGAEA.472484")
    g.add((src_ch4, RDF.type, DCT.BibliographicResource))
    g.add(
        (
            src_ch4,
            DCT.title,
            Literal("EPICA Dome C Methane Record (Spahni & Stocker 2006)", lang="en"),
        )
    )
    g.add((src_ch4, DCT.creator, Literal("Spahni, R.; Stocker, T.F.")))
    g.add((src_ch4, DCT.date, Literal("2006", datatype=XSD.gYear)))
    g.add((dataset, DCT.source, src_ch4))

    # d18O-Quelle
    src_d18o = URIRef("https://doi.org/10.1594/PANGAEA.961024")
    g.add((src_d18o, RDF.type, DCT.BibliographicResource))
    g.add(
        (
            src_d18o,
            DCT.title,
            Literal(
                "EPICA Dome C δ18O Record on AICC2023 (Bouchet et al. 2023)", lang="en"
            ),
        )
    )
    g.add((src_d18o, DCT.creator, Literal("Bouchet, M. et al.")))
    g.add((src_d18o, DCT.date, Literal("2023", datatype=XSD.gYear)))
    g.add((dataset, DCT.source, src_d18o))

    # ── DCAT Catalog ─────────────────────────────────────────────────────
    # dcat:Catalog groups all datasets (entry point for Linked Data)
    catalog = GEOLOD["EPICA_DomeC_Catalog"]
    g.add((catalog, RDF.type, DCAT["Catalog"]))
    g.add(
        (
            catalog,
            RDFS.label,
            Literal("EPICA Dome C Ice Core – Linked Data Catalogue", lang="en"),
        )
    )
    g.add(
        (
            catalog,
            DCT.title,
            Literal("EPICA Dome C Ice Core – Linked Data Catalogue", lang="en"),
        )
    )
    g.add(
        (
            catalog,
            DCT.description,
            Literal(
                "DCAT catalogue aggregating palaeoclimate observation datasets from the EPICA Dome C "
                "ice core, East Antarctica. Includes CH₄ and δ¹⁸O records with raw and smoothed values, "
                "full provenance, site geometry and chronology metadata.",
                lang="en",
            ),
        )
    )
    g.add(
        (
            catalog,
            DCT.publisher,
            Literal("PANGAEA – Data Publisher for Earth & Environmental Science"),
        )
    )
    g.add(
        (catalog, DCT.license, URIRef("https://creativecommons.org/licenses/by/3.0/"))
    )
    g.add(
        (
            catalog,
            DCT.created,
            Literal(datetime.now().strftime("%Y-%m-%d"), datatype=XSD.date),
        )
    )
    g.add((catalog, DCAT["dataset"], dataset))

    # CH4 und d18O als separate dcat:Dataset innerhalb des Katalogs
    ds_ch4 = GEOLOD["EPICA_DomeC_CH4_Dataset"]
    g.add((ds_ch4, RDF.type, DCAT["Dataset"]))
    g.add(
        (ds_ch4, DCT.title, Literal("EPICA Dome C – Methane (CH₄) Record", lang="en"))
    )
    g.add(
        (
            ds_ch4,
            DCT.description,
            Literal(
                "CH₄ concentration measurements from the EPICA Dome C ice core "
                "on the EDC2 chronology (0–649 ka BP, 736 data points).",
                lang="en",
            ),
        )
    )
    g.add((ds_ch4, DCT.source, src_ch4))
    g.add((ds_ch4, DCT.license, URIRef("https://creativecommons.org/licenses/by/3.0/")))
    g.add((ds_ch4, DCAT["distribution"], src_ch4))
    g.add((catalog, DCAT["dataset"], ds_ch4))

    ds_d18o = GEOLOD["EPICA_DomeC_d18O_Dataset"]
    g.add((ds_d18o, RDF.type, DCAT["Dataset"]))
    g.add(
        (
            ds_d18o,
            DCT.title,
            Literal("EPICA Dome C – Stable Water Isotope (δ¹⁸O) Record", lang="en"),
        )
    )
    g.add(
        (
            ds_d18o,
            DCT.description,
            Literal(
                "δ¹⁸O measurements from the EPICA Dome C ice core "
                "on the AICC2023 chronology (102–806 ka BP, 1378 data points).",
                lang="en",
            ),
        )
    )
    g.add((ds_d18o, DCT.source, src_d18o))
    g.add(
        (ds_d18o, DCT.license, URIRef("https://creativecommons.org/licenses/by/3.0/"))
    )
    g.add((ds_d18o, DCAT["distribution"], src_d18o))
    g.add((catalog, DCAT["dataset"], ds_d18o))

    # Observations will be linked to their respective datasets in the loop
    # (set later via geolod:ch4Dataset / geolod:d18oDataset)
    # Store references for later graph linking
    g.__epica_catalog__ = catalog
    g.__epica_ds_ch4__ = ds_ch4
    g.__epica_ds_d18o__ = ds_d18o

    # ── Standort: EPICA Dome C (GeoSPARQL 1.1 / CI pattern + CIDOC-CRM) ────
    site = GEOLOD["EpicaDomeC_Site"]
    g.add((site, RDF.type, GEO["Feature"]))  # geo:Feature (GeoSPARQL)
    g.add((site, RDF.type, GEOLOD["DrillingSite"]))  # domain class
    g.add((site, RDF.type, CRM["E53_Place"]))
    g.add((site, RDF.type, CRM["E27_Site"]))
    g.add((site, RDFS.label, Literal("EPICA Dome C, East Antarctica", lang="en")))
    g.add(
        (
            site,
            CRM["P87_is_identified_by"],
            Literal("75°06'S, 123°21'E", datatype=XSD.string),
        )
    )

    # Geometry — sf:Point only (sf:Point subClassOf geo:Geometry via OWL entailment)
    # WKT with explicit CRS prefix (GeoSPARQL 1.1 / CI_full.py pattern)
    geom = GEOLOD["EpicaDomeC_Geometry"]
    g.add((geom, RDF.type, SF["Point"]))
    g.add(
        (
            geom,
            GEO["asWKT"],
            Literal(
                "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(123.35 -75.1)",
                datatype=GEO["wktLiteral"],
            ),
        )
    )
    g.add((site, GEO["hasGeometry"], geom))

    # FeatureCollection (GeoSPARQL 1.1 — enables Linked Data viewers / QGIS)
    collection = GEOLOD["EPICA_DrillingSite_Collection"]
    g.add((collection, RDF.type, GEO["FeatureCollection"]))
    g.add(
        (
            collection,
            RDFS.label,
            Literal("EPICA Dome C Drilling Site Collection", lang="en"),
        )
    )
    g.add((collection, RDFS.member, site))

    # ── Global Palaeoclimate Sites Collection ──
    # (combined collection across all datasets — EPICA, SISAL, etc.)
    global_collection = GEOLOD["AllPalaeoclimateSites_Collection"]
    g.add((global_collection, RDF.type, GEO["FeatureCollection"]))
    g.add(
        (
            global_collection,
            RDFS.label,
            Literal("All Palaeoclimate Sites Collection", lang="en"),
        )
    )
    g.add(
        (
            global_collection,
            RDFS.comment,
            Literal(
                "Combined collection of all palaeoclimate sampling locations (ice cores, cave sites, etc.)",
                lang="en",
            ),
        )
    )
    g.add((global_collection, RDFS.member, site))

    # ── Eiskern: Probe (SOSA Sample + CIDOC-CRM E22_Human-Made_Object) ───
    core = GEOLOD["EpicaDomeC_IceCore"]
    g.add((core, RDF.type, SOSA["Sample"]))
    g.add((core, RDF.type, CRM["E22_Human-Made_Object"]))  # Bohrkern als Artefakt
    g.add((core, RDFS.label, Literal("EPICA Dome C Ice Core", lang="en")))
    g.add((core, SOSA["isSampleOf"], site))
    g.add((core, CRM["P53_has_former_or_current_location"], site))
    g.add((core, CRM["P2_has_type"], Literal("Ice Core", lang="en")))

    # ── Feldkampagne (CIDOC-CRM E7_Activity + CRMsci S1_Matter_Removal) ─
    campaign = GEOLOD["EPICA_DomeCampaign_1996_2004"]
    g.add((campaign, RDF.type, CRM["E7_Activity"]))
    g.add((campaign, RDF.type, CRMSCI["S1_Matter_Removal"]))
    g.add(
        (
            campaign,
            RDFS.label,
            Literal("EPICA Dome C drilling campaign 1996–2004", lang="en"),
        )
    )
    g.add((campaign, CRM["P7_took_place_at"], site))
    g.add(
        (campaign, CRM["P4_has_time-span"], Literal("1996/2004", datatype=XSD.string))
    )
    g.add((campaign, CRMSCI["O1_removed"], core))

    # ── Observed Properties ───────────────────────────────────────────────
    prop_ch4 = GEOLOD["CH4Concentration"]
    g.add((prop_ch4, RDF.type, SOSA["ObservableProperty"]))
    g.add((prop_ch4, RDF.type, CRMSCI["S9_Property_Type"]))
    g.add((prop_ch4, RDFS.label, Literal("Methane concentration (CH₄)", lang="en")))
    g.add((prop_ch4, QUDT["unit"], UNIT["PPB"]))

    prop_d18o = GEOLOD["Delta18O"]
    g.add((prop_d18o, RDF.type, SOSA["ObservableProperty"]))
    g.add((prop_d18o, RDF.type, CRMSCI["S9_Property_Type"]))
    g.add(
        (prop_d18o, RDFS.label, Literal("Stable water isotope ratio (δ¹⁸O)", lang="en"))
    )
    g.add((prop_d18o, QUDT["unit"], UNIT["PERMILLE"]))

    # ── Chronologien (als Named Individuals dokumentiert) ─────────────────
    chron_edc2 = GEOLOD["EDC2_Chronology"]
    g.add((chron_edc2, RDF.type, CRMSCI["S6_Data_Evaluation"]))
    g.add(
        (
            chron_edc2,
            RDFS.label,
            Literal("EDC2 ice core chronology (Schwander et al. 2001)", lang="en"),
        )
    )

    chron_aicc = GEOLOD["AICC2023_Chronology"]
    g.add((chron_aicc, RDF.type, CRMSCI["S6_Data_Evaluation"]))
    g.add(
        (
            chron_aicc,
            RDFS.label,
            Literal("AICC2023 ice core chronology (Bouchet et al. 2023)", lang="en"),
        )
    )

    # ── Smoothing parameters as named individuals ──────────────────────────
    smooth_median = GEOLOD[f"RollingMedian_w{ROLLING_WINDOW}"]
    g.add((smooth_median, RDF.type, CRMSCI["S6_Data_Evaluation"]))
    g.add(
        (
            smooth_median,
            RDFS.label,
            Literal(f"Rolling median filter, window={ROLLING_WINDOW} pts", lang="en"),
        )
    )
    g.add(
        (
            smooth_median,
            GEOLOD["windowSize"],
            Literal(ROLLING_WINDOW, datatype=XSD.integer),
        )
    )
    g.add(
        (smooth_median, DCT.references, URIRef("https://doi.org/10.1145/1968.1969"))
    )  # Tukey 1977

    smooth_sg = GEOLOD[f"SavitzkyGolay_w{SG_WINDOW}_p{SG_POLYORDER}"]
    g.add((smooth_sg, RDF.type, CRMSCI["S6_Data_Evaluation"]))
    g.add(
        (
            smooth_sg,
            RDFS.label,
            Literal(
                f"Savitzky-Golay filter, window={SG_WINDOW} pts, polyorder={SG_POLYORDER}",
                lang="en",
            ),
        )
    )
    g.add((smooth_sg, GEOLOD["windowSize"], Literal(SG_WINDOW, datatype=XSD.integer)))
    g.add((smooth_sg, GEOLOD["polyOrder"], Literal(SG_POLYORDER, datatype=XSD.integer)))
    g.add(
        (smooth_sg, DCT.references, URIRef("https://doi.org/10.1021/ac60214a047"))
    )  # Savitzky & Golay 1964

    # ── Measurement Types ─────────────────────────────────────────────────────
    mtype_ch4 = GEOLOD["MeasurementType_CH4"]
    g.add((mtype_ch4, RDF.type, GEOLOD["MeasurementType"]))
    g.add((mtype_ch4, RDFS.label, Literal("Methane (CH₄) measurement", lang="en")))
    g.add(
        (
            mtype_ch4,
            RDFS.comment,
            Literal(
                "Indicates that this observation is a CH₄ concentration measurement "
                "from trapped air bubbles in the ice core.",
                lang="en",
            ),
        )
    )

    mtype_d18o = GEOLOD["MeasurementType_d18O"]
    g.add((mtype_d18o, RDF.type, GEOLOD["MeasurementType"]))
    g.add(
        (
            mtype_d18o,
            RDFS.label,
            Literal("δ¹⁸O stable water isotope measurement", lang="en"),
        )
    )
    g.add(
        (
            mtype_d18o,
            RDFS.comment,
            Literal(
                "Indicates that this observation is a stable water isotope ratio "
                "(δ¹⁸O) measurement from the ice matrix.",
                lang="en",
            ),
        )
    )

    # ── CH4-Observationen ────────────────────────────────────────────────
    print("  Writing CH4 observations …")
    df_ch4_valid = df_ch4.dropna(subset=["ch4", "age_edc2_ka", "depth_m"]).reset_index(
        drop=True
    )

    # Pre-calculate smoothed values
    ch4_smooth_median = (
        pd.Series(df_ch4_valid["ch4"].values)
        .rolling(window=ROLLING_WINDOW, center=True, min_periods=1)
        .median()
        .values
    )
    ch4_smooth_sg = savgol_filter(
        df_ch4_valid["ch4"].values, window_length=SG_WINDOW, polyorder=SG_POLYORDER
    )

    for i, row in df_ch4_valid.iterrows():
        obs = GEOLOD[f"Obs_CH4_{i:04d}"]
        age_label = round(float(row["age_edc2_ka"]), 1)
        g.add(
            (
                obs,
                RDFS.label,
                Literal(f"CH₄ observation {i:04d} ({age_label} ka BP)", lang="en"),
            )
        )
        g.add((obs, GEOLOD["measurementType"], mtype_ch4))
        g.add((obs, RDF.type, SOSA["Observation"]))
        g.add((obs, RDF.type, CRMSCI["S4_Observation"]))
        g.add((obs, SOSA["hasFeatureOfInterest"], core))
        g.add((obs, SOSA["observedProperty"], prop_ch4))
        g.add(
            (
                obs,
                SOSA["hasSimpleResult"],
                Literal(round(float(row["ch4"]), 2), datatype=XSD.decimal),
            )
        )
        g.add(
            (
                obs,
                SOSA["resultTime"],
                Literal(round(float(row["age_edc2_ka"]), 4), datatype=XSD.decimal),
            )
        )
        g.add(
            (
                obs,
                GEOLOD["atDepth_m"],
                Literal(round(float(row["depth_m"]), 2), datatype=XSD.decimal),
            )
        )
        g.add((obs, GEOLOD["ageChronology"], chron_edc2))
        g.add((obs, QUDT["unit"], UNIT["PPB"]))
        # Smoothed values tagged with method
        g.add(
            (
                obs,
                GEOLOD["smoothedValue_rollingMedian"],
                Literal(round(float(ch4_smooth_median[i]), 2), datatype=XSD.decimal),
            )
        )
        g.add(
            (
                obs,
                GEOLOD["smoothedValue_savgol"],
                Literal(round(float(ch4_smooth_sg[i]), 2), datatype=XSD.decimal),
            )
        )
        g.add((obs, GEOLOD["smoothingMethod_median"], smooth_median))
        g.add((obs, GEOLOD["smoothingMethod_savgol"], smooth_sg))
        g.add((obs, PROV.wasDerivedFrom, src_ch4))
        g.add((obs, CRM["P7_took_place_at"], site))
        g.add((dataset, GEOLOD["hasObservation"], obs))
        g.add((ds_ch4, DCAT["record"], obs))

    # ── d18O-Observationen ───────────────────────────────────────────────
    print("  Writing δ¹⁸O observations …")
    df_d18o_valid = df_d18o.dropna(subset=["d18o", "age_ka", "depth_m"]).reset_index(
        drop=True
    )

    d18o_smooth_median = (
        pd.Series(df_d18o_valid["d18o"].values)
        .rolling(window=ROLLING_WINDOW, center=True, min_periods=1)
        .median()
        .values
    )
    d18o_smooth_sg = savgol_filter(
        df_d18o_valid["d18o"].values, window_length=SG_WINDOW, polyorder=SG_POLYORDER
    )

    for i, row in df_d18o_valid.iterrows():
        obs = GEOLOD[f"Obs_d18O_{i:04d}"]
        age_label_d = round(float(row["age_ka"]), 1)
        g.add(
            (
                obs,
                RDFS.label,
                Literal(f"δ¹⁸O observation {i:04d} ({age_label_d} ka BP)", lang="en"),
            )
        )
        g.add((obs, GEOLOD["measurementType"], mtype_d18o))
        g.add((obs, RDF.type, SOSA["Observation"]))
        g.add((obs, RDF.type, CRMSCI["S4_Observation"]))
        g.add((obs, SOSA["hasFeatureOfInterest"], core))
        g.add((obs, SOSA["observedProperty"], prop_d18o))
        g.add(
            (
                obs,
                SOSA["hasSimpleResult"],
                Literal(round(float(row["d18o"]), 5), datatype=XSD.decimal),
            )
        )
        g.add(
            (
                obs,
                SOSA["resultTime"],
                Literal(round(float(row["age_ka"]), 4), datatype=XSD.decimal),
            )
        )
        g.add(
            (
                obs,
                GEOLOD["atDepth_m"],
                Literal(round(float(row["depth_m"]), 2), datatype=XSD.decimal),
            )
        )
        g.add((obs, GEOLOD["ageChronology"], chron_aicc))
        g.add((obs, QUDT["unit"], UNIT["PERMILLE"]))
        g.add(
            (
                obs,
                GEOLOD["smoothedValue_rollingMedian"],
                Literal(round(float(d18o_smooth_median[i]), 5), datatype=XSD.decimal),
            )
        )
        g.add(
            (
                obs,
                GEOLOD["smoothedValue_savgol"],
                Literal(round(float(d18o_smooth_sg[i]), 5), datatype=XSD.decimal),
            )
        )
        g.add((obs, GEOLOD["smoothingMethod_median"], smooth_median))
        g.add((obs, GEOLOD["smoothingMethod_savgol"], smooth_sg))
        g.add((obs, PROV.wasDerivedFrom, src_d18o))
        g.add((obs, CRM["P7_took_place_at"], site))
        g.add((dataset, GEOLOD["hasObservation"], obs))
        g.add((ds_d18o, DCAT["record"], obs))

    return g


def export_ontology():
    """
    Writes two OWL ontology files:
      rdf/geo_lod_core.ttl     – shared core (via geo_lod_utils)
      rdf/epica_ontology.ttl   – EPICA-specific extension (imports core)
    Also writes Mermaid diagrams via geo_lod_utils.write_mermaid().
    """
    from datetime import datetime as _dt

    os.makedirs(RDF_DIR, exist_ok=True)

    # ── 1. Core ontology (geo_lod_core.ttl) ─────────────────────────────────
    if GEO_LOD_UTILS_AVAILABLE:
        write_geo_lod_core(RDF_DIR)
    else:
        print("  ⚠  geo_lod_utils not available – geo_lod_core.ttl skipped.")

    # ── 2. EPICA extension ontology (epica_ontology.ttl) ────────────────────
    epica_ttl = f"""@prefix owl:     <http://www.w3.org/2002/07/owl#> .
@prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .
@prefix dct:     <http://purl.org/dc/terms/> .
@prefix dcat:    <http://www.w3.org/ns/dcat#> .
@prefix sosa:    <http://www.w3.org/ns/sosa/> .
@prefix prov:    <http://www.w3.org/ns/prov#> .
@prefix geo:     <http://www.opengis.net/ont/geosparql#> .
@prefix sf:      <http://www.opengis.net/ont/sf#> .
@prefix qudt:    <http://qudt.org/schema/qudt/> .
@prefix unit:    <http://qudt.org/vocab/unit/> .
@prefix crm:     <http://www.cidoc-crm.org/cidoc-crm/> .
@prefix crmsci:  <http://www.ics.forth.gr/isl/CRMsci/> .
@prefix geolod:  <http://w3id.org/geo-lod/> .

# ============================================================================
# EPICA Dome C Ice Core – OWL Ontology Extension
# Imports: geo_lod_core.ttl  (shared classes / properties)
# Generated by: plot_epica_from_tab.py
# Date: {_dt.now().strftime("%Y-%m-%d")}
# Parameters: ROLLING_WINDOW={ROLLING_WINDOW}, SG_WINDOW={SG_WINDOW}, SG_POLYORDER={SG_POLYORDER}
# ============================================================================

<http://w3id.org/geo-lod/epica>
    a owl:Ontology ;
    owl:imports          <http://w3id.org/geo-lod/> ;
    rdfs:label           "EPICA Dome C Ice Core Ontology"@en ;
    dct:title            "EPICA Dome C Ice Core Ontology"@en ;
    dct:description      "OWL ontology for palaeoclimatic ice core observations from EPICA Dome C, \
East Antarctica. Covers CH4 and d18O measurements, smoothing methods, site geometry, \
drilling campaign and data provenance. Imports geo_lod_core.ttl."@en ;
    dct:license          <https://creativecommons.org/licenses/by/4.0/> ;
    dct:created          "{_dt.now().strftime("%Y-%m-%d")}"^^xsd:date ;
    owl:versionInfo      "1.0.0" .

# ============================================================================
# EPICA-SPECIFIC CLASSES  (shared classes live in geo_lod_core.ttl)
# ============================================================================

# ── Catalogue & Dataset ───────────────────────────────────────────────────────

geolod:PalaeoclimateDataCatalogue
    a owl:Class ;
    rdfs:subClassOf     dcat:Catalog ;
    rdfs:label          "Palaeoclimate Data Catalogue"@en ;
    rdfs:comment        "A DCAT catalogue aggregating one or more palaeoclimate datasets."@en .

geolod:IceCoreDataset
    a owl:Class ;
    rdfs:subClassOf     dcat:Dataset ;
    rdfs:label          "Ice Core Dataset"@en ;
    rdfs:comment        "A dataset derived from measurements on an ice core."@en .

geolod:CH4Dataset
    a owl:Class ;
    rdfs:subClassOf     geolod:IceCoreDataset ;
    rdfs:label          "Methane (CH4) Ice Core Dataset"@en ;
    rdfs:comment        "Dataset containing methane concentration observations from an ice core."@en .

geolod:Delta18ODataset
    a owl:Class ;
    rdfs:subClassOf     geolod:IceCoreDataset ;
    rdfs:label          "Stable Water Isotope (d18O) Ice Core Dataset"@en ;
    rdfs:comment        "Dataset containing stable water isotope (d18O) observations from an ice core."@en .

# ── Observation ───────────────────────────────────────────────────────────────

geolod:IceCoreObservation
    a owl:Class ;
    rdfs:subClassOf     geolod:PalaeoclimateObservation ;
    rdfs:label          "Ice Core Observation"@en ;
    rdfs:comment        "A single measurement on an ice core sample (depth, age, value)."@en ;
    rdfs:subClassOf     [
        a owl:Restriction ;
        owl:onProperty      sosa:hasFeatureOfInterest ;
        owl:someValuesFrom  geolod:IceCore
    ] ;
    rdfs:subClassOf     [
        a owl:Restriction ;
        owl:onProperty      prov:wasDerivedFrom ;
        owl:someValuesFrom  geolod:DataSource
    ] .

geolod:CH4Observation
    a owl:Class ;
    rdfs:subClassOf     geolod:IceCoreObservation ;
    rdfs:label          "CH4 Observation"@en ;
    rdfs:comment        "An observation of methane concentration (CH4) in ppbv from an ice core."@en .

geolod:Delta18OObservation
    a owl:Class ;
    rdfs:subClassOf     geolod:IceCoreObservation ;
    rdfs:label          "d18O Observation"@en ;
    rdfs:comment        "An observation of the stable water isotope ratio (d18O) from an ice core."@en .

# ── Sample ────────────────────────────────────────────────────────────────────

geolod:IceCore
    a owl:Class ;
    rdfs:subClassOf     geolod:PalaeoclimateSample ;
    rdfs:subClassOf     crm:E22_Human-Made_Object ;
    rdfs:label          "Ice Core"@en ;
    rdfs:comment        "A cylindrical ice sample extracted by drilling from a glacier or ice sheet."@en ;
    rdfs:subClassOf     [
        a owl:Restriction ;
        owl:onProperty      geolod:extractedFrom ;
        owl:someValuesFrom  geolod:DrillingSite
    ] .

# ── Location ──────────────────────────────────────────────────────────────────

geolod:DrillingSite
    a owl:Class ;
    rdfs:subClassOf     geolod:SamplingLocation ;
    rdfs:label          "Ice Core Drilling Site"@en ;
    rdfs:comment        "The geographical location where an ice core was drilled."@en .

# ── Campaign ──────────────────────────────────────────────────────────────────

geolod:DrillingCampaign
    a owl:Class ;
    rdfs:subClassOf     crm:E7_Activity ;
    rdfs:subClassOf     crmsci:S1_Matter_Removal ;
    rdfs:label          "Drilling Campaign"@en ;
    rdfs:comment        "A scientific field campaign during which an ice core was drilled."@en ;
    rdfs:subClassOf     [
        a owl:Restriction ;
        owl:onProperty      geolod:tookPlaceAt ;
        owl:someValuesFrom  geolod:DrillingSite
    ] ;
    rdfs:subClassOf     [
        a owl:Restriction ;
        owl:onProperty      geolod:removedSample ;
        owl:someValuesFrom  geolod:IceCore
    ] .

# ── Observable Properties ─────────────────────────────────────────────────────

geolod:CH4ConcentrationProperty
    a owl:Class ;
    rdfs:subClassOf     geolod:ObservableProperty ;
    rdfs:label          "Methane Concentration Property"@en ;
    rdfs:comment        "CH4 concentration in ppbv, measured from air bubbles trapped in ice."@en .

# ── Chronology ────────────────────────────────────────────────────────────────

geolod:IceCoreChronology
    a owl:Class ;
    rdfs:subClassOf     geolod:Chronology ;
    rdfs:label          "Ice Core Chronology"@en ;
    rdfs:comment        "A depth-age model for an ice core (e.g. EDC2, AICC2023)."@en .

# ============================================================================
# EPICA-SPECIFIC PROPERTIES
# ============================================================================

geolod:hasDrillingCampaign
    a owl:ObjectProperty ;
    rdfs:domain         geolod:IceCoreDataset ;
    rdfs:range          geolod:DrillingCampaign ;
    rdfs:label          "has drilling campaign"@en .

geolod:atDepth_m
    a owl:DatatypeProperty ;
    rdfs:domain         geolod:IceCoreObservation ;
    rdfs:range          xsd:decimal ;
    rdfs:label          "at depth (m)"@en ;
    qudt:unit           unit:M .

# ============================================================================
# NAMED INDIVIDUALS
# ============================================================================

geolod:AllPalaeoclimateSites_Collection
    a geo:FeatureCollection , owl:NamedIndividual ;
    rdfs:label          "All Palaeoclimate Sites Collection"@en ;
    rdfs:comment        "Combined collection of all palaeoclimate sampling locations (ice cores, cave sites, etc.)"@en .
    # Members are added dynamically from individual datasets (EPICA, SISAL, etc.)

geolod:EPICA_DrillingSite_Collection
    a geo:FeatureCollection , owl:NamedIndividual ;
    rdfs:label          "EPICA Dome C Drilling Site Collection"@en ;
    rdfs:member         geolod:EpicaDomeC_Site .

geolod:EpicaDomeC_Site
    a geolod:DrillingSite , owl:NamedIndividual ;
    rdfs:label          "EPICA Dome C, East Antarctica"@en ;
    crm:P87_is_identified_by  "75 deg 06 S, 123 deg 21 E"^^xsd:string ;
    geo:hasGeometry     geolod:EpicaDomeC_Geometry .

geolod:EpicaDomeC_Geometry
    a sf:Point , owl:NamedIndividual ;
    geo:asWKT           "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(123.35 -75.1)"^^geo:wktLiteral .

geolod:EpicaDomeC_IceCore
    a geolod:IceCore , owl:NamedIndividual ;
    rdfs:label          "EPICA Dome C Ice Core"@en ;
    geolod:extractedFrom geolod:EpicaDomeC_Site .

geolod:EPICA_DrillingCampaign_1996_2004
    a geolod:DrillingCampaign , owl:NamedIndividual ;
    rdfs:label          "EPICA Dome C drilling campaign 1996-2004"@en ;
    geolod:tookPlaceAt   geolod:EpicaDomeC_Site ;
    geolod:removedSample geolod:EpicaDomeC_IceCore ;
    crm:P4_has_time-span  "1996/2004"^^xsd:string .

geolod:CH4Concentration
    a geolod:CH4ConcentrationProperty , owl:NamedIndividual ;
    rdfs:label          "Methane concentration (CH4)"@en ;
    qudt:unit           unit:PPB .

geolod:MeasurementType_CH4
    a geolod:MeasurementType , owl:NamedIndividual ;
    rdfs:label          "CH4 concentration measurement"@en .

geolod:RollingMedian_w{ROLLING_WINDOW}
    a geolod:RollingMedianFilter , owl:NamedIndividual ;
    rdfs:label          "Rolling Median Filter (window={ROLLING_WINDOW})"@en ;
    geolod:windowSize   {ROLLING_WINDOW} .

geolod:SavitzkyGolay_w{SG_WINDOW}_p{SG_POLYORDER}
    a geolod:SavitzkyGolayFilter , owl:NamedIndividual ;
    rdfs:label          "Savitzky-Golay Filter (window={SG_WINDOW}, order={SG_POLYORDER})"@en ;
    geolod:windowSize   {SG_WINDOW} ;
    geolod:polyOrder    {SG_POLYORDER} .

geolod:EDC2_Chronology
    a geolod:IceCoreChronology , owl:NamedIndividual ;
    rdfs:label          "EDC2 Ice Core Chronology"@en .

geolod:AICC2023_Chronology
    a geolod:IceCoreChronology , owl:NamedIndividual ;
    rdfs:label          "AICC2023 Ice Core Chronology"@en .

geolod:PANGAEA_CH4_Source
    a geolod:DataSource , owl:NamedIndividual ;
    rdfs:label          "EPICA Dome C CH4 Record - PANGAEA"@en ;
    dct:title           "EPICA Dome C CH4 Record on EDC2 timescale (Monnin et al. 2004)"@en ;
    dct:creator         "Monnin, E. et al." ;
    dct:date            "2004"^^xsd:gYear ;
    owl:sameAs          <https://doi.org/10.1594/PANGAEA.472484> .

geolod:PANGAEA_d18O_Source
    a geolod:DataSource , owl:NamedIndividual ;
    rdfs:label          "EPICA Dome C d18O Record - PANGAEA"@en ;
    dct:title           "EPICA Dome C d18O Record on AICC2023 (Bouchet et al. 2023)"@en ;
    dct:creator         "Bouchet, M. et al." ;
    dct:date            "2023"^^xsd:gYear ;
    owl:sameAs          <https://doi.org/10.1594/PANGAEA.961024> .

# ============================================================================
# LABELS FOR EPICA-SPECIFIC EXTERNAL TERMS
# ============================================================================

crm:E22_Human-Made_Object
    rdfs:label    "Human-Made Object"@en ;
    rdfs:comment  "CRM E22: A human-made object (e.g. drilled ice core)."@en .

crm:E7_Activity
    rdfs:label    "Activity"@en ;
    rdfs:comment  "CRM E7: An intentional action carried out by an actor."@en .

crm:P7_took_place_at
    rdfs:label    "took place at"@en .

crm:P4_has_time-span
    rdfs:label    "has time-span"@en .

crm:P87_is_identified_by
    rdfs:label    "is identified by"@en .

crm:P2_has_type
    rdfs:label    "has type"@en .

crm:P53_has_former_or_current_location
    rdfs:label    "has former or current location"@en .

crmsci:S1_Matter_Removal
    rdfs:label    "Matter Removal"@en ;
    rdfs:comment  "CRMsci S1: A physical event where matter is removed (e.g. drilling)."@en .

crmsci:O1_removed
    rdfs:label    "removed"@en .

sosa:isSampleOf         rdfs:label "is sample of"@en .
sosa:hasFeatureOfInterest rdfs:label "has feature of interest"@en .
sosa:observedProperty   rdfs:label "observed property"@en .
sosa:hasSimpleResult    rdfs:label "has simple result"@en .
sosa:resultTime         rdfs:label "result time"@en .

dcat:Catalog            rdfs:label "Catalog"@en .
dcat:Dataset            rdfs:label "Dataset"@en .
dcat:Distribution       rdfs:label "Distribution"@en .

dct:BibliographicResource rdfs:label "Bibliographic Resource"@en .
dct:title               rdfs:label "title"@en .
dct:description         rdfs:label "description"@en .
dct:creator             rdfs:label "creator"@en .
dct:date                rdfs:label "date"@en .
dct:license             rdfs:label "license"@en .
dct:created             rdfs:label "created"@en .

qudt:unit               rdfs:label "unit"@en .
unit:PPB                rdfs:label "Parts Per Billion"@en .
unit:PERMILLE           rdfs:label "Per Mille"@en .
unit:M                  rdfs:label "Metre"@en .
"""

    owl_path = os.path.join(RDF_DIR, "epica_ontology.ttl")
    with open(owl_path, "w", encoding="utf-8") as fh:
        fh.write(epica_ttl)
    print(f"  ✓ EPICA ontology: {owl_path}")

    # ── 3. Mermaid diagrams (via geo_lod_utils) ──────────────────────────────
    if GEO_LOD_UTILS_AVAILABLE:
        write_geo_lod_mermaid(
            ONTOLOGY_DIR,
            rolling_window=ROLLING_WINDOW,
            sg_window=SG_WINDOW,
            sg_poly=SG_POLYORDER,
        )
    else:
        print("  ℹ  Mermaid skipped (geo_lod_utils not available)")

    # ── 4. Combined Sites Collection (optional — if SISAL is available) ──────
    # This creates a FeatureCollection that references both EPICA and SISAL sites.
    # The collection itself is defined in the ontology, but members are added
    # dynamically from the data TTLs (epica_dome_c.ttl, sisal_sites.ttl).
    if GEO_LOD_UTILS_AVAILABLE:
        sisal_sites_path = os.path.join(RDF_DIR, "sisal_sites.ttl")
        epica_data_path = os.path.join(RDF_DIR, "epica_dome_c.ttl")

        # Combined sites collection skipped
        print("  ℹ  Combined collection skipped (feature not implemented)")


def export_rdf(df_ch4: pd.DataFrame, df_d18o: pd.DataFrame):
    """Builds the RDF graph and serialises it as Turtle (.ttl)."""
    if not RDF_AVAILABLE:
        return

    print("\n" + "─" * 60)
    print("RDF Export …")
    print("─" * 60)

    g = build_epica_rdf(df_ch4, df_d18o)

    ttl_path = os.path.join(RDF_DIR, "epica_dome_c.ttl")

    g.serialize(destination=ttl_path, format="turtle")

    triples = len(g)
    print(f"  ✓ {triples:,} triples written")
    print(f"  ✓ Turtle:  {ttl_path}")
    export_ontology()


def main():
    report_path = os.path.join(REPORT_DIR, "report.txt")
    tee = Tee(report_path)

    print("=" * 60)
    print("EPICA Dome C – Plot Generator (TAB files, complete)")
    print("=" * 60)

    # ── Daten laden ──────────────────────────────
    print("\n[1/2] Loading CH4 TAB file …")
    df_ch4 = load_ch4_tab("EDC_CH4.tab")

    print("\n[2/2] Loading d18O TAB file …")
    df_d18o = load_d18o_tab("EPICA_Dome_C_d18O.tab")

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

    # RDF Export (data as Turtle, requires rdflib)
    export_rdf(df_ch4, df_d18o)
    # OWL Ontology (no rdflib required – always written)
    if not RDF_AVAILABLE:
        print("\n" + "─" * 60)
        print("OWL Ontology …")
        print("─" * 60)
        export_ontology()  # calls export_mermaid() internally

    print("\n" + "=" * 60)
    print(f"Done! All {len(plots)} plots saved to '{OUTPUT_DIR}/'.")
    print(f"Report saved: {report_path}")
    print("=" * 60)
    tee.close()


if __name__ == "__main__":
    main()
