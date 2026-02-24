# Datei: plot_epica_from_tab.py
import os
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
    print("⚠  rdflib nicht installiert – RDF-Export übersprungen. (pip install rdflib)")

# Arbeitsverzeichnis auf Ordner des Skripts setzen
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Output-Ordner erstellen
OUTPUT_DIR = "plots"
RDF_DIR = "rdf"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RDF_DIR, exist_ok=True)

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

# Glättung
ROLLING_WINDOW = 11  # Rolling Median: Fenstergröße in Datenpunkten (~10 ka bei CH4)
SG_WINDOW = 11  # Savitzky-Golay: Fensterbreite (ungerade, ~10 ka bei CH4)
SG_POLYORDER = 2  # Savitzky-Golay: Polynomgrad (2 = glatt, klassisch)
LINE_COLOR_FADED = "#aaaaaa"  # Originallinie im geglätteten Plot
LINE_WIDTH_SMOOTH = 1.5  # Geglättete Linie etwas dicker   # MIS-Label-Größe
LABEL_PAD = 12

# ──────────────────────────────────────────────
# MIS-Intervalle (Grenzen in ka BP, Quelle: LR04 Lisiecki & Raymo 2005)
# Format: (age_top_ka, age_bottom_ka, label, farbe)
# Warmzeiten (ungerade MIS) = hellblau, Kaltzeiten (gerade MIS) = kein Hintergrund
# ──────────────────────────────────────────────
MIS_COLOR_WARM = "#fddbc7"  # Rot/Orange        – volles Interglazial
MIS_COLOR_INTERSTADIAL = "#fef0e6"  # blasses Rötlich   – Interstadial (MIS 3)
MIS_COLOR_COLD = "#d6e8f7"  # Blau              – Glazial

# MIS-Typ:
#   "warm"       = volles Interglazial   → Rot/Orange, durchgehend
#   "inter"      = Interstadial          → blasses Rötlich, durchgehend (MIS 3)
#   "cold"       = Glazial               → Blau, durchgehend
#   "warm_nodata"= Interglazial, keine CH4-Messdaten → Rot/Orange, gestrichelter Rand
#   "cold_nodata"= Glazial, keine CH4-Messdaten      → Blau, gestrichelter Rand
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
    """Gibt die Anzahl der Kommentar-/Headerzeilen zurück (alles vor der Daten-Headerzeile)."""
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
        3: Gas age [ka BP]  (EDC2 timescale)  ← wir nehmen EDC2 (konsistent mit altem CSV)
        4: CH4 [ppbv]
        5: CH4 std dev [±]
    """
    # Alle Zeilen nach dem /* ... */ Block überspringen
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
            # Erste Zeile nach Block = Spaltenüberschriften → überspringen
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

    print(f"  CH4 geladen: {len(df)} Datenpunkte")
    print(f"  Tiefe: {df['depth_m'].min():.1f} – {df['depth_m'].max():.1f} m")
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

    print(f"  d18O geladen: {len(df)} Datenpunkte")
    print(f"  Tiefe: {df['depth_m'].min():.1f} – {df['depth_m'].max():.1f} m")
    print(f"  Age (ka BP): {df['age_ka'].min():.1f} – {df['age_ka'].max():.1f}")
    print(f"  d18O: {df['d18o'].min():.4f} – {df['d18o'].max():.4f} ‰")

    return df[["depth_m", "age_ka", "d18o"]]


# ──────────────────────────────────────────────
# Plot-Funktion (generisch für beide Achsentypen)
# ──────────────────────────────────────────────


def draw_mis_bands(ax, y_min_ka, y_max_ka):
    """
    Zeichnet MIS-Farbstreifen auf der Y-Achse (ka BP).

    Typen:
      "warm"        → Rot/Orange, durchgehend (volles Interglazial)
      "inter"       → blasses Rötlich, durchgehend (Interstadial, z.B. MIS 3)
      "cold"        → Blau, durchgehend (Glazial)
      "warm_nodata" → Rot/Orange, gestrichelter Rahmen (keine Messdaten)
      "cold_nodata" → Blau, gestrichelter Rahmen (keine Messdaten)
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
    Erstellt einen standardisierten EPICA-Plot.

    x_values      : pd.Series  – die auf der X-Achse dargestellte Messgröße
    y_values      : pd.Series  – die auf der Y-Achse dargestellte Tiefe/Zeit
    xlabel        : str        – X-Achsen-Label (LaTeX ok)
    ylabel        : str        – Y-Achsen-Label
    title_text    : str        – Titel über dem Plot
    output_filename: str       – vollständiger Pfad ohne Extension
    y_major_interval: float   – Haupttick-Abstand Y
    y_minor_interval: float   – Nebentick-Abstand Y
    x_ticks       : list|None – manuelle X-Ticks
    x_padding     : float     – relativer X-Puffer (falls keine manuellen Ticks)
    invert_y      : bool       – Y-Achse invertieren (Tiefe nimmt nach unten zu)
    show_mis      : bool       – MIS-Bänder und Labels einzeichnen (nur für Age-Plots)
    gap_line      : tuple|None – (x1, y1, x2, y2) Gestrichelte Verbindungslinie für Datenlücken
    rolling_window: int|None   – Fenstergröße Rolling Median (None = keine Glättung)
    use_savgol    : bool       – Savitzky-Golay Filter statt Rolling Median
                                 (SG_WINDOW, SG_POLYORDER aus Konfiguration)
                                 Bei True: Original grau, geglättet schwarz
    """
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    ax = fig.add_subplot(111)

    # Y-Achse zuerst setzen (vor MIS-Bändern)
    y_min, y_max = y_values.min(), y_values.max()
    if invert_y:
        ax.set_ylim(y_max, y_min)
    else:
        ax.set_ylim(y_min, y_max)
    ax.margins(y=0)

    # MIS-Bänder im Hintergrund (zorder=0)
    if show_mis:
        draw_mis_bands(ax, y_min_ka=y_min, y_max_ka=y_max)

    if use_savgol:
        # Original grau im Hintergrund
        ax.plot(
            x_values, y_values, linewidth=LINE_WIDTH, color=LINE_COLOR_FADED, zorder=2
        )
        # Savitzky-Golay geglättet schwarz im Vordergrund
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
        # Rolling Median geglättet schwarz im Vordergrund
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

    # Gestrichelte Verbindungslinie für Datenlücken
    # gap_line = (x1, y1, x2, y2): verbindet letzten Punkt vor mit erstem Punkt nach der Lücke
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

    # Glättungs-Untertitel
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

    print(f"  ✓ Gespeichert: {jpg_path}")
    print(f"  ✓ Gespeichert: {svg_path}")


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
#   CRMsci    – CRM-Erweiterung für naturwiss. Beobachtungen
#   QUDT      – Einheiten (ppbv, ‰, m, ka BP)
#   Dublin Core – Metadaten (Titel, Autoren, Lizenz)
# ============================================================================


def build_epica_rdf(df_ch4: pd.DataFrame, df_d18o: pd.DataFrame) -> "Graph":
    """
    Erstellt einen RDF-Graph mit allen EPICA-Daten.

    Modell (je Datenpunkt):
      epica:Obs_CH4_{i}  a  sosa:Observation, crmsci:S4_Observation ;
          sosa:hasFeatureOfInterest  epica:EpicaDomeC_IceCore ;
          sosa:observedProperty      epica:CH4Concentration ;
          sosa:madeBySensor          epica:GasChromatograph ;
          sosa:resultTime            <age als xsd:decimal, ka BP> ;
          sosa:hasSimpleResult       <CH4-Wert als qudt:PPB> ;
          epica:atDepth              <Tiefe in m> ;
          epica:smoothedValue_median <Rolling-Median-Wert> ;
          epica:smoothedValue_savgol <SG-Wert> ;
          epica:smoothingWindow      11 ;
          epica:smoothingPolyorder   2 ;
          prov:wasDerivedFrom        <PANGAEA DOI> ;
          crm:P7_took_place_at       epica:EpicaDomeC_Site .

    Parameters
    ----------
    df_ch4  : DataFrame mit Spalten depth_m, age_edc2_ka, ch4
    df_d18o : DataFrame mit Spalten depth_m, age_ka, d18o

    Returns
    -------
    rdflib.Graph
    """
    g = Graph()

    # ── Namespaces ────────────────────────────────────────────────────────
    EPICA = Namespace("https://purl.org/epica/")
    SOSA = Namespace("http://www.w3.org/ns/sosa/")
    SSN = Namespace("http://www.w3.org/ns/ssn/")
    GEO = Namespace("http://www.opengis.net/ont/geosparql#")
    SF = Namespace("http://www.opengis.net/ont/sf#")
    QUDT = Namespace("http://qudt.org/schema/qudt/")
    UNIT = Namespace("http://qudt.org/vocab/unit/")
    CRM = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
    CRMSCI = Namespace("http://www.ics.forth.gr/isl/CRMsci/")
    DCT = DCTERMS

    g.bind("epica", EPICA)
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
    dataset = EPICA["EPICA_DomeC_Dataset"]
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
    # dcat:Catalog fasst alle Datasets zusammen (Einstiegspunkt für Linked Data)
    catalog = EPICA["EPICA_DomeC_Catalog"]
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
    ds_ch4 = EPICA["EPICA_DomeC_CH4_Dataset"]
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

    ds_d18o = EPICA["EPICA_DomeC_d18O_Dataset"]
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

    # Observations den jeweiligen Datasets zuordnen
    # (wird später beim Loop gesetzt via epica:ch4Dataset / epica:d18oDataset)
    # Referenzen für spätere Verlinkung im Graph speichern
    g.__epica_catalog__ = catalog
    g.__epica_ds_ch4__ = ds_ch4
    g.__epica_ds_d18o__ = ds_d18o

    # ── Standort: EPICA Dome C (GeoSPARQL + CIDOC-CRM) ───────────────────
    site = EPICA["EpicaDomeC_Site"]
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

    geom = EPICA["EpicaDomeC_Geometry"]
    g.add((geom, RDF.type, URIRef(str(SF) + "Point")))
    g.add(
        (geom, GEO["asWKT"], Literal("POINT(123.35 -75.1)", datatype=GEO["wktLiteral"]))
    )
    g.add((site, GEO["hasGeometry"], geom))

    # ── Eiskern: Probe (SOSA Sample + CIDOC-CRM E22_Human-Made_Object) ───
    core = EPICA["EpicaDomeC_IceCore"]
    g.add((core, RDF.type, SOSA["Sample"]))
    g.add((core, RDF.type, CRM["E22_Human-Made_Object"]))  # Bohrkern als Artefakt
    g.add((core, RDFS.label, Literal("EPICA Dome C Ice Core", lang="en")))
    g.add((core, SOSA["isSampleOf"], site))
    g.add((core, CRM["P53_has_former_or_current_location"], site))
    g.add((core, CRM["P2_has_type"], Literal("Ice Core", lang="en")))

    # ── Feldkampagne (CIDOC-CRM E7_Activity + CRMsci S1_Matter_Removal) ─
    campaign = EPICA["EPICA_DomeCampaign_1996_2004"]
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
    prop_ch4 = EPICA["CH4Concentration"]
    g.add((prop_ch4, RDF.type, SOSA["ObservableProperty"]))
    g.add((prop_ch4, RDF.type, CRMSCI["S9_Property_Type"]))
    g.add((prop_ch4, RDFS.label, Literal("Methane concentration (CH₄)", lang="en")))
    g.add((prop_ch4, QUDT["unit"], UNIT["PPB"]))

    prop_d18o = EPICA["Delta18O"]
    g.add((prop_d18o, RDF.type, SOSA["ObservableProperty"]))
    g.add((prop_d18o, RDF.type, CRMSCI["S9_Property_Type"]))
    g.add(
        (prop_d18o, RDFS.label, Literal("Stable water isotope ratio (δ¹⁸O)", lang="en"))
    )
    g.add((prop_d18o, QUDT["unit"], UNIT["PERMILLE"]))

    # ── Chronologien (als Named Individuals dokumentiert) ─────────────────
    chron_edc2 = EPICA["EDC2_Chronology"]
    g.add((chron_edc2, RDF.type, CRMSCI["S6_Data_Evaluation"]))
    g.add(
        (
            chron_edc2,
            RDFS.label,
            Literal("EDC2 ice core chronology (Schwander et al. 2001)", lang="en"),
        )
    )

    chron_aicc = EPICA["AICC2023_Chronology"]
    g.add((chron_aicc, RDF.type, CRMSCI["S6_Data_Evaluation"]))
    g.add(
        (
            chron_aicc,
            RDFS.label,
            Literal("AICC2023 ice core chronology (Bouchet et al. 2023)", lang="en"),
        )
    )

    # ── Glättungs-Parameter als Named Individuals ─────────────────────────
    smooth_median = EPICA[f"RollingMedian_w{ROLLING_WINDOW}"]
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
            EPICA["windowSize"],
            Literal(ROLLING_WINDOW, datatype=XSD.integer),
        )
    )
    g.add(
        (smooth_median, DCT.references, URIRef("https://doi.org/10.1145/1968.1969"))
    )  # Tukey 1977

    smooth_sg = EPICA[f"SavitzkyGolay_w{SG_WINDOW}_p{SG_POLYORDER}"]
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
    g.add((smooth_sg, EPICA["windowSize"], Literal(SG_WINDOW, datatype=XSD.integer)))
    g.add((smooth_sg, EPICA["polyOrder"], Literal(SG_POLYORDER, datatype=XSD.integer)))
    g.add(
        (smooth_sg, DCT.references, URIRef("https://doi.org/10.1021/ac60214a047"))
    )  # Savitzky & Golay 1964

    # ── CH4-Observationen ────────────────────────────────────────────────
    print("  Schreibe CH4-Observationen …")
    df_ch4_valid = df_ch4.dropna(subset=["ch4", "age_edc2_ka", "depth_m"]).reset_index(
        drop=True
    )

    # Glättungswerte vorausberechnen
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
        obs = EPICA[f"Obs_CH4_{i:04d}"]
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
                EPICA["atDepth_m"],
                Literal(round(float(row["depth_m"]), 2), datatype=XSD.decimal),
            )
        )
        g.add((obs, EPICA["ageChronology"], chron_edc2))
        g.add((obs, QUDT["unit"], UNIT["PPB"]))
        # Geglättete Werte mit Tag zur Methode
        g.add(
            (
                obs,
                EPICA["smoothedValue_rollingMedian"],
                Literal(round(float(ch4_smooth_median[i]), 2), datatype=XSD.decimal),
            )
        )
        g.add(
            (
                obs,
                EPICA["smoothedValue_savgol"],
                Literal(round(float(ch4_smooth_sg[i]), 2), datatype=XSD.decimal),
            )
        )
        g.add((obs, EPICA["smoothingMethod_median"], smooth_median))
        g.add((obs, EPICA["smoothingMethod_savgol"], smooth_sg))
        g.add((obs, PROV.wasDerivedFrom, src_ch4))
        g.add((obs, CRM["P7_took_place_at"], site))
        g.add((dataset, EPICA["hasObservation"], obs))
        g.add((ds_ch4, DCAT["record"], obs))

    # ── d18O-Observationen ───────────────────────────────────────────────
    print("  Schreibe δ¹⁸O-Observationen …")
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
        obs = EPICA[f"Obs_d18O_{i:04d}"]
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
                EPICA["atDepth_m"],
                Literal(round(float(row["depth_m"]), 2), datatype=XSD.decimal),
            )
        )
        g.add((obs, EPICA["ageChronology"], chron_aicc))
        g.add((obs, QUDT["unit"], UNIT["PERMILLE"]))
        g.add(
            (
                obs,
                EPICA["smoothedValue_rollingMedian"],
                Literal(round(float(d18o_smooth_median[i]), 5), datatype=XSD.decimal),
            )
        )
        g.add(
            (
                obs,
                EPICA["smoothedValue_savgol"],
                Literal(round(float(d18o_smooth_sg[i]), 5), datatype=XSD.decimal),
            )
        )
        g.add((obs, EPICA["smoothingMethod_median"], smooth_median))
        g.add((obs, EPICA["smoothingMethod_savgol"], smooth_sg))
        g.add((obs, PROV.wasDerivedFrom, src_d18o))
        g.add((obs, CRM["P7_took_place_at"], site))
        g.add((dataset, EPICA["hasObservation"], obs))
        g.add((ds_d18o, DCAT["record"], obs))

    return g


def export_rdf(df_ch4: pd.DataFrame, df_d18o: pd.DataFrame):
    """Baut den RDF-Graph und speichert ihn als Turtle (.ttl) und JSON-LD."""
    if not RDF_AVAILABLE:
        return

    print("\n" + "─" * 60)
    print("RDF-Export …")
    print("─" * 60)

    g = build_epica_rdf(df_ch4, df_d18o)

    ttl_path = os.path.join(RDF_DIR, "epica_dome_c.ttl")
    jsonld_path = os.path.join(RDF_DIR, "epica_dome_c.jsonld")

    g.serialize(destination=ttl_path, format="turtle")
    g.serialize(destination=jsonld_path, format="json-ld", indent=2)

    triples = len(g)
    print(f"  ✓ {triples:,} Triples geschrieben")
    print(f"  ✓ Turtle:  {ttl_path}")
    print(f"  ✓ JSON-LD: {jsonld_path}")


def main():
    print("=" * 60)
    print("EPICA Dome C – Plot Generator (TAB-Dateien, komplett)")
    print("=" * 60)

    # ── Daten laden ──────────────────────────────
    print("\n[1/2] Lade CH4 Tab-Datei …")
    df_ch4 = load_ch4_tab("EDC_CH4.tab")

    print("\n[2/2] Lade d18O Tab-Datei …")
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
            # Gestrichelte Verbindungslinie über Datenlücke MIS 8-10 (243–374 ka)
            # x=CH4-Wert, y=Age; Grenzpunkte direkt aus den Daten
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
        # ── Geglättet: Nach Tiefe (m) ───────────────
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
        # ── Geglättet: Nach Age (ka BP) ──────────────
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
    print("Erstelle Plots …")
    print("─" * 60)

    for i, cfg in enumerate(plots, 1):
        print(f"\n[{i}/{len(plots)}] {cfg['title']} – Y: {cfg['ylabel']}")
        # Nur Zeilen mit gültigen Y-Werten (age kann NaN sein für einzelne Punkte)
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

    # RDF Export
    export_rdf(df_ch4, df_d18o)

    print("\n" + "=" * 60)
    print(f"Fertig! Alle {len(plots)} Plots wurden in '{OUTPUT_DIR}/' gespeichert.")
    print("=" * 60)


if __name__ == "__main__":
    main()
