# Datei: plot_sisal_from_csv.py
# Visualisierung von SISAL-Speleothem-Daten (d18O, d13C) analog zum EPICA-Script
# Basiert auf: plot_epica_from_tab.py

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter, FixedLocator
import matplotlib.transforms as transforms
from scipy.signal import savgol_filter


class Tee:
    """Writes simultaneously to stdout and a file."""

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


# Set working directory to the script's folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create output directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")
REPORT_DIR = os.path.join(SCRIPT_DIR, "report")
RDF_DIR = os.path.join(SCRIPT_DIR, "rdf")
ONTOLOGY_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "ontology")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(RDF_DIR, exist_ok=True)
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

    jpg_path = output_filename + ".jpg"
    svg_path = output_filename + ".svg"
    plt.savefig(jpg_path, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ {jpg_path}")
    print(f"  ✓ {svg_path}")


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


# ══════════════════════════════════════════════════════════════════════════════
# RDF / LOD  –  SISAL Ontologie-Erweiterung + Datenexport
# ══════════════════════════════════════════════════════════════════════════════
#
# Designprinzip: minimale Erweiterung des EPICA geo-lod Schemas.
#   Wiederverwendet:  MeasurementType, SmoothingMethod, DataSource,
#                     ObservableProperty, RollingMedianFilter, SavitzkyGolayFilter
#   Neu:              Speleothem, Cave, SpeleothemSamplingEvent,
#                     SpeleothemObservation, UThChronology,
#                     Delta13CProperty, MeasurementType_d13C
#
# PROV-O Provenienz:
#   SISALv3: Kaushal et al. 2024, ESSD 16, 1933–1963
#            https://doi.org/10.5194/essd-16-1933-2024
#            Data DOI: https://doi.org/10.5287/ora-2nanwp4rk
# ══════════════════════════════════════════════════════════════════════════════

try:
    from rdflib import Graph, Namespace, URIRef, Literal
    from rdflib.namespace import RDF, RDFS, OWL, XSD, DCTERMS, PROV

    RDF_AVAILABLE = True
except ImportError:
    RDF_AVAILABLE = False
    print("⚠  rdflib not installed – RDF export disabled. (pip install rdflib)")

# geo_lod_utils: shared namespaces, GeoSPARQL helpers, core ontology, Mermaid
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "ontology"))
try:
    from geo_lod_utils import (
        NS as GEO_LOD_NS,
        get_graph,
        wkt_point,
        add_geo_site_from_wkt,
        add_feature_collection,
        write_geo_lod_core,
        write_mermaid as write_geo_lod_mermaid,
    )

    GEO_LOD_UTILS_AVAILABLE = True
except ImportError:
    GEO_LOD_UTILS_AVAILABLE = False
    print("⚠  geo_lod_utils not found – falling back to local namespace definitions.")


SISAL_ONTOLOGY_TTL = """\
@prefix geolod:  <http://w3id.org/geo-lod/> .
@prefix sosa:    <http://www.w3.org/ns/sosa/> .
@prefix crm:     <http://www.cidoc-crm.org/cidoc-crm/> .
@prefix crmsci:  <http://www.ics.forth.gr/isl/CRMsci/> .
@prefix geo:     <http://www.opengis.net/ont/geosparql#> .
@prefix sf:      <http://www.opengis.net/ont/sf#> .
@prefix qudt:    <http://qudt.org/schema/qudt/> .
@prefix unit:    <http://qudt.org/vocab/unit/> .
@prefix prov:    <http://www.w3.org/ns/prov#> .
@prefix dct:     <http://purl.org/dc/terms/> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .

# ============================================================================
# ONTOLOGIE-HEADER
# ============================================================================

<http://w3id.org/geo-lod/sisal-ontology>
    a owl:Ontology ;
    rdfs:label   "geo-lod SISAL Ontology"@en ;
    rdfs:comment "Ontology for speleothem palaeoclimate data, aligned with the
                  SISALv3 database schema (Kaushal et al. 2024, ESSD 16,
                  1933-1963). Extends the geo-lod ice-core vocabulary with
                  cave-specific classes, properties, and site-level statistics."@en ;
    owl:imports  <http://w3id.org/geo-lod/> ;
    dct:source   <https://doi.org/10.5194/essd-16-1933-2024> ;
    dct:created  "2024"^^xsd:gYear .

# ============================================================================
# KLASSEN
# ============================================================================

# ── Sample ────────────────────────────────────────────────────────────────────
geolod:Speleothem
    a owl:Class ;
    rdfs:subClassOf geolod:PalaeoclimateSample ;
    rdfs:subClassOf crm:E22_Human-Made_Object ;
    rdfs:label   "Speleothem"@en ;
    rdfs:comment "Secondary cave carbonate precipitate (stalagmite, stalactite,
                  flowstone) used as palaeoclimate archive. Corresponds to
                  entity/persist_id in SISALv3."@en ;
    rdfs:subClassOf [
        a owl:Restriction ;
        owl:onProperty     geolod:collectedFrom ;
        owl:someValuesFrom geolod:Cave
    ] .

# ── Site ──────────────────────────────────────────────────────────────────────
geolod:Cave
    a owl:Class ;
    rdfs:subClassOf geolod:SamplingLocation ;
    rdfs:label   "Cave"@en ;
    rdfs:comment "Geographical cave site from which a speleothem was collected.
                  Corresponds to site in SISALv3 (site_id, latitude, longitude,
                  elevation)."@en .

# ── Sampling Event ────────────────────────────────────────────────────────────
geolod:SpeleothemSamplingEvent
    a owl:Class ;
    rdfs:subClassOf crm:E7_Activity ;
    rdfs:subClassOf crmsci:S1_Matter_Removal ;
    rdfs:label   "Speleothem Sampling Event"@en ;
    rdfs:comment "Scientific field event during which a speleothem was collected
                  from a cave."@en ;
    rdfs:subClassOf [
        a owl:Restriction ;
        owl:onProperty     geolod:tookPlaceAt ;
        owl:someValuesFrom geolod:Cave
    ] ;
    rdfs:subClassOf [
        a owl:Restriction ;
        owl:onProperty     geolod:removedSample ;
        owl:someValuesFrom geolod:Speleothem
    ] .

# ── Observation ───────────────────────────────────────────────────────────────
geolod:SpeleothemObservation
    a owl:Class ;
    rdfs:subClassOf geolod:PalaeoclimateObservation ;
    rdfs:label   "Speleothem Observation"@en ;
    rdfs:comment "Single stable isotope measurement on a speleothem sample,
                  characterised by depth (mm), calendar age (ka BP) and measured
                  value. Corresponds to a row in the d18O or d13C table of
                  SISALv3."@en ;
    rdfs:subClassOf [
        a owl:Restriction ;
        owl:onProperty     sosa:hasFeatureOfInterest ;
        owl:someValuesFrom geolod:Speleothem
    ] ;
    rdfs:subClassOf [
        a owl:Restriction ;
        owl:onProperty     sosa:observedProperty ;
        owl:someValuesFrom geolod:ObservableProperty
    ] ;
    rdfs:subClassOf [
        a owl:Restriction ;
        owl:onProperty     prov:wasDerivedFrom ;
        owl:someValuesFrom geolod:DataSource
    ] .

geolod:Delta18OSpeleothemObservation
    a owl:Class ;
    rdfs:subClassOf geolod:SpeleothemObservation ;
    rdfs:label   "delta-18O Speleothem Observation"@en ;
    rdfs:comment "Oxygen isotope ratio (delta-18O) in permille VPDB from a
                  speleothem. Proxy for hydroclimate, moisture source,
                  temperature."@en .

geolod:Delta13CSpeleothemObservation
    a owl:Class ;
    rdfs:subClassOf geolod:SpeleothemObservation ;
    rdfs:label   "delta-13C Speleothem Observation"@en ;
    rdfs:comment "Carbon isotope ratio (delta-13C) in permille VPDB from a
                  speleothem. Proxy for vegetation density, soil CO2, prior
                  calcite precipitation (Wong & Breecker 2015)."@en .

# ── Chronology ────────────────────────────────────────────────────────────────
geolod:UThChronology
    a owl:Class ;
    rdfs:subClassOf geolod:Chronology ;
    rdfs:label   "U-Th Chronology"@en ;
    rdfs:comment "Depth-age model for a speleothem based on uranium-thorium
                  (U-Th) radiometric dating. Corresponds to the Dating and
                  SISAL_chronology tables in SISALv3. Age-model ensembles
                  built with linear interpolation, Bchron, Bacon, copRa or
                  StalAge."@en ;
    rdfs:seeAlso <https://doi.org/10.5194/essd-16-1933-2024> .

# ── Observable Properties ─────────────────────────────────────────────────────
geolod:Delta13CProperty
    a owl:Class ;
    rdfs:subClassOf geolod:ObservableProperty ;
    rdfs:label   "delta-13C Isotope Ratio Property"@en ;
    rdfs:comment "Stable carbon isotope ratio (delta-13C) in permille VPDB.
                  Sensitive to vegetation dynamics, soil respiration, and prior
                  calcite precipitation."@en .

# ============================================================================
# OBJECT PROPERTIES
# ============================================================================

geolod:collectedFrom
    a owl:ObjectProperty ;
    rdfs:subPropertyOf sosa:isSampleOf ;
    rdfs:domain geolod:Speleothem ;
    rdfs:range  geolod:Cave ;
    rdfs:label  "collected from"@en ;
    rdfs:comment "Links a speleothem to the cave from which it was collected."@en .

geolod:ageChronologySpeleothem
    a owl:ObjectProperty ;
    rdfs:subPropertyOf geolod:ageChronology ;
    rdfs:domain geolod:SpeleothemObservation ;
    rdfs:range  geolod:UThChronology ;
    rdfs:label  "age chronology (speleothem)"@en ;
    rdfs:comment "U-Th chronology used to assign a calendar age to this
                  speleothem observation."@en .

# ============================================================================
# DATATYPE PROPERTIES
# ============================================================================

geolod:atDepth_mm
    a owl:DatatypeProperty ;
    rdfs:domain geolod:SpeleothemObservation ;
    rdfs:range  xsd:decimal ;
    rdfs:label  "at depth (mm)"@en ;
    rdfs:comment "Sampling depth in millimetres from top of speleothem.
                  Corresponds to depth_sample in SISALv3."@en ;
    qudt:unit   unit:MilliM .

geolod:entityId
    a owl:DatatypeProperty ;
    rdfs:domain geolod:Speleothem ;
    rdfs:range  xsd:integer ;
    rdfs:label  "SISAL entity ID"@en ;
    rdfs:comment "Unique dataset identifier in SISALv3 (entity_id). Multiple
                  entity_ids may share the same persist_id (same physical
                  speleothem, different time windows)."@en .

geolod:siteId
    a owl:DatatypeProperty ;
    rdfs:domain geolod:Cave ;
    rdfs:range  xsd:integer ;
    rdfs:label  "SISAL site ID"@en ;
    rdfs:comment "Unique site identifier in SISALv3 (site_id)."@en .

geolod:countD18OSamples
    a owl:DatatypeProperty ;
    rdfs:domain geolod:Cave ;
    rdfs:range  xsd:integer ;
    rdfs:label  "count of d18O samples"@en ;
    rdfs:comment "Total number of delta-18O measurements available for this
                  cave site across all entities in SISALv3
                  (column: n_d18o_samples in v_sites_all)."@en .

geolod:countD13CSamples
    a owl:DatatypeProperty ;
    rdfs:domain geolod:Cave ;
    rdfs:range  xsd:integer ;
    rdfs:label  "count of d13C samples"@en ;
    rdfs:comment "Total number of delta-13C measurements available for this
                  cave site across all entities in SISALv3
                  (column: n_d13c_samples in v_sites_all)."@en .

# ============================================================================
# INSTANZEN  –  MeasurementType & DataSource
# ============================================================================

geolod:MeasurementType_d13C
    a geolod:MeasurementType ;
    rdfs:label   "delta-13C Measurement"@en ;
    rdfs:comment "Stable carbon isotope ratio (delta-13C) in permille VPDB.
                  SISALv3 table: d13C. Proxy: vegetation, soil CO2, prior
                  calcite precipitation."@en .

geolod:SISALv3_DataSource
    a geolod:DataSource ;
    rdfs:label  "SISALv3 Database"@en ;
    rdfs:comment "Speleothem Isotopes Synthesis and AnaLysis database, version 3."@en ;
    dct:identifier
        "https://doi.org/10.5287/ora-2nanwp4rk"^^xsd:anyURI ;
    dct:bibliographicCitation
        "Kaushal, N. et al. (2024): SISALv3: a global speleothem stable isotope and trace element database. Earth Syst. Sci. Data, 16, 1933-1963. https://doi.org/10.5194/essd-16-1933-2024"@en ;
    prov:wasAttributedTo
        <https://doi.org/10.5194/essd-16-1933-2024> .

# ============================================================================
# NAMED INDIVIDUALS
# ============================================================================

geolod:AllPalaeoclimateSites_Collection
    a geo:FeatureCollection , owl:NamedIndividual ;
    rdfs:label   "All Palaeoclimate Sites Collection"@en ;
    rdfs:comment "Combined collection of all palaeoclimate sampling locations (ice cores, cave sites, etc.)"@en .
    # Members are added dynamically from individual datasets (EPICA, SISAL, etc.)

geolod:SISAL_Cave_Collection
    a geo:FeatureCollection , owl:NamedIndividual ;
    rdfs:label   "SISAL Cave Sites Collection"@en .
    # Members are added dynamically from v_sites_all.csv (305 caves)
"""


def build_sisal_rdf(
    df: "pd.DataFrame", site_name: str, site_slug: str
) -> "Graph | None":
    """
    Builds an RDF graph for one SISAL cave site.

    Structure (mirrors build_epica_rdf in the EPICA script):
      - 1 Cave instance  (site)
      - 1 Speleothem instance per entity_id
      - 1 SpeleothemObservation per measurement (d18O / d13C)
      - Smoothed values (rolling median + Savitzky-Golay) as datatype properties
      - PROV-O: prov:wasDerivedFrom → geolod:SISALv3_DataSource
    """
    if not RDF_AVAILABLE:
        return None

    # ── Graph + Namespaces (via geo_lod_utils — single source of truth) ──────
    if GEO_LOD_UTILS_AVAILABLE:
        g = get_graph()
    else:
        g = Graph()

    GEOLOD = Namespace("http://w3id.org/geo-lod/")
    SOSA = Namespace("http://www.w3.org/ns/sosa/")
    GEO = Namespace("http://www.opengis.net/ont/geosparql#")
    SF = Namespace("http://www.opengis.net/ont/sf#")
    QUDT = Namespace("http://qudt.org/schema/qudt/")
    UNIT = Namespace("http://qudt.org/vocab/unit/")

    if not GEO_LOD_UTILS_AVAILABLE:
        g.bind("geolod", GEOLOD)
        g.bind("sosa", SOSA)
        g.bind("geo", GEO)
        g.bind("qudt", QUDT)
        g.bind("unit", UNIT)
        g.bind("prov", PROV)
        g.bind("dct", DCTERMS)
        g.bind("owl", OWL)
        g.bind("rdfs", RDFS)
        g.bind("xsd", XSD)

    # ── DataSource (PROV) – wird referenziert, ist in Ontologie definiert ─────
    src = GEOLOD["SISALv3_DataSource"]

    # ── Cave ──────────────────────────────────────────────────────────────────
    site_id_val = int(df["site_id"].iloc[0])
    cave = GEOLOD[f"Cave_{site_slug}"]
    g.add((cave, RDF.type, GEOLOD["Cave"]))
    g.add((cave, RDFS.label, Literal(site_name, lang="en")))
    g.add((cave, GEOLOD["siteId"], Literal(site_id_val, datatype=XSD.integer)))

    # Geometrie (GeoSPARQL 1.1 / CI_full.py pattern — sf:Point + CRS-prefixed WKT)
    if "latitude" in df.columns and "longitude" in df.columns:
        lat = df["latitude"].iloc[0]
        lon = df["longitude"].iloc[0]
        if pd.notna(lat) and pd.notna(lon):
            geom = GEOLOD[f"Cave_{site_slug}_Geometry"]
            g.add(
                (geom, RDF.type, SF["Point"])
            )  # sf:Point only (subClassOf geo:Geometry)
            g.add(
                (
                    geom,
                    GEO["asWKT"],
                    Literal(
                        f"<http://www.opengis.net/def/crs/EPSG/0/4326> POINT({float(lon):.6f} {float(lat):.6f})",
                        datatype=GEO["wktLiteral"],
                    ),
                )
            )
            g.add((cave, GEO["hasGeometry"], geom))

    # ── Smoothing instances ───────────────────────────────────────────────────
    smooth_median = GEOLOD[f"RollingMedian_w{ROLLING_WINDOW}"]
    g.add((smooth_median, RDF.type, GEOLOD["RollingMedianFilter"]))
    g.add(
        (
            smooth_median,
            GEOLOD["windowSize"],
            Literal(ROLLING_WINDOW, datatype=XSD.integer),
        )
    )

    smooth_sg = GEOLOD[f"SavitzkyGolay_w{SG_WINDOW}_p{SG_POLYORDER}"]
    g.add((smooth_sg, RDF.type, GEOLOD["SavitzkyGolayFilter"]))
    g.add((smooth_sg, GEOLOD["windowSize"], Literal(SG_WINDOW, datatype=XSD.integer)))
    g.add((smooth_sg, GEOLOD["polyOrder"], Literal(SG_POLYORDER, datatype=XSD.integer)))

    # ── MeasurementType instances ─────────────────────────────────────────────
    mtype_d18o = GEOLOD["MeasurementType_d18O"]
    g.add((mtype_d18o, RDF.type, GEOLOD["MeasurementType"]))
    g.add((mtype_d18o, RDFS.label, Literal("delta-18O Measurement", lang="en")))

    mtype_d13c = GEOLOD["MeasurementType_d13C"]
    g.add((mtype_d13c, RDF.type, GEOLOD["MeasurementType"]))
    g.add((mtype_d13c, RDFS.label, Literal("delta-13C Measurement", lang="en")))

    # ── Observable Properties ─────────────────────────────────────────────────
    prop_d18o = GEOLOD["Delta18O_Speleothem"]
    g.add((prop_d18o, RDF.type, GEOLOD["Delta18OProperty"]))
    g.add((prop_d18o, RDFS.label, Literal("delta-18O (speleothem)", lang="en")))

    prop_d13c = GEOLOD["Delta13C_Speleothem"]
    g.add((prop_d13c, RDF.type, GEOLOD["Delta13CProperty"]))
    g.add((prop_d13c, RDFS.label, Literal("delta-13C (speleothem)", lang="en")))

    # ── U-Th chronology (one per site) ───────────────────────────────────────
    chron = GEOLOD[f"UThChronology_{site_slug}"]
    g.add((chron, RDF.type, GEOLOD["UThChronology"]))
    g.add((chron, RDFS.label, Literal(f"U-Th Chronology – {site_name}", lang="en")))
    g.add(
        (
            chron,
            RDFS.comment,
            Literal(
                "Standardised SISAL chronology (linear interpolation / "
                "Bchron / Bacon / copRa / StalAge). "
                "See SISALv3 repository: "
                "https://doi.org/10.5287/ora-2nanwp4rk",
                lang="en",
            ),
        )
    )

    # ── Speleotheme & Observations pro entity_id ──────────────────────────────
    obs_d18o_total = 0
    obs_d13c_total = 0

    for entity_id, grp in df.groupby("entity_id"):
        entity_name = (
            grp["entity_name"].iloc[0]
            if "entity_name" in grp.columns
            else str(entity_id)
        )
        speleothem = GEOLOD[f"Speleothem_{site_slug}_e{entity_id}"]
        g.add((speleothem, RDF.type, GEOLOD["Speleothem"]))
        g.add(
            (speleothem, RDFS.label, Literal(f"{site_name} – {entity_name}", lang="en"))
        )
        g.add(
            (
                speleothem,
                GEOLOD["entityId"],
                Literal(int(entity_id), datatype=XSD.integer),
            )
        )
        g.add((speleothem, GEOLOD["collectedFrom"], cave))

        # ── d18O ──────────────────────────────────────────────────────────────
        sub18 = grp[grp["d18o_permille"].notna() & grp["age_ka"].notna()].copy()
        if not sub18.empty:
            vals18 = sub18["d18o_permille"].values
            med18 = (
                pd.Series(vals18)
                .rolling(window=ROLLING_WINDOW, center=True, min_periods=1)
                .median()
                .values
            )
            try:
                sg18 = savgol_filter(
                    vals18, window_length=SG_WINDOW, polyorder=SG_POLYORDER
                )
            except Exception:
                sg18 = [None] * len(vals18)

            for i, (_, row) in enumerate(sub18.iterrows()):
                obs = GEOLOD[f"Obs_d18O_{site_slug}_e{entity_id}_{obs_d18o_total:05d}"]
                g.add((obs, RDF.type, GEOLOD["Delta18OSpeleothemObservation"]))
                g.add((obs, SOSA["hasFeatureOfInterest"], speleothem))
                g.add((obs, SOSA["observedProperty"], prop_d18o))
                g.add((obs, GEOLOD["measurementType"], mtype_d18o))
                g.add(
                    (
                        obs,
                        GEOLOD["ageKaBP"],
                        Literal(round(float(row["age_ka"]), 4), datatype=XSD.decimal),
                    )
                )
                g.add(
                    (
                        obs,
                        GEOLOD["measuredValue"],
                        Literal(
                            round(float(row["d18o_permille"]), 4), datatype=XSD.decimal
                        ),
                    )
                )
                if "depth_sample" in row and pd.notna(row["depth_sample"]):
                    g.add(
                        (
                            obs,
                            GEOLOD["atDepth_mm"],
                            Literal(
                                round(float(row["depth_sample"]), 3),
                                datatype=XSD.decimal,
                            ),
                        )
                    )
                g.add((obs, GEOLOD["ageChronologySpeleothem"], chron))
                if pd.notna(med18[i]):
                    g.add(
                        (
                            obs,
                            GEOLOD["smoothedValue_rollingMedian"],
                            Literal(round(float(med18[i]), 4), datatype=XSD.decimal),
                        )
                    )
                    g.add((obs, GEOLOD["smoothingMethod_median"], smooth_median))
                if sg18[i] is not None and pd.notna(sg18[i]):
                    g.add(
                        (
                            obs,
                            GEOLOD["smoothedValue_savgol"],
                            Literal(round(float(sg18[i]), 4), datatype=XSD.decimal),
                        )
                    )
                    g.add((obs, GEOLOD["smoothingMethod_savgol"], smooth_sg))
                g.add((obs, PROV.wasDerivedFrom, src))
                obs_d18o_total += 1

        # ── d13C ──────────────────────────────────────────────────────────────
        sub13 = grp[grp["d13c_permille"].notna() & grp["age_ka"].notna()].copy()
        if not sub13.empty:
            vals13 = sub13["d13c_permille"].values
            med13 = (
                pd.Series(vals13)
                .rolling(window=ROLLING_WINDOW, center=True, min_periods=1)
                .median()
                .values
            )
            try:
                sg13 = savgol_filter(
                    vals13, window_length=SG_WINDOW, polyorder=SG_POLYORDER
                )
            except Exception:
                sg13 = [None] * len(vals13)

            for i, (_, row) in enumerate(sub13.iterrows()):
                obs = GEOLOD[f"Obs_d13C_{site_slug}_e{entity_id}_{obs_d13c_total:05d}"]
                g.add((obs, RDF.type, GEOLOD["Delta13CSpeleothemObservation"]))
                g.add((obs, SOSA["hasFeatureOfInterest"], speleothem))
                g.add((obs, SOSA["observedProperty"], prop_d13c))
                g.add((obs, GEOLOD["measurementType"], mtype_d13c))
                g.add(
                    (
                        obs,
                        GEOLOD["ageKaBP"],
                        Literal(round(float(row["age_ka"]), 4), datatype=XSD.decimal),
                    )
                )
                g.add(
                    (
                        obs,
                        GEOLOD["measuredValue"],
                        Literal(
                            round(float(row["d13c_permille"]), 4), datatype=XSD.decimal
                        ),
                    )
                )
                if "depth_sample" in row and pd.notna(row["depth_sample"]):
                    g.add(
                        (
                            obs,
                            GEOLOD["atDepth_mm"],
                            Literal(
                                round(float(row["depth_sample"]), 3),
                                datatype=XSD.decimal,
                            ),
                        )
                    )
                g.add((obs, GEOLOD["ageChronologySpeleothem"], chron))
                if pd.notna(med13[i]):
                    g.add(
                        (
                            obs,
                            GEOLOD["smoothedValue_rollingMedian"],
                            Literal(round(float(med13[i]), 4), datatype=XSD.decimal),
                        )
                    )
                    g.add((obs, GEOLOD["smoothingMethod_median"], smooth_median))
                if sg13[i] is not None and pd.notna(sg13[i]):
                    g.add(
                        (
                            obs,
                            GEOLOD["smoothedValue_savgol"],
                            Literal(round(float(sg13[i]), 4), datatype=XSD.decimal),
                        )
                    )
                    g.add((obs, GEOLOD["smoothingMethod_savgol"], smooth_sg))
                g.add((obs, PROV.wasDerivedFrom, src))
                obs_d13c_total += 1

    print(
        f"  RDF: {obs_d18o_total:,} d18O obs · "
        f"{obs_d13c_total:,} d13C obs · "
        f"{len(g):,} triples"
    )
    return g


def load_sisal_sites_csv(filepath: str) -> "pd.DataFrame":
    """
    Loads the SISAL v_sites_all CSV.

    Expected columns: site_id, site_name, wkt, n_d18o_samples, n_d13c_samples
    WKT format:       POINT(lon lat)  (longitude first, as per GeoSPARQL)
    Returns a clean DataFrame, sorted by site_id.
    """
    df = pd.read_csv(filepath)
    df["site_id"] = pd.to_numeric(df["site_id"], errors="coerce")
    df["n_d18o_samples"] = pd.to_numeric(df["n_d18o_samples"], errors="coerce")
    df["n_d13c_samples"] = pd.to_numeric(df["n_d13c_samples"], errors="coerce")
    df = (
        df.dropna(subset=["site_id", "wkt"])
        .sort_values("site_id")
        .reset_index(drop=True)
    )

    print(f"  Loaded SISAL sites: {len(df)} sites")
    print(f"  d18O samples total: {df['n_d18o_samples'].sum():,}")
    print(f"  d13C samples total: {df['n_d13c_samples'].sum():,}")
    return df


def build_sisal_sites_rdf(df_sites: "pd.DataFrame") -> "Graph | None":
    """
    Builds an RDF graph for ALL 305 SISAL cave sites from v_sites_all.

    Each Cave instance gets:
      - geolod:siteId          (integer)
      - rdfs:label             (site_name)
      - geo:hasGeometry        → sf:Point with geo:asWKT literal
      - geolod:countD18OSamples (integer)
      - geolod:countD13CSamples (integer)
      - prov:wasDerivedFrom    → geolod:SISALv3_DataSource
    """
    if not RDF_AVAILABLE:
        return None

    # ── Graph + Namespaces (via geo_lod_utils) ───────────────────────────────
    if GEO_LOD_UTILS_AVAILABLE:
        g = get_graph()
    else:
        g = Graph()

    GEOLOD = Namespace("http://w3id.org/geo-lod/")
    GEO = Namespace("http://www.opengis.net/ont/geosparql#")
    SF = Namespace("http://www.opengis.net/ont/sf#")

    if not GEO_LOD_UTILS_AVAILABLE:
        g.bind("geolod", GEOLOD)
        g.bind("geo", GEO)
        g.bind("prov", PROV)
        g.bind("rdfs", RDFS)
        g.bind("xsd", XSD)

    src = GEOLOD["SISALv3_DataSource"]

    cave_uris: list = []  # für FeatureCollection
    for _, row in df_sites.iterrows():
        site_id = int(row["site_id"])
        site_name = str(row["site_name"])
        wkt = str(row["wkt"]).strip()

        # Slug: site_id zero-padded for consistent URI sorting
        slug = f"site_{site_id:04d}"
        cave = GEOLOD[f"Cave_{slug}"]

        g.add((cave, RDF.type, GEOLOD["Cave"]))
        g.add((cave, RDFS.label, Literal(site_name, lang="en")))
        g.add((cave, GEOLOD["siteId"], Literal(site_id, datatype=XSD.integer)))
        g.add(
            (
                cave,
                GEOLOD["countD18OSamples"],
                Literal(int(row["n_d18o_samples"]), datatype=XSD.integer),
            )
        )
        g.add(
            (
                cave,
                GEOLOD["countD13CSamples"],
                Literal(int(row["n_d13c_samples"]), datatype=XSD.integer),
            )
        )
        g.add((cave, PROV.wasDerivedFrom, src))

        # Geometry (GeoSPARQL 1.1 / CI_full.py pattern — sf:Point + CRS-prefixed WKT)
        geom = GEOLOD[f"Cave_{slug}_Geometry"]
        g.add((geom, RDF.type, SF["Point"]))  # sf:Point only (subClassOf geo:Geometry)
        # Inject EPSG:4326 CRS prefix if absent (CI_full.py pattern)
        if not wkt.startswith("<"):
            wkt = f"<http://www.opengis.net/def/crs/EPSG/0/4326> {wkt}"
        g.add(
            (
                geom,
                GEO["asWKT"],
                Literal(wkt, datatype=GEO["wktLiteral"]),
            )
        )
        g.add((cave, GEO["hasGeometry"], geom))

        cave_uris.append(cave)

    # ── FeatureCollection (GeoSPARQL 1.1 — enables QGIS / Linked Data viewers) ──
    collection = GEOLOD["SISAL_Cave_Collection"]
    g.add((collection, RDF.type, GEO["FeatureCollection"]))
    g.add((collection, RDFS.label, Literal("SISAL Cave Sites Collection", lang="en")))
    for cave_uri in cave_uris:
        g.add((collection, RDFS.member, cave_uri))
    print(
        f"  FeatureCollection: {len(cave_uris)} members → geolod:SISAL_Cave_Collection"
    )

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
    for cave_uri in cave_uris:
        g.add((global_collection, RDFS.member, cave_uri))
    print(
        f"  Global Collection: {len(cave_uris)} cave sites added → geolod:AllPalaeoclimateSites_Collection"
    )

    print(f"  RDF sites: {len(df_sites)} caves · {len(g):,} triples")
    return g


def export_sisal_rdf(
    all_dfs: list, site_slugs: list, df_sites: "pd.DataFrame | None" = None
) -> None:
    """
    Exports all RDF artefacts:
      rdf/sisal_ontology.ttl    – classes, properties, instances
      rdf/sisal_sites.ttl       – all 305 SISAL cave sites (from v_sites_all)
      rdf/sisal_{slug}_data.ttl – observation data per cave
      rdf/sisal_all_data.ttl    – combined graph (sites + all cave data)
    """
    if not RDF_AVAILABLE:
        print("  ⚠  RDF export skipped (rdflib not available).")
        return

    print("\n" + "─" * 60)
    print("RDF Export  (geo-lod SISAL Ontology + data)")
    print("─" * 60)

    # 1a. Core ontology (geo_lod_core.ttl) via geo_lod_utils
    if GEO_LOD_UTILS_AVAILABLE:
        write_geo_lod_core(RDF_DIR)
        write_geo_lod_mermaid(
            ONTOLOGY_DIR,
            rolling_window=ROLLING_WINDOW,
            sg_window=SG_WINDOW,
            sg_poly=SG_POLYORDER,
            n_sisal_sites=305,
        )
    else:
        print("  ⚠  geo_lod_utils not available – geo_lod_core.ttl / Mermaid skipped.")

    # 1b. SISAL extension ontology (sisal_ontology.ttl)
    onto_path = os.path.join(RDF_DIR, "sisal_ontology.ttl")
    with open(onto_path, "w", encoding="utf-8") as f:
        f.write(SISAL_ONTOLOGY_TTL)
    print(f"  ✓ {onto_path}")

    # 2. Combined graph (will accumulate sites + cave data)
    GEOLOD = Namespace("http://w3id.org/geo-lod/")
    if GEO_LOD_UTILS_AVAILABLE:
        combined = get_graph()
    else:
        combined = Graph()
        combined.bind("geolod", GEOLOD)
        combined.bind("sosa", Namespace("http://www.w3.org/ns/sosa/"))
        combined.bind("geo", Namespace("http://www.opengis.net/ont/geosparql#"))
        combined.bind("prov", PROV)
        combined.bind("dct", DCTERMS)
        combined.bind("rdfs", RDFS)
        combined.bind("xsd", XSD)

    # 3. Sites graph (all 305 SISAL cave sites)
    if df_sites is not None:
        print(f"\n  Building sites graph …")
        g_sites = build_sisal_sites_rdf(df_sites)
        if g_sites is not None:
            sites_path = os.path.join(RDF_DIR, "sisal_sites.ttl")
            g_sites.serialize(destination=sites_path, format="turtle")
            print(f"  ✓ {sites_path}")
            for triple in g_sites:
                combined.add(triple)
    else:
        print("  ⚠  No sites CSV provided – sisal_sites.ttl skipped.")

    # 4. Per-cave observation data
    for df, slug in zip(all_dfs, site_slugs):
        site_name = df["site_name"].iloc[0]
        print(f"\n  {site_name}  ({slug})")
        g = build_sisal_rdf(df, site_name=site_name, site_slug=slug)
        if g is None:
            continue
        out_path = os.path.join(RDF_DIR, f"sisal_{slug}_data.ttl")
        g.serialize(destination=out_path, format="turtle")
        print(f"  ✓ {out_path}")
        for triple in g:
            combined.add(triple)

    # 5. Combined graph
    combined_path = os.path.join(RDF_DIR, "sisal_all_data.ttl")
    combined.serialize(destination=combined_path, format="turtle")
    print(f"\n  ✓ {combined_path}  ({len(combined):,} triples total)")

    # 6. Combined Sites Collection (optional — if EPICA is available)
    # This creates a FeatureCollection that references both EPICA and SISAL sites.
    if GEO_LOD_UTILS_AVAILABLE:
        epica_data_path = os.path.join(RDF_DIR, "epica_dome_c.ttl")
        sisal_sites_path = os.path.join(RDF_DIR, "sisal_sites.ttl")

        # Combined sites collection - feature not implemented
        print("  ℹ  Combined collection skipped (feature not implemented)")
    else:
        print("  ⚠  geo_lod_utils not available – combined collection skipped.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    from datetime import datetime

    report_path = os.path.join(REPORT_DIR, "report.txt")
    tee = Tee(report_path)

    print("=" * 60)
    print("SISAL Speleothem – Plot Generator")
    print("=" * 60)

    # ── Configure SISAL sites file (all 305 sites) ────────────────────────────
    SITES_FILE = "v_sites_all.csv"

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

    # ── Load sites CSV ────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Loading sites: {SITES_FILE}")
    print("─" * 60)
    sites_filepath = (
        SITES_FILE
        if os.path.isabs(SITES_FILE)
        else os.path.join(SCRIPT_DIR, SITES_FILE)
    )
    df_sites = None
    if os.path.exists(sites_filepath):
        df_sites = load_sisal_sites_csv(sites_filepath)
    else:
        print(f"  ⚠  Sites file not found: {sites_filepath} – skipping.")

    total_plots = 0
    loaded_dfs = []  # collected for RDF export
    loaded_slugs = []

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

        # collect for RDF export
        loaded_dfs.append(df)
        loaded_slugs.append(cfg["slug"])

    # ── RDF Export ────────────────────────────────────────────────────────────
    export_sisal_rdf(loaded_dfs, loaded_slugs, df_sites=df_sites)

    print("\n" + "=" * 60)
    print(f"Done! Plots saved to '{OUTPUT_DIR}/'")
    print(f"Total: {total_plots} plots")
    if RDF_AVAILABLE:
        print(f"RDF files saved to '{RDF_DIR}/'")
    print("=" * 60)
    print(f"Report saved: {report_path}")
    tee.close()


if __name__ == "__main__":
    main()
