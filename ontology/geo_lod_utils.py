"""
geo_lod_utils.py
================
Shared utilities for the geo-lod palaeoclimate Linked Data pipeline.

Used by
-------
    EPICA/plot_epica_from_tab.py    ice-core data  (EPICA Dome C)
    SISAL/plot_sisal_from_csv.py    speleothem data (SISALv3)

Geometry pattern
----------------
Follows CI_full.py (Florian Thiery, MIT 2023) exactly:

    site  a  geo:Feature, crm:E53_Place, crm:E27_Site [, domain-class] ;
          rdfs:label        "..."@en ;
          geo:hasGeometry   site_geom .

    site_geom  a  sf:Point ;
               geo:asWKT  "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(lon lat)"
                          ^^geo:wktLiteral .

    # sf:Point is a subclass of geo:Geometry via the SF ontology —
    # the OWL entailment covers it; geo:Geometry is NOT asserted directly
    # (CI_full.py does not assert it either).

FeatureCollection pattern
-------------------------
    collection  a  geo:FeatureCollection ;
                rdfs:label   "..."@en ;
                rdfs:member  site1, site2, ... .

Import in calling scripts
-------------------------
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ontology"))
    from geo_lod_utils import (
        NS, CRS_WGS84, GEOLOD_BASE,
        get_graph,
        wkt_point,
        add_geo_site, add_geo_site_from_wkt,
        add_feature_collection,
        write_geo_lod_core,
        write_mermaid,
        GEO_LOD_CORE_TTL,
    )
"""

from __future__ import annotations

import os
import textwrap
from typing import Iterable

# ---------------------------------------------------------------------------
# Optional rdflib import
# ---------------------------------------------------------------------------
try:
    from rdflib import Graph, Namespace, URIRef, Literal
    from rdflib.namespace import RDF, RDFS, OWL, XSD

    RDF_AVAILABLE = True
except ImportError:
    RDF_AVAILABLE = False


# ===========================================================================
# 1.  NAMESPACES  — single source of truth for the whole pipeline
# ===========================================================================

GEOLOD_BASE: str = "http://w3id.org/geo-lod/"

#: CRS URI embedded in every WKT literal (GeoSPARQL 1.1 / CI_full.py pattern)
CRS_WGS84: str = "http://www.opengis.net/def/crs/EPSG/0/4326"

#: All namespace URI strings used across EPICA and SISAL
NS: dict[str, str] = {
    # W3C standards
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    # Geospatial
    "geo": "http://www.opengis.net/ont/geosparql#",
    "sf": "http://www.opengis.net/ont/sf#",
    # CIDOC-CRM
    "crm": "http://www.cidoc-crm.org/cidoc-crm/",
    "crmsci": "http://www.ics.forth.gr/isl/CRMsci/",
    # Observation & measurement
    "sosa": "http://www.w3.org/ns/sosa/",
    "ssn": "http://www.w3.org/ns/ssn/",
    "qudt": "http://qudt.org/schema/qudt/",
    "unit": "http://qudt.org/vocab/unit/",
    # Provenance & metadata
    "prov": "http://www.w3.org/ns/prov#",
    "dct": "http://purl.org/dc/terms/",
    "dcat": "http://www.w3.org/ns/dcat#",
    "void": "http://rdfs.org/ns/void#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    # Project
    "geolod": GEOLOD_BASE,
}


# ===========================================================================
# 2.  GRAPH FACTORY
# ===========================================================================


def get_graph() -> "Graph":
    """
    Return a fresh rdflib.Graph with every geo-lod namespace pre-bound.
    Raises ImportError if rdflib is not installed.
    """
    if not RDF_AVAILABLE:
        raise ImportError("rdflib is required.  pip install rdflib")
    g = Graph()
    for prefix, uri in NS.items():
        g.bind(prefix, Namespace(uri))
    return g


# ===========================================================================
# 3.  GeoSPARQL GEOMETRY HELPERS  (CI_full.py pattern)
# ===========================================================================


def wkt_point(lon: float, lat: float, precision: int = 6) -> str:
    """
    Return a CRS-prefixed WKT POINT string for use as geo:asWKT value.

    Follows CI_full.py (Florian Thiery, MIT 2023) exactly:
        "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(lon lat)"

    GeoSPARQL WKT uses (longitude latitude) order — NOT (lat lon).
    The CRS prefix is mandatory for correct SPARQL geo-operations and
    compatibility with tools such as the SPARQLing Unicorn QGIS plugin.
    """
    fmt = f"{{:.{precision}f}}"
    return f"<{CRS_WGS84}> POINT({fmt.format(lon)} {fmt.format(lat)})"


def _ensure_crs(wkt: str) -> str:
    """
    Prepend the EPSG:4326 CRS prefix if the WKT string lacks one.
    Idempotent: already-prefixed strings are returned unchanged.

    CI_full.py always adds the prefix when building triples from CSV;
    this helper provides the same guarantee for SISAL CSV imports.
    """
    wkt = wkt.strip()
    if not wkt.startswith("<"):
        wkt = f"<{CRS_WGS84}> {wkt}"
    return wkt


def add_geo_site(
    g: "Graph",
    site_uri: "URIRef",
    geom_uri: "URIRef",
    label: str,
    lon: float,
    lat: float,
    extra_types: Iterable["URIRef"] = (),
) -> None:
    """
    Add a GeoSPARQL Feature + sf:Point geometry to *g* from lon/lat values.

    Writes (CI_full.py pattern):
        site_uri  a  geo:Feature, crm:E53_Place, crm:E27_Site [, extra_types] ;
                  rdfs:label        "<label>"@en ;
                  geo:hasGeometry   geom_uri .

        geom_uri  a  sf:Point ;
                  geo:asWKT  "<CRS> POINT(lon lat)"^^geo:wktLiteral .

    Note: geo:Geometry is NOT asserted explicitly — sf:Point is a subclass
    of geo:Geometry via the OGC Simple Features ontology (OWL entailment).
    """
    GEO = Namespace(NS["geo"])
    SF = Namespace(NS["sf"])
    CRM = Namespace(NS["crm"])

    # Site / Feature
    g.add((site_uri, RDF.type, GEO["Feature"]))
    g.add((site_uri, RDF.type, CRM["E53_Place"]))
    g.add((site_uri, RDF.type, CRM["E27_Site"]))
    for t in extra_types:
        g.add((site_uri, RDF.type, t))
    g.add((site_uri, RDFS.label, Literal(label, lang="en")))
    g.add((site_uri, GEO["hasGeometry"], geom_uri))

    # Geometry  (sf:Point only — no geo:Geometry, matches CI_full.py)
    g.add((geom_uri, RDF.type, SF["Point"]))
    g.add(
        (
            geom_uri,
            GEO["asWKT"],
            Literal(wkt_point(lon, lat), datatype=GEO["wktLiteral"]),
        )
    )


def add_geo_site_from_wkt(
    g: "Graph",
    site_uri: "URIRef",
    geom_uri: "URIRef",
    label: str,
    wkt: str,
    extra_types: Iterable["URIRef"] = (),
) -> None:
    """
    Like add_geo_site() but accepts a pre-formed WKT string (e.g. from CSV).

    The CRS prefix is injected automatically if absent — matching CI_full.py.
    Used by SISAL where v_sites_all.csv already provides WKT POINT values.
    """
    GEO = Namespace(NS["geo"])
    SF = Namespace(NS["sf"])
    CRM = Namespace(NS["crm"])

    wkt = _ensure_crs(wkt)

    # Site / Feature
    g.add((site_uri, RDF.type, GEO["Feature"]))
    g.add((site_uri, RDF.type, CRM["E53_Place"]))
    g.add((site_uri, RDF.type, CRM["E27_Site"]))
    for t in extra_types:
        g.add((site_uri, RDF.type, t))
    g.add((site_uri, RDFS.label, Literal(label, lang="en")))
    g.add((site_uri, GEO["hasGeometry"], geom_uri))

    # Geometry
    g.add((geom_uri, RDF.type, SF["Point"]))
    g.add((geom_uri, GEO["asWKT"], Literal(wkt, datatype=GEO["wktLiteral"])))


def add_feature_collection(
    g: "Graph",
    collection_uri: "URIRef",
    label: str,
    members: Iterable["URIRef"],
) -> None:
    """
    Add a geo:FeatureCollection to *g* with rdfs:member links.

    Pattern (CI_full.py Site_collection page):
        collection_uri  a  geo:FeatureCollection ;
                        rdfs:label   "<label>"@en ;
                        rdfs:member  member1, member2, ... .
    """
    GEO = Namespace(NS["geo"])
    g.add((collection_uri, RDF.type, GEO["FeatureCollection"]))
    g.add((collection_uri, RDFS.label, Literal(label, lang="en")))
    for m in members:
        g.add((collection_uri, RDFS.member, m))


# ===========================================================================
# 4.  CORE OWL ONTOLOGY  — shared classes / properties (EPICA + SISAL)
# ===========================================================================

GEO_LOD_CORE_TTL: str = textwrap.dedent(
    """\
    # ==========================================================================
    # geo_lod_core.ttl
    # geo-lod Core Ontology  —  shared vocabulary for EPICA and SISAL
    # <http://w3id.org/geo-lod/>
    #
    # Domain-specific extensions:
    #   epica_ontology.ttl  (IceCore, DrillingSite, CH4Observation, ...)
    #   sisal_ontology.ttl  (Cave, Speleothem, SpeleothemObservation, ...)
    # Both import this file via owl:imports.
    #
    # GeoSPARQL geometry pattern (CI_full.py, Florian Thiery, MIT 2023):
    #   site  a geo:Feature, crm:E53_Place, crm:E27_Site ;
    #         geo:hasGeometry  site_geom .
    #   site_geom  a sf:Point ;
    #     geo:asWKT "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(lon lat)"
    #              ^^geo:wktLiteral .
    # ==========================================================================

    @prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl:     <http://www.w3.org/2002/07/owl#> .
    @prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .
    @prefix geo:     <http://www.opengis.net/ont/geosparql#> .
    @prefix sf:      <http://www.opengis.net/ont/sf#> .
    @prefix crm:     <http://www.cidoc-crm.org/cidoc-crm/> .
    @prefix crmsci:  <http://www.ics.forth.gr/isl/CRMsci/> .
    @prefix sosa:    <http://www.w3.org/ns/sosa/> .
    @prefix qudt:    <http://qudt.org/schema/qudt/> .
    @prefix unit:    <http://qudt.org/vocab/unit/> .
    @prefix prov:    <http://www.w3.org/ns/prov#> .
    @prefix dct:     <http://purl.org/dc/terms/> .
    @prefix dcat:    <http://www.w3.org/ns/dcat#> .
    @prefix skos:    <http://www.w3.org/2004/02/skos/core#> .
    @prefix geolod:  <http://w3id.org/geo-lod/> .

    <http://w3id.org/geo-lod/>
        a owl:Ontology ;
        rdfs:label   "geo-lod Core Ontology"@en ;
        rdfs:comment "Shared vocabulary for EPICA ice-core and SISAL speleothem \\
    palaeoclimate Linked Data. Domain extensions import this file."@en ;
        owl:versionInfo "1.0" .

    # ==========================================================================
    # CLASSES
    # ==========================================================================

    # -- SamplingLocation  (superclass: DrillingSite in EPICA, Cave in SISAL) --

    geolod:SamplingLocation
        a owl:Class ;
        rdfs:subClassOf geo:Feature ;
        rdfs:subClassOf crm:E53_Place ;
        rdfs:subClassOf crm:E27_Site ;
        rdfs:label   "Sampling Location"@en ;
        rdfs:comment "A geographically identified location from which palaeoclimate \\
    proxy material was obtained (ice-core drilling site or speleothem cave)."@en .

    # -- PalaeoclimateSample  (superclass: IceCore in EPICA, Speleothem in SISAL) --

    geolod:PalaeoclimateSample
        a owl:Class ;
        rdfs:subClassOf sosa:Sample ;
        rdfs:label   "Palaeoclimate Sample"@en ;
        rdfs:comment "A physical archive recording a palaeoclimate signal \\
    (ice core or speleothem)."@en .

    # -- PalaeoclimateObservation  (superclass for all measurement types) --

    geolod:PalaeoclimateObservation
        a owl:Class ;
        rdfs:subClassOf crmsci:S4_Observation ;
        rdfs:subClassOf sosa:Observation ;
        rdfs:label   "Palaeoclimate Observation"@en ;
        rdfs:comment "A single measured value (e.g. δ¹⁸O, δ¹³C, CH₄) at a \\
    known depth or age within a palaeoclimate archive."@en .

    # -- ObservableProperty --

    geolod:ObservableProperty
        a owl:Class ;
        rdfs:subClassOf sosa:ObservableProperty ;
        rdfs:subClassOf crmsci:S9_Property_Type ;
        rdfs:label   "Observable Property"@en ;
        rdfs:comment "A measurable geochemical or physical property of a \\
    palaeoclimate sample (e.g. δ¹⁸O, δ¹³C, CH₄ concentration)."@en .

    geolod:Delta18OProperty
        a owl:Class ;
        rdfs:subClassOf geolod:ObservableProperty ;
        rdfs:label   "δ¹⁸O Property"@en ;
        rdfs:comment "Stable oxygen isotope ratio (δ¹⁸O), shared by both \\
    ice-core and speleothem records."@en .

    # -- Chronology  (superclass: IceCoreChronology in EPICA, UThChronology in SISAL) --

    geolod:Chronology
        a owl:Class ;
        rdfs:label   "Chronology"@en ;
        rdfs:comment "A depth-age model assigning calendar ages to positions \\
    within a palaeoclimate archive."@en .

    # -- MeasurementType --

    geolod:MeasurementType
        a owl:Class ;
        rdfs:subClassOf crmsci:S6_Data_Evaluation ;
        rdfs:label   "Measurement Type"@en ;
        rdfs:comment "Classifies an observation by the physical quantity measured."@en .

    # -- SmoothingMethod and subclasses --

    geolod:SmoothingMethod
        a owl:Class ;
        rdfs:subClassOf crmsci:S6_Data_Evaluation ;
        rdfs:label   "Smoothing Method"@en ;
        rdfs:comment "A numerical method applied to reduce high-frequency \\
    noise in a palaeoclimate time series."@en .

    geolod:RollingMedianFilter
        a owl:Class ;
        rdfs:subClassOf geolod:SmoothingMethod ;
        rdfs:label   "Rolling Median Filter"@en ;
        rdfs:comment "Non-parametric smoother computing the median within \\
    a sliding window of fixed width."@en .

    geolod:SavitzkyGolayFilter
        a owl:Class ;
        rdfs:subClassOf geolod:SmoothingMethod ;
        rdfs:label   "Savitzky-Golay Filter"@en ;
        rdfs:comment "Polynomial least-squares smoothing filter preserving \\
    higher signal moments."@en .

    # -- DataSource --

    geolod:DataSource
        a owl:Class ;
        rdfs:subClassOf prov:Entity ;
        rdfs:label   "Data Source"@en ;
        rdfs:comment "A citable source (database, repository, publication) \\
    from which palaeoclimate observations were obtained."@en .

    # ==========================================================================
    # OBJECT PROPERTIES
    # ==========================================================================

    geolod:ageChronology
        a owl:ObjectProperty ;
        rdfs:label   "age chronology"@en ;
        rdfs:comment "Links an observation to the depth-age model used."@en .

    geolod:measurementType
        a owl:ObjectProperty ;
        rdfs:domain  geolod:PalaeoclimateObservation ;
        rdfs:range   geolod:MeasurementType ;
        rdfs:label   "measurement type"@en .

    geolod:smoothingMethod_median
        a owl:ObjectProperty ;
        rdfs:domain  geolod:PalaeoclimateObservation ;
        rdfs:range   geolod:RollingMedianFilter ;
        rdfs:label   "smoothing method (rolling median)"@en .

    geolod:smoothingMethod_savgol
        a owl:ObjectProperty ;
        rdfs:domain  geolod:PalaeoclimateObservation ;
        rdfs:range   geolod:SavitzkyGolayFilter ;
        rdfs:label   "smoothing method (Savitzky-Golay)"@en .

    geolod:tookPlaceAt
        a owl:ObjectProperty ;
        rdfs:range   geolod:SamplingLocation ;
        rdfs:label   "took place at"@en ;
        rdfs:comment "Links a sampling event or campaign to its location."@en .

    geolod:extractedFrom
        a owl:ObjectProperty ;
        rdfs:domain  geolod:PalaeoclimateSample ;
        rdfs:range   geolod:SamplingLocation ;
        rdfs:label   "extracted from"@en .

    geolod:removedSample
        a owl:ObjectProperty ;
        rdfs:label   "removed sample"@en ;
        rdfs:comment "Links a sampling event to the sample taken."@en .

    geolod:hasObservation
        a owl:ObjectProperty ;
        rdfs:label   "has observation"@en .

    # ==========================================================================
    # DATATYPE PROPERTIES
    # ==========================================================================

    geolod:ageKaBP
        a owl:DatatypeProperty ;
        rdfs:range   xsd:decimal ;
        rdfs:label   "age (ka BP)"@en ;
        rdfs:comment "Age in thousands of years before present (ka BP)."@en .

    geolod:measuredValue
        a owl:DatatypeProperty ;
        rdfs:domain  geolod:PalaeoclimateObservation ;
        rdfs:range   xsd:decimal ;
        rdfs:label   "measured value"@en .

    geolod:smoothedValue_rollingMedian
        a owl:DatatypeProperty ;
        rdfs:domain  geolod:PalaeoclimateObservation ;
        rdfs:range   xsd:decimal ;
        rdfs:label   "smoothed value (rolling median)"@en .

    geolod:smoothedValue_savgol
        a owl:DatatypeProperty ;
        rdfs:domain  geolod:PalaeoclimateObservation ;
        rdfs:range   xsd:decimal ;
        rdfs:label   "smoothed value (Savitzky-Golay)"@en .

    geolod:windowSize
        a owl:DatatypeProperty ;
        rdfs:domain  geolod:SmoothingMethod ;
        rdfs:range   xsd:integer ;
        rdfs:label   "window size"@en ;
        rdfs:comment "Number of data points in the smoothing window."@en .

    geolod:polyOrder
        a owl:DatatypeProperty ;
        rdfs:domain  geolod:SavitzkyGolayFilter ;
        rdfs:range   xsd:integer ;
        rdfs:label   "polynomial order"@en .

    # ==========================================================================
    # NAMED INDIVIDUALS  — shared between EPICA and SISAL
    # ==========================================================================

    geolod:Delta18O
        a geolod:Delta18OProperty, owl:NamedIndividual ;
        rdfs:label   "δ¹⁸O"@en ;
        rdfs:comment "Stable oxygen isotope ratio — shared observable property \\
    used in both ice-core and speleothem observations."@en .

    geolod:MeasurementType_d18O
        a geolod:MeasurementType, owl:NamedIndividual ;
        rdfs:label   "δ¹⁸O measurement"@en .

    # ==========================================================================
    # EXTERNAL VOCABULARY LABELS
    # (so Protégé also shows labels for imported terms)
    # ==========================================================================

    geo:Feature
        rdfs:label   "Feature"@en ;
        rdfs:comment "An abstraction of a real-world phenomenon (OGC GeoSPARQL)."@en .

    geo:FeatureCollection
        rdfs:label   "Feature Collection"@en ;
        rdfs:comment "A collection of geo:Feature instances (GeoSPARQL 1.1)."@en .

    geo:Geometry
        rdfs:label   "Geometry"@en ;
        rdfs:comment "A coherent set of direct positions in space (OGC GeoSPARQL)."@en .

    geo:hasGeometry    rdfs:label "has geometry"@en .
    geo:asWKT          rdfs:label "as WKT"@en .

    sf:Point
        rdfs:label   "Point"@en ;
        rdfs:comment "A single location in n-dimensional space (OGC Simple Features). \\
    Subclass of geo:Geometry."@en .

    crm:E27_Site
        rdfs:label   "Site"@en ;
        rdfs:comment "A place that was or is the focus of human activity (CIDOC-CRM E27)."@en .

    crm:E53_Place
        rdfs:label   "Place"@en ;
        rdfs:comment "An extent in space (CIDOC-CRM E53)."@en .

    crmsci:S4_Observation
        rdfs:label   "Observation"@en ;
        rdfs:comment "A scientific observation event (CRMsci S4)."@en .

    crmsci:S6_Data_Evaluation
        rdfs:label   "Data Evaluation"@en ;
        rdfs:comment "A process producing a value by evaluating data (CRMsci S6)."@en .

    crmsci:S9_Property_Type
        rdfs:label   "Property Type"@en ;
        rdfs:comment "A type of measurable property (CRMsci S9)."@en .

    crmsci:S1_Matter_Removal
        rdfs:label   "Matter Removal"@en ;
        rdfs:comment "A process of removing matter from an object (CRMsci S1)."@en .

    sosa:Observation        rdfs:label "Observation"@en .
    sosa:ObservableProperty rdfs:label "Observable Property"@en .
    sosa:Sample             rdfs:label "Sample"@en .
    prov:Entity             rdfs:label "Entity"@en .
"""
)


def write_geo_lod_core(outdir: str) -> str:
    """
    Write GEO_LOD_CORE_TTL to <outdir>/geo_lod_core.ttl.

    Called by both EPICA and SISAL export functions — whichever runs first
    creates the file; the second overwrites it with the identical content.
    Returns the full path.
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "geo_lod_core.ttl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(GEO_LOD_CORE_TTL)
    print(f"  ✓ Core ontology : {path}")
    return path


# ===========================================================================
# 5.  MERMAID DIAGRAMS
# ===========================================================================

#: Combined taxonomy — Core + EPICA extension + SISAL extension
MERMAID_TAXONOMY: str = textwrap.dedent(
    """\
    flowchart LR

        subgraph EXT["External Ontologies"]
            direction TB

            subgraph GEO_ONT["GeoSPARQL"]
                direction TB
                GF["geo:Feature"]
                GFC["geo:FeatureCollection"]
                GG["geo:Geometry"]
            end

            subgraph SF_ONT["Simple Features"]
                SFP["sf:Point"]
            end

            subgraph CRM_GRP["CIDOC-CRM"]
                direction TB
                CE53["crm:E53_Place"]
                CE27["crm:E27_Site"]
                CE22["crm:E22_Human-Made_Object"]
                CE7["crm:E7_Activity"]
            end

            subgraph CRMSCI_GRP["CRMsci"]
                direction TB
                CS4["crmsci:S4_Observation"]
                CS6["crmsci:S6_Data_Evaluation"]
                CS9["crmsci:S9_Property_Type"]
                CS1["crmsci:S1_Matter_Removal"]
            end

            subgraph SOSA_GRP["SOSA / SSN"]
                direction TB
                SO["sosa:Observation"]
                SS["sosa:Sample"]
                SP["sosa:ObservableProperty"]
            end

            subgraph PROV_GRP["PROV-O"]
                PE["prov:Entity"]
            end

            SFP -->|subClassOf| GG
            GFC -.->|rdfs:member| GF
        end

        subgraph CORE["geo-lod Core  (geo_lod_core.ttl)"]
            direction TB

            subgraph C_LOC["Locations"]
                SL["SamplingLocation"]
                SL -->|subClassOf| GF
                SL -->|subClassOf| CE53
                SL -->|subClassOf| CE27
            end

            subgraph C_SAMP["Samples"]
                PS["PalaeoclimateSample"]
                PS -->|subClassOf| SS
            end

            subgraph C_OBS["Observations"]
                PO["PalaeoclimateObservation"]
                PO -->|subClassOf| CS4
                PO -->|subClassOf| SO
            end

            subgraph C_PROP["Observable Properties"]
                OP["ObservableProperty"]
                D18OP["Delta18OProperty"]
                OP -->|subClassOf| SP
                OP -->|subClassOf| CS9
                D18OP -->|subClassOf| OP
            end

            subgraph C_CHRON["Chronology"]
                CH["Chronology"]
            end

            subgraph C_SMOOTH["Smoothing"]
                SM["SmoothingMethod"]
                RM["RollingMedianFilter"]
                SG["SavitzkyGolayFilter"]
                SM -->|subClassOf| CS6
                RM -->|subClassOf| SM
                SG -->|subClassOf| SM
            end

            subgraph C_MTYPE["Measurement Type"]
                MT["MeasurementType"]
                MT -->|subClassOf| CS6
            end

            subgraph C_DS["Data Source"]
                DS["DataSource"]
                DS -->|subClassOf| PE
            end
        end

        subgraph EPICA["geo-lod EPICA Extension  (epica_ontology.ttl)"]
            direction TB

            subgraph EP_LOC["Ice Core Location"]
                DRSITE["DrillingSite"]
                DRSITE -->|subClassOf| SL
            end

            subgraph EP_SAMP["Ice Core Sample"]
                IC["IceCore"]
                IC -->|subClassOf| PS
                IC -->|subClassOf| CE22
            end

            subgraph EP_OBS["Ice Core Observations"]
                ICO["IceCoreObservation"]
                CH4O["CH4Observation"]
                D18OO["Delta18OObservation"]
                ICO -->|subClassOf| PO
                CH4O -->|subClassOf| ICO
                D18OO -->|subClassOf| ICO
            end

            subgraph EP_PROP["Ice Core Properties"]
                CH4P["CH4ConcentrationProperty"]
                CH4P -->|subClassOf| OP
            end

            subgraph EP_CHRON["Ice Core Chronology"]
                ICC["IceCoreChronology"]
                ICC -->|subClassOf| CH
            end

            subgraph EP_DATA["Catalogue & Dataset"]
                PCC["PalaeoclimateDataCatalogue"]
                ICD["IceCoreDataset"]
                CH4D["CH4Dataset"]
                D18OD["Delta18ODataset"]
                CH4D -->|subClassOf| ICD
                D18OD -->|subClassOf| ICD
            end

            subgraph EP_CAMP["Campaign"]
                DC["DrillingCampaign"]
                DC -->|subClassOf| CE7
            end
        end

        subgraph SISAL["geo-lod SISAL Extension  (sisal_ontology.ttl)"]
            direction TB

            subgraph SI_LOC["Speleothem Location"]
                CAVE["Cave"]
                CAVE -->|subClassOf| SL
            end

            subgraph SI_SAMP["Speleothem Sample"]
                SPEL["Speleothem"]
                SPEL -->|subClassOf| PS
            end

            subgraph SI_EVT["Sampling Event"]
                SSE["SpeleothemSamplingEvent"]
                SSE -->|subClassOf| CS1
            end

            subgraph SI_OBS["Speleothem Observations"]
                SPO["SpeleothemObservation"]
                D18OS["Delta18OSpeleothemObservation"]
                D13CS["Delta13CSpeleothemObservation"]
                SPO -->|subClassOf| PO
                D18OS -->|subClassOf| SPO
                D13CS -->|subClassOf| SPO
            end

            subgraph SI_PROP["Speleothem Properties"]
                D13CP["Delta13CProperty"]
                D13CP -->|subClassOf| OP
            end

            subgraph SI_CHRON["U-Th Chronology"]
                UTHC["UThChronology"]
                UTHC -->|subClassOf| CH
            end
        end
"""
)


def _mermaid_instance_epica(rw: int, sgw: int, sgp: int) -> str:
    """EPICA named-individual instance diagram."""
    return textwrap.dedent(
        f"""\
        flowchart LR

            COLLECTION["geo:FeatureCollection
            geolod:EPICA_DrillingSite_Collection
            1 member"]

            CATALOG["PalaeoclimateDataCatalogue
            geolod:EPICA_DomeC_Catalog"]

            DATASET["IceCoreDataset
            geolod:EPICA_DomeC_Dataset"]

            OBS["IceCoreObservation
            geolod:Obs_CH4_0001 ...
            geolod:Obs_d18O_0001 ..."]

            CORE["IceCore
            geolod:EpicaDomeC_IceCore"]

            SITE["DrillingSite / geo:Feature
            geolod:EpicaDomeC_Site
            crm:E53_Place · crm:E27_Site"]

            GEOM["sf:Point
            geolod:EpicaDomeC_Geometry
            asWKT: EPSG:4326 POINT(123.35 -75.1)"]

            CAMPAIGN["DrillingCampaign
            EPICA Dome C 1996-2004"]

            PROP_CH4["CH4ConcentrationProperty
            geolod:CH4Concentration"]

            PROP_D18O["Delta18OProperty (shared core)
            geolod:Delta18O"]

            MTYPE_CH4["MeasurementType
            geolod:MeasurementType_CH4"]

            MTYPE_D18O["MeasurementType (shared core)
            geolod:MeasurementType_d18O"]

            CHRON_EDC2["IceCoreChronology
            geolod:EDC2_Chronology"]

            CHRON_AICC["IceCoreChronology
            geolod:AICC2023_Chronology"]

            MEDIAN["RollingMedianFilter
            geolod:RollingMedian_w{rw}
            windowSize: {rw}"]

            SAVGOL["SavitzkyGolayFilter
            geolod:SavitzkyGolay_w{sgw}_p{sgp}
            windowSize: {sgw} polyOrder: {sgp}"]

            DATASRC["DataSource
            geolod:EPICA_DomeC_CH4_Source
            geolod:EPICA_DomeC_d18O_Source"]

            COLLECTION -->|"rdfs:member"| SITE
            CATALOG  -->|"dcat:dataset"| DATASET
            DATASET  -->|"geolod:hasObservation"| OBS
            OBS      -->|"sosa:hasFeatureOfInterest"| CORE
            CORE     -->|"geolod:extractedFrom"| SITE
            SITE     -->|"geo:hasGeometry"| GEOM
            CAMPAIGN -->|"crm:P7_took_place_at"| SITE
            OBS      -->|"geolod:measurementType"| MTYPE_CH4
            OBS      -->|"geolod:measurementType"| MTYPE_D18O
            OBS      -->|"geolod:ageChronology"| CHRON_EDC2
            OBS      -->|"geolod:ageChronology"| CHRON_AICC
            OBS      -->|"geolod:smoothingMethod_median"| MEDIAN
            OBS      -->|"geolod:smoothingMethod_savgol"| SAVGOL
            OBS      -->|"prov:wasDerivedFrom"| DATASRC
    """
    )


def _mermaid_instance_sisal(n: int = 305) -> str:
    """SISAL named-individual instance diagram."""
    return textwrap.dedent(
        f"""\
        flowchart LR

            COLLECTION["geo:FeatureCollection
            geolod:SISAL_Cave_Collection
            {n} members"]

            CAVE["Cave / geo:Feature
            geolod:Cave_site_0001 ...
            crm:E53_Place · crm:E27_Site"]

            GEOM["sf:Point
            geolod:Cave_site_0001_Geometry ...
            asWKT: EPSG:4326 POINT(lon lat)"]

            SPEL["Speleothem
            geolod:Speleothem_entity_XXXX"]

            SSE["SpeleothemSamplingEvent"]

            OBS["SpeleothemObservation
            geolod:Obs_d18O_XXXX
            geolod:Obs_d13C_XXXX"]

            MTYPE_D18O["MeasurementType (shared core)
            geolod:MeasurementType_d18O"]

            MTYPE_D13C["MeasurementType
            geolod:MeasurementType_d13C"]

            UTHC["UThChronology
            geolod:UThChronology_entity_XXXX"]

            MEDIAN["RollingMedianFilter
            geolod:RollingMedian_wN"]

            SAVGOL["SavitzkyGolayFilter
            geolod:SavitzkyGolay_wN_pP"]

            DATASRC["DataSource
            geolod:SISALv3_DataSource"]

            COLLECTION -->|"rdfs:member"| CAVE
            CAVE       -->|"geo:hasGeometry"| GEOM
            SPEL       -->|"geolod:collectedFrom"| CAVE
            SSE        -->|"geolod:tookPlaceAt"| CAVE
            SSE        -->|"geolod:removedSample"| SPEL
            OBS        -->|"sosa:hasFeatureOfInterest"| SPEL
            OBS        -->|"geolod:measurementType"| MTYPE_D18O
            OBS        -->|"geolod:measurementType"| MTYPE_D13C
            OBS        -->|"geolod:ageChronologySpeleothem"| UTHC
            OBS        -->|"geolod:smoothingMethod_median"| MEDIAN
            OBS        -->|"geolod:smoothingMethod_savgol"| SAVGOL
            OBS        -->|"prov:wasDerivedFrom"| DATASRC
    """
    )


def write_mermaid(
    outdir: str,
    rolling_window: int = 11,
    sg_window: int = 11,
    sg_poly: int = 2,
    n_sisal_sites: int = 305,
) -> dict[str, str]:
    """
    Write all three Mermaid diagram files to *outdir*.

    Files
    -----
    mermaid_taxonomy.mermaid         combined class hierarchy (Core + EPICA + SISAL)
    mermaid_instance_epica.mermaid   EPICA named-individual instance diagram
    mermaid_instance_sisal.mermaid   SISAL named-individual instance diagram

    Parameters
    ----------
    outdir          : output directory (created if absent)
    rolling_window  : rolling-median window size (from EPICA/SISAL config)
    sg_window       : Savitzky-Golay window size
    sg_poly         : Savitzky-Golay polynomial order
    n_sisal_sites   : number of SISAL cave sites (for collection label)

    Returns dict of {filename: full_path}.
    """
    os.makedirs(outdir, exist_ok=True)
    diagrams = {
        "mermaid_taxonomy.mermaid": MERMAID_TAXONOMY,
        "mermaid_instance_epica.mermaid": _mermaid_instance_epica(
            rolling_window, sg_window, sg_poly
        ),
        "mermaid_instance_sisal.mermaid": _mermaid_instance_sisal(n_sisal_sites),
    }
    paths: dict[str, str] = {}
    for filename, content in diagrams.items():
        path = os.path.join(outdir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        print(f"  ✓ Mermaid       : {path}")
        paths[filename] = path
    return paths


# ===========================================================================
# 6.  SELF-TEST  —  python ontology/geo_lod_utils.py
# ===========================================================================

if __name__ == "__main__":
    import tempfile

    print("geo_lod_utils.py — self-test")
    print("=" * 60)

    # wkt_point
    w = wkt_point(123.35, -75.1)
    assert w == f"<{CRS_WGS84}> POINT(123.350000 -75.100000)", repr(w)
    print(f"✓ wkt_point              : {w}")

    # _ensure_crs — injection
    raw = "POINT(31.9333 41.4167)"
    assert _ensure_crs(raw).startswith(f"<{CRS_WGS84}>")
    print(f"✓ _ensure_crs (inject)   : {_ensure_crs(raw)[:55]}...")

    # _ensure_crs — idempotent
    already = f"<{CRS_WGS84}> POINT(0 0)"
    assert _ensure_crs(already) == already
    print(f"✓ _ensure_crs (idempotent): OK")

    if RDF_AVAILABLE:
        GEOLOD = Namespace(NS["geolod"])
        GEO = Namespace(NS["geo"])
        SF = Namespace(NS["sf"])
        CRM = Namespace(NS["crm"])

        # get_graph
        g = get_graph()
        bound = {p for p, _ in g.namespaces()}
        for prefix in NS:
            assert prefix in bound, f"Missing namespace: {prefix}"
        print(f"✓ get_graph              : {len(bound)} namespaces bound")

        # add_geo_site (EPICA pattern)
        add_geo_site(
            g,
            site_uri=GEOLOD["EpicaDomeC_Site"],
            geom_uri=GEOLOD["EpicaDomeC_Geometry"],
            label="EPICA Dome C, East Antarctica",
            lon=123.35,
            lat=-75.1,
            extra_types=[GEOLOD["DrillingSite"]],
        )
        assert (GEOLOD["EpicaDomeC_Site"], RDF.type, GEO["Feature"]) in g
        assert (GEOLOD["EpicaDomeC_Site"], RDF.type, CRM["E53_Place"]) in g
        assert (GEOLOD["EpicaDomeC_Site"], RDF.type, CRM["E27_Site"]) in g
        assert (GEOLOD["EpicaDomeC_Site"], RDF.type, GEOLOD["DrillingSite"]) in g
        assert (GEOLOD["EpicaDomeC_Geometry"], RDF.type, SF["Point"]) in g
        # geo:Geometry must NOT be asserted (CI pattern — subclass entailment only)
        assert (GEOLOD["EpicaDomeC_Geometry"], RDF.type, GEO["Geometry"]) not in g
        wkts = list(g.objects(GEOLOD["EpicaDomeC_Geometry"], GEO["asWKT"]))
        assert len(wkts) == 1 and str(wkts[0]).startswith(f"<{CRS_WGS84}>")
        print(f"✓ add_geo_site           : WKT = {str(wkts[0])[:55]}...")

        # add_geo_site_from_wkt — CRS injection (SISAL pattern)
        g2 = get_graph()
        add_geo_site_from_wkt(
            g2,
            site_uri=GEOLOD["Cave_site_0001"],
            geom_uri=GEOLOD["Cave_site_0001_Geometry"],
            label="Bittoo Cave",
            wkt="POINT(31.9333 41.4167)",
            extra_types=[GEOLOD["Cave"]],
        )
        wkts2 = list(g2.objects(GEOLOD["Cave_site_0001_Geometry"], GEO["asWKT"]))
        assert str(wkts2[0]).startswith(f"<{CRS_WGS84}>")
        print(f"✓ add_geo_site_from_wkt  : CRS injected OK")

        # add_geo_site_from_wkt — no double CRS prefix when already present
        g3 = get_graph()
        prefixed_wkt = f"<{CRS_WGS84}> POINT(10.0 20.0)"
        add_geo_site_from_wkt(
            g3,
            site_uri=GEOLOD["Cave_site_0002"],
            geom_uri=GEOLOD["Cave_site_0002_Geometry"],
            label="Already prefixed",
            wkt=prefixed_wkt,
        )
        wkts3 = list(g3.objects(GEOLOD["Cave_site_0002_Geometry"], GEO["asWKT"]))
        # The stored value must equal the input exactly (no double prefix)
        assert str(wkts3[0]) == prefixed_wkt, f"double prefix! got: {wkts3[0]!r}"
        assert (
            str(wkts3[0]).count(f"<{CRS_WGS84}>") == 1
        ), "CRS prefix appears more than once"
        print(f"✓ add_geo_site_from_wkt  : no double CRS prefix")

        # add_feature_collection
        add_feature_collection(
            g,
            collection_uri=GEOLOD["SISAL_Cave_Collection"],
            label="SISAL Cave Collection",
            members=[GEOLOD["Cave_site_0001"], GEOLOD["Cave_site_0002"]],
        )
        assert (
            GEOLOD["SISAL_Cave_Collection"],
            RDF.type,
            GEO["FeatureCollection"],
        ) in g
        print(f"✓ add_feature_collection : OK")

    else:
        print("⚠  rdflib not installed — graph tests skipped")

    # write files — test in temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        core = write_geo_lod_core(tmpdir)
        assert os.path.exists(core) and os.path.getsize(core) > 1000
        print(f"✓ write_geo_lod_core     : {os.path.getsize(core):,} bytes")

        paths = write_mermaid(tmpdir, rolling_window=11, sg_window=11, sg_poly=2)
        for name, path in paths.items():
            assert os.path.getsize(path) > 100
            print(f"✓ {name}: {os.path.getsize(path):,} bytes")

    print()
    print("=" * 60)
    print("All tests passed.")
    print()

    # ── Write files next to this script (ontology/ folder) ──────────────────
    here = os.path.dirname(os.path.abspath(__file__))
    print(f"Writing ontology files to: {here}")
    write_geo_lod_core(here)
    write_mermaid(here, rolling_window=11, sg_window=11, sg_poly=2)
    print("Done.")
