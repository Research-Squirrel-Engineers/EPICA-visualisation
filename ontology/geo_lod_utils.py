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

import hashlib
import os
import textwrap
from typing import Iterable

# ---------------------------------------------------------------------------
# Optional rdflib import
# ---------------------------------------------------------------------------
try:
    from rdflib import Graph, Namespace, URIRef, Literal
    from rdflib.namespace import RDF, RDFS, OWL, XSD

    PROV = Namespace(NS_PROV := "http://www.w3.org/ns/prov#")
    DCT = Namespace("http://purl.org/dc/terms/")

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
# 3b.  PROVENANCE  — release date and content fingerprint
# ===========================================================================
#
# Ein erzeugter Datensatz gilt genau für den Stand aus Eingabedaten und
# Generator-Code, aus dem er entstanden ist. Sichtbar gemacht wird das nicht
# über die Uhr des Rechners - die sagt nur, wann jemand das Skript gestartet
# hat -, sondern über einen Fingerabdruck über genau diese Dateien. Er ändert
# sich, wenn sich Daten oder Code ändern, und sonst nie. Zwei Läufe ohne
# Änderung sind damit byte-identisch.
#
# Daneben steht ein von Hand gepflegtes Release-Datum für dct:created, weil
# in einem Katalog ein Datum stehen soll und kein Hash.

# Lives in geo_lod_release.py so that geo_lod_figures can read it without
# importing rdflib through this module. Re-exported here, because every
# generator already imports it from geo_lod_utils.
from geo_lod_release import GEO_LOD_RELEASE  # noqa: E402,F401

FINGERPRINT_LENGTH: int = 12


def file_sha256(path) -> str:
    """Vollständiger SHA-256 einer Datei, hexadezimal."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def content_fingerprint(paths) -> str:
    """Fingerabdruck über mehrere Dateien: 'sha256:xxxxxxxxxxxx'.

    Gehasht wird über Dateiname und Inhalt, in sortierter Reihenfolge -
    damit hängt das Ergebnis weder an der Reihenfolge der Argumente noch am
    absoluten Pfad, also auch nicht daran, wo das Repo liegt.
    """
    entries = sorted(
        (os.path.basename(str(p)), file_sha256(p))
        for p in paths
        if os.path.exists(p)
    )
    h = hashlib.sha256()
    for name, digest in entries:
        h.update(name.encode("utf-8"))
        h.update(digest.encode("ascii"))
    return "sha256:" + h.hexdigest()[:FINGERPRINT_LENGTH]


def add_generation_provenance(
    g,
    dataset_uri,
    activity_uri,
    inputs,
    agents=(),
    label: str = "generation",
) -> str:
    """Hängt die Erzeugungs-Provenienz an einen Datensatzknoten und gibt den
    Fingerabdruck zurück.

    Modelliert wird: der Datensatz entstand durch eine Aktivität, die die
    aufgeführten Dateien benutzt hat; jede davon trägt ihre eigene Prüfsumme.
    Wer diesen TTL-Dump vor sich hat, kann damit prüfen, ob er zu den Daten
    und dem Code passt, aus denen er hervorgegangen ist.

    Bewusst ohne prov:startedAtTime / prov:endedAtTime: die Laufzeit eines
    Konvertierungsskripts ist keine Aussage über die Daten.
    """
    fingerprint = content_fingerprint(inputs)

    g.add((dataset_uri, RDF.type, PROV.Entity))
    g.add((dataset_uri, OWL.versionInfo, Literal(fingerprint)))
    g.add((dataset_uri, DCT.created, Literal(GEO_LOD_RELEASE, datatype=XSD.date)))
    g.add((dataset_uri, PROV.wasGeneratedBy, activity_uri))

    g.add((activity_uri, RDF.type, PROV.Activity))
    g.add((activity_uri, RDFS.label, Literal(label, lang="en")))
    g.add((activity_uri, OWL.versionInfo, Literal(fingerprint)))

    for agent in agents:
        g.add((activity_uri, PROV.wasAssociatedWith, agent))

    for path in inputs:
        if not os.path.exists(path):
            continue
        name = os.path.basename(str(path))
        input_uri = URIRef(f"{activity_uri}_input_{name.replace('.', '_')}")
        g.add((input_uri, RDF.type, PROV.Entity))
        g.add((input_uri, RDFS.label, Literal(name)))
        g.add((input_uri, DCT.identifier, Literal("sha256:" + file_sha256(path))))
        g.add((activity_uri, PROV.used, input_uri))

    return fingerprint


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
    @prefix crm:        <http://www.cidoc-crm.org/cidoc-crm/> .
    @prefix crmsci:     <http://www.ics.forth.gr/isl/CRMsci/> .
    @prefix crmarchaeo: <http://www.cidoc-crm.org/extensions/crmarchaeo/> .
    @prefix sosa:    <http://www.w3.org/ns/sosa/> .
    @prefix qudt:    <http://qudt.org/schema/qudt/> .
    @prefix unit:    <http://qudt.org/vocab/unit/> .
    @prefix prov:    <http://www.w3.org/ns/prov#> .
    @prefix dct:     <http://purl.org/dc/terms/> .
    @prefix dcat:    <http://www.w3.org/ns/dcat#> .
    @prefix skos:    <http://www.w3.org/2004/02/skos/core#> .
    @prefix time:    <http://www.w3.org/2006/time#> .
    @prefix geolod:  <http://w3id.org/geo-lod/> .

    <http://w3id.org/geo-lod/>
        a owl:Ontology ;
        rdfs:label   "geo-lod Core Ontology"@en ;
        rdfs:comment "Shared vocabulary for EPICA ice-core and SISAL speleothem  palaeoclimate Linked Data. Domain extensions import this file."@en ;
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
        rdfs:comment "A geographically identified location from which palaeoclimate  proxy material was obtained (ice-core drilling site or speleothem cave)."@en .

    # -- Borehole  (a point inside a SamplingLocation, EPICA) --

    geolod:Borehole
        a owl:Class ;
        rdfs:subClassOf geolod:SamplingLocation ;
        rdfs:label   "Borehole"@en ;
        rdfs:comment "The individual hole from which core material was recovered,  inside the wider site. Kept apart from the site because published records of  the same drilling can carry different coordinates, and averaging them would  invent a position that no source states."@en .

    # -- PalaeoclimateSample  (superclass: IceCore in EPICA, Speleothem in SISAL) --

    geolod:PalaeoclimateSample
        a owl:Class ;
        rdfs:subClassOf crm:E18_Physical_Thing ;
        rdfs:subClassOf sosa:Sample ;
        rdfs:label   "Palaeoclimate Sample"@en ;
        rdfs:comment "A physical archive recording a palaeoclimate signal  (ice core or speleothem)."@en .

    # -- SampleSection  (a bounded piece of a sample, EPICA deuterium) --

    geolod:SampleSection
        a owl:Class ;
        rdfs:subClassOf geolod:PalaeoclimateSample ;
        rdfs:label   "Sample Section"@en ;
        rdfs:comment "A bounded interval cut from a larger archive, over which a  single value was measured or averaged. Minted only where the source states  the interval; where a record gives a single depth and nothing else, the  observation hangs on the whole archive instead of on an invented section."@en .

    # -- PalaeoclimateObservation  (superclass for all measurement types) --

    geolod:PalaeoclimateObservation
        a owl:Class ;
        rdfs:subClassOf crmsci:S4_Observation ;
        rdfs:subClassOf sosa:Observation ;
        rdfs:label   "Palaeoclimate Observation"@en ;
        rdfs:comment "A single measured value (e.g. δ¹⁸O, δ¹³C, CH₄) at a  known depth or age within a palaeoclimate archive."@en .

    # -- ObservableProperty --

    geolod:ObservableProperty
        a owl:Class ;
        rdfs:subClassOf sosa:ObservableProperty ;
        rdfs:subClassOf crmsci:S9_Property_Type ;
        rdfs:label   "Observable Property"@en ;
        rdfs:comment "A measurable geochemical or physical property of a  palaeoclimate sample (e.g. δ¹⁸O, δ¹³C, CH₄ concentration)."@en .

    geolod:Delta18OProperty
        a owl:Class ;
        rdfs:subClassOf geolod:ObservableProperty ;
        rdfs:label   "δ¹⁸O Property"@en ;
        rdfs:comment "Stable oxygen isotope ratio (δ¹⁸O), shared by both  ice-core and speleothem records."@en .

    # -- Chronology  (superclass: IceCoreChronology in EPICA, UThChronology in SISAL) --

    geolod:Chronology
        a owl:Class ;
        rdfs:subClassOf crmsci:S4_Observation ;
        rdfs:subClassOf time:TRS ;
        rdfs:label   "Chronology"@en ;
        rdfs:comment "A depth-age model assigning calendar ages to positions  within a palaeoclimate archive. Also a temporal reference system: an age is only meaningful together with the model that produced it, so the same node serves as the value of geolod:ageChronology on an observation and of time:hasTRS on its time position."@en .

    geolod:IceCoreChronology
        a owl:Class ;
        rdfs:subClassOf geolod:Chronology ;
        rdfs:label   "Ice Core Chronology"@en ;
        rdfs:comment "A depth-age model for an ice core, such as EDC2, EDC3 or  AICC2023. Ice cores yield two ages per depth - one for the ice, one for the  air trapped in it - so a chronology that publishes both appears here as two  individuals rather than one."@en .

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
        rdfs:comment "A numerical method applied to reduce high-frequency  noise in a palaeoclimate time series."@en .

    geolod:RollingMedianFilter
        a owl:Class ;
        rdfs:subClassOf geolod:SmoothingMethod ;
        rdfs:label   "Rolling Median Filter"@en ;
        rdfs:comment "Non-parametric smoother computing the median within  a sliding window of fixed width."@en .

    geolod:SavitzkyGolayFilter
        a owl:Class ;
        rdfs:subClassOf geolod:SmoothingMethod ;
        rdfs:label   "Savitzky-Golay Filter"@en ;
        rdfs:comment "Polynomial least-squares smoothing filter preserving  higher signal moments."@en .

    # -- ArchaeologicalContext vocabulary (SISAL archaeological enrichment) ---

    geolod:ArchaeologicalContext
        a owl:Class ;
        rdfs:subClassOf crm:E55_Type ;
        rdfs:label   "Archaeological Context"@en ;
        rdfs:comment "Controlled vocabulary class for broader cultural-temporal  context categories used in archaeological cave site classification."@en .

    geolod:PalaeolithicContext
        a geolod:ArchaeologicalContext, owl:NamedIndividual ;
        rdfs:label   "Palaeolithic Context"@en .

    geolod:PrehistoricContext
        a geolod:ArchaeologicalContext, owl:NamedIndividual ;
        rdfs:label   "Prehistoric Context"@en .

    geolod:PalaeontologicalContext
        a geolod:ArchaeologicalContext, owl:NamedIndividual ;
        rdfs:label   "Palaeontological Context"@en .

    geolod:HistoricContext
        a geolod:ArchaeologicalContext, owl:NamedIndividual ;
        rdfs:label   "Historic Context"@en .

    geolod:MesoamericanContext
        a geolod:ArchaeologicalContext, owl:NamedIndividual ;
        rdfs:label   "Mesoamerican Context"@en .

    # The four properties of the curated archaeological reading. They were
    # declared in sisal_ontology.ttl while only caves carried them; since the
    # CI strand annotates findspots the same way, they belong here, and their
    # domain is the class both branches already share. A domain of
    # geolod:Cave would have made every annotated CI findspot a cave under
    # any reasoner - the enrichment says what a place is known for, not what
    # kind of place it is.

    geolod:screenedForArchaeology
        a owl:DatatypeProperty ;
        rdfs:domain  crm:E27_Site ;
        rdfs:range   xsd:boolean ;
        rdfs:label   "screened for archaeology"@en ;
        rdfs:comment "True where geo-lod checked this site against the archaeological literature, whatever the outcome. Absence of the property means the site has not been looked at, which is a different statement from a negative result and has to stay distinguishable."@en .

    geolod:archaeologicalCategory
        a owl:DatatypeProperty ;
        rdfs:domain  crm:E27_Site ;
        rdfs:range   xsd:string ;
        rdfs:label   "archaeological category"@en ;
        rdfs:comment "Free-text classification of the archaeological character, e.g. 'Palaeolithic Art', 'Cave Site'."@en .

    geolod:archaeologicalBroaderContext
        a owl:ObjectProperty ;
        rdfs:domain  crm:E27_Site ;
        rdfs:range   geolod:ArchaeologicalContext ;
        rdfs:label   "archaeological broader context"@en .

    geolod:archaeologicalConfidence
        a owl:DatatypeProperty ;
        rdfs:domain  crm:E27_Site ;
        rdfs:range   xsd:string ;
        rdfs:label   "archaeological confidence"@en ;
        rdfs:comment "Confidence of the archaeological attribution: high, medium or low."@en .

    # -- CI Findspots  (Campanian Ignimbrite tephra documentation sites) -------

    geolod:CIFindspot
        a owl:Class ;
        rdfs:subClassOf crm:E27_Site ;
        rdfs:label   "Campanian Ignimbrite Findspot"@en ;
        rdfs:comment "A documented findspot of Campanian Ignimbrite tephra. Modelled as crm:E27_Site so that geometry, name, and finds can attach via standard CRM patterns."@en .

    geolod:CIArchaeologicalSite
        a owl:Class ;
        rdfs:subClassOf geolod:CIFindspot ;
        rdfs:subClassOf crmarchaeo:A2_Stratigraphic_Volume_Unit ;
        rdfs:label   "CI Archaeological Site"@en ;
        rdfs:comment "A CI Findspot that also carries confirmed or probable archaeological evidence (artefacts, occupation layers, stratified contexts). Mirrors the ArchaeologicalCaveSite pattern in the SISAL extension."@en .

    # -- DataSource --

    geolod:DataSource
        a owl:Class ;
        rdfs:subClassOf crm:E73_Information_Object ;
        rdfs:subClassOf prov:Entity ;
        rdfs:label   "Data Source"@en ;
        rdfs:comment "A citable source (database, repository, publication)  from which palaeoclimate observations were obtained."@en .

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
        rdfs:comment "Stable oxygen isotope ratio — shared observable property  used in both ice-core and speleothem observations."@en .

    geolod:MeasurementType_d18O
        a geolod:MeasurementType, owl:NamedIndividual ;
        rdfs:label   "δ¹⁸O measurement"@en .

    # ==========================================================================
    # MARINE ISOTOPE STAGES
    # Schema for the controlled vocabulary under
    # <http://w3id.org/geo-lod/vocab/mis/>, which is generated from the
    # primary sources by ontology/build_mis_vocab.py.
    # ==========================================================================

    geolod:MarineIsotopeStage
        a owl:Class ;
        rdfs:subClassOf skos:Concept ;
        rdfs:subClassOf crm:E4_Period ;
        rdfs:label   "Marine Isotope Stage"@en ;
        rdfs:comment "A stage of the marine oxygen-isotope record, used as a chronostratigraphic subdivision of the Quaternary and late Neogene."@en .

    geolod:MarineIsotopeSubstage
        a owl:Class ;
        rdfs:subClassOf geolod:MarineIsotopeStage ;
        rdfs:label   "Marine Isotope Substage"@en ;
        rdfs:comment "A lettered subdivision of a Marine Isotope Stage after Railsback et al. (2015). Substages label excursions rather than the transitions between them, so their boundaries are gradational."@en .

    geolod:MISAttributeAssignment
        a owl:Class ;
        rdfs:subClassOf crm:E13_Attribute_Assignment ;
        rdfs:label   "MIS Attribute Assignment"@en ;
        rdfs:comment "Assigns a boundary age, an excursion peak or a climate mode to a Marine Isotope Stage on the authority of one source. Competing readings of the same boundary are kept as separate assignments rather than harmonised."@en .

    geolod:ClimateMode
        a owl:Class ;
        rdfs:subClassOf crm:E55_Type ;
        rdfs:label   "Climate Mode"@en ;
        rdfs:comment "Classifies a Marine Isotope Stage as warm or cold."@en .

    geolod:ClimateMode_Warm
        a geolod:ClimateMode, owl:NamedIndividual ;
        rdfs:label   "warm"@en .

    geolod:ClimateMode_Cold
        a geolod:ClimateMode, owl:NamedIndividual ;
        rdfs:label   "cold"@en .

    geolod:AssignedPropertyType
        a owl:Class ;
        rdfs:subClassOf crm:E55_Type ;
        rdfs:label   "Assigned Property Type"@en ;
        rdfs:comment "The kind of property an attribute assignment establishes (crm:P177)."@en .

    geolod:PeriodBeginning
        a geolod:AssignedPropertyType, owl:NamedIndividual ;
        rdfs:label   "beginning of period"@en ;
        rdfs:comment "The older boundary of a period, in ka BP."@en .

    geolod:PeriodEnd
        a geolod:AssignedPropertyType, owl:NamedIndividual ;
        rdfs:label   "end of period"@en ;
        rdfs:comment "The younger boundary of a period, in ka BP."@en .

    geolod:ExcursionPeak
        a geolod:AssignedPropertyType, owl:NamedIndividual ;
        rdfs:label   "excursion peak"@en ;
        rdfs:comment "The age of the isotopic maximum or minimum a substage labels."@en .

    geolod:ClimateModeType
        a geolod:AssignedPropertyType, owl:NamedIndividual ;
        rdfs:label   "climate mode"@en ;
        rdfs:comment "Warm or cold classification of a stage."@en .

    geolod:AssignmentStatus
        a owl:Class ;
        rdfs:subClassOf crm:E55_Type ;
        rdfs:label   "Assignment Status"@en ;
        rdfs:comment "States whether an attribute assignment is the one geo-lod follows, or a competing reading kept alongside it. Competing readings are never dropped; this only records which one leads."@en .

    geolod:LeadingAssignment
        a geolod:AssignmentStatus, owl:NamedIndividual ;
        rdfs:label   "leading assignment"@en ;
        rdfs:comment "The assignment geo-lod follows for this property. Filtering on it yields one consistent value per property without knowing where a source's coverage ends."@en .

    geolod:AlternativeAssignment
        a geolod:AssignmentStatus, owl:NamedIndividual ;
        rdfs:label   "alternative assignment"@en ;
        rdfs:comment "A competing reading from another source, kept for comparison but not followed."@en .

    geolod:assignmentStatus
        a owl:ObjectProperty ;
        rdfs:domain  crm:E13_Attribute_Assignment ;
        rdfs:range   geolod:AssignmentStatus ;
        rdfs:label   "assignment status"@en .

    geolod:leadingSource
        a owl:ObjectProperty ;
        rdfs:range   geolod:DataSource ;
        rdfs:label   "leading source"@en ;
        rdfs:comment "The source geo-lod follows for this entity where the sources disagree."@en .

    geolod:coverageOldestAgeKaBP
        a owl:DatatypeProperty ;
        rdfs:domain  geolod:DataSource ;
        rdfs:range   xsd:decimal ;
        rdfs:label   "oldest age covered (ka BP)"@en ;
        rdfs:comment "Oldest age a source reaches. Beyond it another source has to lead."@en .

    geolod:climateMode
        a owl:ObjectProperty ;
        rdfs:domain  geolod:MarineIsotopeStage ;
        rdfs:range   geolod:ClimateMode ;
        rdfs:label   "climate mode"@en ;
        rdfs:comment "Materialised warm/cold classification. The assignment carrying the source is kept alongside as a geolod:MISAttributeAssignment."@en .

    geolod:beginAgeKaBP
        a owl:DatatypeProperty ;
        rdfs:subPropertyOf geolod:ageKaBP ;
        rdfs:range   xsd:decimal ;
        rdfs:label   "beginning age (ka BP)"@en ;
        rdfs:comment "Materialised older boundary of a period, taken from the leading source. Competing values are reachable through the attribute assignments."@en .

    geolod:endAgeKaBP
        a owl:DatatypeProperty ;
        rdfs:subPropertyOf geolod:ageKaBP ;
        rdfs:range   xsd:decimal ;
        rdfs:label   "end age (ka BP)"@en ;
        rdfs:comment "Materialised younger boundary of a period, taken from the leading source."@en .

    geolod:peakAgeKaBP
        a owl:DatatypeProperty ;
        rdfs:subPropertyOf geolod:ageKaBP ;
        rdfs:range   xsd:decimal ;
        rdfs:label   "peak age (ka BP)"@en ;
        rdfs:comment "Materialised age of the excursion a substage labels."@en .

    # ==========================================================================
    # STAGE MEMBERSHIP AND STAGE BOUNDARIES IN DEPTH
    # ==========================================================================
    # Two further things get assigned once observations exist, and both are
    # kept as assignments rather than as bare properties, because both depend
    # on a choice that has to stay readable.
    #
    # Membership: which stage an observation falls in follows from its age,
    # and its age follows from the chronology of its record. Two records of
    # the same core can therefore place the same depth in different stages.
    #
    # Boundary in depth: a stage boundary is published as an age. Turning it
    # into a depth means interpolating in one particular depth-age model, so
    # the same boundary sits at different depths depending on which record
    # was used. That is a derived statement about a model, not a measurement,
    # and it says so.

    geolod:MISMembershipAssignment
        a owl:Class ;
        rdfs:subClassOf crm:E13_Attribute_Assignment ;
        rdfs:label   "MIS Membership Assignment"@en ;
        rdfs:comment "Places an observation in a Marine Isotope Stage, on the  authority of one boundary source and through one chronology. Deliberately not  a subclass of geolod:MISAttributeAssignment: that one assigns properties *to*  a stage, this one assigns a stage *to* something else, and the two carry  different constraints."@en .

    geolod:DepthPosition
        a owl:Class ;
        rdfs:subClassOf crm:E54_Dimension ;
        rdfs:label   "Depth Position"@en ;
        rdfs:comment "A position along the depth axis of an archive, in metres.  Used where a value published as an age has been carried into depth through a  depth-age model."@en .

    geolod:MISMembership
        a geolod:AssignedPropertyType, owl:NamedIndividual ;
        rdfs:label   "marine isotope stage membership"@en ;
        rdfs:comment "The stage an observation falls in."@en .

    geolod:MISBoundaryDepth
        a geolod:AssignedPropertyType, owl:NamedIndividual ;
        rdfs:label   "stage boundary in depth"@en ;
        rdfs:comment "The depth at which a stage boundary falls in a given  archive, obtained by interpolation in a given chronology."@en .

    geolod:LinearInterpolation
        a crmsci:S6_Data_Evaluation, owl:NamedIndividual ;
        rdfs:label   "linear interpolation between neighbouring measurements"@en ;
        rdfs:comment "Straight-line interpolation between the two measured  depth-age pairs bracketing the target age. Not applied beyond the outermost  measurement of a record: an extrapolated boundary would state a depth the  record does not reach."@en .

    geolod:interpolationMethod
        a owl:ObjectProperty ;
        rdfs:range   crmsci:S6_Data_Evaluation ;
        rdfs:label   "interpolation method"@en ;
        rdfs:comment "The method by which a value was obtained between two  measured points. Deliberately not the same property as the smoothing methods:  smoothing rewrites a measured value, interpolation produces one where none  was measured, and a consumer has to be able to tell the two apart."@en .

    geolod:inChronology
        a owl:ObjectProperty ;
        rdfs:range   geolod:Chronology ;
        rdfs:label   "in chronology"@en ;
        rdfs:comment "The depth-age model an assignment depends on. Without it a  derived age or depth cannot be compared with one from another record."@en .

    geolod:hasTimePosition
        a owl:ObjectProperty ;
        rdfs:range   time:TimePosition ;
        rdfs:label   "has time position"@en ;
        rdfs:comment "The age of an observation as a time position, carrying both  the numeric value and the reference system it belongs to. Kept alongside the  plain geolod:ageKaBP literal, which stays queryable without a join."@en .

    geolod:atDepth_m
        a owl:DatatypeProperty ;
        rdfs:range   xsd:decimal ;
        rdfs:label   "at depth (m)"@en ;
        rdfs:comment "Depth below the surface of the archive, in metres. In an  ice core this is the measured quantity and the age is the derived one."@en ;
        qudt:unit    unit:M .

    geolod:depthTop_m
        a owl:DatatypeProperty ;
        rdfs:subPropertyOf geolod:atDepth_m ;
        rdfs:range   xsd:decimal ;
        rdfs:label   "top depth (m)"@en ;
        rdfs:comment "Upper, younger bound of a sample section."@en .

    geolod:depthBottom_m
        a owl:DatatypeProperty ;
        rdfs:subPropertyOf geolod:atDepth_m ;
        rdfs:range   xsd:decimal ;
        rdfs:label   "bottom depth (m)"@en ;
        rdfs:comment "Lower, older bound of a sample section."@en .

    geolod:ageMinKaBP
        a owl:DatatypeProperty ;
        rdfs:subPropertyOf geolod:ageKaBP ;
        rdfs:range   xsd:decimal ;
        rdfs:label   "youngest age of interval (ka BP)"@en ;
        rdfs:comment "Age at the top of the interval a value was averaged over."@en .

    geolod:ageMaxKaBP
        a owl:DatatypeProperty ;
        rdfs:subPropertyOf geolod:ageKaBP ;
        rdfs:range   xsd:decimal ;
        rdfs:label   "oldest age of interval (ka BP)"@en ;
        rdfs:comment "Age at the base of the interval a value was averaged over."@en .

    geolod:standardDeviation
        a owl:DatatypeProperty ;
        rdfs:range   xsd:decimal ;
        rdfs:label   "standard deviation"@en ;
        rdfs:comment "Reported standard deviation of a measured value, in the  unit of that value."@en .

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
        rdfs:comment "A single location in n-dimensional space (OGC Simple Features).  Subclass of geo:Geometry."@en .

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

    crm:E4_Period
        rdfs:label   "Period"@en ;
        rdfs:comment "A set of coherent phenomena or cultural manifestations bounded in time and space (CIDOC-CRM E4)."@en .

    crm:E13_Attribute_Assignment
        rdfs:label   "Attribute Assignment"@en ;
        rdfs:comment "An act of assigning a property to an entity, on the authority of some source (CIDOC-CRM E13)."@en .

    sosa:Observation        rdfs:label "Observation"@en .
    sosa:ObservableProperty rdfs:label "Observable Property"@en .
    sosa:Sample             rdfs:label "Sample"@en .
    prov:Entity             rdfs:label "Entity"@en .
    skos:Concept            rdfs:label "Concept"@en .
    skos:ConceptScheme      rdfs:label "Concept Scheme"@en .
    time:TimePosition       rdfs:label "Time Position"@en .
    time:TRS                rdfs:label "Temporal Reference System"@en .
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
    %% External Ontologies
    subgraph EXT["External Ontologies"]
        direction TB
        
        subgraph CRM["CIDOC-CRM"]
            direction TB
            CE53["crm:E53_Place"]
            CE27["crm:E27_Site"]
            CE22["crm:E22_Human-Made_Object"]
            CE7["crm:E7_Activity"]
        end
        
        subgraph CRMSCI["CRMsci"]
            direction TB
            CS4["crmsci:S4_Observation"]
            CS1["crmsci:S1_Matter_Removal"]
            CS6["crmsci:S6_Data_Evaluation"]
            CS9["crmsci:S9_Property_Type"]
        end
        
        subgraph SOSA["SOSA"]
            direction TB
            SO["sosa:Observation"]
            SS["sosa:Sample"]
            SP["sosa:ObservableProperty"]
        end
        
        subgraph GEO["GeoSPARQL"]
            GF["geo:Feature"]
        end
        
        subgraph PROV["PROV-O"]
            PE["prov:Entity"]
        end
    end
    
    %% Core Ontology
    subgraph CORE["geo-lod Core Ontology"]
        direction TB
        PALOBS["PalaeoclimateObservation"]
        PALSAMPLE["PalaeoclimateSample"]
        SAMPLINGLOC["SamplingLocation"]
        CHRONO["Chronology"]
        OBSPROP["ObservableProperty"]
        DATASRC["DataSource"]
        MTYPE["MeasurementType"]
        SMOOTH["SmoothingMethod"]
    end
    
    %% EPICA Extension
    subgraph EPICA["EPICA Ice Core Extension"]
        direction TB
        ICEOBS["IceCoreObservation"]
        CH4OBS["CH4Observation"]
        D18OOBS["Delta18OObservation"]
        ICECORE["IceCore"]
        DRILLSITE["DrillingSite"]
        DRILLCAMP["DrillingCampaign"]
        ICECHRONO["IceCoreChronology"]
    end
    
    %% SISAL Extension
    subgraph SISAL["SISAL Speleothem Extension"]
        direction TB
        SPELOBS["SpeleothemObservation"]
        D18OSPELOBS["Delta18OSpeleothemObservation"]
        D13COBS["Delta13CSpeleothemObservation"]
        SPEL["Speleothem"]
        CAVE["Cave"]
        ARCHCAVE["ArchaeologicalCaveSite"]
        SSE["SpeleothemSamplingEvent"]
        UTHCHRONO["UThChronology"]
    end

    %% CI Extension
    subgraph CI["CI Campanian Ignimbrite Extension"]
        direction TB
        CIFINDSPOT["CIFindspot"]
        CIARCHSITE["CIArchaeologicalSite"]
        CIVOLEVENT["CIVolcanicEvent"]
        CITEPHRA["CITephraDeposit"]
    end

    subgraph CRMEXT["CRMarchaeo / CRMgeo"]
        direction TB
        CA2["crmarchaeo:A2_Stratigraphic_Volume_Unit"]
        CGP6["crmgeo:SP6_Declarative_Place"]
        CE5["crm:E5_Event"]
        CE26["crm:E26_Physical_Feature"]
    end

    %% External to Core relationships
    SO -.-> PALOBS
    CS4 -.-> PALOBS
    SS -.-> PALSAMPLE
    CE53 -.-> SAMPLINGLOC
    CE27 -.-> SAMPLINGLOC
    GF -.-> SAMPLINGLOC
    SP -.-> OBSPROP
    CS9 -.-> OBSPROP
    CS9 -.-> MTYPE
    CS6 -.-> CHRONO
    CS6 -.-> SMOOTH
    PE -.-> DATASRC
    CE7 -.-> DRILLCAMP
    CS1 -.-> DRILLCAMP
    CE7 -.-> SSE
    CS1 -.-> SSE
    
    %% Core to Extensions
    PALOBS --> ICEOBS
    PALOBS --> SPELOBS
    ICEOBS --> CH4OBS
    ICEOBS --> D18OOBS
    SPELOBS --> D18OSPELOBS
    SPELOBS --> D13COBS
    
    PALSAMPLE --> ICECORE
    PALSAMPLE --> SPEL
    CE22 -.-> ICECORE
    
    SAMPLINGLOC --> DRILLSITE
    SAMPLINGLOC --> CAVE
    CAVE --> ARCHCAVE
    CA2 -.-> ARCHCAVE
    
    CHRONO --> ICECHRONO
    CHRONO --> UTHCHRONO

    %% Core to CI
    SAMPLINGLOC --> CIFINDSPOT
    CIFINDSPOT --> CIARCHSITE
    CA2 -.-> CIARCHSITE
    CE5 -.-> CIVOLEVENT
    CE26 -.-> CITEPHRA
    CGP6 -.-> CIFINDSPOT

    %% Styling - External Ontologies
    style EXT fill:#fafafa,stroke:#999,color:#333
    style CRM fill:#fde8e8,stroke:#9b2226,color:#333
    style CRMSCI fill:#ffe8e8,stroke:#e63946,color:#333
    style SOSA fill:#e8f0fb,stroke:#1d3557,color:#333
    style GEO fill:#e8f1f7,stroke:#457b9d,color:#333
    style PROV fill:#fef0e8,stroke:#e76f51,color:#333
    
    style CE53 fill:#9b2226,color:#fff,stroke:#7a1a1d
    style CE27 fill:#9b2226,color:#fff,stroke:#7a1a1d
    style CE22 fill:#9b2226,color:#fff,stroke:#7a1a1d
    style CE7 fill:#9b2226,color:#fff,stroke:#7a1a1d
    style CS4 fill:#e63946,color:#fff,stroke:#c1121f
    style CS1 fill:#e63946,color:#fff,stroke:#c1121f
    style CS6 fill:#e63946,color:#fff,stroke:#c1121f
    style CS9 fill:#e63946,color:#fff,stroke:#c1121f
    style SO fill:#1d3557,color:#fff,stroke:#0d2137
    style SS fill:#1d3557,color:#fff,stroke:#0d2137
    style SP fill:#1d3557,color:#fff,stroke:#0d2137
    style GF fill:#457b9d,color:#fff,stroke:#2c5f7a
    style PE fill:#e76f51,color:#fff,stroke:#c45c3e
    
    %% Styling - Core
    style CORE fill:#e8f4f8,stroke:#457b9d,stroke-width:2px,color:#333
    style PALOBS fill:#74c0fc,color:#000,stroke:#1971c2,stroke-width:2px
    style PALSAMPLE fill:#74c0fc,color:#000,stroke:#1971c2,stroke-width:2px
    style SAMPLINGLOC fill:#74c0fc,color:#000,stroke:#1971c2,stroke-width:2px
    style CHRONO fill:#74c0fc,color:#000,stroke:#1971c2,stroke-width:2px
    style OBSPROP fill:#a5d8ff,color:#000,stroke:#4dabf7
    style DATASRC fill:#a5d8ff,color:#000,stroke:#4dabf7
    style MTYPE fill:#a5d8ff,color:#000,stroke:#4dabf7
    style SMOOTH fill:#a5d8ff,color:#000,stroke:#4dabf7
    
    %% Styling - EPICA
    style EPICA fill:#d4edda,stroke:#2d6a4f,stroke-width:2px,color:#333
    style ICEOBS fill:#2d6a4f,color:#fff,stroke:#1b4332,stroke-width:2px
    style CH4OBS fill:#40916c,color:#fff,stroke:#2d6a4f
    style D18OOBS fill:#40916c,color:#fff,stroke:#2d6a4f
    style ICECORE fill:#2d6a4f,color:#fff,stroke:#1b4332,stroke-width:2px
    style DRILLSITE fill:#2d6a4f,color:#fff,stroke:#1b4332,stroke-width:2px
    style DRILLCAMP fill:#2d6a4f,color:#fff,stroke:#1b4332,stroke-width:2px
    style ICECHRONO fill:#40916c,color:#fff,stroke:#2d6a4f
    
    %% Styling - SISAL
    style SISAL fill:#fff3cd,stroke:#856404,stroke-width:2px,color:#333
    style SPELOBS fill:#856404,color:#fff,stroke:#664d03,stroke-width:2px
    style D18OSPELOBS fill:#b8860b,color:#fff,stroke:#856404
    style D13COBS fill:#b8860b,color:#fff,stroke:#856404
    style SPEL fill:#856404,color:#fff,stroke:#664d03,stroke-width:2px
    style CAVE fill:#856404,color:#fff,stroke:#664d03,stroke-width:2px
    style ARCHCAVE fill:#b8860b,color:#fff,stroke:#856404
    style SSE fill:#856404,color:#fff,stroke:#664d03,stroke-width:2px
    style UTHCHRONO fill:#b8860b,color:#fff,stroke:#856404

    %% Styling - CI
    style CI fill:#fce8d5,stroke:#a0522d,stroke-width:2px,color:#333
    style CIFINDSPOT fill:#a0522d,color:#fff,stroke:#7a3b1e,stroke-width:2px
    style CIARCHSITE fill:#cd6c2a,color:#fff,stroke:#a0522d
    style CIVOLEVENT fill:#cd6c2a,color:#fff,stroke:#a0522d
    style CITEPHRA fill:#cd6c2a,color:#fff,stroke:#a0522d

    %% Styling - CRMarchaeo / CRMgeo
    style CRMEXT fill:#f5e6f0,stroke:#7b2d8b,stroke-width:2px,color:#333
    style CA2 fill:#7b2d8b,color:#fff,stroke:#5c1f6a
    style CGP6 fill:#7b2d8b,color:#fff,stroke:#5c1f6a
    style CE5 fill:#9b2226,color:#fff,stroke:#7a1a1d
    style CE26 fill:#9b2226,color:#fff,stroke:#7a1a1d
"""
)


def _mermaid_instance_epica(rw: int, sgw: int, sgp: int) -> str:
    """EPICA named-individual instance diagram."""
    return textwrap.dedent(
        f"""\
    flowchart LR

    CATALOG(["PalaeoclimateDataCatalogue
    geolod:EPICA_DomeC_Catalog"])
    
    DATASET(["IceCoreDataset
    geolod:EPICA_DomeC_Dataset"])
    
    OBS(["IceCoreObservation
    geolod:Obs_CH4_0001
    geolod:Obs_d18O_0001"])
    
    CORE(["IceCore
    geolod:EpicaDomeC_IceCore"])
    
    SITE(["DrillingSite
    geolod:EpicaDomeC_Site"])
    
    CAMPAIGN(["DrillingCampaign
    EPICA Dome C 1996-2004"])
    
    PROP_CH4(["CH4ConcentrationProperty"])
    PROP_D18O(["Delta18OProperty"])
    MTYPE_CH4(["MeasurementType CH4"])
    MTYPE_D18O(["MeasurementType d18O"])
    CHRON_EDC2(["EDC2 Chronology"])
    CHRON_AICC(["AICC2023 Chronology"])
    MEDIAN(["RollingMedianFilter w11"])
    SG(["SavitzkyGolayFilter w11 p2"])
    SOURCE_CH4(["PANGAEA 472484
    Spahni 2006"])
    SOURCE_D18O(["PANGAEA 961024
    Bouchet 2023"])
    
    GEOM(["sf:Point
    geolod:EpicaDomeC_Geometry"])
    
    LDEPTH((atDepth_m))
    LAGE((ageKaBP))
    LVAL((measuredValue))
    LMEDIAN((smoothed median))
    LSG((smoothed savgol))
    LWKT((asWKT POINT))
    LPPB((unit PPB))
    LPRM((unit PERMILLE))

    CATALOG -->|dcat:dataset| DATASET
    DATASET -->|hasObservation| OBS
    DATASET -->|hasDrillingCampaign| CAMPAIGN
    OBS -->|hasFeatureOfInterest| CORE
    OBS -->|observedProperty| PROP_CH4
    OBS -->|observedProperty| PROP_D18O
    OBS -->|measurementType| MTYPE_CH4
    OBS -->|measurementType| MTYPE_D18O
    OBS -->|ageChronology| CHRON_EDC2
    OBS -->|ageChronology| CHRON_AICC
    OBS -->|smoothingMethod| MEDIAN
    OBS -->|smoothingMethod| SG
    OBS -->|wasDerivedFrom| SOURCE_CH4
    OBS -->|wasDerivedFrom| SOURCE_D18O
    OBS -.->|atDepth_m| LDEPTH
    OBS -.->|ageKaBP| LAGE
    OBS -.->|measuredValue| LVAL
    OBS -.->|smoothed| LMEDIAN
    OBS -.->|smoothed| LSG
    PROP_CH4 -.->|unit| LPPB
    PROP_D18O -.->|unit| LPRM
    CORE -->|extractedFrom| SITE
    CAMPAIGN -->|tookPlaceAt| SITE
    CAMPAIGN -->|removedSample| CORE
    SITE -->|geo:hasGeometry| GEOM
    GEOM -.->|asWKT| LWKT

    %% Main instances - darker green
    style CATALOG fill:#2d6a4f,color:#fff,stroke:#1b4332,stroke-width:2px
    style DATASET fill:#2d6a4f,color:#fff,stroke:#1b4332,stroke-width:2px
    style OBS fill:#2d6a4f,color:#fff,stroke:#1b4332,stroke-width:2px
    style CORE fill:#2d6a4f,color:#fff,stroke:#1b4332,stroke-width:2px
    style SITE fill:#2d6a4f,color:#fff,stroke:#1b4332,stroke-width:2px
    style CAMPAIGN fill:#2d6a4f,color:#fff,stroke:#1b4332,stroke-width:2px
    
    %% Supporting instances - lighter green
    style PROP_CH4 fill:#40916c,color:#fff,stroke:#2d6a4f
    style PROP_D18O fill:#40916c,color:#fff,stroke:#2d6a4f
    style MTYPE_CH4 fill:#40916c,color:#fff,stroke:#2d6a4f
    style MTYPE_D18O fill:#40916c,color:#fff,stroke:#2d6a4f
    style CHRON_EDC2 fill:#40916c,color:#fff,stroke:#2d6a4f
    style CHRON_AICC fill:#40916c,color:#fff,stroke:#2d6a4f
    style MEDIAN fill:#40916c,color:#fff,stroke:#2d6a4f
    style SG fill:#40916c,color:#fff,stroke:#2d6a4f
    style SOURCE_CH4 fill:#40916c,color:#fff,stroke:#2d6a4f
    style SOURCE_D18O fill:#40916c,color:#fff,stroke:#2d6a4f
    
    %% Geometry - blue
    style GEOM fill:#457b9d,color:#fff,stroke:#2c5f7a
    
    %% Literals - bright yellow
    style LDEPTH fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LAGE fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LVAL fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LMEDIAN fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LSG fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LWKT fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LPPB fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LPRM fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
"""
    )


def _mermaid_instance_sisal(n: int = 305) -> str:
    """SISAL named-individual instance diagram."""
    return textwrap.dedent(
        f"""\
    flowchart LR

    COLLECTION(["geo:FeatureCollection
    geolod:SISAL_Cave_Collection
    {n} members"])

    ARCHCOLL(["geo:FeatureCollection
    geolod:SISAL_ArchaeologicalCave_Collection"])

    CAVE(["Cave
    geolod:Cave_site_0001"])

    ARCHCAVE(["ArchaeologicalCaveSite
    geolod:Cave_site_0077
    Chauvet cave"])

    GEOM(["sf:Point
    geolod:Cave_site_0001_Geometry"])

    SPEL(["Speleothem
    geolod:Speleothem_entity_XXXX"])

    SSE(["SpeleothemSamplingEvent"])

    OBS(["SpeleothemObservation
    geolod:Obs_d18O_XXXX
    geolod:Obs_d13C_XXXX"])

    PROP_D18O(["Delta18OProperty"])
    PROP_D13C(["Delta13CProperty"])
    MTYPE_D18O(["MeasurementType d18O"])
    MTYPE_D13C(["MeasurementType d13C"])
    UTHC(["UThChronology"])
    MEDIAN(["RollingMedianFilter w11"])
    SAVGOL(["SavitzkyGolayFilter w11 p2"])
    DATASRC(["SISALv3 DataSource"])
    WIKIDATA(["wikidata:Q191483"])
    UNESCO(["UNESCO WH #1426"])
    ARCHCTX(["geolod:PalaeolithicContext"])

    LAGE((ageKaBP))
    LDEPTH((atDepth_mm))
    LVAL((measuredValue))
    LMEDIAN((smoothed median))
    LSG((smoothed savgol))
    LWKT((asWKT POINT))
    LPRM((unit PERMILLE))
    LCAT(("archaeologicalCategory
    'Palaeolithic Art'"))
    LCONF((confidence: high))

    COLLECTION -->|rdfs:member| CAVE
    ARCHCOLL -->|rdfs:member| ARCHCAVE
    CAVE -->|geo:hasGeometry| GEOM
    ARCHCAVE -->|geo:hasGeometry| GEOM
    ARCHCAVE -->|owl:sameAs| WIKIDATA
    ARCHCAVE -->|geolod:unescoWHId| UNESCO
    ARCHCAVE -->|archaeologicalBroaderContext| ARCHCTX
    ARCHCAVE -.->|archaeologicalCategory| LCAT
    ARCHCAVE -.->|archaeologicalConfidence| LCONF
    SPEL -->|collectedFrom| CAVE
    SSE -->|tookPlaceAt| CAVE
    SSE -->|removedSample| SPEL
    OBS -->|hasFeatureOfInterest| SPEL
    OBS -->|observedProperty| PROP_D18O
    OBS -->|observedProperty| PROP_D13C
    OBS -->|measurementType| MTYPE_D18O
    OBS -->|measurementType| MTYPE_D13C
    OBS -->|ageChronology| UTHC
    OBS -->|smoothingMethod| MEDIAN
    OBS -->|smoothingMethod| SAVGOL
    OBS -->|wasDerivedFrom| DATASRC
    OBS -.->|ageKaBP| LAGE
    OBS -.->|atDepth_mm| LDEPTH
    OBS -.->|measuredValue| LVAL
    OBS -.->|smoothed| LMEDIAN
    OBS -.->|smoothed| LSG
    PROP_D18O -.->|unit| LPRM
    PROP_D13C -.->|unit| LPRM
    GEOM -.->|asWKT| LWKT

    %% Main instances - darker yellow/brown
    style COLLECTION fill:#856404,color:#fff,stroke:#664d03,stroke-width:2px
    style ARCHCOLL fill:#856404,color:#fff,stroke:#664d03,stroke-width:2px
    style CAVE fill:#856404,color:#fff,stroke:#664d03,stroke-width:2px
    style ARCHCAVE fill:#b8860b,color:#fff,stroke:#856404,stroke-width:2px
    style SPEL fill:#856404,color:#fff,stroke:#664d03,stroke-width:2px
    style SSE fill:#856404,color:#fff,stroke:#664d03,stroke-width:2px
    style OBS fill:#856404,color:#fff,stroke:#664d03,stroke-width:2px
    
    %% Supporting instances - lighter brown
    style PROP_D18O fill:#b8860b,color:#fff,stroke:#856404
    style PROP_D13C fill:#b8860b,color:#fff,stroke:#856404
    style MTYPE_D18O fill:#b8860b,color:#fff,stroke:#856404
    style MTYPE_D13C fill:#b8860b,color:#fff,stroke:#856404
    style UTHC fill:#b8860b,color:#fff,stroke:#856404
    style MEDIAN fill:#b8860b,color:#fff,stroke:#856404
    style SAVGOL fill:#b8860b,color:#fff,stroke:#856404
    style DATASRC fill:#b8860b,color:#fff,stroke:#856404
    style WIKIDATA fill:#990000,color:#fff,stroke:#660000
    style UNESCO fill:#005a8c,color:#fff,stroke:#003d61
    style ARCHCTX fill:#b8860b,color:#fff,stroke:#856404
    
    %% Geometry - blue
    style GEOM fill:#457b9d,color:#fff,stroke:#2c5f7a
    
    %% Literals - bright yellow
    style LAGE fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LDEPTH fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LVAL fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LMEDIAN fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LSG fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LWKT fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LPRM fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LCAT fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LCONF fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
"""
    )


def _mermaid_instance_ci(n: int = 74) -> str:
    """CI named-individual instance diagram."""
    return textwrap.dedent(
        f"""\
    flowchart LR

    EVENT(["CIVolcanicEvent
    geolod:ci_volcanic_event
    ~40 ka BP"])

    SOURCE(["crm:E53_Place
    geolod:ci_source_area
    Campi Flegrei"])

    COLLECTION(["geo:FeatureCollection
    geolod:ci/CIFindspotCollection
    {n} members"])

    SITE(["CIFindspot
    geolod:ci/cisite_1 ...
    geolod:ci/cisite_{n}"])

    ARCHSITE(["CIArchaeologicalSite
    geolod:ci/cisite_19
    geolod:ci/cisite_44 ..."])

    GEOM(["sf:Point / SP6_Declarative_Place
    geolod:ci/cisite_1_geom"])

    AGENT(["prov:Agent / foaf:Person
    fsld:agent_0000-0002-3246-3531"])

    ACTIVITY(["prov:Activity / fsl:Georeferencing
    geolod:ci/cisite_1_activity"])

    LWKT((asWKT POINT))
    LLABEL((rdfs:label))
    LCERT((hasCertaintyLevel))
    LLIT((hasLiteratureReference))

    EVENT -->|crm:P7_took_place_at| ARCHSITE
    EVENT -->|crm:P12_occurred_in_presence_of| SITE
    EVENT -->|crm:P7_took_place_at| SOURCE
    COLLECTION -->|rdfs:member| SITE
    COLLECTION -->|rdfs:member| ARCHSITE
    SITE -->|geo:hasGeometry| GEOM
    ARCHSITE -->|geo:hasGeometry| GEOM
    SITE -->|prov:wasGeneratedBy| ACTIVITY
    SITE -->|prov:wasAttributedTo| AGENT
    ACTIVITY -->|prov:wasAssociatedWith| AGENT
    GEOM -.->|asWKT| LWKT
    SITE -.->|rdfs:label| LLABEL
    SITE -.->|hasCertaintyLevel| LCERT
    SITE -.->|hasLiteratureReference| LLIT

    %% Main instances - terracotta
    style EVENT fill:#a0522d,color:#fff,stroke:#7a3b1e,stroke-width:2px
    style SOURCE fill:#a0522d,color:#fff,stroke:#7a3b1e,stroke-width:2px
    style COLLECTION fill:#a0522d,color:#fff,stroke:#7a3b1e,stroke-width:2px
    style SITE fill:#a0522d,color:#fff,stroke:#7a3b1e,stroke-width:2px
    style ARCHSITE fill:#cd6c2a,color:#fff,stroke:#a0522d,stroke-width:2px

    %% Supporting
    style AGENT fill:#cd6c2a,color:#fff,stroke:#a0522d
    style ACTIVITY fill:#cd6c2a,color:#fff,stroke:#a0522d

    %% Geometry - blue
    style GEOM fill:#457b9d,color:#fff,stroke:#2c5f7a

    %% Literals - bright yellow
    style LWKT fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LLABEL fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LCERT fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
    style LLIT fill:#ffd60a,color:#000,stroke:#d4a005,stroke-width:2px
"""
    )


def write_mermaid(
    outdir: str,
    rolling_window: int = 11,
    sg_window: int = 11,
    sg_poly: int = 2,
    n_sisal_sites: int = 305,
    n_ci_sites: int = 74,
) -> dict[str, str]:
    """
    Write all Mermaid diagram files to *outdir*.

    Files
    -----
    mermaid_taxonomy.mermaid         combined class hierarchy (Core + EPICA + SISAL + CI)
    mermaid_instance_epica.mermaid   EPICA named-individual instance diagram
    mermaid_instance_sisal.mermaid   SISAL named-individual instance diagram
    mermaid_instance_ci.mermaid      CI named-individual instance diagram

    Parameters
    ----------
    outdir          : output directory (created if absent)
    rolling_window  : rolling-median window size (from EPICA/SISAL config)
    sg_window       : Savitzky-Golay window size
    sg_poly         : Savitzky-Golay polynomial order
    n_sisal_sites   : number of SISAL cave sites (for collection label)
    n_ci_sites      : number of CI findspot sites (for collection label)

    Returns dict of {filename: full_path}.
    """
    os.makedirs(outdir, exist_ok=True)
    diagrams = {
        "mermaid_taxonomy.mermaid": MERMAID_TAXONOMY,
        "mermaid_instance_epica.mermaid": _mermaid_instance_epica(
            rolling_window, sg_window, sg_poly
        ),
        "mermaid_instance_sisal.mermaid": _mermaid_instance_sisal(n_sisal_sites),
        "mermaid_instance_ci.mermaid": _mermaid_instance_ci(n_ci_sites),
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
