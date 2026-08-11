#!/usr/bin/env python3
"""sisal_rdf.py - S3c.2: the SISAL cut to RDF.

Reads the structure-preserving cut in data/derived/sisal/tables/ and writes

    SISAL/rdf/sisal_ontology.ttl   the SISAL extension of the core ontology
    SISAL/rdf/sisal_v3_data.ttl    sites, speleothems, samples, isotopes, ages

Deliberately not read here: the old SISAL/v_data_*.csv. They carry the
decimal-separator error that sisal-db-v3 was built against, and the cut is
checked against the release. The figures still read them until S3c.4.

What this step decides, and why
-------------------------------
**All eight age models go in.** sisal_chronology holds seven, and
original_chronology an eighth. Only 18 213 of the 35 777 samples in the cut
carry a lin_interp_age; the remaining 17 564 carry another model, mostly
stalage or bacon, and 16 469 of them are in original_chronology. A graph
built on lin_interp alone would drop half the cut and would not say so.

**One age per sample is marked leading.** lin_interp where present, otherwise
the original_chronology age. The rule stops at entity_status: the eight
superseded speleothems hold no lin_interp age at all but 3 621
original_chronology ages, and letting those lead would put data back into
query results that SISAL itself has retired.

**The figures are reproduced through the model, not through the status.**

    ?a geolod:ageModel geolod:AgeModel_lin_interp .

is one triple pattern and yields exactly the 18 178 rows of the flat per-site
export, once δ¹⁸O is required - which is what wdttest-sisal draws. Filtering
on geolod:LeadingAssignment instead answers a different and equally valid
question, and returns about 31 000 samples. Both readings stay in the graph;
that is the whole point of keeping eight models rather than one.

Run through main.py, or standalone from the repository root:

    python SISAL/sisal_rdf.py
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS as DCT
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "data" / "derived" / "sisal" / "tables"
RDF_DIR = SCRIPT_DIR / "rdf"

sys.path.insert(0, str(ROOT / "ontology"))
from geo_lod_utils import add_generation_provenance  # noqa: E402

CRM = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
CRMSCI = Namespace("http://www.ics.forth.gr/isl/CRMsci/")
GEO = Namespace("http://www.opengis.net/ont/geosparql#")
SF = Namespace("http://www.opengis.net/ont/sf#")
GEOLOD = Namespace("http://w3id.org/geo-lod/")
SISAL = Namespace("http://w3id.org/geo-lod/sisal/")
TRS = Namespace("http://w3id.org/geo-lod/trs/")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
QUDT = Namespace("http://qudt.org/schema/qudt/")
UNIT = Namespace("http://qudt.org/vocab/unit/")
PROV = Namespace("http://www.w3.org/ns/prov#")
TIME = Namespace("http://www.w3.org/2006/time#")

ORCID_FLO = URIRef("https://orcid.org/0000-0002-3246-3531")
SISAL_DOI = URIRef("https://doi.org/10.5194/essd-16-1933-2024")
SISAL_LICENSE = URIRef("https://creativecommons.org/licenses/by/4.0/")

# Decimal places. Ages are written in ka BP (S0), so six places keep the
# thousandth of a year the source states; rounding to three would collapse
# distinct samples onto the same age.
DEC_AGE = 6
DEC_VALUE = 4
DEC_DEPTH = 3
DEC_COORD = 4

# --------------------------------------------------------------------------
# The eight age models
# --------------------------------------------------------------------------
# Seven columns triplets in sisal_chronology, plus original_chronology, whose
# own age_model_type names the method the original authors used. Order is the
# column order of the release, not a ranking; which one leads is decided per
# sample below.
CHRONOLOGY_MODELS: list[tuple[str, str, str]] = [
    ("lin_interp", "lin_interp_age", "linear interpolation between dating points"),
    ("lin_reg", "lin_reg_age", "linear regression"),
    ("bchron", "bchron_age", "Bchron age-depth model"),
    ("bacon", "bacon_age", "Bacon age-depth model"),
    ("oxcal", "oxcal_age", "OxCal age-depth model"),
    ("copra", "copra_age", "COPRA age-depth model"),
    ("stalage", "stalage_age", "StalAge age-depth model"),
]
ORIGINAL_MODEL = ("original", "interp_age", "chronology as published by the "
                                            "original authors")

ENTITY_STATUS = {
    "current": ("EntityStatus_current", "current",
                "The record SISAL v3 considers valid."),
    "current partially modified": (
        "EntityStatus_current_partially_modified", "current, partially modified",
        "Valid, with part of the record revised against its first publication."),
    "superseded": (
        "EntityStatus_superseded", "superseded",
        "Replaced by a newer record; corresponding_current names the successor."),
}


# --------------------------------------------------------------------------
# Reading the cut
# --------------------------------------------------------------------------

def read_table(name: str) -> list[dict]:
    path = TABLES_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch the cut first:\n"
            f"    python SISAL/sisal_import.py\n"
            f"or verify an existing one with --verify."
        )
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(value) -> float | None:
    """A number, or None where the cut has an empty field.

    Empty means the release records nothing, which is not the same as zero and
    must not become a triple.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.upper() == "NULL":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def dec(value: float, places: int) -> Literal:
    return Literal(f"{round(float(value), places):.{places}f}", datatype=XSD.decimal)


# --------------------------------------------------------------------------
# The SISAL extension ontology
# --------------------------------------------------------------------------
# Written from this constant rather than maintained as a file, for the same
# reason geo_lod_core.ttl is: a class added in code and forgotten in the TTL
# is a mismatch nobody notices until SHACL runs.
SISAL_ONTOLOGY_TTL = """\
# ==========================================================================
# sisal_ontology.ttl
# geo-lod SISAL extension - speleothem records from SISAL v3
# Generated by SISAL/sisal_rdf.py; do not edit by hand.
# ==========================================================================

@prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:     <http://www.w3.org/2002/07/owl#> .
@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .
@prefix crm:     <http://www.cidoc-crm.org/cidoc-crm/> .
@prefix crmsci:  <http://www.ics.forth.gr/isl/CRMsci/> .
@prefix sosa:    <http://www.w3.org/ns/sosa/> .
@prefix skos:    <http://www.w3.org/2004/02/skos/core#> .
@prefix time:    <http://www.w3.org/2006/time#> .
@prefix qudt:    <http://qudt.org/schema/qudt/> .
@prefix geolod:  <http://w3id.org/geo-lod/> .

<http://w3id.org/geo-lod/sisal/>
    a owl:Ontology ;
    rdfs:label   "geo-lod SISAL Extension"@en ;
    rdfs:comment "Speleothem records from SISAL v3: caves, speleothems, samples, stable-isotope observations and their competing age models."@en ;
    owl:imports  <http://w3id.org/geo-lod/> .

# -- Physical things -------------------------------------------------------

geolod:Cave
    a owl:Class ;
    rdfs:subClassOf geolod:SamplingLocation ;
    rdfs:label   "Cave"@en ;
    rdfs:comment "A karst cave holding one or more speleothem records."@en .

geolod:Speleothem
    a owl:Class ;
    rdfs:subClassOf geolod:PalaeoclimateSample ;
    rdfs:label   "Speleothem"@en ;
    rdfs:comment "One secondary carbonate deposit whose growth records a climate signal. SISAL calls it an entity; several may come from one cave, and their isotope values are offset against one another for reasons unrelated to climate."@en .

geolod:SpeleothemSample
    a owl:Class ;
    rdfs:subClassOf crm:E18_Physical_Thing ;
    rdfs:subClassOf sosa:Sample ;
    rdfs:label   "Speleothem Sample"@en ;
    rdfs:comment "One sampling position along a speleothem, identified by its depth from the top. Carries the isotope measurements and the competing ages."@en .

# -- Observations ----------------------------------------------------------

geolod:SpeleothemObservation
    a owl:Class ;
    rdfs:subClassOf geolod:PalaeoclimateObservation ;
    rdfs:label   "Speleothem Observation"@en .

geolod:Delta18OSpeleothemObservation
    a owl:Class ;
    rdfs:subClassOf geolod:SpeleothemObservation ;
    rdfs:label   "δ¹⁸O Speleothem Observation"@en .

geolod:Delta13CSpeleothemObservation
    a owl:Class ;
    rdfs:subClassOf geolod:SpeleothemObservation ;
    rdfs:label   "δ¹³C Speleothem Observation"@en .

geolod:Delta13CProperty
    a owl:Class ;
    rdfs:subClassOf geolod:ObservableProperty ;
    rdfs:label   "δ¹³C Property"@en .

# -- Age models and their assignments --------------------------------------

geolod:AgeModel
    a owl:Class ;
    rdfs:subClassOf crmsci:S6_Data_Evaluation ;
    rdfs:label   "Age Model"@en ;
    rdfs:comment "A named method for turning dating points into an age at every depth. SISAL v3 carries seven, and the chronology as originally published is an eighth."@en .

geolod:AgeAssignment
    a owl:Class ;
    rdfs:subClassOf crm:E13_Attribute_Assignment ;
    rdfs:label   "Age Assignment"@en ;
    rdfs:comment "One model's age for one sample. Kept as an assignment and not as a bare property because a sample has as many ages as there are models that reached it, and a consumer has to be able to tell which is which."@en .

geolod:ageModel
    a owl:ObjectProperty ;
    rdfs:domain  geolod:AgeAssignment ;
    rdfs:range   geolod:AgeModel ;
    rdfs:label   "age model"@en ;
    rdfs:comment "The model that produced this age. Filtering on it is what reproduces a published figure, which was drawn from one model and not from whichever age happened to lead."@en .

geolod:hasAgeAssignment
    a owl:ObjectProperty ;
    rdfs:domain  geolod:SpeleothemSample ;
    rdfs:range   geolod:AgeAssignment ;
    rdfs:label   "has age assignment"@en .

geolod:ageUncertaintyPos_ka
    a owl:DatatypeProperty ;
    rdfs:domain  geolod:AgeAssignment ;
    rdfs:range   xsd:decimal ;
    rdfs:label   "age uncertainty, older side (ka)"@en .

geolod:ageUncertaintyNeg_ka
    a owl:DatatypeProperty ;
    rdfs:domain  geolod:AgeAssignment ;
    rdfs:range   xsd:decimal ;
    rdfs:label   "age uncertainty, younger side (ka)"@en .

geolod:extrapolatedBeyondDatingRange
    a owl:DatatypeProperty ;
    rdfs:domain  geolod:AgeAssignment ;
    rdfs:range   xsd:boolean ;
    rdfs:label   "extrapolated beyond dating range"@en ;
    rdfs:comment "True where the model ran past its outermost dating point, which is how an age younger than 1950 arises in a record with no modern control point. Deliberately not expressed through geolod:assignmentStatus: the status says whether geo-lod follows this age, this says whether the model had anything to stand on."@en .

# -- Entity status ---------------------------------------------------------

geolod:EntityStatus
    a owl:Class ;
    rdfs:subClassOf skos:Concept ;
    rdfs:label   "Entity Status"@en ;
    rdfs:comment "SISAL's judgement on a speleothem record. A statement about the data, not about the cave."@en .

geolod:entityStatus
    a owl:ObjectProperty ;
    rdfs:domain  geolod:Speleothem ;
    rdfs:range   geolod:EntityStatus ;
    rdfs:label   "entity status"@en .

geolod:correspondingCurrentSpeleothem
    a owl:ObjectProperty ;
    rdfs:domain  geolod:Speleothem ;
    rdfs:range   geolod:Speleothem ;
    rdfs:label   "corresponding current speleothem"@en ;
    rdfs:comment "The record that replaced a superseded one. Without it there is no way to say why two speleothems describe the same lamina."@en .

# -- Identifiers and depth -------------------------------------------------

geolod:entityId
    a owl:DatatypeProperty ;
    rdfs:range   xsd:integer ;
    rdfs:label   "SISAL entity id"@en .

geolod:siteId
    a owl:DatatypeProperty ;
    rdfs:range   xsd:integer ;
    rdfs:label   "SISAL site id"@en .

geolod:sampleId
    a owl:DatatypeProperty ;
    rdfs:range   xsd:integer ;
    rdfs:label   "SISAL sample id"@en .

geolod:atDepth_mm
    a owl:DatatypeProperty ;
    rdfs:range   xsd:decimal ;
    rdfs:label   "at depth (mm)"@en ;
    rdfs:comment "Depth from the top of the speleothem. Millimetres, as the release states them; the ice-core records use metres and the two must not be mixed."@en ;
    qudt:unit    <http://qudt.org/vocab/unit/MilliM> .

geolod:sampleThickness_mm
    a owl:DatatypeProperty ;
    rdfs:range   xsd:decimal ;
    rdfs:label   "sample thickness (mm)"@en .

geolod:mineralogy
    a owl:DatatypeProperty ;
    rdfs:range   xsd:string ;
    rdfs:label   "mineralogy"@en .

geolod:aragoniteCorrection
    a owl:DatatypeProperty ;
    rdfs:range   xsd:string ;
    rdfs:label   "aragonite correction"@en .

geolod:elevation_m
    a owl:DatatypeProperty ;
    rdfs:range   xsd:decimal ;
    rdfs:label   "elevation (m)"@en .
"""


def write_ontology() -> Path:
    RDF_DIR.mkdir(parents=True, exist_ok=True)
    path = RDF_DIR / "sisal_ontology.ttl"
    path.write_text(SISAL_ONTOLOGY_TTL, encoding="utf-8", newline="\n")
    return path


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------

def bind_prefixes(g: Graph) -> None:
    for prefix, ns in [
        ("crm", CRM), ("crmsci", CRMSCI), ("geo", GEO), ("sf", SF),
        ("geolod", GEOLOD), ("sisal", SISAL), ("trs", TRS), ("sosa", SOSA),
        ("qudt", QUDT), ("unit", UNIT), ("prov", PROV), ("time", TIME),
        ("dct", DCT), ("skos", SKOS), ("owl", OWL),
    ]:
        g.bind(prefix, ns)


def add_vocabularies(g: Graph) -> None:
    """The eight models, the three entity statuses, the source, the properties.

    Each model is both an AgeModel and a Chronology: an age is only meaningful
    together with the model that produced it, so the same node serves as the
    value of geolod:ageModel and as the TRS of the time position. The same
    reasoning as for the EPICA chronologies in S2.
    """
    for key, _, comment in CHRONOLOGY_MODELS + [ORIGINAL_MODEL]:
        model = GEOLOD[f"AgeModel_{key}"]
        trs = TRS[f"SISALv3-{key}"]
        g.add((model, RDF.type, GEOLOD["AgeModel"]))
        g.add((model, RDF.type, OWL.NamedIndividual))
        g.add((model, RDFS.label, Literal(key, lang="en")))
        g.add((model, RDFS.comment, Literal(comment, lang="en")))
        g.add((model, GEOLOD["hasChronology"], trs))

        g.add((trs, RDF.type, GEOLOD["UThChronology"]))
        g.add((trs, RDF.type, TIME["TRS"]))
        g.add((trs, RDF.type, OWL.NamedIndividual))
        g.add((trs, RDFS.label, Literal(f"SISAL v3 {key} chronology", lang="en")))
        g.add((trs, DCT.source, SISAL_DOI))

    for _, (local, label, comment) in ENTITY_STATUS.items():
        node = GEOLOD[local]
        g.add((node, RDF.type, GEOLOD["EntityStatus"]))
        g.add((node, RDF.type, OWL.NamedIndividual))
        g.add((node, SKOS.prefLabel, Literal(label, lang="en")))
        g.add((node, RDFS.label, Literal(label, lang="en")))
        g.add((node, RDFS.comment, Literal(comment, lang="en")))

    source = GEOLOD["SISALv3_DataSource"]
    g.add((source, RDF.type, GEOLOD["DataSource"]))
    g.add((source, RDF.type, OWL.NamedIndividual))
    g.add((source, RDFS.label, Literal("SISAL v3 database", lang="en")))
    # The open DOI from the pipeline todo list; the two PANGAEA sources were
    # done with S2, this was the one still missing.
    g.add((source, DCT.source, SISAL_DOI))
    g.add((source, OWL.sameAs, SISAL_DOI))
    g.add((source, DCT.license, SISAL_LICENSE))

    for local, cls, label in [
        ("Delta18OProperty_speleothem", "Delta18OProperty", "δ¹⁸O of speleothem calcite"),
        ("Delta13CProperty_speleothem", "Delta13CProperty", "δ¹³C of speleothem calcite"),
    ]:
        node = GEOLOD[local]
        g.add((node, RDF.type, GEOLOD[cls]))
        g.add((node, RDF.type, OWL.NamedIndividual))
        g.add((node, RDFS.label, Literal(label, lang="en")))


def add_sites(g: Graph, sites: list[dict]) -> dict[str, URIRef]:
    """The caves of the cut, on the existing geolod:Cave_site_NNNN nodes (A1)."""
    uris: dict[str, URIRef] = {}
    for row in sites:
        site_id = int(row["site_id"])
        site = GEOLOD[f"Cave_site_{site_id:04d}"]
        uris[row["site_id"]] = site

        g.add((site, RDF.type, GEOLOD["Cave"]))
        g.add((site, RDF.type, OWL.NamedIndividual))
        g.add((site, RDFS.label, Literal(row["site_name"], lang="en")))
        g.add((site, GEOLOD["siteId"], Literal(site_id, datatype=XSD.integer)))
        g.add((site, PROV.wasDerivedFrom, GEOLOD["SISALv3_DataSource"]))

        lat, lon = num(row["latitude"]), num(row["longitude"])
        if lat is not None and lon is not None:
            geom = SISAL[f"geom_site_{site_id:04d}"]
            wkt = (f"<http://www.opengis.net/def/crs/EPSG/0/4326> "
                   f"POINT({lon:.{DEC_COORD}f} {lat:.{DEC_COORD}f})")
            g.add((site, GEO["hasGeometry"], geom))
            g.add((geom, RDF.type, SF["Point"]))
            g.add((geom, GEO["asWKT"], Literal(wkt, datatype=GEO["wktLiteral"])))
        elevation = num(row.get("elevation"))
        if elevation is not None:
            g.add((site, GEOLOD["elevation_m"], dec(elevation, 1)))
    return uris


def add_entities(g: Graph, entities: list[dict],
                 site_uris: dict[str, URIRef]) -> dict[str, URIRef]:
    """The speleothems, with their status and, where superseded, their successor."""
    uris = {row["entity_id"]: SISAL[f"entity_{int(row['entity_id']):05d}"]
            for row in entities}

    for row in entities:
        entity_id = int(row["entity_id"])
        entity = uris[row["entity_id"]]
        g.add((entity, RDF.type, GEOLOD["Speleothem"]))
        g.add((entity, RDF.type, OWL.NamedIndividual))
        g.add((entity, RDFS.label, Literal(row["entity_name"], lang="en")))
        g.add((entity, GEOLOD["entityId"], Literal(entity_id, datatype=XSD.integer)))
        g.add((entity, PROV.wasDerivedFrom, GEOLOD["SISALv3_DataSource"]))

        site = site_uris.get(row["site_id"])
        if site is not None:
            g.add((entity, GEOLOD["extractedFrom"], site))
            g.add((site, CRM["P53i_is_former_or_current_location_of"], entity))

        status = ENTITY_STATUS.get((row.get("entity_status") or "").strip())
        if status:
            g.add((entity, GEOLOD["entityStatus"], GEOLOD[status[0]]))

        successor = (row.get("corresponding_current") or "").strip()
        if successor and successor in uris:
            g.add((entity, GEOLOD["correspondingCurrentSpeleothem"], uris[successor]))
    return uris


def add_samples(g: Graph, samples: list[dict],
                entity_uris: dict[str, URIRef]) -> dict[str, URIRef]:
    uris: dict[str, URIRef] = {}
    for row in samples:
        sample_id = int(row["sample_id"])
        sample = SISAL[f"sample_{sample_id:07d}"]
        uris[row["sample_id"]] = sample

        g.add((sample, RDF.type, GEOLOD["SpeleothemSample"]))
        g.add((sample, RDF.type, OWL.NamedIndividual))
        g.add((sample, GEOLOD["sampleId"], Literal(sample_id, datatype=XSD.integer)))

        entity = entity_uris.get(row["entity_id"])
        if entity is not None:
            g.add((sample, GEOLOD["collectedFrom"], entity))
            g.add((entity, CRM["P46_is_composed_of"], sample))

        depth = num(row.get("depth_sample"))
        if depth is not None:
            g.add((sample, GEOLOD["atDepth_mm"], dec(depth, DEC_DEPTH)))
            g.add((sample, RDFS.label,
                   Literal(f"sample {sample_id} at {depth:.1f} mm", lang="en")))
        else:
            g.add((sample, RDFS.label, Literal(f"sample {sample_id}", lang="en")))

        thickness = num(row.get("sample_thickness"))
        if thickness is not None:
            g.add((sample, GEOLOD["sampleThickness_mm"], dec(thickness, DEC_DEPTH)))
        for column, prop in [("mineralogy", "mineralogy"),
                             ("arag_corr", "aragoniteCorrection")]:
            text = (row.get(column) or "").strip()
            if text:
                g.add((sample, GEOLOD[prop], Literal(text)))
    return uris


def add_isotopes(g: Graph, rows: list[dict], sample_uris: dict[str, URIRef],
                 isotope: str) -> int:
    """δ¹⁸O or δ¹³C observations, one per measured sample."""
    meta = {
        "d18o": ("Delta18OSpeleothemObservation", "Delta18OProperty_speleothem",
                 "d18o_measurement", "d18o_precision", "δ¹⁸O"),
        "d13c": ("Delta13CSpeleothemObservation", "Delta13CProperty_speleothem",
                 "d13c_measurement", "d13c_precision", "δ¹³C"),
    }[isotope]
    cls, prop, value_col, precision_col, label = meta

    written = 0
    for row in rows:
        value = num(row[value_col])
        if value is None:
            continue
        sample = sample_uris.get(row["sample_id"])
        if sample is None:
            continue
        sample_id = int(row["sample_id"])
        obs = SISAL[f"obs_{isotope}_{sample_id:07d}"]

        g.add((obs, RDF.type, GEOLOD[cls]))
        g.add((obs, RDF.type, OWL.NamedIndividual))
        g.add((obs, RDFS.label,
               Literal(f"{label} observation, sample {sample_id}", lang="en")))
        g.add((obs, SOSA["hasFeatureOfInterest"], sample))
        g.add((obs, SOSA["observedProperty"], GEOLOD[prop]))
        g.add((obs, SOSA["hasSimpleResult"], dec(value, DEC_VALUE)))
        g.add((obs, GEOLOD["measuredValue"], dec(value, DEC_VALUE)))
        g.add((obs, QUDT["unit"], UNIT["PERMILLE"]))
        g.add((obs, PROV.wasDerivedFrom, GEOLOD["SISALv3_DataSource"]))
        g.add((sample, GEOLOD["hasObservation"], obs))

        precision = num(row.get(precision_col))
        if precision is not None:
            g.add((obs, GEOLOD["standardDeviation"], dec(precision, DEC_VALUE)))
        written += 1
    return written


def add_ages(core: Graph, alt: Graph, chronology: list[dict],
             original: list[dict], sample_uris: dict[str, URIRef],
             superseded: set[str]) -> dict[str, int]:
    """Every model's age for every sample, with exactly one marked leading.

    The leading rule, and the reason it stops at entity_status:

      1. lin_interp where the sample has one. That is the age the published
         figures use, and the one the flat per-site export carries.
      2. otherwise the original_chronology age - unless the sample belongs to
         a superseded speleothem. Those hold no lin_interp age at all but
         3621 original ages, and letting them lead would return records SISAL
         has retired to any query that asks for the leading age.
      3. otherwise nothing leads, and the sample carries only alternatives.
    """
    by_sample: dict[str, list[tuple[str, float, float | None, float | None]]] = \
        defaultdict(list)

    for row in chronology:
        for key, column, _ in CHRONOLOGY_MODELS:
            age = num(row[column])
            if age is None:
                continue
            by_sample[row["sample_id"]].append(
                (key, age, num(row.get(f"{column}_uncert_pos")),
                 num(row.get(f"{column}_uncert_neg"))))

    for row in original:
        age = num(row[ORIGINAL_MODEL[1]])
        if age is None:
            continue
        by_sample[row["sample_id"]].append(
            ("original", age, num(row.get("interp_age_uncert_pos")),
             num(row.get("interp_age_uncert_neg"))))

    tally = {"assignments": 0, "leading_lin_interp": 0, "leading_original": 0,
             "no_leading": 0, "extrapolated": 0}

    for sample_id, entries in by_sample.items():
        sample = sample_uris.get(sample_id)
        if sample is None:
            continue
        models = {key for key, _, _, _ in entries}
        if "lin_interp" in models:
            leading = "lin_interp"
            tally["leading_lin_interp"] += 1
        elif "original" in models and sample_id not in superseded:
            leading = "original"
            tally["leading_original"] += 1
        else:
            leading = None
            tally["no_leading"] += 1

        for key, age_years, uncert_pos, uncert_neg in entries:
            age_ka = age_years / 1000.0
            is_leading = key == leading
            target = core if is_leading else alt
            node = SISAL[f"age_{int(sample_id):07d}_{key}"]

            # No rdfs:label and no owl:NamedIndividual on the assignments.
            # At 132 000 nodes a label is 132 000 long literals that say
            # nothing a query could not derive from the three properties
            # below, and the graph is already the largest in the project.
            target.add((node, RDF.type, GEOLOD["AgeAssignment"]))
            target.add((node, GEOLOD["ageModel"], GEOLOD[f"AgeModel_{key}"]))
            target.add((node, GEOLOD["ageKaBP"], dec(age_ka, DEC_AGE)))
            target.add((node, CRM["P140_assigned_attribute_to"], sample))
            target.add((node, GEOLOD["assignmentStatus"],
                        GEOLOD["LeadingAssignment"] if is_leading
                        else GEOLOD["AlternativeAssignment"]))
            target.add((node, GEOLOD["inChronology"], TRS[f"SISALv3-{key}"]))
            target.add((sample, GEOLOD["hasAgeAssignment"], node))

            if uncert_pos is not None:
                target.add((node, GEOLOD["ageUncertaintyPos_ka"],
                            dec(uncert_pos / 1000.0, DEC_AGE)))
            if uncert_neg is not None:
                target.add((node, GEOLOD["ageUncertaintyNeg_ka"],
                            dec(uncert_neg / 1000.0, DEC_AGE)))

            # An age younger than 1950 in a record without a modern dating
            # point is the model running past its outermost control point.
            # Flagged, not dropped: the source states it (A4).
            if age_years < 0:
                target.add((node, GEOLOD["extrapolatedBeyondDatingRange"],
                            Literal(True, datatype=XSD.boolean)))
                tally["extrapolated"] += 1

            if is_leading:
                # The time position is written for the leading age only. For
                # an alternative, geolod:ageKaBP with geolod:inChronology says
                # the same thing in two triples instead of six, and nothing
                # reads a competing model without knowing which it is.
                position = SISAL[f"timepos_{int(sample_id):07d}_{key}"]
                core.add((position, RDF.type, TIME["TimePosition"]))
                core.add((position, TIME["numericPosition"], dec(age_ka, DEC_AGE)))
                core.add((position, TIME["hasTRS"], TRS[f"SISALv3-{key}"]))
                core.add((position, GEOLOD["ageKaBP"], dec(age_ka, DEC_AGE)))
                core.add((node, GEOLOD["hasTimePosition"], position))

                # Materialised on the sample, so that the core file alone
                # reproduces a figure without joining through an assignment:
                #   ?s geolod:ageChronology trs:SISALv3-lin_interp ;
                #      geolod:ageKaBP ?age .
                core.add((sample, GEOLOD["ageKaBP"], dec(age_ka, DEC_AGE)))
                core.add((sample, GEOLOD["ageChronology"], TRS[f"SISALv3-{key}"]))
            tally["assignments"] += 1
    return tally


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def build() -> bool:
    print("\n" + "=" * 72)
    print("  S3c.2 - SISAL cut to RDF")
    print("=" * 72)
    print(f"  Input:  {TABLES_DIR}")

    sites = read_table("site")
    entities = read_table("entity")
    samples = read_table("sample")
    d18o = read_table("d18o")
    d13c = read_table("d13c")
    chronology = read_table("sisal_chronology")
    original = read_table("original_chronology")

    print(f"  {len(sites)} sites, {len(entities)} speleothems, "
          f"{len(samples)} samples")

    superseded_entities = {row["entity_id"] for row in entities
                           if (row.get("entity_status") or "").strip() == "superseded"}
    superseded_samples = {row["sample_id"] for row in samples
                          if row["entity_id"] in superseded_entities}
    print(f"  {len(superseded_entities)} superseded speleothems, "
          f"{len(superseded_samples)} of their samples")

    g = Graph()
    bind_prefixes(g)

    add_vocabularies(g)
    site_uris = add_sites(g, sites)
    entity_uris = add_entities(g, entities, site_uris)
    sample_uris = add_samples(g, samples, entity_uris)

    n_d18o = add_isotopes(g, d18o, sample_uris, "d18o")
    n_d13c = add_isotopes(g, d13c, sample_uris, "d13c")
    print(f"  {n_d18o} δ¹⁸O and {n_d13c} δ¹³C observations")

    alt = Graph()
    bind_prefixes(alt)
    tally = add_ages(g, alt, chronology, original, sample_uris, superseded_samples)
    print(f"  {tally['assignments']} age assignments over eight models")
    print(f"    leading, lin_interp        {tally['leading_lin_interp']:>7d}")
    print(f"    leading, original          {tally['leading_original']:>7d}")
    print(f"    no leading age             {tally['no_leading']:>7d}")
    print(f"    extrapolated past 1950     {tally['extrapolated']:>7d}")

    add_generation_provenance(
        g,
        GEOLOD["SISAL_Dataset"],
        GEOLOD["SISAL_Generation"],
        inputs=[str(TABLES_DIR / f"{t}.csv") for t in
                ("site", "entity", "sample", "d18o", "d13c",
                 "sisal_chronology", "original_chronology")] + [__file__],
        agents=[ORCID_FLO],
        label="SISAL v3 RDF generation (S3c.2)",
    )

    onto_path = write_ontology()
    print(f"\n  ✓ {onto_path.relative_to(ROOT)}")

    core_path = RDF_DIR / "sisal_v3_core.ttl"
    g.serialize(destination=str(core_path), format="turtle")
    print(f"  ✓ {core_path.relative_to(ROOT)}  ({len(g):,} triples)")

    alt_path = RDF_DIR / "sisal_v3_chronologies.ttl"
    alt.serialize(destination=str(alt_path), format="turtle")
    print(f"  ✓ {alt_path.relative_to(ROOT)}  ({len(alt):,} triples)")
    return True


def main() -> int:
    try:
        return 0 if build() else 1
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
