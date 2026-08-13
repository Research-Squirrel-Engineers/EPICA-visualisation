#!/usr/bin/env python3
"""sisal_rdf.py - the SISAL cut to RDF.

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

**Not every run needs every site.** ``--sites dev`` builds the five-site
selection in DEV_SITES, chosen so that every guard this generator has still
fires; ``--sites spannagel`` builds one cave, which is what a consumer
repository asks for. A partial graph says so in the report and in
``owl:versionInfo``, because the file name is the same either way.

Run through main.py, or standalone from the repository root:

    python SISAL/sisal_rdf.py                  the whole cut
    python SISAL/sisal_rdf.py --sites dev      the development selection
    python SISAL/sisal_rdf.py --sites spannagel
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS as DCT
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TABLES_DIR = ROOT / "data" / "derived" / "sisal" / "tables"
CATALOGUE_DIR = ROOT / "data" / "derived" / "sisal" / "catalogue"
CURATED_DIR = ROOT / "data" / "curated"
RDF_DIR = SCRIPT_DIR / "rdf"

sys.path.insert(0, str(ROOT / "ontology"))
from geo_lod_utils import add_generation_provenance  # noqa: E402

CRM = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
CRMSCI = Namespace("http://www.ics.forth.gr/isl/CRMsci/")
CRMARCHAEO = Namespace("http://www.cidoc-crm.org/extensions/crmarchaeo/")
GEO = Namespace("http://www.opengis.net/ont/geosparql#")
SF = Namespace("http://www.opengis.net/ont/sf#")
GEOLOD = Namespace("http://w3id.org/geo-lod/")
SISAL = Namespace("http://w3id.org/geo-lod/sisal/")
TRS = Namespace("http://w3id.org/geo-lod/trs/")
STRAT = Namespace("http://w3id.org/geo-lod/strat/")
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
# Heartbeat
# --------------------------------------------------------------------------

class Heartbeat:
    """Says where the run is, every INTERVAL seconds, from its own thread.

    A progress line printed inside a loop cannot help here: most of the time
    goes into two rdflib calls that do not come back until they are done. A
    daemon thread is the only thing that can speak during them.

    Nothing is printed when a step finishes quickly, so a fast run stays as
    quiet as it was before.
    """

    INTERVAL = 30.0

    def __init__(self, total: int):
        self.total = total
        self.index = 0
        self.label = "starting"
        self.started = time.perf_counter()
        self.step_started = self.started
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "Heartbeat":
        self._thread.start()
        return self

    def step(self, label: str) -> None:
        self.index += 1
        self.label = label
        self.step_started = time.perf_counter()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.INTERVAL):
            now = time.perf_counter()
            # flush: the output goes through a pipe to main.py, where an
            # unflushed line would sit in the buffer until the step ends -
            # which is exactly the moment the message stops being useful.
            print(f"    … still running: step {self.index}/{self.total} "
                  f"{self.label}, {now - self.step_started:.0f}s in this step, "
                  f"{now - self.started:.0f}s total", flush=True)


# --------------------------------------------------------------------------
# Reading the cut
# --------------------------------------------------------------------------

def read_csv(path: Path, hint: str) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {hint}")
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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


# --------------------------------------------------------------------------
# Which sites the run reads
# --------------------------------------------------------------------------
# A development run does not need all twelve sites. The graph over twelve is
# 2.2 million triples and takes five minutes to build and twenty more to
# bundle and validate; a check that costs half an hour is a check that gets
# skipped. What a smaller selection must not do is go green because it never
# saw the awkward cases, so the members are chosen by which guard they
# exercise and not by geography.

#: site_id -> why it is in the development selection.
DEV_SITES = {
    58: "Spannagel - the WD1 evaluation the wdttest family builds on, plus "
        "SPA127 in two versions and δ¹³C on two of eight speleothems",
    145: "Antro del Corchia - the superseded records and their four hiatuses "
         "without a model statement",
    202: "Piani Eterni - the deep end, 84 to 289 ka, and no sample younger "
         "than MIS 5",
    277: "Huagapo - a composite whose dating points carry no depth, and ages "
         "extrapolated past 1950",
    127: "Xiaobailong - no δ¹³C at all, and ICP-MS U/Th Other in date_type",
}


def plain(text: str) -> str:
    """Lower case and without accents, for matching a name on the command line.

    Jaraguá is spelled with an accent in SISAL and without one on a Windows
    console. Matching the two would otherwise be an error message about a
    site that is plainly there.
    """
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", text.strip().lower())
                   if not unicodedata.combining(c))


def resolve_sites(spec: str, sites: list[dict]) -> tuple[set[str] | None, str]:
    """The site_ids to keep and a label for the report.

    *spec* is ``all``, ``dev``, or a comma-separated list of site ids and
    site names - ``--sites spannagel`` builds the graph of one cave, which is
    what a consumer repository needs when it wants that cave and nothing else.
    Names are matched case-insensitively and by prefix, so "spannagel" finds
    "Spannagel cave" without anyone having to type the word cave.
    """
    spec = (spec or "all").strip()
    if spec.lower() == "all":
        return None, f"all {len(sites)} sites of the cut"

    by_id = {row["site_id"]: row["site_name"] for row in sites}
    listing = ", ".join(f"{sid} {by_id[sid]}"
                        for sid in sorted(by_id, key=lambda v: int(v)))
    if spec.lower() == "dev":
        wanted = {str(site_id) for site_id in DEV_SITES}
        missing = wanted - set(by_id)
        if missing:
            raise SystemExit(
                f"  ✗ development selection asks for site(s) "
                f"{', '.join(sorted(missing))}, which the cut does not hold. "
                f"Re-export the cut or adjust DEV_SITES.")
        return wanted, f"development selection, {len(wanted)} of {len(sites)} sites"

    keep: set[str] = set()
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        if token in by_id:
            keep.add(token)
            continue
        matches = [sid for sid, name in by_id.items()
                   if plain(name).startswith(plain(token))]
        if len(matches) == 1:
            keep.add(matches[0])
        elif not matches:
            raise SystemExit(
                f"  ✗ no site called {token!r} in the cut. Available: {listing}")
        else:
            raise SystemExit(
                f"  ✗ {token!r} matches several sites: "
                + ", ".join(f"{sid} {by_id[sid]}" for sid in sorted(matches)))
    return keep, f"{len(keep)} of {len(sites)} sites, named on the command line"


def restrict_to_sites(keep: set[str], tables: dict[str, list[dict]]) -> None:
    """Cut every table down to the chosen sites, in place.

    Done here and not by the guards downstream. Most of them would skip an
    orphan row anyway - add_ages looks its sample up and moves on - but the
    tables that cost the time are read and walked all the same, and the
    savings this whole selection is about would not materialise.
    """
    tables["site"][:] = [r for r in tables["site"] if r["site_id"] in keep]
    tables["entity"][:] = [r for r in tables["entity"] if r["site_id"] in keep]
    entity_ids = {r["entity_id"] for r in tables["entity"]}
    tables["sample"][:] = [r for r in tables["sample"]
                           if r["entity_id"] in entity_ids]
    sample_ids = {r["sample_id"] for r in tables["sample"]}

    for name in ("d18o", "d13c", "sisal_chronology", "original_chronology",
                 "hiatus", "gap"):
        tables[name][:] = [r for r in tables[name]
                           if r["sample_id"] in sample_ids]
    tables["dating"][:] = [r for r in tables["dating"]
                           if r["entity_id"] in entity_ids]
    tables["composite_link_entity"][:] = [
        r for r in tables["composite_link_entity"]
        if r["composite_entity_id"] in entity_ids]


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
@prefix crmarchaeo: <http://www.cidoc-crm.org/extensions/crmarchaeo/> .
@prefix sosa:    <http://www.w3.org/ns/sosa/> .
@prefix skos:    <http://www.w3.org/2004/02/skos/core#> .
@prefix time:    <http://www.w3.org/2006/time#> .
@prefix qudt:    <http://qudt.org/schema/qudt/> .
@prefix geolod:  <http://w3id.org/geo-lod/> .
@prefix strat:   <http://w3id.org/geo-lod/strat/> .

<http://w3id.org/geo-lod/sisal/>
    a owl:Ontology ;
    rdfs:label   "geo-lod SISAL Extension"@en ;
    rdfs:comment "Speleothem records from SISAL v3: caves, speleothems, samples, stable-isotope observations and their competing age models."@en ;
    owl:imports  <http://w3id.org/geo-lod/> ;
    owl:imports  <http://w3id.org/geo-lod/strat/> .

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

geolod:beyondOutermostAgeControlPoint
    a owl:DatatypeProperty ;
    rdfs:domain  geolod:AgeAssignment ;
    rdfs:range   xsd:boolean ;
    rdfs:label   "beyond outermost age control point"@en ;
    rdfs:comment "True where the sample lies outside the depth span this model's own dating points cover, so its age rests on extrapolation and not on interpolation. Stands next to geolod:extrapolatedBeyondDatingRange, which infers the same thing from a negative age; that inference misfires on a still-growing speleothem, where an age of -50 a is the outermost control point rather than a step past it."@en .

# -- Dating points ---------------------------------------------------------
# The class is strat:AgeControlPoint from <http://w3id.org/geo-lod/strat/>,
# not a geolod: one: the WD1 core and a speleothem constrain their age models
# with the same kind of thing, and two names for it would have to be merged
# again in S4. The quantities stay geolod:, because strat: states depths in
# metres and ages in years b2k while SISAL states millimetres and geo-lod
# follows ka BP.

geolod:DatingMethod
    a owl:Class ;
    rdfs:subClassOf skos:Concept ;
    rdfs:label   "Dating Method"@en ;
    rdfs:comment "How a control point was obtained. Four values in the cut: two mass-spectrometric U/Th methods, a combination of methods, and the observation that the speleothem is still growing."@en .

geolod:datingMethod
    a owl:ObjectProperty ;
    rdfs:domain  strat:AgeControlPoint ;
    rdfs:range   geolod:DatingMethod ;
    rdfs:label   "dating method"@en .

geolod:constrainsSpeleothem
    a owl:ObjectProperty ;
    rdfs:domain  strat:AgeControlPoint ;
    rdfs:range   geolod:Speleothem ;
    rdfs:label   "constrains speleothem"@en ;
    rdfs:comment "The record this control point dates. Sits on the speleothem and not on a sample: a dating point has its own depth and rarely coincides with a sampling position."@en .

geolod:usedInAgeModel
    a owl:ObjectProperty ;
    rdfs:domain  strat:AgeControlPoint ;
    rdfs:range   geolod:AgeModel ;
    rdfs:label   "used in age model"@en ;
    rdfs:comment "The models that took this point as a constraint. The models disagree on that: of 1177 dating points in the cut, StalAge used 822 and OxCal 109. Without it, a query asking what an age rests on would answer with points the model never saw."@en .

geolod:datingLabId
    a owl:DatatypeProperty ;
    rdfs:domain  strat:AgeControlPoint ;
    rdfs:range   xsd:string ;
    rdfs:label   "laboratory identifier"@en .

geolod:materialDated
    a owl:DatatypeProperty ;
    rdfs:domain  strat:AgeControlPoint ;
    rdfs:range   xsd:string ;
    rdfs:label   "material dated"@en .

# -- Hiatus and gap --------------------------------------------------------
# Two different statements, and the difference is the one the figures already
# draw: a dashed section means no samples were taken, a hiatus means nothing
# was deposited. The first is about the record, the second about the archive,
# so they get different classes rather than one class with a type.

geolod:GrowthHiatus
    a owl:Class ;
    rdfs:subClassOf crmarchaeo:A3_Stratigraphic_Interface ;
    rdfs:label   "Growth Hiatus"@en ;
    rdfs:comment "A surface inside a speleothem across which nothing was deposited. A statement about the archive: the time above and the time below it do not join up, however densely either is sampled."@en .

geolod:RecordGap
    a owl:Class ;
    rdfs:subClassOf crm:E13_Attribute_Assignment ;
    rdfs:label   "Record Gap"@en ;
    rdfs:comment "SISAL's statement that the isotope record does not cover this position. An assignment, not an interface: the carbonate is there, the measurements are not, and a later study can close the gap without changing the speleothem."@en .

geolod:inSpeleothem
    a owl:ObjectProperty ;
    rdfs:range   geolod:Speleothem ;
    rdfs:label   "in speleothem"@en ;
    rdfs:comment "The record a hiatus or a gap belongs to."@en .

geolod:markedBySample
    a owl:ObjectProperty ;
    rdfs:range   geolod:SpeleothemSample ;
    rdfs:label   "marked by sample"@en ;
    rdfs:comment "The sample row that carries the mark in SISAL. Kept so that the depth in the graph can be traced to the row it came from; the hiatus itself is not that sample."@en .

geolod:respectedByAgeModel
    a owl:ObjectProperty ;
    rdfs:domain  geolod:GrowthHiatus ;
    rdfs:range   geolod:AgeModel ;
    rdfs:label   "respected by age model"@en ;
    rdfs:comment "The models that treated this hiatus as a break instead of interpolating across it. Of the 21 hiatuses in the cut, linear interpolation respected 16 and OxCal none, which is a property of the chronologies and not of the cave."@en .

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

geolod:isComposite
    a owl:DatatypeProperty ;
    rdfs:domain  geolod:Speleothem ;
    rdfs:range   xsd:boolean ;
    rdfs:label   "is composite"@en ;
    rdfs:comment "True where the record is assembled from several speleothems. A composite has no depth scale of its own - its dating points may carry an age and no depth, and the shapes make that exception explicit."@en .

geolod:composedOf
    a owl:ObjectProperty ;
    rdfs:domain  geolod:Speleothem ;
    rdfs:range   geolod:Speleothem ;
    rdfs:label   "composed of"@en ;
    rdfs:comment "A record the composite is built from. SISAL keeps the relation in composite_link_entity; without it a composite looks like an ordinary speleothem whose samples come from nowhere."@en .

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

# -- Archaeological enrichment ---------------------------------------------
# geo-lod's own reading of a cave, not part of SISAL. Carried over unchanged
# from the ontology the plotting script used to write, so no IRI moves.

geolod:ArchaeologicalCaveSite
    a owl:Class ;
    rdfs:subClassOf geolod:Cave ;
    rdfs:subClassOf crmarchaeo:A2_Stratigraphic_Volume_Unit ;
    rdfs:label   "Archaeological Cave Site"@en ;
    rdfs:comment "A SISAL cave site that also carries confirmed or probable archaeological evidence (art, human occupation, skeletal remains, inscriptions). Modelling mirrors the CIArchaeologicalSite pattern from ci_pipeline.py."@en .

geolod:ArchaeologicalContext
    a owl:Class ;
    rdfs:subClassOf crm:E55_Type ;
    rdfs:label   "Archaeological Context"@en ;
    rdfs:comment "Controlled vocabulary class for broader cultural-temporal context categories."@en .

geolod:PalaeolithicContext
    a geolod:ArchaeologicalContext, owl:NamedIndividual ;
    rdfs:label "Palaeolithic Context"@en .

geolod:PrehistoricContext
    a geolod:ArchaeologicalContext, owl:NamedIndividual ;
    rdfs:label "Prehistoric Context"@en .

geolod:PalaeontologicalContext
    a geolod:ArchaeologicalContext, owl:NamedIndividual ;
    rdfs:label "Palaeontological Context"@en .

geolod:HistoricContext
    a geolod:ArchaeologicalContext, owl:NamedIndividual ;
    rdfs:label "Historic Context"@en .

geolod:MesoamericanContext
    a geolod:ArchaeologicalContext, owl:NamedIndividual ;
    rdfs:label "Mesoamerican Context"@en .

geolod:UThChronology
    a owl:Class ;
    rdfs:subClassOf geolod:Chronology ;
    rdfs:label   "U-Th Chronology"@en ;
    rdfs:comment "A depth-age model for a speleothem, built on U-Th dating points: the seven SISAL v3 models and the chronology as originally published. geo_lod_core.ttl names this class as the SISAL counterpart of geolod:IceCoreChronology but does not define it; defined here, so that the eight chronology individuals reach crmsci:S4_Observation through geolod:Chronology like every other class in the graph."@en .

geolod:isUNESCOWorldHeritage
    a owl:DatatypeProperty ;
    rdfs:domain  geolod:Cave ;
    rdfs:range   xsd:boolean ;
    rdfs:label   "is UNESCO World Heritage"@en .

geolod:unescoWHId
    a owl:ObjectProperty ;
    rdfs:domain  geolod:Cave ;
    rdfs:label   "UNESCO WH identifier"@en ;
    rdfs:comment "URI of the UNESCO World Heritage list entry."@en .

geolod:countD18OSamples
    a owl:DatatypeProperty ;
    rdfs:domain  geolod:Cave ;
    rdfs:range   xsd:integer ;
    rdfs:label   "number of δ¹⁸O samples"@en ;
    rdfs:comment "Samples of this cave carrying a δ¹⁸O measurement, over all its speleothems. Recomputed from the database with every export, not carried over from a hand-kept list."@en .

geolod:countD13CSamples
    a owl:DatatypeProperty ;
    rdfs:domain  geolod:Cave ;
    rdfs:range   xsd:integer ;
    rdfs:label   "number of δ¹³C samples"@en .

geolod:collectedFrom
    a owl:ObjectProperty ;
    rdfs:domain  geolod:SpeleothemSample ;
    rdfs:range   geolod:Speleothem ;
    rdfs:label   "collected from"@en .
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
        ("crm", CRM), ("crmsci", CRMSCI), ("crmarchaeo", CRMARCHAEO), ("geo", GEO), ("sf", SF),
        ("geolod", GEOLOD), ("sisal", SISAL), ("trs", TRS), ("sosa", SOSA),
        ("strat", STRAT),
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

    # Unit and source sit here, not on each of the 50 456 observations: both
    # hold for every measurement of the property, and repeating them per node
    # is 100 000 triples that say the same thing over and over.
    for local, label in [
        ("MeasurementType_d18O", "δ¹⁸O measurement"),
        ("MeasurementType_d13C", "δ¹³C measurement"),
    ]:
        node = GEOLOD[local]
        g.add((node, RDF.type, GEOLOD["MeasurementType"]))
        g.add((node, RDF.type, OWL.NamedIndividual))
        g.add((node, RDFS.label, Literal(label, lang="en")))

    for local, cls, label in [
        ("Delta18OProperty_speleothem", "Delta18OProperty", "δ¹⁸O of speleothem calcite"),
        ("Delta13CProperty_speleothem", "Delta13CProperty", "δ¹³C of speleothem calcite"),
    ]:
        node = GEOLOD[local]
        g.add((node, RDF.type, GEOLOD[cls]))
        g.add((node, RDF.type, OWL.NamedIndividual))
        g.add((node, RDFS.label, Literal(label, lang="en")))
        g.add((node, QUDT["unit"], UNIT["PERMILLE"]))
        g.add((node, PROV.wasDerivedFrom, GEOLOD["SISALv3_DataSource"]))


def add_sites(g: Graph, sites: list[dict]) -> dict[str, URIRef]:
    """The six caves of the cut, on the nodes the catalogue already made.

    Only the mapping is built here; the catalogue writes name, position and
    counts for all 365. Kept as its own function so that a run over a
    different site selection needs no other change.
    """
    uris: dict[str, URIRef] = {}
    for row in sites:
        site_id = int(row["site_id"])
        site = GEOLOD[f"Cave_site_{site_id:04d}"]
        uris[row["site_id"]] = site

        g.add((site, RDF.type, GEOLOD["Cave"]))
        g.add((site, RDFS.label, Literal(row["site_name"], lang="en")))
    return uris


def add_cave_catalogue(g: Graph, ann: Graph) -> tuple[int, int, int]:
    """One cave node per site SISAL knows, not only per site in the cut.

    Two files, two graphs, and the split is the point. catalogue/sites.csv
    comes straight out of the database and holds what SISAL states: name,
    position, elevation, measurement counts. That goes into the core graph.
    data/curated/sisal_site_annotations.csv is geo-lod's own work - the
    archaeological reading of a cave, its Wikidata entity, its UNESCO listing -
    and no database export will ever produce it. That goes into its own graph,
    with its own source node.

    The separation is a provenance requirement, not tidiness. The Palaeolithic
    paintings at Villars are not in any SISAL table, and a cave node that
    carries them under prov:wasDerivedFrom geolod:SISALv3_DataSource states
    something false about where they come from.

    Replaces the old SISAL/v_sites_all.csv, which mixed both and was a v2-era
    snapshot: it held 305 sites where SISAL v3 has 365. The 305 ids and names
    are unchanged, so nothing has to be migrated; 60 caves were simply missing.
    """
    sites = read_csv(CATALOGUE_DIR / "sites.csv",
                     "Fetch the cut with SISAL/sisal_import.py; the catalogue "
                     "needs an export repo that carries postgres/queries.yaml "
                     "with a `catalogue:` block.")
    annotations = {
        row["site_id"]: row
        for row in read_csv(CURATED_DIR / "sisal_site_annotations.csv",
                            "This file is maintained by hand in geo-lod and is "
                            "not a database export.")
    }

    add_annotation_source(ann)

    collection = GEOLOD["SISAL_Cave_Collection"]
    all_sites = GEOLOD["AllPalaeoclimateSites_Collection"]
    arch_collection = GEOLOD["SISAL_Archaeological_Cave_Collection"]
    for node, label in [
        (collection, "SISAL cave collection"),
        (arch_collection, "SISAL caves with an archaeological record"),
    ]:
        g.add((node, RDF.type, GEO["FeatureCollection"]))
        g.add((node, RDFS.label, Literal(label, lang="en")))

    n_arch = 0
    n_screened = 0
    for row in sites:
        site_id = int(row["site_id"])
        site = GEOLOD[f"Cave_site_{site_id:04d}"]

        g.add((site, RDF.type, GEOLOD["Cave"]))
        g.add((site, RDF.type, OWL.NamedIndividual))
        g.add((site, RDFS.label, Literal(row["site_name"], lang="en")))
        g.add((site, GEOLOD["siteId"], Literal(site_id, datatype=XSD.integer)))
        g.add((site, PROV.wasDerivedFrom, GEOLOD["SISALv3_DataSource"]))
        g.add((collection, RDFS.member, site))
        g.add((all_sites, RDFS.member, site))

        lat, lon = num(row.get("latitude")), num(row.get("longitude"))
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
        for column, prop in [("n_d18o_samples", "countD18OSamples"),
                             ("n_d13c_samples", "countD13CSamples")]:
            count = num(row.get(column))
            if count is not None:
                g.add((site, GEOLOD[prop],
                       Literal(int(count), datatype=XSD.integer)))

        note = annotations.get(row["site_id"])
        if note:
            n_screened += 1
            n_arch += add_site_annotation(ann, site, note, arch_collection)
    return len(sites), n_screened, n_arch


def add_annotation_source(g: Graph) -> None:
    """The curation as a source in its own right.

    Without this node the annotations would hang off the cave with no author,
    and a consumer merging the file into a wider graph could not tell a SISAL
    measurement from a reading geo-lod added.
    """
    source = GEOLOD["GeoLodSiteAnnotations_DataSource"]
    g.add((source, RDF.type, GEOLOD["DataSource"]))
    g.add((source, RDF.type, OWL.NamedIndividual))
    g.add((source, RDFS.label,
           Literal("geo-lod cave site annotations", lang="en")))
    g.add((source, RDFS.comment,
           Literal("Archaeological, Wikidata and UNESCO annotations added to "
                   "SISAL cave sites by the geo-lod project. Not part of "
                   "SISAL v3 and not reproducible from it.", lang="en")))
    g.add((source, DCT.creator, ORCID_FLO))
    g.add((source, DCT.source, URIRef(
        "https://github.com/Research-Squirrel-Engineers/"
        "GeoScience-FAIRification-LOD")))
    g.add((source, DCT.license, URIRef(
        "https://creativecommons.org/licenses/by/4.0/")))


def add_site_annotation(g: Graph, site: URIRef, note: dict,
                        arch_collection: URIRef) -> int:
    """geo-lod's own reading of a cave: archaeology, Wikidata, UNESCO."""
    def value(column: str) -> str:
        return (note.get(column) or "").strip()

    source = GEOLOD["GeoLodSiteAnnotations_DataSource"]

    # Screened and found nothing is a result, and a different one from never
    # looked at. 305 caves were screened, 37 positively; the 60 caves SISAL v3
    # added since carry neither triple, and a map can tell the three apart.
    g.add((site, GEOLOD["screenedForArchaeology"],
           Literal(True, datatype=XSD.boolean)))
    g.add((site, PROV.wasDerivedFrom, source))

    is_arch = value("isArchaeologicalSite").lower() == "true"
    if is_arch:
        g.add((site, RDF.type, GEOLOD["ArchaeologicalCaveSite"]))
        g.add((site, RDF.type, CRMARCHAEO["A2_Stratigraphic_Volume_Unit"]))
        g.add((arch_collection, RDFS.member, site))
        if value("arch_category"):
            g.add((site, GEOLOD["archaeologicalCategory"],
                   Literal(value("arch_category"), lang="en")))
        if value("arch_broader_context"):
            g.add((site, GEOLOD["archaeologicalBroaderContext"],
                   GEOLOD[value("arch_broader_context")]))
        if value("arch_note"):
            g.add((site, SKOS.note, Literal(value("arch_note"), lang="en")))
        if value("arch_confidence"):
            # No language tag. The shape constrains this with
            # sh:in ( "high" "medium" "low" ), and "high"@en is not "high" -
            # it is a controlled value, not prose.
            g.add((site, GEOLOD["archaeologicalConfidence"],
                   Literal(value("arch_confidence"))))

    if value("wikidata_qid"):
        g.add((site, OWL.sameAs,
               URIRef(f"http://www.wikidata.org/entity/{value('wikidata_qid')}")))
    if value("osm_url"):
        g.add((site, RDFS.seeAlso, URIRef(value("osm_url"))))
    if value("isUNESCO").lower() in ("true", "yes"):
        g.add((site, GEOLOD["isUNESCOWorldHeritage"],
               Literal(True, datatype=XSD.boolean)))
        if value("unesco_wh_id"):
            g.add((site, GEOLOD["unescoWHId"],
                   URIRef(f"https://whc.unesco.org/en/list/{value('unesco_wh_id')}")))
    return 1 if is_arch else 0


def add_entities(g: Graph, entities: list[dict],
                 site_uris: dict[str, URIRef],
                 composite_links: list[dict] | None = None) -> dict[str, URIRef]:
    """The speleothems, with their status and, where superseded, their successor.

    Composites are marked here rather than inferred later. SISAL keeps them in
    composite_link_entity and nowhere else: entity_status calls a composite
    "current" like any other record, so without this table a composite is
    indistinguishable from an ordinary speleothem - and the difference matters
    at the dating points, where a composite carries ages without depths
    because it has no depth scale of its own.
    """
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

    for link in composite_links or []:
        composite = uris.get(link["composite_entity_id"])
        member = uris.get(link["single_entity_id"])
        if composite is None:
            continue
        g.add((composite, GEOLOD["isComposite"],
               Literal(True, datatype=XSD.boolean)))
        if member is not None:
            g.add((composite, GEOLOD["composedOf"], member))
    return uris


def add_samples(g: Graph, samples: list[dict],
                entity_uris: dict[str, URIRef]) -> dict[str, URIRef]:
    uris: dict[str, URIRef] = {}
    for row in samples:
        sample_id = int(row["sample_id"])
        sample = SISAL[f"sample_{sample_id:07d}"]
        uris[row["sample_id"]] = sample

        g.add((sample, RDF.type, GEOLOD["SpeleothemSample"]))
        g.add((sample, GEOLOD["sampleId"], Literal(sample_id, datatype=XSD.integer)))

        entity = entity_uris.get(row["entity_id"])
        if entity is not None:
            g.add((sample, GEOLOD["collectedFrom"], entity))

        # No rdfs:label: at 35 777 samples it would restate the id and the
        # depth, both of which are already triples.
        depth = num(row.get("depth_sample"))
        if depth is not None:
            g.add((sample, GEOLOD["atDepth_mm"], dec(depth, DEC_DEPTH)))

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
                 isotope: str,
                 leading_by_sample: dict[str, tuple[str, float]]) -> tuple[int, int]:
    """δ¹⁸O or δ¹³C observations, one per measured sample.

    The age is repeated here from the sample. SISAL states it once, per
    sample, and that is where it belongs - but ontology/shapes asks for it on
    the observation, EPICA writes it there, and a consumer plotting a series
    should not have to join through the sample to get an x-axis.

    Returns the number written and the number left without an age. The second
    figure is not a defect: 4688 δ¹⁸O and 3093 δ¹³C measurements sit on
    samples no chronology reached, and the old v_data_*.csv simply never
    contained them.
    """
    meta = {
        "d18o": ("Delta18OSpeleothemObservation", "Delta18OProperty_speleothem",
                 "MeasurementType_d18O",
                 "d18o_measurement", "d18o_precision", "δ¹⁸O"),
        "d13c": ("Delta13CSpeleothemObservation", "Delta13CProperty_speleothem",
                 "MeasurementType_d13C",
                 "d13c_measurement", "d13c_precision", "δ¹³C"),
    }[isotope]
    cls, prop, mtype, value_col, precision_col, label = meta

    written = 0
    undated = 0
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
        g.add((obs, SOSA["hasFeatureOfInterest"], sample))
        g.add((obs, SOSA["observedProperty"], GEOLOD[prop]))
        # Both properties, as EPICA/epica_rdf.py writes them. They look like
        # a duplicate and are not: geolod:measuredValue is what
        # ontology/shapes/core_shapes.ttl requires exactly one of, and
        # sosa:hasSimpleResult is what a SOSA consumer reads. Dropping either
        # breaks one of the two.
        g.add((obs, SOSA["hasSimpleResult"], dec(value, DEC_VALUE)))
        g.add((obs, GEOLOD["measuredValue"], dec(value, DEC_VALUE)))
        g.add((obs, GEOLOD["measurementType"], GEOLOD[mtype]))
        g.add((obs, PROV.wasDerivedFrom, GEOLOD["SISALv3_DataSource"]))
        g.add((sample, GEOLOD["hasObservation"], obs))

        chosen = leading_by_sample.get(row["sample_id"])
        if chosen is None:
            undated += 1
        else:
            model, age_ka = chosen
            g.add((obs, GEOLOD["ageKaBP"], dec(age_ka, DEC_AGE)))
            g.add((obs, GEOLOD["ageChronology"], TRS[f"SISALv3-{model}"]))

        precision = num(row.get(precision_col))
        if precision is not None:
            g.add((obs, GEOLOD["standardDeviation"], dec(precision, DEC_VALUE)))
        written += 1
    return written, undated


def split_dating(dating: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """The dating table holds three kinds of row, and they are not alike.

    1177 laboratory measurements, 2 rows stating the speleothem is still
    growing, and 21 rows marking a hiatus. Only the first two carry an age.
    The third is the age models' view of a hiatus that the hiatus table states
    on its own, and running the date_used filter over it would answer a
    question it was never asked: date_used says whether a point entered a
    chronology, not whether a hiatus exists. SB61 proves the point - its row
    reads no, and the hiatus is in the hiatus table all the same.
    """
    measurements, forming, hiatus_events = [], [], []
    for row in dating:
        date_type = (row.get("date_type") or "").strip()
        if date_type == "Event; hiatus":
            hiatus_events.append(row)
        elif date_type.startswith("Event;"):
            forming.append(row)
        else:
            measurements.append(row)
    return measurements, forming, hiatus_events


DATING_METHODS = {
    "MC-ICP-MS U/Th": ("DatingMethod_mc_icp_ms_u_th", "MC-ICP-MS U/Th",
                       "Uranium-thorium disequilibrium dating on a multi-collector "
                       "inductively coupled plasma mass spectrometer."),
    "TIMS": ("DatingMethod_tims", "TIMS",
             "Uranium-thorium disequilibrium dating on a thermal ionisation "
             "mass spectrometer."),
    "ICP-MS U/Th Other": ("DatingMethod_icp_ms_u_th_other", "ICP-MS U/Th (other)",
                    "Uranium-thorium disequilibrium dating on an inductively "
                    "coupled plasma mass spectrometer other than a "
                    "multi-collector one."),
    "U/Th unspecified": ("DatingMethod_u_th_unspecified", "U/Th unspecified",
                         "Uranium-thorium disequilibrium dating; the source "
                         "does not say on which instrument."),
    "Multiple methods": ("DatingMethod_multiple", "multiple methods",
                         "More than one method combined into a single reported age."),
    "Event; actively forming": ("DatingMethod_actively_forming", "actively forming",
                                "Not a laboratory measurement: the top of the record "
                                "was observed to be growing, which dates it to the "
                                "year of collection."),
}


def add_dating_methods(g: Graph) -> None:
    for _, (local, label, comment) in sorted(DATING_METHODS.items()):
        node = GEOLOD[local]
        g.add((node, RDF.type, GEOLOD["DatingMethod"]))
        g.add((node, RDF.type, OWL.NamedIndividual))
        g.add((node, SKOS.prefLabel, Literal(label, lang="en")))
        g.add((node, SKOS.definition, Literal(comment, lang="en")))


def models_using(row: dict) -> list[str]:
    """The models that took this row as a constraint.

    An empty cell is not a no: it means the model was not run for that
    speleothem at all. Only an explicit yes counts, which is why the counts
    per model differ so widely.
    """
    return [key for key, _, _ in CHRONOLOGY_MODELS
            if (row.get(f"date_used_{key}") or "").strip() == "yes"]


def add_dating(g: Graph, measurements: list[dict], forming: list[dict],
               entity_uris: dict[str, URIRef]) -> dict[str, int]:
    """The dating points, as strat:AgeControlPoint.

    date_used = 'yes' filters the measurements (A4). The two actively-forming
    rows pass unfiltered: both read yes, and a record whose top is dated by
    observation has no other point up there.
    """
    tally = {"points": 0, "dropped": 0, "forming": 0, "with_age": 0}

    for row in measurements + forming:
        is_forming = row in forming
        if not is_forming and (row.get("date_used") or "").strip() != "yes":
            tally["dropped"] += 1
            continue

        entity = entity_uris.get(row["entity_id"])
        if entity is None:
            continue

        point = SISAL[f"datingpoint_{int(row['dating_id']):06d}"]
        g.add((point, RDF.type, STRAT["AgeControlPoint"]))
        g.add((point, GEOLOD["constrainsSpeleothem"], entity))
        g.add((point, PROV.wasDerivedFrom, GEOLOD["SISALv3_DataSource"]))

        method = DATING_METHODS.get((row.get("date_type") or "").strip())
        if method:
            g.add((point, GEOLOD["datingMethod"], GEOLOD[method[0]]))

        depth = num(row.get("depth_dating"))
        if depth is not None:
            g.add((point, GEOLOD["atDepth_mm"], dec(depth, DEC_DEPTH)))

        age = num(row.get("corr_age"))
        if age is not None:
            g.add((point, GEOLOD["ageKaBP"], dec(age / 1000.0, DEC_AGE)))
            tally["with_age"] += 1
            for column, prop in [("corr_age_uncert_pos", "ageUncertaintyPos_ka"),
                                 ("corr_age_uncert_neg", "ageUncertaintyNeg_ka")]:
                value = num(row.get(column))
                if value is not None:
                    g.add((point, GEOLOD[prop], dec(value / 1000.0, DEC_AGE)))

        for column, prop in [("lab_num", "datingLabId"),
                             ("material_dated", "materialDated")]:
            text = (row.get(column) or "").strip()
            if text:
                g.add((point, GEOLOD[prop], Literal(text)))

        for key in models_using(row):
            g.add((point, GEOLOD["usedInAgeModel"], GEOLOD[f"AgeModel_{key}"]))

        tally["points"] += 1
        if is_forming:
            tally["forming"] += 1
    return tally


def add_hiatus_and_gaps(g: Graph, hiatus: list[dict], gaps: list[dict],
                        hiatus_events: list[dict], samples_by_id: dict[str, dict],
                        entity_uris: dict[str, URIRef],
                        sample_uris: dict[str, URIRef]) -> dict[str, int]:
    """The 21 hiatuses and 4 gaps, anchored at the sample row that marks them.

    The hiatus table and the Event; hiatus rows of the dating table are the
    same 21, one to one over entity and depth, checked 2026-08-12. The dating
    row is not a second hiatus; it carries which models respected it, and that
    is what is taken from it here.
    """
    tally = {"hiatus": 0, "gaps": 0, "model_statements": 0,
             "no_model_statement": 0, "unmatched_events": 0}

    events_by_key: dict[tuple[str, float], dict] = {}
    for row in hiatus_events:
        depth = num(row.get("depth_dating"))
        if depth is not None:
            events_by_key[(row["entity_id"], round(depth, 3))] = row

    matched: set[tuple[str, float]] = set()

    for rows, cls, counter in [(hiatus, "GrowthHiatus", "hiatus"),
                               (gaps, "RecordGap", "gaps")]:
        for row in rows:
            sample_id = row["sample_id"]
            sample_row = samples_by_id.get(sample_id)
            if sample_row is None:
                continue

            prefix = "hiatus" if cls == "GrowthHiatus" else "gap"
            node = SISAL[f"{prefix}_{int(sample_id):07d}"]
            g.add((node, RDF.type, GEOLOD[cls]))
            g.add((node, PROV.wasDerivedFrom, GEOLOD["SISALv3_DataSource"]))

            entity = entity_uris.get(sample_row["entity_id"])
            if entity is not None:
                g.add((node, GEOLOD["inSpeleothem"], entity))

            sample = sample_uris.get(sample_id)
            if sample is not None:
                g.add((node, GEOLOD["markedBySample"], sample))
                if cls == "RecordGap":
                    g.add((node, CRM["P140_assigned_attribute_to"], sample))

            depth = num(sample_row.get("depth_sample"))
            if depth is not None:
                g.add((node, GEOLOD["atDepth_mm"], dec(depth, DEC_DEPTH)))

            if cls == "GrowthHiatus" and depth is not None:
                key = (sample_row["entity_id"], round(depth, 3))
                event = events_by_key.get(key)
                if event is not None:
                    matched.add(key)
                    models = models_using(event)
                    if not models:
                        tally["no_model_statement"] += 1
                    for model in models:
                        g.add((node, GEOLOD["respectedByAgeModel"],
                               GEOLOD[f"AgeModel_{model}"]))
                        tally["model_statements"] += 1
            tally[counter] += 1

    tally["unmatched_events"] = len(events_by_key) - len(matched)
    return tally


def control_point_depths(measurements: list[dict], forming: list[dict],
                         samples_by_id: dict[str, dict]) -> dict[tuple[str, str],
                                                                tuple[float, float]]:
    """The depth span each model's own control points cover, per speleothem.

    Depth and not age, because extrapolation is a statement about where a
    sample sits relative to the dated part of the record. Measured in ages it
    would test the model against its own output.

    The seven SISAL models read their own date_used_* column. original gets no
    span at all: those columns belong to sisal_chronology and say nothing about
    what the original authors had in front of them. Falling back to the overall
    date_used would put a span under original built from points no triple in
    the graph attributes to it, and the claim would be about someone else's
    publication rather than about this release. Its assignments therefore carry
    no geolod:beyondOutermostAgeControlPoint and count as untestable instead.
    """
    spans: dict[tuple[str, str], tuple[float, float]] = {}

    def extend(entity_id: str, model: str, depth: float) -> None:
        key = (entity_id, model)
        low, high = spans.get(key, (depth, depth))
        spans[key] = (min(low, depth), max(high, depth))

    for row in measurements + forming:
        depth = num(row.get("depth_dating"))
        if depth is None:
            continue
        for model in models_using(row):
            extend(row["entity_id"], model, depth)
    return spans


def collect_ages(chronology: list[dict], original: list[dict]) -> dict:
    """Every model's age per sample, read once and reused twice.

    Its own pass because the observations need the leading age too: SISAL
    hangs the age on the sample, the shapes ask for it on the observation, and
    a second read of the two chronology tables to answer that would be a
    second chance to answer it differently.
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
    return by_sample


def leading_ages(by_sample: dict, superseded: set[str]) -> dict[str, tuple[str, float]]:
    """The one age per sample geo-lod follows, in ka BP.

    lin_interp where the sample has one, otherwise the original_chronology
    age, and never for a sample of a superseded speleothem: those hold no
    lin_interp age at all but 3621 original ages, and letting them lead would
    return records SISAL has retired to any query asking for the leading age.
    """
    leading: dict[str, tuple[str, float]] = {}
    for sample_id, entries in by_sample.items():
        ages = {key: value for key, value, _, _ in entries}
        if "lin_interp" in ages:
            leading[sample_id] = ("lin_interp", ages["lin_interp"] / 1000.0)
        elif "original" in ages and sample_id not in superseded:
            leading[sample_id] = ("original", ages["original"] / 1000.0)
    return leading


def add_ages(core: Graph, alt: Graph, by_sample: dict,
             sample_uris: dict[str, URIRef],
             leading_by_sample: dict[str, tuple[str, float]],
             spans: dict[tuple[str, str], tuple[float, float]] | None = None,
             samples_by_id: dict[str, dict] | None = None) -> dict[str, int]:
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
    tally = {"assignments": 0, "leading_lin_interp": 0, "leading_original": 0,
             "no_leading": 0, "extrapolated": 0,
             "beyond_control": 0, "within_control": 0, "no_control_span": 0}

    for sample_id, entries in by_sample.items():
        sample = sample_uris.get(sample_id)
        if sample is None:
            continue
        chosen = leading_by_sample.get(sample_id)
        leading = chosen[0] if chosen else None
        if leading == "lin_interp":
            tally["leading_lin_interp"] += 1
        elif leading == "original":
            tally["leading_original"] += 1
        else:
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

            # The second reading, from the dating points themselves (S3c.3).
            # It answers the question the first one only guesses at: does this
            # sample lie inside the depth span this model's own control points
            # cover? Written for both answers, because a false here is a
            # statement and its absence is not - and left off entirely where
            # there is no span, which is every original_chronology age and
            # every model that never reached this speleothem.
            if spans is not None and samples_by_id is not None:
                sample_row = samples_by_id.get(sample_id)
                depth = num(sample_row.get("depth_sample")) if sample_row else None
                span = spans.get((sample_row["entity_id"], key)) if sample_row else None
                if depth is not None and span is not None:
                    beyond = depth < span[0] or depth > span[1]
                    target.add((node, GEOLOD["beyondOutermostAgeControlPoint"],
                                Literal(beyond, datatype=XSD.boolean)))
                    if beyond:
                        tally["beyond_control"] += 1
                    else:
                        tally["within_control"] += 1
                else:
                    tally["no_control_span"] += 1

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

# Development format first. N-Triples writes the two large graphs four times
# faster than Turtle (12s against 57s here) at three times the size, so it is
# what a development run gets and .gitignore keeps it out. Turtle is the
# release format: smaller, readable, versioned. Neither carries a blank node,
# so both are byte-stable across runs.
DATA_FORMATS: dict[str, tuple[str, str]] = {
    "nt": ("nt", ".nt"),
    "turtle": ("turtle", ".ttl"),
}
DEFAULT_DATA_FORMAT = "nt"

STEPS = [
    "reading the cut",
    "vocabularies",
    "cave catalogue",
    "speleothems and samples",
    "isotope observations",
    "dating points, hiatuses and gaps",
    "age assignments",
    "provenance",
    "writing the annotations",
    "writing the core graph",
    "writing the chronologies",
    "writing the dating points",
]


def partial_note(selection_label: str, sites: list[dict]) -> Literal:
    """The one sentence that keeps a partial graph from passing as a release.

    Written into every data graph of a partial run rather than into the core
    graph alone: the files are separable, and a consumer who picks up
    sisal_v3_dating.ttl on its own has no other way to find out that the cut
    behind it is five caves rather than 365.
    """
    names = ", ".join(sorted(row["site_name"] for row in sites))
    return Literal(f"Partial graph: {selection_label} ({names}). "
                   f"Not a release.", lang="en")


def build(data_format: str = DEFAULT_DATA_FORMAT,
          sites_spec: str = "all") -> bool:
    serializer, suffix = DATA_FORMATS[data_format]

    print("\n" + "=" * 72)
    print("  SISAL cut to RDF")
    print("=" * 72)
    print(f"  Input:  {TABLES_DIR}")
    print(f"  Format: {data_format} ({suffix}) for the two large graphs, "
          f"Turtle for the rest")

    beat = Heartbeat(len(STEPS)).start()
    try:
        beat.step(STEPS[0])
        tables = {name: read_table(name) for name in
                  ("site", "entity", "sample", "d18o", "d13c",
                   "sisal_chronology", "original_chronology", "dating",
                   "hiatus", "gap", "composite_link_entity")}

        keep, selection_label = resolve_sites(sites_spec, tables["site"])
        partial = keep is not None
        if partial:
            restrict_to_sites(keep, tables)
        print(f"  Sites:  {selection_label}")
        if partial:
            names = ", ".join(sorted(row["site_name"] for row in tables["site"]))
            print(f"          {names}")
            print("  ⚠  PARTIAL GRAPH - not a release. Run with --sites all "
                  "before publishing anything from SISAL/rdf/.")

        sites = tables["site"]
        entities = tables["entity"]
        samples = tables["sample"]
        d18o = tables["d18o"]
        d13c = tables["d13c"]
        chronology = tables["sisal_chronology"]
        original = tables["original_chronology"]
        dating = tables["dating"]
        hiatus = tables["hiatus"]
        gaps = tables["gap"]
        composite_links = tables["composite_link_entity"]
        samples_by_id = {row["sample_id"]: row for row in samples}

        print(f"  {len(sites)} sites, {len(entities)} speleothems, "
              f"{len(samples)} samples")

        superseded_entities = {
            row["entity_id"] for row in entities
            if (row.get("entity_status") or "").strip() == "superseded"}
        superseded_samples = {row["sample_id"] for row in samples
                              if row["entity_id"] in superseded_entities}
        print(f"  {len(superseded_entities)} superseded speleothems, "
              f"{len(superseded_samples)} of their samples")

        beat.step(STEPS[1])
        g = Graph()
        bind_prefixes(g)
        add_vocabularies(g)

        beat.step(STEPS[2])
        ann = Graph()
        bind_prefixes(ann)
        n_catalogue, n_screened, n_arch = add_cave_catalogue(g, ann)
        print(f"  {n_catalogue} cave sites in the catalogue; "
              f"{n_screened} screened for archaeology, {n_arch} positive, "
              f"{n_catalogue - n_screened} not yet screened")

        beat.step(STEPS[3])
        site_uris = add_sites(g, sites)
        entity_uris = add_entities(g, entities, site_uris, composite_links)
        sample_uris = add_samples(g, samples, entity_uris)

        beat.step(STEPS[4])
        by_sample = collect_ages(chronology, original)
        leading = leading_ages(by_sample, superseded_samples)
        n_d18o, undated_d18o = add_isotopes(g, d18o, sample_uris, "d18o", leading)
        n_d13c, undated_d13c = add_isotopes(g, d13c, sample_uris, "d13c", leading)
        print(f"  {n_d18o} δ¹⁸O and {n_d13c} δ¹³C observations "
              f"({undated_d18o} and {undated_d13c} of them undated)")

        beat.step(STEPS[5])
        dat = Graph()
        bind_prefixes(dat)
        add_dating_methods(dat)
        measurements, forming, hiatus_events = split_dating(dating)
        dating_tally = add_dating(dat, measurements, forming, entity_uris)
        break_tally = add_hiatus_and_gaps(dat, hiatus, gaps, hiatus_events,
                                          samples_by_id, entity_uris, sample_uris)
        spans = control_point_depths(measurements, forming, samples_by_id)
        print(f"  {dating_tally['points']} dating points "
              f"({dating_tally['forming']} of them observed growth, "
              f"{dating_tally['dropped']} measurements not used by any chronology)")
        print(f"  {break_tally['hiatus']} hiatuses and {break_tally['gaps']} gaps, "
              f"{break_tally['model_statements']} model statements on them")
        if break_tally["no_model_statement"]:
            print(f"  {break_tally['no_model_statement']} hiatuses without a "
                  f"model statement (superseded Corchia records)")
        if break_tally["unmatched_events"]:
            print(f"  ⚠  {break_tally['unmatched_events']} Event; hiatus rows "
                  f"without a matching hiatus row")

        beat.step(STEPS[6])
        alt = Graph()
        bind_prefixes(alt)
        tally = add_ages(g, alt, by_sample, sample_uris, leading,
                         spans=spans, samples_by_id=samples_by_id)
        print(f"  {tally['assignments']} age assignments over eight models")
        print(f"    leading, lin_interp        {tally['leading_lin_interp']:>7d}")
        print(f"    leading, original          {tally['leading_original']:>7d}")
        print(f"    no leading age             {tally['no_leading']:>7d}")
        print(f"    extrapolated past 1950     {tally['extrapolated']:>7d}")
        print(f"    beyond its control points  {tally['beyond_control']:>7d}")
        print(f"    inside its control points  {tally['within_control']:>7d}")
        print(f"    no control span to test    {tally['no_control_span']:>7d}")

        beat.step(STEPS[7])
        add_generation_provenance(
            g,
            GEOLOD["SISAL_Dataset"],
            GEOLOD["SISAL_Generation"],
            inputs=[str(TABLES_DIR / f"{name}.csv") for name in
                    ("site", "entity", "sample", "d18o", "d13c",
                     "sisal_chronology", "original_chronology",
                     "dating", "hiatus", "gap",
                     "composite_link_entity")] + [__file__],
            agents=[ORCID_FLO],
            label="SISAL v3 RDF generation",
        )
        if partial:
            # In the graph and not only in the report, and on every file this
            # run writes. A partial file carries the same name as a complete
            # one, the report is not what a consumer reads, and the three data
            # graphs travel separately - sisal_v3_dating.ttl in particular is
            # versioned Turtle whatever the run.
            note = partial_note(selection_label, sites)
            for graph in (g, alt, dat):
                graph.add((GEOLOD["SISAL_Dataset"], OWL.versionInfo, note))
        add_generation_provenance(
            ann,
            GEOLOD["SISAL_Site_Annotations_Dataset"],
            GEOLOD["SISAL_Site_Annotations_Generation"],
            inputs=[str(CURATED_DIR / "sisal_site_annotations.csv"), __file__],
            agents=[ORCID_FLO],
            label="geo-lod cave site annotations",
        )

        onto_path = write_ontology()
        print(f"\n  ✓ {onto_path.relative_to(ROOT)}")

        # The annotations stay Turtle whatever the run: 933 triples, read by
        # people as often as by machines, and small enough that the format
        # costs nothing.
        beat.step(STEPS[8])
        ann_path = RDF_DIR / "sisal_site_annotations.ttl"
        ann.serialize(destination=str(ann_path), format="turtle")
        print(f"  ✓ {ann_path.relative_to(ROOT)}  ({len(ann):,} triples)")

        beat.step(STEPS[9])
        core_path = RDF_DIR / f"sisal_v3_core{suffix}"
        g.serialize(destination=str(core_path), format=serializer,
                    encoding="utf-8")
        print(f"  ✓ {core_path.relative_to(ROOT)}  ({len(g):,} triples)")

        beat.step(STEPS[10])
        alt_path = RDF_DIR / f"sisal_v3_chronologies{suffix}"
        alt.serialize(destination=str(alt_path), format=serializer,
                      encoding="utf-8")
        print(f"  ✓ {alt_path.relative_to(ROOT)}  ({len(alt):,} triples)")

        # Turtle whatever the run, like the annotations: at some fifteen
        # thousand triples the format costs nothing, and the file is one a
        # person reads to find out what an age rests on.
        beat.step(STEPS[10])
        dat_path = RDF_DIR / "sisal_v3_dating.ttl"
        dat.serialize(destination=str(dat_path), format="turtle")
        print(f"  ✓ {dat_path.relative_to(ROOT)}  ({len(dat):,} triples)")
    finally:
        beat.stop()
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--format", dest="data_format",
                        choices=sorted(DATA_FORMATS), default=DEFAULT_DATA_FORMAT,
                        help="serialisation of the two large graphs. "
                             "nt is fast and git-ignored, turtle is the "
                             "release format. Default: " + DEFAULT_DATA_FORMAT)
    parser.add_argument("--sites", dest="sites_spec", default="all",
                        help="which sites go into the graph: 'all', 'dev' "
                             "(the guard-covering selection, see DEV_SITES), "
                             "or a comma-separated list of site ids or names, "
                             "e.g. --sites spannagel. Default: all")
    args = parser.parse_args(argv)
    try:
        return 0 if build(args.data_format, args.sites_spec) else 1
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
