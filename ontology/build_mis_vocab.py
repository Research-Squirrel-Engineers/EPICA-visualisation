"""
build_mis_vocab.py
==================

Generates the shared Marine Isotope Stage (MIS) vocabulary of geo-lod from
the two primary sources shipped in ``data/raw/mis/``:

  * ``info/Railsbacketal2015MISSubstagesFig3-TableVersion01-CSV.csv``
    Railsback et al. (2015), stages and lettered substages, 0-1013.1 ka BP.
    This is the *leading* scheme: every MIS assignment made elsewhere in
    geo-lod (EPICA, SISAL, CI, ELSA) refers to these concepts.

  * ``info/LR04_MISboundaries.csv``
    Lisiecki & Raymo (2005), stage boundaries of the LR04 benthic stack,
    0-5315 ka BP. Beyond the coverage of Railsback et al. (2015) this is the
    only source, so the deep-time part of the scheme is LR04-only; the
    provenance is recorded on each concept.

The two sources disagree (LR04 puts the 5/6 boundary at 130 ka, Railsback
puts 5e/6a at 132.2 ka). This is not harmonised. Every boundary is kept as
its own crm:E13_Attribute_Assignment carrying dct:source, so both readings
remain queryable side by side.

Which of the two applies where is not left to the consumer, though. Every
assignment is marked geolod:LeadingAssignment or geolod:AlternativeAssignment,
and every concept names its geolod:leadingSource:

  * where Railsback et al. (2015) covers the concept, it leads and the LR04
    reading of the same boundary stays alongside as the alternative;
  * beyond its coverage LR04 leads, being the only source there;
  * where only one source provides a property at all - the LR04 excursion
    peaks, for instance - that one leads.

A consumer that wants a single consistent age axis filters on
geolod:LeadingAssignment and gets Railsback throughout the Quaternary and
LR04 in deep time, without having to know where the coverage ends.

Outputs (all deterministic - two runs are byte-identical):

  ontology/vocab/mis.ttl    the SKOS scheme, its concepts and all assignments
  ontology/trs.ttl          the temporal reference systems referenced above
  dist/mis_stages.csv       one row per concept, leading values only - the
                            table figures and age-axis code read
  dist/mis_assignments.csv  one row per assignment, both sources side by side
                            with their status - the lossless long form

The two CSVs carry exactly the values the RDF carries; nothing is recomputed
downstream. That is the point: MIS_INTERVALS in SISAL/plot_sisal_from_csv.py
is the last place where boundaries live in code, and mis_stages.csv is what
replaces it.

Run standalone or via main.py.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
RAW_DIR = REPO_DIR / "data" / "raw" / "mis"
VOCAB_DIR = SCRIPT_DIR / "vocab"
DIST_DIR = REPO_DIR / "dist"

RAILSBACK_CSV = RAW_DIR / "Railsbacketal2015MISSubstagesFig3-TableVersion01-CSV.csv"
LR04_CSV = RAW_DIR / "LR04_MISboundaries.csv"

MIS_TTL = VOCAB_DIR / "mis.ttl"
TRS_TTL = SCRIPT_DIR / "trs.ttl"
STAGES_CSV = DIST_DIR / "mis_stages.csv"
ASSIGNMENTS_CSV = DIST_DIR / "mis_assignments.csv"

MIS_NS = "http://w3id.org/geo-lod/vocab/mis/"

# Coverage limit of Railsback et al. (2015): beginning of substage 28c.
RAILSBACK_LIMIT_KA = 1013.1

SRC_RAILSBACK = "railsback2015"
SRC_LR04 = "lr04"

# LR04 lists five peaks inside MIS 5 instead of boundaries. They label the
# same excursions as the lettered substages of Railsback et al. (2015):
# the numbering runs from young to old, as do the letters.
LR04_PEAK_TO_SUBSTAGE = {
    "5.1": "5a",
    "5.2": "5b",
    "5.3": "5c",
    "5.4": "5d",
    "5.5": "5e",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fmt_ka(value: float) -> str:
    """Age in ka with at most four decimals, always with a decimal point."""
    s = f"{value:.4f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


def local_name(stage: str) -> str:
    """'5e' -> 'MIS_5e', 'TG6' -> 'MIS_TG6'."""
    return "MIS_" + stage


def slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")


def stage_of(name: str) -> str | None:
    """Parent stage of a substage: '5e' -> '5'. None if not a substage."""
    m = re.fullmatch(r"(\d+)([a-z])", name)
    return m.group(1) if m else None


def is_numeric_stage(name: str) -> bool:
    return name.isdigit()


def climate_mode(name: str) -> str | None:
    """Parity convention: odd stages warm, even stages cold.

    Applies to numbered stages only. The lettered Pliocene stages of LR04
    (G, K, KM, M, MG, Gi, Co, CN, N, NS, Si, ST, T, TG) are not covered by
    the convention and stay unclassified.
    """
    if not is_numeric_stage(name):
        return None
    return "Warm" if int(name) % 2 == 1 else "Cold"


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_railsback() -> dict[str, dict[str, float]]:
    """{'5e': {'begin': 132.2, 'end': 116.2}, ...} in ka BP."""
    out: dict[str, dict[str, float]] = {}
    with RAILSBACK_CSV.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 3:
                continue
            name = row[0].strip()
            if not re.fullmatch(r"\d+[a-z]?", name):
                continue
            entry: dict[str, float] = {}
            for key, col in (("begin", 1), ("end", 2)):
                raw = row[col].replace(",", "").strip()
                if raw:
                    entry[key] = float(raw) / 1000.0
            if entry:
                out[name] = entry
    if not out:
        raise SystemExit(f"No stage rows parsed from {RAILSBACK_CSV.name}")
    return out


def read_lr04() -> tuple[list[tuple[str, str, float]], list[tuple[str, float]]]:
    """Returns (boundaries, peaks).

    boundaries: [('1', '2', 14.0), ...] meaning the age is the beginning of
    the younger stage and the end of the older one.
    peaks: [('5.1', 82.0), ...]
    """
    boundaries: list[tuple[str, str, float]] = []
    peaks: list[tuple[str, float]] = []
    with LR04_CSV.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            label, raw = row[0].strip(), row[1].strip()
            if not raw or not re.fullmatch(r"-?\d+(\.\d+)?", raw):
                continue
            age = float(raw)
            if "(peak)" in label:
                peaks.append((label.replace("(peak)", "").strip(), age))
            elif "/" in label:
                younger, older = (p.strip() for p in label.split("/", 1))
                boundaries.append((younger, older, age))
    if not boundaries:
        raise SystemExit(f"No boundary rows parsed from {LR04_CSV.name}")
    return boundaries, peaks


# ---------------------------------------------------------------------------
# Turtle emission
# ---------------------------------------------------------------------------

PREFIXES = """\
@prefix rdfs:   <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:    <http://www.w3.org/2002/07/owl#> .
@prefix xsd:    <http://www.w3.org/2001/XMLSchema#> .
@prefix crm:    <http://www.cidoc-crm.org/cidoc-crm/> .
@prefix skos:   <http://www.w3.org/2004/02/skos/core#> .
@prefix time:   <http://www.w3.org/2006/time#> .
@prefix dct:    <http://purl.org/dc/terms/> .
@prefix prov:   <http://www.w3.org/ns/prov#> .
@prefix geolod: <http://w3id.org/geo-lod/> .
@prefix mis:    <http://w3id.org/geo-lod/vocab/mis/> .
@prefix trs:    <http://w3id.org/geo-lod/trs/> .
"""


def build_trs_ttl() -> str:
    return f"""\
# ==========================================================================
# trs.ttl
# geo-lod Temporal Reference Systems  —  <http://w3id.org/geo-lod/trs/>
#
# One TRS per chronology, so that an age can be read without knowing which
# age model produced it. Ages are given in ka BP throughout; time:TimePosition
# carries the numeric value, time:hasTRS the model it belongs to.
#
# GENERATED FILE - do not edit by hand.
# Source: ontology/build_mis_vocab.py
#
# Currently the two chronologies behind the MIS vocabulary. The ice-core
# chronologies (EDC1, EDC2, EDC3, AICC2023) are added in S2.
# ==========================================================================

{PREFIXES}
<http://w3id.org/geo-lod/trs/>
    a owl:Ontology ;
    rdfs:label   "geo-lod Temporal Reference Systems"@en ;
    rdfs:comment "Temporal reference systems used by geo-lod, one per chronology. All express age in thousands of years before present (ka BP)."@en ;
    owl:imports  <http://w3id.org/geo-lod/> .

trs:LR04
    a time:TRS, owl:NamedIndividual ;
    rdfs:label   "LR04 benthic stack age model (ka BP)"@en ;
    rdfs:comment "Age model of the LR04 benthic d18O stack. Ages in thousands of years before present."@en ;
    dct:source   mis:source_lisieckiRaymo2005 .

trs:Railsback2015
    a time:TRS, owl:NamedIndividual ;
    rdfs:label   "Railsback et al. (2015) MIS substage chronology (ka BP)"@en ;
    rdfs:comment "Age scale of the optimised lettered substage scheme of Railsback et al. (2015). Ages in thousands of years before present."@en ;
    dct:source   mis:source_railsback2015 .
"""


def build_mis_ttl(
    railsback: dict[str, dict[str, float]],
    lr04_boundaries: list[tuple[str, str, float]],
    lr04_peaks: list[tuple[str, float]],
) -> str:
    # --- collect concepts -------------------------------------------------
    concepts: dict[str, dict] = {}

    def touch(name: str, source: str) -> dict:
        c = concepts.setdefault(
            name,
            {
                "sources": [],
                "begin": {},   # source -> ka
                "end": {},     # source -> ka
                "peak": {},    # source -> ka
                "derived": False,
            },
        )
        if source not in c["sources"]:
            c["sources"].append(source)
        return c

    for name, ages in railsback.items():
        c = touch(name, SRC_RAILSBACK)
        for key in ("begin", "end"):
            if key in ages:
                c[key][SRC_RAILSBACK] = ages[key]

    for younger, older, age in lr04_boundaries:
        touch(younger, SRC_LR04)["begin"][SRC_LR04] = age
        touch(older, SRC_LR04)["end"][SRC_LR04] = age

    for peak_label, age in lr04_peaks:
        target = LR04_PEAK_TO_SUBSTAGE.get(peak_label)
        if target is None:
            raise SystemExit(f"Unmapped LR04 peak label: {peak_label}")
        touch(target, SRC_LR04)["peak"][SRC_LR04] = age

    # --- derive parent stages from the substage union ---------------------
    substages_by_stage: dict[str, list[str]] = {}
    for name in concepts:
        parent = stage_of(name)
        if parent:
            substages_by_stage.setdefault(parent, []).append(name)

    for parent, subs in substages_by_stage.items():
        subs.sort()
        c = touch(parent, SRC_RAILSBACK)
        if SRC_RAILSBACK in c["begin"] or SRC_RAILSBACK in c["end"]:
            continue  # Railsback lists the stage itself
        begins = [
            concepts[s]["begin"][SRC_RAILSBACK]
            for s in subs
            if SRC_RAILSBACK in concepts[s]["begin"]
        ]
        ends = [
            concepts[s]["end"][SRC_RAILSBACK]
            for s in subs
            if SRC_RAILSBACK in concepts[s]["end"]
        ]
        if not begins or not ends:
            continue
        c["begin"][SRC_RAILSBACK] = max(begins)
        c["end"][SRC_RAILSBACK] = min(ends)
        c["derived"] = True

    # --- leading source per concept, leading assignment per property ------
    # Railsback et al. (2015) leads wherever it reaches. Beyond its coverage
    # LR04 leads, being the only source. Where the leading source says nothing
    # about a property - the excursion peaks, which only LR04 gives - the
    # remaining source leads for that property alone.
    leading_source: dict[str, str] = {}
    for name, c in concepts.items():
        has_railsback = (
            SRC_RAILSBACK in c["begin"] or SRC_RAILSBACK in c["end"]
        )
        leading_source[name] = SRC_RAILSBACK if has_railsback else SRC_LR04

    def is_leading(name: str, key: str, source: str) -> bool:
        sources = concepts[name][key]
        lead = leading_source[name]
        if lead not in sources:
            lead = SRC_RAILSBACK if SRC_RAILSBACK in sources else SRC_LR04
        return source == lead

    def status_iri(name: str, key: str, source: str) -> str:
        return (
            "geolod:LeadingAssignment"
            if is_leading(name, key, source)
            else "geolod:AlternativeAssignment"
        )

    # --- deterministic order: youngest first, name as tie-break -----------
    def sort_key(name: str) -> tuple[float, str]:
        c = concepts[name]
        ages = list(c["begin"].values()) or list(c["end"].values()) or list(c["peak"].values())
        return (min(ages), name)

    ordered = sorted(concepts, key=sort_key)

    out: list[str] = []
    w = out.append
    stage_rows: list[dict] = []
    assignment_rows: list[dict] = []

    w("# ==========================================================================")
    w("# mis.ttl")
    w("# geo-lod Marine Isotope Stage Vocabulary")
    w("# <http://w3id.org/geo-lod/vocab/mis/>")
    w("#")
    w("# Leading scheme: Railsback et al. (2015), stages and lettered substages.")
    w("# Beyond its coverage (1013.1 ka BP) the concepts come from the LR04 stack")
    w("# of Lisiecki & Raymo (2005); dct:source records which.")
    w("#")
    w("# Boundaries are not harmonised. Each source keeps its own reading as a")
    w("# separate crm:E13_Attribute_Assignment, so both remain queryable.")
    w("#")
    w("# GENERATED FILE - do not edit by hand.")
    w("# Source: ontology/build_mis_vocab.py")
    w("# Inputs: info/Railsbacketal2015MISSubstagesFig3-TableVersion01-CSV.csv")
    w("#         info/LR04_MISboundaries.csv")
    w("# ==========================================================================")
    w("")
    w(PREFIXES)

    # --- scheme and sources ----------------------------------------------
    w(f"<{MIS_NS}>")
    w("    a skos:ConceptScheme, owl:Ontology ;")
    w('    rdfs:label   "geo-lod Marine Isotope Stage Vocabulary"@en ;')
    w('    dct:title    "Marine Isotope Stages and Substages"@en ;')
    w(
        '    skos:scopeNote """Railsback et al. (2015) point out that their lettered '
        "substages label excursions rather than the transitions between them, so the "
        "boundary ages of their table are gradational rather than exact instants. The "
        "values are reproduced here as published, without rounding; consumers should "
        'treat them as approximate."""@en ;'
    )
    w(
        '    skos:editorialNote """Concepts up to 1013.1 ka BP follow Railsback et al. '
        "(2015), which is the leading scheme for all MIS assignments in geo-lod. Older "
        "concepts are taken from the LR04 boundaries of Lisiecki & Raymo (2005), the "
        'only source covering that range."""@en ;'
    )
    w("    owl:imports  <http://w3id.org/geo-lod/> ;")
    w("    dct:source   mis:source_railsback2015, mis:source_lisieckiRaymo2005 .")
    w("")

    w("mis:source_railsback2015")
    w("    a geolod:DataSource, owl:NamedIndividual ;")
    w('    rdfs:label "Railsback et al. (2015)"@en ;')
    w(
        '    dct:bibliographicCitation "Railsback, L.B., Gibbard, P.L., Head, M.J., '
        "Voarintsoa, N.R.G., Toucanne, S. (2015): An optimized scheme of lettered marine "
        "isotope substages for the last 1.0 million years, and the climatostratigraphic "
        'nature of isotope stages and substages. Quaternary Science Reviews 111, 94-106."@en ;'
    )
    w(f'    geolod:coverageOldestAgeKaBP "{fmt_ka(RAILSBACK_LIMIT_KA)}"^^xsd:decimal ;')
    w("    dct:source <https://doi.org/10.1016/j.quascirev.2015.01.012> .")
    w("")

    w("mis:source_lisieckiRaymo2005")
    w("    a geolod:DataSource, owl:NamedIndividual ;")
    w('    rdfs:label "Lisiecki & Raymo (2005), LR04"@en ;')
    w(
        '    dct:bibliographicCitation "Lisiecki, L.E., Raymo, M.E. (2005): A '
        "Pliocene-Pleistocene stack of 57 globally distributed benthic d18O records. "
        'Palaeoceanography 20, PA1003."@en ;'
    )
    w("    dct:source <https://doi.org/10.1029/2004PA001071> .")
    w("")

    # --- concepts ---------------------------------------------------------
    for name in ordered:
        c = concepts[name]
        iri = "mis:" + local_name(name)
        parent = stage_of(name)
        is_sub = parent is not None
        cls = "geolod:MarineIsotopeSubstage" if is_sub else "geolod:MarineIsotopeStage"

        w(iri)
        w(f"    a {cls}, skos:Concept ;")
        w(f'    skos:prefLabel "MIS {name}"@en ;')
        w(f'    skos:altLabel "{name}"@en ;')
        w(f"    skos:inScheme <{MIS_NS}> ;")
        if is_sub:
            w(f"    skos:broader mis:{local_name(parent)} ;")
        else:
            w(f"    skos:topConceptOf <{MIS_NS}> ;")

        mode = climate_mode(name)
        if mode:
            w(f"    geolod:climateMode geolod:ClimateMode_{mode} ;")

        # canonical ages: Railsback where available, LR04 otherwise
        for key, prop in (("begin", "beginAgeKaBP"), ("end", "endAgeKaBP")):
            src = SRC_RAILSBACK if SRC_RAILSBACK in c[key] else (
                SRC_LR04 if SRC_LR04 in c[key] else None
            )
            if src:
                w(f'    geolod:{prop} "{fmt_ka(c[key][src])}"^^xsd:decimal ;')
        if SRC_LR04 in c["peak"]:
            w(f'    geolod:peakAgeKaBP "{fmt_ka(c["peak"][SRC_LR04])}"^^xsd:decimal ;')

        if SRC_RAILSBACK not in c["sources"]:
            w(
                '    skos:editorialNote "Beyond the coverage of Railsback et al. (2015); '
                'boundaries from the LR04 stack only."@en ;'
            )
        if c["derived"]:
            w(
                '    skos:editorialNote "Stage-level boundaries derived as the union of '
                'the substage intervals; Railsback et al. (2015) list no stage row."@en ;'
            )
        if name == "3":
            w(
                '    skos:note "Earlier geo-lod figures rendered MIS 3 as an interstadial '
                "(MIS_INTERVALS in SISAL/plot_sisal_from_csv.py). The parity convention "
                'applied here classifies it as warm."@en ;'
            )

        lead = leading_source[name]
        w(
            "    geolod:leadingSource "
            + ("mis:source_railsback2015" if lead == SRC_RAILSBACK
               else "mis:source_lisieckiRaymo2005")
            + " ;"
        )
        def leading_value(key: str) -> str:
            for s in (lead, SRC_RAILSBACK, SRC_LR04):
                if s in c[key]:
                    return fmt_ka(c[key][s])
            return ""

        stage_rows.append(
            {
                "stage": name,
                "label": f"MIS {name}",
                "kind": "substage" if is_sub else "stage",
                "parent": parent or "",
                "begin_ka": leading_value("begin"),
                "end_ka": leading_value("end"),
                "peak_ka": fmt_ka(c["peak"][SRC_LR04]) if SRC_LR04 in c["peak"] else "",
                "climate_mode": (mode or "").lower(),
                "leading_source": "Railsback2015" if lead == SRC_RAILSBACK else "LR04",
                "derived": "true" if c["derived"] else "false",
            }
        )

        sources = " , ".join(
            "mis:source_railsback2015" if s == SRC_RAILSBACK
            else "mis:source_lisieckiRaymo2005"
            for s in c["sources"]
        )
        w(f"    dct:source {sources} .")
        w("")

    # --- assignments ------------------------------------------------------
    def emit_assignment(
        aid: str,
        target: str,
        value_node: str,
        prop_type: str,
        source: str,
        status: str,
        stage: str,
        value: str,
        derived_from: list[str] | None = None,
        comment: str | None = None,
    ) -> None:
        assignment_rows.append(
            {
                "stage": stage,
                "property": prop_type.split(":")[-1],
                "value": value,
                "source": "Railsback2015" if source == SRC_RAILSBACK else "LR04",
                "status": status.split(":")[-1],
                "assignment": aid,
            }
        )
        src_iri = (
            "mis:source_railsback2015" if source == SRC_RAILSBACK
            else "mis:source_lisieckiRaymo2005"
        )
        w(f"mis:{aid}")
        w("    a geolod:MISAttributeAssignment ;")
        w(f"    crm:P140_assigned_attribute_to {target} ;")
        w(f"    crm:P141_assigned {value_node} ;")
        w(f"    crm:P177_assigned_property_of_type {prop_type} ;")
        w(f"    geolod:assignmentStatus {status} ;")
        if derived_from:
            w(f"    prov:wasDerivedFrom {' , '.join(derived_from)} ;")
        if comment:
            w(f'    rdfs:comment "{comment}"@en ;')
        w(f"    dct:source {src_iri} .")
        w("")

    def emit_time_position(tid: str, age: float, trs: str, label: str) -> None:
        w(f"mis:{tid}")
        w("    a time:TimePosition ;")
        w(f'    rdfs:label "{label}"@en ;')
        w(f'    time:numericPosition "{fmt_ka(age)}"^^xsd:decimal ;')
        w(f"    time:hasTRS trs:{trs} ;")
        w(f'    geolod:ageKaBP "{fmt_ka(age)}"^^xsd:decimal .')
        w("")

    w("# --------------------------------------------------------------------------")
    w("# Boundaries after Railsback et al. (2015)")
    w("# --------------------------------------------------------------------------")
    w("")
    for name in ordered:
        c = concepts[name]
        for key, prop_type, word in (
            ("begin", "geolod:PeriodBeginning", "beginning"),
            ("end", "geolod:PeriodEnd", "end"),
        ):
            if SRC_RAILSBACK not in c[key]:
                continue
            age = c[key][SRC_RAILSBACK]
            base = f"{SRC_RAILSBACK}_{slug(local_name(name))}_{key}"
            emit_time_position(
                f"tp_{base}", age, "Railsback2015",
                f"{fmt_ka(age)} ka BP ({word} of MIS {name}, Railsback et al. 2015)",
            )
            derived_from = None
            comment = None
            if c["derived"]:
                derived_from = [
                    "mis:" + local_name(s) for s in sorted(substages_by_stage[name])
                ]
                comment = (
                    "Derived as the union of the substage intervals; "
                    "Railsback et al. (2015) list no stage row for this stage."
                )
            emit_assignment(
                f"assign_{base}",
                "mis:" + local_name(name),
                f"mis:tp_{base}",
                prop_type,
                SRC_RAILSBACK,
                status_iri(name, key, SRC_RAILSBACK),
                name,
                fmt_ka(age),
                derived_from=derived_from,
                comment=comment,
            )

    w("# --------------------------------------------------------------------------")
    w("# Boundaries after Lisiecki & Raymo (2005), LR04")
    w("# --------------------------------------------------------------------------")
    w("")
    for younger, older, age in lr04_boundaries:
        base = f"{SRC_LR04}_boundary_{slug(younger)}_{slug(older)}"
        emit_time_position(
            f"tp_{base}", age, "LR04",
            f"{fmt_ka(age)} ka BP (MIS {younger}/{older} boundary, LR04)",
        )
        emit_assignment(
            f"assign_{base}_begin",
            "mis:" + local_name(younger),
            f"mis:tp_{base}",
            "geolod:PeriodBeginning",
            SRC_LR04,
            status_iri(younger, "begin", SRC_LR04),
            younger,
            fmt_ka(age),
        )
        emit_assignment(
            f"assign_{base}_end",
            "mis:" + local_name(older),
            f"mis:tp_{base}",
            "geolod:PeriodEnd",
            SRC_LR04,
            status_iri(older, "end", SRC_LR04),
            older,
            fmt_ka(age),
        )

    w("# --------------------------------------------------------------------------")
    w("# Excursion peaks of LR04, matched to the lettered substages")
    w("# --------------------------------------------------------------------------")
    w("")
    for peak_label, age in lr04_peaks:
        target = LR04_PEAK_TO_SUBSTAGE[peak_label]
        base = f"{SRC_LR04}_peak_{slug(local_name(target))}"
        emit_time_position(
            f"tp_{base}", age, "LR04",
            f"{fmt_ka(age)} ka BP (peak of MIS {peak_label}, LR04)",
        )
        emit_assignment(
            f"assign_{base}",
            "mis:" + local_name(target),
            f"mis:tp_{base}",
            "geolod:ExcursionPeak",
            SRC_LR04,
            status_iri(target, "peak", SRC_LR04),
            target,
            fmt_ka(age),
            comment=(
                f"LR04 lists this excursion as MIS {peak_label}; it labels the same "
                f"excursion as substage {target} of Railsback et al. (2015)."
            ),
        )

    w("# --------------------------------------------------------------------------")
    w("# Warm / cold classification (parity convention)")
    w("# --------------------------------------------------------------------------")
    w("")
    for name in ordered:
        mode = climate_mode(name)
        if not mode:
            continue
        emit_assignment(
            f"assign_climateMode_{slug(local_name(name))}",
            "mis:" + local_name(name),
            f"geolod:ClimateMode_{mode}",
            "geolod:ClimateModeType",
            SRC_RAILSBACK,
            "geolod:LeadingAssignment",
            name,
            mode.lower(),
            comment=(
                "Parity convention: odd-numbered stages are warm, even-numbered "
                "stages cold, as discussed by Railsback et al. (2015)."
            ),
        )

    return "\n".join(out).rstrip("\n") + "\n", stage_rows, assignment_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """CSV with LF line endings and a fixed column order, so that two runs
    differ in no byte."""
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(verbose: bool = True) -> bool:
    for path in (RAILSBACK_CSV, LR04_CSV):
        if not path.exists():
            print(f"  \u2717 Input missing: {path}")
            print(f"    Expected in {RAW_DIR.relative_to(REPO_DIR)} - the MIS sources "
                  "were moved there out of info/.")
            return False

    railsback = read_railsback()
    lr04_boundaries, lr04_peaks = read_lr04()

    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    mis_ttl, stage_rows, assignment_rows = build_mis_ttl(
        railsback, lr04_boundaries, lr04_peaks
    )
    MIS_TTL.write_text(mis_ttl, encoding="utf-8", newline="\n")
    TRS_TTL.write_text(build_trs_ttl(), encoding="utf-8", newline="\n")

    write_csv(
        STAGES_CSV,
        stage_rows,
        ["stage", "label", "kind", "parent", "begin_ka", "end_ka", "peak_ka",
         "climate_mode", "leading_source", "derived"],
    )
    write_csv(
        ASSIGNMENTS_CSV,
        assignment_rows,
        ["stage", "property", "value", "source", "status", "assignment"],
    )

    if verbose:
        n_stage = sum(1 for line in mis_ttl.splitlines() if line.startswith("    a geolod:MarineIsotopeStage"))
        n_sub = sum(1 for line in mis_ttl.splitlines() if line.startswith("    a geolod:MarineIsotopeSubstage"))
        n_assign = mis_ttl.count("    a geolod:MISAttributeAssignment ;")
        print(f"  \u2713 {MIS_TTL.relative_to(REPO_DIR)}: {n_stage} stages, "
              f"{n_sub} substages, {n_assign} assignments "
              f"({MIS_TTL.stat().st_size / 1024:.1f} KB)")
        print(f"  \u2713 {TRS_TTL.relative_to(REPO_DIR)}: 2 temporal reference systems")
        print(f"  \u2713 {STAGES_CSV.relative_to(REPO_DIR)}: {len(stage_rows)} rows")
        print(f"  \u2713 {ASSIGNMENTS_CSV.relative_to(REPO_DIR)}: "
              f"{len(assignment_rows)} rows")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
