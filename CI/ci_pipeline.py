__author__ = "Florian Thiery"
__copyright__ = "MIT Licence 2025, Florian Thiery"
__credits__ = ["Florian Thiery"]
__license__ = "MIT"
__version__ = "beta"
__maintainer__ = "Florian Thiery"
__email__ = "mail@fthiery.de"
__status__ = "beta"
__update__ = "2025-03-01"

# ==========================================================================
# ci_pipeline.py
# Campanian Ignimbrite findspot data → FAIR RDF/Linked Open Data
#
# Input:  CI/csv/cifindspots_part_full.csv
# Output: CI/rdf/ci_findspots.ttl   — instance data
#         CI/rdf/geo_lod_ci.ttl     — ontology (pre-authored, just copied)
#
# Modelling:
#   - Every findspot → geolod:CIFindspot (+ prov:Entity, fsl:Site)
#   - ArchaeologicalSite IDs → additional geolod:CIArchaeologicalSite typing
#   - GeoSPARQL geometry: CRS-prefixed WKT (canonical CI_full.py pattern)
#   - PROV-O: Entity / Activity / Agent triples per site
#   - CRMgeo SP6_Declarative_Place for geometry bridge
#   - All findspots linked to geolod:ci_volcanic_event via crm:P12
#   - FeatureCollection for QGIS / Linked Data visualisation
#
# Integration:
#   - Imports geo_lod_utils.py from ../ontology/ (shared namespace defs,
#     get_graph(), wkt_point(), add_feature_collection())
#   - fsl: vocabulary referenced as external URI, no owl:imports needed
# ==========================================================================

import os
import sys
import datetime
import codecs
import importlib
import pandas as pd

importlib.reload(sys)

# ---------------------------------------------------------------------------
# Path setup — find geo_lod_utils in ../ontology/
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ONTOLOGY_DIR = os.path.join(_THIS_DIR, "..", "ontology")
sys.path.insert(0, _ONTOLOGY_DIR)

try:
    from rdflib import Graph, Namespace, URIRef, Literal
    from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS, DCTERMS, PROV

    RDF_AVAILABLE = True
except ImportError:
    RDF_AVAILABLE = False
    print("⚠  rdflib not installed — RDF export skipped. (pip install rdflib)")

try:
    from geo_lod_utils import (
        NS as GEO_LOD_NS,
        get_graph,
        wkt_point,
        add_feature_collection,
        add_generation_provenance,
        content_fingerprint,
        GEO_LOD_RELEASE,
    )

    GEO_LOD_UTILS_AVAILABLE = True
    print("✓  geo_lod_utils loaded from", _ONTOLOGY_DIR)
except ImportError:
    GEO_LOD_UTILS_AVAILABLE = False
    print("⚠  geo_lod_utils not found — using local namespace fallback.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_FILE = os.path.join(_THIS_DIR, "cifindspots_part_full.csv")
RDF_DIR = os.path.join(_THIS_DIR, "rdf")
REPORT_DIR = os.path.join(_THIS_DIR, "report")
os.makedirs(RDF_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------
GEOLOD_BASE = "http://w3id.org/geo-lod/"
CI_BASE = "http://w3id.org/geo-lod/ci/"

FSL = Namespace("http://fuzzy-sl.squirrel.link/ontology/") if RDF_AVAILABLE else None
FSLD = Namespace("http://fuzzy-sl.squirrel.link/data/") if RDF_AVAILABLE else None
FOAF = Namespace("http://xmlns.com/foaf/0.1/") if RDF_AVAILABLE else None

# IDs that carry fsl:ArchaeologicalSite spatialtype
# (as declared in project documentation)
ARCHAEOLOGICAL_IDS = {19, 44, 45, 50, 51, 59, 62, 63, 65}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_str(val) -> str:
    """Return stripped string or '' for NaN / 'nan'."""
    s = str(val).strip()
    return "" if s in ("nan", "None", "") else s


def _split(val: str, sep: str = ";") -> list[str]:
    """Split a potentially multi-valued cell on sep, stripping whitespace."""
    return [v.strip() for v in val.split(sep) if v.strip()]


def _uri(base: str, local: str) -> "URIRef":
    return URIRef(base + local)


def _wkt_literal(wkt_raw: str, g: "Graph") -> "Literal":
    """
    Produce a CRS-prefixed geo:wktLiteral from a raw WKT string.
    Uses wkt_point() from geo_lod_utils if available, otherwise
    falls back to the canonical CI_full.py string pattern.
    """
    wkt_clean = wkt_raw.strip()
    crs_prefix = "http://www.opengis.net/def/crs/EPSG/0/4326"
    WKT_TYPE = URIRef("http://www.opengis.net/ont/geosparql#wktLiteral")
    return Literal(f"<{crs_prefix}> {wkt_clean}", datatype=WKT_TYPE)


# ---------------------------------------------------------------------------
# Graph factory (with CI-specific extra bindings)
# ---------------------------------------------------------------------------


def _build_graph() -> "Graph":
    if GEO_LOD_UTILS_AVAILABLE:
        g = get_graph()
    else:
        g = Graph()
        for prefix, uri in {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "dct": "http://purl.org/dc/terms/",
            "prov": "http://www.w3.org/ns/prov#",
            "geo": "http://www.opengis.net/ont/geosparql#",
            "sf": "http://www.opengis.net/ont/sf#",
            "crm": "http://www.cidoc-crm.org/cidoc-crm/",
            "crmarchaeo": "http://www.cidoc-crm.org/extensions/crmarchaeo/",
            "crmgeo": "http://www.ics.forth.gr/isl/CRMgeo/",
            "sosa": "http://www.w3.org/ns/sosa/",
            "foaf": "http://xmlns.com/foaf/0.1/",
            "geolod": GEOLOD_BASE,
        }.items():
            g.bind(prefix, Namespace(uri))

    # CI-specific bindings always added on top
    g.bind("fsl", Namespace("http://fuzzy-sl.squirrel.link/ontology/"))
    g.bind("fsld", Namespace("http://fuzzy-sl.squirrel.link/data/"))
    g.bind("ci", Namespace(CI_BASE))
    g.bind("pleiades", Namespace("https://pleiades.stoa.org/places/vocab#"))
    return g


# ---------------------------------------------------------------------------
# RDF build
# ---------------------------------------------------------------------------


def build_ci_rdf(data: pd.DataFrame) -> "Graph":
    """Convert CI findspot DataFrame to an rdflib Graph."""

    if not RDF_AVAILABLE:
        print("  ⚠  rdflib not available — skipping RDF build.")
        return None

    g = _build_graph()

    # Namespace objects for triple building
    GEO = Namespace("http://www.opengis.net/ont/geosparql#")
    SF = Namespace("http://www.opengis.net/ont/sf#")
    CRM = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
    CRMARCH = Namespace("http://www.cidoc-crm.org/extensions/crmarchaeo/")
    CRMGEO = Namespace("http://www.ics.forth.gr/isl/CRMgeo/")
    GEOLOD = Namespace(GEOLOD_BASE)
    CI = Namespace(CI_BASE)
    PLEIADES = Namespace("https://pleiades.stoa.org/places/vocab#")

    # Shared event URI (defined in geo_lod_ci.ttl)
    ci_event_uri = GEOLOD["ci_volcanic_event"]

    site_uris = []  # collect for FeatureCollection

    for _, row in data.iterrows():
        site_id = str(row["id"]).strip()
        site_uri = CI[f"cisite_{site_id}"]
        geom_uri = CI[f"cisite_{site_id}_geom"]
        act_uri = CI[f"cisite_{site_id}_activity"]
        pyscr_uri = CI[f"cisite_{site_id}_pyscript"]

        site_uris.append(site_uri)

        # ------------------------------------------------------------------
        # 1. Site typing
        # ------------------------------------------------------------------
        g.add((site_uri, RDF.type, GEOLOD["CIFindspot"]))
        g.add((site_uri, RDF.type, PROV.Entity))
        g.add((site_uri, RDF.type, PLEIADES["Place"]))
        g.add((site_uri, RDF.type, FSL["Site"]))

        # CIArchaeologicalSite for known archaeological IDs
        # Also check spatialtype column for fsl:ArchaeologicalSite
        row_id = int(row["id"])
        spatial_types = _split(_safe_str(row["spatialtype"]))
        is_arch = (
            row_id in ARCHAEOLOGICAL_IDS or "fsl:ArchaeologicalSite" in spatial_types
        )
        if is_arch:
            g.add((site_uri, RDF.type, GEOLOD["CIArchaeologicalSite"]))
            g.add((site_uri, RDF.type, CRMARCH["A2_Stratigraphic_Volume_Unit"]))

        # ------------------------------------------------------------------
        # 2. Labels & metadata
        # ------------------------------------------------------------------
        label = _safe_str(row["label"])
        if label:
            g.add((site_uri, RDFS.label, Literal(label, lang="en")))
            g.add((site_uri, SKOS.prefLabel, Literal(label, lang="en")))

        desc = _safe_str(row["desc"])
        if desc:
            g.add((site_uri, SKOS.scopeNote, Literal(desc, lang="en")))
            g.add((site_uri, RDFS.comment, Literal(desc, lang="en")))

        # ------------------------------------------------------------------
        # 3. Certainty
        # ------------------------------------------------------------------
        cert = _safe_str(row["certainty"])
        cert_info = _safe_str(row["certaintyinfo"])
        if cert:
            g.add(
                (
                    site_uri,
                    GEOLOD["hasCertaintyLevel"],
                    URIRef(cert.replace("fsl:", str(FSL))),
                )
            )
        if cert_info:
            g.add(
                (
                    site_uri,
                    GEOLOD["hasCertaintyDescription"],
                    Literal(cert_info, lang="en"),
                )
            )

        # ------------------------------------------------------------------
        # 4. Spatial types (multi-valued, semicolon-separated)
        # ------------------------------------------------------------------
        for st in spatial_types:
            uri_str = st.replace("fsl:", str(FSL))
            g.add((site_uri, GEOLOD["hasSpatialType"], URIRef(uri_str)))

        # ------------------------------------------------------------------
        # 5. Literature references (multi-valued)
        # ------------------------------------------------------------------
        lit = _safe_str(row["literature"])
        if lit:
            for ref in _split(lit):
                g.add((site_uri, GEOLOD["hasLiteratureReference"], Literal(ref)))

        # ------------------------------------------------------------------
        # 6. Related-to links (skos:closeMatch, fsl:spatialCloseMatch, etc.)
        # ------------------------------------------------------------------
        relatedto = _safe_str(row["relatedto"])
        relatedhow = _safe_str(row["relatedtohow"])
        if relatedto and relatedhow:
            # Map relatedhow shorthand → full URI
            pred_str = relatedhow
            if pred_str.startswith("skos:"):
                pred_uri = URIRef(
                    pred_str.replace("skos:", "http://www.w3.org/2004/02/skos/core#")
                )
            elif pred_str.startswith("fsl:"):
                pred_uri = URIRef(pred_str.replace("fsl:", str(FSL)))
            else:
                pred_uri = URIRef(pred_str)
            for target in _split(relatedto):
                if target.startswith("http"):
                    g.add((site_uri, pred_uri, URIRef(target)))

        # ------------------------------------------------------------------
        # 7. Link to CI volcanic event
        #    geoscience sites: crm:P12_occurred_in_the_presence_of
        #    archaeological sites: crm:P7_took_place_at (stronger claim)
        # ------------------------------------------------------------------
        if is_arch:
            g.add((ci_event_uri, CRM["P7_took_place_at"], site_uri))
        else:
            g.add((ci_event_uri, CRM["P12_occurred_in_the_presence_of"], site_uri))

        # ------------------------------------------------------------------
        # 8. GeoSPARQL geometry (canonical CRS-prefixed WKT pattern)
        # ------------------------------------------------------------------
        wkt_raw = _safe_str(row["wkt"])
        if wkt_raw:
            g.add((site_uri, GEO["hasGeometry"], geom_uri))
            g.add((geom_uri, RDF.type, SF["Point"]))
            # CRMgeo SP6_Declarative_Place as bridge
            g.add((geom_uri, RDF.type, CRMGEO["SP6_Declarative_Place"]))
            g.add((geom_uri, GEO["asWKT"], _wkt_literal(wkt_raw, g)))
            if cert:
                g.add(
                    (
                        geom_uri,
                        GEOLOD["hasCertaintyLevel"],
                        URIRef(cert.replace("fsl:", str(FSL))),
                    )
                )
            if cert_info:
                g.add(
                    (
                        geom_uri,
                        GEOLOD["hasCertaintyDescription"],
                        Literal(cert_info, lang="en"),
                    )
                )

        # ------------------------------------------------------------------
        # 9. PROV-O: Agent(s)
        # ------------------------------------------------------------------
        agent_val = _safe_str(row["agent"])
        agent_uris = []
        for orcid in _split(agent_val):
            if orcid.startswith("http"):
                agent_uri = URIRef(orcid)
                g.add((agent_uri, RDF.type, FOAF["Person"]))
                g.add((agent_uri, RDF.type, PROV.Agent))
                g.add((agent_uri, SKOS.exactMatch, agent_uri))
                agent_uris.append(agent_uri)

        # ------------------------------------------------------------------
        # 10. PROV-O: Activity (georeferencing)
        # ------------------------------------------------------------------
        method_type = _safe_str(row["methodtype"])
        if method_type:
            method_uri = URIRef(method_type.replace("fsl:", str(FSL)))
            g.add((act_uri, RDF.type, PROV.Activity))
            g.add((act_uri, RDF.type, method_uri))

        # Activity data
        source = _safe_str(row["source"])
        if source:
            src_uri = URIRef(source.replace("fsl:", str(FSL)))
            g.add((act_uri, FSL["hasSource"], src_uri))
        srctype = _safe_str(row["sourcetype"])
        if srctype:
            g.add(
                (
                    act_uri,
                    FSL["hasSourceType"],
                    URIRef(srctype.replace("fsl:", str(FSL))),
                )
            )
        methoddesc = _safe_str(row["methoddesc"])
        if methoddesc:
            g.add((act_uri, FSL["activityDesc"], Literal(methoddesc, lang="en")))
        if lit:
            for ref in _split(lit):
                g.add((act_uri, GEOLOD["hasLiteratureReference"], Literal(ref)))
        if cert:
            g.add(
                (
                    act_uri,
                    GEOLOD["hasCertaintyLevel"],
                    URIRef(cert.replace("fsl:", str(FSL))),
                )
            )

        # PROV-O links: entity ↔ activity ↔ agents
        g.add((site_uri, PROV.wasGeneratedBy, act_uri))
        g.add((act_uri, PROV.used, site_uri))
        for au in agent_uris:
            g.add((site_uri, PROV.wasAttributedTo, au))
            g.add((act_uri, PROV.wasAssociatedWith, au))

        # ------------------------------------------------------------------
        # 11. PROV-O: Script provenance
        # ------------------------------------------------------------------
        script_uri = URIRef(
            "https://github.com/Research-Squirrel-Engineers/"
            "campanian-ignimbrite-geo/blob/main/py/CI.py"
        )
        g.add((site_uri, PROV.wasAttributedTo, script_uri))
        g.add(
            (
                site_uri,
                PROV.wasDerivedFrom,
                URIRef(
                    "https://github.com/Research-Squirrel-Engineers/"
                    "campanian-ignimbrite-geo"
                ),
            )
        )
        g.add((site_uri, PROV.wasGeneratedBy, pyscr_uri))
        g.add((pyscr_uri, RDF.type, URIRef("http://www.w3.org/ns/prov#Activity")))
        g.add((pyscr_uri, PROV.wasAssociatedWith, script_uri))

        # ------------------------------------------------------------------
        # 12. Licence & attribution
        # ------------------------------------------------------------------
        g.add(
            (
                site_uri,
                DCTERMS.license,
                URIRef("https://creativecommons.org/licenses/by/4.0/"),
            )
        )
        g.add(
            (site_uri, DCTERMS.creator, URIRef("https://orcid.org/0000-0002-3246-3531"))
        )
        g.add(
            (site_uri, DCTERMS.creator, URIRef("https://orcid.org/0000-0003-1100-6494"))
        )
        g.add(
            (
                site_uri,
                DCTERMS.rightsHolder,
                URIRef("https://orcid.org/0000-0002-3246-3531"),
            )
        )
        g.add(
            (
                site_uri,
                DCTERMS.rightsHolder,
                URIRef("https://orcid.org/0000-0003-1100-6494"),
            )
        )

    # ------------------------------------------------------------------
    # FeatureCollection (for QGIS / Linked Data visualisation)
    # ------------------------------------------------------------------
    if GEO_LOD_UTILS_AVAILABLE:
        add_feature_collection(
            g=g,
            collection_uri=URIRef(CI_BASE + "CIFindspotCollection"),
            label="Campanian Ignimbrite Findspot Collection",
            members=site_uris,
        )
    else:
        # Minimal fallback
        GEO = Namespace("http://www.opengis.net/ont/geosparql#")
        coll_uri = URIRef(CI_BASE + "CIFindspotCollection")
        g.add((coll_uri, RDF.type, GEO["FeatureCollection"]))
        g.add(
            (
                coll_uri,
                RDFS.label,
                Literal("Campanian Ignimbrite Findspot Collection", lang="en"),
            )
        )
        for su in site_uris:
            g.add((coll_uri, RDFS.member, su))

    # ----------------------------------------------------------------------
    # Erzeugungs-Provenienz: an welchen Stand aus Daten und Code dieser Dump
    # gebunden ist. Der Fingerabdruck ändert sich, wenn sich die CSV oder
    # dieses Skript ändert - und nur dann.
    # ----------------------------------------------------------------------
    if GEO_LOD_UTILS_AVAILABLE:
        add_generation_provenance(
            g,
            CI["ci_findspots_dataset"],
            CI["ci_findspots_generation"],
            inputs=[CSV_FILE, __file__],
            agents=[
                URIRef("https://orcid.org/0000-0002-3246-3531"),
                URIRef(
                    "https://github.com/Research-Squirrel-Engineers/"
                    "campanian-ignimbrite-geo"
                ),
            ],
            label="CI findspot RDF generation",
        )
        # Jede Fundstellen-Aktivität hängt an der Erzeugung dieses Dumps.
        # prov:wasInformedBy statt prov:hadMember: der Datensatz ist keine
        # prov:Collection, und die Zugehörigkeit der Fundstellen steht schon
        # in der FeatureCollection.
        for _, row in data.iterrows():
            g.add(
                (
                    CI[f"cisite_{str(row['id']).strip()}_pyscript"],
                    PROV.wasInformedBy,
                    CI["ci_findspots_generation"],
                )
            )

    return g


# ---------------------------------------------------------------------------
# Write TTL
# ---------------------------------------------------------------------------


def write_ci_ttl(g: "Graph", out_path: str) -> None:
    """Serialise graph to Turtle and write with explicit prefix block."""
    if g is None:
        return

    ttl_raw = g.serialize(format="turtle")

    # Kein Laufzeitstempel im Kopf: er würde die Datei bei jedem Lauf ändern,
    # ohne etwas über den Inhalt auszusagen. Stattdessen der Fingerabdruck
    # über Eingabedaten und Skript, derselbe Wert wie in owl:versionInfo.
    fingerprint = (
        content_fingerprint([CSV_FILE, __file__])
        if GEO_LOD_UTILS_AVAILABLE
        else "unavailable"
    )
    header = (
        "# ci_findspots.ttl — Campanian Ignimbrite findspot instance data\n"
        "# Generated by ci_pipeline.py\n"
        f"# release: {GEO_LOD_RELEASE}\n"
        f"# fingerprint (input data + generator): {fingerprint}\n"
        "# Source: https://github.com/Research-Squirrel-Engineers/"
        "campanian-ignimbrite-geo\n\n"
    )

    with codecs.open(out_path, "w", "utf-8") as fh:
        fh.write(header)
        fh.write(ttl_raw)

    print(f"  ✓  Written: {out_path}")
    print(f"     Triples: {len(g)}")


# ---------------------------------------------------------------------------
# Tee — writes terminal output simultaneously to stdout and report file
# ---------------------------------------------------------------------------


class Tee:
    """Writes simultaneously to stdout and a file — identical to EPICA/SISAL."""

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Start report logging (identical pattern to EPICA / SISAL)
    report_path = os.path.join(REPORT_DIR, "report.txt")
    tee = Tee(report_path)

    print("*" * 60)
    print("CI Pipeline — Campanian Ignimbrite Findspots → RDF")
    print("*" * 60)

    # 1. Read CSV
    print(f"\n→ Reading {CSV_FILE}")
    data = pd.read_csv(
        CSV_FILE,
        encoding="utf-8",
        sep=",",
        usecols=[
            "id",
            "label",
            "desc",
            "certainty",
            "certaintyinfo",
            "relatedto",
            "relatedtohow",
            "source",
            "sourcetype",
            "spatialtype",
            "methodtype",
            "agent",
            "methoddesc",
            "literature",
            "wkt",
        ],
        na_values=[".", "??", "NULL"],
    )
    print(f"   Rows loaded: {len(data)}")
    arch_rows = data[
        data["id"]
        .astype(str)
        .apply(lambda x: int(x) if x.isdigit() else -1)
        .isin(ARCHAEOLOGICAL_IDS)
        | data["spatialtype"].fillna("").str.contains("ArchaeologicalSite")
    ]
    print(
        f"   Archaeological sites: {len(arch_rows)} "
        f"(IDs: {sorted(arch_rows['id'].tolist())})"
    )

    # 2. Build RDF
    print("\n→ Building RDF graph ...")
    g = build_ci_rdf(data)

    # 3. Write TTL
    print("\n→ Writing TTL ...")
    out_path = os.path.join(RDF_DIR, "ci_findspots.ttl")
    write_ci_ttl(g, out_path)

    print("\n" + "*" * 60)
    print("SUCCESS")
    print("*" * 60)
    print(f"Report saved: {report_path}")

    tee.close()
    return g


if __name__ == "__main__":
    main()
