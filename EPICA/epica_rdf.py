"""
epica_rdf.py
============
Turns the five EPICA Dome C proxy records into RDF (step S2).

Replaces the RDF half of ``plot_epica_from_tab.py``, which covered two of the
five records and put every observation on a single implicit time axis. That
script now only draws.

What the graph says
-------------------
::

    site_dome_c            the place, one node
      +- borehole_edc99    the two PANGAEA events, each with its own
      +- borehole_domec    published coordinates
    core_edc               the ice core, sample of the site
      +- sample_dd_NNNN    a section, only where the source bounds one
    obs_<proxy>_NNNN       one node per measured value
      +- tp_<proxy>_NNNN   its age as a time position, in one named TRS
      +- mis_obs_...       which stage it falls in, and on whose authority

Five decisions are visible in that shape, each taken deliberately (S2,
2026-08-09) and each explained where it is implemented below:

1. **One site, two boreholes.** PANGAEA records the same drilling under two
   events whose coordinates differ by 1.3 km. Both are kept.
2. **Sample nodes only where a section is documented.** Only the deuterium
   record states the interval a value was averaged over.
3. **Every age names its chronology.** The five records sit on four different
   age models, and CH4 carries two. Nothing is silently harmonised.
4. **Stage membership is materialised**, on the leading source (Railsback et
   al. 2015) and through the chronology of the record.
5. **Stage boundaries are also carried into depth**, per record, by linear
   interpolation, never extrapolated.

Outputs
-------
    EPICA/rdf/epica_dome_c.ttl    instance data
    EPICA/rdf/epica_ontology.ttl  the EPICA extension of the core ontology

Both deterministic: no clock is read, ages and values are rounded to a fixed
number of decimals, and no blank nodes are minted. Two runs on unchanged
inputs are byte-identical.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from scipy.signal import savgol_filter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
RDF_DIR = os.path.join(SCRIPT_DIR, "rdf")
REPORT_DIR = os.path.join(SCRIPT_DIR, "report")
ONTOLOGY_DIR = os.path.join(REPO_DIR, "ontology")
DIST_DIR = os.path.join(REPO_DIR, "dist")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ONTOLOGY_DIR)

import epica_data as ed  # noqa: E402

from rdflib import Graph, Namespace, URIRef, Literal  # noqa: E402
from rdflib.namespace import RDF, RDFS, OWL, XSD, DCTERMS as DCT, PROV  # noqa: E402

from geo_lod_utils import (  # noqa: E402
    GEO_LOD_RELEASE,
    add_generation_provenance,
    get_graph,
    wkt_point,
    write_geo_lod_core,
    write_mermaid,
)

# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------
GEOLOD = Namespace("http://w3id.org/geo-lod/")
EPICA = Namespace("http://w3id.org/geo-lod/epica/")
MIS = Namespace("http://w3id.org/geo-lod/vocab/mis/")
TRS = Namespace("http://w3id.org/geo-lod/trs/")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
GEO = Namespace("http://www.opengis.net/ont/geosparql#")
SF = Namespace("http://www.opengis.net/ont/sf#")
QUDT = Namespace("http://qudt.org/schema/qudt/")
UNIT = Namespace("http://qudt.org/vocab/unit/")
CRM = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
CRMSCI = Namespace("http://www.ics.forth.gr/isl/CRMsci/")
DCAT = Namespace("http://www.w3.org/ns/dcat#")
TIME = Namespace("http://www.w3.org/2006/time#")

ORCID_FLO = URIRef("https://orcid.org/0000-0002-3246-3531")

# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
# Kept in the graph rather than left to the figures (decision S2, 2026-08-09).
# The reason is that the published figures show smoothed curves, so the graph
# has to be able to answer what was plotted; the parameters travel with the
# values as their own nodes, so a reader can tell which filter produced which
# number instead of having to trust a caption.
ROLLING_WINDOW = 11
SG_WINDOW = 11
SG_POLYORDER = 2

# Decimals per quantity. Fixed rather than adaptive, so the serialisation does
# not shift when a record is extended.
DEC_AGE = 4
DEC_DEPTH = 3
DEC_VALUE = 5


def dec(value: float, places: int) -> Literal:
    return Literal(f"{round(float(value), places):.{places}f}", datatype=XSD.decimal)


# ===========================================================================
# 1.  MIS BOUNDARIES  (leading reading, from the vocabulary step)
# ===========================================================================


# The three helpers below live in epica_data so that the figures use exactly
# the boundaries and the interpolation the graph uses. They were duplicated
# here first; that is the sort of duplication that stays correct for a week.
from epica_data import (  # noqa: E402
    interpolate_depth,
    read_mis_stages,
    stage_for_age,
)


# ===========================================================================
# 2.  GRAPH CONSTRUCTION
# ===========================================================================


def bind_namespaces(g: Graph) -> None:
    g.bind("epica", EPICA)
    g.bind("mis", MIS)
    g.bind("trs", TRS)
    g.bind("time", TIME)
    g.bind("dcat", DCAT)


def add_place(g: Graph) -> None:
    """The site, its two boreholes, the core and the drilling campaign.

    Decision 1, one site with two boreholes below it. The CH4, deuterium and
    dust files are recorded under the event EDC99 at 123.350/-75.100, the two
    2023 gas files under the event DomeC at 123.395/-75.102. The material is
    from the same drilling, so a single site node is right; but nothing in the
    sources says which of the two coordinates is the better one, so neither is
    dropped and no average is invented. The site reuses the EDC99 geometry by
    reference rather than restating the coordinate, which keeps it obvious
    that the site carries no position of its own.
    """
    site = EPICA["site_dome_c"]
    g.add((site, RDF.type, GEOLOD["DrillingSite"]))
    g.add((site, RDF.type, OWL.NamedIndividual))
    g.add((site, RDFS.label, Literal("EPICA Dome C, East Antarctica", lang="en")))
    g.add(
        (
            site,
            RDFS.comment,
            Literal(
                "Drilling site of the European Project for Ice Coring in "
                "Antarctica on the East Antarctic plateau, 3233 m above sea "
                "level. The published records give two slightly different "
                "positions for it; both are kept on the borehole nodes below.",
                lang="en",
            ),
        )
    )
    g.add((site, CRM["P87_is_identified_by"], Literal("EDC", datatype=XSD.string)))

    for event_id, event in sorted(ed.EVENTS.items()):
        borehole = EPICA[f"borehole_{event_id.lower()}"]
        geom = EPICA[f"geom_borehole_{event_id.lower()}"]
        g.add((borehole, RDF.type, GEOLOD["Borehole"]))
        g.add((borehole, RDF.type, OWL.NamedIndividual))
        g.add((borehole, RDFS.label, Literal(event["label"], lang="en")))
        g.add((borehole, RDFS.comment, Literal(event["comment"], lang="en")))
        g.add((borehole, CRM["P89_falls_within"], site))
        g.add((borehole, GEO["hasGeometry"], geom))
        g.add(
            (
                borehole,
                GEOLOD["elevation_m"],
                dec(event["elevation_m"], 1),
            )
        )
        g.add((geom, RDF.type, SF["Point"]))
        g.add((geom, RDF.type, OWL.NamedIndividual))
        g.add(
            (
                geom,
                GEO["asWKT"],
                Literal(
                    wkt_point(event["lon"], event["lat"]),
                    datatype=GEO["wktLiteral"],
                ),
            )
        )

    # The site points at the EDC99 geometry instead of restating it: that is
    # the event the drilling itself is recorded under, and reusing the node
    # keeps a single coordinate in a single place.
    g.add((site, GEO["hasGeometry"], EPICA["geom_borehole_edc99"]))

    collection = EPICA["collection_drilling_sites"]
    g.add((collection, RDF.type, GEO["FeatureCollection"]))
    g.add((collection, RDF.type, OWL.NamedIndividual))
    g.add(
        (
            collection,
            RDFS.label,
            Literal("EPICA Dome C drilling site collection", lang="en"),
        )
    )
    g.add((collection, RDFS.member, site))

    global_collection = GEOLOD["AllPalaeoclimateSites_Collection"]
    g.add((global_collection, RDF.type, GEO["FeatureCollection"]))
    g.add((global_collection, RDF.type, OWL.NamedIndividual))
    g.add(
        (
            global_collection,
            RDFS.label,
            Literal("All palaeoclimate sites collection", lang="en"),
        )
    )
    g.add((global_collection, RDFS.member, site))

    core = EPICA["core_edc"]
    g.add((core, RDF.type, GEOLOD["IceCore"]))
    g.add((core, RDF.type, OWL.NamedIndividual))
    g.add((core, RDFS.label, Literal("EPICA Dome C ice core (EDC)", lang="en")))
    g.add((core, SOSA["isSampleOf"], site))
    g.add((core, GEOLOD["extractedFrom"], site))
    g.add((core, CRM["P53_has_former_or_current_location"], site))
    g.add((core, CRM["P2_has_type"], Literal("Ice core", lang="en")))

    campaign = EPICA["campaign_1996_2004"]
    g.add((campaign, RDF.type, GEOLOD["DrillingCampaign"]))
    g.add((campaign, RDF.type, OWL.NamedIndividual))
    g.add(
        (
            campaign,
            RDFS.label,
            Literal("EPICA Dome C drilling campaign 1996-2004", lang="en"),
        )
    )
    g.add((campaign, GEOLOD["tookPlaceAt"], site))
    g.add((campaign, GEOLOD["removedSample"], core))
    g.add((campaign, CRM["P4_has_time-span"], Literal("1996/2004", datatype=XSD.string)))
    g.add((campaign, CRMSCI["O1_removed"], core))


def add_smoothing_methods(g: Graph) -> None:
    median = EPICA[f"smoothing_rolling_median_w{ROLLING_WINDOW}"]
    g.add((median, RDF.type, GEOLOD["RollingMedianFilter"]))
    g.add((median, RDF.type, OWL.NamedIndividual))
    g.add(
        (
            median,
            RDFS.label,
            Literal(f"Rolling median filter, window {ROLLING_WINDOW}", lang="en"),
        )
    )
    g.add((median, GEOLOD["windowSize"], Literal(ROLLING_WINDOW, datatype=XSD.integer)))
    g.add((median, DCT.references, URIRef("https://doi.org/10.1145/1968.1969")))

    savgol = EPICA[f"smoothing_savitzky_golay_w{SG_WINDOW}_p{SG_POLYORDER}"]
    g.add((savgol, RDF.type, GEOLOD["SavitzkyGolayFilter"]))
    g.add((savgol, RDF.type, OWL.NamedIndividual))
    g.add(
        (
            savgol,
            RDFS.label,
            Literal(
                f"Savitzky-Golay filter, window {SG_WINDOW}, "
                f"polynomial order {SG_POLYORDER}",
                lang="en",
            ),
        )
    )
    g.add((savgol, GEOLOD["windowSize"], Literal(SG_WINDOW, datatype=XSD.integer)))
    g.add((savgol, GEOLOD["polyOrder"], Literal(SG_POLYORDER, datatype=XSD.integer)))
    g.add((savgol, DCT.references, URIRef("https://doi.org/10.1021/ac60214a047")))


def add_catalogue(g: Graph) -> URIRef:
    catalog = EPICA["catalog"]
    g.add((catalog, RDF.type, DCAT["Catalog"]))
    g.add((catalog, RDF.type, GEOLOD["PalaeoclimateDataCatalogue"]))
    g.add((catalog, RDF.type, OWL.NamedIndividual))
    g.add(
        (
            catalog,
            DCT.title,
            Literal("EPICA Dome C ice core - linked data catalogue", lang="en"),
        )
    )
    g.add(
        (
            catalog,
            RDFS.label,
            Literal("EPICA Dome C ice core - linked data catalogue", lang="en"),
        )
    )
    g.add(
        (
            catalog,
            DCT.description,
            Literal(
                "Five proxy records from the EPICA Dome C ice core, East "
                "Antarctica: methane, deuterium, dust, the oxygen isotope "
                "composition of trapped air, and the oxygen to nitrogen ratio "
                "of trapped air. The records rest on four different depth-age "
                "models and are not on a common time axis; each observation "
                "names the chronology its age comes from.",
                lang="en",
            ),
        )
    )
    g.add(
        (
            catalog,
            DCT.publisher,
            Literal("PANGAEA - Data Publisher for Earth & Environmental Science"),
        )
    )
    g.add((catalog, DCT.created, Literal(GEO_LOD_RELEASE, datatype=XSD.date)))
    return catalog


def add_dataset(g: Graph, catalog: URIRef, dataset_id: str, n_rows: int) -> URIRef:
    """One dcat:Dataset per PANGAEA record, with its source and provenance.

    The source individuals here replace geolod:PANGAEA_CH4_Source and
    geolod:PANGAEA_d18O_Source from the previous ontology, which carried a
    creator and a title but no DOI as dct:source, and in one case named the
    wrong publication. Each of the five now carries the DOI of the record it
    describes and the licence that record was published under - the two
    records from 2023 are CC BY 4.0, the three older ones CC BY 3.0, and
    stating one licence for the catalogue as a whole would have been wrong.
    """
    meta = ed.DATASETS[dataset_id]

    source = EPICA[f"source_{dataset_id}"]
    g.add((source, RDF.type, GEOLOD["DataSource"]))
    g.add((source, RDF.type, OWL.NamedIndividual))
    g.add((source, RDFS.label, Literal(meta["source"], lang="en")))
    g.add((source, DCT.title, Literal(meta["title"], lang="en")))
    g.add((source, DCT.creator, Literal(meta["creator"])))
    g.add((source, DCT.date, Literal(meta["year"], datatype=XSD.gYear)))
    g.add((source, DCT.source, URIRef(meta["doi"])))
    g.add((source, OWL.sameAs, URIRef(meta["doi"])))
    g.add((source, DCT.license, URIRef(meta["license"])))

    dataset = EPICA[f"dataset_{dataset_id}"]
    g.add((dataset, RDF.type, DCAT["Dataset"]))
    g.add((dataset, RDF.type, GEOLOD["IceCoreDataset"]))
    g.add((dataset, RDF.type, OWL.NamedIndividual))
    g.add((dataset, RDFS.label, Literal(meta["title"], lang="en")))
    g.add((dataset, DCT.title, Literal(meta["title"], lang="en")))
    g.add(
        (
            dataset,
            DCT.description,
            Literal(
                f"{meta['label']} from the EPICA Dome C ice core, "
                f"{n_rows} measurements, ages on the {meta['trs']} "
                f"{meta['age_kind']} scale.",
                lang="en",
            ),
        )
    )
    g.add((dataset, DCT.source, source))
    g.add((dataset, DCT.license, URIRef(meta["license"])))
    g.add((dataset, GEOLOD["ageChronology"], TRS[meta["trs"]]))
    g.add((catalog, DCAT["dataset"], dataset))

    # One fingerprint per record, over its own raw file plus the two code
    # files that turn it into triples. A change to the dust file therefore
    # leaves the CH4 dataset untouched, which is what makes the version info
    # worth reading.
    add_generation_provenance(
        g,
        dataset,
        EPICA[f"generation_{dataset_id}"],
        inputs=[
            ed.raw_path(dataset_id),
            os.path.join(SCRIPT_DIR, "epica_data.py"),
            os.path.join(SCRIPT_DIR, "epica_rdf.py"),
        ],
        agents=[ORCID_FLO],
        label=f"EPICA Dome C {meta['short']} RDF generation",
    )
    return dataset


def add_property_and_type(g: Graph, dataset_id: str) -> tuple[URIRef, URIRef]:
    """The observable property and the measurement type of one record."""
    meta = ed.DATASETS[dataset_id]

    prop = EPICA[f"property_{dataset_id}"]
    g.add((prop, RDF.type, GEOLOD["ObservableProperty"]))
    g.add((prop, RDF.type, OWL.NamedIndividual))
    g.add((prop, RDFS.label, Literal(meta["label"], lang="en")))
    g.add((prop, QUDT["unit"], UNIT[meta["unit"]]))

    # The two gas-phase isotope records are about the air trapped in the ice,
    # not about the ice. Spelling that out here is the correction of an error
    # in the previous ontology, which described the d18O record as a water
    # isotope from the ice matrix.
    comments = {
        "ch4": "Methane concentration of the air trapped in the ice, in ppbv.",
        "dd": "Deuterium content of the ice itself, relative to SMOW. This is "
        "the water-isotope record of the core and the usual temperature proxy.",
        "dust": "Mass concentration of insoluble dust particles in the melted "
        "ice, measured with a Coulter counter.",
        "d18o": "Oxygen isotope composition of molecular oxygen in the trapped "
        "air, not of the ice. Reflects the composition of the atmosphere, and "
        "is used for orbital dating rather than as a local temperature proxy.",
        "do2n2": "Ratio of oxygen to nitrogen in the trapped air. Sensitive to "
        "local summer insolation at bubble close-off, hence its use for dating.",
    }
    g.add((prop, RDFS.comment, Literal(comments[dataset_id], lang="en")))

    mtype = EPICA[f"measurement_type_{dataset_id}"]
    g.add((mtype, RDF.type, GEOLOD["MeasurementType"]))
    g.add((mtype, RDF.type, OWL.NamedIndividual))
    g.add(
        (
            mtype,
            RDFS.label,
            Literal(f"{meta['label']} measurement", lang="en"),
        )
    )
    g.add((mtype, CRM["P2_has_type"], Literal(meta["method"], lang="en")))
    return prop, mtype


def add_time_position(
    g: Graph, uri: URIRef, age_ka: float, trs_name: str, label: str
) -> None:
    """An age as a time position in a named reference system.

    Both forms are written: time:numericPosition with time:hasTRS, so the age
    can be read without knowing the project, and geolod:ageKaBP, so a query
    that only wants a number does not have to join through the position node.
    """
    g.add((uri, RDF.type, TIME["TimePosition"]))
    g.add((uri, RDFS.label, Literal(label, lang="en")))
    g.add((uri, TIME["numericPosition"], dec(age_ka, DEC_AGE)))
    g.add((uri, TIME["hasTRS"], TRS[trs_name]))
    g.add((uri, GEOLOD["ageKaBP"], dec(age_ka, DEC_AGE)))


def add_observations(
    g: Graph,
    dataset_id: str,
    df: pd.DataFrame,
    dataset: URIRef,
    stages: list[dict],
) -> int:
    """All observations of one record, with age, depth, smoothing and stage."""
    meta = ed.DATASETS[dataset_id]
    prop, mtype = add_property_and_type(g, dataset_id)
    source = EPICA[f"source_{dataset_id}"]
    core = EPICA["core_edc"]
    site = EPICA["site_dome_c"]
    trs_leading = TRS[meta["trs"]]
    median_method = EPICA[f"smoothing_rolling_median_w{ROLLING_WINDOW}"]
    savgol_method = EPICA[f"smoothing_savitzky_golay_w{SG_WINDOW}_p{SG_POLYORDER}"]

    values = df["value"].to_numpy()
    smooth_median = (
        pd.Series(values).rolling(window=ROLLING_WINDOW, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    smooth_savgol = savgol_filter(
        values, window_length=min(SG_WINDOW, len(values)), polyorder=SG_POLYORDER
    )

    has_section = "depth_top_m" in df.columns
    n_membership = 0

    for i, row in df.iterrows():
        n = i + 1
        obs = EPICA[f"obs_{dataset_id}_{n:04d}"]
        age = float(row["age_ka"])
        depth = float(row["depth_m"])

        g.add((obs, RDF.type, GEOLOD["IceCoreObservation"]))
        g.add((obs, RDF.type, OWL.NamedIndividual))
        g.add(
            (
                obs,
                RDFS.label,
                Literal(
                    f"{meta['label']} observation {n:04d} "
                    f"({age:.1f} ka BP, {depth:.1f} m)",
                    lang="en",
                ),
            )
        )
        g.add((obs, GEOLOD["measurementType"], mtype))
        g.add((obs, SOSA["observedProperty"], prop))
        g.add((obs, SOSA["hasSimpleResult"], dec(row["value"], DEC_VALUE)))
        g.add((obs, GEOLOD["measuredValue"], dec(row["value"], DEC_VALUE)))
        g.add((obs, QUDT["unit"], UNIT[meta["unit"]]))
        g.add((obs, GEOLOD["atDepth_m"], dec(depth, DEC_DEPTH)))
        g.add((obs, GEOLOD["ageKaBP"], dec(age, DEC_AGE)))
        g.add((obs, SOSA["resultTime"], dec(age, DEC_AGE)))
        g.add((obs, GEOLOD["ageChronology"], trs_leading))
        g.add((obs, PROV.wasDerivedFrom, source))
        g.add((obs, CRM["P7_took_place_at"], site))
        g.add((dataset, DCAT["record"], obs))
        g.add((dataset, GEOLOD["hasObservation"], obs))

        # -- feature of interest: the section where one is documented -------
        # Decision 2. The deuterium file gives depth top/bottom and age
        # min/max, so each value there belongs to a real, bounded piece of
        # core. The other four give a single depth, and inventing a section
        # around it would assert a thickness nobody measured.
        if has_section:
            section = EPICA[f"sample_dd_{n:04d}"]
            g.add((section, RDF.type, GEOLOD["SampleSection"]))
            g.add((section, RDF.type, OWL.NamedIndividual))
            g.add(
                (
                    section,
                    RDFS.label,
                    Literal(
                        f"EDC core section {row['depth_top_m']:.2f}-"
                        f"{row['depth_bottom_m']:.2f} m",
                        lang="en",
                    ),
                )
            )
            g.add((section, SOSA["isSampleOf"], core))
            g.add((section, CRM["P46i_forms_part_of"], core))
            g.add((section, GEOLOD["depthTop_m"], dec(row["depth_top_m"], DEC_DEPTH)))
            g.add(
                (section, GEOLOD["depthBottom_m"], dec(row["depth_bottom_m"], DEC_DEPTH))
            )
            g.add((section, GEOLOD["ageMinKaBP"], dec(row["age_min_ka"], DEC_AGE)))
            g.add((section, GEOLOD["ageMaxKaBP"], dec(row["age_max_ka"], DEC_AGE)))
            g.add((obs, SOSA["hasFeatureOfInterest"], section))
        else:
            g.add((obs, SOSA["hasFeatureOfInterest"], core))

        # -- age as a time position ----------------------------------------
        tp = EPICA[f"tp_{dataset_id}_{n:04d}"]
        add_time_position(
            g, tp, age, meta["trs"], f"{age:.4g} ka BP on {meta['trs']}"
        )
        g.add((obs, GEOLOD["hasTimePosition"], tp))

        # Decision 3, second half. The CH4 file publishes the same measurement
        # on two age models. Keeping only EDC2, as the previous script did,
        # made the age look like a property of the measurement. Both are
        # written; geolod:ageChronology on the observation says which of them
        # the materialised geolod:ageKaBP follows.
        for alt_trs, alt_col in meta["trs_alternative"]:
            if alt_col not in df.columns or pd.isna(row[alt_col]):
                continue
            slug = alt_trs.lower().replace("-", "_")
            tp_alt = EPICA[f"tp_{dataset_id}_{n:04d}_{slug}"]
            add_time_position(
                g,
                tp_alt,
                float(row[alt_col]),
                alt_trs,
                f"{float(row[alt_col]):.4g} ka BP on {alt_trs}",
            )
            g.add((obs, GEOLOD["hasTimePosition"], tp_alt))

        # -- reported uncertainty, where the file gives one -----------------
        if "value_stddev" in df.columns and not pd.isna(row["value_stddev"]):
            g.add((obs, GEOLOD["standardDeviation"], dec(row["value_stddev"], 2)))

        # -- smoothed values ------------------------------------------------
        g.add(
            (
                obs,
                GEOLOD["smoothedValue_rollingMedian"],
                dec(smooth_median[i], DEC_VALUE),
            )
        )
        g.add((obs, GEOLOD["smoothedValue_savgol"], dec(smooth_savgol[i], DEC_VALUE)))
        g.add((obs, GEOLOD["smoothingMethod_median"], median_method))
        g.add((obs, GEOLOD["smoothingMethod_savgol"], savgol_method))

        # -- stage membership ------------------------------------------------
        # Decision 4. Which stage a measurement falls in is not a property of
        # the measurement: it depends on whose boundaries are used and on
        # which chronology gave the age. Both are named on the assignment, so
        # the same depth landing in different stages in two records is a
        # readable fact rather than a contradiction.
        stage = stage_for_age(stages, age)
        if stage is not None:
            assignment = EPICA[f"mis_membership_{dataset_id}_{n:04d}"]
            concept = MIS[f"MIS_{stage['stage']}"]
            g.add((assignment, RDF.type, GEOLOD["MISMembershipAssignment"]))
            g.add(
                (
                    assignment,
                    RDFS.label,
                    Literal(
                        f"{meta['label']} observation {n:04d} falls in "
                        f"{stage['label']}",
                        lang="en",
                    ),
                )
            )
            g.add((assignment, CRM["P140_assigned_attribute_to"], obs))
            g.add((assignment, CRM["P141_assigned"], concept))
            g.add(
                (
                    assignment,
                    CRM["P177_assigned_property_of_type"],
                    GEOLOD["MISMembership"],
                )
            )
            g.add((assignment, GEOLOD["assignmentStatus"], GEOLOD["LeadingAssignment"]))
            g.add((assignment, GEOLOD["inChronology"], trs_leading))
            g.add((assignment, DCT.source, MIS["source_railsback2015"]))
            g.add((obs, GEOLOD["fallsWithinStage"], concept))
            n_membership += 1

    return n_membership


def add_boundary_depths(
    g: Graph, dataset_id: str, df: pd.DataFrame, stages: list[dict]
) -> int:
    """Stage boundaries carried into the depth axis of one record.

    Decision 5. A stage boundary is published as an age; a depth for it only
    exists relative to a depth-age model. Because the five records sit on four
    models, the same boundary lands at different depths in different records,
    and that is exactly what the assignment records: the stage, the depth, the
    chronology the interpolation ran in, and the record it came from.

    Nothing is extrapolated. A boundary older than the deepest measurement of
    a record simply gets no assignment there.
    """
    meta = ed.DATASETS[dataset_id]
    written = 0

    for stage in stages:
        depth = interpolate_depth(df, stage["begin"])
        if depth is None:
            continue
        local = f"{dataset_id}_{stage['stage']}_begin"
        position = EPICA[f"depth_position_{local}"]
        assignment = EPICA[f"mis_boundary_depth_{local}"]

        g.add((position, RDF.type, GEOLOD["DepthPosition"]))
        g.add(
            (
                position,
                RDFS.label,
                Literal(
                    f"{depth:.2f} m in the {meta['short']} record "
                    f"(beginning of {stage['label']}, {meta['trs']})",
                    lang="en",
                ),
            )
        )
        g.add((position, GEOLOD["atDepth_m"], dec(depth, DEC_DEPTH)))
        g.add((position, CRM["P90_has_value"], dec(depth, DEC_DEPTH)))
        g.add((position, CRM["P91_has_unit"], UNIT["M"]))
        g.add((position, PROV.wasDerivedFrom, EPICA[f"dataset_{dataset_id}"]))

        g.add((assignment, RDF.type, GEOLOD["MISAttributeAssignment"]))
        g.add(
            (
                assignment,
                RDFS.label,
                Literal(
                    f"Beginning of {stage['label']} at {depth:.2f} m in the "
                    f"{meta['short']} record",
                    lang="en",
                ),
            )
        )
        g.add(
            (
                assignment,
                RDFS.comment,
                Literal(
                    f"Depth obtained by linear interpolation between the two "
                    f"measurements bracketing {stage['begin']} ka BP on the "
                    f"{meta['trs']} scale. Valid for this record only.",
                    lang="en",
                ),
            )
        )
        g.add((assignment, CRM["P140_assigned_attribute_to"], MIS[f"MIS_{stage['stage']}"]))
        g.add((assignment, CRM["P141_assigned"], position))
        g.add(
            (
                assignment,
                CRM["P177_assigned_property_of_type"],
                GEOLOD["MISBoundaryDepth"],
            )
        )
        g.add((assignment, GEOLOD["assignmentStatus"], GEOLOD["LeadingAssignment"]))
        g.add((assignment, GEOLOD["inChronology"], TRS[meta["trs"]]))
        g.add((assignment, GEOLOD["interpolationMethod"], GEOLOD["LinearInterpolation"]))
        g.add((assignment, DCT.source, MIS["source_railsback2015"]))
        g.add(
            (
                assignment,
                PROV.wasDerivedFrom,
                MIS[f"tp_railsback2015_MIS_{stage['stage']}_begin"],
            )
        )
        written += 1

    return written


def build_graph() -> tuple[Graph, dict]:
    g = get_graph()
    bind_namespaces(g)

    stages = read_mis_stages()
    add_place(g)
    add_smoothing_methods(g)
    catalog = add_catalogue(g)

    stats: dict = {"records": {}, "membership": 0, "boundaries": 0}

    for dataset_id, df in ed.load_all():
        dataset = add_dataset(g, catalog, dataset_id, len(df))
        n_membership = add_observations(g, dataset_id, df, dataset, stages)
        n_boundaries = add_boundary_depths(g, dataset_id, df, stages)
        stats["records"][dataset_id] = {
            "rows": len(df),
            "membership": n_membership,
            "boundaries": n_boundaries,
            "trs": ed.DATASETS[dataset_id]["trs"],
        }
        stats["membership"] += n_membership
        stats["boundaries"] += n_boundaries
        print(
            f"  {dataset_id:6s} {len(df):5d} observations, "
            f"{n_membership:5d} stage memberships, "
            f"{n_boundaries:3d} boundary depths  ({ed.DATASETS[dataset_id]['trs']})"
        )

    return g, stats


# ===========================================================================
# 3.  EPICA EXTENSION ONTOLOGY
# ===========================================================================


def build_ontology_ttl() -> str:
    return f"""\
@prefix owl:     <http://www.w3.org/2002/07/owl#> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .
@prefix dct:     <http://purl.org/dc/terms/> .
@prefix dcat:    <http://www.w3.org/ns/dcat#> .
@prefix sosa:    <http://www.w3.org/ns/sosa/> .
@prefix prov:    <http://www.w3.org/ns/prov#> .
@prefix geo:     <http://www.opengis.net/ont/geosparql#> .
@prefix qudt:    <http://qudt.org/schema/qudt/> .
@prefix unit:    <http://qudt.org/vocab/unit/> .
@prefix crm:     <http://www.cidoc-crm.org/cidoc-crm/> .
@prefix crmsci:  <http://www.ics.forth.gr/isl/CRMsci/> .
@prefix time:    <http://www.w3.org/2006/time#> .
@prefix geolod:  <http://w3id.org/geo-lod/> .

# ============================================================================
# EPICA Dome C ice core - OWL ontology extension
# Imports: <http://w3id.org/geo-lod/>
#
# GENERATED FILE - do not edit by hand.
# Source: EPICA/epica_rdf.py
# Release: {GEO_LOD_RELEASE}
#
# Instance data lives under <http://w3id.org/geo-lod/epica/>; the classes and
# properties below stay flat under <http://w3id.org/geo-lod/>, per the IRI
# decision of S0.
# ============================================================================

<http://w3id.org/geo-lod/epica>
    a owl:Ontology ;
    owl:imports     <http://w3id.org/geo-lod/> ;
    rdfs:label      "EPICA Dome C Ice Core Ontology"@en ;
    dct:title       "EPICA Dome C Ice Core Ontology"@en ;
    dct:description "Extension of the geo-lod core ontology for ice-core \
palaeoclimate records. Covers the drilling site and its boreholes, the core \
and its sections, the observations of five proxies and the depth-age models \
their ages rest on."@en ;
    dct:license     <https://creativecommons.org/licenses/by/4.0/> ;
    dct:created     "{GEO_LOD_RELEASE}"^^xsd:date ;
    owl:versionInfo "2.0.0" .

# ============================================================================
# CLASSES
# ============================================================================

geolod:IceCoreDataset
    a owl:Class ;
    rdfs:subClassOf dcat:Dataset ;
    rdfs:label      "Ice Core Dataset"@en ;
    rdfs:comment    "A published set of measurements on one ice core, on one \
depth-age model."@en .

geolod:PalaeoclimateDataCatalogue
    a owl:Class ;
    rdfs:subClassOf dcat:Catalog ;
    rdfs:label      "Palaeoclimate Data Catalogue"@en ;
    rdfs:comment    "A DCAT catalogue aggregating palaeoclimate datasets."@en .

geolod:IceCoreObservation
    a owl:Class ;
    rdfs:subClassOf geolod:PalaeoclimateObservation ;
    rdfs:label      "Ice Core Observation"@en ;
    rdfs:comment    "A single measured value at a known depth in an ice core. \
The depth is what was measured; the age is derived from it through a \
chronology, which the observation names."@en ;
    rdfs:subClassOf [
        a owl:Restriction ;
        owl:onProperty     geolod:ageChronology ;
        owl:someValuesFrom geolod:IceCoreChronology
    ] ;
    rdfs:subClassOf [
        a owl:Restriction ;
        owl:onProperty     prov:wasDerivedFrom ;
        owl:someValuesFrom geolod:DataSource
    ] .

geolod:IceCore
    a owl:Class ;
    rdfs:subClassOf geolod:PalaeoclimateSample ;
    rdfs:subClassOf crm:E22_Human-Made_Object ;
    rdfs:label      "Ice Core"@en ;
    rdfs:comment    "A cylindrical ice sample extracted by drilling from a \
glacier or ice sheet."@en ;
    rdfs:subClassOf [
        a owl:Restriction ;
        owl:onProperty     geolod:extractedFrom ;
        owl:someValuesFrom geolod:DrillingSite
    ] .

geolod:DrillingSite
    a owl:Class ;
    rdfs:subClassOf geolod:SamplingLocation ;
    rdfs:label      "Ice Core Drilling Site"@en ;
    rdfs:comment    "The location at which one or more ice cores were drilled. \
The individual holes are geolod:Borehole instances falling within it."@en .

geolod:DrillingCampaign
    a owl:Class ;
    rdfs:subClassOf crm:E7_Activity ;
    rdfs:subClassOf crmsci:S1_Matter_Removal ;
    rdfs:label      "Drilling Campaign"@en ;
    rdfs:comment    "A field campaign during which an ice core was drilled."@en ;
    rdfs:subClassOf [
        a owl:Restriction ;
        owl:onProperty     geolod:tookPlaceAt ;
        owl:someValuesFrom geolod:DrillingSite
    ] .

# ============================================================================
# PROPERTIES
# ============================================================================

geolod:fallsWithinStage
    a owl:ObjectProperty ;
    rdfs:domain  geolod:PalaeoclimateObservation ;
    rdfs:range   geolod:MarineIsotopeStage ;
    rdfs:label   "falls within stage"@en ;
    rdfs:comment "Materialised stage membership, following the leading \
boundary source. The assignment carrying that source and the chronology used \
is kept alongside as a geolod:MISMembershipAssignment; this property exists so \
that the common case does not need the join."@en .

geolod:elevation_m
    a owl:DatatypeProperty ;
    rdfs:range   xsd:decimal ;
    rdfs:label   "elevation (m)"@en ;
    rdfs:comment "Surface elevation above sea level, in metres."@en ;
    qudt:unit    unit:M .

geolod:hasDrillingCampaign
    a owl:ObjectProperty ;
    rdfs:domain  geolod:IceCoreDataset ;
    rdfs:range   geolod:DrillingCampaign ;
    rdfs:label   "has drilling campaign"@en .

# ============================================================================
# LABELS FOR EXTERNAL TERMS USED HERE
# ============================================================================

crm:E22_Human-Made_Object  rdfs:label "Human-Made Object"@en .
crm:E54_Dimension          rdfs:label "Dimension"@en .
crm:E7_Activity            rdfs:label "Activity"@en .
crm:P89_falls_within       rdfs:label "falls within"@en .
crm:P90_has_value          rdfs:label "has value"@en .
crm:P91_has_unit           rdfs:label "has unit"@en .
crm:P46i_forms_part_of     rdfs:label "forms part of"@en .
crmsci:S1_Matter_Removal   rdfs:label "Matter Removal"@en .
crmsci:O1_removed          rdfs:label "removed"@en .

time:TimePosition          rdfs:label "Time Position"@en .
time:TRS                   rdfs:label "Temporal Reference System"@en .
time:numericPosition       rdfs:label "numeric position"@en .
time:hasTRS                rdfs:label "has temporal reference system"@en .

unit:PPB                   rdfs:label "Parts Per Billion"@en .
unit:PERMILLE              rdfs:label "Per Mille"@en .
unit:MicroGM-PER-KiloGM    rdfs:label "Microgram per Kilogram"@en .
unit:M                     rdfs:label "Metre"@en .
"""


# ===========================================================================
# 4.  MAIN
# ===========================================================================


class Tee:
    """Writes to stdout and to the report file at once."""

    def __init__(self, filepath: str):
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


def main() -> int:
    os.makedirs(RDF_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    tee = Tee(os.path.join(REPORT_DIR, "rdf_report.txt"))

    print("=" * 60)
    print("EPICA Dome C - RDF generator (five records from PANGAEA .tab)")
    print("=" * 60)

    print("\nBuilding graph ...")
    g, stats = build_graph()

    ttl_path = os.path.join(RDF_DIR, "epica_dome_c.ttl")
    g.serialize(destination=ttl_path, format="turtle")

    ontology_path = os.path.join(RDF_DIR, "epica_ontology.ttl")
    with open(ontology_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(build_ontology_ttl())

    total_rows = sum(r["rows"] for r in stats["records"].values())
    print(f"\n  {len(g):,} triples")
    print(f"  {total_rows:,} observations over {len(stats['records'])} records")
    print(f"  {stats['membership']:,} stage memberships")
    print(f"  {stats['boundaries']:,} stage boundaries carried into depth")
    print(f"\n  Turtle:   {ttl_path}")
    print(f"  Ontology: {ontology_path}")

    # The canonical core ontology is written once, by main.py step 2. The
    # sub-scripts used to drop a copy into their own rdf/ directory as well,
    # which meant the bundle could pick up a stale one; that is removed.
    write_mermaid(
        ONTOLOGY_DIR,
        rolling_window=ROLLING_WINDOW,
        sg_window=SG_WINDOW,
        sg_poly=SG_POLYORDER,
    )

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
    tee.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
