"""
epica_data.py
=============
One loading path for the five EPICA Dome C proxy records, shared by the RDF
generator (``epica_rdf.py``) and the plot script (``plot_epica_from_tab.py``).

Why the ``.tab`` and not a derived CSV
-------------------------------------
The PANGAEA ``.tab`` files carry a header block that states, per column, which
age model produced it, which device measured it and who the PI was. A derived
CSV drops all of that. Since S2 models the chronologies as first-class nodes,
that header *is* the data we need, so the ``.tab`` is the only source here.

Why a Python constant and not the YAML
--------------------------------------
``wdttest-epica`` keeps the same provenance in ``data.yaml``. Per the reuse
rule (copy, do not reference) it is reproduced here rather than imported, and
as a Python literal rather than YAML so that geo-lod does not gain a PyYAML
dependency for five records. The values below were transcribed from
``wdttest-epica/data.yaml`` and then checked against the ``.tab`` headers; two
of them did not survive that check and are corrected here:

  * ``d18o`` is *not* a water isotope. The header reads
    ``δ18O, gas [‰] ... COMMENT: of O2`` — this is the isotopic composition of
    molecular oxygen from the trapped air, not of the ice. The water isotope in
    this collection is ``dd`` (δD). The earlier geo-lod ontology labelled it
    "stable water isotope ratio ... from the ice matrix", which is wrong.
  * ``d18o`` and ``do2n2`` are both labelled AICC2023 in ``data.yaml``, but the
    header shows ``Gas age`` for the former and ``Ice age`` for the latter.
    Those are two different axes: at a given depth the trapped air is younger
    than the ice enclosing it, by up to several thousand years at Dome C. They
    therefore get two distinct temporal reference systems.

Loaded frames
-------------
Every loader returns a DataFrame with a common column set, so that downstream
code does not need to know which proxy it holds:

    depth_m       depth in the core, metres
    age_ka        age on the leading chronology of that record, ka BP
    value         the measured value, in the unit given by ``DATASETS[id]``

plus, where the file provides them, the record-specific extras named in
``DATASETS[id]["extra_columns"]``.
"""

from __future__ import annotations

import os
from typing import Iterator

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(REPO_DIR, "data", "raw", "epica")


# ===========================================================================
# 1.  PROVENANCE MANIFEST
# ===========================================================================
# One entry per PANGAEA dataset. ``trs`` names the temporal reference system
# the record's leading age column is expressed in; the IRIs are minted under
# <http://w3id.org/geo-lod/trs/> by ontology/build_mis_vocab.py.
#
# ``event`` is the PANGAEA event the file was recorded under. The two events
# carry coordinates 1.3 km apart although all five records come from the same
# drilling. That difference is a difference between metadata records, not a
# statement that two holes were drilled, and it is kept visible rather than
# silently averaged away (see epica_rdf.py, borehole modelling).

DATASETS: dict[str, dict] = {
    "ch4": {
        "file": "EDC_CH4.tab",
        "title": "EPICA Dome C - methane (CH4)",
        "short": "CH4",
        "label": "Methane (CH\u2084)",
        "creator": "Spahni, R.; Stocker, T. F.",
        "source": "Spahni & Stocker (2006)",
        "doi": "https://doi.org/10.1594/PANGAEA.472484",
        "year": "2006",
        "license": "https://creativecommons.org/licenses/by/3.0/",
        "license_label": "CC BY 3.0",
        "unit": "PPB",
        "unit_label": "ppbv",
        "trs": "EDC2-gas",
        # The file also carries the older EDC1 gas age. It is kept as an
        # alternative reading rather than dropped (decision S2, 2026-08-09).
        # (TRS name, column in the frame) - the column name comes from the
        # file, the TRS name from the vocabulary, and they do not have to match.
        "trs_alternative": [("EDC1-gas", "age_edc1_ka")],
        "age_kind": "gas age",
        "event": "EDC99",
        "method": "Gas chromatography",
        "pi_orcid": "https://orcid.org/0000-0003-1245-2728",
        "extra_columns": ("age_edc1_ka", "value_stddev"),
    },
    "dd": {
        "file": "EPICA_Dome_C_dD.tab",
        "title": "EPICA Dome C - deuterium (\u03b4D)",
        "short": "dD",
        "label": "Deuterium (\u03b4D)",
        "creator": "Augustin, L. et al. (EPICA community members)",
        "source": "Augustin et al. (2004), Nature 429",
        "doi": "https://doi.org/10.1594/PANGAEA.198743",
        "year": "2004",
        "license": "https://creativecommons.org/licenses/by/3.0/",
        "license_label": "CC BY 3.0",
        "unit": "PERMILLE",
        "unit_label": "\u2030 SMOW",
        "trs": "EDC2-ice",
        "trs_alternative": [],
        "age_kind": "ice age",
        "event": "EDC99",
        "method": "Isotope ratio mass spectrometry",
        "pi_orcid": None,
        # The only record that states the interval a value was averaged over.
        # Those four columns are what makes a sample node meaningful here.
        "extra_columns": ("depth_top_m", "depth_bottom_m", "age_min_ka", "age_max_ka"),
    },
    "dust": {
        "file": "EPICA_Dome_C_dust.tab",
        "title": "EPICA Dome C - dust concentration (Coulter counter)",
        "short": "dust",
        "label": "Dust concentration",
        "creator": "Lambert, F. et al.",
        "source": "Lambert et al. (2008)",
        "doi": "https://doi.org/10.1594/PANGAEA.695984",
        "year": "2008",
        "license": "https://creativecommons.org/licenses/by/3.0/",
        "license_label": "CC BY 3.0",
        "unit": "MicroGM-PER-KiloGM",
        "unit_label": "\u00b5g/kg",
        "trs": "EDC3-ice",
        "trs_alternative": [],
        "age_kind": "ice age",
        "event": "EDC99",
        "method": "Coulter counter",
        "pi_orcid": "https://orcid.org/0000-0002-9489-3991",
        "extra_columns": (),
    },
    "d18o": {
        "file": "EPICA_Dome_C_d18O.tab",
        "title": "EPICA Dome C - \u03b418O of atmospheric O2",
        "short": "d18O",
        "label": "\u03b4\u00b9\u2078O of O\u2082",
        "creator": "Bouchet, M. et al.",
        "source": "Bouchet et al. (2023)",
        "doi": "https://doi.org/10.1594/PANGAEA.961024",
        "year": "2023",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "license_label": "CC BY 4.0",
        "unit": "PERMILLE",
        "unit_label": "\u2030",
        "trs": "AICC2023-gas",
        "trs_alternative": [],
        "age_kind": "gas age",
        "event": "DomeC",
        "method": "Isotope Ratio Mass Spectrometer (IRMS)",
        "pi_orcid": "https://orcid.org/0009-0002-0760-1776",
        "extra_columns": (),
    },
    "do2n2": {
        "file": "EPICA_Dome_C_do2n2.tab",
        "title": "EPICA Dome C - \u03b4O2/N2",
        "short": "dO2N2",
        "label": "\u03b4O\u2082/N\u2082",
        "creator": "Bouchet, M. et al.",
        "source": "Bouchet et al. (2023)",
        "doi": "https://doi.org/10.1594/PANGAEA.961025",
        "year": "2023",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "license_label": "CC BY 4.0",
        "unit": "PERMILLE",
        "unit_label": "\u2030",
        "trs": "AICC2023-ice",
        "trs_alternative": [],
        "age_kind": "ice age",
        "event": "DomeC",
        "method": "Isotope Ratio Mass Spectrometer (IRMS)",
        "pi_orcid": "https://orcid.org/0009-0002-0760-1776",
        "extra_columns": (),
    },
}

#: Fixed order. Everything downstream iterates over this, so that the graph,
#: the report and any figure panel come out in the same sequence on every run.
DATASET_ORDER: tuple[str, ...] = ("ch4", "dd", "dust", "d18o", "do2n2")

#: The two PANGAEA events, with the coordinates their headers give.
EVENTS: dict[str, dict] = {
    "EDC99": {
        "label": "EDC99 / EDC (EPICA Dome C)",
        "lon": 123.350000,
        "lat": -75.100000,
        "elevation_m": 3233.0,
        "method": "Ice drill (ICEDRILL)",
        "comment": (
            "Event as recorded in the CH4, deuterium and dust datasets. "
            "Drilling ran 1996-2004; the first hole (EDC96) was abandoned at "
            "788 m in 1999 and the record continues in EDC99."
        ),
    },
    "DomeC": {
        "label": "DomeC",
        "lon": 123.395000,
        "lat": -75.102000,
        "elevation_m": 3233.0,
        "method": "Drilling/drill rig (DRILL)",
        "comment": (
            "Event as recorded in the two 2023 gas datasets. The coordinates "
            "differ from the EDC99 event by about 1.3 km although the material "
            "is from the same core; this is a difference between metadata "
            "records, not evidence of a second borehole."
        ),
    },
}


# ===========================================================================
# 2.  TAB PARSING
# ===========================================================================


def raw_path(dataset_id: str) -> str:
    """Absolute path of the ``.tab`` behind *dataset_id*."""
    return os.path.join(RAW_DIR, DATASETS[dataset_id]["file"])


def all_raw_paths() -> list[str]:
    """All five ``.tab`` in fixed order - the input list for the fingerprint."""
    return [raw_path(d) for d in DATASET_ORDER]


def _read_tab(path: str) -> tuple[list[str], list[list[str]]]:
    """Split a PANGAEA ``.tab`` into its column header and its data rows.

    The header block is everything up to and including the line ``*/``; the
    line after it names the columns. Blank lines are dropped. Nothing else is
    interpreted here - the parsing per record happens in the loaders below,
    where the column meanings are documented.
    """
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")

    end = None
    for i, line in enumerate(lines):
        if line.startswith("*/"):
            end = i
            break
    if end is None:
        raise ValueError(f"{os.path.basename(path)}: no PANGAEA header block found")

    columns = lines[end + 1].split("\t")
    rows = [ln.split("\t") for ln in lines[end + 2:] if ln.strip()]
    return columns, rows


def _frame(path: str, names: list[str], numeric: list[str]) -> pd.DataFrame:
    """Build a DataFrame from a ``.tab``, checking the column count.

    The guard matters: PANGAEA occasionally republishes a dataset with an
    extra column, and a silent positional shift would move ages into the value
    column without anything failing.
    """
    columns, rows = _read_tab(path)
    if len(columns) != len(names):
        raise ValueError(
            f"{os.path.basename(path)}: expected {len(names)} columns "
            f"{names}, found {len(columns)}: {columns}"
        )
    df = pd.DataFrame(rows, columns=names)
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_ch4() -> pd.DataFrame:
    """CH4 in ppbv on the EDC2 gas-age scale, with the EDC1 age kept alongside.

    Columns of ``EDC_CH4.tab``:
        0 depth ice/snow [m]      3 gas age [ka BP], EDC2
        1 depth reference [m]     4 CH4 [ppbv]
        2 gas age [ka BP], EDC1   5 CH4 standard deviation
    """
    df = _frame(
        raw_path("ch4"),
        ["depth_m", "depth_ref_m", "age_edc1_ka", "age_ka", "value", "value_stddev"],
        ["depth_m", "depth_ref_m", "age_edc1_ka", "age_ka", "value", "value_stddev"],
    )
    return df.dropna(subset=["depth_m", "age_ka", "value"]).reset_index(drop=True)


def load_dd() -> pd.DataFrame:
    """Deuterium in per mil SMOW on EDC2, with the averaging interval.

    Columns of ``EPICA_Dome_C_dD.tab``:
        0 depth ice/snow [m]   3 depth bottom [m]   6 dD [per mil SMOW]
        1 age [ka BP]          4 age min [ka]
        2 depth top [m]        5 age max [ka]

    Depth top/bottom and age min/max bound the interval the value was averaged
    over. This is the one record where a sample section is documented, so it is
    the one record that gets sample nodes.
    """
    names = [
        "depth_m",
        "age_ka",
        "depth_top_m",
        "depth_bottom_m",
        "age_min_ka",
        "age_max_ka",
        "value",
    ]
    df = _frame(raw_path("dd"), names, names)
    return df.dropna(subset=["depth_m", "age_ka", "value"]).reset_index(drop=True)


def load_dust() -> pd.DataFrame:
    """Dust concentration in micrograms per kilogram on the EDC3 age model.

    Columns: 0 depth ice/snow [m], 1 age model [ka], 2 dust conc [ug/kg].
    """
    names = ["depth_m", "age_ka", "value"]
    df = _frame(raw_path("dust"), names, names)
    return df.dropna(subset=["depth_m", "age_ka", "value"]).reset_index(drop=True)


def load_d18o() -> pd.DataFrame:
    """delta-18O of atmospheric O2 in per mil, on the AICC2023 *gas* age scale.

    Columns: 0 depth ice/snow [m], 1 gas age [ka BP], 2 d18O-O2 [per mil].
    """
    names = ["depth_m", "age_ka", "value"]
    df = _frame(raw_path("d18o"), names, names)
    return df.dropna(subset=["depth_m", "age_ka", "value"]).reset_index(drop=True)


def load_do2n2() -> pd.DataFrame:
    """delta-O2/N2 in per mil, on the AICC2023 *ice* age scale.

    Columns: 0 depth ice/snow [m], 1 ice age [ka BP], 2 dO2/N2 [per mil].
    """
    names = ["depth_m", "age_ka", "value"]
    df = _frame(raw_path("do2n2"), names, names)
    return df.dropna(subset=["depth_m", "age_ka", "value"]).reset_index(drop=True)


LOADERS = {
    "ch4": load_ch4,
    "dd": load_dd,
    "dust": load_dust,
    "d18o": load_d18o,
    "do2n2": load_do2n2,
}


def load(dataset_id: str) -> pd.DataFrame:
    """Load one record by id."""
    return LOADERS[dataset_id]()


def load_all() -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(dataset_id, frame)`` for all five records, in fixed order."""
    for dataset_id in DATASET_ORDER:
        yield dataset_id, LOADERS[dataset_id]()


# ===========================================================================
# 3.  MARINE ISOTOPE STAGES
# ===========================================================================
# Read from dist/mis_stages.csv, the table the vocabulary step writes from the
# same values it puts in the RDF. Nothing about stage boundaries is defined in
# this repository outside that one generator any more: the hard-coded
# MIS_INTERVALS list that used to sit in the plot script carried boundaries of
# its own - LR04 with two hand-adjusted transitions and MIS 14 dropped
# entirely - so a band in a figure and a membership triple in the graph could
# disagree without anything noticing.

MIS_STAGES_CSV = os.path.join(REPO_DIR, "dist", "mis_stages.csv")


def read_mis_stages() -> list[dict]:
    """Leading stage boundaries, oldest bound first.

    Stages only, no substages: Railsback et al. (2015) resolve substages over
    part of their range only, so substage bands would be present for some
    intervals and absent for others.
    """
    import csv

    if not os.path.exists(MIS_STAGES_CSV):
        raise FileNotFoundError(
            f"{MIS_STAGES_CSV} missing - run the vocabulary step "
            f"(main.py step 3) first."
        )

    stages: list[dict] = []
    with open(MIS_STAGES_CSV, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["kind"] != "stage" or not row["begin_ka"]:
                continue
            stages.append(
                {
                    "stage": row["stage"],
                    "label": row["label"],
                    # begin = older bound, end = younger bound. MIS 1 has no
                    # younger bound; it reaches the present.
                    "begin": float(row["begin_ka"]),
                    "end": float(row["end_ka"]) if row["end_ka"] else 0.0,
                    "mode": row["climate_mode"] or None,
                }
            )
    stages.sort(key=lambda s: s["begin"])
    return stages


def stage_for_age(stages: list[dict], age_ka: float) -> dict | None:
    """The stage an age falls in: end <= age < begin."""
    for st in stages:
        if st["end"] <= age_ka < st["begin"]:
            return st
    return None


#: Longest gap between two measurements that interpolation will still bridge.
#: A boundary falling inside a longer gap gets no depth at all. Without this
#: the CH4 record, which has no data between 214 and 392 ka, produced depths
#: for the beginnings of MIS 8, 9 and 10 by drawing a straight line across
#: 178 ka of nothing - values that look like the others and mean nothing.
MAX_INTERPOLATION_GAP_KA = 15.0


def interpolate_depth(
    df: pd.DataFrame, age_ka: float, max_gap_ka: float = MAX_INTERPOLATION_GAP_KA
) -> float | None:
    """Depth at which *age_ka* falls in this record, by linear interpolation.

    Returns None outside the measured range, and also inside it wherever the
    two bracketing measurements are further apart than *max_gap_ka*.
    Extrapolating past the ends would state a depth the record does not reach;
    interpolating across a long gap states one it does not support. Both are
    refused rather than flagged, because a depth that is present in the data
    will be used as though it were measured.
    """
    pairs = sorted(zip(df["age_ka"].tolist(), df["depth_m"].tolist()))
    if not pairs or age_ka < pairs[0][0] or age_ka > pairs[-1][0]:
        return None
    for (a0, d0), (a1, d1) in zip(pairs, pairs[1:]):
        if a0 <= age_ka <= a1:
            if a1 - a0 > max_gap_ka:
                return None
            if a1 == a0:
                return d0
            return d0 + (d1 - d0) * (age_ka - a0) / (a1 - a0)
    return None
