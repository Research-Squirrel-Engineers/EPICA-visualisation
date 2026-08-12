"""
geo_lod_mis.py
==============
The Marine Isotope Stage table, for everything in this repository that draws
or reasons over stages.

Read from ``dist/mis_stages.csv``, the table the vocabulary step writes from
the same values it puts into the RDF. Nothing about stage boundaries is
defined in this repository outside that one generator: the hard-coded stage
lists that used to sit in the plot scripts carried boundaries of their own -
EPICA had LR04 with two hand-adjusted transitions and MIS 14 dropped entirely,
SISAL had a twelve-entry list ending at 533 ka - so a band in a figure and a
membership triple in the graph could disagree without anything noticing.

Why it lives here and not in ``EPICA/epica_data.py``, where it was written in
S2: the stages are not an EPICA matter. A speleothem falls into MIS 5e in the
same sense an ice core does, and with the reader inside the EPICA package the
SISAL script would have had to import from a sibling strand to draw a band.
Sits next to ``geo_lod_figures`` and ``geo_lod_release`` for the same reason
those do - ``ontology/`` is already on every sub-script's path, and this
module, like them, stays free of rdflib: it reads a CSV.

``EPICA/epica_data`` re-exports both functions, so ``ed.read_mis_stages()``
keeps working at its four existing call sites.
"""

from __future__ import annotations

import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)

MIS_STAGES_CSV = os.path.join(REPO_DIR, "dist", "mis_stages.csv")


def read_mis_stages() -> list[dict]:
    """Leading stage boundaries, oldest bound first.

    Stages only, no substages: Railsback et al. (2015) resolve substages over
    part of their range only, so substage bands would be present for some
    intervals and absent for others.
    """
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
