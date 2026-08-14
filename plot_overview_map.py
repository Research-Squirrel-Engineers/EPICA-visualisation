#!/usr/bin/env python3
"""plot_overview_map.py - every site of the three strands, read from the graph.

    plots/geo_lod_sites_map

The one figure in this repository that does not read a CSV. Its input is the
Turtle the pipeline just wrote, queried with SPARQL, and what it draws is
therefore what the graph says rather than what the generators were fed. If a
site is missing here, it is missing from the published data - which is the
check no figure drawn from the input files can perform.

That makes it slower and more fragile than the strand maps, and both are the
price of the property. It runs last, after the strands have written their RDF,
and it says which files it read.

Query
-----
One query per strand, all of the same shape: a subject with a
``geo:hasGeometry`` whose ``geo:asWKT`` is a point, plus its label and its
class. The WKT literals carry a CRS prefix - ``<...EPSG/0/4326> POINT(lon
lat)`` - so the point is parsed out with a regular expression rather than with
a geometry library.

Run through main.py, or standalone from the repository root:

    python plot_overview_map.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from rdflib import Graph  # noqa: E402

ROOT = Path(__file__).resolve().parent
#: At the root, because the figure belongs to no strand - and as maps/ rather
#: than plots/, because a map is what it is. dist/ would be the other
#: candidate and is wrong: that directory holds what a release ships, and
#: this is a figure of the repository about itself.
OUTPUT_DIR = ROOT / "maps"
REPORT_PATH = ROOT / "maps" / "overview_report.txt"

sys.path.insert(0, str(ROOT / "ontology"))
import geo_lod_basemap as gb  # noqa: E402
import geo_lod_figures as gf  # noqa: E402
from geo_lod_captions import CaptionFile  # noqa: E402

plt.rcParams["svg.hashsalt"] = "geo-lod"

#: Screen dpi of the figure while it is being laid out. It is not the
#: resolution of the JPG - that comes from geo_lod_figures, which reads
#: it from the environment so that main.py --dpi can reach every
#: drawing script.
DPI = 100

#: Which files are read, in order of preference, how a point out of each
#: strand is drawn, and whether the strand is expected to be there yet.
#:
#: The last flag is what makes this list the place a new strand is announced.
#: ELSA is written down before it exists: while its file is absent the map
#: says nothing about it, and the day the strand writes its first Turtle the
#: layer appears without anyone having to remember this file. A strand that is
#: expected and missing, by contrast, is reported in the run and named in the
#: caption - an incomplete map that does not say so is worse than none.
#:
#: SISAL keeps two candidates. sisal_sites.ttl is the cave catalogue, 4,765
#: triples of Turtle, and what this map wants. sisal_v3_core.ttl remains as a
#: fallback for a graph written before the catalogue was split off - it holds
#: the same geometries next to a million measurement triples, and parsing it
#: for 365 points is the reason the split happened.
SOURCES = (
    ("EPICA", ("EPICA/rdf/epica_dome_c.ttl",), "#1a4f8a", "*", 190, True),
    ("SISAL", ("SISAL/rdf/sisal_sites.ttl", "SISAL/rdf/sisal_v3_core.ttl"),
     "#2ca02c", "o", 26, True),
    ("CI", ("CI/rdf/ci_findspots.ttl",), "#d1691f", "^", 46, True),
    # Planned, not yet built: the Eifel maar sediment cores.
    ("ELSA", ("ELSA/rdf/elsa_sites.ttl",), "#7b3fa0", "s", 44, False),
)

QUERY = """
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?site ?label ?wkt WHERE {
    ?site geo:hasGeometry ?geometry .
    ?geometry geo:asWKT ?wkt .
    OPTIONAL { ?site rdfs:label ?label }
}
"""

POINT = re.compile(r"POINT\s*\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s*\)", re.IGNORECASE)

CAPTION_LICENSE = "CC BY 4.0, Florian Thiery"

CAPTIONS = CaptionFile(
    str(OUTPUT_DIR / "captions.yaml"),
    header=(
        "captions.yaml - the figures that do not belong to a single strand.\n"
        "\n"
        "Written by plot_overview_map.py.\n"
        "Edit 'caption' freely: an entry whose caption differs from "
        "'generated'\n"
        "is treated as hand-written and kept on the next run, while "
        "'generated'\n"
        "is refreshed so the diff shows what the code would say now."
    ),
)


class Tee:
    """Writes simultaneously to stdout and the report file."""

    def __init__(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(filepath, "w", encoding="utf-8", newline="\n")
        self.stdout = sys.stdout
        try:
            self.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
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


def sites_of(path: Path) -> list[dict]:
    """Every point geometry in one file, with its label."""
    graph = Graph()
    graph.parse(str(path), format="turtle")
    seen: dict[str, dict] = {}
    for site, label, wkt in graph.query(QUERY):
        match = POINT.search(str(wkt))
        if not match:
            continue
        key = str(site)
        if key in seen:
            continue
        seen[key] = {
            "uri": key,
            "label": str(label) if label else key.rsplit("/", 1)[-1],
            "lon": float(match.group(1)),
            "lat": float(match.group(2)),
        }
    return list(seen.values())


def build() -> bool:
    print("\n" + "=" * 72)
    print("  All palaeoclimate sites - overview map from the graph")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layers, missing = [], []
    for name, candidates, colour, marker, size, expected in SOURCES:
        path = next((ROOT / c for c in candidates if (ROOT / c).exists()), None)
        if path is None:
            if not expected:
                print(f"  ·  {name}: planned, no graph yet")
                continue
            missing.append(name)
            print(f"  ⚠  {name}: none of {', '.join(candidates)} found - the "
                  f"strand is missing from the map")
            continue
        sites = sites_of(path)
        print(f"  {name:<6} {len(sites):>4} sites  "
              f"({path.relative_to(ROOT).as_posix()})")
        layers.append((name, sites, colour, marker, size))

    if not layers:
        print("\n  No graph to read. Nothing drawn.")
        return False

    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=DPI)
    bbox = (-180.0, -90.0, 180.0, 84.0)
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect(1.0)
    gb.draw_land(ax, bbox, str(ROOT), zorder=0)
    gb.draw_graticule(ax, step=30.0, zorder=1)

    for name, sites, colour, marker, size in layers:
        ax.scatter([s["lon"] for s in sites], [s["lat"] for s in sites],
                   s=size, marker=marker, facecolor=colour,
                   edgecolor="#1a1a18", linewidth=0.5, zorder=3,
                   label=f"{name} ({len(sites)})")

    ax.set_xlabel("Longitude [°E]")
    ax.set_ylabel("Latitude [°N]")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.92)
    fig.tight_layout()

    gf.save_figure(fig, str(OUTPUT_DIR / "geo_lod_sites_map"))
    plt.close(fig)

    total = sum(len(sites) for _, sites, _, _, _ in layers)
    gap = ""
    if missing:
        which = " and ".join(", ".join(missing).rsplit(", ", 1))
        verb = "are" if len(missing) > 1 else "is"
        gap = (f" {which} {verb} not on this map: the site geometries were "
               f"not available as Turtle in this run.")
    stated = ", ".join(f"{name} {len(sites)}"
                       for name, sites, _, _, _ in layers)
    CAPTIONS.add(
        "geo_lod_sites_map",
        caption=(f"All {total} sites geo-lod publishes with a position "
                 f"({stated}), drawn from the RDF rather than from the input "
                 f"tables: every point on this map is a subject in the "
                 f"published graph carrying a geo:hasGeometry with a point "
                 f"geometry. Aspect is plate carrée at the equator, which "
                 f"exaggerates the high latitudes; the Antarctic site is "
                 f"drawn in its own projection in the EPICA map." + gap),
        license=CAPTION_LICENSE,
        sources=["http://w3id.org/geo-lod/", gb.LAND_SOURCE],
    )
    print()
    CAPTIONS.write()
    return True


def main() -> int:
    tee = Tee(REPORT_PATH)
    try:
        ok = build()
    except Exception:
        import traceback
        traceback.print_exc()
        ok = False
    finally:
        tee.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
