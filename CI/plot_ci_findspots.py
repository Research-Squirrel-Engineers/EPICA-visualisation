#!/usr/bin/env python3
"""plot_ci_findspots.py - the two figures of the CI strand.

    CI/plots/ci_findspots_map          where the findspots are
    CI/plots/ci_findspots_certainty    the same map plus how well each point
                                       is known

Split from ci_pipeline.py the way EPICA and SISAL are split: the triples are
written first, the figures afterwards, so that an error in the data shows up
at the RDF step rather than after the drawing.

No basemap, and that is a decision, not an omission
---------------------------------------------------
There is no cartopy and no geopandas in this repository, and adding a geo
stack for two figures would be a dependency decision of its own. What a
Campanian Ignimbrite map has to show is not a coastline but a distance: how
far the ash travelled from the Phlegraean Fields. So the background is a
graticule with rings at 500 km intervals around the source, and the reader can
measure off the figure what a coastline would only suggest.

The positions are drawn on a plain equirectangular grid with the aspect ratio
corrected at the mean latitude of the data, which for a 6-45 degree longitude
window is close enough that no point moves visibly.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RAW_CSV = ROOT / "data" / "raw" / "ci" / "cifindspots_part_full.csv"
CURATED_CSV = ROOT / "data" / "curated" / "ci_site_annotations.csv"
OUTPUT_DIR = SCRIPT_DIR / "plots"
REPORT_PATH = SCRIPT_DIR / "report" / "figures_report.txt"

sys.path.insert(0, str(ROOT / "ontology"))
import geo_lod_figures as gf  # noqa: E402
from geo_lod_captions import CaptionFile  # noqa: E402

# Byte-identical SVG across runs: the salt fixes the ids matplotlib gives clip
# paths, save_figure drops the date from the metadata.
plt.rcParams["svg.hashsalt"] = "geo-lod"

DPI = 100

#: The source of the eruption, and the centre of the rings. It is findspot 22
#: in the table; taken from there rather than hard-coded, so that a corrected
#: coordinate moves the rings with it.
SOURCE_ID = "22"

#: Ring spacing in km. The number of rings follows the data rather than a
#: constant: a findspot beyond the outermost ring would be drawn without a
#: scale to read it against, which is the one thing this background is for.
RING_STEP_KM = 500

#: Certainty levels in the order they are drawn and listed, strongest first.
#: The order is the point: a legend sorted alphabetically would put dubious
#: above high and suggest a ranking that is not there.
CERTAINTY_ORDER = ("fsl:high", "fsl:representative", "fsl:medium",
                   "fsl:low", "fsl:dubious")
CERTAINTY_COLOUR = {
    "fsl:high": "#1a4f8a",
    "fsl:representative": "#3b8ea5",
    "fsl:medium": "#e0a300",
    "fsl:low": "#d1691f",
    "fsl:dubious": "#a02c2c",
}
CERTAINTY_LABEL = {
    "fsl:high": "high",
    "fsl:representative": "representative",
    "fsl:medium": "medium",
    "fsl:low": "low",
    "fsl:dubious": "dubious",
}

CAPTION_LICENSE = "CC BY 4.0, Florian Thiery"
CAPTION_SOURCES = [
    "https://github.com/Research-Squirrel-Engineers/campanian-ignimbrite-geo"
]

CAPTIONS = CaptionFile(
    str(SCRIPT_DIR / "captions.yaml"),
    header=(
        "captions.yaml - one entry per generated figure of the CI strand.\n"
        "\n"
        "Written by CI/plot_ci_findspots.py.\n"
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


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

POINT = re.compile(r"POINT\s*\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s*\)", re.IGNORECASE)


def parse_point(wkt: str) -> tuple[float, float] | None:
    """(lon, lat) from a WKT point, or None if the cell holds something else."""
    match = POINT.search(wkt or "")
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_findspots() -> list[dict]:
    """The findspot table, with the archaeological reading merged in.

    Same two sources and the same rule as ci_pipeline.py: spatialtype from the
    table, plus what the curated annotations add. The map and the graph cannot
    disagree about which sites are archaeological, because neither of them
    decides it.
    """
    annotations = {(row.get("ci_id") or "").strip(): row
                   for row in read_csv(CURATED_CSV)}
    curated_arch = {ci_id for ci_id, note in annotations.items()
                    if (note.get("isArchaeologicalSite") or "").strip().lower()
                    == "true"}

    sites = []
    for row in read_csv(RAW_CSV):
        point = parse_point(row.get("wkt", ""))
        if point is None:
            continue
        ci_id = (row.get("id") or "").strip()
        spatial = (row.get("spatialtype") or "").strip()
        sites.append({
            "id": ci_id,
            "label": (row.get("label") or "").strip(),
            "lon": point[0],
            "lat": point[1],
            "certainty": (row.get("certainty") or "").strip(),
            "spatialtype": spatial,
            "is_arch": "ArchaeologicalSite" in spatial or ci_id in curated_arch,
            "curated": ci_id in curated_arch and "ArchaeologicalSite" not in spatial,
        })
    return sites


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def ring(lon0: float, lat0: float, radius_km: float,
         steps: int = 361) -> tuple[list[float], list[float]]:
    """A circle of constant great-circle distance, in lon/lat.

    Drawn as a real geodesic circle rather than as an ellipse in degrees: at
    45 degrees north the two differ by enough to misplace a 2000 km ring by
    several degrees of longitude, and the rings are the only scale the figure
    has.
    """
    p0, l0 = math.radians(lat0), math.radians(lon0)
    d = radius_km / EARTH_RADIUS_KM
    lons, lats = [], []
    for i in range(steps):
        bearing = math.radians(i * 360.0 / (steps - 1))
        lat = math.asin(math.sin(p0) * math.cos(d)
                        + math.cos(p0) * math.sin(d) * math.cos(bearing))
        lon = l0 + math.atan2(
            math.sin(bearing) * math.sin(d) * math.cos(p0),
            math.cos(d) - math.sin(p0) * math.sin(lat))
        lons.append(math.degrees(lon))
        lats.append(math.degrees(lat))
    return lons, lats


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_map(ax, sites: list[dict], source: dict, legend: bool = True) -> None:
    lons = [s["lon"] for s in sites]
    lats = [s["lat"] for s in sites]
    lon_lo, lon_hi = min(lons) - 2.5, max(lons) + 2.5
    lat_lo, lat_hi = min(lats) - 2.0, max(lats) + 2.0

    furthest = max(haversine_km(source["lon"], source["lat"], s["lon"], s["lat"])
                   for s in sites)
    for step in range(1, math.ceil(furthest / RING_STEP_KM) + 1):
        radius = step * RING_STEP_KM
        rx, ry = ring(source["lon"], source["lat"], radius)
        ax.plot(rx, ry, color="#b8b3aa", linewidth=0.8, linestyle=(0, (4, 3)),
                zorder=1)
        # Labelled due north of the source, where the point cloud is thinnest.
        # A ring whose northern arc is off the window is labelled where it
        # crosses the top edge instead - otherwise the outer rings, which are
        # the ones a reader actually needs, would be the unlabelled ones.
        north_lat = source["lat"] + radius / 111.32
        if north_lat < lat_hi - 0.5:
            ax.text(source["lon"], north_lat - 0.35, f"{radius} km",
                    fontsize=8, color="#8a857c", ha="center", va="top",
                    zorder=1)
            continue
        for x, y in zip(rx, ry):
            if lon_lo < x < lon_hi and abs(y - lat_hi) < 0.35:
                ax.text(x, y - 0.6, f"{radius} km", fontsize=8,
                        color="#8a857c", ha="center", va="top", zorder=1)
                break

    ax.grid(True, color="#e5e1d8", linewidth=0.6, zorder=0)

    for level in CERTAINTY_ORDER:
        group = [s for s in sites if s["certainty"] == level and not s["is_arch"]]
        if group:
            ax.scatter([s["lon"] for s in group], [s["lat"] for s in group],
                       s=34, marker="o", facecolor=CERTAINTY_COLOUR[level],
                       edgecolor="white", linewidth=0.6, zorder=3,
                       label=CERTAINTY_LABEL[level] if legend else None)
    arch = [s for s in sites if s["is_arch"]]
    if arch:
        ax.scatter([s["lon"] for s in arch], [s["lat"] for s in arch],
                   s=78, marker="^",
                   facecolor=[CERTAINTY_COLOUR.get(s["certainty"], "#777777")
                              for s in arch],
                   edgecolor="#1a1a18", linewidth=0.8, zorder=4,
                   label="archaeological site" if legend else None)

    ax.scatter([source["lon"]], [source["lat"]], s=180, marker="*",
               facecolor="#a02c2c", edgecolor="#1a1a18", linewidth=0.8,
               zorder=5, label="Phlegraean Fields" if legend else None)

    ax.set_xlim(lon_lo, lon_hi)
    ax.set_ylim(lat_lo, lat_hi)
    ax.set_xlabel("Longitude [°E]")
    ax.set_ylabel("Latitude [°N]")
    # Equirectangular: one degree of longitude is shorter than one of latitude
    # by the cosine of the latitude. Without this the map is stretched east.
    ax.set_aspect(1.0 / math.cos(math.radians(sum(lats) / len(lats))))
    if legend:
        ax.legend(loc="lower right", fontsize=9, framealpha=0.92)


def draw_certainty_bars(ax, sites: list[dict]) -> None:
    counts = [sum(1 for s in sites if s["certainty"] == level)
              for level in CERTAINTY_ORDER]
    positions = list(range(len(CERTAINTY_ORDER)))
    ax.bar(positions, counts,
           color=[CERTAINTY_COLOUR[level] for level in CERTAINTY_ORDER],
           edgecolor="white", linewidth=0.8)
    for x, value in zip(positions, counts):
        if value:
            ax.text(x, value + 0.6, str(value), ha="center", fontsize=9,
                    color="#1a1a18")
    ax.set_xticks(positions)
    ax.set_xticklabels([CERTAINTY_LABEL[level] for level in CERTAINTY_ORDER],
                       rotation=30, ha="right")
    ax.set_ylabel("Findspots")
    ticks, limits, decimals = gf.nice_ticks(0, max(counts), target=5)
    ax.set_yticks([t for t in ticks if t >= 0])
    ax.set_ylim(0, limits[1])
    ax.grid(True, axis="y", color="#e5e1d8", linewidth=0.6)
    ax.set_axisbelow(True)


def draw_distance(ax, sites: list[dict], source: dict) -> None:
    for level in CERTAINTY_ORDER:
        group = [s for s in sites if s["certainty"] == level]
        if not group:
            continue
        distances = [haversine_km(source["lon"], source["lat"],
                                  s["lon"], s["lat"]) for s in group]
        ax.scatter(distances, [CERTAINTY_ORDER.index(level)] * len(group),
                   s=30, facecolor=CERTAINTY_COLOUR[level], edgecolor="white",
                   linewidth=0.5, zorder=3)
    ax.set_yticks(list(range(len(CERTAINTY_ORDER))))
    ax.set_yticklabels([CERTAINTY_LABEL[level] for level in CERTAINTY_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Distance from the Phlegraean Fields [km]")
    ax.grid(True, color="#e5e1d8", linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def caption_stem(sites: list[dict], source: dict) -> str:
    arch = sum(1 for s in sites if s["is_arch"])
    curated = sum(1 for s in sites if s["curated"])
    distances = [haversine_km(source["lon"], source["lat"], s["lon"], s["lat"])
                 for s in sites]
    return (
        f"{len(sites)} documented findspots of Campanian Ignimbrite tephra, "
        f"{arch} of them archaeological sites ({curated} of those added by "
        f"geo-lod's curated annotations, the rest stated in the source table). "
        f"Distances from the Phlegraean Fields run to "
        f"{max(distances):.0f} km."
    )


def figure_map(sites: list[dict], source: dict) -> None:
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=DPI)
    draw_map(ax, sites, source)
    fig.tight_layout()
    gf.save_figure(fig, str(OUTPUT_DIR / "ci_findspots_map"), dpi=DPI)
    plt.close(fig)
    CAPTIONS.add(
        "ci_findspots_map",
        caption=(caption_stem(sites, source) + " Positions are drawn on an "
                 "equirectangular grid without a basemap; the dashed rings "
                 f"mark great-circle distances at {RING_STEP_KM} km intervals "
                 "around the source. Colour is the certainty of the findspot, "
                 "triangles are archaeological sites."),
        license=CAPTION_LICENSE,
        sources=CAPTION_SOURCES,
    )


def figure_certainty(sites: list[dict], source: dict) -> None:
    fig = plt.figure(figsize=(15, 7.5), dpi=DPI)
    grid = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0],
                            height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.35)
    draw_map(fig.add_subplot(grid[:, 0]), sites, source)
    draw_certainty_bars(fig.add_subplot(grid[0, 1]), sites)
    draw_distance(fig.add_subplot(grid[1, 1]), sites, source)
    gf.save_figure(fig, str(OUTPUT_DIR / "ci_findspots_certainty"), dpi=DPI)
    plt.close(fig)

    counts = {level: sum(1 for s in sites if s["certainty"] == level)
              for level in CERTAINTY_ORDER}
    stated = ", ".join(f"{CERTAINTY_LABEL[level]} {counts[level]}"
                       for level in CERTAINTY_ORDER if counts[level])
    CAPTIONS.add(
        "ci_findspots_certainty",
        caption=(caption_stem(sites, source) + " Left: the findspot map, "
                 "coloured by certainty, triangles are archaeological sites. "
                 "Top right: how many findspots carry each certainty level "
                 f"({stated}). Bottom right: the same levels against "
                 "great-circle distance from the source, which is where a "
                 "distant findspot that is only weakly attested becomes "
                 "visible as such."),
        license=CAPTION_LICENSE,
        sources=CAPTION_SOURCES,
    )


def build() -> bool:
    print("\n" + "=" * 72)
    print("  Campanian Ignimbrite findspots - figures")
    print("=" * 72)

    sites = load_findspots()
    source = next((s for s in sites if s["id"] == SOURCE_ID), None)
    if source is None:
        raise ValueError(f"findspot {SOURCE_ID} (the Phlegraean Fields, the "
                         f"centre of the distance rings) is not in the table")
    print(f"  {len(sites)} findspots with a geometry")
    print(f"  {sum(1 for s in sites if s['is_arch'])} archaeological, "
          f"{sum(1 for s in sites if s['curated'])} of them from the curated "
          f"annotations")
    print(f"  source: {source['label']} "
          f"({source['lon']:.4f}, {source['lat']:.4f})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print()
    figure_map(sites, source)
    figure_certainty(sites, source)

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
