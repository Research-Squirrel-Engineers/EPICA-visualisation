#!/usr/bin/env python3
"""plot_ci_findspots.py - the two figures of the CI strand.

    CI/plots/ci_findspots_map          where the findspots are
    CI/plots/ci_findspots_certainty    the same map plus how well each point
                                       is known

Split from ci_pipeline.py the way EPICA and SISAL are split: the triples are
written first, the figures afterwards, so that an error in the data shows up
at the RDF step rather than after the drawing.

    CI/plots/ci_findspots_campania       the cluster around the source

Three maps, because the record has two scales. Two thirds of the findspots
sit within 500 km of the Phlegraean Fields, and on a map that reaches Kostenki
they are a blob. The third figure is that blob at its own scale.

Background
----------
Land polygons from ``geo_lod_basemap`` - Natural Earth, clipped to the window
and thinned to the size the figure is drawn at. No country borders: the
eruption is 39 ka old and a modern boundary drawn across it states something
that was not there.

Over the land go rings of constant great-circle distance from the source, at
500 km intervals, as many as the furthest findspot needs. What a Campanian
Ignimbrite map has to show is a distance, and a coastline alone would only
suggest it.

One thing the background cannot say, and the caption therefore does: the
coastline is the modern one. At the time of the eruption sea level stood some
80 m lower, and the northern Adriatic in particular was dry land.

The positions are drawn on an equirectangular grid with the aspect corrected
at the mean latitude of the window.
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
#: Maps go beside the data figures, not among them: CI/plots holds what a
#: reader plots against an axis, CI/maps what is drawn against a
#: coastline. Same split in every strand.
OUTPUT_DIR = SCRIPT_DIR / "maps"
REPORT_PATH = SCRIPT_DIR / "report" / "figures_report.txt"

sys.path.insert(0, str(ROOT / "ontology"))
import geo_lod_basemap as gb  # noqa: E402
import geo_lod_figures as gf  # noqa: E402
from geo_lod_captions import CaptionFile  # noqa: E402

# Byte-identical SVG across runs: the salt fixes the ids matplotlib gives clip
# paths, save_figure drops the date from the metadata.
plt.rcParams["svg.hashsalt"] = "geo-lod"

#: Screen dpi of the figure while it is being laid out. It is not the
#: resolution of the JPG - that comes from geo_lod_figures, which reads
#: it from the environment so that main.py --dpi can reach every
#: drawing script.
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
    "https://github.com/Research-Squirrel-Engineers/campanian-ignimbrite-geo",
    gb.LAND_SOURCE,
]

#: The window of the close-up, as a margin in degrees around the source. Two
#: thirds of the findspots fall inside it.
CAMPANIA_MARGIN = 2.2

#: What the caption has to say about a modern coastline under a 39 ka event.
SEA_LEVEL_NOTE = (
    "The coastline is the modern one; at the time of the eruption sea level "
    "stood some 80 m lower and the shelf areas, the northern Adriatic above "
    "all, were dry land."
)

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


#: Label placement. The offsets are tried in this order, and they grow: a
#: label that cannot sit close is moved out and connected back to its point
#: with a hairline, rather than dropped or written across its neighbour. The
#: pairs are in typographic points, x then y.
LABEL_OFFSETS = (
    (8, 4), (8, -12), (-8, 4), (-8, -12),
    (16, 16), (16, -24), (-16, 16), (-16, -24),
    (28, 30), (28, -40), (-28, 30), (-28, -40),
    (40, 48), (40, -60), (-40, 48), (-40, -60),
)
#: Beyond this distance from its point a label gets a leader line. Closer than
#: that the eye makes the connection on its own and a line is clutter.
LEADER_FROM_POINTS = 14.0
LABEL_FONTSIZE = 7
#: Rough width of a character at LABEL_FONTSIZE, in points. Matplotlib can
#: measure the real thing, but only after a draw, and the estimate is close
#: enough for a collision test.
LABEL_CHAR_WIDTH = 0.52


def place_labels(ax, sites: list[dict], bbox) -> tuple[int, int]:
    """Label the findspots without overlapping. Returns (labelled, dropped).

    Twenty-five of them lie within half a degree of Naples. The first pass
    tries to seat a label next to its point; where that collides, the label
    moves progressively further out and keeps a hairline back to where it
    belongs. Only when even the outermost ring is taken does a label go
    unwritten, and the count is reported rather than silently swallowed.
    """
    lon_lo, lat_lo, lon_hi, lat_hi = bbox
    width_points = ax.get_window_extent().width * 72.0 / ax.figure.dpi or 700.0
    height_points = ax.get_window_extent().height * 72.0 / ax.figure.dpi or 700.0
    per_x = (lon_hi - lon_lo) / width_points
    per_y = (lat_hi - lat_lo) / height_points

    placed: list[tuple[float, float, float, float]] = []
    labelled = dropped = 0
    # Archaeological first, then west to east: a stable order, so the same
    # labels survive from one run to the next.
    order = sorted(sites, key=lambda s: (not s["is_arch"], s["lon"], s["id"]))
    for site in order:
        text_width = len(site["label"]) * LABEL_FONTSIZE * LABEL_CHAR_WIDTH
        for dx, dy in LABEL_OFFSETS:
            x0 = site["lon"] + dx * per_x
            if dx < 0:
                x0 -= text_width * per_x
            y0 = site["lat"] + dy * per_y
            pad_x, pad_y = 2.5 * per_x, 2.0 * per_y
            box = (x0 - pad_x, y0 - pad_y,
                   x0 + text_width * per_x + pad_x,
                   y0 + LABEL_FONTSIZE * 1.25 * per_y + pad_y)
            # Inside the frame, or not at all. A label that runs off the
            # right edge is worse than one that moved to the other side.
            if (box[0] < lon_lo or box[2] > lon_hi
                    or box[1] < lat_lo or box[3] > lat_hi):
                continue
            if any(not (box[2] < other[0] or box[0] > other[2]
                        or box[3] < other[1] or box[1] > other[3])
                   for other in placed):
                continue
            far = math.hypot(dx, dy) > LEADER_FROM_POINTS
            ax.annotate(
                site["label"], (site["lon"], site["lat"]),
                textcoords="offset points", xytext=(dx, dy),
                fontsize=LABEL_FONTSIZE, color="#3a3833", zorder=6,
                ha="right" if dx < 0 else "left",
                va="bottom" if dy >= 0 else "top",
                arrowprops=dict(arrowstyle="-", linewidth=0.5,
                                color="#8a857c", shrinkA=1.0, shrinkB=3.0)
                if far else None)
            placed.append(box)
            labelled += 1
            break
        else:
            dropped += 1
    return labelled, dropped


def draw_map(ax, sites: list[dict], source: dict, legend: bool = True,
             bbox: tuple[float, float, float, float] | None = None,
             ring_labels: bool = True, label_sites: bool = False) -> None:
    """The findspot map. *bbox* overrides the window derived from the data."""
    lons = [s["lon"] for s in sites]
    lats = [s["lat"] for s in sites]
    if bbox is None:
        lon_lo, lon_hi = min(lons) - 2.5, max(lons) + 2.5
        lat_lo, lat_hi = min(lats) - 2.0, max(lats) + 2.0
    else:
        lon_lo, lat_lo, lon_hi, lat_hi = bbox

    # Limits before the land: the simplification tolerance follows the drawn
    # width of the axes, and that is only known once they are sized.
    ax.set_xlim(lon_lo, lon_hi)
    ax.set_ylim(lat_lo, lat_hi)
    ax.set_aspect(gb.aspect_for(lat_lo, lat_hi))
    gb.draw_land(ax, (lon_lo, lat_lo, lon_hi, lat_hi), str(ROOT), zorder=0)

    furthest = max(haversine_km(source["lon"], source["lat"], s["lon"], s["lat"])
                   for s in sites
                   if lon_lo <= s["lon"] <= lon_hi and lat_lo <= s["lat"] <= lat_hi)
    for step in range(1, math.ceil(furthest / RING_STEP_KM) + 1):
        radius = step * RING_STEP_KM
        rx, ry = ring(source["lon"], source["lat"], radius)
        ax.plot(rx, ry, color="#b8b3aa", linewidth=0.8, linestyle=(0, (4, 3)),
                zorder=1)
        # Labelled due north of the source, where the point cloud is thinnest.
        # A ring whose northern arc is off the window is labelled where it
        # crosses the top edge instead - otherwise the outer rings, which are
        # the ones a reader actually needs, would be the unlabelled ones.
        if not ring_labels:
            continue
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

    gb.draw_graticule(ax, step=gf.nice_step(lon_hi - lon_lo, 6), zorder=1)

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

    if label_sites:
        inside = [s for s in sites
                  if lon_lo <= s["lon"] <= lon_hi
                  and lat_lo <= s["lat"] <= lat_hi]
        labelled, dropped = place_labels(
            ax, inside, (lon_lo, lat_lo, lon_hi, lat_hi))
        print(f"    {labelled} of {len(inside)} findspots labelled"
              + (f", {dropped} had no room" if dropped else ""))

    ax.set_xlabel("Longitude [°E]")
    ax.set_ylabel("Latitude [°N]")
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
    gf.save_figure(fig, str(OUTPUT_DIR / "ci_findspots_map"))
    plt.close(fig)
    CAPTIONS.add(
        "ci_findspots_map",
        caption=(caption_stem(sites, source) + " Positions are drawn on an "
                 "equirectangular grid without a basemap; the dashed rings "
                 f"mark great-circle distances at {RING_STEP_KM} km intervals "
                 "around the source. Colour is the certainty of the findspot, "
                 "triangles are archaeological sites. " + SEA_LEVEL_NOTE),
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
    gf.save_figure(fig, str(OUTPUT_DIR / "ci_findspots_certainty"))
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


def figure_campania(sites: list[dict], source: dict) -> None:
    """The close-up: the cluster the wide map cannot resolve.

    The window follows the source rather than the data - a close-up whose
    extent moved with every findspot added would not be comparable between
    two releases of the table.
    """
    bbox = (source["lon"] - CAMPANIA_MARGIN, source["lat"] - CAMPANIA_MARGIN,
            source["lon"] + CAMPANIA_MARGIN, source["lat"] + CAMPANIA_MARGIN)
    inside = [s for s in sites
              if bbox[0] <= s["lon"] <= bbox[2] and bbox[1] <= s["lat"] <= bbox[3]]

    fig, ax = plt.subplots(figsize=(10, 9), dpi=DPI)
    draw_map(ax, sites, source, bbox=bbox, ring_labels=False, label_sites=True)
    fig.tight_layout()
    gf.save_figure(fig, str(OUTPUT_DIR / "ci_findspots_campania"))
    plt.close(fig)

    arch = sum(1 for s in inside if s["is_arch"])
    CAPTIONS.add(
        "ci_findspots_campania",
        caption=(f"The {len(inside)} findspots within {CAMPANIA_MARGIN:.0f} "
                 f"degrees of the Phlegraean Fields, {arch} of them "
                 f"archaeological sites, at a scale the overview map cannot "
                 f"resolve. Colour is the certainty of the findspot, the star "
                 f"is the source of the eruption, the dashed rings are the "
                 f"same {RING_STEP_KM} km great-circle intervals. "
                 + SEA_LEVEL_NOTE),
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
    figure_campania(sites, source)
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
