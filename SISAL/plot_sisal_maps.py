#!/usr/bin/env python3
"""plot_sisal_maps.py - where the SISAL caves are.

    SISAL/plots/sisal_sites_map                all twelve, on the world
    SISAL/plots/sisal_sites_map_europe         the European cluster
    SISAL/plots/sisal_sites_map_east_asian_monsoon
    SISAL/plots/sisal_sites_map_south_american_monsoon

Four maps and three layers, and the layers are the point. The catalogue -
365 caves SISAL v3 knows - lies underneath in pale grey. The caves geo-lod
screened and found archaeological carry a ring. The twelve of the cut are
drawn in the colour of their climate system, the same colours the cluster
plates use, so that a reader moving between the two figures does not have to
relearn them.

What that shows, which no single-layer map does: the twelve are a selection,
and one can see out of what. A map of twelve dots states that twelve caves
exist.

Sources
-------
``data/derived/sisal/catalogue/sites.csv`` for the catalogue, the same file
the RDF cave nodes come from, and ``data/curated/sisal_site_annotations.csv``
for the archaeological reading. The cut itself is not read: which twelve sites
it holds, and in which cluster, stands in ``plot_sisal_from_csv.SITES``, and
a second list here would be a second truth.

Run through main.py, or standalone from the repository root:

    python SISAL/plot_sisal_maps.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CATALOGUE_CSV = ROOT / "data" / "derived" / "sisal" / "catalogue" / "sites.csv"
CURATED_CSV = ROOT / "data" / "curated" / "sisal_site_annotations.csv"
OUTPUT_DIR = SCRIPT_DIR / "maps"
REPORT_PATH = SCRIPT_DIR / "report" / "maps_report.txt"

sys.path.insert(0, str(ROOT / "ontology"))
import geo_lod_basemap as gb  # noqa: E402
import geo_lod_figures as gf  # noqa: E402
from geo_lod_captions import CaptionFile  # noqa: E402

sys.path.insert(0, str(SCRIPT_DIR))
import sisal_plates  # noqa: E402
from plot_sisal_from_csv import SITES  # noqa: E402

plt.rcParams["svg.hashsalt"] = "geo-lod"

#: Screen dpi of the figure while it is being laid out. It is not the
#: resolution of the JPG - that comes from geo_lod_figures, which reads
#: it from the environment so that main.py --dpi can reach every
#: drawing script.
DPI = 100

#: One colour per climate system. Taken from the same tab10 order the cluster
#: plates draw their panels in, so the two figure families agree.
CLUSTER_COLOUR = {
    "europe": "#1f77b4",
    "east_asian_monsoon": "#d62728",
    "south_american_monsoon": "#2ca02c",
}

CATALOGUE_COLOUR = "#a9a49a"
ARCHAEOLOGY_EDGE = "#7a3e12"

#: Margin around a cluster window, in degrees. Enough that no cave sits on the
#: frame, little enough that the cluster fills the figure.
CLUSTER_MARGIN = 9.0

CAPTION_LICENSE = "CC BY 4.0, Florian Thiery"
CAPTION_SOURCES = [
    "https://doi.org/10.5194/essd-16-1933-2024",
    gb.LAND_SOURCE,
]

CAPTIONS = CaptionFile(
    str(SCRIPT_DIR / "captions.yaml"),
    header=(
        "captions.yaml - one entry per generated figure of the SISAL strand.\n"
        "\n"
        "Written by SISAL/plot_sisal_from_csv.py and SISAL/plot_sisal_maps.py.\n"
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

def read_csv(path: Path, hint: str = "") -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {hint}".strip())
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def cut_clusters() -> dict[int, str]:
    """site_id -> cluster, from the list the plot script already keeps.

    The slug carries the id in front - "58_spannagel" - which is how the cut
    names its figures, so the id can be read back out of it without a second
    table.
    """
    clusters = {}
    for site in SITES:
        clusters[int(site["slug"].split("_", 1)[0])] = site["cluster"]
    return clusters


def load_sites() -> list[dict]:
    catalogue = read_csv(CATALOGUE_CSV,
                         "Fetch the cut with SISAL/sisal_import.py.")
    annotations = {row["site_id"]: row for row in read_csv(
        CURATED_CSV, "This file is maintained by hand in geo-lod.")}
    clusters = cut_clusters()

    sites = []
    for row in catalogue:
        lat, lon = number(row.get("latitude")), number(row.get("longitude"))
        if lat is None or lon is None:
            continue
        site_id = int(row["site_id"])
        note = annotations.get(row["site_id"], {})
        sites.append({
            "id": site_id,
            "label": (row.get("site_name") or "").strip(),
            "lon": lon,
            "lat": lat,
            "cluster": clusters.get(site_id),
            "is_arch": (note.get("isArchaeologicalSite") or "").strip().lower()
                       == "true",
        })
    return sites


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_map(ax, sites: list[dict], bbox, label_cut: bool = False,
             graticule_step: float = 30.0, focus: str | None = None) -> None:
    """The three layers. With *focus*, only that cluster is a cluster.

    On a cluster map the other two climate systems have no business in the
    legend - their sites are thousands of kilometres outside the window, and a
    legend entry for a symbol that does not appear is a promise the figure
    does not keep. Sites of another cluster that do fall inside the window are
    drawn as what they are here: catalogue caves.
    """
    lon_lo, lat_lo, lon_hi, lat_hi = bbox
    ax.set_xlim(lon_lo, lon_hi)
    ax.set_ylim(lat_lo, lat_hi)
    ax.set_aspect(gb.aspect_for(lat_lo, lat_hi))
    gb.draw_land(ax, bbox, str(ROOT), zorder=0)
    gb.draw_graticule(ax, step=graticule_step, zorder=1)

    catalogue = [s for s in sites
                 if s["cluster"] is None
                 or (focus is not None and s["cluster"] != focus)]
    ax.scatter([s["lon"] for s in catalogue], [s["lat"] for s in catalogue],
               s=12, marker="o", facecolor=CATALOGUE_COLOUR, edgecolor="none",
               alpha=0.85, zorder=2, label="SISAL v3 catalogue")

    archaeological = [s for s in sites if s["is_arch"]]
    if archaeological:
        ax.scatter([s["lon"] for s in archaeological],
                   [s["lat"] for s in archaeological],
                   s=62, marker="o", facecolor="none",
                   edgecolor=ARCHAEOLOGY_EDGE, linewidth=1.2, zorder=3,
                   label="archaeological record")

    for key, label, _ in sisal_plates.CLUSTERS:
        if focus is not None and key != focus:
            continue
        group = [s for s in sites if s["cluster"] == key]
        if not group:
            continue
        ax.scatter([s["lon"] for s in group], [s["lat"] for s in group],
                   s=90, marker="^", facecolor=CLUSTER_COLOUR[key],
                   edgecolor="#1a1a18", linewidth=0.8, zorder=4, label=label)

    if label_cut:
        for site in sites:
            if site["cluster"] is None or (focus and site["cluster"] != focus):
                continue
            if not (lon_lo <= site["lon"] <= lon_hi
                    and lat_lo <= site["lat"] <= lat_hi):
                continue
            ax.annotate(site["label"], (site["lon"], site["lat"]),
                        textcoords="offset points", xytext=(9, -3), fontsize=8,
                        color="#3a3833", zorder=6)

    ax.set_xlabel("Longitude [°E]")
    ax.set_ylabel("Latitude [°N]")


def counts(sites: list[dict]) -> tuple[int, int, int]:
    return (len(sites),
            sum(1 for s in sites if s["is_arch"]),
            sum(1 for s in sites if s["cluster"] is not None))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_world(sites: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(15, 8), dpi=DPI)
    draw_map(ax, sites, (-180.0, -60.0, 180.0, 80.0), graticule_step=30.0)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    gf.save_figure(fig, str(OUTPUT_DIR / "sisal_sites_map"))
    plt.close(fig)

    total, arch, cut = counts(sites)
    CAPTIONS.add(
        "sisal_sites_map",
        caption=(f"The {total} cave sites of the SISAL v3 catalogue, the "
                 f"{arch} of them with an archaeological record, and the "
                 f"{cut} sites of the geo-lod cut in the colours of their "
                 f"climate system. The cut is a selection; the grey layer is "
                 f"what it was selected from."),
        license=CAPTION_LICENSE,
        sources=CAPTION_SOURCES,
    )


def figure_cluster(sites: list[dict], key: str, label: str,
                   phrase: str) -> None:
    group = [s for s in sites if s["cluster"] == key]
    if not group:
        print(f"  ⚠  no sites in cluster {key} - no figure")
        return
    lons = [s["lon"] for s in group]
    lats = [s["lat"] for s in group]
    bbox = (min(lons) - CLUSTER_MARGIN, min(lats) - CLUSTER_MARGIN,
            max(lons) + CLUSTER_MARGIN, max(lats) + CLUSTER_MARGIN)

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=DPI)
    draw_map(ax, sites, bbox, label_cut=True, focus=key,
             graticule_step=gf.nice_step(bbox[2] - bbox[0], 6))
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    gf.save_figure(fig, str(OUTPUT_DIR / f"sisal_sites_map_{key}"))
    plt.close(fig)

    inside = [s for s in sites
              if bbox[0] <= s["lon"] <= bbox[2] and bbox[1] <= s["lat"] <= bbox[3]]
    total, arch, _ = counts(inside)
    CAPTIONS.add(
        f"sisal_sites_map_{key}",
        caption=(f"The {len(group)} sites of the {phrase} cluster, named, with "
                 f"the {total} catalogue caves of the same window behind them "
                 f"and the {arch} of those carrying an archaeological record "
                 f"ringed. Colours are the ones the {phrase} cluster plate "
                 f"uses."),
        license=CAPTION_LICENSE,
        sources=CAPTION_SOURCES,
    )


def build() -> bool:
    print("\n" + "=" * 72)
    print("  SISAL cave sites - maps")
    print("=" * 72)

    sites = load_sites()
    total, arch, cut = counts(sites)
    print(f"  {total} catalogue sites, {arch} with an archaeological record")
    print(f"  {cut} sites of the cut, in "
          f"{len({s['cluster'] for s in sites if s['cluster']})} climate systems")

    missing = {s["cluster"] for s in sites if s["cluster"]} - {
        key for key, _, _ in sisal_plates.CLUSTERS}
    if missing:
        print(f"  ⚠  cluster without a definition in sisal_plates: "
              f"{', '.join(sorted(missing))}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print()
    figure_world(sites)
    for key, label, phrase in sisal_plates.CLUSTERS:
        figure_cluster(sites, key, label, phrase)

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
