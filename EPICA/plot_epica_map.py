#!/usr/bin/env python3
"""plot_epica_map.py - where Dome C is.

    EPICA/plots/epica_dome_c_map

One drilling site, so the map has to earn its place by saying something the
coordinates do not. It says two things.

The continent, in south polar stereographic. A single point on a world map in
plate carrée tells a reader nothing about Antarctica, and plate carrée is the
wrong projection for a continent that surrounds the pole: it puts the pole
along the bottom edge as a line and stretches the coast into a band. The
projection here is ten lines of arithmetic in ``geo_lod_basemap``, no
projection library.

And the inset: the two PANGAEA events of this core, EDC99 and DomeC, at the
coordinates their headers carry - about 1.3 km apart. That is a difference
between metadata records, not a second borehole, and it is the kind of thing
that disappears when a figure rounds a position to the nearest degree. The
graph keeps both events; so does the map.

Run through main.py, or standalone from the repository root:

    python EPICA/plot_epica_map.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "plots"
REPORT_PATH = SCRIPT_DIR / "report" / "map_report.txt"

sys.path.insert(0, str(ROOT / "ontology"))
import geo_lod_basemap as gb  # noqa: E402
import geo_lod_figures as gf  # noqa: E402
from geo_lod_captions import CaptionFile  # noqa: E402

sys.path.insert(0, str(SCRIPT_DIR))
import epica_data as ed  # noqa: E402

plt.rcParams["svg.hashsalt"] = "geo-lod"

DPI = 100

#: How far north the map reaches. Far enough that the Southern Ocean frames
#: the continent, not so far that Patagonia and New Zealand join in.
MAP_MAX_LAT = -58.0

EVENT_COLOUR = {"EDC99": "#1a4f8a", "DomeC": "#d1691f"}
SITE_COLOUR = "#1a4f8a"

CAPTION_LICENSE = "CC BY 4.0, Florian Thiery"
CAPTION_SOURCES = ["https://www.pangaea.de/", gb.LAND_SOURCE]

CAPTIONS = CaptionFile(
    str(SCRIPT_DIR / "captions.yaml"),
    header=(
        "captions.yaml - one entry per generated figure of the EPICA strand.\n"
        "\n"
        "Written by EPICA/epica_plates.py and EPICA/plot_epica_map.py.\n"
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


def local_km(lon: float, lat: float, lon0: float, lat0: float
             ) -> tuple[float, float]:
    """Offset in km from a reference point, east and north.

    At 75 degrees south a degree of longitude is a quarter of a degree of
    latitude, and an inset in degrees would show the two events four times
    further apart east-west than they are.
    """
    return ((lon - lon0) * 111.320 * math.cos(math.radians(lat)),
            (lat - lat0) * 110.574)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def draw_continent(ax) -> tuple[int, int]:
    rings, vertices = gb.draw_land_polar(ax, str(ROOT), max_lat=MAP_MAX_LAT)
    gb.draw_graticule_polar(ax, max_lat=MAP_MAX_LAT)

    site = ed.EVENTS["EDC99"]
    x, y = gb.south_polar(site["lon"], site["lat"])
    ax.scatter([x], [y], s=190, marker="*", facecolor=SITE_COLOUR,
               edgecolor="#1a1a18", linewidth=0.8, zorder=5)
    ax.annotate("EPICA Dome C", (x, y), textcoords="offset points",
                xytext=(12, -4), fontsize=11, color="#1a1a18", zorder=6)
    ax.annotate(f"{abs(site['lat']):.2f}°S {site['lon']:.2f}°E, "
                f"{site['elevation_m']:.0f} m",
                (x, y), textcoords="offset points", xytext=(12, -18),
                fontsize=8, color="#5f5e5a", zorder=6)
    return rings, vertices


def draw_events_inset(ax) -> float:
    """The two PANGAEA events at their own scale, in km."""
    reference = ed.EVENTS["EDC99"]
    for name, event in ed.EVENTS.items():
        x, y = local_km(event["lon"], event["lat"],
                        reference["lon"], reference["lat"])
        ax.scatter([x], [y], s=70, marker="o",
                   facecolor=EVENT_COLOUR.get(name, "#777777"),
                   edgecolor="#1a1a18", linewidth=0.7, zorder=4, label=name)
    # A square window around the two points. With plain autoscaling and an
    # aspect of one, a pair 1.3 km apart east-west and 0.2 km apart
    # north-south gives an inset the shape of a pencil.
    xs, ys = [], []
    for event in ed.EVENTS.values():
        x, y = local_km(event["lon"], event["lat"],
                        reference["lon"], reference["lat"])
        xs.append(x)
        ys.append(y)
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    half = max(max(xs) - min(xs), max(ys) - min(ys), 0.4) * 0.9
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    ax.set_xlabel("East [km]", fontsize=8)
    ax.set_ylabel("North [km]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect(1.0)
    ax.grid(True, color=gb.GRATICULE_COLOUR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.92)
    ax.set_title("PANGAEA events", fontsize=9, color="#3a3833")
    return haversine_km(ed.EVENTS["EDC99"]["lon"], ed.EVENTS["EDC99"]["lat"],
                        ed.EVENTS["DomeC"]["lon"], ed.EVENTS["DomeC"]["lat"])


def build() -> bool:
    print("\n" + "=" * 72)
    print("  EPICA Dome C - map")
    print("=" * 72)

    fig = plt.figure(figsize=(9.5, 9), dpi=DPI)
    main = fig.add_axes((0.02, 0.02, 0.96, 0.96))
    rings, vertices = draw_continent(main)
    print(f"  Antarctic coastline: {rings} rings, {vertices} vertices")

    inset = fig.add_axes((0.70, 0.06, 0.26, 0.24))
    separation = draw_events_inset(inset)
    print(f"  EDC99 to DomeC: {separation:.2f} km apart in the metadata")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gf.save_figure(fig, str(OUTPUT_DIR / "epica_dome_c_map"), dpi=DPI)
    plt.close(fig)

    site = ed.EVENTS["EDC99"]
    CAPTIONS.add(
        "epica_dome_c_map",
        caption=(f"EPICA Dome C on the Antarctic ice sheet, "
                 f"{abs(site['lat']):.2f}°S {site['lon']:.2f}°E at "
                 f"{site['elevation_m']:.0f} m, in south polar stereographic "
                 f"projection; parallels at 60, 70 and 80°S. The inset shows "
                 f"the two PANGAEA events of this core at their own scale: "
                 f"EDC99, which carries the methane, deuterium and dust "
                 f"records, and DomeC, which carries the two 2023 gas "
                 f"datasets. Their headers place them {separation:.1f} km "
                 f"apart although the material is from the same core - a "
                 f"difference between metadata records, not a second "
                 f"borehole."),
        license=CAPTION_LICENSE,
        sources=CAPTION_SOURCES,
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
