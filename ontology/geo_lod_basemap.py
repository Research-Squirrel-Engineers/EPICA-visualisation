"""
geo_lod_basemap.py
==================
A coastline under the points, without a geo stack.

Every strand of this repository draws a map at some point, and every one of
them would otherwise either go without a background or pull in cartopy. This
module is the third way: the land polygons of Natural Earth, read from a
GeoJSON with the standard library, clipped to the window and thinned to the
scale they are drawn at. No projection library, no GEOS, no PROJ, nothing that
has to be installed on Windows before a figure appears.

Lives next to ``geo_lod_figures.py``, which it complements: that module knows
how a figure reaches disk, this one knows what goes behind the data.

The source
----------
``data/raw/basemap/ne_50m_land.geojson`` is Natural Earth 1:50m land, public
domain, unchanged. One file for every map in the repository, because the
alternative - a coarse file for world maps and a fine one for regional ones -
puts a second decision in every call site and still gets it wrong at some
intermediate window.

Clipping and thinning
---------------------
Two steps, and both are about the SVG rather than about the picture.

``clip_ring`` cuts polygons at the window with Sutherland-Hodgman. Without it
a map of the Mediterranean carries the whole of Eurasia in its file, invisible
outside the axes but fully present in the bytes.

``simplify`` drops points that lie closer than a tolerance to the line their
neighbours span (Douglas-Peucker), and the tolerance is derived from the axis:
degrees per drawn point, times a fraction of a point. A vertex the plot cannot
resolve is not detail, it is weight. Measured on the world window: 60 669
points and 1.6 MB become 10 428 points and 353 KB with no visible change. On
a regional window the tolerance falls to a few hundredths of a degree and
almost nothing is dropped, which is the point of deriving it rather than
setting it.

What is deliberately not drawn
------------------------------
Country borders. The records in this repository run from the Holocene to
800 ka; a modern political boundary drawn across them states something that
was not there. Land, coastline, graticule - nothing else.

Also not drawn: the coastline of the period in question. What is available
here is the modern one, and where that matters - the CI map, where sea level
stood some 80 m lower at the time of the eruption - it belongs in the caption
rather than in a silent approximation.
"""

from __future__ import annotations

import json
import math
import os

#: The land file, relative to the repository root.
LAND_RELATIVE = os.path.join("data", "raw", "basemap", "ne_50m_land.geojson")

#: Source of the land polygons, for captions and for the provenance of any
#: figure that carries them.
LAND_SOURCE = "https://www.naturalearthdata.com/"
LAND_CREDIT = "Land polygons: Natural Earth 1:50m, public domain"

#: How much of a drawn point a vertex has to be worth to survive. Below this
#: the simplification cannot be seen at the size the figure is written at.
SIMPLIFY_POINT_FRACTION = 0.6

#: House colours of the background. Warm grey against the saturated series
#: colours of the strands, and a coastline a shade darker than the fill rather
#: than black: the coastline is context, not data.
LAND_FACE = "#eceae3"
LAND_EDGE = "#c3bfb5"
LAND_LINEWIDTH = 0.6
GRATICULE_COLOUR = "#e5e1d8"
GRATICULE_LINEWIDTH = 0.6

_CACHE: dict[str, list] = {}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def land_path(root: str) -> str:
    return os.path.join(root, LAND_RELATIVE)


def _rings_of(path: str) -> list[list[tuple[float, float]]]:
    """Every ring of the land file, read once per process."""
    if path in _CACHE:
        return _CACHE[path]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. It is Natural Earth 1:50m land, public domain, "
            f"and travels with the repository."
        )
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    rings = []
    for feature in data["features"]:
        geometry = feature["geometry"]
        polygons = (geometry["coordinates"]
                    if geometry["type"] == "MultiPolygon"
                    else [geometry["coordinates"]])
        for polygon in polygons:
            for ring in polygon:
                rings.append([(float(x), float(y)) for x, y in ring])
    _CACHE[path] = rings
    return rings


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

def _clip_side(ring, inside, cross):
    if not ring:
        return []
    out = []
    previous = ring[-1]
    previous_in = inside(previous)
    for point in ring:
        point_in = inside(point)
        if point_in:
            if not previous_in:
                out.append(cross(previous, point))
            out.append(point)
        elif previous_in:
            out.append(cross(previous, point))
        previous, previous_in = point, point_in
    return out


def clip_ring(ring, bbox):
    """*ring* cut to the rectangle *bbox* = (lon0, lat0, lon1, lat1)."""
    x0, y0, x1, y1 = bbox

    def at_x(x):
        return lambda a, b: (x, a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))

    def at_y(y):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]), y)

    sides = (
        (lambda p: p[0] >= x0, at_x(x0)),
        (lambda p: p[0] <= x1, at_x(x1)),
        (lambda p: p[1] >= y0, at_y(y0)),
        (lambda p: p[1] <= y1, at_y(y1)),
    )
    for inside, cross in sides:
        ring = _clip_side(ring, inside, cross)
        if not ring:
            return []
    return ring


# ---------------------------------------------------------------------------
# Thinning
# ---------------------------------------------------------------------------

def _perpendicular(point, start, end) -> float:
    (x, y), (x1, y1), (x2, y2) = point, start, end
    dx, dy = x2 - x1, y2 - y1
    if dx == 0.0 and dy == 0.0:
        return math.hypot(x - x1, y - y1)
    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))


def simplify(points, tolerance: float):
    """Douglas-Peucker, iterative so that a long coastline cannot recurse away."""
    if tolerance <= 0 or len(points) < 3:
        return points
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        worst, index = tolerance, None
        for k in range(first + 1, last):
            distance = _perpendicular(points[k], points[first], points[last])
            if distance > worst:
                worst, index = distance, k
        if index is not None:
            keep.add(index)
            stack.append((first, index))
            stack.append((index, last))
    return [points[k] for k in sorted(keep)]



def densify(ring, max_step: float):
    """Split segments longer than *max_step* degrees.

    Needed before a projection: a straight line in degrees is a curve on the
    map, and the longest straight lines in a clipped ring are exactly the ones
    the clip introduced along the window edge. Undensified, the southern
    boundary of an Antarctic map comes out as a chord across the ocean.
    """
    if max_step <= 0 or len(ring) < 2:
        return ring
    out = []
    for index, point in enumerate(ring):
        out.append(point)
        nxt = ring[(index + 1) % len(ring)]
        distance = math.hypot(nxt[0] - point[0], nxt[1] - point[1])
        pieces = int(distance / max_step)
        for k in range(1, pieces + 1):
            t = k / (pieces + 1.0)
            out.append((point[0] + (nxt[0] - point[0]) * t,
                        point[1] + (nxt[1] - point[1]) * t))
    return out


def tolerance_for(ax, span_degrees: float) -> float:
    """Simplification tolerance from the size the axes are drawn at.

    Degrees per typographic point, times ``SIMPLIFY_POINT_FRACTION``. Falls
    back to a world-scale value when the figure has no size yet, which happens
    only if a caller asks before layout.
    """
    try:
        width_points = ax.get_window_extent().width * 72.0 / ax.figure.dpi
    except Exception:
        width_points = 700.0
    if width_points <= 0:
        width_points = 700.0
    return span_degrees / width_points * SIMPLIFY_POINT_FRACTION


# ---------------------------------------------------------------------------
# Drawing, plate carrée
# ---------------------------------------------------------------------------

def draw_land(ax, bbox, root: str, face: str = LAND_FACE,
              edge: str = LAND_EDGE, linewidth: float = LAND_LINEWIDTH,
              zorder: int = 0) -> tuple[int, int]:
    """Land inside *bbox* on an axes in degrees. Returns (rings, vertices)."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon

    ax.figure.canvas.draw_idle()
    tolerance = tolerance_for(ax, bbox[2] - bbox[0])

    patches, vertices = [], 0
    for ring in _rings_of(land_path(root)):
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        if (max(xs) < bbox[0] or min(xs) > bbox[2]
                or max(ys) < bbox[1] or min(ys) > bbox[3]):
            continue
        cut = clip_ring(ring, bbox)
        if len(cut) < 3:
            continue
        cut = simplify(cut, tolerance)
        if len(cut) < 3:
            continue
        patches.append(Polygon(cut, closed=True))
        vertices += len(cut)

    ax.add_collection(PatchCollection(
        patches, facecolor=face, edgecolor=edge, linewidth=linewidth,
        zorder=zorder))
    return len(patches), vertices


def aspect_for(lat0: float, lat1: float) -> float:
    """Plate carrée aspect at the middle latitude of the window.

    One degree of longitude is shorter than one of latitude by the cosine of
    the latitude; without this correction every map in the repository is
    stretched east-west, and the further from the equator the worse.
    """
    middle = math.radians((lat0 + lat1) / 2.0)
    return 1.0 / max(math.cos(middle), 0.05)


def draw_graticule(ax, step: float = 10.0, zorder: int = 0) -> None:
    """A grid on whole degrees, drawn from the axis limits."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    for x in _multiples(x0, x1, step):
        ax.plot([x, x], [y0, y1], color=GRATICULE_COLOUR,
                linewidth=GRATICULE_LINEWIDTH, zorder=zorder)
    for y in _multiples(y0, y1, step):
        ax.plot([x0, x1], [y, y], color=GRATICULE_COLOUR,
                linewidth=GRATICULE_LINEWIDTH, zorder=zorder)


def _multiples(low: float, high: float, step: float) -> list[float]:
    first = math.ceil(low / step) * step
    values, value = [], first
    while value <= high:
        values.append(round(value, 6))
        value += step
    return values


# ---------------------------------------------------------------------------
# Drawing, south polar stereographic
# ---------------------------------------------------------------------------

def south_polar(lon: float, lat: float) -> tuple[float, float]:
    """Southern polar stereographic, in units of Earth radii.

    Ten lines instead of a projection library. A map of Antarctica in plate
    carrée is not a map of Antarctica: the continent surrounds the pole, and
    the pole is a line at the bottom of the frame rather than a point.
    """
    phi = math.radians(lat)
    lam = math.radians(lon)
    # r = 2 tan(45° + phi/2): zero at the south pole, growing towards the
    # equator. The factor of two is the usual scale and cancels out here,
    # since the axes are set from the projected extent.
    r = 2.0 * math.tan(math.pi / 4.0 + phi / 2.0)
    # 0 degrees longitude points down, 90 East to the right - the orientation
    # every published Antarctic map uses.
    return r * math.sin(lam), -r * math.cos(lam)


def draw_land_polar(ax, root: str, max_lat: float = -55.0,
                    face: str = LAND_FACE, edge: str = LAND_EDGE,
                    linewidth: float = LAND_LINEWIDTH,
                    zorder: int = 0) -> tuple[int, int]:
    """Land south of *max_lat*, projected. Returns (rings, vertices)."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon

    bbox = (-180.0, -90.0, 180.0, max_lat)
    patches, vertices = [], 0
    for ring in _rings_of(land_path(root)):
        if min(p[1] for p in ring) > max_lat:
            continue
        cut = clip_ring(ring, bbox)
        if len(cut) < 3:
            continue
        # Simplified in degrees before projecting: the tolerance is a distance
        # on the sphere, and after the projection it would mean something
        # different at the pole than at the edge of the map.
        cut = simplify(cut, 0.08)
        if len(cut) < 3:
            continue
        cut = densify(cut, 1.0)
        patches.append(Polygon([south_polar(x, y) for x, y in cut],
                               closed=True))
        vertices += len(cut)

    ax.add_collection(PatchCollection(
        patches, facecolor=face, edgecolor=edge, linewidth=linewidth,
        zorder=zorder))
    return len(patches), vertices


def draw_graticule_polar(ax, max_lat: float = -55.0,
                         parallels=(-60.0, -70.0, -80.0),
                         meridian_step: float = 30.0,
                         zorder: int = 0) -> None:
    """Parallels as circles, meridians as rays, with the parallels labelled."""
    for lat in parallels:
        if lat > max_lat:
            continue
        points = [south_polar(lon, lat) for lon in range(0, 361, 2)]
        ax.plot([p[0] for p in points], [p[1] for p in points],
                color=GRATICULE_COLOUR, linewidth=GRATICULE_LINEWIDTH,
                zorder=zorder)
        x, y = south_polar(45.0, lat)
        ax.text(x, y, f"{abs(lat):.0f}°S", fontsize=7, color="#8a857c",
                ha="center", va="center", zorder=zorder)
    edge = south_polar(0.0, max_lat)
    radius = math.hypot(*edge)
    for lon in _multiples(0.0, 359.0, meridian_step):
        x, y = south_polar(lon, max_lat)
        ax.plot([0.0, x], [0.0, y], color=GRATICULE_COLOUR,
                linewidth=GRATICULE_LINEWIDTH, zorder=zorder)
    ax.set_xlim(-radius * 1.04, radius * 1.04)
    ax.set_ylim(-radius * 1.04, radius * 1.04)
    ax.set_aspect(1.0)
    ax.axis("off")
