# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Two Sides of the FAIR4RS Coin — querying the published RDF
#
# This notebook runs three SPARQL queries against local Turtle
# snapshots of the *GeoScience-FAIRification-LOD* graph and turns each
# result into a view: a reconstructed proxy curve, a map of SISAL cave
# sites, and a map of Campanian Ignimbrite findspots. It is the local
# (Jupyter) companion to the browser-executable
# `local_ttl-fair4rs-coin-live.qmd`; both run the same queries on the
# same data, only the runtime and the mapping library differ.
#
# ## About this notebook
#
# The three queries each exercise one axis of the paper *Two Sides of
# the FAIR4RS Coin* (Thiery & Schenk, 2026): the **methodological**
# axis (can a published curve be traced back to, and rebuilt from, the
# choices that produced it?), the **spatial** axis (where are the
# sites?), and the **cross-domain** axis (which sites also belong to the
# cultural-heritage record?).
#
# ### Why this dataset
#
# The graph is compact, globally distributed, and deliberately
# multi-domain: EPICA Dome C ice-core observations, SISALv3 speleothem
# caves, and a Campanian Ignimbrite findspot catalogue share one
# `geo-lod` vocabulary. That makes it a good teaching graph for showing
# how reproducible computation and semantic interoperability are two
# sides of the same publication act, rather than separate concerns.
#
# ### What you'll learn
#
# - Loading several Turtle files into one in-memory `rdflib` graph and
#   querying it with SPARQL.
# - Turning SPARQL result bindings into a tidy `pandas` DataFrame.
# - Rebuilding a smoothed time series, and drawing two categorical
#   marker maps, straight from the queried data.
#
# ### Data-context notes
#
# - **Coordinate convention.** Geometries are GeoSPARQL WKT literals of
#   the form `<…EPSG/0/4326> POINT(lon lat)` — note the CRS prefix and
#   the `lon lat` axis order. Mapping libraries expect `[lat, lon]`, so
#   we swap the order once on parse.
# - **EPICA δ¹⁸O is atmospheric.** The δ¹⁸O series here is δ¹⁸O of
#   atmospheric O₂ (≈ −0.5…1.5 ‰), not the δ¹⁸O of the ice.
# - **Smoothing is modelled explicitly.** Each observation links to a
#   `SavitzkyGolayFilter` individual carrying `windowSize` and
#   `polyOrder`, alongside the raw value and the source DOI — that is
#   what makes the lineage query possible.
# - **Two spatial types per findspot.** A few CI findspots carry two
#   `hasSpatialType` values; the CI query groups them so each findspot
#   stays one row.
#
# ### Tooling notes
#
# `rdflib` parses and queries the graph, `pandas` holds the tables,
# `matplotlib` draws the reconstruction, and `folium` draws the maps.
# The browser companion swaps `folium` for hand-written Leaflet (folium
# writes HTML files, which is awkward under Pyodide) but is otherwise
# identical.
#
# ### Requirements
#
# ```
# pip install rdflib pandas matplotlib folium
# ```
#
# Place the four Turtle snapshots (`geo_lod_core.ttl`,
# `epica_dome_c.ttl`, `sisal_sites.ttl`, `ci_findspots.ttl`) next to
# this notebook, or run it from the repository so they are found
# automatically.

# %% [markdown]
# ## 0  Setup

# %%
import re
import json
import urllib.request
from pathlib import Path

from rdflib import Graph
import pandas as pd
import folium


def _in_notebook() -> bool:
    """True only inside a Jupyter kernel (ZMQInteractiveShell)."""
    try:
        from IPython import get_ipython

        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


import matplotlib

if not _in_notebook():
    matplotlib.use("Agg")  # plain-script run: no GUI window, save PNG instead
import matplotlib.pyplot as plt

try:
    HERE = Path(__file__).resolve().parent
except NameError:  # notebook: no __file__
    HERE = Path.cwd()


# Locate the four Turtle snapshots: next to the notebook, in the
# current directory, or within a sibling repository checkout.
def find_ttls() -> dict[str, Path]:
    names = {
        "core": "geo_lod_core.ttl",
        "epica": "epica_dome_c.ttl",
        "sisal": "sisal_sites.ttl",
        "ci": "ci_findspots.ttl",
    }
    try:
        base = Path(__file__).resolve().parent
    except NameError:
        base = Path.cwd()
    candidates = [base, Path.cwd()]
    # also try the canonical repo sub-paths relative to a repo root
    for start in (base, Path.cwd()):
        for p in [start, *start.parents]:
            if (p / "EPICA").is_dir() and (p / "SISAL").is_dir():
                return {
                    "core": p / "ontology" / "geo_lod_core.ttl",
                    "epica": p / "EPICA" / "rdf" / "epica_dome_c.ttl",
                    "sisal": p / "SISAL" / "rdf" / "sisal_sites.ttl",
                    "ci": p / "CI" / "rdf" / "ci_findspots.ttl",
                }
    for cand in candidates:
        if (cand / names["sisal"]).exists():
            return {k: cand / v for k, v in names.items()}
    raise FileNotFoundError(
        "Turtle snapshots not found - put the four .ttl files next to "
        "this notebook or run it from the repository."
    )


TTL = find_ttls()


def load_graph(*keys: str) -> Graph:
    g = Graph()
    for k in keys:
        with TTL[k].open("rb") as fh:
            g.parse(fh, format="turtle", publicID=TTL[k].as_uri())
    return g


_WKT_RE = re.compile(r"point\s*\(\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s*\)", re.IGNORECASE)


def parse_wkt(wkt):
    """Parse a WKT 'POINT(lon lat)' literal into (lat, lon)."""
    m = _WKT_RE.search(str(wkt) if wkt is not None else "")
    if not m:
        return (None, None)
    try:
        return (float(m.group(2)), float(m.group(1)))
    except ValueError:
        return (None, None)


def localname(uri) -> str:
    return str(uri).rsplit("/", 1)[-1].rsplit("#", 1)[-1]


# --- static map rendering for paper figures (matplotlib only) ----------
# Coastlines come from a small Natural Earth 110 m land GeoJSON, parsed
# with plain json and drawn as polygons -- no GEOS/GDAL/cartopy needed.
_LAND_FILE = "ne_110m_land.geojson"
_LAND_URL = (
    "https://raw.githubusercontent.com/nvkelso/"
    "natural-earth-vector/master/geojson/ne_110m_land.geojson"
)


def _load_land():
    """Return the Natural Earth land GeoJSON, fetching+caching if absent."""
    p = HERE / _LAND_FILE
    if not p.exists():
        try:
            urllib.request.urlretrieve(_LAND_URL, p)
        except Exception:
            return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _exterior_rings(geom):
    """Yield each polygon's exterior ring from a Polygon/MultiPolygon."""
    if geom["type"] == "Polygon":
        yield geom["coordinates"][0]
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            yield poly[0]


def render_static_map(points, out_png, title="", extent=None, legend_order=None):
    """Render coloured points over a coastline outline and save a PNG.

    points : list of {lat, lon, cat, color}
    extent : (lon_min, lon_max, lat_min, lat_max) or None for world
    """
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.collections import PatchCollection
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9, 4.8))
    land = _load_land()
    if land:
        patches = [
            MplPoly(ring, closed=True)
            for feat in land["features"]
            for ring in _exterior_rings(feat["geometry"])
        ]
        ax.add_collection(
            PatchCollection(
                patches, facecolor="#eceff1", edgecolor="#b0bec5", linewidths=0.4
            )
        )
    else:
        ax.text(
            0.5,
            0.5,
            "(coastlines unavailable - points only)",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            color="#999",
        )

    cat_color = {}
    for cat in legend_order or []:
        cat_color.setdefault(cat, None)
    for pt in points:
        cat_color[pt["cat"]] = pt["color"]
    for cat, color in cat_color.items():
        xs = [p["lon"] for p in points if p["cat"] == cat]
        ys = [p["lat"] for p in points if p["cat"] == cat]
        ax.scatter(
            xs,
            ys,
            s=22,
            c=color,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
            label=cat,
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c,
            markeredgecolor="white",
            markersize=8,
            label=cat,
        )
        for cat, c in cat_color.items()
        if c
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)

    if extent:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    else:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


print("Turtle files:", {k: v.name for k, v in TTL.items()})


# %% [markdown]
# ## A  Samples and smoothing lineage (EPICA)
#
# One query returns, for every ice-core observation, its age, raw value,
# Savitzky-Golay-smoothed value, the smoothing method with its window
# size and polynomial order, and the source DOI. The `method / window /
# order / source` columns are the **provenance lineage**; the `age / raw
# / savgol` columns are enough to **redraw the smoothed curve from the
# published RDF alone**, which the next cell does.

# %%
Q_SAMPLES = """
PREFIX geolod: <http://w3id.org/geo-lod/>
PREFIX sosa:   <http://www.w3.org/ns/sosa/>
PREFIX prov:   <http://www.w3.org/ns/prov#>

SELECT ?obs ?type ?age ?raw ?savgol ?method ?window ?order ?source WHERE {
  ?obs a sosa:Observation ;
       geolod:measurementType ?type ;
       sosa:resultTime ?age ;
       sosa:hasSimpleResult ?raw ;
       geolod:smoothedValue_savgol ?savgol ;
       geolod:smoothingMethod_savgol ?method ;
       prov:wasDerivedFrom ?source .
  ?method geolod:windowSize ?window .
  OPTIONAL { ?method geolod:polyOrder ?order }
}
ORDER BY ?type ?age
"""

g_epica = load_graph("core", "epica")

df_samples = pd.DataFrame(
    [
        {
            "obs": localname(r["obs"]),
            "type": localname(r["type"]).replace("MeasurementType_", ""),
            "age_ka": float(r["age"]),
            "raw": float(r["raw"]),
            "savgol": float(r["savgol"]),
            "method": localname(r["method"]),
            "window": int(r["window"]),
            "order": int(r["order"]) if r["order"] is not None else None,
            "source": str(r["source"]),
        }
        for r in g_epica.query(Q_SAMPLES)
    ]
)

print(
    f"{len(df_samples)} observations "
    f"({', '.join(f'{t}: {n}' for t, n in df_samples['type'].value_counts().items())})"
)
df_samples.head()


# %% [markdown]
# ### Rebuild the smoothed curves from the RDF
#
# One panel per proxy: the raw measurements and the Savitzky-Golay
# series, both read straight from the graph above — the published figure
# regenerated from the Linked Open Data rather than from the original
# pipeline.

# %%
proxies = list(df_samples["type"].unique())
fig, axes = plt.subplots(
    len(proxies), 1, figsize=(9, 3.0 * len(proxies)), squeeze=False
)
for ax, proxy in zip(axes[:, 0], proxies):
    sub = df_samples[df_samples["type"] == proxy].sort_values("age_ka")
    ax.plot(sub["age_ka"], sub["raw"], lw=0.5, alpha=0.5, label="raw")
    ax.plot(sub["age_ka"], sub["savgol"], lw=1.3, label="Savitzky-Golay")
    ax.set_xlabel("age [ka BP]")
    ax.set_ylabel(proxy)
    ax.set_title(f"EPICA Dome C {proxy} rebuilt from the published RDF")
    ax.legend(loc="best", fontsize=8)
    ax.invert_xaxis()
fig.tight_layout()
if not _in_notebook():
    fig.savefig(
        HERE / "reconstruction.png", dpi=110, bbox_inches="tight", pad_inches=0.1
    )
    print("saved reconstruction.png")
fig


# %% [markdown]
# ## B  SISAL cave sites (marker map)
#
# Every SISALv3 cave with coordinates, plus whether it is also an
# archaeological cave site, whether it is UNESCO World Heritage, and its
# Wikidata cross-link. `EXISTS` keeps the flags to one row per cave. The
# marker colour encodes the highest-priority status.

# %%
Q_SISAL = """
PREFIX geolod: <http://w3id.org/geo-lod/>
PREFIX geo:   <http://www.opengis.net/ont/geosparql#>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:   <http://www.w3.org/2002/07/owl#>

SELECT ?site ?label ?wkt ?isArch ?isUnesco ?wikidata WHERE {
  ?site a geolod:Cave ;
        rdfs:label ?label ;
        geo:hasGeometry / geo:asWKT ?wkt .
  BIND(EXISTS { ?site a geolod:ArchaeologicalCaveSite } AS ?isArch)
  BIND(EXISTS { ?site geolod:isUNESCOWorldHeritage true } AS ?isUnesco)
  OPTIONAL { ?site owl:sameAs ?wikidata
             FILTER(STRSTARTS(STR(?wikidata), "https://www.wikidata.org/")) }
}
ORDER BY ?site
"""

g_sisal = load_graph("core", "sisal")

df_sisal = (
    pd.DataFrame(
        [
            {
                "site": localname(r["site"]),
                "label": str(r["label"]),
                **dict(zip(("lat", "lon"), parse_wkt(r["wkt"]))),
                "isArch": bool(r["isArch"]),
                "isUnesco": bool(r["isUnesco"]),
                "wikidata": str(r["wikidata"]) if r["wikidata"] else "",
            }
            for r in g_sisal.query(Q_SISAL)
        ]
    )
    .dropna(subset=["lat", "lon"])
    .reset_index(drop=True)
)


def sisal_status(row):
    if row["isUnesco"]:
        return "UNESCO World Heritage"
    if row["isArch"]:
        return "Archaeological cave"
    return "Speleothem site"


df_sisal["status"] = df_sisal.apply(sisal_status, axis=1)
print(
    f"{len(df_sisal)} caves | "
    f"archaeological: {int(df_sisal['isArch'].sum())} | "
    f"UNESCO: {int(df_sisal['isUnesco'].sum())} | "
    f"Wikidata: {int((df_sisal['wikidata'] != '').sum())}"
)
df_sisal.head()


# %%
SISAL_COLORS = {
    "Speleothem site": "#3186cc",
    "Archaeological cave": "#e6550d",
    "UNESCO World Heritage": "#31a354",
}

m_sisal = folium.Map(
    location=[df_sisal["lat"].mean(), df_sisal["lon"].mean()],
    zoom_start=2,
    tiles="OpenStreetMap",
)
groups = {s: folium.FeatureGroup(name=s) for s in SISAL_COLORS}
for _, row in df_sisal.iterrows():
    popup = f"<b>{row['label']}</b><br>{row['status']}"
    if row["wikidata"]:
        popup += f'<br><a href="{row["wikidata"]}" target="_blank">Wikidata</a>'
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5,
        color=SISAL_COLORS[row["status"]],
        fill=True,
        fill_color=SISAL_COLORS[row["status"]],
        fill_opacity=0.8,
        popup=folium.Popup(popup, max_width=250),
    ).add_to(groups[row["status"]])
for g in groups.values():
    g.add_to(m_sisal)
folium.LayerControl(collapsed=True).add_to(m_sisal)

# static figure for the paper (always); interactive HTML in script mode
render_static_map(
    [
        {"lat": r.lat, "lon": r.lon, "cat": r.status, "color": SISAL_COLORS[r.status]}
        for r in df_sisal.itertuples()
    ],
    HERE / "map_sisal.png",
    title="SISAL cave sites",
    legend_order=list(SISAL_COLORS),
)
print("saved map_sisal.png")
if not _in_notebook():
    m_sisal.save(str(HERE / "map_sisal.html"))
    print("saved map_sisal.html")
m_sisal


# %% [markdown]
# ## C  Campanian Ignimbrite findspots (marker map)
#
# The 74 CI findspots with coordinates, archaeological flag and
# georeferencing certainty; `GROUP_CONCAT` keeps the six twice-typed
# findspots to one row each. The marker colour encodes the certainty of
# the georeferencing — the honest "how sure are we about this point?"
# layer that a fuzzy-linked findspot catalogue needs.

# %%
Q_CI = """
PREFIX geolod: <http://w3id.org/geo-lod/>
PREFIX geo:   <http://www.opengis.net/ont/geosparql#>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?f ?label ?wkt ?isArch
       (GROUP_CONCAT(DISTINCT ?st; SEPARATOR="|") AS ?stypes)
       (SAMPLE(?cl) AS ?certainty)
WHERE {
  ?f a geolod:CIFindspot ;
     rdfs:label ?label ;
     geo:hasGeometry / geo:asWKT ?wkt .
  OPTIONAL { ?f geolod:hasSpatialType ?st }
  OPTIONAL { ?f geolod:hasCertaintyLevel ?cl }
  BIND(EXISTS { ?f a geolod:CIArchaeologicalSite } AS ?isArch)
}
GROUP BY ?f ?label ?wkt ?isArch
ORDER BY ?f
"""

g_ci = load_graph("core", "ci")

df_ci = (
    pd.DataFrame(
        [
            {
                "findspot": localname(r["f"]),
                "label": str(r["label"]),
                **dict(zip(("lat", "lon"), parse_wkt(r["wkt"]))),
                "spatial": "|".join(
                    localname(s) for s in str(r["stypes"]).split("|") if s
                ),
                "isArch": bool(r["isArch"]),
                "certainty": localname(r["certainty"]) if r["certainty"] else "(none)",
            }
            for r in g_ci.query(Q_CI)
        ]
    )
    .dropna(subset=["lat", "lon"])
    .reset_index(drop=True)
)

print(f"{len(df_ci)} findspots | archaeological: {int(df_ci['isArch'].sum())}")
print("certainty:", dict(df_ci["certainty"].value_counts()))
df_ci.head()


# %%
CERTAINTY_COLORS = {
    "high": "#31a354",
    "medium": "#fd8d3c",
    "low": "#de2d26",
    "dubious": "#756bb1",
    "representative": "#3186cc",
}
DEFAULT_COLOR = "#999999"

m_ci = folium.Map(
    location=[df_ci["lat"].mean(), df_ci["lon"].mean()],
    zoom_start=4,
    tiles="OpenStreetMap",
)
present = [c for c in CERTAINTY_COLORS if c in set(df_ci["certainty"])]
present += [c for c in df_ci["certainty"].unique() if c not in CERTAINTY_COLORS]
groups = {c: folium.FeatureGroup(name=f"certainty: {c}") for c in present}
for _, row in df_ci.iterrows():
    color = CERTAINTY_COLORS.get(row["certainty"], DEFAULT_COLOR)
    popup = (
        f"<b>{row['label']}</b><br>type: {row['spatial'] or 'n/a'}"
        f"<br>certainty: {row['certainty']}"
        f"<br>{'archaeological' if row['isArch'] else 'findspot'}"
    )
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        popup=folium.Popup(popup, max_width=260),
    ).add_to(groups[row["certainty"]])
for g in groups.values():
    g.add_to(m_ci)
folium.LayerControl(collapsed=True).add_to(m_ci)

# static figure for the paper (always); interactive HTML in script mode
_pad = 3.0
_ci_extent = (
    df_ci["lon"].min() - _pad,
    df_ci["lon"].max() + _pad,
    df_ci["lat"].min() - _pad,
    df_ci["lat"].max() + _pad,
)
render_static_map(
    [
        {
            "lat": r.lat,
            "lon": r.lon,
            "cat": r.certainty,
            "color": CERTAINTY_COLORS.get(r.certainty, DEFAULT_COLOR),
        }
        for r in df_ci.itertuples()
    ],
    HERE / "map_ci.png",
    title="Campanian Ignimbrite findspots (by certainty)",
    extent=_ci_extent,
    legend_order=["high", "medium", "low", "dubious", "representative"],
)
print("saved map_ci.png")
if not _in_notebook():
    m_ci.save(str(HERE / "map_ci.html"))
    print("saved map_ci.html")
m_ci


# %% [markdown]
# ## Explore
#
# The three DataFrames (`df_samples`, `df_sisal`, `df_ci`) stay in scope.
# A few starting points — adapt freely:

# %%
# Archaeological SISAL caves that also carry a Wikidata link:
df_sisal[(df_sisal["isArch"]) & (df_sisal["wikidata"] != "")][
    ["site", "label", "wikidata"]
].head(10)

# %%
# CI findspots typed as caves (the natural overlap with the SISAL side):
df_ci[df_ci["spatial"].str.contains("Cave")][["findspot", "label", "certainty"]]

# %% [markdown]
# ---
#
# *Companion to the browser-executable `local_ttl-fair4rs-coin-live.qmd`.
# Data and RDF conversion: the
# [GeoScience-FAIRification-LOD](https://github.com/Research-Squirrel-Engineers/GeoScience-FAIRification-LOD)
# repository (Research Squirrel Engineers). Part of an Open Educational
# Resource series on knowledge graphs and linked open data, produced in
# the context of [NFDI4Objects](https://www.nfdi4objects.net/).*
