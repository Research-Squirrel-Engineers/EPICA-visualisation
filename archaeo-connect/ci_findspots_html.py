__author__ = "Florian Thiery"
__copyright__ = "MIT Licence 2025, Florian Thiery"
__license__ = "MIT"
__version__ = "beta"
__maintainer__ = "Florian Thiery"
__email__ = "mail@fthiery.de"
__status__ = "beta"
__update__ = "2026-03-28"

# ==========================================================================
# ci_findspots_html.py
# Campanian Ignimbrite findspots → interactive HTML (Leaflet map + table)
#
# Input:  data/raw/ci/cifindspots_part_full.csv
#         data/curated/ci_site_annotations.csv
# Output: CI_findspots_CAA.html
#
# Archaeological site logic (mirrors ci_pipeline.py):
#   is_arch = spatialtype contains "ArchaeologicalSite"  OR  listed as
#            isArchaeologicalSite in data/curated/ci_site_annotations.csv
#
# Update instructions:
#   data/curated/ci_site_annotations.csv — add rows as curation progresses
#   CSV columns used:
#     id, label, wkt, spatialtype, certainty, certaintyinfo,
#     relatedto, relatedtohow, literature, source, sourcetype
#   Wikidata QID and OSM parsed from 'relatedto' (semicolon-separated URIs)
# ==========================================================================

import json
import re
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — one source, shared with CI/ci_pipeline.py
# ---------------------------------------------------------------------------
# There used to be a copy of the findspot table in this directory, and the
# copy was what this script read. It was byte-identical, which is exactly why
# it was dangerous: nothing would have shown that the two had drifted apart.
_HERE = Path(__file__).parent
ROOT = _HERE.parent
CSV_FILE = ROOT / "data" / "raw" / "ci" / "cifindspots_part_full.csv"
CURATED_CSV = ROOT / "data" / "curated" / "ci_site_annotations.csv"
OUT_FILE = _HERE / "CI_findspots_CAA.html"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_wkt(wkt: str):
    """Return (lat, lon) from POINT(lon lat) WKT string."""
    try:
        coords = wkt.replace("POINT(", "").replace(")", "").replace("I", "").strip().split()
        return float(coords[1]), float(coords[0])
    except Exception:
        return None, None


def parse_related(val: str):
    """Parse semicolon-separated URIs from relatedto column.
    Returns (qid, osm, wikipedia, geonames) — all str or None."""
    qid, osm, wikipedia, geonames = None, None, None, None
    for part in str(val).split(";"):
        part = part.strip()
        if "wikidata.org/entity/Q" in part:
            m = re.search(r"(Q\d+)", part)
            if m:
                qid = m.group(1)
        elif "openstreetmap.org" in part:
            m = re.search(r"openstreetmap\.org/(node|way|relation)/(\d+)", part)
            if m:
                osm = f"{m.group(1)}/{m.group(2)}"
        elif "wikipedia.org" in part:
            wikipedia = part
        elif "geonames.org" in part:
            geonames = part
    return qid, osm, wikipedia, geonames


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def curated_archaeological_ids(path: Path) -> set:
    """The ids geo-lod adds on top of what the source table states.

    Same file and same rule as CI/ci_pipeline.py. The map and the graph used
    to hold two hand-written sets that had drifted apart - twelve sites here,
    nine there - and neither file said which was right.
    """
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    flag = df["isArchaeologicalSite"].astype(str).str.strip().str.lower()
    return {int(v) for v in df.loc[flag == "true", "ci_id"]}


def load_data(csv_path: Path) -> list:
    df = pd.read_csv(csv_path, na_values=[".", "??", "NULL"])
    curated = curated_archaeological_ids(CURATED_CSV)

    sites = []
    for _, row in df.iterrows():
        sid = int(row["id"])
        spatial_types = str(row.get("spatialtype", "")).split(";")
        spatial_types = [s.strip() for s in spatial_types]

        is_arch = (
            sid in curated
            or "fsl:ArchaeologicalSite" in spatial_types
        )

        lat, lon = parse_wkt(str(row["wkt"]))
        qid, osm, wp, geonames = parse_related(str(row.get("relatedto", "")))

        sites.append({
            "id":           sid,
            "label":        str(row["label"]),
            "lat":          lat,
            "lon":          lon,
            "spatialtype":  str(row.get("spatialtype", "")),
            "certainty":    str(row.get("certainty", "fsl:medium")),
            "relatedtohow": str(row.get("relatedtohow", "")),
            "literature":   str(row.get("literature", "")),
            "qid":          qid,
            "osm":          osm,
            "wp":           wp,
            "is_arch":      is_arch,
            "curated":      sid in curated and "fsl:ArchaeologicalSite" not in spatial_types,
            "proposed":     False,   # no longer used — all arch are confirmed
        })

    n_arch  = sum(1 for s in sites if s["is_arch"])
    n_other = sum(1 for s in sites if not s["is_arch"])
    n_qid   = sum(1 for s in sites if s["qid"])
    n_osm   = sum(1 for s in sites if s["osm"])
    n_high  = sum(1 for s in sites if s["certainty"] == "fsl:high")

    print(f"  Sites loaded:        {len(sites)}")
    print(f"  Archaeological:      {n_arch}")
    print(f"  Other findspots:     {n_other}")
    print(f"  High certainty:      {n_high}")
    print(f"  Wikidata QID:        {n_qid}/{len(sites)}")
    print(f"  OSM present:         {n_osm}/{len(sites)}")
    return sites


# ---------------------------------------------------------------------------
# CSS — plain string, no f-string
# ---------------------------------------------------------------------------
CSS = """\
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; font-size: 14px; background: #f4f3ef; color: #1a1a18; }
header { padding: 20px 24px 14px; background: #1a1a18; color: #f4f3ef; }
header h1 { font-size: 18px; font-weight: 500; }
header p  { font-size: 12px; color: #888780; margin-top: 3px; }
#map { width: 100%; height: 400px; border-bottom: 2px solid #e8e6e0; }
.controls { display:flex; gap:8px; flex-wrap:wrap; align-items:center; padding:10px 24px; background:#faf9f6; border-bottom:1px solid #e8e6e0; }
.controls span { font-size:12px; color:#888780; }
.btn { padding:4px 12px; border-radius:16px; border:1px solid #b4b2a9; background:transparent; color:#5f5e5a; font-size:12px; cursor:pointer; transition:all .15s; }
.btn.on { background:#1a1a18; color:#f4f3ef; border-color:#1a1a18; }
.btn.f-arch.on   { background:#9a3412; border-color:#9a3412; color:#fff; }
.btn.f-other.on  { background:#3B8BD4; border-color:#3B8BD4; color:#fff; }
.btn.f-high.on   { background:#085041; border-color:#085041; color:#fff; }
.info-box { margin:14px 24px 0; background:#fff; border:1px solid #e8e6e0; border-radius:8px; padding:11px 16px; font-size:12px; color:#5f5e5a; display:flex; gap:20px; flex-wrap:wrap; }
.info-box strong { color:#1a1a18; font-size:11px; text-transform:uppercase; letter-spacing:.04em; display:block; margin-bottom:3px; }
.sections { padding: 16px 24px 32px; display: flex; flex-direction: column; gap: 22px; }
.section-block { background: #fff; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
.section-header { padding: 12px 16px; display: flex; align-items: center; gap: 12px; }
.section-header.arch-hdr  { background: #9a3412; color: #fff; }
.section-header.other-hdr { background: #e8f3fd; color: #0C447C; border-bottom: 1px solid #c5dcf0; }
.section-title { font-size: 15px; font-weight: 500; }
.section-sub   { font-size: 12px; opacity: .8; margin-top: 2px; }
.section-count { margin-left: auto; font-size: 12px; opacity: .75; white-space: nowrap; }
table { width:100%; border-collapse:collapse; }
th { padding:9px 12px; font-size:11px; font-weight:500; text-align:left; white-space:nowrap; cursor:pointer; user-select:none; border-bottom:1px solid #e8e6e0; color:#5f5e5a; background:#fafaf8; }
th:hover { background:#f0efeb; }
th.sorted-asc::after  { content:' \u2191'; }
th.sorted-desc::after { content:' \u2193'; }
.arch-hdr-row th { background:#fef5f2; color:#7c2d12; border-bottom:2px solid #fca89a; }
td { padding:9px 12px; border-bottom:1px solid #f1efe8; vertical-align:top; font-size:13px; line-height:1.5; }
tr:last-child td { border-bottom:none; }
tr:hover td { background:#f9f8f4; }
tr.high-cert td { background:#f0faf6; }
tr.high-cert:hover td { background:#e4f5ef; }
.site-name { font-weight:500; }
.ext-links { display:flex; gap:5px; margin-top:4px; flex-wrap:wrap; }
.ext-link  { display:inline-flex; align-items:center; font-size:11px; padding:1px 7px; border-radius:8px; text-decoration:none; border:1px solid; white-space:nowrap; }
.ext-link:hover { opacity:.75; }
.lnk-wd      { background:#f0f0ff; color:#3C3489; border-color:#c8c4f0; }
.lnk-osm     { background:#f0fff4; color:#085041; border-color:#9fd8b8; }
.lnk-wp      { background:#f5f5f5; color:#444; border-color:#ccc; }
.lnk-missing { opacity:.3; cursor:default; pointer-events:none; }
.badge { display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:500; white-space:nowrap; }
.b-cert-high { background:#E1F5EE; color:#085041; }
.b-cert-med  { background:#f4f3ef; color:#5f5e5a; border:1px solid #d3d1c7; }
.b-cert-low  { background:#FEF9EE; color:#633806; }
.b-cert-dub  { background:#fcebeb; color:#7c2d12; }
.b-cert-rep  { background:#EEEDFE; color:#3C3489; }
.lit  { font-size:12px; color:#5f5e5a; font-style:italic; }
.sptype { font-size:11px; color:#888; }
.popup-name  { font-weight:600; font-size:14px; margin-bottom:4px; }
.popup-row   { font-size:12px; line-height:1.7; color:#333; }
.popup-badge { display:inline-block; padding:1px 6px; border-radius:8px; font-size:11px; font-weight:500; margin-right:3px; }
.popup-links { margin-top:6px; display:flex; gap:6px; flex-wrap:wrap; }
.popup-link  { font-size:11px; text-decoration:none; padding:2px 7px; border-radius:6px; border:1px solid; }"""

# ---------------------------------------------------------------------------
# JS — plain string, SITES_JSON_PLACEHOLDER replaced via str.replace()
# No Python f-string here — JS curly braces are literal
# ---------------------------------------------------------------------------
JS = """\
const CERT_ORDER = {'fsl:high':0,'fsl:medium':1,'fsl:representative':2,'fsl:low':3,'fsl:dubious':4};
const CERT_LABEL = {'fsl:high':'high','fsl:medium':'medium','fsl:representative':'representative','fsl:low':'low','fsl:dubious':'dubious'};
const CERT_CLASS = {'fsl:high':'b-cert-high','fsl:medium':'b-cert-med','fsl:representative':'b-cert-rep','fsl:low':'b-cert-low','fsl:dubious':'b-cert-dub'};
const SPTYPE_LABEL = v => v.replace(/fsl:/g,'').replace(/;/g,' \u00b7 ');
const MATCH_LABEL  = v => ({'skos:closeMatch':'close match','fsl:spatialCloseMatch':'spatial close','fsl:partlyMatch':'partial','fsl:dubiousMatch':'dubious'}[v]||v);

const SITES = SITES_JSON_PLACEHOLDER;

// Map
const map = L.map('map').setView([43, 18], 4);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'\u00a9 OpenStreetMap contributors',maxZoom:18}).addTo(map);

// Fullscreen control (native, no plugin needed)
const FullscreenControl = L.Control.extend({
  options: {position:'topleft'},
  onAdd: function(){
    const btn = L.DomUtil.create('button','leaflet-bar leaflet-control');
    btn.innerHTML = '\u26f6';
    btn.title = 'Toggle fullscreen';
    btn.style.cssText = 'width:30px;height:30px;font-size:16px;cursor:pointer;background:#fff;border:none;display:flex;align-items:center;justify-content:center;';
    L.DomEvent.on(btn,'click',function(e){
      L.DomEvent.stopPropagation(e);
      const el = document.getElementById('map');
      if(!document.fullscreenElement){
        el.requestFullscreen().catch(()=>{});
        btn.innerHTML = '\u2715';
      } else {
        document.exitFullscreen();
        btn.innerHTML = '\u26f6';
      }
    });
    document.addEventListener('fullscreenchange',function(){
      if(!document.fullscreenElement) btn.innerHTML = '\u26f6';
    });
    return btn;
  }
});
new FullscreenControl().addTo(map);

// Legend control (bottom right, inside map)
const LegendControl = L.Control.extend({
  options: {position:'bottomright'},
  onAdd: function(){
    const div = L.DomUtil.create('div','leaflet-bar');
    div.style.cssText = 'background:#fff;padding:8px 12px;font-size:12px;line-height:1.9;color:#1a1a18;min-width:190px;pointer-events:none;';
    div.innerHTML =
      '<div style="font-weight:500;margin-bottom:4px;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#888">Legend</div>'+
      '<div style="display:flex;align-items:center;gap:7px"><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="#9a3412"/></svg><strong style="color:#9a3412">archaeological site</strong></div>'+
      '<div style="display:flex;align-items:center;gap:7px"><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="#3B8BD4"/></svg>other findspot</div>'+
      '<div style="display:flex;align-items:center;gap:7px;margin-top:3px"><svg width="16" height="12"><circle cx="4" cy="6" r="2.2" fill="#aaa"/><circle cx="12" cy="6" r="5" fill="#aaa"/></svg><span style="color:#888">size = certainty</span></div>'+
      '<div style="display:flex;align-items:center;gap:7px"><span style="opacity:.4;font-size:11px">filled</span> high \u00b7 <span style="opacity:.3;font-size:11px">faded</span> dubious</div>';
    return div;
  }
});
new LegendControl().addTo(map);

const markers = {};
function markerColor(s){ return s.is_arch ? '#9a3412' : '#3B8BD4'; }
function markerRadius(s){ const co = CERT_ORDER[s.certainty]??2; return 10 - co*1.8; }
function markerOpacity(s){ return s.certainty==='fsl:dubious' ? 0.35 : 0.85; }

SITES.forEach(s => {
  const m = L.circleMarker([s.lat,s.lon],{
    radius: markerRadius(s), fillColor: markerColor(s),
    color: '#fff', weight: 1.5, fillOpacity: markerOpacity(s)
  }).addTo(map);

  const wdL  = s.qid ? `<a href="https://www.wikidata.org/wiki/${s.qid}" target="_blank" class="popup-link lnk-wd">Wikidata ${s.qid}</a>` : '';
  const osmL = s.osm ? `<a href="https://www.openstreetmap.org/${s.osm}" target="_blank" class="popup-link lnk-osm">OSM ${s.osm}</a>` : '';
  const wpL  = s.wp  ? `<a href="${s.wp}" target="_blank" class="popup-link lnk-wp">Wikipedia</a>` : '';
  const tag  = s.is_arch
    ? `<span class="popup-badge" style="background:#9a3412;color:#fff">archaeological</span>`
    : '';

  m.bindPopup(
    `<div class="popup-name">${s.label} <span style="font-weight:400;color:#888;font-size:11px">(ID ${s.id})</span></div>`+
    `<div class="popup-row" style="margin-bottom:4px">${tag}<span class="popup-badge ${CERT_CLASS[s.certainty]}">${CERT_LABEL[s.certainty]}</span></div>`+
    `<div class="popup-row"><b>Spatial type:</b> ${SPTYPE_LABEL(s.spatialtype)}</div>`+
    `<div class="popup-row"><b>Match:</b> ${MATCH_LABEL(s.relatedtohow)}</div>`+
    `<div class="popup-row"><b>Literature:</b> <i>${s.literature}</i></div>`+
    `<div class="popup-links">${wdL}${osmL}${wpL}</div>`,
    {maxWidth:320}
  );
  markers[s.id] = m;
});

let activeFilter = 'all';
const sortState = {a:{key:'id',dir:1}, o:{key:'id',dir:1}};

function setFilter(f, btn){
  activeFilter = f;
  document.querySelectorAll('.btn').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  render();
}

function sortBy(key, tbl){
  const st = sortState[tbl];
  if(st.key===key) st.dir*=-1; else { st.key=key; st.dir=1; }
  render();
}

function applyFilter(sites){
  if(activeFilter==='high') return sites.filter(s=>s.certainty==='fsl:high');
  return sites;
}

function sortSites(sites, tbl){
  const {key,dir} = sortState[tbl];
  return [...sites].sort((a,b)=>{
    let av=a[key], bv=b[key];
    if(key==='certainty'){ av=CERT_ORDER[av]??9; bv=CERT_ORDER[bv]??9; }
    if(typeof av==='string'){ av=av.toLowerCase(); bv=bv.toLowerCase(); }
    return av<bv?-dir:av>bv?dir:0;
  });
}

function linksHtml(s){
  const wd  = s.qid ? `<a href="https://www.wikidata.org/wiki/${s.qid}" target="_blank" class="ext-link lnk-wd">WD ${s.qid}</a>` : `<span class="ext-link lnk-wd lnk-missing">WD \u2013</span>`;
  const osm = s.osm ? `<a href="https://www.openstreetmap.org/${s.osm}" target="_blank" class="ext-link lnk-osm">OSM \u2197</a>` : `<span class="ext-link lnk-osm lnk-missing">OSM \u2013</span>`;
  const wp  = s.wp  ? `<a href="${s.wp}" target="_blank" class="ext-link lnk-wp">WP \u2197</a>` : '';
  return `<div class="ext-links">${wd}${osm}${wp}</div>`;
}

function archRowHtml(s){
  return `<tr class="${s.certainty==='fsl:high'?'high-cert':''}">
    <td style="color:#888;font-size:12px">${s.id}</td>
    <td><div class="site-name">${s.label}</div>${linksHtml(s)}</td>
    <td><span class="sptype">${SPTYPE_LABEL(s.spatialtype)}</span></td>
    <td><span class="badge ${CERT_CLASS[s.certainty]}">${CERT_LABEL[s.certainty]}</span></td>
    <td style="font-size:12px;color:#5f5e5a">${MATCH_LABEL(s.relatedtohow)}</td>
    <td><span class="lit">${s.literature}</span></td>
  </tr>`;
}

function otherRowHtml(s){
  return `<tr class="${s.certainty==='fsl:high'?'high-cert':''}">
    <td style="color:#888;font-size:12px">${s.id}</td>
    <td><div class="site-name">${s.label}</div>${linksHtml(s)}</td>
    <td><span class="sptype">${SPTYPE_LABEL(s.spatialtype)}</span></td>
    <td><span class="badge ${CERT_CLASS[s.certainty]}">${CERT_LABEL[s.certainty]}</span></td>
    <td style="font-size:12px;color:#5f5e5a">${MATCH_LABEL(s.relatedtohow)}</td>
    <td><span class="lit">${s.literature}</span></td>
  </tr>`;
}

function render(){
  const archSites  = SITES.filter(s=>s.is_arch);
  const otherSites = SITES.filter(s=>!s.is_arch);

  const hideArch  = activeFilter==='other';
  const hideOther = activeFilter==='arch';

  const aRows = hideArch  ? [] : sortSites(applyFilter(archSites),  'a');
  const oRows = hideOther ? [] : sortSites(applyFilter(otherSites), 'o');

  document.getElementById('count-arch').textContent  = aRows.length+' sites';
  document.getElementById('count-other').textContent = oRows.length+' sites';

  document.getElementById('block-arch').style.opacity  = hideArch  ? '0.25' : '1';
  document.getElementById('block-other').style.opacity = hideOther ? '0.25' : '1';

  document.getElementById('tbody-arch').innerHTML  = aRows.map(archRowHtml).join('');
  document.getElementById('tbody-other').innerHTML = oRows.map(otherRowHtml).join('');

  const vis = new Set([...aRows,...oRows].map(s=>s.id));
  Object.values(markers).forEach(m=>m.setStyle({opacity:.1,fillOpacity:.08}));
  vis.forEach(id=>{
    const orig = markerOpacity(SITES.find(s=>s.id===id));
    markers[id]?.setStyle({opacity:1,fillOpacity:orig});
  });
}

render();"""


# ---------------------------------------------------------------------------
# HTML builder — concatenation only, JS injected last
# ---------------------------------------------------------------------------
def build_html(sites: list) -> str:
    n_arch  = sum(1 for s in sites if s["is_arch"])
    n_other = sum(1 for s in sites if not s["is_arch"])
    n_qid   = sum(1 for s in sites if s["qid"])
    n_osm   = sum(1 for s in sites if s["osm"])
    arch_ids_str = ", ".join(
        str(i) for i in sorted(s["id"] for s in sites if s["is_arch"]))
    n_curated = sum(1 for s in sites if s["curated"])

    sites_json = json.dumps(sites, ensure_ascii=False)
    js_final   = JS.replace("SITES_JSON_PLACEHOLDER", sites_json)

    head = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '<title>Campanian Ignimbrite Findspots \u2014 CAA 2026</title>\n'
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>\n'
        f'<style>\n{CSS}\n</style>\n'
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>\n'
        '</head>\n'
    )

    body = (
        '<body>\n'
        '<header>\n'
        '  <h1>Campanian Ignimbrite (CI) Findspots</h1>\n'
        f'  <p>{len(sites)} sites total \u00b7 source: cifindspots_part_full.csv + ci_pipeline.py \u00b7 CAA 2026 \u00b7 LEIZA / Research Squirrel Engineers</p>\n'
        '</header>\n'
        '<div id="map"></div>\n'
        '<div class="info-box">\n'
        '  <div>\n'
        '    <strong>Archaeological site logic (ci_pipeline.py)</strong>\n'
        '    <code style="font-size:11px;background:#f4f3ef;padding:2px 6px;border-radius:4px">'
        'is_arch = spatialtype contains \u201cArchaeologicalSite\u201d  OR  '
        'listed in data/curated/ci_site_annotations.csv</code>\n'
        f'    <div style="font-size:11px;margin-top:5px;color:#5f5e5a">The same two files decide it for the RDF graph, so this map and the graph cannot disagree. {n_curated} of the {n_arch} come from the curated annotations rather than from the source table.</div>\n'
        '  </div>\n'
        f'  <div><strong>Confirmed ({n_arch} sites)</strong>IDs: {arch_ids_str}</div>\n'
        f'  <div><strong>Linked Data coverage</strong>Wikidata QID: {n_qid}/{len(sites)} \u00b7 OSM: {n_osm}/{len(sites)}</div>\n'
        '</div>\n'
        '<div class="controls">\n'
        '  <span>Filter:</span>\n'
        f'  <button class="btn on"        onclick="setFilter(\'all\',this)">All ({len(sites)})</button>\n'
        f'  <button class="btn f-arch"    onclick="setFilter(\'arch\',this)">Archaeological ({n_arch})</button>\n'
        f'  <button class="btn f-other"   onclick="setFilter(\'other\',this)">Other findspots ({n_other})</button>\n'
        '  <button class="btn f-high"    onclick="setFilter(\'high\',this)">High certainty</button>\n'
        '</div>\n'
        '<div class="sections">\n'
        '  <div class="section-block" id="block-arch">\n'
        '    <div class="section-header arch-hdr">\n'
        '      <div>\n'
        '        <div class="section-title">Archaeological sites \u2014 confirmed (ci_pipeline.py)</div>\n'
        '        <div class="section-sub">spatialtype = fsl:ArchaeologicalSite or fsl:Cave;fsl:ArchaeologicalSite \u00b7 includes Castelcivita, Kozarnika, Temnata (pending pipeline update)</div>\n'
        '      </div>\n'
        '      <div class="section-count" id="count-arch"></div>\n'
        '    </div>\n'
        '    <table>\n'
        '      <thead><tr class="arch-hdr-row">\n'
        '        <th onclick="sortBy(\'id\',\'a\')">ID</th>\n'
        '        <th onclick="sortBy(\'label\',\'a\')">Site</th>\n'
        '        <th onclick="sortBy(\'spatialtype\',\'a\')">Spatial Type</th>\n'
        '        <th onclick="sortBy(\'certainty\',\'a\')">Certainty</th>\n'
        '        <th onclick="sortBy(\'relatedtohow\',\'a\')">Match Type</th>\n'
        '        <th onclick="sortBy(\'literature\',\'a\')">Literature</th>\n'
        '      </tr></thead>\n'
        '      <tbody id="tbody-arch"></tbody>\n'
        '    </table>\n'
        '  </div>\n'
        '  <div class="section-block" id="block-other">\n'
        '    <div class="section-header other-hdr">\n'
        '      <div>\n'
        '        <div class="section-title" style="color:#0C447C">All other findspots</div>\n'
        '        <div class="section-sub">Lakes, inhabited places, plateaux, maars, geological/sediment sites</div>\n'
        '      </div>\n'
        '      <div class="section-count" id="count-other"></div>\n'
        '    </div>\n'
        '    <table>\n'
        '      <thead><tr>\n'
        '        <th onclick="sortBy(\'id\',\'o\')">ID</th>\n'
        '        <th onclick="sortBy(\'label\',\'o\')">Site</th>\n'
        '        <th onclick="sortBy(\'spatialtype\',\'o\')">Spatial Type</th>\n'
        '        <th onclick="sortBy(\'certainty\',\'o\')">Certainty</th>\n'
        '        <th onclick="sortBy(\'relatedtohow\',\'o\')">Match Type</th>\n'
        '        <th onclick="sortBy(\'literature\',\'o\')">Literature</th>\n'
        '      </tr></thead>\n'
        '      <tbody id="tbody-other"></tbody>\n'
        '    </table>\n'
        '  </div>\n'
        '</div>\n'
        '<div style="height:32px"></div>\n'
    )

    return head + body + '<script>\n' + js_final + '\n</script>\n</body>\n</html>'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("CI Findspots --> HTML")
    print("=" * 60)
    print(f"\n-> Reading {CSV_FILE}")
    sites = load_data(CSV_FILE)

    print("\n-> Building HTML ...")
    html = build_html(sites)

    # newline="\n": matplotlib is not the only thing on Windows that turns LF
    # into CRLF on the way out. .gitattributes stores LF, so a page written in
    # text mode differs from its own committed form on every checkout.
    with open(OUT_FILE, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html)
    print(f"  OK  Written: {OUT_FILE}  ({len(html):,} chars)")
    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)
    print("\nTo mark a findspot archaeological:  data/curated/"
          "ci_site_annotations.csv, or spatialtype in the source table")
    print("To update Wikidata/OSM:             relatedto column in the "
          "source table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
