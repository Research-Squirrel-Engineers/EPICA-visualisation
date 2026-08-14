# GeoScience-FAIRification-LOD: Palaeoclimate Data Processing Pipeline

![Squilly Logo](img/logo.png)

A Python pipeline that turns palaeoclimate measurements into FAIR Linked Open Data. It covers three strands: five proxy records from the EPICA Dome C ice core (CH₄, δD, dust, δ¹⁸O of O₂ and δO₂/N₂), speleothem isotopes from the SISAL database, and Campanian Ignimbrite findspots. The graph is built on SOSA/SSN for the measurements, GeoSPARQL for the geometries, PROV-O for the provenance and CIDOC-CRM as the anchor that lets it enter the NFDI4Objects Knowledge Graph; the same run produces the figures, so a plot and a triple cannot disagree about what a source file said.

Two properties are treated as requirements rather than niceties. **Every age names the chronology it comes from.** The five EPICA records rest on four different depth-age models, and two of them additionally differ in phase — an ice age and a gas age at the same depth are not the same age. Six temporal reference systems under `http://w3id.org/geo-lod/trs/` keep them apart, which is what makes it visible that the beginning of MIS 5 sits at 1734 m on one scale and at 1782 m on another, in the same core. **Nothing is stated where nothing was measured.** A stage boundary is not carried into depth across a data gap, a filter is not run across one, and a figure marks the gap instead of drawing a line over it.

Every run is byte-reproducible: no output carries a timestamp, and each generated dataset states the fingerprint of the input data and the generator code it came from.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18814640.svg)](https://doi.org/10.5281/zenodo.18814640)

## 📁 Structure

```
project/
├── main.py                       ← MAIN SCRIPT (run everything)
├── pipeline_report.txt           ← Execution log
│
├── EPICA/                        ← EPICA Dome C (ice core), five proxies
│   ├── epica_data.py             ← one loader for figures and RDF, plus provenance
│   ├── epica_rdf.py              ← RDF generator
│   ├── epica_plates.py           ← multi-panel plates and the paper collages
│   ├── plot_epica_from_tab.py    ← single figures
│   ├── captions.yaml             ← figure captions (generated, editable)
│   ├── plots/                    ← JPG + SVG
│   │   ├── ch4_vs_age_ka_full.jpg      … 30 single figures
│   │   ├── plate_columns_smooth11.jpg  … 7 plates
│   │   └── plate_pipeline_outputs.jpg  … 2 collages
│   ├── rdf/
│   │   ├── epica_ontology.ttl
│   │   └── epica_dome_c.ttl
│   └── report/
│
├── SISAL/                        ← SISAL (speleothems)
│   ├── plot_sisal_from_csv.py
│   ├── plots/                    ← 24 plots × 2 formats
│   ├── rdf/
│   │   ├── sisal_ontology.ttl
│   │   ├── sisal_sites.ttl       ← all 365 caves, positions only
│   │   └── sisal_<id>_<cave>_data.ttl
│   └── report/
│
├── CI/                           ← Campanian Ignimbrite findspots
│   ├── ci_pipeline.py            ← findspots to RDF
│   ├── plot_ci_findspots.py      ← maps and certainty plate
│   ├── maps/                     ← findspot maps
│   ├── captions.yaml
│   ├── rdf/                      ← ci_findspots.ttl, ci_site_annotations.ttl
│   ├── plots/
│   └── report/
│
├── ontology/                     ← shared, on every sub-script's PYTHONPATH
│   ├── geo_lod_utils.py          ← core ontology, provenance, Mermaid
│   ├── geo_lod_figures.py        ← saving, axes, the data-gap convention
│   ├── geo_lod_basemap.py        ← coastlines, clipping, polar projection
│   ├── geo_lod_captions.py       ← caption files
│   ├── build_mis_vocab.py        ← MIS vocabulary and TRS generator
│   ├── geo_lod_core.ttl          ← base ontology (generated)
│   ├── crm_bridging.ttl          ← CIDOC-CRM bridging axioms
│   ├── trs.ttl                   ← temporal reference systems (generated)
│   ├── vocab/mis.ttl             ← Marine Isotope Stages (generated)
│   ├── shapes/                   ← SHACL: core_shapes.ttl, mis_shapes.ttl
│   └── *.mermaid                 ← taxonomy and instance diagrams
│
├── data/raw/                     ← primary sources, read-only
│   ├── epica/                    ← the five PANGAEA .tab files
│   └── mis/                      ← LR04 and Railsback tables
│
├── dist/                         ← generated
│   ├── geo-lod-bundle.ttl        ← all triples in one file
│   ├── mis_stages.csv            ← one row per MIS concept
│   └── mis_assignments.csv       ← one row per boundary assignment
│
├── PRIMER.md                     ← working plan and decision log
├── README.md
└── LICENSE
```

Anything under `data/raw/` is a source and is only ever read; anything a script
makes goes to `dist/`, `plots/` or `rdf/`. That is what makes it readable off
any file whether it is evidence or output.

## 🚀 Usage

### Run everything (recommended)

```bash
python main.py
```

This executes, in order:

1. Preparation — check the directory layout and the sub-scripts
2. Regenerate the canonical ontology from `geo_lod_utils.py`
3. Regenerate the controlled vocabularies and the temporal reference systems
4. EPICA Dome C — RDF for five proxy records
5. EPICA Dome C — 30 single figures, 7 plates, 2 collages, captions
6. SISAL — RDF and 24 figures for four caves, plus 305 site records
7. Campanian Ignimbrite — findspot RDF
8. Bundle everything into `dist/`, then validate: CIDOC-CRM coverage and SHACL

The RDF step runs before the figures on purpose: a fault in the data then
surfaces before half an hour of plotting.

**Duration:** ~3-4 minutes

### Bundle output formats

Turtle is the default: compact, readable, and byte-stable across runs. Other
formats are produced on demand:

```bash
python main.py --bundle-format nt            # fastest, for tight iteration
python main.py --bundle-format release       # nt, turtle, jsonld, xml
```

Serialising Turtle costs about 13 seconds more per run than N-Triples, which
is a fair price for a versioned bundle that always matches the code. Formats
not rewritten in a run are listed at the end, so a stale JSON-LD is noticed
before publication rather than after. `dist/geo-lod-bundle.nt` is git-ignored:
unlike Turtle it is not byte-stable, because rdflib assigns fresh blank-node
labels on every parse and N-Triples writes them out verbatim.

### Clean outputs before running

```bash
python main.py --clean
```

Removes all generated files (plots, RDF, Mermaid, reports, Python cache) before execution.

### EPICA only

```bash
python main.py --epica-only
```

### SISAL only

```bash
python main.py --sisal-only
```

## 📊 Output

### Figures (JPG + SVG)

**EPICA Dome C — 30 single figures.** Five records × two axes (depth, age) ×
three smoothing variants (`full`, `full_smooth11`, `full_savgol11p2`). Named
`{proxy}_vs_{depth|age_ka}_full{variant}`, with proxy one of `ch4`, `dd`,
`dust`, `d18o`, `do2n2`.

**EPICA Dome C — 7 plates**, the comparisons a single-proxy figure cannot make:

- `plate_columns_{variant}` — five records side by side on one vertical age axis
- `plate_rows_{variant}` — the same five stacked on a horizontal age axis
- `plate_boundary_depths` — the same MIS boundaries in the depth axis of each
  record, and each record's departure from their mean

**EPICA Dome C — 2 collages** for the paper: `plate_pipeline_outputs` (δ¹⁸O and
CH₄, each against age and depth) and `plate_pipeline_outputs_five` (all five
records). Both show the measured series in grey behind the smoothed one.

**SISAL — 24 plots** for four caves: Botuverá (144), Antro del Corchia (145),
Sanbao (140, δ¹⁸O only), Buraca Gloriosa (275). Format
`{site_id}_{cave}_{isotope}_age_{variant}.{jpg,svg}`.

Axis limits are derived from the data (`geo_lod_figures.nice_ticks`), never
from a fixed tick list, so a record cannot be drawn outside its own axis.
Per-record overrides live in `AXIS_OVERRIDES` in `plot_epica_from_tab.py`.

### Data gaps are drawn as gaps

Where consecutive measurements are more than 15 ka apart, the line is dashed
rather than solid and the last sample before and the first after the break are
ringed — the convention used in the `wdttest-*` family. In depth figures the
gap itself carries a neutral band labelled *no samples*, and a Marine Isotope
Stage with only one interpolatable boundary is drawn hatched, up to the edge of
the data. A stage lying entirely inside a gap gets no band at all: there is no
depth it could be assigned to.

A dashed segment states *no samples here* — not *no ice accumulated here*.

### Figure captions

`EPICA/captions.yaml` carries one entry per figure: `caption`, optionally
`captiondetail`, the licence and the source DOIs. Field names follow
`captions.yaml` in `wdttest-tables`.

The captions are generated from what was drawn — the chronology, the filter,
the number of measurements, which stages a record cannot cover — because those
are exactly the statements that go stale silently when a caption is maintained
by hand. They can still be edited: each entry keeps the last generated text
under `generated`, and once `caption` differs from it the caption is treated as
the author's and left alone, while `generated` is refreshed so the diff shows
where prose and data have drifted apart.

### RDF/Linked Open Data (TTL)

**Core Ontology:**
- `ontology/geo_lod_core.ttl` — Shared base classes (PalaeoclimateObservation, SamplingLocation, etc.)

**EPICA:**
- `EPICA/rdf/epica_ontology.ttl` — EPICA classes (IceCoreObservation, DrillingSite, Borehole, …)
- `EPICA/rdf/epica_dome_c.ttl` — 4,904 observations over five records, one site with two boreholes, 814 core sections, 5,587 time positions, 4,904 stage memberships, 77 stage boundaries carried into depth
- **187,554 triples**

**SISAL:**
- `SISAL/rdf/sisal_ontology.ttl` — SISAL-specific classes (SpeleothemObservation, Cave, etc.)
- `SISAL/rdf/sisal_sites.ttl` — All 365 SISAL v3 caves with WGS84 geometries and measurement counts (4,765 triples). Separate from the core graph so that a consumer after positions does not have to parse the measurements
- `SISAL/rdf/sisal_site_annotations.ttl` — geo-lod's archaeological, Wikidata and UNESCO reading of those caves, with its own source node
- `SISAL/rdf/sisal_144_botuvera_data.ttl` — 907 δ¹⁸O + 907 δ¹³C observations (21,795 triples)
- `SISAL/rdf/sisal_145_corchia_data.ttl` — 1,234 δ¹⁸O + 1,234 δ¹³C observations (29,651 triples)
- `SISAL/rdf/sisal_140_sanbao_data.ttl` — 5,832 δ¹⁸O observations (70,075 triples)
- `SISAL/rdf/sisal_275_buracagloriosa_data.ttl` — 1,137 δ¹⁸O + 1,137 δ¹³C observations (27,327 triples)
- `SISAL/rdf/sisal_all_data.ttl` — Combined file (**152,169 triples total**)


**Maps** live in `maps/` beside the `plots/` of each strand: `EPICA/maps/epica_dome_c_map` (Antarctica, polar stereographic), `SISAL/maps/sisal_sites_map*` (world and one per climate system), `CI/maps/ci_findspots_map`, `ci_findspots_campania` and `ci_findspots_certainty`, and `maps/geo_lod_sites_map` at the root, which is read from the RDF rather than from the input tables. Land polygons: Natural Earth 1:50m, public domain, in `data/raw/basemap/`.

Every figure is written twice: an SVG, which has no resolution, and a JPG whose quality follows `main.py --dpi` — `draft` (100 dpi) by default, `print` (300 dpi and at least 3000 px on the shorter side) for anything that goes to a printer. A release run (`--bundle-format release`) raises draft to print on its own.

**CI (Campanian Ignimbrite):**
- `CI/rdf/ci_findspots.ttl` — Findspot data with GeoSPARQL geometries and PROV-O provenance
- `CI/rdf/ci_site_annotations.ttl` — geo-lod's archaeological reading, kept apart from the source data

### Mermaid Diagrams (Ontology Visualisation)

All diagrams generated in `ontology/`:

- **`mermaid_taxonomy.mermaid`** — Complete class hierarchy (Core + EPICA + SISAL)
  - Includes external ontologies (SOSA, GeoSPARQL, DCAT, PROV)
  - LR (left-right) layout for readability
  
- **`mermaid_instance_epica.mermaid`** — EPICA named individuals
  - EPICA Dome C site, ice core sample, chronology
  - Green color scheme (#d4edda)
  
- **`mermaid_instance_sisal.mermaid`** — SISAL named individuals
  - 305 cave sites, FeatureCollections, archaeological cave sites
  - Yellow/brown color scheme (#fff3cd)

- **`mermaid_instance_ci.mermaid`** — CI named individuals
  - Campanian Ignimbrite volcanic event, findspots, archaeological sites
  - Terracotta color scheme (#fce8d5)

**Rendering to PNG:**
```bash
# Install Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Generate PNG images
mmdc -i ontology/mermaid_taxonomy.mermaid -o img/taxonomy.png
mmdc -i ontology/mermaid_instance_epica.mermaid -o img/instance_epica.png
mmdc -i ontology/mermaid_instance_sisal.mermaid -o img/instance_sisal.png
mmdc -i ontology/mermaid_instance_ci.mermaid -o img/instance_ci.png
```

## 🖼️ RDF Model Visualisations

### Ontology Taxonomy

![Ontology Class Hierarchy](img/taxonomy.png)

*Complete class hierarchy showing Core, EPICA, and SISAL classes with external ontology integration (SOSA, GeoSPARQL, DCAT, PROV)*

### EPICA Instance Model

![EPICA RDF Model](img/instance_epica.png)

*EPICA Dome C drilling site with ice core sample, observations, and chronology*

### SISAL Instance Model

![SISAL RDF Model](img/instance_sisal.png)

*SISAL cave sites (305 caves) organized in GeoSPARQL FeatureCollections*

## 🔍 SPARQL Queries

After export, you can load the TTL files into a triplestore (e.g., Apache Jena Fuseki, GraphDB) and query them:

### All Sites (EPICA + SISAL)

```sparql
PREFIX geolod: <http://w3id.org/geo-lod/>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?site ?label ?wkt
WHERE {
  ?collection rdfs:member ?site .
  ?site rdfs:label ?label ;
        geo:hasGeometry/geo:asWKT ?wkt .
}
```

Result: 380 sites — 305 SISAL caves, 74 Campanian Ignimbrite findspots and
the EPICA Dome C drilling site. The two EPICA boreholes are not members of a
collection; they hang below the site with `crm:P89_falls_within`.

### EPICA CH₄ observations, with the chronology their age belongs to

```sparql
PREFIX geolod: <http://w3id.org/geo-lod/>
PREFIX epica:  <http://w3id.org/geo-lod/epica/>
PREFIX time:   <http://www.w3.org/2006/time#>

SELECT ?obs ?age ?trs ?value ?smoothed
WHERE {
  ?obs geolod:measurementType epica:measurement_type_ch4 ;
       geolod:hasTimePosition ?tp ;
       geolod:ageChronology ?trs ;
       geolod:measuredValue ?value ;
       geolod:smoothedValue_rollingMedian ?smoothed .
  ?tp time:hasTRS ?trs ;
      time:numericPosition ?age .
}
ORDER BY ?age
```

Result: 736 observations. Binding `?trs` twice is what pins the age to the
leading chronology; drop the second binding and each observation returns twice,
once on EDC2 and once on the older EDC1 reading the source also publishes.

### The same stage boundary in five depth axes

```sparql
PREFIX geolod: <http://w3id.org/geo-lod/>
PREFIX crm:    <http://www.cidoc-crm.org/cidoc-crm/>
PREFIX mis:    <http://w3id.org/geo-lod/vocab/mis/>

SELECT ?trs ?depth
WHERE {
  ?a crm:P140_assigned_attribute_to mis:MIS_5 ;
     crm:P177_assigned_property_of_type geolod:MISBoundaryDepth ;
     crm:P141_assigned ?position ;
     geolod:inChronology ?trs .
  ?position geolod:atDepth_m ?depth .
}
ORDER BY ?depth
```

Result: 1733.94 m on AICC2023-ice, 1739.87 m on EDC3-ice, 1755.63 m on
EDC2-ice, 1759.64 m on AICC2023-gas, 1782.26 m on EDC2-gas. The three ice-age
scales cluster; the two gas-age scales sit systematically deeper.

### SISAL Sites with Sample Counts

```sparql
PREFIX geolod: <http://w3id.org/geo-lod/>

SELECT ?cave ?name ?d18o_count ?d13c_count
WHERE {
  ?cave a geolod:Cave ;
        rdfs:label ?name ;
        geolod:countD18OSamples ?d18o_count ;
        geolod:countD13CSamples ?d13c_count .
}
ORDER BY DESC(?d18o_count)
```

Result: 305 caves with sample counts

## 🛠️ Dependencies

```bash
pip install numpy pandas matplotlib scipy rdflib pyshacl pyyaml
```

**Optional (for Mermaid PNG rendering):**
```bash
npm install -g @mermaid-js/mermaid-cli
```

## 📝 Ontology Overview

### Class Hierarchy

```
geolod:PalaeoclimateObservation
  ├── geolod:IceCoreObservation (EPICA — five proxies)
  └── geolod:SpeleothemObservation (SISAL)
        ├── geolod:Delta18OSpeleothemObservation
        └── geolod:Delta13CSpeleothemObservation

geolod:SamplingLocation
  ├── geolod:DrillingSite (EPICA)
  ├── geolod:Borehole (EPICA — the hole inside the site)
  ├── geolod:Cave (SISAL)
  │     └── geolod:ArchaeologicalCaveSite
  └── geolod:CIFindspot (CI)
        └── geolod:CIArchaeologicalSite

geolod:PalaeoclimateSample
  ├── geolod:IceCore (EPICA)
  ├── geolod:SampleSection (a bounded interval, only where a source states one)
  └── geolod:Speleothem (SISAL)

geolod:Chronology  ⊑ time:TRS
  ├── geolod:IceCoreChronology (EPICA)
  └── geolod:UThChronology (SISAL)

crm:E13_Attribute_Assignment
  ├── geolod:MISAttributeAssignment      (assigns a property TO a stage)
  └── geolod:MISMembershipAssignment     (assigns a stage TO an observation)
```

A chronology is also a `time:TRS`, so one node serves both roles: an
observation names it with `geolod:ageChronology`, and the `time:TimePosition`
of that observation names the same node with `time:hasTRS`.

### Temporal reference systems

Under `http://w3id.org/geo-lod/trs/`, one per chronology **and phase**:

| TRS | Used by |
|---|---|
| `EDC1-gas` | CH₄, kept as the alternative reading alongside EDC2 |
| `EDC2-gas` | CH₄ |
| `EDC2-ice` | δD |
| `EDC3-ice` | dust |
| `AICC2023-gas` | δ¹⁸O of O₂ |
| `AICC2023-ice` | δO₂/N₂ |
| `LR04`, `Railsback2015` | the MIS vocabulary |

The gas/ice split is not bookkeeping. At a given depth the air trapped in the
bubbles is younger than the ice around it, so the same published boundary age
falls at a different depth depending on which branch a record uses. Read as one
scale, CH₄ and δD place the beginning of MIS 5 twenty-seven metres apart with
nothing in the graph to explain why.

### FeatureCollections (GeoSPARQL)

- `geolod:EPICA_DrillingSite_Collection` — 1 member
- `geolod:SISAL_Cave_Collection` — 305 members
- `geolod:SISAL_ArchaeologicalCave_Collection` — 37 members
- `geolod:AllPalaeoclimateSites_Collection` — 306 members
- `geolod:CIFindspotCollection` — CI findspots

## 🌐 W3ID URIs

All resources use persistent W3ID.org URIs:

Classes and properties stay flat under the base namespace; instance data lives
in one branch per strand.

| Path | Content |
|---|---|
| `http://w3id.org/geo-lod/` | core ontology |
| `http://w3id.org/geo-lod/epica/` | ice-core instance data |
| `http://w3id.org/geo-lod/sisal/` | speleothem instance data |
| `http://w3id.org/geo-lod/ci/` | Campanian Ignimbrite findspots |
| `http://w3id.org/geo-lod/vocab/mis/` | Marine Isotope Stage vocabulary |
| `http://w3id.org/geo-lod/trs/` | temporal reference systems |

Examples: `…/epica/site_dome_c`, `…/epica/obs_ch4_0001`, `…/epica/tp_ch4_0001`,
`…/trs/AICC2023-gas`, `…/vocab/mis/MIS_5e`.

## 📈 Statistics

### EPICA Dome C
- **1 drilling site**, two boreholes (75.10°S/123.35°E and 75.102°S/123.395°E — both published, neither discarded)
- **4,904 observations** across five records: 736 CH₄, 814 δD, 1,154 dust, 1,378 δ¹⁸O of O₂, 822 δO₂/N₂
- **Time span:** 0–805.8 ka BP · **depth range:** 8.5–3,191.1 m
- **187,554 RDF triples**

### SISAL
- **305 cave sites** worldwide (37 typed as `geolod:ArchaeologicalCaveSite`, 27 with Wikidata `owl:sameAs`, 7 UNESCO World Heritage)
- **9,110 observations** in four example caves
- **152,169 RDF triples**

### Bundle
- **354,901 triples**, 57 distinct classes, 121 distinct properties
- CIDOC-CRM coverage complete, 0 SHACL violations

## ♻️ Reproducibility

Two consecutive runs produce byte-identical output — 155 generated files at the
time of writing; only `pipeline_report.txt` differs, since it logs wall-clock
times. Four things make that work:

- **No clock in the output.** No generator reads the current time. Dates come
  from `GEO_LOD_RELEASE` in `geo_lod_utils.py`.
- **Content fingerprints instead of run timestamps.** Every generated dataset
  carries `owl:versionInfo` with a SHA-256 over its input data and generator
  script, and a `prov:Activity` naming each input with its own checksum. The
  fingerprint changes when the data or the model changes — and only then, so a
  dump can be checked against the state it claims to come from.
- **Deterministic figures.** `svg.hashsalt` fixes matplotlib's clip-path ids
  and the SVG metadata date is suppressed.
- **Line endings that survive a checkout.** SVGs are written through a
  binary file handle (`geo_lod_figures.save_figure`). Matplotlib otherwise
  opens the target in text mode, and Python then writes CRLF on Windows while
  `.gitattributes` stores LF — leaving every figure permanently different from
  its own committed form. The same rule applies to the log files.

## 🕰️ Marine Isotope Stage Vocabulary

`ontology/vocab/mis.ttl` holds 315 concepts (228 stages, 87 substages) with
792 boundary assignments, generated from the primary sources in
`data/raw/mis/`. Railsback et al. (2015) is the leading scheme wherever it
reaches; beyond its coverage of 1013.1 ka BP the LR04 boundaries of Lisiecki &
Raymo (2005) take over. The two sources disagree — LR04 puts the 5/6 boundary
at 130 ka, Railsback puts 5e/6a at 132.2 ka — and that disagreement is kept
rather than resolved: each reading is its own `crm:E13_Attribute_Assignment`
with its own `dct:source`, marked `geolod:LeadingAssignment` or
`geolod:AlternativeAssignment`. Filtering on the former yields one consistent
age axis without having to know where a source's coverage ends.

The same run writes `dist/mis_stages.csv` (one row per concept, leading values)
and `dist/mis_assignments.csv` (one row per assignment, both readings) for
figures and age-axis code, so plots and RDF cannot drift apart.

## 📖 Literature

**EPICA:**
- Lüthi et al. (2008): High-resolution carbon dioxide concentration record 650,000-800,000 years before present. *Nature* 453, 379-382. https://doi.org/10.1038/nature06949
- Loulergue et al. (2008): Orbital and millennial-scale features of atmospheric CH₄ over the past 800,000 years. *Nature* 453, 383-386. https://doi.org/10.1038/nature06950

**SISAL:**
- Kaushal et al. (2024): SISALv3: a global speleothem stable isotope and trace element database. *Earth System Science Data* 16, 1933-1963. https://doi.org/10.5194/essd-16-1933-2024

**EPICA chronologies:**
- Bouchet et al. (2023): The Antarctic Ice Core Chronology 2023 (AICC2023). *Climate of the Past* 19, 2257-2286. https://doi.org/10.5194/cp-19-2257-2023
- Parrenin et al. (2007): The EDC3 chronology for the EPICA Dome C ice core. *Climate of the Past* 3, 485-497. https://doi.org/10.5194/cp-3-485-2007

**MIS Boundaries:**
- Railsback et al. (2015): An optimized scheme of lettered marine isotope substages for the last 1.0 million years. *Quaternary Science Reviews* 111, 94-106. https://doi.org/10.1016/j.quascirev.2015.01.012
- Lisiecki & Raymo (2005): A Plio-Pleistocene stack of 57 globally distributed benthic δ¹⁸O records. *Paleoceanography* 20, PA1003. https://doi.org/10.1029/2004PA001071

## 🐛 Troubleshooting

### Import Error: `geo_lod_utils not found`

The scripts automatically set `PYTHONPATH` to include the `ontology/` directory. If you still get import errors:

1. **Check structure:**
   ```
   project/
   ├── main.py
   ├── EPICA/
   │   └── plot_epica_from_tab.py
   ├── SISAL/
   │   └── plot_sisal_from_csv.py
   └── ontology/
       └── geo_lod_utils.py  ← must be here!
   ```

2. **Run via main.py** (not individual scripts):
   ```bash
   python main.py
   ```

### No Mermaid diagrams generated

If `ontology/*.mermaid` files are missing:
- Check `pipeline_report.txt` for import errors
- Ensure `geo_lod_utils.py` is in `ontology/` directory
- Run with `--clean` flag: `python main.py --clean`

### No data found

Primary sources live under `data/raw/`, read-only:

```bash
ls data/raw/epica/*.tab data/raw/mis/*.csv
```

Required for EPICA: `EDC_CH4.tab`, `EPICA_Dome_C_dD.tab`,
`EPICA_Dome_C_dust.tab`, `EPICA_Dome_C_d18O.tab`, `EPICA_Dome_C_do2n2.tab`.
The SISAL CSV exports are still read from `SISAL/`; they move to `data/raw/`
with the SISAL rebuild.

### `ModuleNotFoundError: No module named 'yaml'`

The caption layer reads its own output back, to recognise hand-written
captions. Install PyYAML:

```bash
pip install pyyaml
```

### RDF export not working

→ Install rdflib:
```bash
pip install rdflib
```

## 🧭 Working document

`PRIMER.md` is the working plan: the steps S0 to S5, what each one changes, and
a decision log recording every choice with its date and its reason. Where this
README says *why* something is the way it is, the long version is there. It is
in German — an internal document that happens to lie open, not one addressed
outwards.

## 🤝 Authors

**Florian Thiery**
ORCID: https://orcid.org/0000-0002-3246-3531
Research Squirrel Engineers, Mainz — LEIZA, Leibniz-Zentrum für Archäologie

**Fiona Schenk**
Johannes Gutenberg-Universität Mainz — geoscientific input on the speleothem
and tephra side.

## 🤖 Use of AI

Large parts of the Python code in this repository, and parts of this README,
were written with the assistance of a large language model (Anthropic's Claude)
in an iterative session with the author, who set the task, took every design
decision and reviewed the result. The RDF, the ontology and the figures are
produced by that code; none of the measurements, and none of the statements
about the data, were generated by a model.

The point of saying so is not disclosure for its own sake. Code written this
way is fluent, and fluent code reads as correct whether or not it is — so this
repository leans on checks that do not care how something was written:

- The pipeline is validated on every run: CIDOC-CRM coverage over every typed
  instance, SHACL over the whole bundle, and a triple and class inventory in
  `pipeline_report.txt`.
- Two consecutive runs must produce byte-identical output. A file that always
  shows a diff trains everyone to skip its diff, and then a real change passes
  unnoticed.
- Every generated dataset carries a fingerprint over its input data and its
  generator script, so a dump can be checked against the state it claims to
  come from.
- Every decision is recorded in `PRIMER.md` with its reason and its date,
  including the ones later revised. Several of the corrections logged there —
  a filter running across a data gap, an axis clipping six per cent of a
  record, a chronology conflated with another — were bugs in exactly this kind
  of plausible-looking code, found by the checks rather than by reading.

Responsibility for the content, and for the errors that remain, lies with the
authors.

## 📄 Licence

Code: MIT (see `LICENSE`).

The data are not ours to license. Each PANGAEA record keeps its own terms —
three of the EPICA files are CC BY 3.0, two CC BY 4.0 — and the licence and DOI
of each are recorded per figure in `EPICA/captions.yaml` and per dataset in the
RDF. SISALv3 is cited via Kaushal et al. (2024). Anything reused from this
repository should carry those attributions on, not this one alone.
