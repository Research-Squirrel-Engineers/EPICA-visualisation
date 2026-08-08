# geo-lod Ontology — CIDOC-CRM Crosswalk

This document describes how the `geo-lod` ontology
(`http://w3id.org/geo-lod/`) anchors all of its instance-level data into
the **CIDOC-CRM family** of ontologies, as required for ingestion into the
**NFDI4Objects Knowledge Graph (N4O KG)**.

> **Hard rule for N4O:** every typed instance in the published RDF must be
> reachable, via the transitive closure of `rdfs:subClassOf`, from a class
> in the CIDOC-CRM family namespaces:
>
> - `crm:` — `http://www.cidoc-crm.org/cidoc-crm/`
> - `crmsci:` — `http://www.ics.forth.gr/isl/CRMsci/`
> - `crmgeo:` — `http://www.ics.forth.gr/isl/CRMgeo/`
> - `crmarchaeo:` — `http://www.ics.forth.gr/isl/CRMarchaeo/`
>
> External standard vocabularies (SOSA, PROV-O, GeoSPARQL, DCAT, FOAF,
> Pleiades, fuzzy-sl, …) do **not** count as anchors on their own. They
> are bridged into CRM via local `rdfs:subClassOf` axioms inside the
> geo-lod ontology files (see *Bridging external vocabularies* below).
>
> The CRM-coverage check in `bundle_rdf.py` enforces this rule on every
> pipeline run.

---

## 1. Anchoring strategy

The geo-lod ontology uses a **shallow bridging layer**: rather than
re-modelling everything in CRM, each external class that we instantiate
gets exactly one `rdfs:subClassOf` axiom pointing it into the CRM family.
This keeps the original semantics of the external vocabulary intact while
making the data CRM-discoverable.

There are two flavours:

1. **Direct CRM subclassing** for our own classes (`geo-lod:*`):
   declared in `ontology/geo_lod_core.ttl` and the per-domain extensions
   (`epica_ontology.ttl`, `sisal_ontology.ttl`, CI ontology).
2. **Bridging axioms** for instantiated external classes
   (`sosa:Observation`, `prov:Activity`, `geo:Feature`, …): declared in a
   dedicated file `ontology/crm_bridging.ttl` (TODO — see below) so they
   stay separate from the original external ontologies.

---

## 2. Crosswalk table

The mapping below is the canonical reference. Each row is one rule that
must hold in the merged bundle. The "Instances" column gives the rough
count from the most recent pipeline run as orientation.

### 2.1 Own classes (`geo-lod:`) → CRM

| `geo-lod:` class                        | `rdfs:subClassOf` (CRM family)             | Rationale |
|---|---|---|
| `SamplingLocation`                      | `crm:E27_Site`                              | Geographic place where samples are taken |
| `DrillingSite` ⊑ `SamplingLocation`     | (via parent) `crm:E27_Site`                 | Ice-core drilling location |
| `Cave` ⊑ `SamplingLocation`             | (via parent) `crm:E27_Site`                 | Speleothem cave site |
| `ArchaeologicalCaveSite` ⊑ `Cave`       | (via parent) `crm:E27_Site`                 | SISAL site with archaeological context |
| `CIArchaeologicalSite`                  | `crm:E27_Site`                              | Findspot with archaeological context |
| `CIFindspot`                            | `crm:E27_Site`                              | Documented findspot of CI tephra |
| `PalaeoclimateSample`                   | `crm:E20_Biological_Object` *or* `crm:E18_Physical_Thing` | Physical specimen — pick one and document |
| `IceCore` ⊑ `PalaeoclimateSample`       | (via parent)                                | Ice-core specimen |
| `Speleothem` ⊑ `PalaeoclimateSample`    | (via parent)                                | Stalagmite/-tite specimen |
| `Delta18OSpeleothemObservation`         | `crmsci:S4_Observation`                     | δ¹⁸O measurement event |
| `Delta13CSpeleothemObservation`         | `crmsci:S4_Observation`                     | δ¹³C measurement event |
| `CH4ConcentrationProperty`              | `crmsci:S9_Property_Type`                   | Property type (CH₄ ppbv) |
| `Delta18OProperty`                      | `crmsci:S9_Property_Type`                   | Property type (δ¹⁸O ‰) |
| `Delta13CProperty`                      | `crmsci:S9_Property_Type`                   | Property type (δ¹³C ‰) |
| `MeasurementType`                       | `crmsci:S9_Property_Type`                   | Generic property type |
| `IceCoreChronology`                     | `crmsci:S4_Observation` *or* `crm:E13_Attribute_Assignment` | Age-depth assignment |
| `UThChronology`                         | `crmsci:S4_Observation`                     | U/Th dating event |
| `RollingMedianFilter`                   | `crmsci:S6_Data_Evaluation`                 | Smoothing as derived data |
| `SavitzkyGolayFilter`                   | `crmsci:S6_Data_Evaluation`                 | Smoothing as derived data |
| `DataSource`                            | `crm:E73_Information_Object`                | Citable source (TAB/CSV file, paper) |
| `MarineIsotopeStage`                    | `crm:E4_Period`                             | Chronostratigraphic subdivision, also a `skos:Concept` |
| `MarineIsotopeSubstage` ⊑ `MarineIsotopeStage` | (via parent) `crm:E4_Period`         | Lettered substage after Railsback et al. (2015) |
| `MISAttributeAssignment`                | `crm:E13_Attribute_Assignment`              | Boundary, peak or climate mode on the authority of one source |
| `ClimateMode`                           | `crm:E55_Type`                              | Warm / cold classification |
| `AssignedPropertyType`                  | `crm:E55_Type`                              | Value of `crm:P177` on an attribute assignment |
| `AssignmentStatus`                      | `crm:E55_Type`                              | Leading vs. alternative reading of a disputed value |

### 2.2 External vocabularies → CRM (bridging axioms)

These are written into `ontology/crm_bridging.ttl`. Each external class
that we actually instantiate in our data needs exactly one bridging
axiom.

| External class                         | `rdfs:subClassOf` (CRM family)             | Notes |
|---|---|---|
| `sosa:Observation`                      | `crmsci:S4_Observation`                     | Aligned by intent |
| `sosa:Sample`                           | `crm:E20_Biological_Object` *or* `crm:E18_Physical_Thing` | Same target as `PalaeoclimateSample` |
| `sosa:ObservableProperty`               | `crmsci:S9_Property_Type`                   |  |
| `prov:Entity`                           | `crm:E70_Thing`                             | Most general physical/conceptual thing |
| `prov:Activity`                         | `crm:E7_Activity`                           | Direct mapping |
| `prov:Agent`                            | `crm:E39_Actor`                             |  |
| `foaf:Person`                           | `crm:E21_Person`                            |  |
| `geo:Feature`                           | `crm:E1_CRM_Entity`                         | Most general — refine when feature type is known |
| `geo:FeatureCollection`                 | `crm:E78_Curated_Holding`                   | Curated set of features |
| `sf:Point`                              | `crmgeo:SP6_Declarative_Place`              | Geometry as place declaration |
| `dcat:Dataset`                          | `crm:E73_Information_Object`                |  |
| `dcat:Catalog`                          | `crm:E78_Curated_Holding`                   |  |
| `dcterms:BibliographicResource`         | `crm:E31_Document`                          |  |
| `skos:Concept`                          | `crm:E55_Type`                              | Vocabulary term |
| `skos:ConceptScheme`                    | `crm:E32_Authority_Document`                | Published vocabulary |
| `skos:Collection`                        | `crm:E78_Curated_Holding`                   | Curated subset of a vocabulary |
| `time:TimePosition`                     | `crm:E54_Dimension`                         | Value with unit and reference system |
| `time:TRS`                              | `crm:E29_Design_or_Procedure`               | Mirrors `crmgeo:SP4` for spatial CRS |
| `pleiades:Place`                        | `crm:E53_Place`                             |  |
| `fuzzy-sl:Site`                         | `crm:E27_Site`                              | Squirrel.link Site model |
| `fuzzy-sl:Georeferencing`               | `crm:E13_Attribute_Assignment`              | Statement of position |
| `crmarchaeo:A2_Stratigraphic_Volume_Unit` | (already in CRM family)                   | No bridging needed |
| `crmgeo:SP6_Declarative_Place`          | (already in CRM family)                     | No bridging needed |
| `crmsci:S4_Observation` etc.            | (already in CRM family)                     | No bridging needed |

> **Choices that need final review** are marked with *or* — where two CRM
> targets are defensible. Once chosen, the alternative should be removed
> here and the decision recorded in the commit message.

---

## 3. Bridging file structure

Recommended layout once `crm_bridging.ttl` exists:

```turtle
@prefix crm:        <http://www.cidoc-crm.org/cidoc-crm/> .
@prefix crmsci:     <http://www.ics.forth.gr/isl/CRMsci/> .
@prefix crmgeo:     <http://www.ics.forth.gr/isl/CRMgeo/> .
@prefix crmarchaeo: <http://www.ics.forth.gr/isl/CRMarchaeo/> .
@prefix sosa:       <http://www.w3.org/ns/sosa/> .
@prefix prov:       <http://www.w3.org/ns/prov#> .
@prefix geo:        <http://www.opengis.net/ont/geosparql#> .
@prefix sf:         <http://www.opengis.net/ont/sf#> .
@prefix dcat:       <http://www.w3.org/ns/dcat#> .
@prefix dcterms:    <http://purl.org/dc/terms/> .
@prefix foaf:       <http://xmlns.com/foaf/0.1/> .
@prefix pleiades:   <https://pleiades.stoa.org/places/vocab#> .
@prefix fuzzysl:    <http://fuzzy-sl.squirrel.link/ontology/> .
@prefix rdfs:       <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:        <http://www.w3.org/2002/07/owl#> .

<http://w3id.org/geo-lod/crm-bridging> a owl:Ontology ;
  rdfs:label "geo-lod ↔ CIDOC-CRM bridging axioms" ;
  rdfs:comment """Bridges external standard vocabularies into the CIDOC-CRM
family so that every instance in the geo-lod knowledge graph has a CRM
ancestor. Required for ingestion into the NFDI4Objects Knowledge Graph.""" .

# --- SOSA ------------------------------------------------------------------
sosa:Observation        rdfs:subClassOf crmsci:S4_Observation .
sosa:Sample             rdfs:subClassOf crm:E20_Biological_Object .
sosa:ObservableProperty rdfs:subClassOf crmsci:S9_Property_Type .

# --- PROV-O ----------------------------------------------------------------
prov:Entity   rdfs:subClassOf crm:E70_Thing .
prov:Activity rdfs:subClassOf crm:E7_Activity .
prov:Agent    rdfs:subClassOf crm:E39_Actor .

# --- FOAF ------------------------------------------------------------------
foaf:Person rdfs:subClassOf crm:E21_Person .

# --- GeoSPARQL / Simple Features ------------------------------------------
geo:Feature           rdfs:subClassOf crm:E1_CRM_Entity .
geo:FeatureCollection rdfs:subClassOf crm:E78_Curated_Holding .
sf:Point              rdfs:subClassOf crmgeo:SP6_Declarative_Place .

# --- DCAT / Dublin Core ----------------------------------------------------
dcat:Dataset                  rdfs:subClassOf crm:E73_Information_Object .
dcat:Catalog                  rdfs:subClassOf crm:E78_Curated_Holding .
dcterms:BibliographicResource rdfs:subClassOf crm:E31_Document .

# --- Pleiades / fuzzy-sl ---------------------------------------------------
pleiades:Place         rdfs:subClassOf crm:E53_Place .
fuzzysl:Site           rdfs:subClassOf crm:E27_Site .
fuzzysl:Georeferencing rdfs:subClassOf crm:E13_Attribute_Assignment .
```

---

## 4. Verifying coverage

The pipeline runs `bundle_rdf.py` as step 5. Its CRM-coverage check
enumerates every instance class in the merged bundle and reports any that
do not transitively subclass a CRM-family class:

```text
▶ CIDOC-CRM Coverage-Check (strict CRM-family) ...
  Instanz-Klassen geprüft: 45
  ✓ mit CRM-Anker:         45
  ✗ ohne CRM-Anker:         0
```

A green check here is the gate for N4O readiness.

---

## 5. Open questions / decisions to make

1. **Sample target class.** Choose between `crm:E20_Biological_Object`
   (organic origin) and `crm:E18_Physical_Thing` (physical specimen) for
   `PalaeoclimateSample`, `sosa:Sample`. The choice propagates to
   `IceCore` and `Speleothem`. *Note: ice cores are not biological;
   speleothems contain organic inclusions but are mineral. `E18` is
   probably the safer general choice.*
2. **`IceCoreChronology` target.** `crmsci:S4_Observation` (treating an
   age-depth assignment as observation) vs. `crm:E13_Attribute_Assignment`
   (treating it as an attribution event). The latter is closer to the
   epistemics — chronologies are scholarly attributions, not direct
   observations.
3. **Smoothing as `crmsci:S6_Data_Evaluation`.** Confirm that derived
   smoothed series are best modelled as data evaluations (CRMsci) rather
   than fresh observations.
4. **Refinement of `geo:Feature`.** Currently bridged to the very general
   `crm:E1_CRM_Entity`. If all our `geo:Feature` instances are in fact
   places, the bridge could be tightened to `crm:E53_Place`.

---

## 6. Changelog

- *unreleased* — initial crosswalk drafted alongside `bundle_rdf.py`
  step-5 CRM-coverage gate. Bridging file `crm_bridging.ttl` to be
  written next.
- *S1* — MIS vocabulary added under
  `http://w3id.org/geo-lod/vocab/mis/`. New own classes for stages,
  substages and their attribute assignments; SKOS and OWL-Time bridged
  into the CRM family so that concepts, time positions and temporal
  reference systems pass the coverage gate.
