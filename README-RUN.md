# README-RUN.md — was man womit anstösst

Jeder Einstiegspunkt des Repositoriums, seine Schalter, und was danach auf der
Platte liegt. Gedacht zum Nachschlagen, nicht zum Durchlesen.

Alle Befehle stehen auf **einer** Zeile und laufen aus dem Wurzelverzeichnis,
sofern nicht anders angegeben. Windows-Schreibweise mit `py`; unter Linux und
macOS `python3` und Schrägstriche.

Diese Datei wird bei jeder Änderung an den Skripten mitgeführt. Wenn ein
Schalter hier fehlt, ist das ein Fehler und kein Geheimnis.

---

## Der kurze Weg

| Ich will … | Befehl |
|---|---|
| alles neu bauen, schnell | `py main.py` |
| alles neu bauen, vollständig | `py main.py --sisal-sites all` |
| eine Release-Fassung | `py main.py --bundle-format release` |
| nur am CI-Strang arbeiten | `py main.py --ci-only --archaeo-only --no-bundle` |
| Abbildungen für den Druck | `py main.py --sisal-sites all --dpi print` |
| sehen, was erzeugt wurde | `py clean.py` |

---

## `main.py` — die Pipeline

Führt die Einzelskripte in fester Reihenfolge aus, protokolliert alles nach
`pipeline_report.txt` und misst jeden Schritt. Die Schrittnummern kommen aus
einem Zähler und folgen dem Lauf: ein voller Lauf zählt bis 14, ein
`--ci-only` bis 8.

```cmd
py main.py
```

### Welche Stränge

| Schalter | Wirkung |
|---|---|
| *(keiner)* | alle Stränge |
| `--epica-only` | nur EPICA |
| `--sisal-only` | nur SISAL |
| `--ci-only` | nur der CI-Strang |
| `--archaeo-only` | nur die archäologischen HTML-Seiten |

Die Schalter sind **kombinierbar**. `--ci-only --archaeo-only` läuft die
beiden zusammengehörigen Schritte, ohne EPICA und SISAL anzufassen — der
schnellste sinnvolle Lauf, wenn man am CI-Strang arbeitet.

### Umfang und Format

| Schalter | Voreinstellung | Wirkung |
|---|---|---|
| `--sisal-sites dev` | `dev` | fünf Sites, so gewählt, dass jede Prüfung des Generators noch feuert |
| `--sisal-sites all` | | alle zwölf Sites des Ausschnitts |
| `--sisal-sites spannagel` | | eine oder mehrere Sites, nach Name oder `site_id`, kommagetrennt |
| `--bundle-format nt` | `nt` | Bundle nur als N-Triples, schnell |
| `--bundle-format release` | | N-Triples, Turtle, JSON-LD und RDF/XML |
| `--no-bundle` | | den Bundle-Schritt ganz überspringen |
| `--no-overview` | | die Übersichtskarte überspringen |
| `--no-clean` | | Schritt 0 überspringen und in vorhandene Ausgaben schreiben |
| `--clean` | an | Schritt 0 ausführen. Ist ohnehin die Voreinstellung; der Schalter bleibt, damit ältere Aufrufe weiter gelten |

Ein **Release-Lauf** ist alles, was `--bundle-format release` setzt. Er zieht
drei Dinge nach: `--sisal-sites all`, Turtle statt N-Triples für die grossen
SISAL-Graphen, und Druckauflösung für alle Rastergrafiken. Jedes davon wird
gemeldet, wenn es eine Voreinstellung überschreibt.

### Auflösung der Rastergrafiken

| Schalter | Wirkung |
|---|---|
| *(keiner)* | `draft` — 100 dpi, keine Pixelgrenze, kleine Dateien |
| `--dpi print` | 300 dpi **und** mindestens 3 000 px auf der kürzeren Seite |
| `--dpi 150` | genau 150 dpi. Ab 300 kommt die Pixelgrenze mit, darunter nicht |

Gerechnet wird auf das fertig beschnittene Bild, nicht auf die Figurgrösse —
deshalb steht in der Ausgabe selten glatt 300 dpi. Bei breiten Abbildungen
gewinnt die Pixelgrenze: die SISAL-Weltkarte landet bei 495 dpi.

Das SVG neben jedem JPG hat keine Auflösung und ist von alldem unberührt.

Der Wert erreicht die Zeichenskripte über die Umgebungsvariable
`GEO_LOD_RASTER_DPI`, weil jedes von ihnen ein eigener Prozess ist. Wer ein
Skript einzeln startet, setzt sie selbst:

```cmd
set GEO_LOD_RASTER_DPI=print & py SISAL\plot_sisal_maps.py
```

### Was ein voller Lauf schreibt

```
ontology/geo_lod_core.ttl          die kanonische Ontologie
ontology/vocab/mis.ttl             MIS-Vokabular
ontology/trs.ttl                   zeitliche Bezugssysteme
dist/mis_stages.csv                MIS-Stufen als Tabelle
dist/mis_assignments.csv           MIS-Zuordnungen als Tabelle
EPICA/rdf/                         epica_dome_c.ttl, epica_ontology.ttl
EPICA/plots/                       78 Abbildungen der fünf Proxy-Aufzeichnungen
EPICA/maps/                        epica_dome_c_map
EPICA/captions.yaml                eine Bildunterschrift je Abbildung
SISAL/rdf/                         sisal_sites.ttl, sisal_site_annotations.ttl,
                                   sisal_ontology.ttl, sisal_v3_core,
                                   sisal_v3_chronologies, sisal_v3_dating.ttl
SISAL/plots/                       Profile je Speläothem und die Cluster-Tafeln
SISAL/maps/                        Weltkarte und drei Cluster-Karten
SISAL/captions.yaml                73 Einträge
CI/rdf/                            ci_findspots.ttl, ci_site_annotations.ttl
CI/maps/                           drei Karten
CI/captions.yaml                   drei Einträge
archaeo-connect/CI_findspots_CAA.html
maps/geo_lod_sites_map             die Übersichtskarte, aus dem Graphen gelesen
maps/captions.yaml
dist/geo-lod-bundle.*              das validierte Bundle
pipeline_report.txt                das Protokoll des Laufs
```

Laufzeiten zur Orientierung, gemessen an einem vollen Lauf mit zwölf Sites:
rund 15 Minuten, davon zwei Drittel im Bundle-Schritt mit der
SHACL-Validierung. Ein `--ci-only --archaeo-only --no-bundle` ist in wenigen
Sekunden durch.

---

## Die Einzelskripte

Alle laufen auch für sich, mit `PYTHONPATH` auf `ontology`. `main.py` setzt
das selbst; von Hand geht es so:

```cmd
set PYTHONPATH=C:\git\GeoScience-FAIRification-LOD\ontology & py CI\ci_pipeline.py
```

### EPICA

| Skript | Schalter | Ergebnis |
|---|---|---|
| `EPICA/epica_rdf.py` | keine | `EPICA/rdf/epica_dome_c.ttl`, `epica_ontology.ttl` |
| `EPICA/plot_epica_from_tab.py` | keine | 78 Abbildungen nach `EPICA/plots/` |
| `EPICA/epica_plates.py` | keine | die Tafeln, dieselben Verzeichnisse |
| `EPICA/plot_epica_map.py` | keine | `EPICA/maps/epica_dome_c_map` |

### SISAL

| Skript | Schalter | Ergebnis |
|---|---|---|
| `SISAL/sisal_import.py` | `--from PFAD`, `--verify` | holt den Ausschnitt aus `squirrels-sisal-db-v3` nach `data/derived/sisal/`; `--verify` prüft nur gegen das Manifest |
| `SISAL/sisal_rdf.py` | `--format nt\|turtle`, `--sites all\|dev\|NAME` | die sechs Dateien in `SISAL/rdf/` |
| `SISAL/plot_sisal_from_csv.py` | keine | Profile je Speläothem nach `SISAL/plots/` |
| `SISAL/plot_sisal_maps.py` | keine | vier Karten nach `SISAL/maps/` |

`--sites` nimmt `all`, `dev` (fünf Sites) oder eine kommagetrennte Liste aus
Namen und `site_id`. Namen werden nach Präfix und ohne Rücksicht auf Gross-
und Kleinschreibung gesucht: `--sites spannagel` findet „Spannagel cave".

### CI und Archäologie

| Skript | Schalter | Ergebnis |
|---|---|---|
| `CI/ci_pipeline.py` | keine | `CI/rdf/ci_findspots.ttl`, `ci_site_annotations.ttl` |
| `CI/plot_ci_findspots.py` | keine | drei Karten nach `CI/maps/` |
| `archaeo-connect/ci_findspots_html.py` | keine | `archaeo-connect/CI_findspots_CAA.html` |

### Strangübergreifend

| Skript | Schalter | Ergebnis |
|---|---|---|
| `plot_overview_map.py` | keine | `maps/geo_lod_sites_map`, gelesen aus den TTL |
| `ontology/build_mis_vocab.py` | keine | `ontology/vocab/mis.ttl`, `trs.ttl`, die beiden CSV in `dist/` |
| `ontology/geo_lod_utils.py` | keine | Selbsttest der geteilten Bausteine |

`plot_overview_map.py` liest die frisch geschriebenen Turtle-Dateien, nicht die
Eingabetabellen. Ein Strang, dessen TTL fehlt, wird gemeldet und in der
Bildunterschrift genannt; ein geplanter Strang wie ELSA schweigt, bis es ihn
gibt.

---

## `bundle_rdf.py` — Bundle und Validierung

```cmd
py bundle_rdf.py
```

| Schalter | Wirkung |
|---|---|
| `--format nt,turtle,jsonld,xml` | Ausgabeformate, kommagetrennt |
| `--sites spannagel` | statt des Gesamtbundles ein Site-Bundle |

Verfügbare Formate: `nt`, `turtle`, `longturtle`, `xml`, `jsonld`.
Voreinstellung ist `nt` — das Bundle mit über zwei Millionen Tripeln als Turtle
zu schreiben dauert ein Vielfaches, und im Entwicklungslauf liest es niemand
mit den Augen.

Ein Site-Bundle setzt einen Graphen voraus, der die Site enthält:

```cmd
py main.py --sisal-only --sisal-sites spannagel --no-bundle & py bundle_rdf.py --sites spannagel --format turtle
```

Danach liegt `dist/spannagel-bundle.ttl` — die Datei, die `wdttest-sisal`
erwartet. Sie steht in `.gitignore`.

---

## `check_docs.py` — bleibt diese Datei aktuell?

```cmd
py check_docs.py
```

Sammelt alle Skripte mit `__main__`, zieht ihre `add_argument`-Schalter heraus
und hält beides gegen `README-RUN.md`. Gemeldet wird dreierlei: ein Skript, das
hier fehlt; ein Schalter, der hier fehlt; und ein Schalter, der hier steht, den
aber kein Skript mehr annimmt.

`main.py` ruft es im Vorbereitungsschritt auf. Es ist immer eine **Warnung** —
eine Dokumentationsprüfung, die einen Lauf abbricht, ist eine, die man
abschaltet.

| Schalter | Wirkung |
|---|---|
| `--verbose` | jeden Einstiegspunkt mit seinen Schaltern auflisten |

Die Prosa bleibt Handarbeit. Was ein Skript *schreibt*, weiss kein Parser, und
das ist die nützlichere Hälfte dieser Datei — die Prüfung deckt die andere ab.

---

## `clean.py` — was ist erzeugt, was ist übrig

Ohne Schalter wird **nichts** gelöscht, sondern nur aufgelistet:

```cmd
py clean.py
```

Drei Rubriken, und der Unterschied ist wichtig:

- **Generated** — schreibt der Schritt, dem die Datei gehört, bei jedem Lauf neu
- **Unused leftovers** — einmal geschrieben, von nichts mehr gelesen
- **Scheduled** — in Benutzung, aber mit Verfallsdatum. Wird hier **nie**
  entfernt, egal mit welchem Schalter. Die Meldung am Ende eines Laufs ist eine
  Erinnerung, keine Aufforderung.

| Schalter | Wirkung |
|---|---|
| `--delete` | die erzeugten Dateien wirklich entfernen |
| `--delete --stale` | zusätzlich die Leftovers |
| `--group NAME` | auf eine Gruppe beschränken, wiederholbar |
| `--list-groups` | die Gruppen zeigen und beenden |

Gruppen: `vocab`, `diagrams`, `epica`, `sisal`, `overview`, `ci`, `archaeo`,
`bundle`, `cache` und `log`, wobei `log` nicht im Standardsatz ist.

`main.py` ruft `clean.py` in Schritt 0 mit genau den Gruppen auf, die der Lauf
danach neu schreibt — deshalb bleiben die Ausgaben eines `--ci-only`-Laufs von
EPICA und SISAL unberührt.

---

## Prüfungen, die sich lohnen

**Byte-Stabilität.** Zwei Läufe auf unveränderten Eingaben müssen dieselben
Bytes erzeugen. Einzige zugelassene Ausnahme ist `pipeline_report.txt`.

```cmd
py main.py --ci-only --archaeo-only --no-bundle & git add -A & py main.py --ci-only --archaeo-only --no-bundle & git status --short
```

Sauber ist es, wenn nur `pipeline_report.txt` in der **zweiten** Spalte ein `M`
trägt. Beide Läufe müssen dieselbe Auflösung verwenden, sonst ändern sich die
JPG selbstverständlich, und das wäre kein Befund.

**Der Ausschnitt gegen sein Manifest.**

```cmd
py SISAL\sisal_import.py --verify
```

**Vollständigkeit vor einem Release.**

```cmd
py main.py --bundle-format release
```

Erwartet: CRM-Abdeckung ohne Lücke, null SHACL-Verstösse. Eine Warnung zu
`cisite_59` ist bekannt und offen — Kostenki-Borschtschewo trägt noch keinen
`skos:closeMatch` auf eine externe Normdatei.

---

## Wenn etwas nicht läuft

| Meldung | Ursache |
|---|---|
| `ModuleNotFoundError: geo_lod_utils` | `PYTHONPATH` zeigt nicht auf `ontology/`. `main.py` setzt das selbst |
| `data/raw/basemap/ne_50m_land.geojson not found` | die Kartengrundlage fehlt; sie ist 1,6 MB gross und reist nicht in jedem Bundle mit |
| `none of SISAL/rdf/sisal_sites.ttl … found` | die Übersichtskarte lief vor dem SISAL-Schritt |
| `columns missing from cifindspots_part_full.csv` | die Fundstellentabelle stammt aus einer anderen Fassung von `campanian-ignimbrite-geo` |
| Bundle-Schritt dauert ewig | das ist die SHACL-Validierung über zwei Millionen Tripel. `--no-bundle` überspringt sie |
