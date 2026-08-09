# Primer — Erweiterung des geo-lod-Datensatzes

Arbeitsplan für die Erweiterung von `GeoScience-FAIRification-LOD` um zusätzliche
EPICA- und SISAL-Daten, die Angleichung an das WD1-Modell und den ELSA-Strang.

**Ort.** <https://github.com/Research-Squirrel-Engineers/GeoScience-FAIRification-LOD/blob/main/PRIMER.md>

**So wird es benutzt.** Es wird in jedem Chat vollständig hochgeladen. Danach
genügt ein Satz: „Wir machen S3a." Teil A gilt immer, Teil B ist die Übersicht,
Teil C beschreibt den einzelnen Schritt. Die Statusspalte in Teil B und die
Beschlusslage in A4 werden nach jedem Chat nachgeführt, damit spätere Chats den
aktuellen Stand sehen.

---

# Teil A — Immer gültig

## A1. Ausgangslage

**Zwei Repo-Familien, ein Namensraum.** Beide arbeiten unter
`http://w3id.org/geo-lod/`:

| | Repo | Inhalt |
|---|---|---|
| geo-lod | `GeoScience-FAIRification-LOD` | `geolod:` Kernontologie, EPICA/SISAL/CI-Tripel, SHACL, CRM-Bridging |
| geteilt | `Research-Squirrel-Engineers/sisal-db-v3` | Restore der SISAL-v3-Datenbank; Zugriffsschicht für beide Familien |
| WD1 | `wdttest-epica` | 5 PANGAEA-`.tab`, Abbildungen, FDO-Metadaten |
| WD1 | `wdttest-sisal` | Abbildungen aus drei Site-CSVs |
| WD1 | `wdttest-tables` (Semantic Layer) | `strat:`, `an:`, `crm_bridge`, repo-lokale Vokabulare |

**Was schon zusammenpasst.** `strat.ttl` liegt unter
`http://w3id.org/geo-lod/strat/`, `crm_bridge.ttl` importiert
`http://w3id.org/geo-lod/` und hängt `an:Specimen ⊑ geolod:PalaeoclimateSample`.
Die Anbindung ist vorgesehen, nur noch nicht genutzt.

**Was nicht neu gebaut werden muss.** Alle 305 SISAL-Sites stehen bereits in
`SISAL/rdf/sisal_sites.ttl` als `geolod:Cave_site_NNNN` mit `geolod:siteId` —
einschliesslich 58 Spannagel, 202 Piani Eterni, 275 Buraca Gloriosa. Neue Daten
hängen sich an bestehende Knoten an.

**Datenmengen.** EPICA rund 4 900 Messwerte über fünf Proxies. Das vollständige
SISAL-v3-Release hat 448 000 Samples — die Datenbank wird vollständig
restauriert, nach RDF geht nur eine Auswahl.

## A2. Zielbild

**geo-lod ist der Haupttreiber.** Alles Semantische — Ontologie, Vokabulare,
Instanzdaten — entsteht in `GeoScience-FAIRification-LOD`. Die `wdttest-*` /
`wd-*`-Repos sind davon unabhängig; sie leisten für Fionas Paper einiges, was
geo-lod nicht braucht, und erben aus geo-lod ausschliesslich die Ontologie.

- Keine Laufzeitkopplung in beide Richtungen: geo-lod läuft ohne die WD1-Repos,
  die WD1-Repos laufen ohne geo-lod.
- Rohdaten und Code werden nachgenutzt, nicht referenziert — jede Seite hält,
  was sie zum Laufen braucht.
- Geteilt wird genau ein Artefakt: die Ontologie unter
  `http://w3id.org/geo-lod/`. Sie wird importiert, nicht kopiert.
- ELSA kommt zuletzt und dient als Abnahmetest: fügt es sich ein, ohne dass an
  `geolod:` oder `strat:` nachgepatcht werden muss, war die Angleichung richtig.

## A2b. Verhältnis zum ECEASST-Beitrag

Der Proceedings-Beitrag ist im Review. Eine überarbeitete Fassung ist danach
möglich und wahrscheinlich nötig. Damit gilt:

- Der Beitrag ist **kein eingefrorener Stand**, an dem sich die Arbeit hier
  ausrichten muss. Ergebnisse aus S0 bis S5 können und sollen in die
  Überarbeitung einfliessen.
- Das betrifft nicht nur die RDF-Daten, sondern auch die Abbildungen. Wo ein
  Schritt hier eine Abbildung inhaltlich verändert — andere Altersachse,
  korrigierte SISAL-Werte, zusätzliche Proxies — wird das für die Überarbeitung
  vermerkt, statt die alte Abbildung zu konservieren.
- Der IRI-Umzug aus S0.4 ist damit deutlich entspannter: die im Beitrag
  genannten IRIs lassen sich in der überarbeiteten Fassung mitziehen. Die
  Abbildung alt → neu bleibt trotzdem nötig, weil der Preprint zitierbar ist.
- Umgekehrt gilt weiter A3: der Beitrag ist kein Grund, hier Umwege zu gehen.
  Erst wird die Sache richtig modelliert, dann wird der Text nachgeführt.

## A3. Querschnittsregeln

- **`metadata/ontology.ttl` und `metadata/shapes.ttl` sind tabu.** Das ist die
  gemeinsame `wdt:`-FDO-Schicht der WD1-Familie, verbatim über alle Repos,
  zentral versioniert. Sie ist etwas anderes als die geo-lod-Ontologie und wird
  nicht mit ihr vermischt. Der fachliche Semantic Layer gehört dort ins
  Verzeichnis `metadata/ontology/` — so macht es `wdttest-tables` bereits.
- Nachnutzung heisst kopieren, nicht referenzieren. Einzige Ausnahme ist die
  Ontologie über ihre w3id-IRI.
- **Rohdaten liegen unter `data/raw/<strang>/`**, unverändert wie bezogen, und
  werden nur gelesen. Was ein Skript daraus macht, geht nach `dist/`. Damit ist
  an jeder Datei ablesbar, ob sie Quelle oder Erzeugnis ist; `info/` als
  Ablage für „brauche ich noch" entfällt.
- Kopierter Code trägt seine Prüfungen mit. Ein Guard, der beim Kopieren
  wegfällt, ist die gefährlichste Form von Nachnutzung.
- **Kein Spreadsheet und kein Import-Assistent im Ladeweg.** Das gilt auch für
  den pgAdmin-Importdialog. `COPY ... (FORMAT csv)` ist der einzige Pfad.
- **Zweimal laufen lassen, `git status` muss sauber bleiben.** Tripel sortiert
  ausgeben, keine zufälligen Blank-Node-IDs. Bei matplotlib-SVG kommen zwei
  weitere Quellen dazu: das `<dc:date>` in den Metadaten und die zufälligen
  Clip-Path-IDs. Beides ist abgestellt, `plt.rcParams["svg.hashsalt"]` und
  `metadata={"Date": None}` beim Speichern — bei jedem neuen Plot-Skript
  mitnehmen.
- **Keine Uhr im Output.** Kein Generator liest `datetime.now()`. Ein Datum im
  RDF kommt aus `GEO_LOD_RELEASE` in `geo_lod_utils.py`, die Bindung an den
  Stand aus `content_fingerprint()` über Eingabedaten und Generator-Skript.
  Damit ändert sich ein erzeugter Datensatz genau dann, wenn sich Daten oder
  Modell ändern — was ihn für den, der den TTL-Dump vor sich hat, prüfbar
  macht. Neue Generatoren rufen `add_generation_provenance()` auf.
- Ein Thema pro Chat, ein Repo pro Chat.
- **Offene Entscheidungen werden als interaktives Formular gestellt**, nicht als
  Aufzählung im Fliesstext. Ein Klickpfad ist schneller zu beantworten als eine
  Liste, und die Antwort kommt in einer Form zurück, die sich direkt nach A4
  übertragen lässt.
- **Jedes Formular hat immer ein freies Kommentarfeld**, unabhängig davon, wie
  eindeutig die Optionen scheinen. Die wichtigsten Antworten sind regelmässig
  die, an die beim Formulieren der Frage niemand gedacht hat — etwa dass eine
  Datei anderswo hingehört, als die Frage unterstellt. Ohne Feld gehen sie
  entweder verloren oder erzwingen eine zweite Runde.
- Sprache: Konversation deutsch, Code/Ontologie/Dokumentation englisch.
  Ausnahme: dieses `PRIMER.md` bleibt deutsch — es ist ein internes
  Arbeitsdokument, das offen liegt, aber nicht nach aussen adressiert ist.

## A5. Was in welchem Chat hochgeladen wird

Die Zeile **Uploads** bei jedem Schritt in Teil C nennt, was zusätzlich
gebraucht wird. Grundregel: lieber das Bundle als einzelne Dateien — bei
Einzeldateien fehlt regelmässig der Kontext, und Nachfragen kosten mehr als der
Upload.

**Das geo-lod-Bundle.** Das Repo ist mit Abbildungen und generiertem RDF rund
54 MB gross, das meiste davon erzeugt. Gebraucht wird der Code, die Ontologie,
die kleinen Eingabedaten und die kleinen TTL. Ein Grössenfilter erledigt die
Auswahl zuverlässiger als eine Dateiliste:

```cmd
cd /d C:\git
robocopy GeoScience-FAIRification-LOD bundle\geo-lod /E /MAX:1000000 ^
  /XD plots img dist .git .venv __pycache__ example_query ^
  /XF *.jpg *.svg *.png
powershell -NoProfile -Command "Compress-Archive -Path 'bundle\geo-lod' -DestinationPath 'geo-lod_bundle.zip' -Force"
```

`/MAX:1000000` lässt `geo_lod_core.ttl`, die Ontologiemodule, `core_shapes.ttl`,
`sisal_sites.ttl` und `ci_findspots.ttl` durch und filtert die fünf grossen
generierten Datendateien heraus. Ergebnis: wenige MB. Robocopy meldet Exitcode 1
bei Erfolg.

**Was gitignoriert ist, kommt nicht über GitHub.** Geprüft 2026-08-08: die
`.gitignore` ist die Python-Vorlage, `dist/` steht dort auskommentiert und
`data/` gar nicht — beide sind also versioniert, entsprechend liegt
`dist/geo-lod-bundle.ttl` im Repo. Der frühere Satz, sie lägen nicht im Repo,
war falsch. Das ist so gewollt: die kleinen Rohdaten unter `data/raw/` und die
erzeugten Tabellen in `dist/` sollen über GitHub zitierbar sein. Grosse
generierte Dateien bleiben trotzdem aus dem Upload-Bundle draussen, dafür sorgt
der Grössenfilter, nicht die `.gitignore`.

**Wenn eine Datei gezielt geändert werden soll**, genügt sie einzeln zusätzlich
zum Bundle; dann sehe ich den aktuellen Stand und den Kontext gleichzeitig.

**Nicht hochladen:** `config.ini`, `.venv/`, `.git/`, generierte Abbildungen,
die grossen Daten-TTL, die SISAL-CSVs.

## A4. Beschlusslage

Wird nach S0 gefüllt und danach nur noch fortgeschrieben. Alle späteren Schritte
lesen hier ab, statt neu zu diskutieren.

| Frage | Beschluss | seit |
|---|---|---|
| Kanonische Altersskala | ka BP | 2026-08-08 |
| Natives Alter zusätzlich speichern? | nein — ein Alterswert je Beobachtung, in ka BP. Die Chronologie bleibt als eigener Knoten erhalten | 2026-08-08 |
| `crmarchaeo:`-Namensraum | `http://www.cidoc-crm.org/extensions/crmarchaeo/`; `crm_bridge.ttl` und `ontology/README.md` ziehen nach | 2026-08-08 |
| MIS: Grenzen, Warm/Kalt-Einstufung, wer sie zugewiesen bekommt | Grenzen als konkurrierende E13 je Quelle; Zuweisung an Beobachtungen materialisiert, ebenfalls je Quelle als E13; Warm/Kalt als Property im Schema, mit Literaturbeleg | 2026-08-08 |
| IRI-Muster Instanzdaten | ein Zweig je Strang: `…/epica/`, `…/sisal/`, `…/ci/`, `…/elsa/`; Klassen bleiben flach unter `…/geo-lod/` | 2026-08-08 |
| Bestand auf das neue IRI-Muster ziehen? | ja, einmalig breaking — CI-Strang und die 305 `Cave_site_NNNN` wandern mit | 2026-08-08 |
| Achsenbeschriftung in Darstellungen | immer `Age [ka]`, ohne Zusatz BP oder b2k — in allen Abbildungen, auch den bereits publizierten | 2026-08-08 |
| Alter direkt in `[ka]` abfragbar | ja — `time:TimePosition` mit `time:numericPosition` und `time:hasTRS`, keine Umrechnung in der Query | 2026-08-08 |
| TRS-IRI für `ka BP` | eigene TRS je Chronologie unter `…/trs/`; keine passende externe vorhanden (Recherche 2026-08-08) | 2026-08-08 |
| Verhältnis `geolod:ageKaBP` zu `time:TimePosition` | beides parallel, kein Deprecation | 2026-08-08 |
| Auslieferung `sisalv3_csv.zip` vs. Download | in S3b entscheiden — keine Rückwirkung auf IRIs | 2026-08-08 |
| SISAL-Site-Auswahl für RDF | in S3 entscheiden, wenn die vollständige Datenbank vorliegt | 2026-08-08 |
| Waisen bei FK-Aktivierung: abweisen oder laden | in S3b entscheiden — zeigt sich erst beim Ladelauf | 2026-08-08 |
| PRIMER.md-Sprache | deutsch — internes Arbeitsdokument | 2026-08-08 |
| MIS-Leitschema | Railsback et al. 2015, durchgehend. Auch die WD1-Familie zieht darauf um; `wdttest-wd1--ager-corg` ist bereits dort, `wdttest-tables` folgt in S4 | 2026-08-08 |
| Umfang des MIS-Schemas | vollständig bis TG5/TG6 (5315 ka). Bis 1013,1 ka Railsback, darüber hinaus LR04 als einzige Quelle; Herkunft steht als `dct:source` am Konzept | 2026-08-08 |
| Welche Quelle gilt, wenn beide etwas sagen | Railsback führt, solange es reicht; darüber hinaus LR04. Beide Lesarten bleiben erhalten, aber jede Zuweisung trägt `geolod:assignmentStatus` (leitend oder alternativ) und jedes Konzept `geolod:leadingSource` | 2026-08-08 |
| LR04-Peaks 5.1 bis 5.5 | keine Grenzen, sondern Exkursionen: an die Substadien 5a bis 5e gehängt, als `geolod:ExcursionPeak` | 2026-08-08 |
| Elternstadien ohne Railsback-Zeile | aus der Vereinigung der Substadien abgeleitet, mit `prov:wasDerivedFrom` und Vermerk am E13 | 2026-08-08 |
| MIS-Labels | `skos:prefLabel "MIS 5e"`, `skos:altLabel "5e"` | 2026-08-08 |
| Warm/Kalt | Paritätskonvention, ungerade warm, gerade kalt, Beleg Railsback et al. 2015. Nur nummerierte Stadien; die Buchstabenstadien des Pliozäns bleiben unklassifiziert | 2026-08-08 |
| Ort der Rohdaten | `data/raw/<strang>/`, unverändert und nur lesend. Die MIS-Quellen sind aus `info/` dorthin gezogen; `info/` entfällt | 2026-08-08 |
| Aufbereitete Tabellen aus dem Vokabular | ja, aus demselben Lauf nach `dist/`: `mis_stages.csv` (eine Zeile je Konzept, leitende Werte) und `mis_assignments.csv` (eine Zeile je Zuweisung, beide Lesarten mit Status) | 2026-08-08 |
| SVG-Determinismus | `svg.hashsalt` gesetzt, `<dc:date>` unterdrückt — 33 Abbildungen über zwei Läufe byte-identisch, JPG ohnehin | 2026-08-08 |
| SHACL-Inferenz | `inference="none"` statt `"rdfs"`. SHACL folgt `rdfs:subClassOf` bei `sh:targetClass` und `sh:class` selbst; die Subklassen-Axiome liegen im Bundle. Ergebnis identisch, Schritt von 42 s auf 14 s | 2026-08-08 |
| Laufzeitmessung | je Schritt eine Dauer im Report, dazu eine Tabelle mit Anteilen in der Summary | 2026-08-08 |
| Log der Sub-Skripte | eingefangen und zeilenweise durchgereicht: `pipeline_report.txt` und Terminal zeigen dasselbe. `PYTHONIOENCODING=utf-8` im Kindprozess, sonst scheitern ✓, ‰ und δ an der Pipe | 2026-08-08 |
| Zeitstempel im RDF | keine. `dct:created` aus `GEO_LOD_RELEASE`, dazu `owl:versionInfo` mit einem SHA-256-Fingerabdruck über Eingabedaten und Generator-Skript. Die Aktivitätszeiten in `ci_findspots.ttl` sind entfallen — die Laufzeit eines Konvertierungsskripts ist keine Aussage über die Daten | 2026-08-08 |
| Umfang des Fingerabdrucks | Eingabedaten **und** Generator-Skript. Ein geänderter Kommentar im Code ändert ihn mit; das ist gewollt, weil der Datensatz nur für den Codestand gilt, aus dem er stammt | 2026-08-08 |
| `pipeline_report.txt` | wird weiter versioniert; er ändert sich als einzige Datei bei jedem Lauf, das ist der Preis für den Laufprotokoll-Charakter | 2026-08-08 |
| Zeilenenden | `.gitattributes` mit `eol=lf`. Ohne sie schreibt Git auf Windows beim Checkout um, und die Byte-Gleichheit gilt nur lokal | 2026-08-08 |
| Bundle-Format | Turtle als Voreinstellung, weil byte-stabil und versioniert; N-Triples spart gemessen nur 13 von 187 Sekunden und lohnt den möglichen Versatz zwischen `.ttl` und Code nicht. Über `--bundle-format` sind `nt`, `jsonld`, `xml` und `longturtle` erreichbar, `release` schreibt alle | 2026-08-08 |
| `dist/geo-lod-bundle.nt` | gitignoriert. N-Triples ist nicht byte-stabil, weil rdflib beim Parsen neue Blank-Node-Labels vergibt und sie im Klartext ausgibt; Turtle bleibt stabil, dort stehen dieselben Knoten inline als `[ ]` | 2026-08-08 |
| Aggregatdateien im Bundle | `sisal_all_data.ttl` wird übersprungen. Sie ist die Vereinigung der vier Höhlendateien, trägt geprüft null Tripel bei und kostete ein Drittel der Parse-Zeit | 2026-08-08 |
| Logdateien | mit `newline="\n"` geschrieben. Sonst schreibt Python auf Windows CRLF, während Git nach `.gitattributes` LF ablegt — die Arbeitskopie wiche dauerhaft von ihrer eigenen abgelegten Form ab | 2026-08-08 |
| Ebene, an der eine EPICA-Beobachtung hängt | Kern als Feature of Interest; ein `geolod:SampleSection` nur dort, wo die Quelle das Intervall nennt — das ist allein der δD-Datensatz mit Tiefe top/bottom und Alter min/max. Bei den übrigen vier wäre ein Probenknoten eine erfundene Mächtigkeit | 2026-08-09 |
| CH₄ mit zwei Altersspalten | beide ins RDF, EDC2 leitend. `geolod:ageChronology` an der Beobachtung sagt, welcher der beiden `time:TimePosition` das materialisierte `geolod:ageKaBP` folgt | 2026-08-09 |
| MIS-Zuweisung je Beobachtung | ja, nur die leitende Lesart (Railsback). Eigene Klasse `geolod:MISMembershipAssignment`, weil `geolod:MISAttributeAssignment` Eigenschaften *an* ein Stadium hängt und diese hier ein Stadium *an* etwas anderes; die Shapes unterscheiden sich entsprechend | 2026-08-09 |
| Nur Stadien, keine Substadien für die Zugehörigkeit | Railsback löst Substadien nur über einen Teil des Bereichs auf. Stadien-Zugehörigkeit ist durchgehend vollständig, Substadien wären lückenhaft; über `skos:broader` bleiben sie erreichbar | 2026-08-09 |
| Glättungswerte im RDF | bleiben. Die publizierten Abbildungen zeigen geglättete Kurven, also muss der Graph beantworten können, was gezeichnet wurde; die Parameter hängen als eigene Knoten daran | 2026-08-09 |
| Bohrstelle Dome C | ein `geolod:DrillingSite`, darunter zwei `geolod:Borehole` (EDC99 und DomeC) mit je eigener Geometrie und `crm:P89_falls_within`. Die beiden PANGAEA-Events geben Koordinaten 1,3 km auseinander; keine wird verworfen, keine gemittelt. Der Site verweist per `geo:hasGeometry` auf die EDC99-Geometrie, statt die Koordinate ein zweites Mal hinzuschreiben | 2026-08-09 |
| IRI-Muster Instanzdaten EPICA | klein mit Unterstrich: `epica:obs_ch4_0001`, `epica:tp_ch4_0001`, `epica:site_dome_c`. Klassen und Properties bleiben flach unter `…/geo-lod/` | 2026-08-09 |
| MIS-Grenze in Tiefe ins RDF | ja, als `geolod:MISAttributeAssignment` mit `geolod:MISBoundaryDepth`, einer `geolod:DepthPosition` als Wert und der Chronologie an der Zuweisung. Lineare Interpolation, **nicht** extrapoliert: eine Grenze jenseits des tiefsten Messpunkts bekommt keine Zuweisung. 81 Zuweisungen über fünf Datensätze | 2026-08-09 |
| TRS je Chronologie **und Phase** | sechs statt drei: `EDC1-gas`, `EDC2-gas`, `EDC2-ice`, `EDC3-ice`, `AICC2023-gas`, `AICC2023-ice`. **Korrigiert A6 vom 2026-08-08**, wo nur AICC2023 getrennt war. Beleg aus den Headern: CH₄ ist Gasalter auf EDC2, δD Eisalter auf EDC2 — ungetrennt liegt die MIS-5-Grenze bei beiden 27 m auseinander, ohne dass der Graph sagen könnte warum. Mit Trennung liegen die drei Eisalter-Skalen bei 1734–1756 m, die beiden Gasalter-Skalen bei 1760–1782 m, was der Gas-Eis-Altersdifferenz entspricht | 2026-08-09 |
| `geolod:Chronology` als `time:TRS` | umgesetzt. Ein Knoten in zwei Rollen: `geolod:ageChronology` an der Beobachtung und `time:hasTRS` an ihrer `time:TimePosition` zeigen auf dasselbe Individuum unter `…/trs/` | 2026-08-09 |
| δ¹⁸O-Datensatz | ist δ¹⁸O **von O₂ aus der Luft**, kein Wasserisotop. Der Header sagt `δ18O, gas [‰] … COMMENT: of O2`. Die alte Ontologie beschrieb ihn als „stable water isotope ratio … from the ice matrix“ — falsch. Das Wasserisotop des Kerns ist δD. Betrifft auch die Achsenbeschriftung im ECEASST-Beitrag | 2026-08-09 |
| Provenienz aus `data.yaml` | als Python-Konstante `DATASETS` in `EPICA/epica_data.py` nachgenutzt, nicht als YAML. Nachnutzung heisst kopieren (A3), und für fünf Datensätze lohnt keine PyYAML-Abhängigkeit. Werte gegen die `.tab`-Header geprüft, zwei korrigiert (δ¹⁸O-Beschreibung, Gas/Eis-Trennung) | 2026-08-09 |
| EPICA-Schritt in `main.py` | zweigeteilt: erst RDF (`epica_rdf.py`), dann Abbildungen (`plot_epica_from_tab.py`). So fällt ein Datenfehler auf, bevor die Abbildungen entstehen | 2026-08-09 |
| Tafel-Layout | beides: `plate_columns_*` (fünf Spalten, senkrechte Altersachse, wie der Bestand) und `plate_rows_*` (fünf Zeilen, waagerechte Achse, wie die Eiskern-Literatur) | 2026-08-09 |
| Altersfenster der Tafeln | nur voll 0–806 ka. Ausschnitte erst, wenn der Text sie braucht | 2026-08-09 |
| Dust-Skala | logarithmisch; die Werte spannen Faktor 560 (2,7 bis 1525 µg/kg) | 2026-08-09 |
| Glättung auf den Tafeln | je ein vollständiger Satz ungeglättet, Median und Savitzky-Golay, nicht roh und geglättet in einer Abbildung | 2026-08-09 |
| MIS-Bänder in **allen** Abbildungen | aus `dist/mis_stages.csv`, also Railsback. Die zwölf Einzelabbildungen ändern sich dadurch sichtbar: die alte Liste war LR04 mit zwei von Hand ans CH₄-Signal angepassten Übergängen und ohne MIS 14, und MIS 3 galt dort als Interstadial statt nach Paritätskonvention als warm. Für die Überarbeitung des Beitrags vermerkt | 2026-08-09 |
| „Kein Datum“-Bänder | nicht mehr hartcodiert. `draw_mis_bands` bekommt die Alter der Messpunkte und schraffiert die Stadien ohne einen einzigen Messwert selbst — gilt damit für alle fünf Datensätze statt nur für die drei früher für CH₄ eingetragenen | 2026-08-09 |
| Achsenbeschriftung `Age [ka]` | umgesetzt, in Einzelabbildungen wie Tafeln. Der Bezugspunkt steht am Chronologieknoten im Graphen, nicht in jeder Bildunterschrift | 2026-08-09 |
| Interpolation über Datenlücken | verboten. `interpolate_depth` liefert `None`, wenn die beiden umgebenden Messpunkte weiter als 15 ka auseinanderliegen. Vorher bekamen die Beginne von MIS 8, 9 und 10 im CH₄-Datensatz eine Tiefe, die über 178 ka ohne Daten geradlinig hinweggerechnet war — auf der Abweichungstafel ein Ausschlag von 36 m, der wie ein Modellunterschied aussah und keiner war. 77 Grenzen statt 81 | 2026-08-09 |
| SVG-Zeilenenden | über ein binär geöffnetes Handle geschrieben (`epica_style.save_figure`). matplotlib öffnet sonst im Textmodus, und Python übersetzt auf Windows jedes Zeilenende zu CRLF, während `.gitattributes` LF ablegt — die Arbeitskopie wich damit bei **jeder** Abbildung von ihrer eigenen abgelegten Form ab. Dasselbe Muster wie bei den Logdateien am 2026-08-08, eine Ebene weiter | 2026-08-09 |
| Wertachsen | aus den Daten, `epica_style.nice_ticks`. Schritte aus 1, 2, 2,5 oder 5 mal einer Zehnerpotenz, Nachkommastellen aus dem Schritt, Grenzen umschliessen den Wertebereich immer. Handgesetzte Achsen bleiben über `AXIS_OVERRIDES` in `plot_epica_from_tab.py` möglich | 2026-08-09 |
| **Abgeschnittene Messwerte** | behoben. Die feste Tick-Liste `D18O_TICKS` endete bei 1,0 und setzte damit die Achsengrenze auf 1,075; δ¹⁸O reicht bis 1,457. **82 von 1378 Messwerten (6 %) lagen ausserhalb der Achse und wurden weggeschnitten** — in jeder δ¹⁸O-Abbildung einschliesslich der publizierten Collage. Für die Überarbeitung des Beitrags vermerkt | 2026-08-09 |
| Umfang der Einzelabbildungen | alle fünf Proxies, Tiefe und Alter, drei Glättungsvarianten: 30 statt 12. Die Konfigurationen kommen aus `DATASETS`, nicht mehr aus einer handgeschriebenen Liste; δD, Dust und δO₂/N₂ hatten bis dahin überhaupt keine Einzelabbildung | 2026-08-09 |
| MIS-Bänder in Tiefendarstellungen | ja, nur in den Einzelabbildungen. Die Tiefen kommen aus `ed.interpolate_depth`, also aus derselben Funktion wie `geolod:MISBoundaryDepth` im Graphen. Wo die Interpolation verweigert, fehlt das Band — bei CH₄ betrifft das MIS 8 bis 10, und ein fehlendes Band ist dort die ehrliche Auskunft | 2026-08-09 |
| Collage des Beitrags | zwei Zuschnitte aus derselben Funktion: `plate_pipeline_outputs` mit den vier bisherigen Panels (δ¹⁸O und CH₄, je Alter und Tiefe), damit die bestehende Abbildung ersetzbar ist, ohne den Text zu ändern, und `plate_pipeline_outputs_five` mit allen fünf Datensätzen auf der Altersachse. Beide zeigen Rohwerte grau hinter der Glättung — das ist die Aussage der Abbildung | 2026-08-09 |
| Benennung der Collagen | `plate_pipeline_outputs` und `plate_pipeline_outputs_five`. Alles Mehrteilige heisst `plate_*`, `fig02_*` entfällt — eine Abbildungsnummer aus einem Beitrag ist kein Dateiname, sie ändert sich beim nächsten Umbruch | 2026-08-09 |
| Bildüberschrift in der Grafik | nur bei den mehrteiligen Tafeln entfernt. Die Einzelabbildungen behalten ihren Titel — dort ist er die einzige Kennzeichnung. Die Panel-Titel innerhalb einer Tafel bleiben ebenfalls, sonst wäre nicht erkennbar, welche Spalte welche ist | 2026-08-09 |
| Captions | `EPICA/captions.yaml`, eine Datei je Strang. Feldnamen wie in `wdttest-tables`: `caption`, `captiondetail`, `license`, dazu `sources` mit den DOIs und `generated` | 2026-08-09 |
| Wem gehört die Caption | erzeugt, aber nacheditierbar. Jeder Eintrag führt unter `generated` den zuletzt maschinell erzeugten Text mit. Weicht `caption` davon ab, gilt er als von Hand geschrieben und bleibt stehen, während `generated` nachgeführt wird — der Diff zeigt damit, wo Prosa und Daten auseinandergelaufen sind. Von Hand ändert man `caption` und sonst nichts | 2026-08-09 |
| Neue Abhängigkeit `pyyaml` | nötig, weil die Caption-Datei nicht nur geschrieben, sondern auch wieder gelesen wird, um die handgeschriebenen Einträge zu erkennen. Damit ist die Begründung vom Vormittag, für fünf Konstanten lohne kein YAML, nicht mehr tragfähig; `EPICA/epica_data.py` bleibt trotzdem Python, weil es niemand editiert | 2026-08-09 |
| Ort der geteilten Abbildungs-Helfer | `ontology/geo_lod_figures.py` und `ontology/geo_lod_captions.py`, neben `geo_lod_utils.py`. Jedes Sub-Skript läuft ohnehin mit `ontology/` auf dem `PYTHONPATH`. `EPICA/epica_style.py` entfällt | 2026-08-09 |
| SISAL-Abbildungen | auf `geo_lod_figures.save_figure` umgestellt, sofort und nicht erst in S3c. Der CRLF-Fehler war dort derselbe, und ein bekannter Fehler, der auf einen späteren Schritt vertagt wird, wird bei jedem Lauf dazwischen erneut geschrieben. SISAL bekommt seine `captions.yaml` in S3c | 2026-08-09 |
| Darstellung von Datenlücken | Konvention aus `wdttest-sisal` übernommen: innerhalb eines Laufs durchgezogen, über die Unterbrechung gestrichelt, die letzte Probe davor und die erste danach geringelt. Liegt als `geo_lod_figures.find_breaks` / `draw_profile` im geteilten Modul, gilt für Einzelabbildungen, Tafeln und Collagen. Der Vorbehalt wandert mit: ein gestrichelter Abschnitt heisst „hier keine Proben", nicht „hier kein Eintrag" | 2026-08-09 |
| Lückenschwelle EPICA | 15 ka, nicht die 5 kyr aus `wdttest-sisal`. Bei 5 ka markiert die Konvention 14 Lücken im Dust-Datensatz und 5 im δD, die nur dünne Beprobung sind; bei 15 ka bleibt genau die eine echte Lücke im CH₄. Der Wert ist derselbe wie `ed.MAX_INTERPOLATION_GAP_KA` und wird von dort bezogen — was in der Abbildung eine Lücke ist, ist genau die Strecke, über die der Generator keine Stadiengrenze interpoliert | 2026-08-09 |
| Glättung über Lücken | verboten, laufweise statt über die ganze Reihe (`geo_lod_figures.smooth_by_run`). Ein zentriertes Fenster von 11 Punkten griff bei CH₄ über 178 ka ohne Daten hinweg: der geglättete Wert bei 214 ka enthielt Messungen von 392 ka. Betrifft 9 der 736 CH₄-Werte, maximale Abweichung 63 ppbv. **Gilt für Abbildung und RDF**, aus derselben Funktion — `geolod:smoothedValue_rollingMedian` und `…_savgol` haben sich für diese neun Beobachtungen geändert | 2026-08-09 |
| Ringe an der Bruchstelle | auf **beiden** Kurven, Rohwerte wie Glättung. Der Ring markiert die letzte gemessene Probe vor der Unterbrechung — eine Aussage über die Messreihe, nicht über die Glättung. Der Versatz zwischen grauem und schwarzem Ring zeigt nebenbei, wie weit die Glättung dort vom Messwert abweicht | 2026-08-09 |
| Teilweise bekannte MIS-Bänder in Tiefe | werden gezeichnet, schraffiert, bis an den Rand der Messreihe. Vorher entfiel ein Band, sobald **eine** seiner Grenzen in eine Datenlücke fiel — bei CH₄ verschwanden so MIS 7 und MIS 11, obwohl von beiden je eine Kante interpolierbar ist. Stadien, die vollständig in der Lücke liegen (MIS 8 bis 10), bekommen weiterhin kein Band; dort gibt es keine Tiefe, der sie zuzuordnen wären | 2026-08-09 |
| Datenlücke in Tiefendarstellungen | bekommt ein eigenes neutrales Band mit der Beschriftung „no samples". Vorher stand dort weisse Fläche, und weiss ist in diesen Abbildungen sonst nichts — eine Lücke sah aus wie ein Zeichenfehler | 2026-08-09 |

## A6. IRI-Landkarte unter `http://w3id.org/geo-lod/`

Was unter welchem Pfad liegt, und was davon bei w3id als Redirect eingetragen
werden muss. `http://` ist durchgängig korrekt, nicht `https://`.

**Status:** `aktiv` = wird bereits ausgeliefert oder ist vergeben;
`beschlossen` = in A4 festgelegt, noch nicht angelegt; `geplant` = Vorschlag,
noch nicht entschieden.

| Pfad | Inhalt | Ziel des Redirects | Status |
|---|---|---|---|
| `/geo-lod/` | Kernontologie: Klassen und Properties, flach | `geo_lod_core.ttl` | aktiv |
| `/geo-lod/strat/` | `strat:` — Semantic Layer der WD1-Familie | `strat.ttl` in `wdttest-tables` | aktiv |
| `/geo-lod/vocab/mis/` | MIS-Stadien und -Substadien, Grenzen als E13 | `ontology/vocab/mis.ttl` | aktiv (S1) |
| `/geo-lod/vocab/tephra/` | Marker-Tephren, Join zwischen WD1, ELSA und CI | `ontology/vocab/tephra.ttl` | reserviert (S5) |
| `/geo-lod/epica/` | Instanzdaten Eiskern | `EPICA/rdf/` | aktiv (S2) |
| `/geo-lod/sisal/` | Instanzdaten Speläotheme, inkl. der 305 Cave-Sites | `SISAL/rdf/` | beschlossen (S3c) |
| `/geo-lod/ci/` | Instanzdaten Campanian-Ignimbrite-Fundstellen | `CI/rdf/` | beschlossen (S4-Umzug) |
| `/geo-lod/elsa/` | Instanzdaten Maarsedimente | noch nicht vorhanden | beschlossen (S5) |
| `/geo-lod/shapes/` | SHACL-Gate, `core_shapes.ttl` plus `strat`-Shapes | `ontology/shapes/` | geplant (S4) |
| `/geo-lod/trs/` | Temporal Reference Systems, eine je Chronologie **und Phase**: `LR04`, `Railsback2015`, `EDC1-gas`, `EDC2-gas`, `EDC2-ice`, `EDC3-ice`, `AICC2023-gas`, `AICC2023-ice`, später SISAL und ELSA | `ontology/trs.ttl` | aktiv (S1 + S2) |

**Was hier bewusst nicht steht.** `metadata/ontology.ttl` und
`metadata/shapes.ttl` der WD1-Familie liegen unter `wdt:`, nicht unter
`geo-lod`. Sie sind nach A3 tabu und bekommen hier keinen Pfad.

**Zu klären beim Eintragen:**

- Content Negotiation ist Voraussetzung für S1: solange `/geo-lod/` kein TTL
  ausliefert, zeigt der `owl:imports` in `crm_bridge.ttl` ins Leere und jeder
  Konsument hält doch eine lokale Kopie.
- Der Namensraum von `an:` aus `wdttest-tables` ist noch nicht geprüft. In S4
  nachsehen: liegt er unter `/geo-lod/`, gehört er in diese Tabelle, sonst wird
  er hier als externer Namensraum vermerkt.
- Ein Redirect auf GitHub Pages liefert nur genau eine Repräsentation aus. Für
  echte Content Negotiation braucht es entweder w3id-seitige `Accept`-Regeln
  oder je Pfad einen `.ttl`- und einen `.html`-Eintrag.

---

# Teil B — Schrittübersicht

| ID | Schritt | Repo | hängt ab von | Status |
|---|---|---|---|---|
| S0 | Festlegungen, kein Code | — | — | erledigt 2026-08-08 |
| S1 | Gemeinsame Vokabulare | geo-lod | S0 | erledigt 2026-08-08 |
| S2 | EPICA nach RDF | geo-lod | S0, S1 | erledigt 2026-08-09 |
| S3a | SISAL: DDL MySQL → Postgres | sisal-db-v3 | — | offen |
| S3b | SISAL: Loader, Guard, Aufräumen | sisal-db-v3 | S3a | offen |
| S3c | SISAL nach RDF | geo-lod | S0, S1, S3b | offen |
| S4 | Ontologie-Angleichung | geo-lod + wdttest-tables | S2, S3c | offen |
| S5 | ELSA | geo-lod | S4 | offen |

S2 und S3a/S3b laufen unabhängig voneinander und können in beliebiger
Reihenfolge oder parallel gemacht werden. S3a hängt an keiner Festlegung aus S0
und kann sofort beginnen.

---

# Teil C — Die Schritte

## S0 — Festlegungen (kein Code)

**Ziel:** die Entscheidungen treffen, die IRIs und Klassenidentität betreffen.
Werden sie später getroffen, müssen erzeugte Tripel neu geschrieben werden.

**Uploads:** für S0.2 bis S0.4 keine, die Belege stehen unten. Für S0.1 die
fünf EPICA-`.tab` — die Kopfzeilen zeigen, welche Altersskalen tatsächlich
ausgeliefert werden, und davon hängt ab, ob native und umgerechnete Werte
nebeneinander gespeichert werden müssen.

**Ergebnis:** Tabelle A4 ausgefüllt.

### S0.1 Altersskala

Der wichtigste Befund. Die Umrechnung auf eine gemeinsame Achse steckt derzeit
ausschliesslich in Code:

- `wdttest-epica/py/main.py`, `AGE_TRANSFORMS`: `ch4`, `d18o`, `do2n2` →
  `age × 1000 + 50`; `dust` → `age × 1000 − 8`; `dd` → unverändert
- `sisal-db-v3/py/export_sites.py`: Alter werden als BP exportiert, das
  Plotting addiert 50
- `data.yaml` deklariert `age_scale` nur als Freitext (`"EDC2 gas age"`,
  `"AICC2023"`, `"EDC3"`)
- `geo-lod/SISAL/plot_sisal_from_csv.py`: `MIS_INTERVALS`, hartcodiert in ka BP
  nach LR04, mit einer Warm/Kalt/Interstadial-Einstufung, die in keiner der
  MIS-CSVs steht

Damit ist die Chronologie-Information nicht FAIR: weder maschinenlesbar noch mit
dem Rohdatum verknüpft.

**Befund aus den PANGAEA-Kopfzeilen** (geprüft 2026-08-08):

| id | Alterspalte(n) im `.tab` | Einheit | Skala laut Header |
|---|---|---|---|
| ch4 | Gas age (EDC1) **und** Gas age (EDC2) | ka BP | EDC1 + EDC2 |
| d18o | **Gas** age | ka BP | AICC2023 |
| do2n2 | **Ice** age | ka BP | AICC2023 |
| dd | AGE (GEOCODE), dazu Age min/max | ka BP | EDC2 („years before 1950") |
| dust | Age model | ka (ohne BP) | EDC3 |

- Keiner der fünf Datensätze liefert b2k; alle sind BP. Die Annahme, `dd` sei
  bereits b2k, ist falsch — `dd` ist ka BP auf EDC2, wie `ch4`. Die fehlende
  Umrechnung in `AGE_TRANSFORMS` ist ein Fehler, kein Sonderfall.
- Der Offset `−8` bei `dust` ist durch den Header nicht gedeckt und verschiebt
  den Datensatz um 58 Jahre gegen die übrigen vier.
- `ch4` liefert zwei Altersspalten; die Wahl EDC2 existiert derzeit nur als
  Spaltenindex im Code. Die EDC1-Spalte fällt still weg.
- `d18o` trägt Gasalter, `do2n2` Eisalter — beide „AICC2023", aber nicht
  dieselbe Achse. Ein einzelner Chronologie-Knoten „AICC2023" würde die
  Δage-Differenz wegmodellieren.
- Die Bouchet-Datensätze nennen −75.102/123.395, die übrigen −75.100/123.350:
  zwei Bohrpunkte rund 200 m auseinander unter einem `geolod:DrillingSite`.
- Die Header tragen deutlich mehr Provenienz als `data.yaml`: ORCID der PI,
  `METHOD/DEVICE`, Lizenz-URI, Event und Kampagne, Förderung. Das bestätigt die
  Festlegung in S2, die `.tab` als Quelle zu nehmen.

Zu entscheiden:

- Welcher Wert wird als Alter gespeichert — das native (mit deklarierter Skala),
  das umgerechnete, oder beide?
- Welche Skala ist kanonisch? `geolod:ageKaBP` sagt ka BP, `strat:ageB2k` sagt
  Jahre b2k, ELSA-23 ist b2k. Vorschlag: b2k als kanonischer numerischer Wert,
  `[ka]` als Darstellungseinheit über QUDT.
- `data.yaml` bekommt pro Eintrag einen expliziten Offset/Faktor; die Lambdas
  lesen ihn von dort. Damit landet die Umrechnung im TTL statt im Code.

**Beschluss (2026-08-08).** Kanonisch ist **ka BP**. Jede Beobachtung trägt genau
einen Alterswert; ein zusätzliches natives Literal entfällt. Das ist deshalb
tragbar, weil alle fünf EPICA-Datensätze bereits ka BP liefern — für EPICA
findet gar keine Umrechnung mehr statt, und `AGE_TRANSFORMS` entfällt
vollständig. Betroffen sind nur die anderen Stränge: SISAL rechnet Jahre BP
durch 1000, ELSA-23 rechnet b2k über `(b2k − 50) / 1000`.

Was das **nicht** bedeutet: Die Chronologie verschwindet nicht. Jede Beobachtung
verweist weiterhin auf ihren `geolod:Chronology`-Knoten (EDC1, EDC2, EDC3,
AICC2023), und dieser trägt die Umrechnungsregel. Ohne diesen Verweis wären die
fünf Proxies numerisch vergleichbar, aber sachlich falsch gleichgesetzt.

Folgt daraus für die Modellierung:

- `geolod:ageKaBP` bleibt die Property; `strat:ageB2k` wird in S4 dagegen
  aufgelöst, nicht umgekehrt.
- Der Offset `−8` bei `dust` und die `+50` bei den übrigen entfallen ersatzlos.
- `d18o` (Gasalter) und `do2n2` (Eisalter) brauchen trotz gemeinsamer Skala
  AICC2023 zwei unterscheidbare Chronologie-Knoten oder ein Attribut am
  Alterswert. Sonst geht die Δage-Differenz verloren.
- Für `ch4` liegen zwei Alterswerte in ka BP vor, EDC1 und EDC2. Beide bleiben
  erhalten, jeweils mit Verweis auf ihre Chronologie — es sind konkurrierende
  Zuweisungen, kein nativ-gegen-umgerechnet-Fall. Dasselbe Muster wie bei den
  MIS-Grenzen.

**Darstellungskonvention.** Achsen tragen ausschliesslich `Age [ka]`, ohne den
Zusatz BP oder b2k — in allen Abbildungen, auch in den bereits für den
ECEASST-Beitrag erzeugten.

**Abfragbarkeit.** Die Beschriftung darf nicht dazu führen, dass der Bezugspunkt
nur noch im Fliesstext steht. Umgekehrt soll eine Query den Wert in `[ka]`
nehmen können, wie er dasteht, ohne Faktor oder Offset. Beides zusammen leistet
`time:TimePosition` aus OWL-Time:

```turtle
geolod:obs_… geolod:hasAgePosition [
    a                     time:TimePosition ;
    time:numericPosition  "142.7"^^xsd:decimal ;
    time:hasTRS           <…noch festzulegen…> ;
    qudt:unit             unit:KiloYR
] .
```

`time:numericPosition` trägt die Zahl in ka, `time:hasTRS` den Bezugspunkt. Eine
SPARQL-Abfrage filtert direkt auf `time:numericPosition`; wer den Datumsbezug
braucht, liest ihn am TRS ab. Damit ist die Achsenbeschriftung `Age [ka]` keine
Auslassung mehr, sondern die korrekte Kurzform.

`geolod:ageKaBP` bleibt als einfaches Literal daneben bestehen, damit bestehende
Queries nicht brechen; das TimePosition-Objekt ist die vollständige Form. Beide
Formen werden dauerhaft geschrieben, keine wird als `owl:deprecated` markiert.

**TRS: Recherchebefund (2026-08-08).** OWL-Time definiert `time:TRS` nur als
Stub-Klasse und legt die Definition ausdrücklich ausserhalb des eigenen Umfangs.
Festgelegt ist im Standard allein der Gregorianische Kalender über
`http://www.opengis.net/def/uom/ISO-8601/0/Gregorian`. Für tiefe Zeit nennt das
Beispiel im Standard `http://www.opengis.net/def/crs/OGC/0/ChronometricGeologicTime`
— eine Skala in Millionen Jahren, rückwärts positiv.

Sie wird **nicht** übernommen. Ihre Einheit ist Ma, nicht ka; sie zu verwenden
hiesse, entweder in Ma zu speichern und die Abfrage in `[ka]` wieder zu einem
Rechenschritt zu machen, oder sie gegen ihre eigene Definition zu benutzen.
Ausserdem beschreibt sie einen einzigen Deep-Time-Bezug, während hier je
Chronologie eine TRS gebraucht wird; ein Register auf dieser Granularität
existiert nicht — EDC2, EDC3 und AICC2023 sind nirgends als TRS publiziert.

**Beschluss.** Eigene TRS-Instanzen unter `http://w3id.org/geo-lod/trs/`, eine
je Chronologie: `trs/EDC1`, `trs/EDC2`, `trs/EDC3`, `trs/AICC2023`, später die
SISAL- und ELSA-Chronologien. Jede ist `a time:TRS`, deklariert Ursprung (1950),
Einheit (ka) und Richtung (rückwärts positiv) und trägt ihre Quelle über
`dct:source`. `ChronometricGeologicTime` wird per `rdfs:seeAlso` erwähnt, nicht
per `owl:sameAs` gleichgesetzt — die Einheiten unterscheiden sich.

Damit fällt der Chronologie-Knoten aus S0.1 mit der TRS zusammen: eine
Chronologie *ist* ein temporales Bezugssystem, kein separates Objekt daneben.
`geolod:Chronology` wird entsprechend als Unterklasse von `time:TRS` geführt.
Das löst nebenbei die Gas-/Eisalter-Frage: `AICC2023` liefert zwei Achsen und
bekommt zwei TRS, `trs/AICC2023-gas` und `trs/AICC2023-ice`.

### S0.2 CRMarchaeo-Namensraum

Kollision: `wdttest-tables/metadata/ontology/crm_bridge.ttl` verwendet
`http://www.ics.forth.gr/isl/CRMarchaeo/`, geo-lod verwendet
`http://www.cidoc-crm.org/extensions/crmarchaeo/`. `strat:StratigraphicUnit ⊑ A2`
und `geolod:CIArchaeologicalSite ⊑ A2` sind derzeit zwei verschiedene Klassen.
Die geo-lod-Variante ist die richtige; `crm_bridge.ttl` und
`GeoScience-FAIRification-LOD/ontology/README.md` nachziehen.

**Entschieden (2026-08-08):** `http://www.cidoc-crm.org/extensions/crmarchaeo/`.
Der Nachzug in `crm_bridge.ttl` gehört zu S4, der in `ontology/README.md` kann
sofort erfolgen.

### S0.3 MIS

Substadien nach **Railsback et al. 2015** (Zeitbasis BP), Stadiengrenzen nach
**Lisiecki & Raymo 2005 (LR04)**.

Die Quellen widersprechen sich, und das ist kein Fehler: LR04 setzt MIS 5/6 auf
130 000, Railsback MIS 5e/6a auf 132 200. Das wird **nicht harmonisiert**,
sondern als zwei konkurrierende Zuweisungen modelliert —
`crm:E13_Attribute_Assignment` mit `dct:source`, dasselbe Muster wie
`strat:AgeControlPoint`.

Zu prüfen: `mis_stage_boundaries.csv` deklariert für die LR04-Werte
`timebase = b2k`, `mis_literature.csv` für Railsback `BP`. LR04 wird
konventionell in BP angegeben — vermutlich ein Übertragungsfehler.

**Geprüft (2026-08-08):** Beide Originalquellen sind BP —
`LR04_MISboundaries.csv` führt die Spalte als `Age(ka)` in der LR04-Konvention,
Railsback Tabelle als `(Yrs BP)`. Die `b2k`-Deklaration für LR04 ist damit ein
Übertragungsfehler und wird korrigiert. Railsback merkt in der Kopfzeile selbst
an, dass die Substadien Exkursionen bezeichnen und nicht die Grenzen zwischen
ihnen — die Grenzwerte der Tabelle sind entsprechend unscharf und sollten nicht
als exakte Zeitpunkte modelliert werden.

**Beschluss (2026-08-08).**

- Grenzen bleiben zwei konkurrierende E13 je Quelle, ohne Harmonisierung.
- Die MIS-Zuweisung an Beobachtungen wird **materialisiert**, ebenfalls als E13
  je Quelle. Damit wird das Schema tatsächlich konsumiert (S1) und die
  Zugehörigkeit ist ohne Intervall-Join abfragbar. Preis: bei rund 4 900 EPICA-
  plus den SISAL-Beobachtungen verdoppelt jede zusätzliche Quelle die
  Zuweisungstripel. Der Generator muss die Zuweisung deterministisch erzeugen,
  sonst bricht die Byte-Identität zweier Läufe.
- Die Warm/Kalt-Einstufung wird eine Property im Schema und braucht einen
  Literaturbeleg. Die Einstufung in `MIS_INTERVALS` hat derzeit keinen. Als
  Beleg kommt die Paritätskonvention in Frage — ungerade Stadien warm, gerade
  kalt — die Railsback et al. 2015 in ihrer climatostratigraphischen Diskussion
  behandelt. Beim Befüllen ist der hartcodierte Bestand gegen die Konvention
  abzugleichen; die Abweichungen sind der interessante Teil, insbesondere die
  Einstufung „Interstadial" bei Substadien.
- Die Unschärfe der Railsback-Grenzen wird am Schema vermerkt, nicht
  weggerundet.

### S0.4 Ort der Instanzdaten

geo-lod legt Klassen und Instanzen flach in denselben Namensraum
(`geolod:Cave` neben `geolod:Cave_site_0275`). Die WD1-Familie trennt sauber.
Umzug des Bestands wäre breaking; Vorschlag: Bestand belassen, alles Neue unter
`http://w3id.org/geo-lod/vocab/…` bzw. `…/elsa/…` und die Trennung ab hier
durchhalten.

**Beschluss (2026-08-08).** Drei Ebenen:

| Ebene | Muster |
|---|---|
| Klassen und Properties | `http://w3id.org/geo-lod/` — flach, unverändert |
| Kontrollierte Vokabulare | `http://w3id.org/geo-lod/vocab/<scheme>/` |
| Instanzdaten | ein Zweig je Strang: `…/epica/`, `…/sisal/`, `…/ci/`, `…/elsa/` |

Der Bestand zieht mit — einmalig breaking, dafür ohne dauerhafte Zweiteilung.
EPICA und SISAL werden in S2 und S3c ohnehin neu erzeugt und tragen das neue
Muster von Anfang an. Echter Umzug sind damit nur der CI-Strang und die 305
`Cave_site_NNNN`.

**Nachzuhalten beim Umzug:** die alten IRIs sind unter w3id bereits publiziert
und stehen im ECEASST-Beitrag sowie potenziell im N4O-KG. Der Umzug braucht
deshalb eine Abbildung alt → neu, die mit ausgeliefert wird, statt eines stillen
Austauschs. Ob das als `owl:sameAs`, `dct:replaces` oder als
w3id-Weiterleitung gelöst wird, entscheidet S4 zusammen mit der
Content-Negotiation.

---

## S1 — Gemeinsame Vokabulare

**Ziel:** übergreifende SKOS-Schemata, auf die alle Stränge zeigen. Muss vor den
Instanzdaten stehen.

**Uploads:** geo-lod-Bundle (A5). Die WD1-CSVs `mis_literature.csv`,
`mis_stage_boundaries.csv` und `mis_boundaries.csv` wurden nicht gebraucht:
gebaut wird aus den Primärquellen `data/raw/mis/LR04_MISboundaries.csv` und
`data/raw/mis/Railsbacketal2015MISSubstagesFig3-TableVersion01-CSV.csv`, dieselbe
Begründung wie in S2 für die `.tab` gegenüber den abgeleiteten CSVs.

**Ergebnis:** `ontology/vocab/mis.ttl` in geo-lod, IRI-Muster dokumentiert.

**Fertig, wenn:** das MIS-Schema durch SHACL läuft und `wdttest-tables` per
`skos:exactMatch` darauf zeigen könnte.

**Erledigt 2026-08-08.** 315 Konzepte (228 Stadien, 87 Substadien), 792
Zuweisungen, 461 Zeitpositionen, 9 806 Tripel. CRM-Coverage 14/14, SHACL ohne
Violation, zwei Läufe byte-identisch. Neu im Repo:

| Datei | Rolle |
|---|---|
| `ontology/build_mis_vocab.py` | Generator, liest `info/`, schreibt beide TTL |
| `ontology/vocab/mis.ttl` | erzeugt, nicht von Hand editieren |
| `ontology/vocab/README.md` | IRI-Muster `…/vocab/<scheme>/` und Leitschema-Regel |
| `ontology/trs.ttl` | erzeugt; `trs:LR04` und `trs:Railsback2015` |
| `dist/mis_stages.csv` | erzeugt; eine Zeile je Konzept, leitende Werte |
| `dist/mis_assignments.csv` | erzeugt; eine Zeile je Zuweisung, beide Lesarten |
| `data/raw/mis/` | die drei Quelldateien aus `info/`, unverändert |
| `ontology/shapes/mis_shapes.ttl` | Shapes für Konzepte, Zuweisungen, Zeitpositionen |

Geändert: `geo_lod_utils.py` (MIS-Terme im Kern), `crm_bridging.ttl` (SKOS und
OWL-Time verankert), `ontology/README.md` (Crosswalk), `main.py` (Schritt 3
erzeugt die Vokabulare, danach ist alles um eins verschoben), `bundle_rdf.py`
(`ontology/vocab/` kommt ins Bundle).

**Wie mit zwei Quellen umgegangen wird.** Nichts wird verworfen, aber die
Auswahl steht in den Daten statt im Kopf des Konsumenten. Jede Zuweisung trägt
`geolod:assignmentStatus`, jedes Konzept `geolod:leadingSource`:

- Wo Railsback reicht, führt Railsback. Die LR04-Lesart derselben Grenze bleibt
  als `geolod:AlternativeAssignment` daneben stehen — 55 der 792 Zuweisungen.
- Jenseits von 1013,1 ka führt LR04, mangels Alternative.
- Wo nur eine Quelle etwas zu einer Eigenschaft sagt, etwa die Exkursionsgipfel
  von LR04, führt diese.

Ein Filter auf `geolod:LeadingAssignment` liefert damit genau eine Altersachse,
ohne dass die Abfrage die Reichweite der Quellen kennen muss. Die Grenze selbst
ist ebenfalls Datum: `geolod:coverageOldestAgeKaBP` an
`mis:source_railsback2015`.

**Warum zwei Tabellen und nicht eine.** Für Abbildungen und Achsencode ist die
breite Form richtig: eine Zeile je Konzept, nur die leitenden Werte, direkt als
Ersatz für `MIS_INTERVALS` lesbar. Für alles, was die Uneinigkeit der Quellen
sehen muss, ist sie falsch, weil sie sie wegwirft. Die lange Form daneben
kostet fast nichts und hält den Graphen und die Tabellen deckungsgleich: keine
Grenze wird stromabwärts noch einmal berechnet.

**Abgleich `MIS_INTERVALS`.** Genau eine Abweichung von der Parität: MIS 3
steht im Plotting als Interstadial, die Konvention sagt warm. Die übrigen zwölf
Einträge stimmen. Die Abweichung hängt als `skos:note` an `mis:MIS_3`.

**Folge für die Abbildungen (A2b).** Die MIS-Bänder verschieben sich, weil
`MIS_INTERVALS` auf LR04 steht und das Leitschema jetzt Railsback ist: 14 → 14,5;
29 → 35; 57 → 57,3; 71 → 72,7; 130 → 132,2. MIS 3 wandert am stärksten. Für die
Überarbeitung des Beitrags vermerkt.

Gepflegt wird in `GeoScience-FAIRification-LOD`, unter `ontology/vocab/` neben
`geo_lod_core.ttl` und `crm_bridging.ttl`. Alle anderen Repos zeigen darauf,
statt eigene Kopien zu halten — dieselbe Begründung, die `crm_bridge.ttl` schon
für den Kern anführt.

- `http://w3id.org/geo-lod/vocab/mis/` — Stadien und Substadien, Grenzen als E13
  mit Quellenangabe (Railsback / LR04).
- IRI-Muster für weitere Schemata festlegen: `…/vocab/<scheme>/`.
- **Voraussetzung:** `http://w3id.org/geo-lod/` muss per Content Negotiation das
  TTL ausliefern. Solange der `owl:imports` in `crm_bridge.ttl` ins Leere zeigt,
  hält jeder Konsument doch eine lokale Kopie — und der Drift, den die Aufteilung
  vermeiden soll, kommt hintenrum zurück.
- `…/vocab/tephra/` wird nur reserviert, nicht befüllt. Es ist der Join zwischen
  WD1, ELSA und dem CI-Strang und gehört in S5.
- **Wer konsumiert das Vokabular?** Ohne eine MIS-Zuweisung an Beobachtungen in
  S2 oder S3c bliebe das Schema unbenutzt. Die dritte hartcodierte Stelle,
  `MIS_INTERVALS` in `SISAL/plot_sisal_from_csv.py`, trägt zusätzlich eine
  Warm/Kalt-Einstufung — die gehört als Eigenschaft ins Schema, nicht ins
  Plotting.

**Nicht in diesem Chat:** Tephra-Vokabular befüllen, Instanzdaten.

**Offen geblieben:** Content Negotiation unter `http://w3id.org/geo-lod/` ist
weiterhin nicht eingerichtet. Das blockiert das Vokabular nicht, aber solange
sie fehlt, bleibt die Aussage unbelegt, dass andere Repos darauf zeigen statt
zu kopieren. Gehört zu S4.

---

## S2 — EPICA nach RDF

**Ziel:** alle fünf Proxies direkt aus den `.tab`-Dateien nach RDF, vollständig
in geo-lod.

**Uploads:** geo-lod-Bundle (A5); die fünf `.tab` aus `wdttest-epica/data/`;
`wdttest-epica/data.yaml`. Die `.tab` sind gitignoriert und müssen von der
Platte kommen.

**Ergebnis:** neues Sub-Skript im geo-lod-`main.py`, Ausgabe nach `EPICA/rdf/`,
dazu mehrteilige Abbildungen neben den bestehenden Einzeldateien.

**Fertig, wenn:** die Pipeline grün durchläuft, SHACL sauber ist und zwei Läufe
byte-identisch sind.

**Stand 2026-08-09: erledigt.** Pipeline grün, 0 SHACL-
Violations, 48 erzeugte Dateien über zwei Läufe byte-identisch. Bundle
354 969 Tripel (vorher 207 591), davon EPICA 187 622. Was entstanden ist:

| Datei | Rolle |
|---|---|
| `data/raw/epica/*.tab` | die fünf PANGAEA-Dateien, unverändert |
| `EPICA/epica_data.py` | ein Ladeweg für Generator und Plots, plus Provenienzmanifest |
| `EPICA/epica_rdf.py` | der Generator, Ausgabe nach `EPICA/rdf/` |
| `EPICA/plot_epica_from_tab.py` | nur noch Abbildungen; RDF-Hälfte entfernt |

4 904 Beobachtungen, 5 587 `time:TimePosition` (die Differenz sind die 683
EDC1-Alter des CH₄-Datensatzes), 4 904 Stadien-Zugehörigkeiten, 814
`geolod:SampleSection` aus dem δD-Datensatz, 81 Grenzen in Tiefe.

Entfallen: `EPICA/*.tab` und `src/EPICA_Dome_C_*.csv` (Rohdaten jetzt unter
`data/raw/epica/`), `EPICA/rdf/geo_lod_core.ttl` (das doppelte Schreiben aus
Teil D, für EPICA erledigt), `geolod:PANGAEA_CH4_Source` und
`geolod:PANGAEA_d18O_Source` (ersetzt durch `epica:source_*` mit DOI als
`dct:source` — damit sind zwei der drei offenen DOI-Todos erledigt).

**Was der Graph jetzt zeigt und vorher nicht konnte.** Der Beginn von MIS 5
(132,2 ka BP nach Railsback) liegt je nach Datensatz bei 1733,94 m
(AICC2023-ice), 1739,87 m (EDC3-ice), 1755,63 m (EDC2-ice), 1759,64 m
(AICC2023-gas) oder 1782,26 m (EDC2-gas). Die drei Eisalter-Skalen liegen
beisammen, die beiden Gasalter-Skalen systematisch tiefer — das ist die
Gas-Eis-Altersdifferenz, und sie steht jetzt abfragbar im Graphen statt in
einer Fussnote. Das ist ein Ergebnis für die Überarbeitung des Beitrags.

**Abbildungen.** Die zwölf Einzeldateien bleiben, dazu sieben Tafeln aus
`EPICA/epica_plates.py`, je als SVG und JPG: `plate_columns_*` und
`plate_rows_*` in den drei Glättungsvarianten, dazu `plate_boundary_depths`.
Letztere hat zwei Hälften, weil eine nicht reicht — links die Tiefen-Alters-
Beziehung als Kontext, auf deren 3200-m-Achse der Unterschied zwischen den
Modellen unsichtbar bleibt, rechts die Abweichung jeder Kurve vom Mittel
derselben Grenze, auf einer Achse von Zehnermetern. Bezugsgrösse ist das
Mittel über die Datensätze, die eine Grenze überhaupt abdecken, nicht ein
gewählter Referenzdatensatz: es gibt hier keine Wahrheit, nur Modelle, die
voneinander abweichen. Insgesamt 150 erzeugte Dateien, über zwei Läufe
byte-identisch — 30 Einzelabbildungen, sieben Tafeln und zwei Collagen, je als
SVG und JPG.

**Nachlese zum ersten Auslieferungslauf.** Drei Fehler sind erst am Ergebnis
auf Windows aufgefallen und am 2026-08-09 behoben: die CRLF-Zeilenenden in
allen SVG, die feste δ¹⁸O-Tick-Liste, die 6 % der Messwerte aus der Abbildung
schnitt, und die drei Datensätze ohne jede Einzelabbildung. Beim Übernehmen
sind ausserdem `EPICA/EDC_CH4.tab`, `EPICA/EPICA_Dome_C_d18O.tab` und
`EPICA/rdf/geo_lod_core.ttl` zu löschen; `py main.py --clean` erledigt die
letzte.

- Die fünf `.tab` ziehen nach geo-lod um und ersetzen dort die bisherigen
  Rohdaten (`src/EPICA_Dome_C_*.csv`, `EPICA/*.tab`). `wdttest-epica` behält
  seine eigene Kopie — die Repos sind unabhängig.
- Quelle sind die `.tab`, nicht die abgeleiteten CSVs — die PANGAEA-Header
  tragen Provenienz, die im CSV verloren geht.
- Neu gegenüber dem Bestand: `dD`, `dust`, `dO2/N2` (bisher nur CH₄ und δ¹⁸O).
- Chronologien als eigene Knoten: EDC2, EDC3, AICC2023 als `geolod:Chronology`,
  jede Beobachtung verweist auf die ihre. Damit wird sichtbar, dass die fünf
  Proxies *nicht* auf derselben Zeitachse liegen — genau das, was die Lambdas
  heute stillschweigend wegrechnen.
- Der bestehende `geolod:DrillingSite` für Dome C bleibt Ankerpunkt.
- Die Provenienz aus `data.yaml` (Creator, DOI, `age_scale`, Lizenz je Datensatz)
  wandert mit und ersetzt die unvollständigen `DataSource`-Instanzen in
  `geo_lod_utils.py`. Damit erledigt sich der offene Todo, DOIs als
  `dct:source` zu ergänzen.

**Abbildungen: Collagen statt nur Einzeldateien.** Bisher entsteht je Proxy,
Achse und Glättung eine eigene Datei, bei fünf Proxies wird das unübersichtlich.
Gebraucht werden zusätzlich mehrteilige Tafeln, die den Vergleich zeigen, den
Einzelbilder nicht leisten: mehrere Proxies übereinander auf gemeinsamer
Altersachse, dieselbe Grösse in verschiedenen Chronologien nebeneinander,
geglättet gegen ungeglättet. Welche genau, wird zu Beginn des S2-Chats
entschieden, wenn die fünf `.tab` und ihre Wertebereiche vorliegen — vorher
lässt sich nicht sagen, was nebeneinander überhaupt lesbar ist.

Was dabei gilt: die Einzeldateien bleiben, die Collagen kommen dazu. Sie tragen
dieselben MIS-Bänder aus `dist/mis_stages.csv` wie die SISAL-Abbildungen, damit
Eiskern und Speläotheme dieselbe Stratigraphie zeigen, und sie sind wie alles
andere byte-reproduzierbar.

**MIS-Bänder auch in den Tiefenplots.** Tiefenplots sind bisher chronologiefrei,
obwohl die Tiefe die gemessene Grösse ist und das Alter das Abgeleitete. Die
Bänder geben der Tiefenachse eine indirekte Altersinformation. Dabei ist zu
beachten:

- Eine MIS-Grenze in Metern ist keine Beobachtung, sondern eine Interpolation
  im Tiefen-Alters-Modell. Sie hängt vollständig an der benutzten Chronologie.
- Die fünf Proxies liegen nicht auf derselben: CH₄ auf EDC1/EDC2, δ¹⁸O und
  dO₂/N₂ auf AICC2023, dust auf EDC3. Dieselbe Grenze landet je nach Kurve in
  anderer Tiefe — auf einer Tafel mit mehreren Proxies wird das sichtbar, und
  genau das ist erwünscht.
- An jeder Tafel mit Bändern steht, welche Chronologie die Umrechnung geliefert
  hat. Ohne diese Angabe ist die Abbildung nicht interpretierbar.
- Grenzen jenseits des tiefsten Messpunktes werden nicht extrapoliert.

**Zu entscheiden in S2:** ob die Grenze-in-Tiefe auch ins RDF geht, als weitere
`geolod:MISAttributeAssignment` mit der Chronologie als Quelle. Konsequente
Fortsetzung des Musters, verdoppelt aber die Zuweisungen je Kern und
Chronologie.

**Nicht in diesem Chat:** `wdttest-epica` anfassen.

---

## S3a — SISAL: DDL von MySQL nach Postgres

**Ziel:** das vollständige SISAL-v3-Schema als reviewbares
`postgres/schema.sql` — 21 Tabellen mit Typen, Primärschlüsseln,
Fremdschlüsseln und Constraints, lauffähig auf Postgres 13.

**Uploads:** ein `sisal_bundle.zip`, erzeugt mit den Befehlen unten. Inhalt:
`ddl.sql`, `alter.sql`, `dump_header.txt`, `csv_info.txt` und das Repo ohne die
grossen Dateien. Sollte deutlich unter 1 MB liegen.

**Nicht hochladen:** die CSVs, den vollen SQL-Dump, `config.ini`.

### Bundle erzeugen (Windows cmd)

```cmd
cd /d C:\pfad\zum\sisal-release
mkdir bundle
```

DDL aus dem MySQL-Dump. Nicht „alles vor dem ersten INSERT" — MySQL-Dumps
verschachteln Struktur und Daten pro Tabelle. Der Befehl schneidet jeden Block
von `CREATE TABLE` bis `) ENGINE=` heraus und liest zeilenweise:

```cmd
powershell -NoProfile -Command "$o=New-Object System.Collections.Generic.List[string]; $in=$false; foreach($l in [IO.File]::ReadLines((Resolve-Path 'sisalv3.sql'))){ if($l -match '^CREATE TABLE'){$in=$true}; if($in){$o.Add($l)}; if($in -and $l -match '^\) ENGINE'){$in=$false; $o.Add('')} }; Set-Content -Path 'bundle\ddl.sql' -Value $o -Encoding UTF8"
```

Constraints, die manche Dumps ans Ende stellen, plus Dump-Kopf:

```cmd
findstr /b /c:"ALTER TABLE" /c:"CREATE INDEX" sisalv3.sql > bundle\alter.sql
powershell -NoProfile -Command "Get-Content 'sisalv3.sql' -TotalCount 25 | Set-Content 'bundle\dump_header.txt' -Encoding UTF8"
```

CSV-Kopfzeilen, Zeilenzahlen und je drei Beispielzeilen. Die Beispielzeilen
zeigen, wie NULL, Anführungszeichen und Dezimaltrenner tatsächlich aussehen —
das entscheidet über die `COPY`-Optionen:

```cmd
powershell -NoProfile -Command "Get-ChildItem 'sisalv3_csv\*.csv' | ForEach-Object { $n=$_.Name; $c=0; $h=''; $s=New-Object System.Collections.Generic.List[string]; foreach($l in [IO.File]::ReadLines($_.FullName)){ if($c -eq 0){$h=$l} elseif($c -le 3){$s.Add($l)}; $c++ }; '=== '+$n+'  ('+$c+' Zeilen)'; 'HEADER: '+$h; foreach($x in $s){'SAMPLE: '+$x}; '' } | Set-Content 'bundle\csv_info.txt' -Encoding UTF8"
```

Das Repo ohne die grossen Dateien. Robocopy meldet Exitcode 1, wenn es Dateien
kopiert hat — das ist Erfolg, kein Fehler:

```cmd
robocopy C:\git\sisal-db-v3 bundle\sisal-db-v3 /E /XD sisalv3_csv __pycache__ .git .venv /XF *.sql config.ini
```

Packen:

```cmd
powershell -NoProfile -Command "Compress-Archive -Path 'bundle\*' -DestinationPath 'sisal_bundle.zip' -Force"
```

Falls `ddl.sql` leer bleibt, hat der Dump die Tabellen anders eingeleitet — dann
stattdessen die ersten 3000 Zeilen schicken.

**Ergebnis:** `postgres/schema.sql`, plus eine kurze Notiz zur
Namenszuordnung MySQL → Postgres.

**Fertig, wenn:** das Schema auf einer leeren Datenbank ohne Fehler durchläuft.

**Warum DDL aus dem Dump und Daten aus den CSVs.** Nur der Dump trägt Typen,
Schlüssel, `NOT NULL` und `ON DELETE CASCADE` — genau das, was den CSVs fehlt;
ohne ihn müsste die Integrität aus dem ER-Diagramm rekonstruiert werden. Die
143 MB `INSERT INTO … VALUES` werden dagegen nicht angefasst: langsamer als
`COPY`, und ein Konverter zwischen Quelle und Datenbank wäre eine neue Stelle
für genau die Fehlerklasse, gegen die das Repo gebaut wurde. Die Werte sind
lange Float-Literale (`0.0344083333333334`); ein verlustbehafteter
Zwischenschritt fiele dort nicht auf.

**Das Nicht-Triviale an der Übersetzung:**

- `int(10) unsigned` → `integer` genügt (`sample_id` liegt bei rund 475 000),
  `double` → `double precision`.
- `ENGINE`, `CHARSET`, `COLLATE`, die `/*!40101 */`-Kommentare und `LOCK TABLES`
  entfallen. `KEY` wird zu `CREATE INDEX`.
- Die `enum`-Spalten sind der eigentliche Aufwand. Als `CHECK (col IN (…))`
  abbilden statt `CREATE TYPE … AS ENUM` — leichter zu laden und leichter zu
  lockern, falls die CSVs Werte enthalten, die das Schema nicht deklariert.
- Bezeichner konsequent kleinschreiben (`Ba_Ca` → `ba_ca`, `d18O` → `d18o`).
  Dauerhaft gequotete Bezeichner sind in jeder späteren Query lästig; der
  bestehende Code macht es bei `d18o`/`d13c` ohnehin so. Zuordnung dokumentieren.
- Vollständig heisst auch spaltenvollständig. Derzeit lädt `build_database.py`
  nur die Spalten, die das Manuskript braucht.

**Nicht in diesem Chat:** Loader, Guard, Ladereihenfolge, Aufräumarbeiten. Das
Schema muss stehen, bevor Ladelogik dagegen geschrieben wird — sonst diskutiert
man beides gegen ein bewegliches Ziel.

---

## S3b — SISAL: Loader, Guard, Aufräumen

**Ziel:** `sisal-db-v3` restauriert die vollständige Datenbank mit
referenzieller Integrität und ist danach der einzige Zugriffsweg auf SISAL —
für geo-lod wie für die WD1-Repos.

**Uploads:** `sisal-db-v3` als ZIP ohne `data/sisalv3_csv/` und ohne
`postgres/sisal-v3.sql`; das in S3a entstandene `postgres/schema.sql`; die
Zeilenzahlen je CSV (`wc -l` bzw. unter Windows `find /c /v ""`).

**Ergebnis:** angepasstes `build_database.py`, neuer Guard, aufgeräumtes Repo.

**Fertig, wenn:** ein Lauf von leerer Datenbank bis geladener v3-DB durchläuft
und der Guard grün meldet.

**Ladeweg.** `COPY ... (FORMAT csv)` aus den Original-CSVs, Ladereihenfolge
entlang der Fremdschlüssel: `site` → `entity` → `sample` → Messtabellen, dann
`reference`, `notes`, Link-Tabellen. Kein pgAdmin-Import; Datenbank anlegen und
hineinschauen gern, laden nicht.

**Was der Guard prüft.** Nicht mehr drei Sites gegen handgepflegte Zahlen,
sondern: Zeilenzahl je Tabelle gegen die CSV-Zeilenzahl, Wertebereiche gegen
`PLAUSIBLE_RANGE` als direkten Test auf die 1000×-Signatur, und
Fremdschlüssel-Integrität. Site-bezogene Sollwerte bleiben möglich, sind aber
optional und nicht mehr WD1-fest.

**Erwartbarer Nebenbefund.** Beim Aktivieren der Fremdschlüssel werden
wahrscheinlich Waisen auffallen, die im Sechs-Tabellen-Ausschnitt gar nicht
sichtbar sein konnten. Das ist ein Gewinn, kein Problem — aber es braucht eine
Entscheidung, ob solche Zeilen abgewiesen oder protokolliert und geladen werden.

**Datenhaltung.** Entpackte 127 MB werden nicht ausgeliefert. Entweder
`sisalv3_csv.zip` (36 MB) mitliefern und beim Build entpacken, oder von ORA
ziehen. Damit erledigt sich auch die Frage nach dem alten 43-MB-Dump.

**Anpassungen am Repo.** Das Repo war bisher auf die drei WD1-Sites und sechs
Tabellen zugeschnitten. Mit dem vollständigen Restore ändert sich der Zweck, und
diese Punkte ziehen mit:

- **`postgres/sisal-v3.sql` (43 MB) entfällt.** Es ist ein `pg_dump` der
  geladenen Datenbank mit denselben sechs Tabellen, die `build_database.py`
  ohnehin aus den CSVs baut — dieselben Daten also zweimal, einmal als Quelle
  plus Code, einmal als Ergebnis. Nach dem Umbau ist der Build der Restore-Pfad,
  damit hat der Dump keine Rolle mehr. Er ist ausserdem an eine
  Werkzeugversion gebunden (mit `pg_dump` 18.0 aus Postgres 13.9 gezogen, mit
  `\restrict`-Token). Falls eine Abkürzung ohne vollen Ladelauf gewünscht ist,
  gehört sie als Release-Asset oder nach Zenodo — 43 MB in der Git-Historie
  bleiben dauerhaft darin.
- **`postgres/datamodel.gml` / `datamodel.jpg` ersetzen.** Der DbVis-Export
  zeigt den Sechs-Tabellen-Ausschnitt; nach dem Restore beschreibt er nicht mehr
  das, was in der Datenbank steht. Das offizielle ER-Diagramm aus der
  ESSD-Veröffentlichung deckt das vollständige Modell ab. Für ein Diagramm ist
  ausserdem ein Vektorformat oder PNG besser geeignet als JPEG.
- **Datenbankname auf v3 ziehen.** Die Datenbank heisst an allen Stellen
  `sisal-v2`, hält aber v3-Daten: `config.ini`, `config.example.ini`,
  README-Diagramm, `main.py`-Docstring, Phasenbeschreibung. Solange nur ein
  Ausschnitt geladen war, fiel das nicht ins Gewicht; mit dem vollen Restore ist
  eine eindeutige Benennung wichtiger.
- **Repo-Umbenennung nachziehen.** `main.py`, der Strukturblock im README und
  `CITATION.cff` tragen noch `wdttest-sisal-db-v3`; `repository-code` zeigt auf
  einen Platzhalter. Der richtige Wert steht jetzt fest. Die verbleibenden
  `TODO`-Marker in der `.cff` (ORCIDs, Releasedatum) werden bei der Gelegenheit
  mit erledigt.
- **Site-Auswahl parametrisieren.** `data/derived/` und `EXPECTED_SITES`
  kodieren die drei WD1-Sites fest, `FIGURE_WINDOW` das Altersfenster der
  Abbildungen — der Kommentar dort merkt selbst an, dass das Fenster ins
  Plot-Repo gehört. Für ein Repo, das beide Familien bedient, werden daraus
  Parameter. Die `.gitignore` sieht den Fall bereits vor.
- **`py/__pycache__/`** steht in `.gitignore`; falls dennoch getrackt,
  `git rm -r --cached`.
- **`config.ini`** ist korrekt git-ignoriert, hat aber das Passwortfeld
  ausgefüllt — genau das, wovor `config.example.ini` warnt. Auf Umgebungs-
  variable oder `pgpass` umstellen.

**Nicht in diesem Chat:** RDF, geo-lod anfassen.

---

## S3c — SISAL nach RDF

**Ziel:** Neuaufbau der SISAL-Tripel aus der restaurierten Datenbank, statt aus
den bestehenden `v_data_*.csv`. Der Generator läuft in geo-lod.

**Uploads:** geo-lod-Bundle (A5); der Exportcode aus `sisal-db-v3`; eine
strukturerhaltende Exportdatei für die gewählten Sites.

**Ergebnis:** neues Sub-Skript im geo-lod-`main.py`, Ausgabe nach `SISAL/rdf/`.

**Fertig, wenn:** die Pipeline grün ist, SHACL sauber, und die drei unten
belegten Fehler nachweislich nicht mehr auftreten.

**Warum Neuaufbau, nicht Weiterrechnen.** Die aktuellen geo-lod-CSVs tragen
genau den Fehler, gegen den `sisal-db-v3` gebaut wurde:

- Sample 328059 (Sanbao): im Release `lin_interp_age_uncert_pos = 1884.137`, in
  `v_data_140_sanbao.csv` steht `1884137.0`. Dezimaltrenner-Signatur, Wert mit
  exakt drei Nachkommastellen, Faktor 1000. Drei betroffene Werte über die vier
  Dateien; die Isotopenwerte selbst sind sauber.
- Zeilenverlust: Buraca Gloriosa hat im Release 1178 Samples mit δ¹⁸O und
  Chronologie, in geo-lod stehen 1137 — 41 fehlen, ohne gesetztes Fenster. Bei
  Sanbao 5832 gegenüber 9535; dort kann zusätzlich ein anderer Join oder eine
  Deduplizierung mitspielen und ist hier zu klären.

**Vorgehen.** Der Code aus `sisal-db-v3` wird nachgenutzt (kopiert, nicht
referenziert): Export aus der Datenbank nach CSV, und diese CSVs sind die Basis
für den geo-lod-Generator. Damit bleibt die Struktur
Site → Entity → Sample → Chronology erhalten, und geo-lod hängt weder an einer
laufenden Datenbank noch am anderen Repo.

- Mitwandern muss der `COPY`-Weg **und** der Zeilenzahl-Abgleich.
- Die Sites hängen an bestehenden `geolod:Cave_site_NNNN`-Knoten (A1).
- `age_uncert_pos` / `age_uncert_neg` stehen in den CSVs bereits, werden aber
  beim TTL-Schreiben verworfen. Kein neuer Datenweg nötig, nur zwei Properties
  mehr im Modell.
- Site-Auswahl nach A4. geo-lod hat Beobachtungen für 140, 144, 145, 275,
  Fionas Auswahl ist 58, 202, 275; die Site-Ebene liegt für alle 305 vor.
- Mit dem vollen Restore kommen Tabellen in Reichweite, die modelliert werden
  wollen: `dating` (U/Th-Datierungen als eigene Messereignisse), `hiatus` und
  `gap` (Lücken im Archiv), `original_chronology` neben `sisal_chronology` als
  konkurrierende Altersmodelle. Dasselbe Muster wie bei den MIS-Grenzen, passt
  zu `strat:AgeControlPoint`. Nicht Pflicht für S3c, aber der Grund, den Restore
  überhaupt vollständig zu machen.

---

## S4 — Ontologie-Angleichung

**Ziel:** `geolod:` und `strat:` so zusammenführen, dass ein gemeinsames Bundle
widerspruchsfrei lädt.

**Uploads:** geo-lod-Bundle (A5); `metadata/ontology/` aus `wdttest-tables`;
die in S2 und S3c erzeugten Ontologie-TTL.

**Ergebnis:** angepasste Ontologiemodule, zusammengeführte SHACL-Shapes.

**Fertig, wenn:** beide Graphen gemeinsam laden, SHACL ohne Verletzungen
durchläuft und keine Klasse doppelt unter zwei Namensräumen existiert.

Erst wenn beide Stränge Tripel liefern, weil sich hier zeigt, was wirklich
kollidiert. Bekannte Punkte:

- `strat:Core` ist nur `crm:E18_Physical_Thing`, aber nicht
  `geolod:PalaeoclimateSample` — obwohl Eiskern, Speläothem und `an:Specimen`
  es sind. Eine Zeile im Bridge.
- `strat:ageKaBP` und `geolod:ageKaBP` existieren doppelt in verschiedenen
  Namensräumen. Nach S0.1 auflösen.
- `strat:Core` hat keinen Link auf eine Locality; `wd1:WD1` ist nirgends
  verortet, obwohl `locality/Walsdorfer_maar` existiert. `strat:drilledAt`
  ergänzen.
- `geolod:CIArchaeologicalSite ⊑ crmarchaeo:A2_Stratigraphic_Volume_Unit` und
  ebenso `geolod:Cave_site_0275` — ein Ort ist keine stratigraphische
  Volumeneinheit. Er *enthält* welche. Umbauen.
- SHACL zusammenführen: `strat`-Shapes und `core_shapes.ttl` in einen Gate.
- Der CI-Strang gehört mit in diesen Schritt, nicht nur EPICA und SISAL. Der
  A2-Umbau betrifft seine Instanzdaten unmittelbar.

---

## S5 — ELSA

**Ziel:** vierter Datenstrang als Integrationstest der Angleichung.

**Uploads:** geo-lod-Bundle (A5); die MDPI-Quaternary-PDFs (Sirocko et al. 2024,
Schenk et al. 2024); Tabelle 1 des ELSA-23-Stacks als CSV, falls verfügbar.

**Ergebnis:** neues Sub-Skript, `…/vocab/tephra/` befüllt, `strat:`-Erweiterung.

**Fertig, wenn:** ELSA lädt, ohne dass an `geolod:` oder `strat:` nachgepatcht
werden muss. Muss doch gepatcht werden, zeigt die Stelle, was in S4 offen blieb.

- Umfang: ELSA-23-Tephra-Stack (Sirocko et al. 2024, Tabelle 1: 14 Marker ×
  9 Kerne mit Alter und Top-/Basis-Tiefe) plus die CI-Kryptotephra aus Auel
  (Schenk et al. 2024). Kein Pollen-Stack, keine Spektralindizes, kein µXRF.
- Ausnahme: die ELSA-20/23-Chronologie kommt mit — sie macht die Tiefen
  überhaupt erst mit EPICA und SISAL vergleichbar.
- `…/vocab/tephra/` wird befüllt und ist der eigentliche Join: WD1, ELSA und der
  CI-Strang zeigen auf dieselben Marker-Concepts. Der Campanian Ignimbrite ist
  selbst eine Marker-Tephra — die bestehenden CI-Tripel müssen hier auf das
  gemeinsame Concept umgehängt werden.
- Maare: Seen mit Geometrie aus OSM (`prov:wasDerivedFrom`, kein `owl:sameAs`
  auf einen Way), `owl:sameAs` nach Wikidata; Trockenmaare über
  `fuzzysl:Georeferencing` mit Sicherheit. Kern-IDs und Marker in der fuzzy-sl
  Wikibase.
- Erweiterung von `strat:`: `strat:TephraCorrelation` als E13, damit dieselbe
  Marker-Tephra über mehrere Kerne korreliert werden kann. Die Allen-Relationen
  sind kernintern; ELSA bringt die Achse quer über Kerne.
- Lizenz: nur die MDPI-CC-BY-Quellen speisen den Graphen, nicht das
  Elsevier-Supplement.

---

# Teil D — Offene Punkte

Nicht einem Schritt zugeordnet, aber nicht zu vergessen:

- ~~Zeitbasis-Auszeichnung in `mis_stage_boundaries.csv` prüfen (S0.3).~~
  Erledigt 2026-08-08: beide Quellen sind BP, die `b2k`-Deklaration ist zu
  korrigieren.
- Sanbao: 5832 gegenüber 9535 Samples im Release. Beim Neuaufbau klären, ob
  Zeilenverlust oder bewusster Join (S3c).
- Abbildung alt → neu für die migrierten CI- und `Cave_site`-IRIs, zusammen mit
  der Content-Negotiation (S4).
- ~~Beleg für die Warm/Kalt-Einstufung suchen und den hartcodierten Bestand in
  `MIS_INTERVALS` dagegen abgleichen (S1).~~ Erledigt 2026-08-08: Parität mit
  Railsback et al. 2015 als Beleg, eine Abweichung (MIS 3).
- `MIS_INTERVALS` in `SISAL/plot_sisal_from_csv.py` durch `dist/mis_stages.csv`
  ersetzen — die letzte hartcodierte MIS-Stelle (S3c).
- ~~Die übrigen Rohdaten nach `data/raw/` nachziehen: `src/EPICA_Dome_C_*.csv`
  und die `.tab` in S2~~ — für EPICA erledigt 2026-08-09. Offen bleiben die
  SISAL-CSVs (S3c) und `CI/cifindspots_part_full.csv` (S4).
- `src/plot_epica_115--250.py` liest fünf CSVs, von denen nur zwei je in `src/`
  lagen; das Skript war schon vorher nicht lauffähig und hängt in keiner
  Pipeline. Entweder auf `EPICA/epica_data.py` umstellen oder löschen — in S2
  bewusst nicht angefasst.
- ~~Die mehrteiligen Tafeln aus S2~~ — erledigt 2026-08-09, sieben Tafeln.
- ~~`MIS_INTERVALS` in `EPICA/plot_epica_from_tab.py`~~ — erledigt 2026-08-09.
  Offen bleibt die Stelle in `SISAL/plot_sisal_from_csv.py` (S3c); danach ist
  keine MIS-Grenze mehr im Code hinterlegt.
- ~~Die Tiefenplots tragen keine MIS-Bänder.~~ Erledigt 2026-08-09.
- Die Fünfer-Collage zeigt nur die Altersachse. Eine Tiefenvariante ist eine
  Zeile in `build_all`, falls der Text sie braucht.
- ~~`epica_style.py` gilt nur für EPICA.~~ Erledigt 2026-08-09: liegt als
  `ontology/geo_lod_figures.py` neben `geo_lod_utils.py`, SISAL nutzt es.
- SISAL hat noch keine `captions.yaml`. Die Mechanik liegt in
  `ontology/geo_lod_captions.py` bereit; einzutragen sind die Unterschriften
  der 36 Abbildungen, das gehört in S3c.
- Die SISAL-Achsen laufen weiter über handgesetzte Grenzen. Ob dort dasselbe
  Abschneiden passiert wie bei δ¹⁸O, ist ungeprüft — beim Umbau in S3c gegen
  `geo_lod_figures.nice_ticks` stellen und die Wertebereiche vergleichen.
- `wdttest-tables` auf Railsback umstellen und per `skos:exactMatch` auf
  `…/vocab/mis/` zeigen lassen (S4). `wdttest-wd1--ager-corg` ist bereits dort.
- Der Bundle-Schritt bleibt der grösste Posten, jetzt aber im Parsen der
  Einzeldateien, nicht mehr in SHACL oder der Serialisierung. Falls das
  irgendwann stört: die Sub-Skripte könnten zusätzlich N-Triples schreiben,
  die das Bundle dann schneller einliest.
- Aus den bestehenden Pipeline-Todos: doppeltes Schreiben von
  `geo_lod_core.ttl` — für EPICA erledigt 2026-08-09, für SISAL offen
  (`SISAL/plot_sisal_from_csv.py` legt weiter eine Kopie in `SISAL/rdf/`);
  `cisite_59` auf fehlenden Pleiades-/Wikidata-Link prüfen. Von den DOIs an den
  `DataSource`-Instanzen sind die beiden PANGAEA-Quellen mit S2 erledigt, offen
  bleibt `SISALv3_DataSource` (S3c).