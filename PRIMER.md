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

## A3. Querschnittsregeln

- **`metadata/ontology.ttl` und `metadata/shapes.ttl` sind tabu.** Das ist die
  gemeinsame `wdt:`-FDO-Schicht der WD1-Familie, verbatim über alle Repos,
  zentral versioniert. Sie ist etwas anderes als die geo-lod-Ontologie und wird
  nicht mit ihr vermischt. Der fachliche Semantic Layer gehört dort ins
  Verzeichnis `metadata/ontology/` — so macht es `wdttest-tables` bereits.
- Nachnutzung heisst kopieren, nicht referenzieren. Einzige Ausnahme ist die
  Ontologie über ihre w3id-IRI.
- Kopierter Code trägt seine Prüfungen mit. Ein Guard, der beim Kopieren
  wegfällt, ist die gefährlichste Form von Nachnutzung.
- **Kein Spreadsheet und kein Import-Assistent im Ladeweg.** Das gilt auch für
  den pgAdmin-Importdialog. `COPY ... (FORMAT csv)` ist der einzige Pfad.
- **Zweimal laufen lassen, `git status` muss sauber bleiben.** Tripel sortiert
  ausgeben, keine zufälligen Blank-Node-IDs.
- Ein Thema pro Chat, ein Repo pro Chat.
- Sprache: Konversation deutsch, Code/Ontologie/Dokumentation englisch.
  Ausnahme: dieses `PRIMER.md` bleibt deutsch — es ist ein internes
  Arbeitsdokument, das offen liegt, aber nicht nach aussen adressiert ist.

## A4. Beschlusslage

Wird nach S0 gefüllt und danach nur noch fortgeschrieben. Alle späteren Schritte
lesen hier ab, statt neu zu diskutieren.

| Frage | Beschluss | seit |
|---|---|---|
| Kanonische Altersskala | offen | |
| Natives Alter zusätzlich speichern? | offen | |
| `crmarchaeo:`-Namensraum | offen | |
| MIS: Grenzen, Warm/Kalt-Einstufung, wer sie zugewiesen bekommt | offen | |
| IRI-Muster Instanzdaten | offen | |
| Auslieferung `sisalv3_csv.zip` vs. Download | offen | |
| SISAL-Site-Auswahl für RDF | offen | |
| Waisen bei FK-Aktivierung: abweisen oder laden | offen | |
| PRIMER.md-Sprache | deutsch — internes Arbeitsdokument | 2026-08 |

---

# Teil B — Schrittübersicht

| ID | Schritt | Repo | hängt ab von | Status |
|---|---|---|---|---|
| S0 | Festlegungen, kein Code | — | — | offen |
| S1 | Gemeinsame Vokabulare | geo-lod | S0 | offen |
| S2 | EPICA nach RDF | geo-lod | S0, S1 | offen |
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

**Uploads:** keine. Die Belege stehen unten.

**Ergebnis:** Tabelle A4 ausgefüllt.

### S0.1 Altersskala

Der wichtigste Befund. Die Umrechnung auf eine gemeinsame Achse steckt derzeit
ausschliesslich in Code:

- `wdttest-epica/py/main.py`, `AGE_TRANSFORMS`: `ch4`, `d18o`, `do2n2` →
  `age × 1000 + 50`; `dust` → `age × 1000 − 8`; `dd` → unverändert (bereits b2k)
- `sisal-db-v3/py/export_sites.py`: Alter werden als BP exportiert, das
  Plotting addiert 50
- `data.yaml` deklariert `age_scale` nur als Freitext (`"EDC2 gas age"`,
  `"AICC2023"`, `"EDC3"`)
- `geo-lod/SISAL/plot_sisal_from_csv.py`: `MIS_INTERVALS`, hartcodiert in ka BP
  nach LR04, mit einer Warm/Kalt/Interstadial-Einstufung, die in keiner der
  MIS-CSVs steht

Damit ist die Chronologie-Information nicht FAIR: weder maschinenlesbar noch mit
dem Rohdatum verknüpft. Zu entscheiden:

- Welcher Wert wird als Alter gespeichert — das native (mit deklarierter Skala),
  das umgerechnete, oder beide?
- Welche Skala ist kanonisch? `geolod:ageKaBP` sagt ka BP, `strat:ageB2k` sagt
  Jahre b2k, ELSA-23 ist b2k. Vorschlag: b2k als kanonischer numerischer Wert,
  `[ka]` als Darstellungseinheit über QUDT.
- `data.yaml` bekommt pro Eintrag einen expliziten Offset/Faktor; die Lambdas
  lesen ihn von dort. Damit landet die Umrechnung im TTL statt im Code.

### S0.2 CRMarchaeo-Namensraum

Kollision: `wdttest-tables/metadata/ontology/crm_bridge.ttl` verwendet
`http://www.ics.forth.gr/isl/CRMarchaeo/`, geo-lod verwendet
`http://www.cidoc-crm.org/extensions/crmarchaeo/`. `strat:StratigraphicUnit ⊑ A2`
und `geolod:CIArchaeologicalSite ⊑ A2` sind derzeit zwei verschiedene Klassen.
Die geo-lod-Variante ist die richtige; `crm_bridge.ttl` und
`GeoScience-FAIRification-LOD/ontology/README.md` nachziehen.

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

### S0.4 Ort der Instanzdaten

geo-lod legt Klassen und Instanzen flach in denselben Namensraum
(`geolod:Cave` neben `geolod:Cave_site_0275`). Die WD1-Familie trennt sauber.
Umzug des Bestands wäre breaking; Vorschlag: Bestand belassen, alles Neue unter
`http://w3id.org/geo-lod/vocab/…` bzw. `…/elsa/…` und die Trennung ab hier
durchhalten.

---

## S1 — Gemeinsame Vokabulare

**Ziel:** übergreifende SKOS-Schemata, auf die alle Stränge zeigen. Muss vor den
Instanzdaten stehen.

**Uploads:** `ontology/` aus geo-lod; `mis_literature.csv`,
`mis_stage_boundaries.csv`, `mis_boundaries.csv`.

**Ergebnis:** `ontology/vocab/mis.ttl` in geo-lod, IRI-Muster dokumentiert.

**Fertig, wenn:** das MIS-Schema durch SHACL läuft und `wdttest-tables` per
`skos:exactMatch` darauf zeigen könnte.

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

---

## S2 — EPICA nach RDF

**Ziel:** alle fünf Proxies direkt aus den `.tab`-Dateien nach RDF, vollständig
in geo-lod.

**Uploads:** geo-lod-Repo (mindestens `EPICA/`, `ontology/`, `main.py`,
`geo_lod_utils.py`); die fünf `.tab` aus `wdttest-epica/data/`;
`wdttest-epica/data.yaml`.

**Ergebnis:** neues Sub-Skript im geo-lod-`main.py`, Ausgabe nach `EPICA/rdf/`.

**Fertig, wenn:** die Pipeline grün durchläuft, SHACL sauber ist und zwei Läufe
byte-identisch sind.

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

**Nicht in diesem Chat:** Abbildungen, `wdttest-epica` anfassen.

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

**Uploads:** geo-lod-Repo (mindestens `SISAL/`, `ontology/`, `main.py`,
`geo_lod_utils.py`); der Exportcode aus `sisal-db-v3`; eine
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

**Uploads:** `ontology/` aus geo-lod; `metadata/ontology/` aus `wdttest-tables`;
die in S2 und S3c erzeugten TTL.

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

**Uploads:** geo-lod-Repo; die MDPI-Quaternary-PDFs (Sirocko et al. 2024,
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

- Zeitbasis-Auszeichnung in `mis_stage_boundaries.csv` prüfen (S0.3).
- Sanbao: 5832 gegenüber 9535 Samples im Release. Beim Neuaufbau klären, ob
  Zeilenverlust oder bewusster Join (S3c).
- Aus den bestehenden Pipeline-Todos: doppeltes Schreiben von
  `geo_lod_core.ttl` in den Unterskripten entfernen; `cisite_59` auf fehlenden
  Pleiades-/Wikidata-Link prüfen. Die DOIs an den `DataSource`-Instanzen
  erledigt S2 mit.