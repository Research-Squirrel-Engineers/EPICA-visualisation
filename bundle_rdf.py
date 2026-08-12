#!/usr/bin/env python3
"""
bundle_rdf.py - Aggregate ontology + instance RDF into a single distributable
                 Turtle file under dist/, then run validations.

Pipeline-Schritt 5:
  1. Merge alle TTL-Dateien aus ontology/, EPICA/rdf/, SISAL/rdf/, CI/rdf/
     in einen rdflib.Graph -> dist/geo-lod-bundle.ttl
     (mit defensivem Auto-Repair für mehrzeilige String-Literale)
  2. CIDOC-CRM-Coverage: jede getypte Instanz muss (transitiv über
     rdfs:subClassOf) unter einer CRM-Familien-Klasse hängen
     (crm:, crmsci:, crmarchaeo:, crmgeo:)
  3. SHACL-Validierung gegen ontology/shapes/*.ttl bzw. ontology/*-shapes.ttl
  4. Sanity-Checks (Triple-Count, Klassen-/Property-Inventar)

Aufruf direkt:    python bundle_rdf.py
Aufruf über main: ruft run_bundle_step(...) auf.
"""

from __future__ import annotations

import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef
except ImportError as e:
    print(f"  ✗ rdflib nicht verfügbar: {e}")
    print("    Installiere mit:  pip install rdflib")
    raise

# pyshacl ist optional - SHACL-Validierung wird übersprungen, wenn nicht da.
try:
    from pyshacl import validate as shacl_validate

    PYSHACL_AVAILABLE = True
except ImportError:
    PYSHACL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Namespaces - CIDOC-CRM-Familie
# ---------------------------------------------------------------------------
# Strikte CRM-Coverage: ALLE Klassen, auch externe Standardvokabulare
# (SOSA, PROV, GeoSPARQL, DCAT, FOAF, ...), müssen einen CRM-Anker
# bekommen, da das Bundle in den NFDI4Objects Knowledge Graph einfließt.
# Für externe Vokabulare geschieht das per Bridging-Axiom in der eigenen
# Ontologie (siehe ontology/README.md - Crosswalk).

CRM_FAMILY_PREFIXES = (
    "http://www.cidoc-crm.org/cidoc-crm/",
    "http://www.ics.forth.gr/isl/CRMsci/",
    "http://www.ics.forth.gr/isl/CRMgeo/",
    "http://www.ics.forth.gr/isl/CRMarchaeo/",
    "http://www.cidoc-crm.org/extensions/crmsci/",
    "http://www.cidoc-crm.org/extensions/crmgeo/",
    "http://www.cidoc-crm.org/extensions/crmarchaeo/",
)

GEO_LOD = Namespace("http://w3id.org/geo-lod/")


# ---------------------------------------------------------------------------
# TTL Auto-Repair
# ---------------------------------------------------------------------------
# Bekanntes Problem in den Generator-Skripten: mehrzeilige String-Literale
# werden mit normalen "..."-Quotes geschrieben, was Turtle nicht erlaubt.
# Wir reparieren das im Speicher beim Re-Parse - die Quelle muss aber
# trotzdem gefixt werden. Siehe Hinweise im Output und ontology/README.md.

# Matcht: einfaches "..."-Literal, das mindestens ein Newline enthält.
_BROKEN_LITERAL = re.compile(
    r'"((?:[^"\\]|\\.)*?\n(?:[^"\\]|\\.)*?)"',
    re.DOTALL,
)


def _repair_multiline_strings(ttl_text: str) -> tuple[str, int]:
    """Ersetzt einfache "..."-Literale, die Newlines enthalten, durch
    Triple-Quoted-Literale.

    Liefert (repariertes Text, Anzahl Ersetzungen). Konservativ: ersetzt
    nur, wenn der String nicht selbst schon \"\"\" enthält.
    """
    n = 0

    def _repl(m: re.Match) -> str:
        nonlocal n
        content = m.group(1)
        if '"""' in content:
            return m.group(0)  # nicht anfassen
        n += 1
        return f'"""{content}"""'

    return _BROKEN_LITERAL.sub(_repl, ttl_text), n


def _parse_with_repair(graph: Graph, path: Path) -> tuple[int, int, bool]:
    """Versucht erst normales Parsen, fällt bei Fehler auf Auto-Repair zurück.

    Returns: (added_triples, file_triples, repaired_flag)
      - added_triples: wieviele Triples NEU im Graph durch dieses Parsing
      - file_triples:  wieviele Triples die Datei selbst enthält (für Anzeige
                       von Aggregaten, deren Inhalt schon im Bundle ist)
      - repaired_flag: True wenn Auto-Repair greifen musste

    Wirft Exception nur, wenn auch der Repair-Versuch fehlschlägt.
    """
    before = len(graph)
    repaired_flag = False

    try:
        graph.parse(path, format="turtle")
    except Exception as first_err:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8-sig")

        repaired, n_fixes = _repair_multiline_strings(text)
        if n_fixes == 0:
            raise first_err

        graph.parse(data=repaired, format="turtle")
        repaired_flag = True

    added = len(graph) - before

    # Datei isoliert parsen, um die intrinsische Triple-Zahl zu kennen.
    # Das ist nur nötig, wenn 'added' nicht der vollen Datei entspricht
    # (sprich: Set-Duplikate mit bereits geladenem Material).
    # Wir zählen immer mit, weil's billig ist.
    file_graph = Graph()
    try:
        file_graph.parse(path, format="turtle")
    except Exception:
        if repaired_flag:
            file_graph.parse(data=repaired, format="turtle")
        else:
            # Sollte nicht passieren - wenn das erste Parsen klappte, klappt das hier auch
            pass
    file_triples = len(file_graph)

    return added, file_triples, repaired_flag


# ---------------------------------------------------------------------------
# Bundle Builder
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Ausgabeformate
# ---------------------------------------------------------------------------
#
# Gemessen an einem Bundle mit rund 208.000 Tripeln (10 MB Turtle):
#
#   nt          1.4s   30.1 MB   schnell, zeilenweise, gut zu grep-en
#   xml         2.7s   21.3 MB
#   turtle      6.6s   10.4 MB   kompakt und lesbar - das Release-Format
#   longturtle  6.4s   10.6 MB
#   json-ld     8.9s   28.7 MB
#
# Voreinstellung ist Turtle. N-Triples wäre schneller, spart aber gemessen nur
# rund 13 der 187 Sekunden eines vollen Laufs - zu wenig, um dafür in Kauf zu
# nehmen, dass die versionierte .ttl hinter dem Code herhinkt und man an ein
# Flag denken muss. Wer eng iteriert, nimmt --bundle-format nt; für eine
# Veröffentlichung schreibt --bundle-format release alle Formate.

BUNDLE_FORMATS: dict[str, tuple[str, str, str]] = {
    # Schlüssel: (rdflib-Format, Dateiendung, Beschreibung)
    # Achtung: N-Triples ist nicht byte-stabil. rdflib vergibt beim Parsen
    # neue Blank-Node-Labels, und N-Triples schreibt sie im Klartext aus -
    # rund 30 Zeilen mit OWL-Restriktionen aendern sich damit bei jedem Lauf.
    # Deshalb steht dist/geo-lod-bundle.nt in der .gitignore. Turtle bleibt
    # stabil, weil dieselben Knoten dort inline als [ ] serialisiert werden.
    "nt": ("nt", ".nt", "N-Triples - eine Zeile je Tripel, am schnellsten"),
    "turtle": ("turtle", ".ttl", "Turtle - kompakt und lesbar, Release-Format"),
    "longturtle": ("longturtle", ".lttl", "Turtle mit einem Prädikat je Zeile"),
    "xml": ("xml", ".rdf", "RDF/XML - für Werkzeuge, die nichts anderes lesen"),
    "jsonld": ("json-ld", ".jsonld", "JSON-LD - für Web-Clients"),
}

# N-Triples for a development run: writing the 1.2-million-triple bundle as
# Turtle took 373s against roughly 60s here, and nothing in a development run
# reads the bundle with its eyes. Turtle is still written by a release run,
# where RELEASE_BUNDLE_FORMATS asks for all four.
DEFAULT_BUNDLE_FORMATS: tuple[str, ...] = ("nt",)
RELEASE_BUNDLE_FORMATS: tuple[str, ...] = ("nt", "turtle", "jsonld", "xml")

# Aggregatdateien: Vereinigungen von Dateien, die einzeln ohnehin geladen
# werden. Sie tragen nichts zum Bundle bei, kosten aber den vollen Parse.
# Geprüft 2026-08-08: ohne sisal_all_data.ttl bleibt das Bundle bei
# identischer Tripelzahl, der Schritt wird rund ein Drittel schneller.
AGGREGATE_FILENAMES: set[str] = {"sisal_all_data.ttl"}

# Files a development run leaves out. sisal_v3_chronologies.ttl holds the
# 101 470 competing ages from the seven models geo-lod does not follow: 885 000
# triples that no figure and no shape needs, and roughly half the parse time of
# the whole bundle. They belong in a release, not in every run.
#
# Nothing is lost by leaving them out. The age geo-lod follows is materialised
# on the sample in sisal_v3_core.ttl, and the file stands on its own next to
# the bundle for anyone comparing models.
# Matched on the stem, because the same graph is written as .nt in a
# development run and as .ttl in a release one.
RELEASE_ONLY_STEMS: set[str] = {"sisal_v3_chronologies"}

# Instance graphs may arrive in either format. Turtle first so that a stale
# .ttl next to a fresh .nt is still found and reported rather than silently
# ignored; _dedupe_formats then keeps one per stem.
INSTANCE_SUFFIXES: tuple[str, ...] = (".ttl", ".nt")


def _dedupe_formats(paths: list[Path]) -> list[Path]:
    """One file per stem, preferring the format written most recently.

    A release run writes sisal_v3_core.ttl next to the sisal_v3_core.nt a
    development run left behind. Reading both would load every triple twice,
    which rdflib survives and the triple count does not - and a stale one
    would quietly outvote the fresh one.
    """
    by_stem: dict[str, Path] = {}
    for path in paths:
        current = by_stem.get(path.stem)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            by_stem[path.stem] = path
    return [by_stem[stem] for stem in sorted(by_stem)]


def _collect_ttl_files(source_dirs: Iterable[Path],
                       release: bool = False,
                       suffixes: tuple[str, ...] = (".ttl",)) -> list[Path]:
    files: list[Path] = []
    skipped: list[Path] = []
    for d in source_dirs:
        if not d.exists():
            continue
        found: list[Path] = []
        for suffix in suffixes:
            found.extend(sorted(d.glob(f"*{suffix}")))
        for path in _dedupe_formats(found):
            if not release and path.stem in RELEASE_ONLY_STEMS:
                skipped.append(path)
                continue
            files.append(path)
    for path in skipped:
        print(f"  ℹ  {path.name} übersprungen - nur im Release-Bundle. "
              f"Mit --bundle-format release ist es dabei.")
    return files


def _dedupe_core_copies(files: list[Path], canonical_ontology_dir: Path) -> list[Path]:
    """Wenn sub-Skripte eine Kopie von z.B. geo_lod_core.ttl in ihre rdf/-
    Verzeichnisse schreiben, ignorieren wir die - die kanonische Version
    steht in ontology/.
    """
    canonical_names = {p.name for p in canonical_ontology_dir.glob("*.ttl")}

    kept: list[Path] = []
    skipped: list[Path] = []
    for f in files:
        if f.parent == canonical_ontology_dir:
            kept.append(f)
            continue
        if f.name in canonical_names:
            skipped.append(f)
            continue
        kept.append(f)

    if skipped:
        print("    (Duplikate ignoriert - kanonische Version liegt in ontology/):")
        for f in skipped:
            print(f"      ~ {f}")
    return kept


def build_bundle(
    ontology_dir: Path,
    rdf_dirs: Iterable[Path],
    output_path: Path,
    formats: Iterable[str] = DEFAULT_BUNDLE_FORMATS,
) -> Graph | None:
    """Merge alle TTLs in einen Graph, schreibe nach output_path."""
    print("\n  ▶ Sammle TTL-Quellen ...")

    # ontology/*.ttl plus die kontrollierten Vokabulare in ontology/vocab/.
    # ontology/shapes/ bleibt draussen - Shapes gehören nicht in den Datengraph.
    # "Release" is read off the formats rather than passed as a second switch:
    # a run that asks for the release formats is the release run, and two
    # independent flags would eventually disagree.
    release = set(formats) >= set(RELEASE_BUNDLE_FORMATS)
    ontology_files = _collect_ttl_files([ontology_dir, ontology_dir / "vocab"])
    instance_files = _collect_ttl_files(rdf_dirs, release=release,
                                        suffixes=INSTANCE_SUFFIXES)

    print(f"    Ontologie-Dateien: {len(ontology_files)}")
    for f in ontology_files:
        try:
            print(f"      • {f.relative_to(ontology_dir.parent)}")
        except ValueError:
            print(f"      • {f}")

    instance_files = _dedupe_core_copies(instance_files, ontology_dir)

    aggregates = [f for f in instance_files if f.name in AGGREGATE_FILENAMES]
    if aggregates:
        print("    (Aggregate übersprungen - Inhalt liegt schon in den Einzeldateien):")
        for f in aggregates:
            print(f"      ~ {f.name}")
        instance_files = [f for f in instance_files if f.name not in AGGREGATE_FILENAMES]

    print(f"    Instanz-Dateien:   {len(instance_files)}")
    for f in instance_files:
        try:
            print(f"      • {f.relative_to(ontology_dir.parent)}")
        except ValueError:
            print(f"      • {f}")

    all_files = ontology_files + instance_files
    if not all_files:
        print("  ✗ Keine TTL-Dateien gefunden - Bundle nicht erstellt.")
        return None

    g = Graph()
    g.bind("crm", Namespace(CRM_FAMILY_PREFIXES[0]))
    g.bind("crmsci", Namespace("http://www.ics.forth.gr/isl/CRMsci/"))
    g.bind("crmgeo", Namespace("http://www.ics.forth.gr/isl/CRMgeo/"))
    g.bind("crmarchaeo", Namespace("http://www.ics.forth.gr/isl/CRMarchaeo/"))
    g.bind("geo-lod", GEO_LOD)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)

    print("\n  ▶ Parse & merge ...")
    parse_errors: list[tuple[Path, str]] = []
    repaired_files: list[Path] = []

    for f in all_files:
        try:
            added, file_triples, was_repaired = _parse_with_repair(g, f)
            tag = "  (auto-repaired)" if was_repaired else ""
            # Wenn alle Triples neu sind: nur eine Zahl zeigen.
            # Wenn die Datei mehr Triples enthält als sie zum Bundle beiträgt
            # (= Aggregat / Set-Duplikate), beide Zahlen zeigen.
            if added == file_triples:
                print(f"    + {f.name}: {file_triples:>8,} Triples{tag}")
            else:
                print(
                    f"    + {f.name}: {file_triples:>8,} Triples "
                    f"({added:,} new, {file_triples - added:,} dup){tag}"
                )
            if was_repaired:
                repaired_files.append(f)
        except Exception as e:
            short = str(e).splitlines()[0][:120]
            print(f"    ✗ Fehler beim Parsen von {f.name}: {short}")
            parse_errors.append((f, str(e)))

    if repaired_files:
        print()
        print("  ⚠  Auto-Repair angewendet - FIX IN GENERATOR-SKRIPTEN NÖTIG:")
        for f in repaired_files:
            print(f"      • {f}")
        print('    → Mehrzeilige String-Literale müssen mit """..."""')
        print('      statt "..." geschrieben werden (Turtle-Spec).')

    if parse_errors:
        print(f"\n  ⚠  {len(parse_errors)} Datei(en) konnten nicht geparst werden.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n  ▶ Serialisiere Bundle ({len(g):,} Triples) ...")

    written: set[str] = set()

    for key in formats:
        if key not in BUNDLE_FORMATS:
            print(f"    ⚠  Unbekanntes Format übersprungen: {key}")
            continue
        rdflib_format, suffix, description = BUNDLE_FORMATS[key]
        target = output_path.with_suffix(suffix)
        t_start = time.perf_counter()
        g.serialize(destination=target, format=rdflib_format, encoding="utf-8")
        size_mb = target.stat().st_size / 1048576
        print(
            f"  ✓ {target.name:<26} {size_mb:6.1f} MB  "
            f"{time.perf_counter() - t_start:5.1f}s  ({description})"
        )
        written.add(target.name)

    # Bundles aus früheren Läufen, die diesmal nicht neu geschrieben wurden,
    # sind jetzt veraltet. Stillschweigend liegen lassen wäre die gefährlichere
    # Variante: sie sehen aktuell aus und sind es nicht.
    stale = sorted(
        f.name
        for f in output_path.parent.glob(output_path.stem + ".*")
        if f.name not in written
    )
    if stale:
        print("\n  ℹ  Aus früheren Läufen, in diesem Lauf nicht neu geschrieben:")
        for name in stale:
            print(f"      • {name}")
        print("     Vor einer Veröffentlichung: python main.py --bundle-format release")

    return g


# ---------------------------------------------------------------------------
# Validation: CIDOC-CRM Coverage (strict)
# ---------------------------------------------------------------------------


def _is_crm_class(uri: URIRef) -> bool:
    s = str(uri)
    return any(s.startswith(p) for p in CRM_FAMILY_PREFIXES)


def validate_crm_coverage(g: Graph) -> bool:
    """Strikt: jede getypte Instanz muss über die rdfs:subClassOf-Schließung
    unter einer Klasse aus der CIDOC-CRM-Familie hängen
    (crm: / crmsci: / crmgeo: / crmarchaeo:).

    Externe Vokabulare (SOSA, PROV, GeoSPARQL, DCAT, FOAF, ...) werden NICHT
    als 'ok' gewertet - sie brauchen ein Bridging-Axiom in der eigenen
    Ontologie. Siehe ontology/README.md (Crosswalk).
    """
    print("\n  ▶ CIDOC-CRM Coverage-Check (strict CRM-family) ...")

    instance_classes: set[URIRef] = {
        o for _, _, o in g.triples((None, RDF.type, None)) if isinstance(o, URIRef)
    }

    skip = {
        OWL.Class,
        OWL.NamedIndividual,
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        OWL.Ontology,
        OWL.AnnotationProperty,
        OWL.Restriction,
        OWL.AllDisjointClasses,
        OWL.AllDifferent,
        RDFS.Class,
        RDFS.Datatype,
        RDF.Property,
        RDF.List,
        # Eigenschaftscharakteristiken. Sie stehen als zweiter rdf:type neben
        # owl:ObjectProperty und sagen aus, wie eine Property sich verhält -
        # keine Entität, die einen CRM-Vorfahren haben könnte. Aufgefallen mit
        # S3c.3: strat.ttl deklariert die Allen-Relationen auf der Tiefenachse
        # als transitiv beziehungsweise symmetrisch, und der Check las das als
        # fünf Instanzen zweier nicht verankerter Klassen.
        OWL.TransitiveProperty,
        OWL.SymmetricProperty,
        OWL.AsymmetricProperty,
        OWL.ReflexiveProperty,
        OWL.IrreflexiveProperty,
        OWL.FunctionalProperty,
        OWL.InverseFunctionalProperty,
        OWL.DeprecatedClass,
        OWL.DeprecatedProperty,
    }
    instance_classes -= skip

    if not instance_classes:
        print("    ⚠  Keine getypten Instanzen gefunden - Check übersprungen.")
        return True

    def ancestors(cls: URIRef, seen: set[URIRef] | None = None) -> set[URIRef]:
        if seen is None:
            seen = set()
        for parent in g.objects(cls, RDFS.subClassOf):
            if isinstance(parent, URIRef) and parent not in seen:
                seen.add(parent)
                ancestors(parent, seen)
        return seen

    covered: list[URIRef] = []
    uncovered: list[URIRef] = []
    for cls in sorted(instance_classes):
        chain = {cls} | ancestors(cls)
        if any(_is_crm_class(c) for c in chain):
            covered.append(cls)
        else:
            uncovered.append(cls)

    print(f"    Instanz-Klassen geprüft: {len(instance_classes)}")
    print(f"    ✓ mit CRM-Anker:         {len(covered)}")
    print(f"    ✗ ohne CRM-Anker:        {len(uncovered)}")

    if uncovered:
        print("\n    Klassen ohne CRM-Anker (Crosswalk in ontology/README.md):")
        with_counts = sorted(
            ((cls, sum(1 for _ in g.subjects(RDF.type, cls))) for cls in uncovered),
            key=lambda x: (-x[1], str(x[0])),
        )
        for cls, n_inst in with_counts:
            print(f"      • {cls}  ({n_inst} Instanz/en)")
        return False

    return True


# ---------------------------------------------------------------------------
# Validation: SHACL
# ---------------------------------------------------------------------------


def _find_shape_files(ontology_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    shapes_dir = ontology_dir / "shapes"
    if shapes_dir.exists():
        candidates.extend(sorted(shapes_dir.glob("*.ttl")))
    candidates.extend(sorted(ontology_dir.glob("*-shapes.ttl")))
    candidates.extend(sorted(ontology_dir.glob("*_shapes.ttl")))
    seen: set[Path] = set()
    unique: list[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def validate_shacl(g: Graph, ontology_dir: Path) -> bool:
    print("\n  ▶ SHACL-Validierung ...")

    if not PYSHACL_AVAILABLE:
        print("    ⚠  pyshacl nicht installiert - übersprungen.")
        print("       Installation:  pip install pyshacl")
        return True

    shape_files = _find_shape_files(ontology_dir)
    if not shape_files:
        print("    ⚠  Keine Shape-Dateien gefunden - übersprungen.")
        print(f"       Erwartet z.B. unter: {ontology_dir / 'shapes'}/*.ttl")
        return True

    shapes_graph = Graph()
    for sf in shape_files:
        print(f"    + lade Shapes: {sf.name}")
        shapes_graph.parse(sf, format="turtle")

    # inference="none", nicht "rdfs": der RDFS-Reasoner über 200k Triples
    # kostete zwei Drittel der Laufzeit des Schrittes (42 s gegenüber 14 s)
    # und ändert am Ergebnis nichts. SHACL folgt rdfs:subClassOf bei
    # sh:targetClass und sh:class von sich aus; die Subklassen-Axiome müssen
    # dafür nur im Bundle liegen, und das tun sie (geo_lod_core.ttl,
    # crm_bridging.ttl). Geprüft 2026-08-08: identische Ergebnisse.
    t_start = time.perf_counter()
    conforms, results_graph, results_text = shacl_validate(
        data_graph=g,
        shacl_graph=shapes_graph,
        ont_graph=None,
        inference="none",
        abort_on_first=False,
        meta_shacl=False,
        advanced=True,
        debug=False,
    )
    print(f"    Dauer: {time.perf_counter() - t_start:.1f}s")

    if conforms:
        print("    ✓ Alle SHACL-Shapes erfüllt.")
        return True

    # Conforms=False kann durch Violations ODER Warnings ausgelöst werden.
    # Nur sh:Violation soll das Bundle blockieren - Warnings sind Audit-Hinweise.
    SH = Namespace("http://www.w3.org/ns/shacl#")
    n_violations = 0
    n_warnings = 0
    n_other = 0
    for r in results_graph.subjects(RDF.type, SH.ValidationResult):
        sev = next(results_graph.objects(r, SH.resultSeverity), None)
        if sev == SH.Violation:
            n_violations += 1
        elif sev == SH.Warning:
            n_warnings += 1
        else:
            n_other += 1

    # Output: erst Übersicht, dann die Details aus pyshacl
    print(
        f"    SHACL-Ergebnis: "
        f"{n_violations} Violation(s), "
        f"{n_warnings} Warning(s)" + (f", {n_other} sonstige" if n_other else "")
    )
    for line in results_text.splitlines():
        print(f"      {line}")

    if n_violations > 0:
        # Hard fail - es gibt strukturelle Probleme im Bundle
        return False

    # Nur Warnings: Bundle gilt als gültig, Hinweise sind Audit-Material
    print()
    print("    ⚠  Nur Warnings - Bundle gilt als gültig.")
    return True


# ---------------------------------------------------------------------------
# Sanity Checks
# ---------------------------------------------------------------------------


def sanity_report(g: Graph) -> None:
    print("\n  ▶ Sanity-Report ...")

    n_triples = len(g)
    n_subjects = len(set(g.subjects()))
    classes = Counter(
        o for _, _, o in g.triples((None, RDF.type, None)) if isinstance(o, URIRef)
    )
    properties = Counter(p for _, p, _ in g)

    print(f"    Triples:         {n_triples:,}")
    print(f"    Subjekte:        {n_subjects:,}")
    print(f"    Distinkte Typen: {len(classes):,}")
    print(f"    Distinkte Props: {len(properties):,}")

    print("\n    Top-10 Klassen nach Instanzzahl:")
    for cls, count in classes.most_common(10):
        print(f"      {count:>6,}  {cls}")

    print("\n    Top-10 Properties nach Vorkommen:")
    for prop, count in properties.most_common(10):
        print(f"      {count:>6,}  {prop}")


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------


def run_bundle_step(
    script_dir: Path,
    ontology_dir: Path,
    rdf_dirs: Iterable[Path],
    dist_dir: Path | None = None,
    formats: Iterable[str] = DEFAULT_BUNDLE_FORMATS,
) -> bool:
    """Bauen + validieren. True nur wenn Bundle gebaut UND alle harten
    Checks ok (CRM-Coverage, SHACL).
    """
    if dist_dir is None:
        dist_dir = script_dir / "dist"

    # Basisname ohne Endung; build_bundle hängt je Format die richtige an.
    bundle_path = dist_dir / "geo-lod-bundle.ttl"
    g = build_bundle(ontology_dir, rdf_dirs, bundle_path, formats=formats)
    if g is None:
        return False

    crm_ok = validate_crm_coverage(g)
    shacl_ok = validate_shacl(g, ontology_dir)
    sanity_report(g)

    print()
    print(f"    CRM-Coverage:   {'✓' if crm_ok   else '✗'}")
    print(f"    SHACL:          {'✓' if shacl_ok else '✗'}")
    return crm_ok and shacl_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> int:
    here = Path(__file__).parent.absolute()
    ontology_dir = here / "ontology"
    rdf_dirs = [
        here / "EPICA" / "rdf",
        here / "SISAL" / "rdf",
        here / "CI" / "rdf",
    ]
    ok = run_bundle_step(here, ontology_dir, rdf_dirs, formats=DEFAULT_BUNDLE_FORMATS)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main())
