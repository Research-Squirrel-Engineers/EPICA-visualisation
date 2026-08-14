#!/usr/bin/env python3
"""
main.py - the geo-lod pipeline: EPICA, SISAL, CI, the archaeological
HTML pages, and the RDF bundle, with a run log over all of them.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import time

import clean

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
EPICA_RDF_SCRIPT  = SCRIPT_DIR / "EPICA" / "epica_rdf.py"
EPICA_PLOT_SCRIPT = SCRIPT_DIR / "EPICA" / "plot_epica_from_tab.py"
SISAL_RDF_SCRIPT = SCRIPT_DIR / "SISAL" / "sisal_rdf.py"
SISAL_SCRIPT = SCRIPT_DIR / "SISAL" / "plot_sisal_from_csv.py"
CI_RDF_SCRIPT  = SCRIPT_DIR / "CI" / "ci_pipeline.py"
CI_PLOT_SCRIPT = SCRIPT_DIR / "CI" / "plot_ci_findspots.py"
ARCHAEO_SCRIPTS = [SCRIPT_DIR / "archaeo-connect" / "ci_findspots_html.py"]
DOCS_CHECK_SCRIPT = SCRIPT_DIR / "check_docs.py"
EPICA_MAP_SCRIPT = SCRIPT_DIR / "EPICA" / "plot_epica_map.py"
SISAL_MAP_SCRIPT = SCRIPT_DIR / "SISAL" / "plot_sisal_maps.py"
OVERVIEW_SCRIPT = SCRIPT_DIR / "plot_overview_map.py"
ONTOLOGY_DIR = SCRIPT_DIR / "ontology"
DIST_DIR     = SCRIPT_DIR / "dist"

EPICA_RDF_DIR = SCRIPT_DIR / "EPICA" / "rdf"
SISAL_RDF_DIR = SCRIPT_DIR / "SISAL" / "rdf"
CI_RDF_DIR = SCRIPT_DIR / "CI" / "rdf"

# Global log file
LOG_FILE = SCRIPT_DIR / "pipeline_report.txt"


class TeeOutput:
    """Writes to both stdout and a file"""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        # newline="\n": sonst schreibt Python auf Windows CRLF, während Git
        # die Datei nach .gitattributes als LF speichert - die Arbeitskopie
        # wiche dann dauerhaft von ihrer eigenen abgelegten Form ab.
        self.log = open(filepath, "w", encoding="utf-8", newline="\n")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def print_header(text: str, char: str = "=", width: int = 80):
    print()
    print(char * width)
    print(text.center(width))
    print(char * width)
    print()


# Laufzeit je Schritt. print_section startet die Uhr, end_section hält sie an;
# print_summary zeigt daraus, wo die Zeit geblieben ist.
STEP_TIMES: list[tuple[str, float]] = []
_current_step: list = []


def print_section(text: str):
    print()
    print("─" * 80)
    print(f"  {text}")
    print("─" * 80)
    print(f"  Start: {datetime.now().strftime('%H:%M:%S')}")
    _current_step.clear()
    _current_step.extend([text, time.perf_counter()])


def end_section() -> None:
    """Hält die Uhr des laufenden Abschnitts an und meldet die Dauer."""
    if not _current_step:
        return
    label, started = _current_step[0], _current_step[1]
    duration = time.perf_counter() - started
    STEP_TIMES.append((label, duration))
    print(f"\n  ⏱  {label}: {duration:.1f}s")
    _current_step.clear()


def check_file_exists(filepath: Path, description: str) -> bool:
    if not filepath.exists():
        print(f"  ⚠  {description} not found: {filepath}")
        return False
    print(f"  ✓ {description} found: {filepath.name}")
    return True


def check_directory_exists(dirpath: Path, description: str) -> bool:
    if not dirpath.exists():
        print(f"  ⚠  {description} not found: {dirpath}")
        return False
    print(f"  ✓ {description} found: {dirpath.name}/")
    return True


#: The strands main.py can run, in the order it runs them. Each has an
#: --only switch of its own, and naming them once is what keeps the switch,
#: the clean group and the summary from drifting apart - the old code spelled
#: the same three conditions out at four call sites.
STRANDS = ("epica", "sisal", "ci", "archaeo")

#: Filled in from --dpi before the first step and handed to every drawing
#: script. Empty means the scripts fall back to their own default, which is
#: draft.
RASTER_QUALITY: dict[str, str] = {}


class StepNumber:
    """Hands out the next section number.

    The numbers used to be written into the section titles by hand, and every
    step inserted in the middle meant renumbering the ones after it - which
    was done wrongly at least once. A counter cannot drift from the order the
    steps actually run in.
    """

    def __init__(self):
        self.value = -1

    def __call__(self, title: str) -> str:
        self.value += 1
        return f"{self.value}. {title}"


step = StepNumber()


def strand_requested(args, strand: str) -> bool:
    """True unless another strand's --only switch excludes this one."""
    only = [name for name in STRANDS
            if getattr(args, f"{name}_only", False)]
    return not only or strand in only


def clean_groups_for(args) -> list[str]:
    """Which clean groups belong to the steps this run will actually execute.

    Scoped deliberately: `--sisal-only` must not remove the EPICA figures.
    They would not be redrawn in that run, and half an hour of plotting would
    be gone for nothing. Steps 2 and 3 always run, so their groups are always
    in.

    The four Mermaid files hang on EPICA alone. They used to be written by
    both strands, but the SISAL side went with the RDF export in S3c.2, and
    while this said "epica or sisal" a --sisal-only run deleted them and left
    them deleted.
    """
    groups = ["vocab", "cache"]
    if strand_requested(args, "epica"):
        groups.append("epica")
        groups.append("diagrams")
    if strand_requested(args, "sisal"):
        groups.append("sisal")
    if strand_requested(args, "ci"):
        groups.append("ci")
    if strand_requested(args, "archaeo"):
        groups.append("archaeo")
    if not args.no_overview:
        groups.append("overview")
    if not args.no_bundle:
        groups.append("bundle")
    return groups


def clean_before_run(args) -> None:
    """Remove what this run is about to rewrite, and name what it will not.

    Sweeping first is what keeps an orphan out of the repository. A figure
    whose proxy has been renamed is not overwritten by the next run - it is
    simply never written again, and without the sweep it would stay in
    plots/ indistinguishable from a current one. The inventory itself lives
    in clean.py, together with the two registers this step does not touch.
    """
    print_section(step("Clean generated outputs"))
    groups = clean_groups_for(args)
    print(f"  Groups: {', '.join(groups)}")
    result = clean.sweep(groups, delete=True, verbose=True)
    if result.removed:
        print(f"\n  Removed {result.removed} item(s), "
              f"{clean.human(result.bytes_freed)} freed.")
    else:
        print("  Nothing to remove.")
    end_section()


def regenerate_canonical_ontology() -> bool:
    """Schritt 0: Schreibt die kanonischen Ontologie-Dateien aus
    geo_lod_utils.py nach ontology/. Single Source of Truth - die Sub-
    Skripte sollen keine eigenen Kopien mehr in ihre rdf/-Verzeichnisse
    legen.

    Aktuell wird nur ontology/geo_lod_core.ttl regeneriert; weitere
    TTL-Konstanten (z.B. EPICA_ONTOLOGY_TTL, SISAL_ONTOLOGY_TTL) können
    hier ergänzt werden, sobald sie nach geo_lod_utils.py gewandert sind.
    """
    print("\n  ▶ Regeneriere kanonische Ontologie-Dateien aus geo_lod_utils.py ...")

    # geo_lod_utils.py liegt in ONTOLOGY_DIR
    sys.path.insert(0, str(ONTOLOGY_DIR))
    try:
        from geo_lod_utils import GEO_LOD_CORE_TTL
    except ImportError as e:
        print(f"  ✗ geo_lod_utils.py konnte nicht importiert werden: {e}")
        return False

    ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    target = ONTOLOGY_DIR / "geo_lod_core.ttl"
    try:
        target.write_text(GEO_LOD_CORE_TTL, encoding="utf-8")
        size_kb = target.stat().st_size / 1024
        print(f"  ✓ {target.name} geschrieben ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"  ✗ Fehler beim Schreiben von {target}: {e}")
        return False


def regenerate_vocabularies() -> bool:
    """Schritt 0b: Erzeugt die kontrollierten Vokabulare unter
    ontology/vocab/ aus den Primärquellen in data/raw/.

    Nebenprodukt sind die aufbereiteten Tabellen in dist/ - dieselben Werte
    wie im TTL, für Abbildungen und Achsencode.

    Aktuell nur das MIS-Vokabular (plus ontology/trs.ttl); weitere Schemata
    (z.B. tephra) kommen hier dazu.
    """
    print("\n  ▶ Regeneriere kontrollierte Vokabulare aus data/raw/ ...")

    sys.path.insert(0, str(ONTOLOGY_DIR))
    try:
        from build_mis_vocab import build as build_mis
    except ImportError as e:
        print(f"  ✗ build_mis_vocab.py konnte nicht importiert werden: {e}")
        return False

    try:
        return build_mis()
    except Exception as e:
        print(f"  ✗ MIS-Vokabular konnte nicht erzeugt werden: {e}")
        return False


def run_script(script_path: Path, description: str,
               script_args: list[str] | None = None) -> bool:
    """Execute Python script with PYTHONPATH set correctly.

    Die Ausgabe des Sub-Skripts wird zeilenweise eingefangen und über den
    TeeOutput weitergereicht. Damit steht im pipeline_report.txt exakt das,
    was auch im Terminal steht - vorher schrieben die Sub-Skripte direkt auf
    die Konsole und kamen am Log vorbei.

    Zwei Umgebungsvariablen sind dafür nötig: sobald die Ausgabe durch eine
    Pipe geht, fällt Python im Kindprozess auf die Locale-Kodierung zurück
    (auf Windows cp1252), und Zeichen wie ✓, ‰ oder δ würden dort scheitern.
    PYTHONIOENCODING erzwingt UTF-8, PYTHONUNBUFFERED sorgt dafür, dass die
    Zeilen sofort ankommen statt erst am Ende des Schrittes.
    """
    if not script_path.exists():
        print(f"  ✗ {description} not found: {script_path}")
        return False

    print(f"\n  ▶ Starting {description} ...")
    print(f"    Path: {script_path}")
    if script_args:
        print(f"    Args: {' '.join(script_args)}")

    # Set up environment with PYTHONPATH
    env = os.environ.copy()
    pythonpath = str(ONTOLOGY_DIR)

    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = pythonpath + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = pythonpath

    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    # Raster quality reaches the drawing scripts through the environment:
    # every one of them is its own process, and a flag parsed here has no
    # other way in. RASTER_QUALITY is set once from --dpi before the run.
    env.update(RASTER_QUALITY)

    print(f"    PYTHONPATH: {pythonpath}")

    try:
        process = subprocess.Popen(
            [sys.executable, str(script_path), *(script_args or [])],
            cwd=str(script_path.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line.rstrip("\n"))
        returncode = process.wait()

        if returncode == 0:
            print(f"  ✓ {description} completed successfully")
            return True
        else:
            print(f"  ✗ {description} failed with exit code {returncode}")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def _bundle_format_choices():
    """Formatliste aus bundle_rdf importieren, ohne rdflib zu erzwingen."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from bundle_rdf import (
        BUNDLE_FORMATS,
        DEFAULT_BUNDLE_FORMATS,
        RELEASE_BUNDLE_FORMATS,
    )

    return BUNDLE_FORMATS, DEFAULT_BUNDLE_FORMATS, RELEASE_BUNDLE_FORMATS


BUNDLE_FORMATS, DEFAULT_BUNDLE_FORMATS, RELEASE_BUNDLE_FORMATS = (
    _bundle_format_choices()
)


def run_bundle(epica_ok: bool, sisal_ok: bool, ci_ok: bool, formats) -> bool:
    """Schritt 5: Ontologie + alle RDF-Outputs zu dist/geo-lod-bundle.ttl
    zusammenführen und validieren (CRM-Coverage + SHACL).

    Wird nur ausgeführt, wenn mindestens ein Subschritt erfolgreich war -
    sonst ergibt das Bundle keinen Sinn.
    """
    if not (epica_ok or sisal_ok or ci_ok):
        print("  ⚠  Kein Subschritt erfolgreich - Bundle wird übersprungen.")
        return False

    # bundle_rdf.py liegt neben main.py
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from bundle_rdf import run_bundle_step
    except ImportError as e:
        print(f"  ✗ bundle_rdf.py konnte nicht importiert werden: {e}")
        return False

    # Nur die RDF-Verzeichnisse einbeziehen, deren Subschritt erfolgreich war.
    # So vermeiden wir, dass veraltete Outputs in ein neues Bundle geraten.
    rdf_dirs = []
    if epica_ok:
        rdf_dirs.append(EPICA_RDF_DIR)
    if sisal_ok:
        rdf_dirs.append(SISAL_RDF_DIR)
    if ci_ok:
        rdf_dirs.append(CI_RDF_DIR)

    try:
        return run_bundle_step(
            script_dir=SCRIPT_DIR,
            ontology_dir=ONTOLOGY_DIR,
            rdf_dirs=rdf_dirs,
            dist_dir=DIST_DIR,
            formats=formats,
        )
    except Exception as e:
        print(f"  ✗ Bundle-Schritt fehlgeschlagen: {e}")
        return False


def _status(ok: bool, requested: bool) -> str:
    """Three outcomes, not two.

    A step that was never asked for has not failed, and reporting it as a
    failure trains the reader to ignore the summary - which is the one line
    that has to stay trustworthy.
    """
    if not requested:
        return "– Not requested"
    return "✓ Success" if ok else "✗ Failed"


#: How a strand is named in the summary. Longest label sets the column.
SUMMARY_LABELS = {
    "epica": "EPICA",
    "sisal": "SISAL",
    "ci": "CI",
    "archaeo": "Archaeo HTML",
    "bundle": "Bundle",
}


def print_summary(results: dict[str, bool], requested: dict[str, bool],
                  start: datetime):
    print_header("Summary", char="═")
    duration = datetime.now() - start
    width = max(len(label) for label in SUMMARY_LABELS.values()) + 2
    for name, label in SUMMARY_LABELS.items():
        print(f"  {(label + ':').ljust(width)} "
              f"{_status(results[name], requested[name])}")

    if STEP_TIMES:
        print("\n  Duration per step:")
        width = max(len(label) for label, _ in STEP_TIMES)
        total_steps = sum(d for _, d in STEP_TIMES) or 1.0
        for label, secs in STEP_TIMES:
            share = 100.0 * secs / total_steps
            print(f"    {label.ljust(width)}   {secs:7.1f}s   {share:4.1f}%")

    print(f"\n  Total duration: {duration.total_seconds():.1f} seconds")
    print(f"  Log saved to: {LOG_FILE}")

    # Was der Sweep nicht anfasst: Reste, die kein Schritt mehr schreibt, und
    # Dateien, die noch gebraucht werden und mit einem benannten Schritt
    # entfallen. Eine Zeile, damit die Liste im Blick bleibt, ohne den Bericht
    # zu fluten - Einzelheiten liefert clean.py.
    print()
    print(clean.summarise())


def main():
    parser = argparse.ArgumentParser(
        description="geo-lod pipeline: EPICA, SISAL and CI palaeoclimate data "
                    "to RDF, figures, the archaeological HTML pages, and "
                    "the validated bundle"
    )
    # One --<strand>-only switch per strand, and they combine: --ci-only
    # --archaeo-only runs the two that belong together without touching the
    # long-running EPICA and SISAL steps.
    for strand in STRANDS:
        parser.add_argument(
            f"--{strand}-only", action="store_true",
            help=f"run only the {strand} strand. Several --only switches may "
                 f"be given; the run then covers exactly those strands.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Generierte Ausgaben vor dem Lauf entfernen. Voreingestellt an; "
            "die Option bleibt, damit bestehende Aufrufe weiter gelten."
        ),
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help=(
            "Schritt 0 überspringen und in die vorhandenen Ausgaben "
            "hineinschreiben. Dann kann eine Datei überleben, die kein "
            "Schritt mehr schreibt."
        ),
    )
    parser.add_argument(
        "--no-overview", action="store_true",
        help="die Übersichtskarte überspringen. Sie liest die frisch "
             "geschriebenen TTL und braucht dafür ein paar Sekunden mehr.")
    parser.add_argument(
        "--no-bundle",
        action="store_true",
        help="den Bundle-Schritt (RDF-Bundle + Validierung) überspringen",
    )
    parser.add_argument(
        "--bundle-format",
        default=",".join(DEFAULT_BUNDLE_FORMATS),
        help=(
            "Ausgabeformate des Bundles, kommagetrennt. "
            "Verfügbar: " + ", ".join(BUNDLE_FORMATS) + ". "
            "'release' schreibt " + ", ".join(RELEASE_BUNDLE_FORMATS) + ". "
            "Voreinstellung: " + ",".join(DEFAULT_BUNDLE_FORMATS)
            + " - schnell für Entwicklungsläufe."
        ),
    )
    parser.add_argument(
        "--sisal-sites",
        default="dev",
        help=(
            "Welche SISAL-Sites in den Graphen gehen: 'dev' (fünf Sites, so "
            "gewählt, dass jede Prüfung des Generators noch feuert), 'all', "
            "oder eine Liste von site_ids bzw. Namen, z. B. --sisal-sites "
            "spannagel. Ein Release-Lauf setzt 'all' durch. "
            "Voreinstellung: dev - schnell für Entwicklungsläufe."
        ),
    )
    parser.add_argument(
        "--dpi",
        default="draft",
        help=(
            "Auflösung der JPG: 'draft' (100 dpi, klein und schnell), "
            "'print' (300 dpi und mindestens 3000 px auf der kürzeren Seite) "
            "oder eine Zahl. Ab 300 kommt die Pixelgrenze mit, darunter "
            "nicht. Ein Release-Lauf setzt 'print' durch. Das SVG neben jedem "
            "JPG hat ohnehin keine Auflösung. "
            "Voreinstellung: draft - schnell für Entwicklungsläufe."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.bundle_format.strip().lower() in ("release", "all"):
        bundle_formats = list(RELEASE_BUNDLE_FORMATS)
    else:
        bundle_formats = [
            f.strip() for f in args.bundle_format.split(",") if f.strip()
        ]

    # One notion of "this is a release run", read off the bundle formats and
    # passed down. Two independent switches would eventually disagree.
    release_run = set(bundle_formats) >= set(RELEASE_BUNDLE_FORMATS)

    # Ein Release darf keinen Teilgraphen enthalten. Die Voreinstellung des
    # Schalters ist 'dev', und ein Release-Lauf, der sie erbt, wäre ein
    # unvollständiges Bundle unter einem vollständigen Namen.
    # A release run is high resolution, whatever --dpi says. The point of the
    # switch is a fast development run; a release that quietly shipped 100 dpi
    # rasters would be the one case where the default does real damage.
    raster = args.dpi.strip().lower()
    if release_run and raster in ("draft", "dev", ""):
        if raster:
            print("  ℹ  --dpi draft wird für den Release-Lauf auf 'print' "
                  "angehoben")
        raster = "print"
    RASTER_QUALITY["GEO_LOD_RASTER_DPI"] = raster

    sisal_sites = args.sisal_sites.strip()
    if release_run and sisal_sites.lower() != "all":
        sisal_sites = "all"

    # Set up logging
    tee = TeeOutput(LOG_FILE)
    sys.stdout = tee

    start = datetime.now()

    print_header("geo-lod pipeline", char="═")
    print(f"  Timestamp: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Directory: {SCRIPT_DIR}")
    print()

    if not args.no_clean:
        clean_before_run(args)

    print_section(step("Preparation"))
    print("\n  Directory structure:")
    check_directory_exists(SCRIPT_DIR / "EPICA", "EPICA directory")
    check_directory_exists(SCRIPT_DIR / "SISAL", "SISAL directory")
    check_directory_exists(SCRIPT_DIR / "CI",    "CI directory")
    check_directory_exists(ONTOLOGY_DIR, "Ontology directory")

    print("\n  Scripts:")
    epica_rdf_exists  = check_file_exists(EPICA_RDF_SCRIPT,  "EPICA RDF script")
    epica_plot_exists = check_file_exists(EPICA_PLOT_SCRIPT, "EPICA plot script")
    epica_exists = epica_rdf_exists and epica_plot_exists
    sisal_rdf_exists = check_file_exists(SISAL_RDF_SCRIPT, "SISAL RDF script")
    sisal_plot_exists = check_file_exists(SISAL_SCRIPT, "SISAL plot script")
    sisal_exists = sisal_rdf_exists and sisal_plot_exists
    ci_rdf_exists = check_file_exists(CI_RDF_SCRIPT, "CI RDF script")
    ci_plot_exists = check_file_exists(CI_PLOT_SCRIPT, "CI plot script")
    epica_map_exists = check_file_exists(EPICA_MAP_SCRIPT, "EPICA map script")
    sisal_map_exists = check_file_exists(SISAL_MAP_SCRIPT, "SISAL map script")
    overview_exists = check_file_exists(OVERVIEW_SCRIPT, "overview map script")
    ci_exists = ci_rdf_exists and ci_plot_exists
    archaeo_exists = all(
        check_file_exists(script, f"archaeo-connect: {script.name}")
        for script in ARCHAEO_SCRIPTS
    )

    # Documentation drift, reported here and nowhere else. In the preparation
    # step because it costs nothing and belongs with the other "is this
    # repository in the state it claims to be" checks - and as a warning,
    # because a documentation check that stops a run is one people switch off.
    if DOCS_CHECK_SCRIPT.exists():
        print("\n  Run reference:")
        try:
            sys.path.insert(0, str(SCRIPT_DIR))
            import check_docs
            check_docs.check()
        except Exception as error:
            print(f"  ⚠  check_docs konnte nicht laufen: {error}")
    end_section()

    overview_ok = True
    epica_ok   = False
    sisal_ok   = False
    ci_ok      = False
    archaeo_ok = False
    bundle_ok  = False

    print_section(step("Regenerate canonical ontology"))
    canonical_ok = regenerate_canonical_ontology()
    if not canonical_ok:
        print("\n  ⚠  Ontologie konnte nicht regeneriert werden - Bundle wird")
        print("     vermutlich mit veralteter ontology/geo_lod_core.ttl arbeiten.")
    end_section()

    print_section(step("Regenerate controlled vocabularies"))
    vocab_ok = regenerate_vocabularies()
    if not vocab_ok:
        print("\n  ⚠  Vokabulare konnten nicht regeneriert werden - das Bundle")
        print("     arbeitet dann mit einem veralteten ontology/vocab/mis.ttl.")
    end_section()

    # EPICA ist seit S2 zweigeteilt: erst die Tripel, dann die Abbildungen.
    # Der Generator liest die fuenf .tab aus data/raw/epica/, das Plot-Skript
    # dieselben Dateien ueber denselben Loader. Die Reihenfolge ist so gewaehlt,
    # dass ein Fehler in den Daten am RDF-Schritt auffaellt, bevor eine halbe
    # Stunde Abbildungen entsteht.
    if not args.sisal_only and not args.ci_only and epica_exists:
        print_section(step("EPICA Dome C (RDF)"))
        epica_ok = run_script(EPICA_RDF_SCRIPT, "EPICA Dome C RDF generation")
        end_section()

        print_section(step("EPICA Dome C (figures)"))
        epica_plots_ok = run_script(EPICA_PLOT_SCRIPT, "EPICA Dome C figures")
        end_section()
        epica_ok = epica_ok and epica_plots_ok

        if epica_map_exists:
            print_section(step("EPICA Dome C (map)"))
            epica_ok = run_script(EPICA_MAP_SCRIPT,
                                  "EPICA Dome C locator map") and epica_ok
            end_section()

    if not args.epica_only and not args.ci_only and sisal_exists:
        print_section(step("SISAL (RDF)"))
        # The two large SISAL graphs are written as N-Triples in a development
        # run: four times faster to write and to parse, at three times the
        # size, and .gitignore keeps them out. A release run asks for Turtle,
        # which is the format that gets versioned.
        sisal_format = ["--format", "turtle" if release_run else "nt"]
        if release_run and args.sisal_sites.strip().lower() != "all":
            print(f"  ℹ  --sisal-sites {args.sisal_sites.strip()} wird für "
                  f"diesen Release-Lauf auf 'all' gesetzt.")
        sisal_ok = run_script(SISAL_RDF_SCRIPT, "SISAL v3 RDF generation",
                              sisal_format + ["--sites", sisal_sites])
        end_section()

        print_section(step("SISAL (figures)"))
        sisal_plots_ok = run_script(SISAL_SCRIPT, "SISAL figures")
        end_section()
        sisal_ok = sisal_ok and sisal_plots_ok

        # After the profiles, not before: both write into SISAL/captions.yaml,
        # and the maps carry over what the profiles put there.
        if sisal_map_exists:
            print_section(step("SISAL (maps)"))
            sisal_ok = run_script(SISAL_MAP_SCRIPT,
                                  "SISAL cave site maps") and sisal_ok
            end_section()

    # CI is split the way EPICA and SISAL are: triples first, figures after,
    # so that a fault in the findspot table shows up before anything is drawn.
    if strand_requested(args, "ci") and ci_exists:
        print_section(step("Campanian Ignimbrite (RDF)"))
        ci_ok = run_script(CI_RDF_SCRIPT, "CI findspot RDF generation")
        end_section()

        print_section(step("Campanian Ignimbrite (figures)"))
        ci_plots_ok = run_script(CI_PLOT_SCRIPT, "CI findspot figures")
        end_section()
        ci_ok = ci_ok and ci_plots_ok

    # The archaeological HTML pages read the same two files the CI graph is
    # built from, so they run after it: a page that disagrees with the graph
    # it illustrates is worse than no page.
    if strand_requested(args, "archaeo") and archaeo_exists:
        print_section(step("Archaeological connection (HTML)"))
        archaeo_ok = all(
            run_script(script, f"archaeo-connect: {script.stem}")
            for script in ARCHAEO_SCRIPTS
        )
        end_section()

    # Last of the figures, because it reads the RDF the strands have just
    # written rather than their input tables. A strand that did not run is
    # reported as absent instead of silently dropped.
    if overview_exists and not args.no_overview:
        print_section(step("Overview map (from the graph)"))
        overview_ok = run_script(OVERVIEW_SCRIPT,
                                 "all sites, read from the RDF")
        end_section()

    if not args.no_bundle:
        print_section(step("RDF Bundle & Validation"))
        bundle_ok = run_bundle(epica_ok, sisal_ok, ci_ok, bundle_formats)
        end_section()

    # Was dieser Lauf überhaupt vorhatte. Ein nicht angeforderter Schritt ist
    # nicht fehlgeschlagen; vorher gab `--sisal-only` selbst bei fehlerfreiem
    # Lauf Exit-Code 1 zurück, was einen CI-Job reihenweise rot färbt.
    exists = {"epica": epica_exists, "sisal": sisal_exists, "ci": ci_exists,
              "archaeo": archaeo_exists}
    requested = {name: strand_requested(args, name) and exists[name]
                 for name in STRANDS}
    requested["bundle"] = not args.no_bundle

    results = {"epica": epica_ok, "sisal": sisal_ok, "ci": ci_ok,
               "archaeo": archaeo_ok, "bundle": bundle_ok}
    print_summary(results, requested, start)

    overall_ok = (
        canonical_ok
        and vocab_ok
        and all(ok for name, ok in results.items() if requested[name])
    )

    # Schlusszeile noch durch den Tee, damit Log und Terminal Zeile für Zeile
    # dasselbe zeigen; erst danach die Logdatei schliessen.
    if overall_ok:
        print("✓ Pipeline completed successfully!")
    else:
        print("⚠  Some steps failed - see errors above.")

    tee.close()
    sys.stdout = tee.terminal

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
