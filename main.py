#!/usr/bin/env python3
"""
main.py - EPICA + SISAL + CI Pipeline with logging and bundle step
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import shutil

# Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
EPICA_SCRIPT = SCRIPT_DIR / "EPICA" / "plot_epica_from_tab.py"
SISAL_SCRIPT = SCRIPT_DIR / "SISAL" / "plot_sisal_from_csv.py"
CI_SCRIPT    = SCRIPT_DIR / "CI" / "ci_pipeline.py"
ONTOLOGY_DIR = SCRIPT_DIR / "ontology"
DIST_DIR     = SCRIPT_DIR / "dist"

EPICA_PLOTS_DIR = SCRIPT_DIR / "EPICA" / "plots"
EPICA_RDF_DIR = SCRIPT_DIR / "EPICA" / "rdf"
EPICA_REPORT_DIR = SCRIPT_DIR / "EPICA" / "report"
SISAL_PLOTS_DIR = SCRIPT_DIR / "SISAL" / "plots"
SISAL_RDF_DIR = SCRIPT_DIR / "SISAL" / "rdf"
SISAL_REPORT_DIR = SCRIPT_DIR / "SISAL" / "report"
CI_RDF_DIR = SCRIPT_DIR / "CI" / "rdf"

# Global log file
LOG_FILE = SCRIPT_DIR / "pipeline_report.txt"


class TeeOutput:
    """Writes to both stdout and a file"""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

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


def print_section(text: str):
    print()
    print("─" * 80)
    print(f"  {text}")
    print("─" * 80)


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


def clean_directory(dirpath: Path, description: str) -> int:
    if not dirpath.exists():
        return 0
    count = 0
    try:
        for item in dirpath.iterdir():
            if item.is_file():
                item.unlink()
                count += 1
            elif item.is_dir():
                shutil.rmtree(item)
                count += 1
        if count > 0:
            print(f"  ✓ Cleaned {description}: {count} items removed")
    except Exception as e:
        print(f"  ⚠  Error cleaning {description}: {e}")
    return count


def clean_pycache(root_dir: Path) -> int:
    count = 0
    for pycache in root_dir.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            count += 1
        except:
            pass
    if count > 0:
        print(f"  ✓ Removed {count} __pycache__ directories")
    return count


def clean_all_outputs() -> None:
    print_section("Cleaning Output Directories")
    total = 0
    total += clean_directory(EPICA_PLOTS_DIR, "EPICA plots")
    total += clean_directory(EPICA_RDF_DIR, "EPICA RDF")
    total += clean_directory(EPICA_REPORT_DIR, "EPICA reports")
    total += clean_directory(SISAL_PLOTS_DIR, "SISAL plots")
    total += clean_directory(SISAL_RDF_DIR, "SISAL RDF")
    total += clean_directory(SISAL_REPORT_DIR, "SISAL reports")
    total += clean_directory(CI_RDF_DIR, "CI RDF")
    total += clean_directory(DIST_DIR, "dist (bundle)")

    if ONTOLOGY_DIR.exists():
        count = 0
        for f in ONTOLOGY_DIR.glob("*.mermaid"):
            try:
                f.unlink()
                count += 1
            except:
                pass
        if count > 0:
            print(f"  ✓ Removed {count} Mermaid files from ontology/")
            total += count

    print("\n  Python cache cleanup:")
    total += clean_pycache(SCRIPT_DIR)
    print(f"\n  Total items removed: {total}")


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


def run_script(script_path: Path, description: str) -> bool:
    """Execute Python script with PYTHONPATH set correctly."""
    if not script_path.exists():
        print(f"  ✗ {description} not found: {script_path}")
        return False

    print(f"\n  ▶ Starting {description} ...")
    print(f"    Path: {script_path}")

    # Set up environment with PYTHONPATH
    env = os.environ.copy()
    pythonpath = str(ONTOLOGY_DIR)

    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = pythonpath + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = pythonpath

    print(f"    PYTHONPATH: {pythonpath}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            env=env,
            capture_output=False,
        )

        if result.returncode == 0:
            print(f"  ✓ {description} completed successfully")
            return True
        else:
            print(f"  ✗ {description} failed with exit code {result.returncode}")
            return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def run_bundle(epica_ok: bool, sisal_ok: bool, ci_ok: bool) -> bool:
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
        )
    except Exception as e:
        print(f"  ✗ Bundle-Schritt fehlgeschlagen: {e}")
        return False


def print_summary(epica: bool, sisal: bool, ci: bool, bundle: bool, start: datetime):
    print_header("Summary", char="═")
    duration = datetime.now() - start
    print(f"  EPICA:   {'✓ Success' if epica  else '✗ Failed / skipped'}")
    print(f"  SISAL:   {'✓ Success' if sisal  else '✗ Failed / skipped'}")
    print(f"  CI:      {'✓ Success' if ci     else '✗ Failed / skipped'}")
    print(f"  Bundle:  {'✓ Success' if bundle else '✗ Failed / skipped'}")
    print(f"\n  Total duration: {duration.total_seconds():.1f} seconds")
    print(f"  Log saved to: {LOG_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description="EPICA + SISAL + CI Palaeoclimate Data Processing Pipeline"
    )
    parser.add_argument("--epica-only", action="store_true")
    parser.add_argument("--sisal-only", action="store_true")
    parser.add_argument("--ci-only", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--no-bundle",
        action="store_true",
        help="Schritt 5 (RDF-Bundle + Validierung) überspringen",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Set up logging
    tee = TeeOutput(LOG_FILE)
    sys.stdout = tee

    start = datetime.now()

    print_header("EPICA + SISAL + CI Pipeline", char="═")
    print(f"  Timestamp: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Directory: {SCRIPT_DIR}")
    print()

    if args.clean:
        clean_all_outputs()

    print_section("1. Preparation")
    print("\n  Directory structure:")
    check_directory_exists(SCRIPT_DIR / "EPICA", "EPICA directory")
    check_directory_exists(SCRIPT_DIR / "SISAL", "SISAL directory")
    check_directory_exists(SCRIPT_DIR / "CI",    "CI directory")
    check_directory_exists(ONTOLOGY_DIR, "Ontology directory")

    print("\n  Scripts:")
    epica_exists = check_file_exists(EPICA_SCRIPT, "EPICA script")
    sisal_exists = check_file_exists(SISAL_SCRIPT, "SISAL script")
    ci_exists    = check_file_exists(CI_SCRIPT,    "CI script")

    epica_ok  = False
    sisal_ok  = False
    ci_ok     = False
    bundle_ok = False

    print_section("2. Regenerate canonical ontology")
    canonical_ok = regenerate_canonical_ontology()
    if not canonical_ok:
        print("\n  ⚠  Ontologie konnte nicht regeneriert werden - Bundle wird")
        print("     vermutlich mit veralteter ontology/geo_lod_core.ttl arbeiten.")

    print_section("3. Regenerate controlled vocabularies")
    vocab_ok = regenerate_vocabularies()
    if not vocab_ok:
        print("\n  ⚠  Vokabulare konnten nicht regeneriert werden - das Bundle")
        print("     arbeitet dann mit einem veralteten ontology/vocab/mis.ttl.")

    if not args.sisal_only and not args.ci_only and epica_exists:
        print_section("4. EPICA Dome C (Ice Core)")
        epica_ok = run_script(EPICA_SCRIPT, "EPICA Dome C Processing")

    if not args.epica_only and not args.ci_only and sisal_exists:
        print_section("5. SISAL (Speleothems)")
        sisal_ok = run_script(SISAL_SCRIPT, "SISAL Processing")

    if not args.epica_only and not args.sisal_only and ci_exists:
        print_section("6. Campanian Ignimbrite (CI Findspots)")
        ci_ok = run_script(CI_SCRIPT, "CI Findspot Processing")

    if not args.no_bundle:
        print_section("7. RDF Bundle & Validation")
        bundle_ok = run_bundle(epica_ok, sisal_ok, ci_ok)

    print_summary(epica_ok, sisal_ok, ci_ok, bundle_ok, start)

    # Close log file
    tee.close()
    sys.stdout = tee.terminal

    # Pipeline insgesamt nur grün, wenn alle aktivierten Schritte ok sind.
    bundle_required = not args.no_bundle
    overall_ok = (
        canonical_ok
        and vocab_ok
        and epica_ok
        and sisal_ok
        and ci_ok
        and (bundle_ok if bundle_required else True)
    )

    if not overall_ok:
        print("⚠  Some steps failed - see errors above.")
        sys.exit(1)
    else:
        print("✓ Pipeline completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
