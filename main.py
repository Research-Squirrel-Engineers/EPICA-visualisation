#!/usr/bin/env python3
"""
main.py - EPICA + SISAL Pipeline with logging
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
ONTOLOGY_DIR = SCRIPT_DIR / "ontology"

EPICA_PLOTS_DIR = SCRIPT_DIR / "EPICA" / "plots"
EPICA_RDF_DIR = SCRIPT_DIR / "EPICA" / "rdf"
EPICA_REPORT_DIR = SCRIPT_DIR / "EPICA" / "report"
SISAL_PLOTS_DIR = SCRIPT_DIR / "SISAL" / "plots"
SISAL_RDF_DIR = SCRIPT_DIR / "SISAL" / "rdf"
SISAL_REPORT_DIR = SCRIPT_DIR / "SISAL" / "report"

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


def print_summary(epica: bool, sisal: bool, start: datetime):
    print_header("Summary", char="═")
    duration = datetime.now() - start
    print(f"  EPICA:  {'✓ Success' if epica else '✗ Failed'}")
    print(f"  SISAL:  {'✓ Success' if sisal else '✗ Failed'}")
    print(f"\n  Total duration: {duration.total_seconds():.1f} seconds")
    print(f"  Log saved to: {LOG_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description="EPICA + SISAL Palaeoclimate Data Processing Pipeline"
    )
    parser.add_argument("--epica-only", action="store_true")
    parser.add_argument("--sisal-only", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Set up logging
    tee = TeeOutput(LOG_FILE)
    sys.stdout = tee

    start = datetime.now()

    print_header("EPICA + SISAL Pipeline", char="═")
    print(f"  Timestamp: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Directory: {SCRIPT_DIR}")
    print()

    if args.clean:
        clean_all_outputs()

    print_section("1. Preparation")
    print("\n  Directory structure:")
    check_directory_exists(SCRIPT_DIR / "EPICA", "EPICA directory")
    check_directory_exists(SCRIPT_DIR / "SISAL", "SISAL directory")
    check_directory_exists(ONTOLOGY_DIR, "Ontology directory")

    print("\n  Scripts:")
    epica_exists = check_file_exists(EPICA_SCRIPT, "EPICA script")
    sisal_exists = check_file_exists(SISAL_SCRIPT, "SISAL script")

    epica_ok = False
    sisal_ok = False

    if not args.sisal_only and epica_exists:
        print_section("2. EPICA Dome C (Ice Core)")
        epica_ok = run_script(EPICA_SCRIPT, "EPICA Dome C Processing")

    if not args.epica_only and sisal_exists:
        print_section("3. SISAL (Speleothems)")
        sisal_ok = run_script(SISAL_SCRIPT, "SISAL Processing")

    print_summary(epica_ok, sisal_ok, start)

    # Close log file
    tee.close()
    sys.stdout = tee.terminal

    if not epica_ok or not sisal_ok:
        print("⚠  Some steps failed — see errors above.")
        sys.exit(1)
    else:
        print("✓ Pipeline completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
