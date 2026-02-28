# main.py mit subprocess execution - FIXED VERSION

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
EPICA_OUTPUT_DIR = SCRIPT_DIR / "EPICA"
SISAL_OUTPUT_DIR = SCRIPT_DIR / "SISAL"
EPICA_PLOTS_DIR = EPICA_OUTPUT_DIR / "plots"
EPICA_RDF_DIR = EPICA_OUTPUT_DIR / "rdf"
EPICA_REPORT_DIR = EPICA_OUTPUT_DIR / "report"
SISAL_PLOTS_DIR = SISAL_OUTPUT_DIR / "plots"
SISAL_RDF_DIR = SISAL_OUTPUT_DIR / "rdf"
SISAL_REPORT_DIR = SISAL_OUTPUT_DIR / "report"

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

def clean_directory(dirpath: Path, description: str, keep_dir: bool = True) -> int:
    if not dirpath.exists():
        return 0
    count = 0
    try:
        if keep_dir:
            for item in dirpath.iterdir():
                if item.is_file():
                    item.unlink()
                    count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    count += 1
            print(f"  ✓ Cleaned {description}: {count} items removed")
        else:
            if dirpath.exists():
                shutil.rmtree(dirpath)
                count = 1
                print(f"  ✓ Removed {description}")
    except Exception as e:
        print(f"  ⚠  Error cleaning {description}: {e}")
    return count

def clean_pycache(root_dir: Path) -> int:
    count = 0
    for pycache in root_dir.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            count += 1
        except Exception as e:
            print(f"  ⚠  Could not remove {pycache}: {e}")
    if count > 0:
        print(f"  ✓ Removed {count} __pycache__ directories")
    return count

def clean_all_outputs(clean_cache: bool = True) -> None:
    print_section("Cleaning Output Directories")
    total_removed = 0
    total_removed += clean_directory(EPICA_PLOTS_DIR, "EPICA plots")
    total_removed += clean_directory(EPICA_RDF_DIR, "EPICA RDF")
    total_removed += clean_directory(EPICA_REPORT_DIR, "EPICA reports")
    total_removed += clean_directory(SISAL_PLOTS_DIR, "SISAL plots")
    total_removed += clean_directory(SISAL_RDF_DIR, "SISAL RDF")
    total_removed += clean_directory(SISAL_REPORT_DIR, "SISAL reports")
    
    if ONTOLOGY_DIR.exists():
        mermaid_count = 0
        for mermaid_file in ONTOLOGY_DIR.glob("*.mermaid"):
            try:
                mermaid_file.unlink()
                mermaid_count += 1
            except Exception as e:
                print(f"  ⚠  Could not remove {mermaid_file.name}: {e}")
        if mermaid_count > 0:
            print(f"  ✓ Removed {mermaid_count} Mermaid files from ontology/")
            total_removed += mermaid_count
    
    if clean_cache:
        print("\n  Python cache cleanup:")
        total_removed += clean_pycache(SCRIPT_DIR)
    
    print(f"\n  Total items removed: {total_removed}")

def run_script(script_path: Path, description: str, skip: bool = False) -> bool:
    """Executes a Python script as subprocess."""
    if skip:
        print(f"  ⊘ {description} skipped (--skip option)")
        return True

    if not script_path.exists():
        print(f"  ✗ {description} not found: {script_path}")
        return False

    print(f"\n  ▶ Starting {description} ...")
    print(f"    Path: {script_path}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"  ✓ {description} completed successfully")
            return True
        else:
            print(f"  ✗ {description} failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error in {description}:")
        print(f"    {type(e).__name__}: {e}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback
            traceback.print_exc()
        return False

def print_summary(epica_success: bool, sisal_success: bool, start_time: datetime):
    print_header("Summary", char="═")
    duration = datetime.now() - start_time
    print(f"  EPICA:  {'✓ Success' if epica_success else '✗ Failed'}")
    print(f"  SISAL:  {'✓ Success' if sisal_success else '✗ Failed'}")
    print(f"\n  Total duration: {duration.total_seconds():.1f} seconds")

def main():
    parser = argparse.ArgumentParser(
        description="EPICA + SISAL Palaeoclimate Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Execute everything
  python main.py --clean            # Clean outputs first, then execute everything
  python main.py --epica-only       # EPICA only
  python main.py --sisal-only       # SISAL only
  python main.py --no-plots         # Generate RDF only
  python main.py --no-rdf           # Generate plots only
        """
    )
    
    parser.add_argument("--epica-only", action="store_true", help="Execute EPICA only")
    parser.add_argument("--sisal-only", action="store_true", help="Execute SISAL only")
    parser.add_argument("--no-plots", action="store_true", help="Do not generate plots (RDF/TTL only)")
    parser.add_argument("--no-rdf", action="store_true", help="Do not generate RDF (plots only)")
    parser.add_argument("--clean", action="store_true", help="Clean all output directories and Python cache before running")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detailed output")

    args = parser.parse_args()
    start_time = datetime.now()

    print_header("EPICA + SISAL Pipeline", char="═")
    print(f"  Timestamp: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Directory: {SCRIPT_DIR}")
    print()

    if args.clean:
        clean_all_outputs(clean_cache=True)

    print_section("1. Preparation")
    print("\n  Directory structure:")
    epica_dir_ok = check_directory_exists(EPICA_OUTPUT_DIR, "EPICA directory")
    sisal_dir_ok = check_directory_exists(SISAL_OUTPUT_DIR, "SISAL directory")
    ontology_ok = check_directory_exists(ONTOLOGY_DIR, "Ontology directory")

    print("\n  Scripts:")
    epica_exists = check_file_exists(EPICA_SCRIPT, "EPICA script")
    sisal_exists = check_file_exists(SISAL_SCRIPT, "SISAL script")

    if not epica_exists and not sisal_exists:
        print("\n  ✗ ERROR: No scripts found!")
        sys.exit(1)

    epica_success = False
    if not args.sisal_only and epica_exists:
        print_section("2. EPICA Dome C (Ice Core)")
        epica_success = run_script(EPICA_SCRIPT, "EPICA Dome C Processing", skip=False)

    sisal_success = False
    if not args.epica_only and sisal_exists:
        print_section("3. SISAL (Speleothems)")
        sisal_success = run_script(SISAL_SCRIPT, "SISAL Processing", skip=False)

    print_summary(epica_success, sisal_success, start_time)

    if (not args.epica_only and not epica_success) or (not args.sisal_only and not sisal_success):
        print("⚠  Some steps failed — see errors above.")
        sys.exit(1)
    else:
        print("✓ Pipeline completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
