#!/usr/bin/env python3
"""clean.py - what the pipeline writes, and what it no longer needs.

Three registers, and the difference between them is the whole point of this
file:

    GENERATED   written afresh by a pipeline step at every run. Removing it
                costs nothing but the time to run the step again, so main.py
                removes it before the step starts. That is what keeps an
                orphan out of the repository: a figure whose proxy has been
                renamed is not overwritten by the next run, it is simply
                never written again, and without a sweep it would sit in
                plots/ forever, indistinguishable from a current one.

    STALE       written by an earlier version of the pipeline, read by
                nothing today. A sweep of the output directories does not
                catch these, because they sit outside them. Reported at every
                run, removed only when asked: `python clean.py --stale
                --delete`.

    PENDING     still in use, but scheduled to go at a named step. Never
                removed here, only listed, so that the list is a to-do and
                not a trap.

Standalone use, from the repository root:

    python clean.py                    report everything, delete nothing
    python clean.py --delete           remove the generated files
    python clean.py --stale --delete   also remove the unused leftovers
    python clean.py --group epica      restrict to one step
    python clean.py --list-groups      show the groups and exit

The default is to report. Deleting takes a flag, in both directions: nothing
here removes a file unless it was asked to, and main.py asks for exactly the
groups whose step it is about to run.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.absolute()


@dataclass(frozen=True)
class Entry:
    """One line of the inventory.

    ``pattern`` is a path or glob relative to the repository root. With
    ``contents=True`` the directory itself stays and only what is inside it
    goes, which is what the output directories want: git keeps no empty
    directory, and a step that writes into a directory it has to create first
    is one failure mode more.
    """

    pattern: str
    why: str
    contents: bool = False
    note: str = ""

    def resolve(self) -> list[Path]:
        """Existing paths this entry stands for, sorted, deepest first.

        Deepest first so that a caller may delete the list in order without
        removing a parent before its children.
        """
        matches = sorted(ROOT.glob(self.pattern))
        if not self.contents:
            return sorted(matches, key=lambda p: len(p.parts), reverse=True)
        inner: list[Path] = []
        for directory in matches:
            if directory.is_dir():
                inner.extend(directory.iterdir())
        return sorted(inner, key=lambda p: len(p.parts), reverse=True)


# --------------------------------------------------------------------------
# GENERATED - one group per pipeline step, named as main.py names its steps.
# --------------------------------------------------------------------------
# The grouping is not cosmetic. `main.py --sisal-only` must not wipe the EPICA
# figures: half an hour of drawing, removed by a run that never intended to
# replace them. main.py therefore asks for the groups whose step it is about
# to run, and for no others.
GENERATED: dict[str, list[Entry]] = {
    "vocab": [
        Entry("ontology/geo_lod_core.ttl",
              "ontology step, from geo_lod_utils.GEO_LOD_CORE_TTL"),
        Entry("ontology/vocab/mis.ttl", "vocabulary step, build_mis_vocab"),
        Entry("ontology/trs.ttl", "vocabulary step, build_mis_vocab"),
        Entry("dist/mis_stages.csv", "vocabulary step, build_mis_vocab"),
        Entry("dist/mis_assignments.csv", "vocabulary step, build_mis_vocab"),
    ],
    "diagrams": [
        # write_mermaid writes all four in one call, and since S3c.2 only the
        # EPICA step calls it. Their own group, because a run without EPICA
        # would otherwise remove them and leave nothing to write them back.
        Entry("ontology/*.mermaid", "geo_lod_utils.write_mermaid, EPICA step"),
    ],
    "epica": [
        Entry("EPICA/rdf", "EPICA/epica_rdf.py", contents=True),
        Entry("EPICA/report",
              "EPICA/epica_rdf.py, plot_epica_from_tab.py, plot_epica_map.py", contents=True),
        Entry("EPICA/plots", "EPICA/plot_epica_from_tab.py", contents=True),
        Entry("EPICA/maps", "EPICA/plot_epica_map.py", contents=True),
        Entry("EPICA/captions.yaml",
              "EPICA/epica_plates.py, plot_epica_map.py; one entry per figure"),
    ],
    "sisal": [
        # Both .ttl and .nt: the directory is swept whole, so a release run's
        # Turtle cannot survive next to a later development run's N-Triples
        # and be picked up as if it were current.
        Entry("SISAL/rdf", "SISAL/sisal_rdf.py", contents=True),
        # data/curated/ is NOT here: it is hand-maintained input, not output.
        Entry("SISAL/report",
              "SISAL/plot_sisal_from_csv.py, plot_sisal_maps.py",
              contents=True),
        Entry("SISAL/plots", "SISAL/plot_sisal_from_csv.py", contents=True),
        Entry("SISAL/maps", "SISAL/plot_sisal_maps.py", contents=True),
        Entry("SISAL/captions.yaml",
              "SISAL/plot_sisal_from_csv.py, one entry per figure"),
    ],
    "overview": [
        # maps/ rather than plots/: the figure is drawn against a coastline,
        # not against an axis, and it belongs to no strand.
        Entry("maps", "plot_overview_map.py", contents=True),
    ],
    "ci": [
        Entry("CI/rdf", "CI/ci_pipeline.py", contents=True),
        Entry("CI/report", "CI/ci_pipeline.py, plot_ci_findspots.py",
              contents=True),
        Entry("CI/maps", "CI/plot_ci_findspots.py", contents=True),
        Entry("CI/captions.yaml",
              "CI/plot_ci_findspots.py, one entry per figure"),
    ],
    "archaeo": [
        # The HTML pages, not their inputs: the findspot table lives under
        # data/raw/ and the annotations under data/curated/, and neither is
        # written by a step.
        Entry("archaeo-connect/CI_findspots_CAA.html",
              "archaeo-connect/ci_findspots_html.py"),
    ],
    "bundle": [
        # Only the bundle itself. dist/ also holds the MIS tables, and those
        # belong to the vocab step: a --no-bundle run must not lose them.
        Entry("dist/geo-lod-bundle.*", "bundle_rdf.py, bundle step"),
    ],
    "log": [
        Entry("pipeline_report.txt", "main.py, TeeOutput"),
    ],
    "cache": [
        Entry("**/__pycache__", "Python bytecode"),
    ],
}

# Groups a plain `python clean.py --delete` acts on. The log is left out: it
# is the record of the run that is writing it, and removing it from a
# standalone call would only puzzle whoever reads the directory next.
DEFAULT_GROUPS: tuple[str, ...] = (
    "vocab", "diagrams", "epica", "sisal", "ci", "archaeo", "overview",
    "bundle", "cache",
)


# --------------------------------------------------------------------------
# STALE - written once, read by nothing.
# --------------------------------------------------------------------------
# Everything here has been checked to be unreferenced, and where it is a copy,
# byte-identical to its original. The check is worth repeating before adding a
# line: a file that looks like a leftover and turns out to be an input is the
# expensive kind of mistake.
STALE: list[Entry] = [
    Entry("SISAL/v_sites_all.csv",
          "the old 305-site list, replaced by the database catalogue",
          note="a v2-era snapshot: SISAL v3 has 365 sites. Ids and names are "
               "unchanged, so nothing had to be migrated. The curated columns "
               "live on in data/curated/sisal_site_annotations.csv"),
    Entry("EPICA/epica_ontology.ttl",
          "identical to the generated EPICA/rdf/epica_ontology.ttl",
          note="the bundle reads EPICA/rdf/*.ttl; nothing reads this one"),
    Entry("EPICA/ch4_vs_age_ka_full_smooth11.jpg",
          "figure outside plots/, identical to EPICA/plots/",
          note="predates the plots/ directory"),
    Entry("EPICA/ch4_vs_depth_full_smooth11.jpg",
          "figure outside plots/, identical to EPICA/plots/",
          note="predates the plots/ directory"),
    Entry("SISAL/v_data_*.csv",
          "the old figure input, replaced by data/derived/sisal/sites/",
          note="carried the decimal-separator error - 907 rows for Botuvera "
               "against 920, 5832 for Sanbao against 6085. Nothing reads them "
               "since S3c.4"),
]


# --------------------------------------------------------------------------
# PENDING - in use, and dated.
# --------------------------------------------------------------------------
# Listed, never removed. Each line names the step that ends it, so that the
# report answers "why is this still here" without a look into PRIMER.md.
PENDING: list[Entry] = [
    Entry("archaeo-connect/v_sites_all.csv",
          "the last input the SISAL archaeology page had",
          note="an older cut than SISAL/v_sites_all.csv, not a copy of it. "
               "Nothing reads it today - see sisal_arch_html.py below - and "
               "whether it stays is decided in S3f with the page itself"),
    Entry("archaeo-connect/sisal_arch_html.py",
          "a copy of ci_findspots_html.py, not a SISAL script",
          note="the generator of SISAL_arch_sites_CAA.html was overwritten "
               "with a reformatted copy of the CI script at some point: it "
               "reads the findspot table and writes the CI page. The SISAL "
               "page in the same directory therefore has no generator. "
               "Rebuilt or dropped in S3f, not here"),
    Entry("archaeo-connect/SISAL_arch_sites_CAA.html",
          "a page no script writes any more",
          note="kept because it is the only surviving statement of what the "
               "lost generator produced"),
    Entry("s3c1_import.txt",
          "run log of the S3c.1 import",
          note="evidence, not an input; keep or remove by hand"),
]


# --------------------------------------------------------------------------
# Sizes, safety, reporting
# --------------------------------------------------------------------------

def _size(path: Path) -> int:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    try:
        return path.stat().st_size
    except OSError:
        return 0


def human(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _inside_root(path: Path) -> bool:
    """True if *path* is below the repository root and is not the root itself.

    A glob that matches nothing is harmless; a glob that matches the root is
    not. Every removal passes through here.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return False
    root = ROOT.resolve()
    return resolved != root and root in resolved.parents


def _remove(path: Path) -> bool:
    if not _inside_root(path):
        print(f"  ⚠  refused, outside the repository: {path}")
        return False
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    except OSError as exc:
        print(f"  ⚠  {path.relative_to(ROOT)}: {exc}")
        return False


@dataclass
class Result:
    """What a sweep found, and what it did about it."""

    removed: int = 0
    bytes_freed: int = 0
    listed: list[tuple[Entry, list[Path]]] = field(default_factory=list)

    @property
    def files(self) -> int:
        return sum(len(paths) for _, paths in self.listed)


def sweep(groups=None, *, delete: bool = False, verbose: bool = False) -> Result:
    """Report - and with ``delete=True`` remove - the generated files.

    Returns the tally rather than printing it, so that main.py can decide how
    loud to be.
    """
    wanted = list(DEFAULT_GROUPS) if groups is None else list(groups)
    result = Result()

    for group in wanted:
        entries = GENERATED.get(group)
        if entries is None:
            print(f"  ⚠  unknown group: {group}")
            continue
        for entry in entries:
            paths = entry.resolve()
            if not paths:
                continue
            result.listed.append((entry, paths))
            size = sum(_size(p) for p in paths)
            if delete:
                gone = sum(1 for p in paths if _remove(p))
                result.removed += gone
                result.bytes_freed += size
                if verbose:
                    print(f"  ✓ {entry.pattern:<34s} {gone:>4d} gone     "
                          f"{human(size):>9s}")
            elif verbose:
                print(f"    {entry.pattern:<34s} {len(paths):>4d} file(s)  "
                      f"{human(size):>9s}   {entry.why}")
    return result


def sweep_stale(*, delete: bool = False, verbose: bool = True) -> Result:
    """The same for the leftovers. Deleting is opt-in twice over."""
    result = Result()
    for entry in STALE:
        paths = entry.resolve()
        if not paths:
            continue
        result.listed.append((entry, paths))
        size = sum(_size(p) for p in paths)
        if delete:
            gone = sum(1 for p in paths if _remove(p))
            result.removed += gone
            result.bytes_freed += size
            if verbose:
                print(f"  ✓ {entry.pattern:<44s} removed   {human(size):>9s}")
        elif verbose:
            for path in paths:
                print(f"    {str(path.relative_to(ROOT)):<44s} {human(_size(path)):>9s}")
            print(f"      {entry.why}")
            if entry.note:
                print(f"      {entry.note}")
    return result


def report_pending(verbose: bool = True) -> Result:
    result = Result()
    for entry in PENDING:
        paths = entry.resolve()
        if not paths:
            continue
        result.listed.append((entry, paths))
        if verbose:
            size = sum(_size(p) for p in paths)
            label = entry.pattern if len(paths) > 1 else str(paths[0].relative_to(ROOT))
            print(f"    {label:<44s} {len(paths):>3d}  {human(size):>9s}")
            print(f"      {entry.why}")
            if entry.note:
                print(f"      {entry.note}")
    return result


def summarise() -> str:
    """One line for the end of a pipeline run: what is still lying around."""
    stale = sweep_stale(delete=False, verbose=False)
    pending = report_pending(verbose=False)
    parts = []
    if stale.files:
        parts.append(f"{stale.files} unused leftover(s)")
    if pending.files:
        parts.append(f"{pending.files} file(s) scheduled for removal")
    if not parts:
        return "  ✓ No leftovers - every file in the repository is current."
    return ("  ℹ  " + ", ".join(parts)
            + " - details: python clean.py")


# --------------------------------------------------------------------------
# Standalone
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report or remove what the geo-lod pipeline writes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Standalone use")[-1],
    )
    parser.add_argument("--delete", action="store_true",
                        help="actually remove; without it nothing is touched")
    parser.add_argument("--stale", action="store_true",
                        help="include the unused leftovers")
    parser.add_argument("--group", action="append", metavar="NAME",
                        help="restrict to one group, repeatable")
    parser.add_argument("--list-groups", action="store_true",
                        help="list the groups and exit")
    args = parser.parse_args(argv)

    if args.list_groups:
        print("Groups:\n")
        width = max(len(g) for g in GENERATED)
        for name, entries in GENERATED.items():
            marker = " " if name in DEFAULT_GROUPS else "-"
            print(f"  {marker} {name.ljust(width)}  {len(entries)} entries")
        print("\n  (- is not in the default set)")
        return 0

    groups = args.group or list(DEFAULT_GROUPS)

    print("\nGenerated - rewritten by the step that owns it")
    print("─" * 78)
    generated = sweep(groups, delete=args.delete, verbose=True)
    if not generated.listed:
        print("    nothing found")

    print("\nUnused leftovers - written once, read by nothing")
    print("─" * 78)
    stale = sweep_stale(delete=args.delete and args.stale, verbose=True)
    if not stale.listed:
        print("    nothing found")

    print("\nScheduled - in use, and dated; never removed here")
    print("─" * 78)
    pending = report_pending(verbose=True)
    if not pending.listed:
        print("    nothing found")

    print("\n" + "─" * 78)
    if args.delete:
        freed = generated.bytes_freed + (stale.bytes_freed if args.stale else 0)
        print(f"  Removed {generated.removed + stale.removed} item(s), "
              f"{human(freed)} freed.")
        if stale.listed and not args.stale:
            print("  The leftovers were kept: add --stale to remove them too.")
    else:
        print(f"  Nothing was removed. {generated.files} generated file(s), "
              f"{stale.files} leftover(s), {pending.files} scheduled.")
        print("  Add --delete to remove the generated files, "
              "--delete --stale for the leftovers too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
