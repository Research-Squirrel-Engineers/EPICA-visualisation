#!/usr/bin/env python3
"""Take over the SISAL v3 cut from the sister repository `sisal-db-v3`.

`sisal-db-v3` owns the database and the SQL; this repository owns the graph. The
handover is a set of CSV files, and this script is the handover: it copies them
out of a checkout of the sister repository, records what it took, and can later
confirm that what lies here is still that.

    python SISAL/sisal_import.py                 # pull from the sister checkout
    python SISAL/sisal_import.py --from ../sisal-db-v3
    python SISAL/sisal_import.py --verify        # check only, no copying

The distinction matters for reproducibility. Pulling needs the sister checkout
and a run of its export phase, which needs a running PostgreSQL. Verifying needs
neither: it reads the manifest and the files next to it. Every later step of the
pipeline verifies, never pulls, so a clone of this repository builds the graph
without a database anywhere in sight -- which is the entire reason the cut is
committed here rather than queried at build time.

What is taken over:

    tables/*.csv     the structure-preserving cut, one file per release table.
                     site -> entity -> sample -> chronology is intact, which is
                     what the RDF generator needs.
    sites/*.csv      the flat per-site files, one row per measurement, used by
                     the figures.
    queries.yaml     the export definition that produced both. Copied rather
                     than referenced, so this repository can always answer which
                     SQL its data came from, sister checkout or not.

The manifest holds a SHA-256 and a row count per file, plus the origin of the
pull. A checksum mismatch says the file was edited after the pull, which is the
one thing that must never happen: the cut is a query result, and editing it here
would make the SQL a lie.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "data" / "derived" / "sisal"
MANIFEST = TARGET / "MANIFEST.json"

# Where the sister checkout is looked for, in order, relative to this repository.
# The repository has been renamed once already, so both names are tried before
# giving up and asking for --from.
CANDIDATES = (
    Path("..") / "squirrels-sisal-db-v3",
    Path("..") / "sisal-db-v3",
    Path("..") / "GeoScience-FAIRification-LOD" / ".." / "sisal-db-v3",
)

# Free-text columns run past the default limit; csv.reader raises rather than
# truncating.
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


def find_source(explicit: str | None) -> Path:
    """Return the sister checkout, or exit with the paths that were tried."""
    if explicit:
        source = Path(explicit).expanduser().resolve()
        if not (source / "data" / "derived").is_dir():
            raise SystemExit(
                f"ERROR: {source} does not look like a sisal-db-v3 checkout\n"
                f"       (no data/derived/ below it)"
            )
        return source

    for candidate in CANDIDATES:
        source = (ROOT / candidate).resolve()
        if (source / "data" / "derived" / "tables").is_dir():
            return source

    tried = "\n".join(f"       {(ROOT / c).resolve()}" for c in CANDIDATES)
    raise SystemExit(
        "ERROR: no sisal-db-v3 checkout found. Tried:\n" + tried + "\n"
        "       Pass one with --from, or run `python py/main.py export` there\n"
        "       first if the checkout exists but data/derived/tables/ does not."
    )


# --------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------
def sha256(path: Path) -> str:
    """Return the SHA-256 of a file, read in blocks so size does not matter."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def count_rows(path: Path) -> int:
    """Return the number of data rows, counted with csv.reader.

    Not by counting newlines: free-text columns such as notes.notes carry
    embedded line breaks inside quoted fields, and a newline count would report
    them as extra rows.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def describe(path: Path) -> dict:
    return {"sha256": sha256(path), "rows": count_rows(path), "bytes": path.stat().st_size}


def git_commit(repo: Path) -> str | None:
    """Return the checked-out commit of the sister repository, if git is there.

    Best effort on purpose: the origin is worth recording, but a missing git or
    a downloaded ZIP instead of a clone is no reason to refuse the handover.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


# --------------------------------------------------------------------------
# Pull
# --------------------------------------------------------------------------
def pull(source: Path) -> int:
    """Copy the cut out of the sister checkout and write the manifest."""
    src_tables = source / "data" / "derived" / "tables"
    src_queries = source / "postgres" / "queries.yaml"

    if not src_tables.is_dir():
        raise SystemExit(
            f"ERROR: {src_tables} does not exist.\n"
            f"       Run `python py/main.py export` in {source.name} first."
        )
    if not src_queries.exists():
        raise SystemExit(f"ERROR: {src_queries} does not exist")

    stems = site_stems(src_queries)
    missing = [f"{stem}.csv" for stem in stems.values()
               if not (source / "data" / "derived" / f"{stem}.csv").exists()]
    if missing:
        raise SystemExit(
            f"ERROR: the export in {source.name} is incomplete, missing: "
            f"{', '.join(missing)}\n"
            f"       Run `python py/main.py export` there before pulling."
        )

    # Replaced wholesale rather than merged: a file that disappeared from the
    # export upstream must disappear here too, or the next run would build
    # triples from something that is no longer part of the cut.
    if TARGET.exists():
        shutil.rmtree(TARGET)
    (TARGET / "tables").mkdir(parents=True)
    (TARGET / "sites").mkdir(parents=True)

    manifest: dict = {
        # No local path here. The manifest is versioned, and an absolute path is
        # a property of the machine that ran the pull, not of the data; it would
        # differ between clones and show up as a diff that means nothing. The
        # commit identifies the origin, and it does so everywhere.
        "source": {
            "repository": source.name,
            "commit": git_commit(source),
        },
        "sites": sorted(stems),
        "files": {},
    }

    for path in sorted(src_tables.glob("*.csv")):
        target = TARGET / "tables" / path.name
        shutil.copyfile(path, target)
        manifest["files"][f"tables/{path.name}"] = describe(target)
        print(f"  tables/{path.name:<32s} {manifest['files'][f'tables/{path.name}']['rows']:>7d} rows")

    print()
    for site_id, stem in sorted(stems.items()):
        path = source / "data" / "derived" / f"{stem}.csv"
        target = TARGET / "sites" / f"{stem}.csv"
        shutil.copyfile(path, target)
        entry = describe(target)
        manifest["files"][f"sites/{stem}.csv"] = entry
        print(f"  sites/{stem + '.csv':<33s} {entry['rows']:>7d} rows  site {site_id}")

    shutil.copyfile(src_queries, TARGET / "queries.yaml")
    manifest["files"]["queries.yaml"] = describe_text(TARGET / "queries.yaml")

    # sort_keys and a trailing newline: the manifest is versioned, and a diff
    # should show what changed in the data, not how a dict happened to iterate.
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")

    print(f"\n  manifest: {MANIFEST.relative_to(ROOT)}")
    commit = manifest["source"]["commit"]
    print(f"  origin  : {source.name}"
          + (f" @ {commit[:12]}" if commit else " (no git commit recorded)"))
    return 0


def describe_text(path: Path) -> dict:
    """Fingerprint a non-CSV file. Row counting would be meaningless there."""
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def site_stems(queries_file: Path) -> dict[int, str]:
    """Return site_id -> output stem from the export definition."""
    import yaml

    with open(queries_file, encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    stems = {int(k): v for k, v in (spec.get("per_site_stems") or {}).items()}
    sites = spec.get("sites") or []
    missing = [s for s in sites if s not in stems]
    if missing:
        raise SystemExit(f"ERROR: {queries_file.name}: no stem for site(s) {missing}")
    return {s: stems[s] for s in sites}


# --------------------------------------------------------------------------
# Verify
# --------------------------------------------------------------------------
def verify() -> int:
    """Check the files here against the manifest. No sister checkout needed."""
    if not MANIFEST.exists():
        print(f"  ERROR: {MANIFEST.relative_to(ROOT)} not found; run without "
              f"--verify to pull the cut first", file=sys.stderr)
        return 1

    with open(MANIFEST, encoding="utf-8") as handle:
        manifest = json.load(handle)

    ok = True
    for name, entry in sorted(manifest["files"].items()):
        path = TARGET / name
        if not path.exists():
            print(f"  {name:<45s} MISSING")
            ok = False
            continue
        digest = sha256(path)
        if digest != entry["sha256"]:
            print(f"  {name:<45s} CHANGED since the pull")
            ok = False
            continue
        print(f"  {name:<45s} OK")

    # Files nobody put in the manifest are the other half of the question: a
    # stray CSV here would be read by the generator and stand in no relation to
    # any query.
    known = set(manifest["files"])
    for path in sorted(TARGET.rglob("*")):
        if path.is_file() and path != MANIFEST:
            name = path.relative_to(TARGET).as_posix()
            if name not in known:
                print(f"  {name:<45s} NOT IN MANIFEST")
                ok = False

    commit = manifest["source"].get("commit")
    print(f"\n  origin  : {manifest['source']['repository']}"
          + (f" @ {commit[:12]}" if commit else " (no git commit recorded)"))
    print(f"  sites   : {', '.join(str(s) for s in manifest['sites'])}")
    if not ok:
        print("\n  the cut does not match its manifest; do not build RDF from it",
              file=sys.stderr)
    return 0 if ok else 1


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--from", dest="source", metavar="PATH",
                        help="path to the sisal-db-v3 checkout")
    parser.add_argument("--verify", action="store_true",
                        help="check the cut against its manifest and exit")
    args = parser.parse_args(argv)

    if args.verify:
        if args.source:
            print("  note   : --from is ignored with --verify", file=sys.stderr)
        print(f"  target : {TARGET.relative_to(ROOT)}\n")
        return verify()

    source = find_source(args.source)
    print(f"  source : {source}")
    print(f"  target : {TARGET.relative_to(ROOT)}\n")
    return pull(source)


if __name__ == "__main__":
    sys.exit(main())
