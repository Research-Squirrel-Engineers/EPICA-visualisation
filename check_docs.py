#!/usr/bin/env python3
"""check_docs.py - does README-RUN.md still describe this repository?

The run reference is prose, and prose goes stale silently. This repository has
been bitten by that twice already: README.md promised a
``SISAL/rdf/sisal_sites.ttl`` that did not exist for months, and three scripts
each carried their own hand-written list of archaeological findspots which had
drifted apart. Both times the truth lived in a sentence instead of in a check.

So the prose stays hand-written - what a script *writes* is the useful half of
that file and no parser knows it - and this checks the half a parser does know:

- every script with a ``__main__`` block is named in README-RUN.md
- every ``--flag`` those scripts accept appears there too
- no flag is documented that no script accepts any more

It reports and does not fix, in the manner of ``clean.py``: a register that
says what is out of step, leaving the judgement to a person. A missing entry
is a warning, never a failed run - documentation that blocks a pipeline is
documentation people learn to switch off.

Run through main.py's preparation step, or on its own:

    python check_docs.py
    python check_docs.py --verbose
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "README-RUN.md"

#: Directories that hold no entry points. ``ontology`` does hold two, so it is
#: deliberately not here.
SKIP_DIRECTORIES = {".git", ".venv", "__pycache__", "bundle", "dist", "data",
                    "img", "plots", "maps", "example_query"}

#: Scripts that are entry points but have no business in a run reference:
#: this file, and anything the register below marks as dead.
SKIP_SCRIPTS = {
    # A copy of ci_findspots_html.py that overwrote the SISAL generator. It is
    # in clean.py under "scheduled", and documenting it as if it worked would
    # be worse than leaving it out. Decided with the page itself in S3f.
    "archaeo-connect/sisal_arch_html.py",
}

#: Flags a reader never types: argparse's own, and the ones kept only so that
#: an older invocation does not break.
SKIP_FLAGS = {"--help"}

MAIN_BLOCK = re.compile(r'^if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:', re.M)
#: The whole argument list of a call, so that a flag declared alongside a
#: short form is found too: add_argument("-v", "--verbose") would otherwise
#: yield only "-v".
ADD_ARGUMENT = re.compile(r'add_argument\((.*?)\)', re.S)
FLAG_LITERAL = re.compile(r'[\'"](--[a-z][a-z0-9-]*)[\'"]', re.I)
#: f-string flags, as main.py builds its four --<strand>-only switches
ADD_ARGUMENT_FSTRING = re.compile(r'add_argument\(\s*f[\'"]--\{(\w+)\}([a-z0-9-]*)[\'"]',
                                  re.I)
#: A letter has to follow the two dashes, or every table rule in the file
#: ("---") counts as a flag.
DOCUMENTED_FLAG = re.compile(r'(--[a-z][a-z0-9-]*)', re.I)

#: Commands in the examples that are not ours. Their switches would otherwise
#: be read as flags this repository no longer accepts - the reference shows a
#: "git status --short", and --short is git's, not ours.
FOREIGN_COMMANDS = ("git", "set", "robocopy", "powershell", "findstr", "dir",
                    "mkdir", "rmdir", "cd", "pip", "npx", "mmdc", "copy")


def entry_points() -> dict[str, set[str]]:
    """Every runnable script, with the flags it accepts.

    Read as text rather than imported: importing a drawing script to find out
    what it takes would run it.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if any(part in SKIP_DIRECTORIES for part in path.relative_to(ROOT).parts):
            continue
        if relative in SKIP_SCRIPTS or path.name in SKIP_SCRIPTS:
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        if not MAIN_BLOCK.search(source):
            continue
        flags = {flag for call in ADD_ARGUMENT.findall(source)
                 for flag in FLAG_LITERAL.findall(call)}
        for stem, tail in ADD_ARGUMENT_FSTRING.findall(source):
            # The loop variable is the strand name; the names are the one
            # thing a regex cannot resolve, so they come from the module.
            flags.update(f"--{name}{tail}" for name in strand_names(source, stem))
        found[relative] = {f for f in flags if f not in SKIP_FLAGS}
    return found


def strand_names(source: str, variable: str) -> list[str]:
    """The values a ``for <variable> in NAMES`` loop iterates over.

    Only the literal-tuple case, which is the one that occurs here
    (``STRANDS = ("epica", "sisal", "ci", "archaeo")``). Anything else returns
    nothing and the flag simply goes unchecked - a checker that guesses is
    worse than one that admits it does not know.
    """
    loop = re.search(rf'for\s+{variable}\s+in\s+(\w+)\s*:', source)
    if not loop:
        return []
    literal = re.search(rf'^{loop.group(1)}\s*=\s*\(([^)]*)\)', source, re.M)
    if not literal:
        return []
    return re.findall(r'[\'"](\w+)[\'"]', literal.group(1))


def documented() -> tuple[str, set[str]]:
    if not REFERENCE.exists():
        raise FileNotFoundError(f"{REFERENCE} not found")
    text = REFERENCE.read_text(encoding="utf-8")
    flags: set[str] = set()
    for line in text.splitlines():
        # A single line can hold several commands chained with "&", and only
        # some of them are ours; each part is judged on its own.
        for part in line.split("&"):
            token = part.strip().lstrip("`").split(" ", 1)[0].lower()
            if token in FOREIGN_COMMANDS:
                continue
            flags.update(DOCUMENTED_FLAG.findall(part))
    return text, {f for f in flags if f not in SKIP_FLAGS}


def check(verbose: bool = False) -> int:
    """Compare, report, return the number of findings."""
    scripts = entry_points()
    text, documented_flags = documented()

    missing_scripts = [name for name in scripts if name not in text
                       and Path(name).name not in text]
    used_flags = {flag for flags in scripts.values() for flag in flags}
    missing_flags = sorted(used_flags - documented_flags)
    stale_flags = sorted(documented_flags - used_flags)

    findings = len(missing_scripts) + len(missing_flags) + len(stale_flags)

    if verbose:
        print(f"  {len(scripts)} entry points, {len(used_flags)} flags")
        for name in sorted(scripts):
            flags = ", ".join(sorted(scripts[name])) or "no flags"
            print(f"    {name:<40} {flags}")

    if not findings:
        print(f"  ✓ README-RUN.md covers all {len(scripts)} entry points and "
              f"{len(used_flags)} flags")
        return 0

    print(f"  ⚠  README-RUN.md is {findings} item(s) behind the code:")
    for name in missing_scripts:
        print(f"     script not documented: {name}")
    for flag in missing_flags:
        owners = ", ".join(sorted(n for n, f in scripts.items() if flag in f))
        print(f"     flag not documented:   {flag}  ({owners})")
    for flag in stale_flags:
        print(f"     documented, no longer accepted by any script: {flag}")
    print("     - a warning, not an error. Nothing is blocked by it.")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check README-RUN.md against the scripts it describes.")
    parser.add_argument("--verbose", action="store_true",
                        help="list every entry point and its flags")
    args = parser.parse_args()
    try:
        check(verbose=args.verbose)
    except FileNotFoundError as error:
        print(f"  ⚠  {error}")
    # Always zero: this never fails a run.
    return 0


if __name__ == "__main__":
    sys.exit(main())
