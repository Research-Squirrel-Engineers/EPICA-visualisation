"""
geo_lod_captions.py
===================
The caption layer: what used to be printed across the top of a figure now
lives in a ``captions.yaml`` next to the script that draws it, one file per
strand (``EPICA/captions.yaml``, later ``SISAL/captions.yaml``).

Why the caption left the image
------------------------------
A heading burnt into an SVG cannot be translated, cannot be indexed, and has
to be cropped out by hand when the figure goes into a paper that supplies its
own caption. Outside the image it is text: quotable, diffable, and available
to the metadata layer. The field names follow ``captions.yaml`` in
``wdttest-tables`` - ``caption``, ``captiondetail``, ``license`` - so that the
two repository families describe an output the same way.

Generated, and still editable
-----------------------------
Captions state things the drawing code knows and prose does not keep up with:
which chronology an age axis belongs to, which filter ran, how many points
there are, which stages a record cannot cover. Hand-written, exactly those
facts go stale silently, and a wrong caption is not something a reader can
notice.

So the generator writes them. But an entry is not overwritten once it has been
edited by hand: each entry keeps the text the generator last produced under
``generated``, and if ``caption`` no longer matches it, the caption is treated
as the author's and left alone - while ``generated`` is refreshed, so the diff
shows what the machine would say now. Editing a caption therefore means
editing ``caption`` and nothing else, and the file still tells you where prose
and data have drifted apart.

The file is written in a fixed key order with sorted entries, so two runs on
unchanged inputs produce identical bytes like everything else here.
"""

from __future__ import annotations

import os
from typing import Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for the caption layer.  pip install pyyaml"
    ) from exc

#: Key order inside one entry. Fixed rather than alphabetical, because a
#: reader opening the file should meet the caption first and the bookkeeping
#: last.
FIELD_ORDER = (
    "caption",
    "captiondetail",
    "generated",
    "license",
    "sources",
)


class CaptionFile:
    """Collects captions during a run and merges them into a YAML file."""

    def __init__(self, path: str, header: str = ""):
        self.path = path
        self.header = header
        self.entries: dict[str, dict] = {}

    def add(
        self,
        key: str,
        caption: str,
        captiondetail: str = "",
        license: str = "",
        sources: Iterable[str] = (),
    ) -> None:
        """Register one output. *key* is the file name without extension."""
        entry = {"caption": caption, "generated": caption}
        if captiondetail:
            entry["captiondetail"] = captiondetail
        if license:
            entry["license"] = license
        if sources:
            entry["sources"] = list(sources)
        self.entries[key] = entry

    def _load_existing(self) -> dict:
        if not os.path.exists(self.path):
            return {}
        with open(self.path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}

    def write(self, verbose: bool = True) -> int:
        """Merge with what is on disk and write. Returns the number of entries
        whose caption is hand-written and was therefore kept."""
        existing = self._load_existing()
        kept = 0

        for key, entry in self.entries.items():
            old = existing.get(key)
            if not isinstance(old, dict):
                continue
            old_caption = old.get("caption")
            # Edited by hand exactly when the caption on disk is not the text
            # the generator produced last time.
            if old_caption and old_caption != old.get("generated"):
                entry["caption"] = old_caption
                kept += 1
            # Hand-written extras survive; the generator does not supply them.
            for field in ("captiondetail", "license", "sources"):
                if field in old and field not in entry:
                    entry[field] = old[field]

        lines = []
        if self.header:
            lines.extend(f"# {line}".rstrip() for line in self.header.split("\n"))
            lines.append("")

        for key in sorted(self.entries):
            entry = self.entries[key]
            lines.append(f"{key}:")
            for field in FIELD_ORDER:
                if field not in entry:
                    continue
                value = entry[field]
                if isinstance(value, list):
                    lines.append(f"  {field}:")
                    lines.extend(f"    - {item}" for item in value)
                else:
                    lines.append(f"  {field}: {_quote(value)}")
            lines.append("")

        with open(self.path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines).rstrip("\n") + "\n")

        if verbose:
            note = f", {kept} hand-written kept" if kept else ""
            print(f"  ✓ Captions: {self.path} ({len(self.entries)} entries{note})")
        return kept


def _quote(value: str) -> str:
    """Double-quoted YAML scalar, escaped just enough to round-trip."""
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'
