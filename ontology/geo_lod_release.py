"""The one hand-kept release date, and nothing else.

It sat in ``geo_lod_utils`` until the figures needed it too. That module
imports rdflib, and ``geo_lod_figures`` is deliberately free of it - drawing
and triples do not share dependencies. A constant is a poor reason to break
that rule, and a second copy of the string would be a worse one, so the
constant moved here and both import it.

``geo_lod_utils`` re-exports it, so every existing
``from geo_lod_utils import GEO_LOD_RELEASE`` keeps working.
"""

GEO_LOD_RELEASE: str = "2026-08-08"
