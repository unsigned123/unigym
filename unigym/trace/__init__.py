"""unigym.trace -- standardized per-tick debug bundles (debug toolkit A).

docs/contracts/trace_contract.md is the format twin.  Layout: one
``manifest.json`` + one ``tick_NNNNNN.npz`` per recorded (sampled) tick.
Everything here is numpy-only -- the recorder drives the live run through
the daemon control API (console eval-dump transport), the diff engine and
the shard importer are pure file readers; no backend imports anywhere.
"""

from .io import TraceReader, load_manifest, tick_path, write_manifest

__all__ = ["TraceReader", "load_manifest", "write_manifest", "tick_path"]
