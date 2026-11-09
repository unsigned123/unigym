"""Test-support faces that ship with the product.

The conformance kit lives here rather than under ``tests/`` so the
contract it asserts is available to a cut source package as well: a
source lane calls the same kit the core synthetic providers use.
"""

from .conformance import (  # noqa: F401
    ConformanceError,
    ConformanceManifest,
    StructureFacts,
    check_build_determinism,
    check_compile_determinism,
    check_dispatch_integrity,
    check_naming,
    check_region_namespace,
    check_scale_face,
    check_structure,
    run_conformance,
)
