# tests/_support

Shared, non-collected test infrastructure (helpers, factories, environment
resolution, fakes). This directory contains no tests.

It is a regular Python package (`__init__.py`) so the harness can import it as
`tests._support.markers`; the `tests` root itself is an implicit namespace
package on the pytest `pythonpath`.

The marker policy module and the package marker are the base infrastructure.
The one backend fake lives in `fakes/` with its own manifest; factories are
added only when a real external boundary requires them, following
`docs/testing.md` section 10.
