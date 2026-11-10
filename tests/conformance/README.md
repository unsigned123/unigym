# tests/conformance

Cross-backend behavior that must agree through the shared `EnvBackend`
contract. The suite uses the deterministic pure-torch fake in
`tests/_support/fakes/backend.py`; conformance tests import it only from
`tests._support.fakes.backend`.

Real backends are selected by CI lanes. No real simulator, GPU or external
asset belongs in this directory.
