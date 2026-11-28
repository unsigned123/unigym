"""Pure backend capability vocabulary for neutral layers.

This package owns the backend-neutral vocabulary that neutral layers may use
without importing the backend composition graph:

* :mod:`unigym.capabilities.traits` — trait/scope vocabulary, model-edit
  fields, per-backend trait tables and canonical backend names;
* :mod:`unigym.capabilities.specs` — pure ``BackendSpec`` metadata and the
  canonical spec table;
* :mod:`unigym.capabilities.versions` — stdlib-only engine version probes;
* :mod:`unigym.capabilities.selection` — canonical backend selection and
  stack-probe validation.

Nothing in this package imports ``unigym.backends`` (concrete backends or the
composition root), ``unigym.registration`` or ``unigym.run``.  Backend
construction faces stay in the composition layer
(:mod:`unigym.backends.bindings` and :mod:`unigym.backends.compose`).
"""
