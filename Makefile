PYTHON ?= .venv/bin/python

.PHONY: test-fast test test-full test-integration

test-fast:
	$(PYTHON) -m pytest -m fast -x -q

test:
	$(PYTHON) -m pytest -q

test-full:
	$(PYTHON) -m pytest tests -q

test-integration:
	$(PYTHON) -m pytest tests/integration -q
