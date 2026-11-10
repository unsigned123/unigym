PYTHON ?= .venv/bin/python

.PHONY: test-fast test test-full test-integration

test-fast:
	$(PYTHON) -m pytest -m fast -x -q

test:
	$(PYTHON) -m pytest -q

test-full:
	$(PYTHON) -m pytest tests -q

# Phase 0 only: tests/integration has no collected tests yet, so pytest exits 5
# (no tests collected). Tolerate that code and surface the empty layer.
test-integration:
	@$(PYTHON) -m pytest tests/integration -q; rc=$$?; \
	if [ $$rc -eq 5 ]; then \
		echo "test-integration: no integration tests collected yet (Phase 0 rc-5 tolerance)"; \
	elif [ $$rc -ne 0 ]; then \
		exit $$rc; \
	fi
