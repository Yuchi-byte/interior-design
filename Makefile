# A wrapper over the gate in CLAUDE.md, nothing more: every recipe is the
# command from that file verbatim, so `make check` and CI can never mean
# different things. Requires no tool the repo does not already need — uv, plus
# the make that ships with every machine.

SOURCES := packages/roomgen-geometry/src packages/roomgen-guide/src
TESTS := packages/roomgen-geometry/tests packages/roomgen-guide/tests
LINT_PATHS := $(SOURCES) $(TESTS)

.PHONY: help sync lint format-check format typecheck test check hooks

help:
	@echo "sync         build the one repo venv (CPU torch)"
	@echo "check        the full gate: lint, format-check, typecheck, test"
	@echo "lint         ruff check"
	@echo "format       ruff format (rewrites)"
	@echo "format-check ruff format --check"
	@echo "typecheck    mypy over both src trees"
	@echo "test         pytest, unfiltered (CI deselects the models marker)"
	@echo "hooks        install the pre-commit hook"

sync:
	uv sync

lint:
	uv run ruff check $(LINT_PATHS)

format-check:
	uv run ruff format --check $(LINT_PATHS)

format:
	uv run ruff format $(LINT_PATHS)

typecheck:
	uv run mypy $(SOURCES)

# Unfiltered on purpose. CI runs `-m "not models"` because its runner has no
# weights; a dev machine does, and those tests are the ones that exercise the
# real backends.
test:
	uv run pytest -q

check: lint format-check typecheck test

hooks:
	uv run pre-commit install
