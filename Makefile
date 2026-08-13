UV ?= uv

.PHONY: install lint format format-check typecheck security test build run all
install:
	$(UV) sync --locked
format:
	$(UV) run --locked ruff format kairos_llm tests
format-check:
	$(UV) run --locked ruff format --check kairos_llm tests
lint:
	$(UV) run --locked ruff check kairos_llm tests
typecheck:
	$(UV) run --locked mypy kairos_llm
security:
	$(UV) run --locked bandit -q -r kairos_llm -x tests
test:
	$(UV) run --locked pytest -q --tb=short
build:
	$(UV) build --no-sources
run:
	$(UV) run --locked python -m kairos_llm
all: lint format-check typecheck security test build
