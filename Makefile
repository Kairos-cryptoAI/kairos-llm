.PHONY: install lint test format run
install:
	pip install -e ".[dev]"
format:
	ruff format kairos_llm tests
lint:
	ruff check kairos_llm tests
test:
	pytest -q
run:
	python -m kairos_llm
