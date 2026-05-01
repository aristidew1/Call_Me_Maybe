SRC = src

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -Wall -m src $(ARGS)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

lint:
		python3 -m flake8 $(SRC)
		python3 -m mypy $(SRC) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
		python3 -m flake8 $(SRC)
		metpython3 -m mypy $(SRC) --strict