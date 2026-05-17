SRC = src

.PHONY: install run debug clean lint lint-strict check-uv

check-uv:
	@which uv > /dev/null 2>&1 || (echo "Error: 'uv' is not installed." && echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" && echo "Or on Windows: powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"" && exit 1)

install: check-uv
	uv sync

run: check-uv
	uv run python -m src $(ARGS)

debug: check-uv
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
		python3 -m mypy $(SRC) --strict