
py = python3

run:
	uv run $(py) -m src

install:
	@pip install poetry
	poetry install

debug:
	uv run ${py} -m src

clean:
	@rm -rf */__pycache__
	@rm -rf .mypy_cache
lint:
	flake8 src
	mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
