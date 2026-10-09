# Installation

## For users

Install the published package, [`spherinator`](https://pypi.org/project/spherinator/), from PyPI.

### Using pip

We recommend installing into a virtual environment rather than system-wide:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install spherinator
```

### Using uv

If you have [uv](https://docs.astral.sh/uv/) installed, it manages the virtual environment for you:

```bash
uv venv
source .venv/bin/activate

uv pip install spherinator
```

## For developers

Follow this path if you want to contribute to Spherinator, fix a bug, add a new extractor/transformation/loader, or run the test suite locally.

### Clone the repository

```bash
git clone https://github.com/HITS-AIN/Spherinator.git
cd Spherinator
```

### Set up the development environment

uv reads `pyproject.toml` and creates an isolated virtual environment with the exact locked dependency versions from `uv.lock`:

```bash
uv sync --extra dev
```

This installs Spherinator itself in editable mode plus the `dev` extras (`pytest`, `ruff`, `ipykernel`, ...), so any change you make to the source under `src/spherinator/` is picked up immediately.

You don't need to manually activate the virtual environment — prefix commands with `uv run` and uv takes care of it. If you prefer an activated shell:

```bash
source .venv/bin/activate
```

### Run the test suite

```bash
uv run --extra dev pytest
```

Run a single test file or test:

```bash
uv run --extra dev pytest tests/test_fits_converter.py
uv run --extra dev pytest tests/test_fits_converter.py::test_some_case
```

### Lint and format

Spherinator uses [ruff](https://docs.astral.sh/ruff/) for both linting and formatting.

```bash
uv run --extra dev ruff check --no-fix
uv run --extra dev ruff format --check
```

Run these before opening a pull request.

### Pre-commit hooks

The same ruff checks can run automatically on every `git commit` via [pre-commit](https://pre-commit.com/), configured in `.pre-commit-config.yaml`. Install the git hook once after cloning:

```bash
uv run --extra dev pre-commit install
```

From then on, each `git commit` runs `ruff check` and `ruff format` on the staged files and aborts the commit if a hook fails. `ruff format` reformats files in place, while lint errors from `ruff check` have to be fixed by hand. In both cases, review the changes, `git add` the files again, and re-commit.

The hook only checks staged files, so run it against the whole repository after changing `.pre-commit-config.yaml` or when setting it up for the first time:

```bash
uv run --extra dev pre-commit run --all-files
```

To bypass the hook for a single commit (not recommended), use `git commit --no-verify`.

### Build the documentation

The documentation (this site) is built with [Sphinx](https://www.sphinx-doc.org/) from the `docs/` directory using the `docs` extra:

```bash
uv sync --extra docs
uv run --extra docs sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser to preview it. For live-reloading while editing:

```bash
uv run --extra docs sphinx-autobuild docs docs/_build/html
```
