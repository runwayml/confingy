# confingy

An implicit configuration system for Python. Tracks constructor arguments, supports lazy instantiation, and serializes/deserializes configurations to JSON ("fingys").

## Project structure

- `src/confingy/` — library source (src layout)
  - `tracking.py` — core `@track`, `lazy`, `lens`, `Lazy[T]`, `disable_validation`
  - `fingy.py` — `serialize_fingy`, `deserialize_fingy`, `save_fingy`, `load_fingy`, `transpile_fingy`, `prettify_fingy`
  - `serde.py` — serialization/deserialization internals (handlers, registry, keys)
  - `exceptions.py` — `ValidationError`, `SerializationError`, `DeserializationError`
  - `mypy_plugin.py` — mypy plugin for tracked classes
  - `cli/` — typer CLI (`confingy serialize`, `confingy transpile`, `confingy viz`)
  - `viz/` — graph visualization (optional `[viz]` extra: fastapi + uvicorn)
  - `utils/` — hashing, imports, type checks
- `tests/` — pytest test suite
- `examples/` — usage examples (dataloading, lens, training loop, transpile, pydantic validation)
- `docs/` — mkdocs documentation

## Development

### Setup

```bash
uv sync --group dev --extra viz
```

### Running things

Always use `uv run` or `make` targets to run commands. Never activate the venv manually or use pip.

| Command | What it does |
|---------|-------------|
| `make pytest` | Run tests (`uv run --group dev --extra viz pytest -vv`) |
| `make mypy` | Type check (`uv run --group dev --extra viz mypy -p confingy`) |
| `make lint` | Lint (`uv run --group dev --extra viz ruff check`) |
| `make format-check` | Check formatting (`uv run --group dev --extra viz ruff format --check`) |
| `make docs` | Build mkdocs site |
| `make serve-docs` | Serve docs locally |

To run a one-off Python script: `uv run python myscript.py`

### Testing

- Tests are in `tests/` and run with `make pytest`
- CI runs against Python 3.10, 3.11, 3.12, 3.13
- The local `.python-version` is 3.10.12

### Linting and formatting

- Ruff linting and formatting is enforced automatically via a Claude Code hook (`.claude/settings.json`) — it runs `ruff check --fix` and `ruff format` after every Python file edit.
- Never use `py_compile`
- Per-file ignores: `ANN001` is suppressed in `examples/` and `tests/`

### Type checking

- Uses **mypy** with the `confingy.mypy_plugin` plugin (configured in `pyproject.toml`)
- Run with `make mypy`

## Coding conventions

- Google style docstrings
- Avoid `typing.Any` unless absolutely necessary
- Minimize dependencies — the core has only pydantic, typer, and typing_extensions
- Prefer composition and dependency injection over inheritance
- `uv` is the package manager — never use pip directly

## CI

GitHub Actions workflows on push/PR to `main`:
- `pytest.yml` — tests across Python 3.10–3.13
- `mypy.yml` — type checking
- `ruff.yml` — lint + format check
- `docs.yml` — docs build
- `publish.yml` — PyPI publishing on release
