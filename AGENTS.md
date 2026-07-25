# Agent Guide: Python Environment and Execution with uv

This project uses [uv](https://github.com/astral-sh/uv) as the primary tool for Python package management, virtual environment management, and tool execution.

## Why uv?

`uv` is an extremely fast Python package installer and resolver written in Rust. It replaces `pip`, `pip-tools`, `pipenv`, `poetry`, and `virtualenv` with a single, unified, and blazing-fast tool.

## Common Workflows

### 1. Environment Setup

To initialize or synchronize the virtual environment with the dependencies defined in `pyproject.toml` and `uv.lock`, run:

```bash
uv sync
```

This project targets Python 3.12 via `.python-version`. `uv sync` will automatically create a virtual environment in the `.venv` directory and install all required dependencies, including editable development dependencies (like `jittor` and `jittor-geometric` located in `third_party/`).

### 2. Running Commands and Scripts

You can run any command or script within the context of the virtual environment without manually activating it by using `uv run`:

* **Run a Python script:**
  ```bash
  uv run python scripts/tune_temporal_graph.py
  ```
* **Run tests:**
  ```bash
  uv run pytest
  ```
* **Run project CLI entry points:**
  ```bash
  uv run jgrec-build --help
  ```

### 3. Linting (Required Habit)

Run ruff after any code change and before considering a task done:

```bash
uv run ruff check          # 检查
uv run ruff check --fix    # 自动修复安全项
```

Configuration lives in `pyproject.toml` under `[tool.ruff]` (line-length 120, py312). Notes:

* `RUF001/RUF002/RUF003` are intentionally ignored — project comments and docstrings are in Chinese, full-width punctuation is deliberate.
* `E501` is ignored; line width is left to the formatter.
* Intentional deviating code should carry a scoped `noqa` comment (e.g. deferred imports guarded by `pytest.importorskip` use `# noqa: PLC0415`).

### 4. Managing Dependencies

To add or remove dependencies, use `uv add` or `uv remove`. This will automatically update `pyproject.toml` and regenerate `uv.lock`.

* **Add a production dependency:**
  ```bash
  uv add package-name
  ```
* **Add a development dependency:**
  ```bash
  uv add --group dev package-name
  ```
* **Remove a dependency:**
  ```bash
  uv remove package-name
  ```

### 5. Activating the Environment (Optional)

If you prefer traditional virtual environment activation, you can activate the `.venv` created by `uv`:

```bash
source .venv/bin/activate
```
