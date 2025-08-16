---
title: "Python Tooling in 2025"
summary: "
    A 2025 guide to the best Python tools for quants---from fast package managers and powerful linters to type checkers and test runners. Includes example configs.
"

date: "2025-08-14"
type: paper
mathjax: false
authors:
    - Adrian Letchford
categories:
    - engineering
---

Today, Python's ecosystem has an abundance of tooling designed to support every aspect of the development workflow. From dependency management to static analysis, from linting to environment setup, there are more options than ever!

This article presents a modern, opinionated toolchain for Python development in quantitative research and development. The focus is on **code quality**, ensuring that your codebase is clean, tested, typed, and consistent.

Each tool has been selected for its performance, reliability, and integration with the broader Python and data science ecosystem. At the end of the article, I've included a skeleton config that puts everything together into a coherent project structure--ready to use, or adapt to your own trading or research needs.

Whether you're starting fresh or refining an existing workflow, this guide should provide a solid foundation for modern, professional-grade Python development in quant environments.

<sectionbreak></sectionbreak>

Four pillars underpin a high-quality Python codebase: **package management**, **code linting & formatting**, **static typing**, and **testing**. Each has a crowded ecosystem of tools, but the modern ones worth adopting share two traits: speed and integration with `pyproject.toml`.

Speed matters because modern development often means running checks and builds many times a day. A tool that executes in seconds instead of minutes directly shortens feedback loops, encourages frequent use, and keeps you in the flow.

The `pyproject.toml` file was introduced by [PEP 518](https://peps.python.org/pep-0518/) (and later [PEP 517](https://peps.python.org/pep-0517/))[^1] to give Python projects a single, tool-agnostic place to declare build requirements and configuration. In the past, settings were scattered across `setup.cfg`, `tox.ini`, and assorted dotfiles. But now modern tools have converged on `pyproject.toml` as the canonical source of truth. In practice, it's the front door to your project's tooling: one file that defines how the project is built, tested, linted, and run.

Let's examine each of the four pillars in turn and see how to implement them effectively in a modern Python stack.

# Package management

Package management controls how your project declares, resolves, installs, and reproduces dependencies. Done well, it reduces "works on my machine" risk and keeps CI/CD, development, and research sandboxes aligned.

Historically, the de-facto approach with `pip` was a plain `requirements.txt`---a flat list of packages to install. It usually pins only your top-level dependencies (e.g., `numpy`, `pandas`), leaving [transitive versions](https://en.wikipedia.org/wiki/Transitive_dependency) to float. Rebuilding the environment days or weeks later can resolve a different dependency tree, causing surprise breakages and hard-to-trace drift across machines, CI, or research environments. Fine for quick scripts; brittle for anything that needs reproducibility.

The modern approach is to keep a manifest of your top-level dependencies and automatically generate a **lock file** that pins the *entire* dependency tree (including transitive packages). You commit the lock file and rebuild from it. Installs are now completely deterministic. Upgrades or new packages become deliberate: adjust the manifest, regenerate the lock, and review the diff. Dependency drift is eliminated.

## UV

In 2025, [uv](https://docs.astral.sh/uv/) is the fastest, most complete option for day-to-day Python dependency management. It handles project creation, dependency resolution, a cross-platform lockfile, virtualenvs, tool running, and even Python runtime installation---while staying compatible with pyproject.toml. It's built for speed (think milliseconds-level operations, written in Rust) and reproducibility, which matters when you're iterating on models and shipping to CI frequently.

Core workflow:

### Install
```bash
pip install uv
```
From this point on, only use `uv`, not `pip`. For other installation options, check out the ["Installing uv"](https://docs.astral.sh/uv/getting-started/installation/) guide.

### Create a new project
```bash
uv init myproject
```
This adds a `pyproject.toml` file.
### Add dependencies
```bash
cd myproject
uv add pandas numpy scipy 
uv add --dev pytest ruff   # dependency groups for clean separation
```
This adds the dependencies to `pyproject.toml` and updates the lock file.

### Create/sync the environment to the lockfile
```bash
uv sync
```
This will automatically fully a [virtual environment](https://docs.python.org/3/library/venv.html) for you in  `.venv`.

### Execute inside the managed venv
```bash
uv run python scripts/backtest.py
```
No manual activation needed!


## Alternatives (and why I prefer uv)

The top alternative is [poetry](https://python-poetry.org/) which remains a strong, integrated solution (dependency management + build/publish), with a familiar UX and mature ecosystem.

I prefer uv for a modern quant project because it's significantly faster. For teams running frequent CI and spinning up many ephemeral envs, the speed and lockfile model are tangible wins. 

# Code linting & formatting

You write code; you read code. Some is clean and easy to scan. Other code---sometimes your own---resists a first pass. Unused imports, awkward line breaks, dead code, vague names, and stray typos create confusion, hide bugs, and fuel [bikeshedding](https://en.wikipedia.org/wiki/Law_of_triviality). Linting & formatting tools keep your codebase in a consistent style, catch small mistakes early, and frees you to think about models, not minutiae.

**Linting** statically analyzes your code for defects and style issues: unused imports, shadowed variables, dead code, unsafe patterns, complexity, naming conventions, and more.

For example, using multiple `isinstance` calls for the same object is uncessary and verbose. This is caught using a rule called [duplicate-isinstance-call](https://docs.astral.sh/ruff/rules/duplicate-isinstance-call/):

{{<figure src="lint-example.png" width="medium" >}} {{</figure>}}

**Formatting** is complementary: it rewrites code to a consistent style (spacing, quotes, import order, line wraps) so devs don't argue about it and diffs stay clean.

## Ruff

Use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting. It's fast (written in Rust), batteries-included (replaces flake8 + isort and can replace Black), and reads all config from `pyproject.toml`. Most of all, it will lint and format your notebooks!

Core workflow:

### Install

With `uv` (recommended), add `ruff` as a dev dependency:
```bash
uv add --dev ruff
```

### Check
Check you code with:
```bash
uv run ruff check .
```

### Fix
Automatically fix many of the linting and formatting errors with:
```bash
uv run ruff check . --fix    # lint + auto-fix
uv run ruff format .         # code formatter
```

## Config

The team behind Ruff, [Astral](https://astral.sh/), provide [configuration docs](https://docs.astral.sh/ruff/configuration/) and [rule docs](https://docs.astral.sh/ruff/rules/). But to get you started, here is a sensible `pyproject.toml` config for a quant codebase:


```toml
[tool.ruff]
line-length = 100

# If you hit false positives in quick-and-dirty
# experiment folders, exclude these here.
exclude = [
    ".venv",
]

[tool.ruff.lint]
# Full list of rules here:
# https://docs.astral.sh/ruff/rules/
select = [
    # Core rules
    "E",     # pycodestyle errors
    "F",     # Pyflakes
    "UP",    # pyupgrade
    "I",     # isort
    
    # Quality and style
    "B",     # flake8-bugbear
    "Q",     # flake8-quotes
    "SIM",   # flake8-simplify
    "FLY",   # flynt
    "PERF",  # Perflint
]
```

# Static typing

Static type checking analyzes your code without running it to ensure values match expectations. In Python, those expectations are written as type annotations in function signatures and variables; the checker reads these (and infers the rest) to catch mismatches--e.g., a function expecting a `pd.Series` won't silently receive a `pd.DataFrame`.

Those same annotations double as executable documentation: they make refactors safer and surface edge cases in data pipelines. On large research codebases, the result is fewer "works in a notebook, breaks in production" failures.

Some quick examples of typing errors:
```python
# 1) Series vs DataFrame mixup
def zscore(x: pd.Series) -> pd.Series: ...
zscore(df)  # ❌ flagged: DataFrame where Series expected

# 2) Optional values used as definite
def sharpe(ret: pd.Series | None) -> float:
    return ret.mean() / ret.std()   # ❌ flagged: 'ret' could be None

# 3) Wrong argument types
def load_csv(path: Path | str) -> pd.DataFrame: ...
load_csv(123)  # ❌ flagged: int is not Path | str
```

## Pyright

Use [Pyright](https://github.com/microsoft/pyright) as your static type checker: it's fast, mature, and powers VS Code's Pylance, so editor feedback is excellent. It also plays well with pandas-stubs and numpy.typing, which improves day-to-day ergonomics in quant code. 

Core workflow:

### Install
```bash
uv add --dev pyright pandas-stubs
```
Add pandas-stubs (already shown); it materially improves Pandas ergonomics.

### Check for typing errors
```bash
uv run pyright
```

## Config
Pyright is configurable in the `pyproject.toml` file. Full [documentation here](https://microsoft.github.io/pyright/#/configuration). To get you started, here is a config suitable for a quant project:

```toml
[tool.pyright]
# Turn useful diagnostics up early
reportUnknownVariableType = true
reportUnknownMemberType = true
reportUnknownArgumentType = true
reportOptionalSubscript = true
reportUnusedImport = "error"
reportMissingTypeStubs = true
```

## Alternatives

[Pyrefly](https://pyrefly.org/) --- very fast and has nice migration tooling (it can auto-insert ignores so you can enable it and fix issues incrementally). In practice, it still struggles with Pandas-heavy code; you may find yourself fighting the checker (e.g., `df = df.loc[idx, :]` narrows to `(Series | Unknown)` and fails).

[Astral ty](https://docs.astral.sh/ty/) --- a new Rust type checker from the Ruff/uv team. It's in [preview/alpha](https://github.com/astral-sh/ty/releases) today; promising performance, but not production-ready yet. Worth tracking and trying.


# Testing

Automated tests are executable specifications: every run confirms the code still behaves as intended. Without them, every edit becomes a time-consuming, error-prone recheck. In quant teams, skipping tests turns every change into a manual audit; a fast test suite collapses that loop from hours to seconds and lets you make changes and ship with confidence.

## Pytest
pytest is the de-facto test runner for Python. It's simple, fast, and extensible.

Core workflow:

### Install
```bash
uv add --dev pytest
```

### Write a test
```python
# tests/test_sharpe.py
import numpy as np
import pandas as pd

from src.metrics import sharpe  # your code

def test_sharpe_zero_on_zero_returns():
    r = pd.Series(np.zeros(10))
    assert sharpe(r) == 0.0

def test_sharpe_handles_nans():
    r = pd.Series([0.01, np.nan, 0.02, -0.01]).fillna(0)
    assert np.isfinite(sharpe(r))
```

Some best practices here:
* Put test code under a top level folder `tests/`.
* Name test files `test_*.py` and test functions `test_*`.

### Run tests
```bash
uv run pytest tests/
```

## Config

Pytest requires very little configuration. I like to put this into my `pyproject.toml` to make it explict where the tests are and so that I only have to run `uv run pytest`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

# Example config

To bootstrap a new project or modernize an existing one:

1. Add the configuration below to a pyproject.toml at the repository root.
1. Install **uv** (`pip install uv`) or ensure it's available.
1. Create the environment with `uv sync`.

`uv sync` resolves and installs dependencies from your project manifest/lockfile and typically completes in seconds on a fresh project.

```toml
# ================================================
# Project Configuration
# ================================================

[project]
name = "My Project"
version = "0.1.0"
description = "Add your description here"
requires-python = ">=3.13"
dependencies = [
    "numpy>=2.3.2",
    "pandas>=2.3.1",
]

# ================================================
# Development Dependencies
# ================================================

[dependency-groups]
dev = [
    "pandas-stubs>=2.3.0.250703",
    "pyright>=1.1.403",
    "pytest>=8.4.1",
    "ruff>=0.12.9",
]

# ================================================
# Linting and Formatting
# ================================================

[tool.ruff]
line-length = 100

# If you hit false positives in quick-and-dirty
# experiment folders, exclude these here.
exclude = [
    ".venv",
]

[tool.ruff.lint]
# Full list of rules here:
# https://docs.astral.sh/ruff/rules/
select = [
    # Core rules
    "E",     # pycodestyle errors
    "F",     # Pyflakes
    "UP",    # pyupgrade
    "I",     # isort
    
    # Quality and style
    "B",     # flake8-bugbear
    "Q",     # flake8-quotes
    "SIM",   # flake8-simplify
    "FLY",   # flynt
    "PERF",  # Perflint
]

# ================================================
# Static Type Checking
# ================================================

[tool.pyright]
# Turn useful diagnostics up early
reportUnknownVariableType = true
reportUnknownMemberType = true
reportUnknownArgumentType = true
reportOptionalSubscript = true
reportUnusedImport = "error"
reportMissingTypeStubs = true

# ================================================
# Testing
# ================================================

[tool.pytest.ini_options]
testpaths = ["tests"]
```

# Summary

A modern, reliable Python codebase for quant work rests on four pillars: **package management** with `uv`, **linting & formatting** with `ruff`, **static type checking** with `pyright`, and **testing** with `pytest`. Together they deliver reproducible environments, cleaner diffs, earlier bug detection, and safer refactors—so research code promotes to production with fewer surprises.

What to do next:

* Start from the example config, run `uv sync`, and keep the lockfile committed.
* Enable `ruff` and `pyright` in your editor; run them and `pytest` in CI on every PR.
* Grow tests alongside features.
* Review lockfile diffs and lint/type failures like any other change---these are quality signals, not noise.

Adopt this baseline, and you get a fast feedback loop and a codebase that stays readable, testable, and reproducible as it scales.

[^1]: Even though PEP 517 has a smaller number than PEP 518, PEP 517 does indeed referece PEP 518 as the source of the `pyproject.toml` file. It says "Here we define a new style of source tree based around the pyproject.toml file defined in PEP 518..." You can check the documents' post history for their latest publication date.