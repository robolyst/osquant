---
title: "Python Tooling in 2025"
summary: "
    A 2025 guide to the best Python tools for quants--from fast package managers and powerful linters to devcontainers and task runners. Includes example configs and a ready-to-use project template.
"

date: "2025-08-14"
type: paper
mathjax: false
authors:
    - Adrian Letchford
categories:
    - engineering
acknowledgements: "All figures in this article were made with [Figma](http://figma.com)."
---

<todo>Replace "skeleton repository" with the link</todo>

Today, Python's ecosystem has abundance of tooling designed to support every aspect of the development workflow. From dependency management to static analysis, from linting to environment setup, there are more options than ever!

This article presents a modern, opinionated toolchain for Python development in quantitative research and development. The focus is:

1. **Code quality** - ensuring that your codebase is clean, tested, typed, and consistent.
2. **Developer productivity** - improving the day-to-day development experience with better workflows and automation.

Each tool has been selected for its performance, reliability, and integration with the broader Python and data science ecosystem. To accompany the article, I've published a [skeleton repository](https://www.notion.so/Article-ideas-ab575e137dff46f9a63d73d42a47b769?pvs=21) that puts everything together into a coherent project structure--ready to use, or adapt to your own trading or research needs.

Whether you're starting fresh or refining an existing workflow, this guide should provide a solid foundation for modern, professional-grade Python development in quant environments.

# Code Quality

Four pillars underpin a high-quality Python codebase: **package management**, **code linting & formatting**, **static typing**, and **testing**. Each has a crowded ecosystem of tools, but the modern ones worth adopting share two traits: speed and integration with `pyproject.toml`.

Speed matters because modern development often means running checks and builds many times a day. A tool that executes in seconds instead of minutes directly shortens feedback loops, encourages frequent use, and keeps you in the flow.

The `pyproject.toml` file was introduced by [PEP 518](https://peps.python.org/pep-0518/) (and later [PEP 517](https://peps.python.org/pep-0517/))[^1] to give Python projects a single, tool-agnostic place to declare build requirements and configuration. In the past, settings were scattered across `setup.cfg`, `tox.ini`, and assorted dotfiles. But now modern tools have converged on `pyproject.toml` as the canonical source of truth. In practice, it's the front door to your project's tooling: one file that defines how the project is built, tested, linted, and run.

Let's examine each of the four pillars in turn and see how to implement them effectively in a modern Python stack.

## Package management

Package management controls how your project declares, resolves, installs, and reproduces dependencies. Done well, it shrinks "works on my machine" risk and keeps CI/CD, development, and research sandboxes aligned.

Historically, the de-facto approach with `pip` was a plain `requirements.txt`---a flat list of packages to install. It usually pins only your top-level dependencies (e.g., `numpy`, `pandas`), leaving [transitive versions](https://en.wikipedia.org/wiki/Transitive_dependency) to float. Rebuilding the environment days or weeks later can resolve a different dependency tree, causing surprise breakages and hard-to-trace drift across machines, CI, or research environments. Fine for quick scripts; brittle for anything that needs reproducibility.

The modern approach is to keep a manifest of your top-level dependencies and automatically generate a **lock file** that pins the *entire* dependency tree (including transitive packages). You commit the lock file and rebuild from it. Installs are now completely deterministic. Upgrades or new packages become deliberate: adjust the manifest, regenerate the lock, and review the diff. Dependency drift is eliminated.

### UV

In 2025, [uv](https://docs.astral.sh/uv/) is the fastest, most complete option for day-to-day Python dependency management. It handles project creation, dependency resolution, a cross-platform lockfile, virtualenvs, tool running, and even Python runtime installation---while staying compatible with pyproject.toml. It's built for speed (think milliseconds-level operations) and reproducibility, which matters when you're iterating on models and shipping to CI frequently.

Core workflow:

#### Create a new project
```bash
uv init myproject
```
This adds a `pyproject.toml` file.
#### Add dependencies
```bash
cd myproject
uv add pandas numpy scipy 
uv add --dev pytest ruff   # dependency groups for clean separation
```
This adds the dependencies to `pyproject.toml` and updates the lock file.

#### Create/sync the environment to the lockfile
<todo>Make sure the reader knows how to sync just the main deps or the dev deps.</todo>
```bash
uv sync
```
This will automatically create a `.venv` for you.

#### Run commands inside the managed venv
```bash
uv run pytest -q
uv run python scripts/backtest.py
```
No manual activation needed!


### Alternatives (and why I prefer uv)

The top alternative is [poetry](https://python-poetry.org/) which remains a strong, integrated solution (dependency management + build/publish), with a familiar UX and mature ecosystem.

I prefer uv for a modern quant project because it's significantly faster. For teams running frequent CI and spinning up many ephemeral envs, the speed and lockfile model are tangible wins. 


## Code linting & formatting

## Static typing

## Testing

# Productivity

# Skeleton repo

# Summary

[^1]: Even though PEP 517 has a smaller number than PEP 518, PEP 517 does indeed referece PEP 518 as the source of the `pyproject.toml` file. It says "Here we define a new style of source tree based around the pyproject.toml file defined in PEP 518..." You can check the documents' post history for their latest publication date.