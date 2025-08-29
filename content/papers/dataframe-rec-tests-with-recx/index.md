---
title: "DataFrame Rec Tests with Recx"
summary: "
    Keep research and production outputs aligned with rec tests. This article explains the concept and introduces a tiny Python library for clean, tolerance-aware tests.
"
date: "2025-08-29"
type: paper
mathjax: false
authors:
    - Adrian Letchford
categories:
    - engineering
hover_color: "#5dcabfff"
---

Code changes. Data changes. Outputs change. Somewhere between the first analysis and an odd position in production, little mismatches creep in: a misstated value, off-by-one date ranges, rounding shifts, subtle drift in calculations, missing IDs. The most reliable way to catch them is to compare a new DataFrame to a previously validated one--a reconciliation, or **rec**, test.

`recx` is a lightweight library that makes these recs declarative, repeatable, and pleasant to read. It's early and experimental, but I've been using it in my production trading pipeline for a long while. I'm open-sourcing it so others can use it and give feedback.

# What is a rec test?

A rec test compares a baseline (known-good) DataFrame to a candidate (new run) and checks that they align on keys (same dates or IDs) and agree on values (within tolerances you care about).

For example, you fetch daily prices from a broker and build features for trading. On day 1 you get:

| date       | price |
|------------|-------|
| 2024-01-01 | 100.0 |
| 2024-01-02 | 101.0 |
| 2024-01-03 | 103.0 |
| 2024-01-04 | 102.0 |

On day 2 you get:

| date       | price |
|------------|-------|
| 2024-01-01 | 100.0 |
| 2024-01-02 | 101.0 |
| 2024-01-03 | 103.0 |
| 2024-01-04 | <ins>**101.5**</ins> |
| 2024-01-05 | 100.0 |

The price on 2024-01-04 has changed slightly. A rec test catches it.

Rec failures don't just happen in vendor feeds. Most mismatches show up in your own outputs: a refactor, a feature flag, a changed default. Things end up "almost" equal. A rec test defines what "almost" means for your project and highlights exactly where the baseline and candidate diverge.

# Why do you need rec tests?

That tiny 0.5 drift on 2024-01-04 doesn't stay tiny: it rolls into features, changes model inputs, nudges decisions, and leaks into P&L. Without a rec test, you can't be sure live results follow the same process as your backtests.

Where the drift comes from:

* **Source data moves.** Vendors backfill/correct, adjust rounding, rename IDs, or apply corporate actions differently.
* **Your code moves.** Refactors, feature flags, updating defaults, dtype/precision changes, library upgrades, window alignment, timezone handling, random seeds.
* **Pipelines wobble.** Partial loads, duplicate rows, missing/extra keys, off-by-one joins, calendar mismatches, stale snapshots.

The idea is that every time you run your pipeline, you do a rec test between the new run (candidate) and the previous run (baseline).

Why rec tests help:

* **Early warning:** you see differences the moment they appear--before they hit P&L.
* **Focused diffs:** you learn exactly which rows/fields diverged, not just that something changed.
* **Confidence to change:** you can optimise, refactor, or swap vendors and still keep research and production in lockstep.

# Meet `recx`

`recx` focuses on Pandas DataFrames and keeps the API small:

* **Declarative column mapping.** Assign checks per column.
* **Built-ins:** `EqualCheck`, `AbsTolCheck`, `RelTolCheck`.
* **Index integrity:** automatic missing/extra index checks.
* **Nice summaries:** compact, log-friendly report; rich objects for assertions.
* **Extensible:** subclass `ColumnCheck` to add custom logic.
* **Quality-of-life:** select columns by regex, skip columns, and optionally align on a date column to ignore future rows.

A simple example:
```python
import pandas as pd
from recx import Rec, EqualCheck, AbsTolCheck

# Create a baseline DataFrame
baseline = pd.DataFrame({
    "price": [100.00, 200.00, 300.00],
    "status": ["active", "inactive", "active"]
})

# Candidate has a small price change in the
# last record. Statuses match.
candidate = pd.DataFrame({
    "price": [100.00, 200.00, 301.00],
    "status": ["active", "inactive", "active"]
})

# Declare the rec
rec = Rec({
    "price": AbsTolCheck(tol=0.01),
    "status": EqualCheck(),
})

result = rec.run(baseline, candidate)

# Prints a concise pass/fail report
result.summary()
```

The summary output:
```text
───────────────────────────────────────────────────────────────────────
                    DataFrame Reconciliation Summary                   
───────────────────────────────────────────────────────────────────────
Baseline: rows=3 cols=2
Candidate: rows=3 cols=2

1 check(s) FAILED ❌
missing_indices_check ...................................... PASSED ꪜ
extra_indices_check ........................................ PASSED ꪜ
Column 'price' with AbsTolCheck(tol=0.01) ... [1/3 (33.33%)] FAILED ❌
Column 'status' with EqualCheck ............................ PASSED ꪜ

Failing rows:

Column 'price':
 │   
 │   Showing up to 10 rows
 │   
 │      baseline  candidate  abs_error
 │   2     300.0      301.0        1.0
 │   
```

The `Rec` object maps columns to checks, runs them, and returns a `RecResult` that gives you a readable summary plus programmatic results:
* `passed()` - True if all checks passed.
* `failures()` - List of all failures.
* `raise_for_failures()` - Raises an exception if there are any failures.

Use `recx` when you are rebuilding datasets in your pipeline and want to ensure they stay consistent over time.

# Getting started

Install with:
```bash
pip install recx
```

Then skim the ["Getting Started"](https://recx.readthedocs.io/en/latest/getting-started/) and ["Usage"](https://recx.readthedocs.io/en/latest/usage/) docs for patterns like regex selection, skipping columns, and writing custom checks. The repo README notes the project is early/experimental, so expect API polish over time.

- **GitHub:** https://github.com/robolyst/recx
- **Docs:** https://recx.readthedocs.io
- **PyPI:** https://pypi.org/project/recx/

Contributions and feedback are welcome!
