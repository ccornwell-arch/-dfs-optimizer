"""Build smoke test: the full showdown pipeline must produce valid lineups.

Lesson from Oct 7, 2026: commit d5f5249 broke prepare_showdown_pool (returned
None) and three follow-up regressions shipped before anyone ran a build.
Unit tests on helpers are not enough — this test exercises the actual
generate path end-to-end so a broken build can never reach production.

Run from repo root:  python3 tests/test_build_smoke.py
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "tests")

import pandas as pd

from dfs_lab.showdown import generate_showdown_lineups
from test_exclusion_trust import _pool


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def test_smoke_build_produces_valid_lineups():
    """The exact call pattern from the UI must return a non-empty, valid portfolio."""
    df = _pool()
    result = generate_showdown_lineups(
        df, 50000, "GPP / top-heavy", 20, 500, 40000, 50000,
        {"3-3": 1, "4-2": 1, "5-1": 0}, "Neutral", "", {}, "20-Max",
        0, 80, 35, 1, 1, 1, 42)
    _check("smoke: result is not None", result is not None)
    _check("smoke: result is not empty", not result.empty)
    _check("smoke: built requested count", len(result) == 20)


def test_smoke_lineups_under_cap():
    df = _pool()
    result = generate_showdown_lineups(
        df, 50000, "GPP / top-heavy", 10, 500, 40000, 50000,
        {"3-3": 1, "4-2": 1, "5-1": 0}, "Neutral", "", {}, "20-Max",
        0, 80, 35, 1, 1, 1, 42)
    _check("smoke: all lineups under $50k cap",
           bool((result["Salary"] <= 50000).all()))
    _check("smoke: all lineups above min salary",
           bool((result["Salary"] >= 40000).all()))


def test_smoke_captain_present():
    df = _pool()
    result = generate_showdown_lineups(
        df, 50000, "GPP / top-heavy", 10, 500, 40000, 50000,
        {"3-3": 1, "4-2": 1, "5-1": 0}, "Neutral", "", {}, "20-Max",
        0, 80, 35, 1, 1, 1, 42)
    _check("smoke: every lineup has a captain",
           bool(result["CPT"].notna().all() & (result["CPT"] != "").all()))
    _check("smoke: every lineup has 5 flex",
           all(bool(pd.notna(result[f"FLEX{i}"]).all()) for i in range(1, 6)))


def test_smoke_with_exclusions_and_locks():
    """Realistic settings: exclusions + a lock + captain pool restriction."""
    df = _pool()
    sm = {
        "108": {"Exclude": True},  # Jalon Daniels out
        "103": {"Lock": True},     # CeeDee Lamb locked
        "100": {"CPT Eligible": True},
        "103": {"CPT Eligible": True, "Lock": True},
    }
    result = generate_showdown_lineups(
        df, 50000, "GPP / top-heavy", 10, 500, 40000, 50000,
        {"3-3": 1, "4-2": 1, "5-1": 0}, "Neutral", "", sm, "20-Max",
        0, 80, 35, 1, 1, 1, 42)
    _check("smoke+rules: result is not None", result is not None)
    _check("smoke+rules: result is not empty", not result.empty)
    ids = set()
    for _, row in result.iterrows():
        ids.add(str(row["CPT_ID"]))
        for i in range(1, 6):
            ids.add(str(row.get(f"FLEX{i}_ID", "")))
    _check("smoke+rules: excluded player absent", "108" not in ids)
    # Locked player should appear in every lineup
    _check("smoke+rules: locked player in every lineup",
           bool(all("103" in {str(row["CPT_ID"])} | {str(row.get(f"FLEX{i}_ID", "")) for i in range(1, 6)}
                   for _, row in result.iterrows())))


if __name__ == "__main__":
    test_smoke_build_produces_valid_lineups()
    test_smoke_lineups_under_cap()
    test_smoke_captain_present()
    test_smoke_with_exclusions_and_locks()
    print("All build smoke tests passed.")
