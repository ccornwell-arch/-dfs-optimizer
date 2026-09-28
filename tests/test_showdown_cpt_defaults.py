"""Regression tests for the Showdown CPT? default-unchecked change.

The user wants the CPT? (captain eligibility) column unchecked by default so
they opt players INTO the captain pool instead of opting out. Covers:

  1. The Players editor defaults CPT Eligible to False (both the fresh-table
     init and the saved-strategy fallback).
  2. captain_pool_ids() mirrors the optimizer: active + not excluded +
     CPT-eligible; a missing strategy entry still defaults to eligible
     (first-run: everyone is in the pool until the user applies choices).
  3. The optimizer and the post-build audit keep their True default for
     missing entries, so a user who never opens Players still gets a build.

Run from repo root:  python3 tests/test_showdown_cpt_defaults.py
"""
import re
import sys

sys.path.insert(0, ".")

import pandas as pd

from dfs_lab.showdown import captain_pool_ids


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def _pool_df():
    return pd.DataFrame({
        "ID": ["1", "2", "3", "4"],
        "Name": ["A", "B", "C", "D"],
        "ActiveForBuild": [True, True, True, False],
    })


def test_editor_defaults_unchecked():
    src = open("dfs_lab/ui/main.py").read()
    _check("editor table inits CPT Eligible=False",
           '"CPT Eligible":False' in src)
    _check("editor saved-strategy fallback is False",
           '("CPT Eligible","CPT Eligible",False)' in src)
    _check("no lingering True default in editor",
           '"CPT Eligible":True,"Priority"' not in src)


def test_pool_all_default_strategy():
    pool = captain_pool_ids(_pool_df(), {})
    _check("missing entries default eligible (active only)",
           pool == ["1", "2", "3"])


def test_pool_explicit_unchecked_excluded():
    strat = {"1": {"CPT Eligible": False}, "2": {"CPT Eligible": True}}
    pool = captain_pool_ids(_pool_df(), strat)
    _check("explicit False removes player from pool", pool == ["2", "3"])


def test_pool_respects_exclude_and_inactive():
    strat = {"2": {"Exclude": True}, "3": {"Priority": "Exclude"}}
    pool = captain_pool_ids(_pool_df(), strat)
    _check("excluded players not in pool", pool == ["1"])


def test_pool_empty_when_nothing_checked():
    strat = {pid: {"CPT Eligible": False} for pid in ["1", "2", "3", "4"]}
    _check("empty pool when nobody checked", captain_pool_ids(_pool_df(), strat) == [])


def test_optimizer_and_audit_keep_true_default():
    src = open("dfs_lab/showdown.py").read()
    _check("optimizer defaults missing entry to eligible",
           'bool(strat.get("CPT Eligible", True))' in src)
    _check("audit defaults missing entry to eligible",
           'bool(cap_strat.get("CPT Eligible", True))' in src)


if __name__ == "__main__":
    test_editor_defaults_unchecked()
    test_pool_all_default_strategy()
    test_pool_explicit_unchecked_excluded()
    test_pool_respects_exclude_and_inactive()
    test_pool_empty_when_nothing_checked()
    test_optimizer_and_audit_keep_true_default()
    print("OK")


def test_cpt_lock_implies_eligibility_in_pool():
    """Regression: checking the CPT (lock) column alone must put the player in
    the captain pool. The user hit 'No captain-eligible players' after checking
    CPT locks but not CPT? -- the lock is the stronger intent ('WILL captain')
    and must imply eligibility, as it did before the CPT? opt-in change."""
    strat = {"1": {"CPT Eligible": False, "CPT Lock": True},
             "2": {"CPT Eligible": False}}
    pool = captain_pool_ids(_pool_df(), strat)
    _check("CPT-locked player is in pool despite CPT? unchecked", pool == ["1", "3"])
    _check("CPT lock does not override an exclusion",
           captain_pool_ids(_pool_df(), {"1": {"CPT Eligible": False, "CPT Lock": True, "Exclude": True}}) == ["2", "3"])


def test_optimizer_source_treats_lock_as_eligible():
    """The MILP's own cpt_ok must agree with captain_pool_ids: a CPT-locked
    player may not have the CPT slot upper-bounded to 0 (that + the lock's
    hard 'must captain' constraint = infeasible model)."""
    src = open("dfs_lab/showdown.py").read()
    _check("optimizer cpt_ok includes CPT Lock",
           'bool(strat.get("CPT Lock", False))) and not excluded' in src)
    msrc = open("dfs_lab/ui/main.py").read()
    _check("apply handler normalizes lock -> eligible",
           "cpt_elig=(bool(r[\"CPT Eligible\"]) or cptlock) and not ex" in msrc)


if __name__ == "__main__":
    test_cpt_lock_implies_eligibility_in_pool()
    test_optimizer_source_treats_lock_as_eligible()
    print("lock-implies-eligibility regression tests done")
