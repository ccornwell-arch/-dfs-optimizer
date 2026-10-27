"""Showdown exposure enforcement: Min/Max/CPT targets are promises.

Regression suite for the Oct 26, 2026 DAL@PHI failure, where the
delivered sets silently broke the user's settings (Pierce Max 50 ->
70%, McKee Max 50 -> 67%, Schoonmaker Max 10 -> 25%, Barkley Min 50 ->
3/20 without a lock) and two runs returned 9/20 and 13/20 lineups with
no warning anywhere. The contract now:
  - Max Exposure / CPT Max are hard ceilings in exact lineup counts;
  - Min Exposure / CPT Min are reached by forcing + repair, without
    needing a hard Lock;
  - a requested count is returned, or the result's
    attrs["showdown_notes"] names the shortfall and its cause;
  - a setting that cannot be honored is named (player, setting,
    achievable range) before the build runs.

Run from repo root:  python3 tests/test_showdown_exposure_enforcement.py
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "tests")

import pandas as pd

from dfs_lab.showdown import (
    generate_showdown_lineups,
    select_showdown_portfolio,
)
from test_exclusion_trust import _pool


def _check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  [{extra}]" if extra else ""))
    if not cond:
        raise AssertionError(name)


def _gen(df, sm, count=20, seed=42):
    return generate_showdown_lineups(
        df, 5000, "GPP / top-heavy", count, 800, 30000, 50000,
        {"3-3": 1, "4-2": 1, "5-1": 0}, "Neutral", "", sm, "20-Max",
        0, 80, 35, 1, 1, 2, seed)


def _counts(result):
    tot, cpt = {}, {}
    for _, row in result.iterrows():
        cpt[str(row["CPT_ID"])] = cpt.get(str(row["CPT_ID"]), 0) + 1
        tot[str(row["CPT_ID"])] = tot.get(str(row["CPT_ID"]), 0) + 1
        for i in range(1, 6):
            pid = str(row[f"FLEX{i}_ID"])
            tot[pid] = tot.get(pid, 0) + 1
    return tot, cpt


def _notes(result):
    return list(result.attrs.get("showdown_notes") or [])


def test_max_exposure_is_a_hard_cap():
    df = _pool()
    res = _gen(df, {"109": {"Max Exposure": 50}})  # Bucky Irving
    tot, _ = _counts(res)
    _check("max: full count returned", len(res) == 20, f"got {len(res)}")
    _check("max: Irving <= 10 of 20", tot.get("109", 0) <= 10,
           f"got {tot.get('109', 0)}")


def test_cpt_max_is_a_hard_cap():
    df = _pool()
    res = _gen(df, {"103": {"CPT Max": 20}})  # CeeDee Lamb
    _, cpt = _counts(res)
    _check("cpt max: full count returned", len(res) == 20)
    _check("cpt max: Lamb captains <= 4 of 20", cpt.get("103", 0) <= 4,
           f"got {cpt.get('103', 0)}")


def test_min_exposure_without_lock():
    df = _pool()
    # Hunter Luepke (102): $3,200, proj 4.0 — the sim rarely picks him
    # unforced, so 10 of 20 can only come from enforcement.
    res = _gen(df, {"102": {"Min Exposure": 50}})
    tot, _ = _counts(res)
    _check("min: full count returned", len(res) == 20, f"got {len(res)}")
    _check("min: Luepke >= 10 of 20 without a lock",
           tot.get("102", 0) >= 10, f"got {tot.get('102', 0)}")


def test_cpt_min_without_lock():
    df = _pool()
    res = _gen(df, {"101": {"CPT Min": 25}})  # Javonte Williams
    _, cpt = _counts(res)
    _check("cpt min: full count returned", len(res) == 20)
    _check("cpt min: Javonte captains >= 5 of 20",
           cpt.get("101", 0) >= 5, f"got {cpt.get('101', 0)}")


def test_min_on_excluded_player_is_named():
    df = _pool()
    res = _gen(df, {"109": {"Min Exposure": 50, "Exclude": True}})
    tot, _ = _counts(res)
    notes = _notes(res)
    _check("excluded min: Irving never appears", tot.get("109", 0) == 0)
    _check("excluded min: named as unhonorable",
           any("Min Exposure" in n and "Bucky Irving" in n
               and "cannot be honored" in n for n in notes),
           "; ".join(notes))


def test_min_above_max_is_named_and_capped():
    df = _pool()
    res = _gen(df, {"109": {"Min Exposure": 60, "Max Exposure": 50}})
    tot, _ = _counts(res)
    notes = _notes(res)
    _check("min>max: named contradiction",
           any("exceeds Max Exposure" in n and "Bucky Irving" in n
               for n in notes), "; ".join(notes))
    _check("min>max: Irving <= 10 of 20", tot.get("109", 0) <= 10,
           f"got {tot.get('109', 0)}")


def test_cpt_min_on_ineligible_player_is_named():
    df = _pool()
    res = _gen(df, {"109": {"CPT Min": 25, "CPT Eligible": False}})
    _, cpt = _counts(res)
    notes = _notes(res)
    _check("ineligible cpt min: Irving never captains",
           cpt.get("109", 0) == 0)
    _check("ineligible cpt min: named as unhonorable",
           any("CPT Min" in n and "not captain-eligible" in n
               for n in notes), "; ".join(notes))


def test_lock_outranks_cap_and_is_named():
    df = _pool()
    res = _gen(df, {"103": {"Lock": True, "Max Exposure": 50}})
    tot, _ = _counts(res)
    notes = _notes(res)
    _check("lock: Lamb in all 20", tot.get("103", 0) == 20,
           f"got {tot.get('103', 0)}")
    _check("lock: cap conflict named",
           any("locked" in n and "Max Exposure" in n for n in notes),
           "; ".join(notes))


def test_aggregate_caps_shortfall_is_named():
    df = _pool()
    sm = {str(100 + i): {"Max Exposure": 10} for i in range(16)}
    res = _gen(df, sm)
    notes = _notes(res)
    _check("aggregate: build falls short", len(res) < 20,
           f"got {len(res)}")
    _check("aggregate: capacity named up front",
           any("leave room for only" in n for n in notes),
           "; ".join(notes))
    _check("aggregate: shortfall named",
           any(n.startswith("Shortfall:") for n in notes),
           "; ".join(notes))
    tot, _ = _counts(res)
    _check("aggregate: no player above 2",
           all(v <= 2 for v in tot.values()),
           f"max seen {max(tot.values()) if tot else 0}")


def test_selection_layer_enforces_against_delivered_count():
    df = _pool()
    # The API shape: the SAME strategy map goes to the generator (which
    # enforces against its own larger count) and to the selection (which
    # re-enforces against the delivered count).
    sm = {"109": {"Max Exposure": 50}, "102": {"Min Exposure": 50}}
    cands = _gen(df, sm, count=40)
    _check("selection: candidate pool built", len(cands) == 40,
           f"got {len(cands)}")
    sel, notes = select_showdown_portfolio(cands, df, sm, 20, min_unique=2)
    tot, _ = _counts(sel)
    _check("selection: 20 delivered", len(sel) == 20, f"got {len(sel)}")
    _check("selection: Irving <= 10 of delivered 20",
           tot.get("109", 0) <= 10, f"got {tot.get('109', 0)}")
    _check("selection: Luepke >= 10 of delivered 20",
           tot.get("102", 0) >= 10, f"got {tot.get('102', 0)}")


def test_no_settings_no_notes_full_count():
    df = _pool()
    res = _gen(df, {})
    _check("defaults: full count returned", len(res) == 20)
    _check("defaults: no notes when nothing was promised",
           _notes(res) == [], "; ".join(_notes(res)))


if __name__ == "__main__":
    test_max_exposure_is_a_hard_cap()
    test_cpt_max_is_a_hard_cap()
    test_min_exposure_without_lock()
    test_cpt_min_without_lock()
    test_min_on_excluded_player_is_named()
    test_min_above_max_is_named_and_capped()
    test_cpt_min_on_ineligible_player_is_named()
    test_lock_outranks_cap_and_is_named()
    test_aggregate_caps_shortfall_is_named()
    test_selection_layer_enforces_against_delivered_count()
    test_no_settings_no_notes_full_count()
    print("All showdown exposure enforcement tests passed.")
