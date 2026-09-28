"""audit_min_exposure(): missed min-exposure targets are reported with a reason.

Regression test for the silent-miss UX gap: a user sets Min Exposure 30% on a
player, rebuilds, and the player still has 0% — with no message explaining why.
The audit must flag the miss and name the cause (excluded / inactive /
target not in force).
"""
import pandas as pd

from dfs_lab.showdown import audit_min_exposure


def _lineups(names, n=10):
    rows = []
    for i in range(n):
        flex = [names[(i + j) % len(names)] for j in range(5)]
        rows.append({"CPT": "Jalen Hurts", "FLEX1": flex[0], "FLEX2": flex[1],
                     "FLEX3": flex[2], "FLEX4": flex[3], "FLEX5": flex[4]})
    return pd.DataFrame(rows)


def _df():
    return pd.DataFrame([
        {"ID": "p1", "Name": "Jalen Hurts", "ActiveForBuild": True},
        {"ID": "p2", "Name": "Saquon Barkley", "ActiveForBuild": True},
        {"ID": "p3", "Name": "DeVonta Smith", "ActiveForBuild": True},
        {"ID": "p4", "Name": "Zach Ertz", "ActiveForBuild": False},
    ])


def test_miss_on_excluded_player_names_exclusion():
    df = _df()
    # Smith in zero lineups, min 30%, marked Out.
    res = _lineups(["Saquon Barkley"])
    strat = {"p3": {"Min Exposure": 30.0, "Exclude": True}}
    misses = audit_min_exposure(res, df, strat)
    assert len(misses) == 1
    assert misses[0]["name"] == "DeVonta Smith"
    assert misses[0]["target"] == 30.0
    assert misses[0]["actual"] == 0.0
    assert "Out" in misses[0]["reason"] or "excluded" in misses[0]["reason"]


def test_miss_on_inactive_player_names_inactivity():
    df = _df()
    res = _lineups(["Saquon Barkley"])
    strat = {"p4": {"Min Exposure": 50.0}}
    misses = audit_min_exposure(res, df, strat)
    assert len(misses) == 1
    assert misses[0]["name"] == "Zach Ertz"
    assert "inactive" in misses[0]["reason"]


def test_miss_with_no_blocker_says_target_not_in_force():
    df = _df()
    res = _lineups(["Saquon Barkley"])
    strat = {"p3": {"Min Exposure": 30.0}}
    misses = audit_min_exposure(res, df, strat)
    assert len(misses) == 1
    assert "not in force" in misses[0]["reason"]


def test_met_target_is_not_flagged():
    df = _df()
    res = _lineups(["DeVonta Smith"])
    strat = {"p3": {"Min Exposure": 30.0}}
    assert audit_min_exposure(res, df, strat) == []


def test_no_targets_no_misses():
    df = _df()
    res = _lineups(["Saquon Barkley"])
    assert audit_min_exposure(res, df, {"p3": {}}) == []
    assert audit_min_exposure(res, df, {}) == []
