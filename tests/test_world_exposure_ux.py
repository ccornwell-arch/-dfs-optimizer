"""UX tests for the two user complaints of Sep 28, 2026 (PHI@CHI slate):

1. Game Worlds: picking a world showed only a summary table (Captain / Rank /
   Construction / Proj / $ Left / Dup Risk) -- "not full lineups". The world
   view now renders full-roster lineup cards via _sd_lineup_card_html().
   The card must contain the captain AND all five FLEX players.

2. Exposure Lab: the "Adjust Exposure" tile landed on a read-only report, so
   the user "couldn't adjust exposure" without knowing to go to Players.
   The tab now has an editable targets table; edits are applied to the
   strategy map by _apply_exposure_targets(). Edits map by row position, so
   this also guards the min>max skip path and unchanged-value preservation.

Run from repo root:  python3 tests/test_world_exposure_ux.py
"""
import sys

sys.path.insert(0, ".")

import pandas as pd

from dfs_lab.ui.main import _sd_lineup_card_html, _apply_exposure_targets
import dfs_lab.showdown as sd


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def _lineup_row():
    return pd.Series({
        "Rank": 7, "Rating": "A", "Captain": "Jalen Hurts",
        "FLEX1": "Saquon Barkley", "FLEX2": "D'Andre Swift",
        "FLEX3": "Kyle Monangai", "FLEX4": "Cairo Santos", "FLEX5": "Zach Ertz",
        "Construction": "3-3", "Projection": 87.2, "Salary": 49800,
        "Salary Left": 200, "Dup Risk": "Low",
        "Correlation Grade": "B", "Leverage Grade": "A",
        "Game World": "PHI passing ceiling",
    })


def test_world_card_shows_full_roster():
    html = _sd_lineup_card_html(_lineup_row(), {})
    import html as _h
    for name in ["Jalen Hurts", "Saquon Barkley", "D'Andre Swift",
                 "Kyle Monangai", "Cairo Santos", "Zach Ertz"]:
        _check(f"world card contains full roster player: {name}", _h.escape(name) in html)
    _check("world card shows rank", "#7" in html)
    _check("world card shows projection", "87.2" in html)


def _exp_inputs():
    df = pd.DataFrame({
        "ID": ["p1", "p2", "p3"], "Name": ["Jalen Hurts", "Saquon Barkley", "Zach Ertz"],
        "Position": ["QB", "RB", "TE"], "Team": ["PHI", "PHI", "PHI"],
        "FlexSalary": [10800, 10000, 4200], "My Proj": [20.1, 16.9, 7.5],
        "My Own": [30.0, 25.0, 9.8], "CPT Own": [20.0, 10.0, 2.0],
    })
    result = pd.DataFrame([{
        "Rank": 1, "CPT": "Jalen Hurts", "FLEX1": "Saquon Barkley",
        "FLEX2": "Zach Ertz", "FLEX3": "Saquon Barkley", "FLEX4": "Zach Ertz",
        "FLEX5": "Saquon Barkley",
    }])
    strategy = {}
    exp = sd.showdown_exposure_table(df, result, strategy)
    ed = exp.rename(columns={"Name": "Player", "Actual %": "My Exp %", "Min %": "Target Min %",
                             "Max %": "Target Max %", "CPT Actual %": "CPT Exp %",
                             "CPT Min %": "CPT Target Min %", "CPT Max %": "CPT Target Max %"})
    ed_cols = ["Player", "Pos", "Team", "My Exp %", "Target Min %", "Target Max %",
               "CPT Exp %", "CPT Target Min %", "CPT Target Max %"]
    ed = ed[[c for c in ed_cols if c in ed.columns]]
    return df, exp, ed, strategy


def _row_of(ed, name):
    return int(ed.index[ed["Player"].eq(name)].tolist()[0])


def test_exposure_targets_apply_to_right_player():
    df, exp, ed, strategy = _exp_inputs()
    ri = _row_of(ed, "Jalen Hurts")
    n, bad = _apply_exposure_targets(
        ed, exp, {ri: {"Target Min %": 40.0, "Target Max %": 90.0}}, strategy)
    _check("one row applied", n == 1 and not bad)
    _check("targets land on Hurts' strategy entry",
           strategy.get("p1", {}).get("Min Exposure") == 40.0
           and strategy.get("p1", {}).get("Max Exposure") == 90.0)
    _check("untouched player has no strategy entry", "p2" not in strategy)


def test_exposure_min_above_max_is_skipped_and_reported():
    df, exp, ed, strategy = _exp_inputs()
    ri = _row_of(ed, "Jalen Hurts")
    n, bad = _apply_exposure_targets(
        ed, exp, {ri: {"Target Min %": 80.0, "Target Max %": 20.0}}, strategy)
    _check("invalid row skipped", n == 0)
    _check("invalid row reported by player name", bad == ["Jalen Hurts"])
    _check("nothing written for invalid row", "p1" not in strategy)


def test_exposure_unchanged_values_preserve_existing_targets():
    df, _, _, _ = _exp_inputs()
    strategy = {"p1": {"Min Exposure": 10.0, "Max Exposure": 100.0,
                       "CPT Min": 5.0, "CPT Max": 60.0, "Lock": True}}
    result = pd.DataFrame([{
        "Rank": 1, "CPT": "Jalen Hurts", "FLEX1": "Saquon Barkley",
        "FLEX2": "Zach Ertz", "FLEX3": "Saquon Barkley", "FLEX4": "Zach Ertz",
        "FLEX5": "Saquon Barkley",
    }])
    exp = sd.showdown_exposure_table(df, result, strategy)
    ed = exp.rename(columns={"Name": "Player", "Actual %": "My Exp %", "Min %": "Target Min %",
                             "Max %": "Target Max %", "CPT Actual %": "CPT Exp %",
                             "CPT Min %": "CPT Target Min %", "CPT Max %": "CPT Target Max %"})
    ed_cols = ["Player", "Pos", "Team", "My Exp %", "Target Min %", "Target Max %",
               "CPT Exp %", "CPT Target Min %", "CPT Target Max %"]
    ed = ed[[c for c in ed_cols if c in ed.columns]]
    # Edit only the CPT max; everything else must carry through unchanged.
    ri = _row_of(ed, "Jalen Hurts")
    n, bad = _apply_exposure_targets(
        ed, exp, {ri: {"CPT Target Max %": 80.0}}, strategy)
    _check("applied cleanly", n == 1 and not bad)
    s = strategy["p1"]
    _check("changed value updated", s["CPT Max"] == 80.0)
    _check("unchanged targets preserved",
           s["Min Exposure"] == 10.0 and s["Max Exposure"] == 100.0 and s["CPT Min"] == 5.0)
    _check("unrelated strategy keys untouched", s["Lock"] is True)


if __name__ == "__main__":
    test_world_card_shows_full_roster()
    test_exposure_targets_apply_to_right_player()
    test_exposure_min_above_max_is_skipped_and_reported()
    test_exposure_unchanged_values_preserve_existing_targets()
    print("world/exposure UX tests done")
