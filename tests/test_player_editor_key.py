"""Regression test for the "I marked him Out but he's in my lineups" bug.

Root cause: the Players-tab data_editor sits inside a form, so checkbox edits
stay pending (row-positional: {row_index: changes}) until APPLY is pressed. The
editor used a FIXED widget key, so if the user changed the sort/filter between
checking a box and pressing apply, the pending edit silently landed on the
WRONG player — the intended player was never excluded and showed up in builds.

Fix: the widget key is scoped to the exact visible row order
(player_editor_widget_key). Any resort/refilter changes the key, which resets
pending positional edits instead of misapplying them.

Run from repo root:  python3 tests/test_player_editor_key.py
"""
import io
import sys

sys.path.insert(0, ".")

import pandas as pd

import dfs_lab.data as _data_mod
from dfs_lab.common import player_editor_widget_key
from dfs_lab.data import prepare_player_pool

# Synthetic players have no nflverse record: stub live availability so the
# guard fails open instead of excluding everyone as teamless.
_data_mod._load_live_nfl_availability = lambda season=2026: (pd.DataFrame(), pd.DataFrame())
from dfs_lab.classic import generate_lineups

FP = "Classic|500|GPP / top-heavy|DKSalaries.csv"
IDS_SALARY_ORDER = ["9", "3", "7", "1", "5"]      # e.g. sorted by salary
IDS_PROJ_ORDER = ["1", "5", "9", "3", "7"]        # same pool, sorted by projection


def test_key_stable_for_same_view():
    a = player_editor_widget_key("v4_classic_players", FP, IDS_SALARY_ORDER)
    b = player_editor_widget_key("v4_classic_players", FP, IDS_SALARY_ORDER)
    assert a == b, "same view must produce a stable key (no spurious widget reset)"
    print("PASS key stable for identical view")


def test_key_changes_on_resort():
    a = player_editor_widget_key("v4_classic_players", FP, IDS_SALARY_ORDER)
    b = player_editor_widget_key("v4_classic_players", FP, IDS_PROJ_ORDER)
    assert a != b, "resorted view must change the key so stale positional edits reset"
    print("PASS key changes when the row order changes (resort)")


def test_key_changes_on_filter():
    a = player_editor_widget_key("v4_classic_players", FP, IDS_SALARY_ORDER)
    b = player_editor_widget_key("v4_classic_players", FP, IDS_SALARY_ORDER[:3])
    assert a != b, "filtered view must change the key"
    print("PASS key changes when the view is filtered")


def test_key_changes_on_new_slate():
    a = player_editor_widget_key("v4_classic_players", FP, IDS_SALARY_ORDER)
    b = player_editor_widget_key("v4_classic_players", "Classic|500|GPP / top-heavy|OTHER.csv", IDS_SALARY_ORDER)
    assert a != b, "new slate upload must change the key"
    print("PASS key changes on a new slate")


def test_positional_edit_would_hit_wrong_player_after_resort():
    # Documents the original bug mechanics: the user checks "Out" on the
    # player at row position 2 (ID "7") while sorted by salary...
    pending = {2: {"Exclude": True}}
    before = IDS_SALARY_ORDER[pending_row := 2]
    assert before == "7"
    # ...then sorts by projection before pressing apply. The same positional
    # edit now points at ID "9" — the wrong player.
    after = IDS_PROJ_ORDER[pending_row]
    assert after == "9" and after != before, "fixture must demonstrate the misalignment"
    # The fix: keys differ, so Streamlit resets the widget instead of applying
    # the stale positional edit to player "9".
    assert player_editor_widget_key("v4_classic_players", FP, IDS_SALARY_ORDER) != \
           player_editor_widget_key("v4_classic_players", FP, IDS_PROJ_ORDER)
    print("PASS stale positional edit cannot survive a resort (would have hit ID 9 instead of 7)")


# --- Build-level: an ID-keyed exclusion (what APPLY records) always holds ---
# Mirrors the user report: a $3,000 punt WR ("Hurst") marked Out must never appear.

PLAYERS = [
    # (ID, Name, Pos, Team, Salary, Proj, Own)
    (1001, "T1 Quarterback", "QB", "T1", 7000, 20.0, 10.0),
    (1002, "T1 Runner", "RB", "T1", 6500, 16.0, 12.0),
    (1003, "T1 Change", "RB", "T1", 4500, 9.0, 5.0),
    (1004, "T1 Wide1", "WR", "T1", 6000, 14.0, 10.0),
    (1005, "T1 Wide2", "WR", "T1", 5200, 11.0, 7.0),
    (1006, "T1 Punt", "WR", "T1", 3000, 6.66, 1.4),   # the "Hurst"
    (1007, "T1 Tight", "TE", "T1", 4500, 10.0, 8.0),
    (1008, "T1 Defense", "DST", "T1", 3000, 7.0, 6.0),
    (2001, "T2 Quarterback", "QB", "T2", 6800, 18.0, 9.0),
    (2002, "T2 Runner", "RB", "T2", 6200, 15.0, 11.0),
    (2003, "T2 Change", "RB", "T2", 4500, 9.0, 5.0),
    (2004, "T2 Wide1", "WR", "T2", 5900, 13.0, 9.0),
    (2005, "T2 Wide2", "WR", "T2", 5100, 11.0, 6.0),
    (2006, "T2 Punt", "WR", "T2", 3000, 6.0, 1.2),
    (2007, "T2 Tight", "TE", "T2", 4300, 9.0, 7.0),
    (2008, "T2 Defense", "DST", "T2", 2800, 6.0, 5.0),
]
GAME = "T1@T2 09/28/2026 01:00PM ET"
SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]


def _dk_file():
    lines = ["Position,Name + ID,Salary,Name,ID,Roster Position,Game Info,TeamAbbrev,AvgPointsPerGame"]
    for pid, name, pos, team, sal, proj, own in PLAYERS:
        rp = {"RB": "RB/FLEX", "WR": "WR/FLEX", "TE": "TE/FLEX"}.get(pos, pos)
        lines.append(f"{pos},{name} ({pid}),{sal},{name},{pid},{rp},{GAME},{team},{proj}")
    return io.BytesIO("\n".join(lines).encode("utf-8"))


def _ss_file():
    lines = ["Name,My Proj,My Own"]
    for pid, name, pos, team, sal, proj, own in PLAYERS:
        lines.append(f"{name},{proj},{own}")
    return io.BytesIO("\n".join(lines).encode("utf-8"))


def test_excluded_punt_wr_never_appears():
    # What the APPLY handler records when the user checks "Out" on T1 Punt.
    sm = {"1006": {"Lock": False, "Exclude": True, "Priority": "Exclude",
                   "Min Exposure": 0.0, "Max Exposure": 100.0}}
    df = prepare_player_pool(_dk_file(), _ss_file())
    res = generate_lineups(df, 500, "GPP / top-heavy", 20, 400, 40000, 1,
                           "Optional", [], sm, {}, False, False, 7)
    assert not res.empty, "build produced no lineups"
    n = sum(1 for _, r in res.iterrows() for s in SLOTS if r[s] == "T1 Punt")
    assert n == 0, f"excluded punt WR appeared {n}x in {len(res)} lineups"
    # ...while the other punt WR (not excluded) is still eligible.
    assert (df["ID"].astype(str) == "2006").any()
    print(f"PASS excluded punt WR in 0 of {len(res)} lineups")


if __name__ == "__main__":
    test_key_stable_for_same_view()
    test_key_changes_on_resort()
    test_key_changes_on_filter()
    test_key_changes_on_new_slate()
    test_positional_edit_would_hit_wrong_player_after_resort()
    test_excluded_punt_wr_never_appears()
    print("\nALL PLAYER-EDITOR-KEY TESTS PASSED")
