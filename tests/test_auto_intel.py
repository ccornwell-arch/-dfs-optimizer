"""Tests for Game Intel auto-fill (dfs_lab/auto_intel.py).

The feature: Aytia researches defensive matchups, usage trends, rest
edges and travel from nflverse and pre-fills Game Intel; the user only
overrides. All builders are pure -- synthetic dataframes, no network.

Run from repo root:  python3 tests/test_auto_intel.py
"""
import sys

sys.path.insert(0, ".")

import pandas as pd

from dfs_lab import auto_intel as ai
from dfs_lab.ui.main import _auto_intel_adopt_manual


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def _stats(rows):
    """rows: dicts with week/player_name/position/team/opponent_team/targets/carries/receiving_yards."""
    df = pd.DataFrame(rows)
    df["season"] = 2026
    for c in ("targets", "carries", "receiving_yards", "receptions"):
        if c not in df.columns:
            df[c] = 0
    return df


def _wr_row(week, name, team, opp, yds):
    return {"week": week, "player_name": name, "position": "WR",
            "team": team, "opponent_team": opp, "targets": 8, "carries": 0,
            "receptions": 6, "receiving_yards": yds}


def test_defense_dvp_values():
    # Four defenses allow 30 / 20 / 12 / 8 DK pts per game to WRs over weeks 1-3.
    rows = []
    for wk in (1, 2, 3):
        rows.append(_wr_row(wk, "WRA", "XXX", "DA", 300))
        rows.append(_wr_row(wk, "WRB", "XXX", "DB", 200))
        rows.append(_wr_row(wk, "WRC", "XXX", "DC", 120))
        rows.append(_wr_row(wk, "WRD", "XXX", "DD", 80))
    dvp = ai.defense_dvp(_stats(rows), 2026, week=4)
    _check("softest defense allows ~39/g", abs(dvp[("DA", "WR")] - 39.0) < 0.01)
    _check("toughest defense allows ~14/g", abs(dvp[("DD", "WR")] - 14.0) < 0.01)


def test_defense_rating_percentiles():
    league = [30.0, 20.0, 12.0, 8.0]
    _check("softest -> +3", ai.defense_rating(30.0, league)[0] == 3)
    _check("second -> +1", ai.defense_rating(20.0, league)[0] == 1)
    _check("third -> -1", ai.defense_rating(12.0, league)[0] == -1)
    _check("toughest -> -3", ai.defense_rating(8.0, league)[0] == -3)
    _check("no data -> 0", ai.defense_rating(None, league)[0] == 0)


def test_usage_trend_up():
    rows = []
    for wk, tg in [(1, 5), (2, 5), (3, 8), (4, 9), (5, 8)]:
        rows.append({"week": wk, "player_name": "Riser", "position": "WR",
                     "team": "PHI", "opponent_team": "CHI",
                     "targets": tg, "carries": 0})
    r, note = ai.usage_trend(_stats(rows), 2026, week=6, namekey="riser")
    _check("rising usage -> +2", r == 2)
    _check("note explains the trend", "up 67%" in note)


def test_usage_trend_needs_baseline():
    rows = [{"week": 5, "player_name": "Newb", "position": "WR", "team": "PHI",
             "opponent_team": "CHI", "targets": 9, "carries": 0}]
    r, _ = ai.usage_trend(_stats(rows), 2026, week=6, namekey="newb")
    _check("no baseline -> 0, not a guess", r == 0)


def test_injury_usage_bump_next_man_up():
    df = pd.DataFrame({
        "ID": ["g1", "e1", "c1", "s1"],
        "Name": ["Dallas Goedert", "Zach Ertz", "Grant Calcaterra", "Saquon Barkley"],
        "Position": ["TE", "TE", "TE", "RB"], "Team": ["PHI"] * 4,
        "ActiveForBuild": [False, True, True, True],
        "My Proj": [12.0, 7.5, 3.0, 16.9],
    })
    bumps = ai.injury_usage_bumps(df)
    _check("top replacement gets +2", bumps["e1"][0] == 2)
    _check("note names the absence", "Goedert" in bumps["e1"][1] and "OUT" in bumps["e1"][1])
    _check("second replacement gets +1", bumps["c1"][0] == 1)
    _check("other positions untouched", "s1" not in bumps)


def test_injury_bump_when_star_is_zeroed():
    # The real-world case: Goedert is OUT and his projection is zeroed to 0.0,
    # but he averaged big points before the injury -- still a significant absence.
    df = pd.DataFrame({
        "ID": ["g1", "e1"],
        "Name": ["Dallas Goedert", "Zach Ertz"],
        "Position": ["TE", "TE"], "Team": ["PHI", "PHI"],
        "ActiveForBuild": [False, True],
        "My Proj": [0.0, 7.5],
    })
    stats = _stats([{"week": wk, "player_name": "Dallas Goedert", "position": "TE",
                     "team": "PHI", "opponent_team": "CHI",
                     "targets": 6, "carries": 0, "receptions": 5, "receiving_yards": 60}
                    for wk in (1, 2)])
    star_avg = ai._season_avg_points(stats, 2026, week=4)
    _check("Goedert's season average is significant", star_avg["dallas goedert"] >= 6)
    bumps = ai.injury_usage_bumps(df, star_avg)
    _check("zeroed star still triggers the bump", bumps["e1"][0] == 2)
    _check("practice-squad-level absence does not",
           ai.injury_usage_bumps(df, {}) == {})


def _sched(rows):
    df = pd.DataFrame(rows)
    df["season"] = 2026
    return df


def test_rest_travel_home_and_no_penalty():
    sched = _sched([
        {"week": 3, "home_team": "CHI", "away_team": "PHI", "gameday": "2026-09-28"},
        {"week": 2, "home_team": "CHI", "away_team": "GB", "gameday": "2026-09-21"},
        {"week": 2, "home_team": "PHI", "away_team": "DAL", "gameday": "2026-09-20"},
    ])
    game = sched.iloc[0]
    rt = ai.rest_travel_for_game(sched, 2026, game)
    _check("home team gets Home/Rest +1", rt["CHI"]["Home/Rest"][0] == 1)
    _check("1-day rest diff is no edge", rt["PHI"]["Home/Rest"][0] == 0)
    _check("1-tz trip is no travel penalty", rt["PHI"]["Travel"][0] == 0)


def test_rest_edge_and_long_travel():
    sched = _sched([
        {"week": 3, "home_team": "MIA", "away_team": "SEA", "gameday": "2026-09-27"},
        {"week": 2, "home_team": "MIA", "away_team": "BUF", "gameday": "2026-09-20"},
        {"week": 2, "home_team": "SEA", "away_team": "SF", "gameday": "2026-09-13"},
    ])
    rt = ai.rest_travel_for_game(sched, 2026, sched.iloc[0])
    _check("7-day rest edge -> +1", rt["SEA"]["Home/Rest"][0] == 1)
    _check("tired home team loses its edge", rt["MIA"]["Home/Rest"][0] == 0)
    _check("3-tz trip -> Travel -1", rt["SEA"]["Travel"][0] == -1)
    _check("travel note names the trip", "MIA" in rt["SEA"]["Travel"][1])


def _slate_df():
    return pd.DataFrame({
        "ID": ["h1", "b1", "g1", "e1", "k1"],
        "Name": ["Jalen Hurts", "Saquon Barkley", "Dallas Goedert", "Zach Ertz", "Case Keenum"],
        "Position": ["QB", "RB", "TE", "TE", "QB"],
        "Team": ["PHI", "PHI", "PHI", "PHI", "CHI"],
        "ActiveForBuild": [True, True, False, True, True],
        "My Proj": [20.1, 16.9, 0.0, 7.5, 12.0],
    })


def _slate_sched():
    return _sched([
        {"week": 3, "home_team": "CHI", "away_team": "PHI", "gameday": "2026-09-28"},
        {"week": 2, "home_team": "CHI", "away_team": "GB", "gameday": "2026-09-21"},
        {"week": 2, "home_team": "PHI", "away_team": "DAL", "gameday": "2026-09-20"},
    ])


def test_suggest_game_intel_end_to_end():
    rows = []
    for wk in (1, 2, 3):
        rows.append(_wr_row(wk, "Some WR", "PHI", "CHI", 300))  # CHI soft vs WRs
        rows.append({"week": wk, "player_name": "Dallas Goedert", "position": "TE",
                     "team": "PHI", "opponent_team": "CHI",
                     "targets": 6, "carries": 0, "receptions": 5, "receiving_yards": 60})
    sched = _slate_sched()
    sugg, msg = ai.suggest_game_intel(_slate_df(), _stats(rows), sched, 2026)
    _check("Ertz gets the injury usage bump", sugg["e1"]["Usage"] == 2)
    _check("auto note names the absence", sugg["e1"]["Note"].startswith("Auto:") and "Goedert" in sugg["e1"]["Note"])
    _check("auto confidence is heuristic-grade", sugg["e1"]["Confidence"] == 60)
    _check("home team gets Home/Rest +1", sugg["k1"]["Home/Rest"] == 1)
    _check("away QB with no edges gets no suggestion", "h1" not in sugg)


def test_suggest_unknown_slate_returns_honest_empty():
    df = _slate_df()
    df["Team"] = ["XX", "XX", "XX", "XX", "YY"]
    sugg, msg = ai.suggest_game_intel(df, _stats([]), _slate_sched(), 2026)
    _check("unmatched slate -> no suggestions", sugg == {})
    _check("message says why", "schedule" in msg)


def test_manual_adopt_marks_user_cells():
    context = {"p1": {"Defense": 2, "Usage": 0, "Home/Rest": 0, "Travel": 0,
                       "Time/Split": 0, "Confidence": 60, "Note": "Auto: x"}}
    snapshot = {"p1": {"Defense": 1, "Usage": 0, "Home/Rest": 0, "Travel": 0,
                       "Time/Split": 0, "Confidence": 60, "Note": "Auto: x"}}
    manual = set()
    _auto_intel_adopt_manual(context, snapshot, manual)
    _check("changed cell marked manual", "p1|Defense" in manual)
    _check("untouched cell not manual", "p1|Usage" not in manual)
    # User reverts to the auto value -> no longer manual.
    context["p1"]["Defense"] = 1
    _auto_intel_adopt_manual(context, snapshot, manual)
    _check("reverted cell unmarked", "p1|Defense" not in manual)


def test_manual_adopt_before_first_autofill():
    # User typed before ever running auto-fill: non-default values are theirs.
    context = {"p9": {"Defense": 0, "Usage": 3, "Home/Rest": 0, "Travel": 0,
                      "Time/Split": 0, "Confidence": 80, "Note": "my take"}}
    manual = set()
    _auto_intel_adopt_manual(context, {}, manual)
    _check("pre-existing Usage kept", "p9|Usage" in manual)
    _check("pre-existing Confidence kept", "p9|Confidence" in manual)
    _check("pre-existing Note kept", "p9|Note" in manual)
    _check("neutral Defense stays auto-fillable", "p9|Defense" not in manual)


if __name__ == "__main__":
    test_defense_dvp_values()
    test_defense_rating_percentiles()
    test_usage_trend_up()
    test_usage_trend_needs_baseline()
    test_injury_usage_bump_next_man_up()
    test_injury_bump_when_star_is_zeroed()
    test_rest_travel_home_and_no_penalty()
    test_rest_edge_and_long_travel()
    test_suggest_game_intel_end_to_end()
    test_suggest_unknown_slate_returns_honest_empty()
    test_manual_adopt_marks_user_cells()
    test_manual_adopt_before_first_autofill()
    print("auto intel tests done")
