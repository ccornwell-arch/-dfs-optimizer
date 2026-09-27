"""Teamless players must be auto-excluded by the live availability guard.

Regression test for the Brandon Aiyuk case (Sep 26, 2026): DK's slate file
listed him (SF, $3,000) while nflverse's 2026 weekly roster data had zero rows
for him — he is not on a team. The old (name, team) lookup missed and failed
open, so the optimizer rostered a teamless player.

Run from repo root:  python3 tests/test_teamless_exclusion.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import dfs_lab.data as data

CHECKS = []
def _check(name, cond, detail=""):
    CHECKS.append(name)
    if not cond:
        raise AssertionError(f"FAIL {name} {detail}")
    print(f"  ok: {name}")

def _fake_availability(roster_rows):
    roster = pd.DataFrame(roster_rows)
    injuries = pd.DataFrame(columns=["full_name", "team", "week", "report_status"])
    data._load_live_nfl_availability = lambda season=2026: (roster, injuries)

def _pool(rows):
    df = pd.DataFrame(rows)
    df["ActiveForBuild"] = True
    return df

def test_teamless_player_excluded():
    _fake_availability([
        {"full_name": "Patrick Mahomes", "team": "KC", "week": 3, "status": "ACT",
         "headshot_url": "http://x/mahomes.png"},
        {"full_name": "Derrick Henry", "team": "BAL", "week": 3, "status": "ACT",
         "headshot_url": ""},
    ])
    df = _pool([
        {"Name": "Patrick Mahomes", "Position": "QB", "Team": "KC", "Salary": 6200, "My Proj": 21.8},
        {"Name": "Brandon Aiyuk", "Position": "WR", "Team": "SF", "Salary": 3000, "My Proj": 8.93},
        {"Name": "Ravens", "Position": "DST", "Team": "BAL", "Salary": 2800, "My Proj": 5.01},
    ])
    out = data.apply_live_availability_guard(df, "ActiveForBuild", 2026)
    mah = out[out["Name"] == "Patrick Mahomes"].iloc[0]
    _check("rostered star stays active", bool(mah["ActiveForBuild"]))
    _check("rostered star keeps headshot", mah["Headshot URL"] == "http://x/mahomes.png")
    aiy = out[out["Name"] == "Brandon Aiyuk"].iloc[0]
    _check("teamless player excluded", not bool(aiy["ActiveForBuild"]))
    _check("teamless reason names the problem",
           "not on a team" in str(aiy["Auto Excluded Reason"]),
           str(aiy["Auto Excluded Reason"]))
    dst = out[out["Name"] == "Ravens"].iloc[0]
    _check("DST exempt from roster check", bool(dst["ActiveForBuild"]))

def test_blocking_status_still_excluded():
    _fake_availability([
        {"full_name": "Joe Burrow", "team": "CIN", "week": 3, "status": "RES",
         "headshot_url": ""},
    ])
    df = _pool([
        {"Name": "Joe Burrow", "Position": "QB", "Team": "CIN", "Salary": 6800, "My Proj": 20.1},
    ])
    out = data.apply_live_availability_guard(df, "ActiveForBuild", 2026)
    bur = out.iloc[0]
    _check("reserve status still excluded", not bool(bur["ActiveForBuild"]))
    _check("reserve reason preserved",
           "Live roster status" in str(bur["Auto Excluded Reason"]))

def test_name_matched_despite_team_mismatch_stays_active():
    # Traded player: nflverse has the new team, DK file still shows the old one.
    # A valid roster record is what matters — do not punish the team mismatch.
    _fake_availability([
        {"full_name": "Deebo Samuel", "team": "WAS", "week": 3, "status": "ACT",
         "headshot_url": "http://x/deebo.png"},
    ])
    df = _pool([
        {"Name": "Deebo Samuel", "Position": "WR", "Team": "SF", "Salary": 6100, "My Proj": 14.2},
    ])
    out = data.apply_live_availability_guard(df, "ActiveForBuild", 2026)
    dee = out.iloc[0]
    _check("team-mismatch but rostered stays active", bool(dee["ActiveForBuild"]))
    _check("team-mismatch still gets headshot", dee["Headshot URL"] == "http://x/deebo.png")

def test_fail_open_when_live_data_unavailable():
    def _boom(season=2026):
        raise RuntimeError("nflverse unreachable")
    data._load_live_nfl_availability = _boom
    df = _pool([
        {"Name": "Brandon Aiyuk", "Position": "WR", "Team": "SF", "Salary": 3000, "My Proj": 8.93},
    ])
    out = data.apply_live_availability_guard(df, "ActiveForBuild", 2026)
    _check("unreachable data fails open", bool(out.iloc[0]["ActiveForBuild"]))
    _check("warning recorded", "availability_warning" in out.attrs)

if __name__ == "__main__":
    print("teamless-exclusion tests:")
    test_teamless_player_excluded()
    test_blocking_status_still_excluded()
    test_name_matched_despite_team_mismatch_stays_active()
    test_fail_open_when_live_data_unavailable()
    print(f"\nALL TEAMLESS-EXCLUSION TESTS PASSED ({len(CHECKS)} checks)")
