"""Repro for DFS LAB exclusion-guard bugs.

Covers the real user reports:
  1. Backup QBs must be auto-excluded (ActiveForBuild == False).
  2. An injured (0-proj) "starter" must NOT keep the primary-QB slot over a healthy backup.
  3. 0-projection players must be deactivated AND carry an exclusion reason.
  4. A player manually excluded via the Exclude checkbox must never appear in built lineups.
  5. classic_lineup_coherence must hard-reject a backup QB even if the guard is bypassed.
  6. A projection edited to 0 AFTER pool creation (Players-tab override flow) must
     take the player out of the build.

Run from repo root:  python tests/test_exclusion_guards.py
"""
import io
import sys

sys.path.insert(0, ".")

import pandas as pd

import dfs_lab.data as _data_mod
from dfs_lab.data import prepare_player_pool, apply_football_reality_guard

# Synthetic players ("AAron Starter", ...) have no nflverse roster record. Stub
# live availability as unreachable so the guard fails open (its documented
# path) instead of excluding every synthetic player as teamless.
_EMPTY_DF = pd.DataFrame()
_data_mod._load_live_nfl_availability = lambda season=2026: (_EMPTY_DF, _EMPTY_DF)
from dfs_lab.classic import generate_lineups, classic_lineup_coherence

try:
    from dfs_lab.data import apply_post_edit_availability_gate
    HAS_GATE = True
except ImportError:
    HAS_GATE = False

SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]

# (ID, Name, Position, Roster Position, Team, Salary, Proj, Own)
PLAYERS = [
    # Team AAA (game AAA@BBB) — starter + backup QB
    (1001, "AAron Starter", "QB", "QB", "AAA", 7000, 20.0, 10.0),
    (1002, "AAron Backup", "QB", "QB", "AAA", 5000, 12.0, 2.0),
    (1003, "AAron Runner", "RB", "RB", "AAA", 6500, 16.0, 12.0),
    (1004, "AAron Change", "RB", "RB", "AAA", 4500, 9.0, 5.0),
    (1005, "AAron Wide1", "WR", "WR", "AAA", 6000, 14.0, 10.0),
    (1006, "AAron Wide2", "WR", "WR", "AAA", 5200, 11.0, 7.0),
    (1007, "AAron Wide3", "WR", "WR", "AAA", 4200, 8.0, 4.0),
    (1008, "AAron Tight", "TE", "TE", "AAA", 4500, 10.0, 8.0),
    (1009, "AAron Defense", "DST", "DST", "AAA", 3000, 7.0, 6.0),
    # Team BBB — injured starter (0 proj, high salary) + healthy backup
    (2001, "Baker Injured", "QB", "QB", "BBB", 7500, 0.0, 0.0),
    (2002, "Baker Healthy", "QB", "QB", "BBB", 5200, 14.0, 6.0),
    (2003, "Baker Runner", "RB", "RB", "BBB", 6200, 15.0, 11.0),
    (2004, "Baker Wide1", "WR", "WR", "BBB", 5800, 13.0, 9.0),
    (2005, "Baker Wide2", "WR", "WR", "BBB", 5000, 10.0, 6.0),
    (2006, "Baker Tight", "TE", "TE", "BBB", 4300, 9.0, 7.0),
    (2007, "Baker Defense", "DST", "DST", "BBB", 2800, 6.0, 5.0),
    # Team CCC (game CCC@DDD) — star RB for the manual-exclude test
    (3001, "Caleb Thrower", "QB", "QB", "CCC", 6800, 18.0, 9.0),
    (3002, "Caleb Star", "RB", "RB", "CCC", 8000, 22.0, 15.0),
    (3003, "Caleb Runner", "RB", "RB", "CCC", 5500, 12.0, 8.0),
    (3004, "Caleb Wide1", "WR", "WR", "CCC", 6300, 15.0, 11.0),
    (3005, "Caleb Wide2", "WR", "WR", "CCC", 4800, 10.0, 6.0),
    (3006, "Caleb Tight", "TE", "TE", "CCC", 4000, 8.0, 5.0),
    (3007, "Caleb Defense", "DST", "DST", "CCC", 3200, 8.0, 7.0),
    # Team DDD — includes a 0-projection WR
    (4001, "Derek Thrower", "QB", "QB", "DDD", 6600, 17.0, 8.0),
    (4002, "Derek Runner", "RB", "RB", "DDD", 6000, 13.0, 9.0),
    (4003, "Derek Wide1", "WR", "WR", "DDD", 5900, 13.0, 9.0),
    (4004, "Derek Wide2", "WR", "WR", "DDD", 5100, 11.0, 6.0),
    (4005, "Derek Zeroed", "WR", "WR", "DDD", 4000, 0.0, 0.0),
    (4006, "Derek Tight", "TE", "TE", "DDD", 4400, 9.0, 6.0),
    (4007, "Derek Defense", "DST", "DST", "DDD", 2900, 7.0, 5.0),
]

GAMES = {"AAA": "AAA@BBB 09/28/2026 01:00PM ET", "BBB": "AAA@BBB 09/28/2026 01:00PM ET",
         "CCC": "CCC@DDD 09/28/2026 04:05PM ET", "DDD": "CCC@DDD 09/28/2026 04:05PM ET"}


def dk_file():
    lines = ["Position,Name + ID,Salary,Name,ID,Roster Position,Game Info,TeamAbbrev,AvgPointsPerGame"]
    for pid, name, pos, rp, team, sal, proj, own in PLAYERS:
        # DK marks skill-position FLEX eligibility in Roster Position; the app
        # requires a FLEX token to fill the FLEX slot (is_FLEX regex \bFLEX\b).
        rp_out = {"RB": "RB/FLEX", "WR": "WR/FLEX", "TE": "TE/FLEX"}.get(rp, rp)
        lines.append(f"{pos},{name} ({pid}),{sal},{name},{pid},{rp_out},{GAMES[team]},{team},{proj}")
    return io.BytesIO("\n".join(lines).encode("utf-8"))


def ss_file():
    lines = ["Name,My Proj,My Own"]
    for pid, name, pos, rp, team, sal, proj, own in PLAYERS:
        lines.append(f"{name},{proj},{own}")
    return io.BytesIO("\n".join(lines).encode("utf-8"))


def build(count=20, strategy_map=None, seed=7, df=None, attempts=400):
    df = prepare_player_pool(dk_file(), ss_file()) if df is None else df
    return generate_lineups(
        df, 500, "GPP / top-heavy", count, attempts, 40000, 1, "Optional",
        [], strategy_map or {}, {}, False, False, seed,
    )


def appearances(res, name):
    n = 0
    for _, r in res.iterrows():
        for s in SLOTS:
            if r[s] == name:
                n += 1
    return n


def test1_backup_qb_auto_excluded():
    df = prepare_player_pool(dk_file(), ss_file())
    bq = df[df["Name"] == "AAron Backup"].iloc[0]
    st = df[df["Name"] == "AAron Starter"].iloc[0]
    assert not bool(bq["ActiveForBuild"]), "backup QB still ActiveForBuild"
    assert "Backup QB" in str(bq["Role Confidence"]), f"backup QB mislabeled: {bq['Role Confidence']}"
    assert str(bq["Auto Excluded Reason"]).strip() != "", "backup QB has no exclusion reason"
    assert bool(st["ActiveForBuild"]) and bool(st["Primary QB"]), "starter should be primary+active"
    print("PASS test1: backup QB auto-excluded, starter primary")


def test2_injured_starter_does_not_keep_primary():
    df = prepare_player_pool(dk_file(), ss_file())
    inj = df[df["Name"] == "Baker Injured"].iloc[0]
    healthy = df[df["Name"] == "Baker Healthy"].iloc[0]
    assert not bool(inj["ActiveForBuild"]), "injured 0-proj starter is still active"
    assert not bool(inj["Primary QB"]), "injured starter kept the Primary QB flag"
    assert bool(healthy["ActiveForBuild"]), "healthy backup was deactivated (salary-first bug)"
    assert bool(healthy["Primary QB"]), "healthy backup not marked primary"
    print("PASS test2: healthy backup is primary+active, injured starter out")


def test3_zero_projection_deactivated_with_reason():
    df = prepare_player_pool(dk_file(), ss_file())
    z = df[df["Name"] == "Derek Zeroed"].iloc[0]
    assert not bool(z["ActiveForBuild"]), "0-proj player still active"
    assert str(z["Auto Excluded Reason"]).strip() != "", "0-proj player has no exclusion reason"
    print("PASS test3: 0-proj player deactivated with reason:", str(z["Auto Excluded Reason"]).strip()[:60])


def test4_manual_exclude_never_appears():
    sm = {"3002": {"Lock": False, "Exclude": True, "Priority": "Exclude",
                   "Min Exposure": 0.0, "Max Exposure": 100.0}}
    res = build(count=20, strategy_map=sm)
    assert not res.empty, "build produced no lineups"
    n = appearances(res, "Caleb Star")
    assert n == 0, f"manually excluded player appeared {n}x in {len(res)} lineups"
    print(f"PASS test4: excluded star RB in 0 of {len(res)} lineups")


def test5_coherence_hard_rejects_backup_qb():
    df = prepare_player_pool(dk_file(), ss_file())

    def ix(name):
        return df.index[df["Name"] == name][0]

    chosen = [("QB", ix("AAron Backup")), ("RB1", ix("AAron Runner")), ("RB2", ix("Caleb Runner")),
              ("WR1", ix("AAron Wide1")), ("WR2", ix("Caleb Wide1")), ("WR3", ix("Derek Wide1")),
              ("TE", ix("AAron Tight")), ("FLEX", ix("Baker Runner")), ("DST", ix("Caleb Defense"))]
    coh = classic_lineup_coherence(df, chosen)
    assert not coh["Accept"], "coherence ACCEPTED a lineup with a backup QB"
    print("PASS test5: coherence hard-rejects backup QB:", coh["Coherence Flags"][:80])


def test6_zeroed_projection_after_pool_creation():
    # Simulates the Players-tab flow: pool built from upload, user then edits a
    # projection to 0 ("not playing"). Without a re-gate, ActiveForBuild is stale.
    df = prepare_player_pool(dk_file(), ss_file())
    df.loc[df["Name"] == "Caleb Star", "My Proj"] = 0.0
    if HAS_GATE:
        # This is exactly what the fixed Classic UI does after applying overrides.
        df = apply_post_edit_availability_gate(df, "My Proj", "ActiveForBuild", 0.05)
    else:
        print("  (no post-edit gate available: exercising the raw stale pool)")
    row = df[df["Name"] == "Caleb Star"].iloc[0]
    assert not bool(row["ActiveForBuild"]), "zeroed-projection player still ActiveForBuild"
    assert str(row["Auto Excluded Reason"]).strip() != "", "zeroed player has no exclusion reason"
    res = build(count=20, df=df, seed=11)
    assert not res.empty, "build produced no lineups"
    n = appearances(res, "Caleb Star")
    assert n == 0, f"zeroed-projection player appeared {n}x in {len(res)} lineups"
    print(f"PASS test6: post-edit zeroed player out, 0 of {len(res)} lineups")


if __name__ == "__main__":
    test1_backup_qb_auto_excluded()
    test2_injured_starter_does_not_keep_primary()
    test3_zero_projection_deactivated_with_reason()
    test4_manual_exclude_never_appears()
    test5_coherence_hard_rejects_backup_qb()
    test6_zeroed_projection_after_pool_creation()
    print("\nALL EXCLUSION-GUARD TESTS PASSED")
