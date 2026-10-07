"""Trust sweep: excluded players must never appear in generated lineups.

Covers every exclusion path:
1. Exclude=True in strategy_map (Out checkbox / Quick Out button)
2. Priority="Exclude" (alternate exclusion signal)
3. Exclude + Lock combined (Exclude wins — player must not appear)
4. Exclude + CPT Lock combined (Exclude wins — player must not captain)

Also verifies the post-build audit catches violations if they ever occur.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np

from dfs_lab.showdown import (
    generate_showdown_lineups,
    audit_showdown_portfolio,
    captain_pool_ids,
)


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def _pool():
    # 16-player 2-team showdown pool with all required columns.
    # DAL: QB, 2 RB, 3 WR, TE, K, DST. TB: QB, 2 RB, 3 WR, TE, K.
    names = ["Dak Prescott", "Javonte Williams", "Hunter Luepke", "CeeDee Lamb",
             "George Pickens", "Ryan Flournoy", "Jake Ferguson", "Brandon Aubrey",
             "Jalon Daniels", "Bucky Irving", "Sean Tucker", "Chris Godwin",
             "Emeka Egbuka", "Tez Johnson", "Cade Otton", "Chase McLaughlin"]
    poss = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "K",
            "QB", "RB", "RB", "WR", "WR", "WR", "TE", "K"]
    teams = ["DAL"] * 8 + ["TB"] * 8
    sals = [10800, 6200, 3200, 10200, 8400, 4200, 5600, 4800,
            7500, 8200, 3800, 7800, 6800, 3600, 5200, 4600]
    projs = [18.5, 11.0, 4.0, 16.2, 13.5, 6.0, 8.5, 8.0,
             12.0, 14.5, 5.5, 12.8, 10.2, 5.0, 7.8, 7.5]
    owns = [25.0, 18.0, 3.0, 30.0, 22.0, 5.0, 12.0, 10.0,
            8.0, 20.0, 4.0, 16.0, 12.0, 4.0, 9.0, 9.0]
    ids = [str(100 + i) for i in range(len(names))]
    return pd.DataFrame({
        "ID": ids,
        "Name": names,
        "CPT_NameID": [f"{n} ({i})" for n, i in zip(names, ids)],
        "FLEX_NameID": [f"{n} ({i})" for n, i in zip(names, ids)],
        "Position": poss,
        "Team": teams,
        "FlexSalary": sals,
        "CaptainSalary": [int(s * 1.5) for s in sals],
        "My Proj": projs,
        "Aytia Proj": projs,
        "My Own": owns,
        "CPT Own": [o * 0.5 for o in owns],
        "Game Info": ["TB@DAL"] * len(names),
        "Away": ["TB"] * len(names),
        "Home": ["DAL"] * len(names),
        "Opponent": ["TB"] * 8 + ["DAL"] * 8,
        "is_QB": [p == "QB" for p in poss],
        "is_RB": [p == "RB" for p in poss],
        "is_WR": [p == "WR" for p in poss],
        "is_TE": [p == "TE" for p in poss],
        "is_DST": [False] * len(names),
        "is_K": [p == "K" for p in poss],
        "is_passcatcher": [p in ("WR", "TE") for p in poss],
        "ActiveForBuild": [True] * len(names),
    })


def _lineup_ids(result):
    ids = set()
    for _, row in result.iterrows():
        ids.add(str(row["CPT_ID"]))
        for i in range(1, 6):
            ids.add(str(row.get(f"FLEX{i}_ID", "")))
    return ids


def test_exclude_true_never_appears():
    df = _pool()
    # Exclude Jalon Daniels (ID 102) via the Out checkbox path.
    sm = {"108": {"Exclude": True, "CPT Eligible": False}}
    res = generate_showdown_lineups(
        df, 5000, "GPP / top-heavy", 10, 500, 30000, 50000,
        {"3-3": 1, "4-2": 1, "5-1": 0}, "Neutral", "", sm, "20-Max",
        0, 80, 35, 1, 1, 1, 42)
    _check("exclude-true: build produced lineups", res is not None and not res.empty)
    _check("exclude-true: Daniels in zero lineups", "108" not in _lineup_ids(res))


def test_priority_exclude_never_appears():
    df = _pool()
    sm = {"109": {"Priority": "Exclude"}}  # Bucky Irving via Priority path
    res = generate_showdown_lineups(
        df, 5000, "GPP / top-heavy", 10, 500, 30000, 50000,
        {"3-3": 1, "4-2": 1, "5-1": 0}, "Neutral", "", sm, "20-Max",
        0, 80, 35, 1, 1, 1, 42)
    _check("priority-exclude: build produced lineups", res is not None and not res.empty)
    _check("priority-exclude: Irving in zero lineups", "109" not in _lineup_ids(res))


def test_exclude_beats_lock():
    df = _pool()
    # User checked both Lock AND Out (Out should win).
    sm = {"103": {"Exclude": True, "Lock": True}}
    res = generate_showdown_lineups(
        df, 5000, "GPP / top-heavy", 10, 500, 30000, 50000,
        {"3-3": 1, "4-2": 1, "5-1": 0}, "Neutral", "", sm, "20-Max",
        0, 80, 35, 1, 1, 1, 42)
    _check("exclude-beats-lock: build produced lineups", res is not None and not res.empty)
    _check("exclude-beats-lock: Lamb in zero lineups", "103" not in _lineup_ids(res))


def test_exclude_beats_cpt_lock():
    df = _pool()
    sm = {"100": {"Exclude": True, "CPT Lock": True, "CPT Eligible": True}}
    res = generate_showdown_lineups(
        df, 5000, "GPP / top-heavy", 10, 500, 30000, 50000,
        {"3-3": 1, "4-2": 1, "5-1": 0}, "Neutral", "", sm, "20-Max",
        0, 80, 35, 1, 1, 1, 42)
    _check("exclude-beats-cpt-lock: build produced lineups", res is not None and not res.empty)
    _check("exclude-beats-cpt-lock: Dak in zero lineups", "100" not in _lineup_ids(res))


def test_audit_catches_violation():
    df = _pool()
    sm = {"108": {"Exclude": True}}
    # Fabricate a violating lineup (Daniels in FLEX1).
    bad = pd.DataFrame([{
        "Rank": 1, "CPT": "Dak Prescott", "CPT_ID": "100",
        "FLEX1": "Jalon Daniels", "FLEX1_ID": "108",
        "FLEX2": "CeeDee Lamb", "FLEX2_ID": "103",
        "FLEX3": "Bucky Irving", "FLEX3_ID": "109",
        "FLEX4": "Jake Ferguson", "FLEX4_ID": "104",
        "FLEX5": "Brandon Aubrey", "FLEX5_ID": "106",
    }])
    id_to_name = {str(r["ID"]): str(r["Name"]) for _, r in df.iterrows()}
    viols = audit_showdown_portfolio(bad, sm, id_to_name)
    _check("audit catches excluded player in lineup", len(viols) > 0 and "108" in viols[0] or any("Out" in v for v in viols))


def test_captain_pool_respects_exclude():
    df = _pool()
    sm = {"100": {"Exclude": True, "CPT Eligible": True},
          "103": {"CPT Eligible": True}}
    ids = captain_pool_ids(df, sm)
    _check("excluded QB not in captain pool", "100" not in ids)
    _check("eligible WR in captain pool", "103" in ids)


if __name__ == "__main__":
    test_exclude_true_never_appears()
    test_priority_exclude_never_appears()
    test_exclude_beats_lock()
    test_exclude_beats_cpt_lock()
    test_audit_catches_violation()
    test_captain_pool_respects_exclude()
    print("All exclusion trust tests passed.")
