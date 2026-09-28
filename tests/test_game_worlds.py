"""Regression tests for Game Worlds correctness.

Two user-reported bugs (Sep 28, 2026, PHI@CHI slate):

1. A "Caleb Williams ceiling" world was generated even though Williams is OUT
   (inactive). Root cause: configure_game_worlds() picked the "primary QB"
   from the nflverse depth-chart flag, which still pointed at the inactive
   starter. Worlds must only be named after active (ActiveForBuild) players.

2. Under a "Low-scoring game" scenario the portfolio still drew ~16% from the
   inactive QB's ceiling world and ~14% each from both passing-ceiling worlds:
   only the "shootout" family was downweighted, while pass_ceiling/qb_ceiling
   kept full weight -- nearly contradicting the user's thesis. Low-scoring
   scripts now downweight those families too (kept as rare alternatives, not
   removed).

Run from repo root:  python3 tests/test_game_worlds.py
"""
import sys

sys.path.insert(0, ".")

import pandas as pd

import dfs_lab.showdown as sd


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def _slate_df():
    # CHI QBs: Williams (OUT, still flagged Primary QB by the depth chart),
    # Keenum (active starter). PHI: Hurts active.
    return pd.DataFrame({
        "Name": ["Caleb Williams", "Case Keenum", "Jalen Hurts", "Saquon Barkley"],
        "Team": ["CHI", "CHI", "PHI", "PHI"],
        "Position": ["QB", "QB", "QB", "RB"],
        "is_QB": [True, True, True, False],
        "Primary QB": [True, False, True, False],
        "ActiveForBuild": [False, True, True, True],
        "FlexSalary": [9000, 5600, 10800, 10000],
        "AvgPointsPerGame": [0.0, 8.0, 24.0, 18.0],
        "My Proj": [0.0, 12.0, 20.1, 16.9],
        "Game Info": ["PHI@CHI"] * 4,
    })


def test_no_world_named_after_inactive_qb():
    sd.configure_game_worlds(_slate_df())
    names = list(sd.GAME_WORLDS.keys())
    _check("no Caleb Williams ceiling world", "Caleb Williams ceiling" not in names)
    _check("active backup gets the ceiling world instead",
           "Case Keenum ceiling" in names)
    _check("active PHI QB keeps his world", "Jalen Hurts ceiling" in names)


def test_no_active_qb_means_no_qb_world():
    df = _slate_df()
    df.loc[df["Team"].eq("CHI"), "ActiveForBuild"] = False
    sd.configure_game_worlds(df)
    names = list(sd.GAME_WORLDS.keys())
    _check("no CHI qb-ceiling world when no active CHI QB",
           "Case Keenum ceiling" not in names and "Caleb Williams ceiling" not in names)


def test_low_script_downweights_passing_worlds():
    sd.configure_game_worlds(_slate_df())
    neutral = sd.contest_world_weights("20-Max", 50000, "Neutral")
    low = sd.contest_world_weights("20-Max", 50000, "Low-scoring game")
    for fam, world in [("pass_ceiling", "CHI passing ceiling"),
                       ("qb_ceiling", "Case Keenum ceiling"),
                       ("comeback", "PHI leads / CHI comeback")]:
        _check(f"{fam} world downweighted under low-scoring script",
               low[world] < neutral[world])
    _check("low-scoring world boosted under low-scoring script",
           low["Low-scoring game"] > neutral["Low-scoring game"])
    # Alternatives survive: nothing is zeroed out entirely.
    _check("passing worlds remain possible (not eliminated)",
           all(low[w] > 0 for w in ["CHI passing ceiling", "Case Keenum ceiling"]))


if __name__ == "__main__":
    test_no_world_named_after_inactive_qb()
    test_no_active_qb_means_no_qb_world()
    test_low_script_downweights_passing_worlds()
    print("game world tests done")
