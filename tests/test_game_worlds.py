"""Regression tests for Game Worlds correctness.

User-reported issues (TB@DAL slate, Oct 7, 2026):

1. Duplicate ceiling worlds: every slate generated both "{TEAM} passing ceiling"
   and "{QB NAME} ceiling" worlds for the same team -- the same story twice,
   eating ~20% of the portfolio. Worse, the QB-named world could go stale: a
   "Baker Mayfield ceiling" world survived after Mayfield was ruled OUT
   (Sep 28 had the same bug with Caleb Williams). QB-named worlds were removed
   entirely; one team-named passing-ceiling world per team. Teams can't be
   inactive, so the stale-name class is gone.

2. Under a "Low-scoring game" scenario the portfolio still drew ~16% from the
   inactive QB's ceiling world and ~14% each from both passing-ceiling worlds:
   only the "shootout" family was downweighted. Low-scoring scripts now
   downweight passing families too (kept as rare alternatives, not removed).

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


def test_no_qb_named_worlds():
    # QB-named ceiling worlds were removed: they duplicated the team's
    # passing-ceiling world and could name an inactive player (Baker Mayfield,
    # Oct 7 2026; Caleb Williams, Sep 28 2026). Only team-named worlds remain.
    sd.configure_game_worlds(_slate_df())
    names = list(sd.GAME_WORLDS.keys())
    _check("no Caleb Williams ceiling world", "Caleb Williams ceiling" not in names)
    _check("no Case Keenum ceiling world", "Case Keenum ceiling" not in names)
    _check("no Jalen Hurts ceiling world", "Jalen Hurts ceiling" not in names)
    _check("no world named after any rostered QB",
           not any(n.endswith("ceiling") and "passing" not in n for n in names))
    _check("team passing-ceiling worlds still exist",
           "CHI passing ceiling" in names and "PHI passing ceiling" in names)


def test_low_script_downweights_passing_worlds():
    sd.configure_game_worlds(_slate_df())
    neutral = sd.contest_world_weights("20-Max", 50000, "Neutral")
    low = sd.contest_world_weights("20-Max", 50000, "Low-scoring game")
    for fam, world in [("pass_ceiling", "CHI passing ceiling"),
                       ("comeback", "PHI leads / CHI comeback")]:
        _check(f"{fam} world downweighted under low-scoring script",
               low[world] < neutral[world])
    _check("low-scoring world boosted under low-scoring script",
           low["Low-scoring game"] > neutral["Low-scoring game"])
    # Alternatives survive: nothing is zeroed out entirely.
    _check("passing worlds remain possible (not eliminated)",
           low["CHI passing ceiling"] > 0)


def test_directional_thesis_leads_world_mix():
    # Oct 1, 2026 (PIT@CLE): user picked "Low-scoring game" but the portfolio
    # drew 30% Balanced shootout / 25% CLE passing ceiling and zero
    # low-scoring worlds. A directional thesis must lead the mix (~45%
    # plurality); other worlds stay as alternatives, never eliminated.
    sd.configure_game_worlds(_slate_df())
    for script, fam in [("Low-scoring game", "low"),
                       ("Shootout", "shootout"),
                       ("Ground-and-pound", "rb_control")]:
        w = sd.contest_world_weights("Single Entry", 2000, script)
        tot = sum(w.values())
        fam_share = sum(v for n, v in w.items()
                        if sd.GAME_WORLDS[n].get("family") == fam) / tot
        _check(f"{script}: thesis family leads at ~45% (got {fam_share:.2f})",
               0.40 <= fam_share <= 0.50)
        _check(f"{script}: no world eliminated",
               all(v > 0 for v in w.values()))
    # Neutral keeps the old contest-driven mix (no plurality lift).
    w = sd.contest_world_weights("Single Entry", 2000, "Neutral")
    tot = sum(w.values())
    low_share = sum(v for n, v in w.items()
                    if sd.GAME_WORLDS[n].get("family") == "low") / tot
    _check(f"Neutral: low family stays a small side bet (got {low_share:.3f})",
           low_share < 0.10)


if __name__ == "__main__":
    test_no_qb_named_worlds()
    test_low_script_downweights_passing_worlds()
    test_directional_thesis_leads_world_mix()
    print("game world tests done")
