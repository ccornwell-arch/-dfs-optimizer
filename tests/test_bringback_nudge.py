"""Bring-back nudge tests (Sep 2026).

"Optional" bring-back mode used to be a no-op: the per-player MILP objective
has no correlation term, so bring-backs appeared only by accident — the exact
failure mode in Fantasy Footballers' "DFS Optimizer Top 10 Mistakes" #9
("it only sees individual projections, not how players interact").
GPP best practice (FantasyLabs, Footballers) is ~50% game-stack rate: a real
lever, not a mandate.

The fix: when the first-pass lineup rides a QB whose opposing offense is
worthy of a shootout (league-average scoring or better) but carries no
bring-back, solve_one re-solves with a soft BRINGBACK_NUDGE bonus on that
opponent's active skill players. Same noise draw, so the only difference
between passes is the incentive.

Covers, on synthetic data only:
  1. bringback_nudge_vector decision rules (all branches, deterministic).
  2. Nudge bonus lands exactly on active opposing skill players.
  3. End-to-end: Optional + worthy opponent raises the bring-back rate vs
     the no-worthy control, across seeds.
  4. Required still forces; None still forbids.
  5. Nudge constant is sane (positive, below the hard stack bonuses).

Run from repo root:  python3 tests/test_bringback_nudge.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import io
import numpy as np
import pandas as pd

import dfs_lab.data as _data_mod
# Synthetic players have no nflverse roster record. Stub live availability as
# unreachable so the guard fails open instead of excluding everyone.
_EMPTY_DF = pd.DataFrame()
_data_mod._load_live_nfl_availability = lambda season=2026: (_EMPTY_DF, _EMPTY_DF)

from dfs_lab.data import prepare_player_pool
from dfs_lab.leverage import compute_mobile_qb_tags
from dfs_lab.classic import generate_lineups, bringback_nudge_vector
from dfs_lab.config import BRINGBACK_NUDGE

CHECKS = []
def _check(name, cond, detail=""):
    CHECKS.append(name)
    if not cond:
        raise AssertionError(f"FAIL {name} {detail}")
    print(f"  ok: {name}")


def _tiny_df():
    # (Name, is_QB, Team, Opponent, is_RB, is_WR, is_TE, ActiveForBuild)
    rows = [
        ("QB1",  True,  "AAA", "BBB", False, False, False, True),
        ("WR1",  False, "AAA", "BBB", False, True,  False, True),
        ("RB1",  False, "BBB", "AAA", True,  False, False, True),
        ("WR2",  False, "BBB", "AAA", False, True,  False, True),
        ("TE1",  False, "BBB", "AAA", False, False, True,  False),  # inactive
        ("DST1", False, "BBB", "AAA", False, False, False, True),   # DST: no bonus
    ]
    return pd.DataFrame(rows, columns=["Name", "is_QB", "Team", "Opponent",
                                       "is_RB", "is_WR", "is_TE", "ActiveForBuild"])


def test_nudge_vector_rules():
    df = _tiny_df()
    base = np.zeros(len(df))
    no_bb = [("QB", 0), ("WR", 1)]          # QB1 + stack partner, no bring-back
    with_bb = [("QB", 0), ("RB", 2)]       # QB1 + opposing RB
    no_qb = [("WR", 1), ("RB", 2)]

    v = bringback_nudge_vector(df, base, no_bb, "Optional", {"BBB"})
    _check("happy path returns a vector", v is not None)
    _check("bonus hits exactly the active opposing skill players",
           [round(x, 6) for x in v] == [0.0, 0.0, BRINGBACK_NUDGE, BRINGBACK_NUDGE, 0.0, 0.0],
           f"got {[round(x,3) for x in v]}")
    _check("input vector not mutated", (base == 0.0).all())
    _check("Required mode -> no nudge", bringback_nudge_vector(df, base, no_bb, "Required", {"BBB"}) is None)
    _check("None mode -> no nudge", bringback_nudge_vector(df, base, no_bb, "None", {"BBB"}) is None)
    _check("worthy=None -> no nudge", bringback_nudge_vector(df, base, no_bb, "Optional", None) is None)
    _check("worthy empty -> no nudge", bringback_nudge_vector(df, base, no_bb, "Optional", set()) is None)
    _check("unworthy opponent -> no nudge",
           bringback_nudge_vector(df, base, no_bb, "Optional", {"CCC"}) is None)
    _check("lineup already has bring-back -> no nudge",
           bringback_nudge_vector(df, base, with_bb, "Optional", {"BBB"}) is None)
    _check("no QB in lineup -> no nudge",
           bringback_nudge_vector(df, base, no_qb, "Optional", {"BBB"}) is None)
    _check("no chosen lineup -> no nudge",
           bringback_nudge_vector(df, base, None, "Optional", {"BBB"}) is None)


def test_nudge_constant_sane():
    _check("nudge positive", BRINGBACK_NUDGE > 0)
    _check("nudge below hard stack bonuses", BRINGBACK_NUDGE < 1.1,
           f"BRINGBACK_NUDGE={BRINGBACK_NUDGE}")


# (ID, Name, Position, Roster Position, Team, Salary, Proj, Own)
PLAYERS = [
    (101, "Alpha QB", "QB", "QB", "AAA", 7600, 24.0, 12.0),
    (102, "Alpha W1", "WR", "WR", "AAA", 6600, 17.0, 14.0),
    (103, "Alpha W2", "WR", "WR", "AAA", 5600, 13.0, 9.0),
    (104, "Alpha TE", "TE", "TE", "AAA", 4900, 11.0, 8.0),
    (105, "Alpha RB", "RB", "RB", "AAA", 7200, 18.0, 15.0),
    (106, "Alpha DST", "DST", "DST", "AAA", 3000, 7.0, 6.0),
    (201, "Bravo W1", "WR", "WR", "BBB", 6000, 10.4, 8.0),
    (202, "Bravo W2", "WR", "WR", "BBB", 5000, 8.5, 5.0),
    (203, "Bravo RB", "RB", "RB", "BBB", 6400, 12.0, 9.0),
    (204, "Bravo TE", "TE", "TE", "BBB", 4300, 8.0, 4.0),
    (205, "Bravo DST", "DST", "DST", "BBB", 2900, 6.5, 5.0),
    (301, "Charlie QB", "QB", "QB", "CCC", 7000, 19.0, 8.0),
    (302, "Charlie RB", "RB", "RB", "CCC", 7800, 21.0, 22.0),
    (303, "Charlie W1", "WR", "WR", "CCC", 6300, 14.8, 12.0),
    (304, "Charlie W2", "WR", "WR", "CCC", 5300, 11.6, 7.0),
    (305, "Charlie TE", "TE", "TE", "CCC", 4600, 10.0, 6.0),
    (306, "Charlie DST", "DST", "DST", "CCC", 3100, 7.5, 6.0),
    (401, "Delta QB", "QB", "QB", "DDD", 6800, 18.0, 7.0),
    (402, "Delta RB", "RB", "RB", "DDD", 6100, 13.8, 8.0),
    (403, "Delta W1", "WR", "WR", "DDD", 6100, 13.2, 9.0),
    (404, "Delta W2", "WR", "WR", "DDD", 5100, 10.6, 5.0),
    (405, "Delta DST", "DST", "DST", "DDD", 2800, 6.0, 4.0),
    (501, "Echo QB", "QB", "QB", "EEE", 6900, 18.5, 7.0),
    (502, "Echo RB", "RB", "RB", "EEE", 6800, 15.5, 10.0),
    (503, "Echo W1", "WR", "WR", "EEE", 5800, 12.8, 8.0),
    (504, "Echo W2", "WR", "WR", "EEE", 5200, 11.0, 6.0),
    (505, "Echo TE", "TE", "TE", "EEE", 4400, 9.0, 5.0),
    (506, "Echo DST", "DST", "DST", "EEE", 2950, 6.8, 5.0),
    (601, "Foxtrot QB", "QB", "QB", "FFF", 6700, 17.5, 6.0),
    (602, "Foxtrot RB", "RB", "RB", "FFF", 6300, 13.0, 7.0),
    (603, "Foxtrot W1", "WR", "WR", "FFF", 5700, 12.0, 7.0),
    (604, "Foxtrot W2", "WR", "WR", "FFF", 5000, 10.0, 5.0),
    (605, "Foxtrot DST", "DST", "DST", "FFF", 2850, 6.2, 4.0),
]
GAMES = {"AAA": "AAA@BBB 09/28/2026 01:00PM ET", "BBB": "AAA@BBB 09/28/2026 01:00PM ET",
         "CCC": "CCC@DDD 09/28/2026 04:05PM ET", "DDD": "CCC@DDD 09/28/2026 04:05PM ET",
         "EEE": "EEE@FFF 09/28/2026 08:15PM ET", "FFF": "EEE@FFF 09/28/2026 08:15PM ET"}
BBB_SKILL_IDS = {"201", "202", "203", "204"}


def _pool():
    lines = ["Position,Name + ID,Salary,Name,ID,Roster Position,Game Info,TeamAbbrev,AvgPointsPerGame"]
    for pid, name, pos, rp, team, sal, proj, own in PLAYERS:
        rp_out = {"RB": "RB/FLEX", "WR": "WR/FLEX", "TE": "TE/FLEX"}.get(rp, rp)
        lines.append(f"{pos},{name} ({pid}),{sal},{name},{pid},{rp_out},{GAMES[team]},{team},{proj}")
    dk = io.BytesIO("\n".join(lines).encode("utf-8"))
    ss = io.BytesIO(("Name,My Proj,My Own\n" + "\n".join(
        f"{name},{proj},{own}" for _, name, _, _, _, _, proj, own in PLAYERS)).encode("utf-8"))
    df = prepare_player_pool(dk, ss)
    df["Rush Share"] = 0.05
    df["Mobile QB"] = compute_mobile_qb_tags(df)
    return df


def _bb_rate(res):
    n = bb = 0
    for _, r in res.iterrows():
        if r["QB"] != "Alpha QB":
            continue
        n += 1
        ids = {str(r[c]) for c in ["RB1_ID", "RB2_ID", "WR1_ID", "WR2_ID", "WR3_ID", "TE_ID", "FLEX_ID"]}
        if ids & BBB_SKILL_IDS:
            bb += 1
    return bb / n if n else 0.0


def _build(mode, worthy, seed):
    df = _pool()
    return generate_lineups(
        df, 500, "GPP / top-heavy", 30, 800, 40000, 1, mode,
        [], {"101": {"Lock": True}}, {}, False, False, seed,
        bringback_worthy=worthy,
    )


def test_nudge_lifts_bringback_rate():
    rn = _build("Optional", {"BBB"}, seed=7)
    rc = _build("Optional", None, seed=7)
    r_n, r_c = _bb_rate(rn), _bb_rate(rc)
    _check("nudge portfolio builds", len(rn) > 0)
    _check("control portfolio builds", len(rc) > 0)
    _check("Optional+worthy out-bring-backs the no-worthy control",
           r_n > r_c, f"nudge={r_n:.2f} control={r_c:.2f}")


def test_other_modes_unchanged():
    rq = _build("Required", {"BBB"}, seed=7)
    _check("Required still forces a bring-back vs worthy offense",
           _bb_rate(rq) == 1.0, f"got {_bb_rate(rq):.2f}")
    nn = _build("None", {"BBB"}, seed=7)
    _check("None still forbids opposing skill players",
           _bb_rate(nn) == 0.0, f"got {_bb_rate(nn):.2f}")


if __name__ == "__main__":
    test_nudge_vector_rules()
    test_nudge_constant_sane()
    test_nudge_lifts_bringback_rate()
    test_other_modes_unchanged()
    print(f"ALL BRING-BACK NUDGE TESTS PASSED ({len(CHECKS)} checks)")
