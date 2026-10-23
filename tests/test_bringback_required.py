"""Bring-back "Required" is a hard user rule (Oct 2026).

Before this fix, the bring-back worthiness gate (commit 4ca9867) silently
dropped the Required constraint in solve_one whenever the opposing offense
rated below league average: the user set "Required", the build came back
with no bring-back against LV/ATL-class offenses and no warning (Chris's
Week 7 export: 11 of 20 lineups missing the required bring-back). The gate
was designed for Aytia's SUGGESTED defaults; it must never override an
explicit user rule.

Covers, on synthetic data only:
  1. Required forces an opposing PASS CATCHER (WR/TE) even when the
     opponent is rated unworthy (worthy set empty / pointing elsewhere).
  2. An opposing RB alone does NOT satisfy Required: with every opposing
     WR/TE excluded, a locked QB anchored on that game builds ZERO
     lineups rather than one that violates the rule.
  3. bringback_worthy_teams returns DraftKings spellings ("LAR", not
     the nflverse "LA") so the remaining gated paths (Optional nudge,
     details label) can find a worthy offense by its pool name.

Run from repo root:  python3 tests/test_bringback_required.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import io
import pandas as pd

import dfs_lab.data as _data_mod
# Synthetic players have no nflverse roster record. Stub live availability as
# unreachable so the guard fails open instead of excluding everyone.
_EMPTY_DF = pd.DataFrame()
_data_mod._load_live_nfl_availability = lambda season=2026: (_EMPTY_DF, _EMPTY_DF)

from dfs_lab.data import prepare_player_pool
from dfs_lab.leverage import compute_mobile_qb_tags
import dfs_lab.classic as classic_mod
from dfs_lab.classic import generate_lineups, bringback_worthy_teams

CHECKS = []
def _check(name, cond, detail=""):
    CHECKS.append(name)
    if not cond:
        raise AssertionError(f"FAIL {name} {detail}")
    print(f"  ok: {name}")


# Same synthetic slate as tests/test_bringback_nudge.py: Alpha (AAA) hosts
# Bravo (BBB); Charlie/Echo/Foxtrot fill out the pool.
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
BBB_CATCHER_IDS = {"201", "202", "204"}   # Bravo WR/TE (203 is the RB)
SLOT_ID_COLS = ["QB_ID", "RB1_ID", "RB2_ID", "WR1_ID", "WR2_ID", "WR3_ID",
                "TE_ID", "FLEX_ID", "DST_ID"]


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


def _build(worthy, strategy=None, seed=7):
    df = _pool()
    strat = {"101": {"Lock": True}}
    strat.update(strategy or {})
    return generate_lineups(
        df, 500, "GPP / top-heavy", 12, 500, 40000, 1, "Required",
        [], strat, {}, False, False, seed,
        bringback_worthy=worthy,
    )


def _alpha_lineups(res):
    return [r for _, r in res.iterrows() if r["QB"] == "Alpha QB"]


def test_required_forces_catcher_vs_unworthy():
    for worthy in (set(), {"CCC"}, None):
        res = _build(worthy)
        rows = _alpha_lineups(res)
        _check(f"Required builds with a locked QB (worthy={worthy})", len(rows) > 0)
        bad = [i for i, r in enumerate(rows)
               if not ({str(r[c]) for c in SLOT_ID_COLS} & BBB_CATCHER_IDS)]
        _check(f"every locked-QB lineup carries a Bravo WR/TE (worthy={worthy})",
               not bad, f"violating rows: {bad}")


def test_required_rb_alone_cannot_satisfy():
    # Exclude every Bravo pass catcher. Bravo's RB remains, but a
    # bring-back is a pass catcher: the locked Alpha QB must build
    # NOTHING rather than a lineup whose only bring-back is the RB.
    strat = {pid: {"Exclude": True} for pid in ("201", "202", "204")}
    res = _build({"BBB"}, strategy=strat)
    _check("no lineup is produced when Required cannot be satisfied",
           len(res) == 0, f"got {len(res)} lineups")


def test_worthy_set_uses_dk_spellings():
    orig = classic_mod._nflverse_team_ratings
    classic_mod._nflverse_team_ratings = lambda season: (
        {"LA": 30.0, "LV": 10.0}, {}, 22.0)
    try:
        worthy = bringback_worthy_teams(pd.DataFrame({"x": [1]}))
    finally:
        classic_mod._nflverse_team_ratings = orig
    _check("worthy set is DK-keyed (LAR in, LA out)",
           worthy == {"LAR"}, f"got {worthy}")


if __name__ == "__main__":
    test_required_forces_catcher_vs_unworthy()
    test_required_rb_alone_cannot_satisfy()
    test_worthy_set_uses_dk_spellings()
    print(f"ALL BRING-BACK REQUIRED TESTS PASSED ({len(CHECKS)} checks)")
