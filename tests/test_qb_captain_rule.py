"""Regression test: the QB-Captain pass-catcher rule must not brick the slate.

Background: with "When QB is CPT - pass catchers" set to "Minimum 1", every
Showdown build returned zero lineups and the app fell back to coherence
scoring with a warning. Root cause: solve_showdown_one() used `n` both for
the player count (n = len(df)) and for the pass-catcher minimum
(n = abs(int(cpt_qb_passcatchers))). The QB-constraint block clobbered n to 1,
so post-solve slot extraction only inspected player 0 and rejected every
solver result.

Covers:
  1. cpt_qb_passcatchers=1 and =2 return real lineups (not None).
  2. When the Captain is a QB, the FLEX contains >= N same-team pass catchers.
  3. cpt_qb_passcatchers=-1 ("No more than 1") also returns lineups.

Run from repo root:  python3 tests/test_qb_captain_rule.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from dfs_lab.showdown import solve_showdown_one, SHOWDOWN_SLOTS


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def _pool():
    players = [
        ("p1", "Jalen Hurts", "QB", "PHI", 10800, 20.1, 40.5, True),
        ("p2", "Saquon Barkley", "RB", "PHI", 9800, 16.9, 30.0, True),
        ("p3", "DeVonta Smith", "WR", "PHI", 8600, 14.2, 45.5, True),
        ("p4", "A.J. Brown", "WR", "PHI", 9200, 15.1, 35.0, True),
        ("p6", "Zach Ertz", "TE", "PHI", 3800, 7.5, 9.8, True),
        ("p7", "Jake Elliott", "K", "PHI", 4800, 8.0, 12.0, True),
        ("p8", "PHI DST", "DST", "PHI", 4200, 7.0, 10.0, True),
        ("q1", "Case Keenum", "QB", "CHI", 9000, 12.0, 35.5, True),
        ("q2", "D'Andre Swift", "RB", "CHI", 7400, 13.5, 25.0, True),
        ("q3", "Rome Odunze", "WR", "CHI", 6800, 11.0, 20.0, True),
        ("q4", "Luther Burden III", "WR", "CHI", 5200, 9.0, 15.0, True),
        ("q5", "Cole Kmet", "TE", "CHI", 4600, 8.5, 18.0, True),
        ("q6", "Cairo Santos", "K", "CHI", 4600, 8.2, 14.0, True),
        ("q7", "CHI DST", "DST", "CHI", 4000, 6.5, 8.0, True),
    ]
    df = pd.DataFrame(
        players,
        columns=["ID", "Name", "Position", "Team", "Salary", "My Proj", "My Own", "ActiveForBuild"],
    )
    pos = df["Position"].str.upper()
    df["is_QB"] = pos.eq("QB")
    df["is_RB"] = pos.eq("RB")
    df["is_WR"] = pos.eq("WR")
    df["is_TE"] = pos.eq("TE")
    df["is_DST"] = pos.eq("DST")
    df["is_K"] = pos.eq("K")
    df["is_passcatcher"] = df["is_WR"] | df["is_TE"]
    df["CPT Own"] = df["My Own"]
    df["Aytia Proj"] = df["My Proj"]
    df["FlexSalary"] = df["Salary"]
    df["CaptainSalary"] = (df["Salary"] * 1.5).round(-2)
    df["Opponent"] = df["Team"].map({"PHI": "CHI", "CHI": "PHI"})
    return df


def _lineups(qb_pc, seeds):
    df = _pool()
    strategy_map = {pid: {"CPT Eligible": True} for pid in df["ID"]}
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        res = solve_showdown_one(
            df, 0.5, rng, strategy_map, 0, 50000, ("PHI", 4),
            "Neutral", "", qb_pc, 0, 0, 2, 1, 0, [],
        )
        out.append((df, res))
    return out


def _qb_captain_ok(df, chosen, n_min):
    slot = {s: i for s, i in chosen}
    cpt = df.loc[slot["CPT"]]
    if not bool(cpt["is_QB"]):
        return True
    flex = df.loc[[i for s, i in chosen if s != "CPT"]]
    pcs = flex[flex["Team"].eq(cpt["Team"]) & flex["is_passcatcher"]]
    return len(pcs) >= n_min


def main():
    seeds = [100, 101, 102, 103, 104]

    for qb_pc, n_min in ((1, 1), (2, 2)):
        runs = _lineups(qb_pc, seeds)
        got = [res for _, res in runs if res]
        _check(f"Minimum {n_min}: {len(got)}/{len(seeds)} attempts return a lineup", len(got) == len(seeds))
        ok = all(_qb_captain_ok(df, res, n_min) for df, res in runs if res)
        _check(f"Minimum {n_min}: QB captains all have >= {n_min} same-team pass catcher(s)", ok)

    runs = _lineups(-1, seeds)
    got = [res for _, res in runs if res]
    _check(f"No more than 1: {len(got)}/{len(seeds)} attempts return a lineup", len(got) == len(seeds))

    print("All QB-captain pass-catcher regression tests passed.")


if __name__ == "__main__":
    main()
