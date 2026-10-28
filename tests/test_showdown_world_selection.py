"""WS26 Piece 1 — sim-driven showdown selection and rating.

The projection MILP is demoted to candidate supplier: world-authored
lineups (the optimal legal lineup of sampled sim worlds) join the
pool, selection walks the pool in world-score order, and the rating
is anchored on world metrics (Best-of-pool %, Top-decile %) instead
of percentiles of the projection column inside the delivered set.

Invariants pinned here:
  - world metrics are exact on hand-set worlds;
  - world-authored lineups obey the build's legality (excludes,
    locks, the captain pool, the salary band, both teams, the
    QB-linkage constraint);
  - select_showdown_portfolio(order_scores=...) changes the pick
    order and nothing else — with order_scores=None the legacy
    rating order is byte-identical;
  - the world rating promotes the lineup that wins worlds over the
    lineup that merely projects best, and falls back to the legacy
    composite when the world columns are absent.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from dfs_lab.showdown import (
    add_showdown_ratings,
    select_showdown_portfolio,
    showdown_candidate_row,
    showdown_lineup_world_metrics,
    showdown_world_optimal_lineups,
    showdown_world_scores,
)


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def _mk(rows):
    """rows: (id, name, pos, team, flex_sal, proj, own) tuples."""
    return pd.DataFrame({
        "ID": [r[0] for r in rows],
        "Name": [r[1] for r in rows],
        "CPT_NameID": [f"{r[1]} ({r[0]})" for r in rows],
        "FLEX_NameID": [f"{r[1]} ({r[0]})" for r in rows],
        "Position": [r[2] for r in rows],
        "Team": [r[3] for r in rows],
        "FlexSalary": [r[4] for r in rows],
        "CaptainSalary": [int(r[4] * 1.5) for r in rows],
        "My Proj": [r[5] for r in rows],
        "Aytia Proj": [r[5] for r in rows],
        "My Own": [r[6] for r in rows],
        "CPT Own": [r[6] * 0.5 for r in rows],
        "Game Info": ["TB@DAL"] * len(rows),
        "Away": ["TB"] * len(rows),
        "Home": ["DAL"] * len(rows),
        "Opponent": ["TB" if r[3] == "DAL" else "DAL" for r in rows],
        "is_QB": [r[2] == "QB" for r in rows],
        "is_RB": [r[2] == "RB" for r in rows],
        "is_WR": [r[2] == "WR" for r in rows],
        "is_TE": [r[2] == "TE" for r in rows],
        "is_DST": [r[2] == "DST" for r in rows],
        "is_K": [r[2] == "K" for r in rows],
        "is_passcatcher": [r[2] in ("WR", "TE") for r in rows],
        "ActiveForBuild": [True] * len(rows),
    })


def _pool():
    return _mk([
        ("100", "Dak Prescott", "QB", "DAL", 10800, 18.5, 25.0),
        ("101", "Javonte Williams", "RB", "DAL", 6200, 11.0, 18.0),
        ("102", "Hunter Luepke", "RB", "DAL", 3200, 4.0, 3.0),
        ("103", "CeeDee Lamb", "WR", "DAL", 10200, 16.2, 30.0),
        ("104", "George Pickens", "WR", "DAL", 8400, 13.5, 22.0),
        ("105", "Ryan Flournoy", "WR", "DAL", 4200, 6.0, 5.0),
        ("106", "Jake Ferguson", "TE", "DAL", 5600, 8.5, 12.0),
        ("107", "Brandon Aubrey", "K", "DAL", 4800, 8.0, 10.0),
        ("108", "DAL DST", "DST", "DAL", 4000, 7.0, 8.0),
        ("109", "Baker Mayfield", "QB", "TB", 9800, 17.0, 20.0),
        ("110", "Bucky Irving", "RB", "TB", 6600, 11.5, 15.0),
        ("111", "Mike Evans", "WR", "TB", 9000, 14.5, 24.0),
        ("112", "Chris Godwin", "WR", "TB", 7600, 12.0, 18.0),
        ("113", "Emeka Egbuka", "WR", "TB", 5400, 7.5, 9.0),
        ("114", "Cade Otton", "TE", "TB", 4600, 6.5, 7.0),
        ("115", "Chase McLaughlin", "K", "TB", 4400, 7.5, 6.0),
        ("116", "TB DST", "DST", "TB", 3600, 6.5, 5.0),
    ])


def _row(pool, cpt_id, flex_ids, strategy=None):
    pos = {str(pid): i for i, pid in enumerate(pool["ID"].astype(str))}
    chosen = [("CPT", pos[cpt_id])] + [
        (f"FLEX{j}", pos[p]) for j, p in enumerate(flex_ids, start=1)]
    row = showdown_candidate_row(pool, chosen, strategy or {},
                                 "Neutral", "")
    assert row is not None, (cpt_id, flex_ids)
    return row


def test_world_metrics_exact():
    pool = _pool()
    W = np.zeros((10, len(pool)))
    pos = {str(pid): i for i, pid in enumerate(pool["ID"].astype(str))}
    for pid in ("103", "101", "111", "112", "106"):  # shared flexes
        W[:, pos[pid]] = 10.0
    W[:6, pos["100"]] = 40.0   # Dak spikes in worlds 0-5
    W[6:, pos["109"]] = 20.0   # Baker spikes in worlds 6-9
    shared = ("103", "101", "111", "112", "106")
    pairs = [("100", ("100",) + shared), ("109", ("109",) + shared)]
    m = showdown_lineup_world_metrics(W, pool, pairs)
    _check("metrics: L1 best-of-pool 60%",
           abs(m["Best-of-pool %"].iloc[0] - 60.0) < 1e-6)
    _check("metrics: L2 best-of-pool 40%",
           abs(m["Best-of-pool %"].iloc[1] - 40.0) < 1e-6)
    _check("metrics: L1 sim mean 86",
           abs(m["Sim Mean"].iloc[0] - 86.0) < 1e-6)
    _check("metrics: L2 sim mean 62",
           abs(m["Sim Mean"].iloc[1] - 62.0) < 1e-6)
    _check("metrics: L1 ceiling P90 110",
           abs(m["Ceiling P90"].iloc[0] - 110.0) < 1e-6)
    s = showdown_world_scores(m)
    _check("world scores rank the world-winner first", s[0] > s[1])


def test_world_optimal_legality():
    pool = _pool()
    rng = np.random.default_rng(3)
    W = rng.gamma(shape=4.0, scale=1.0,
                  size=(400, len(pool))) * (pool["My Proj"].to_numpy()
                                            / 4.0)
    strategy = {str(pid): {"CPT Eligible": False}
                for pid in pool["ID"]}
    for pid in ("100", "109", "103"):
        strategy[pid] = {"CPT Eligible": True}
    strategy["111"] = {"CPT Eligible": False, "Exclude": True}
    strategy["112"] = {"CPT Eligible": False, "Lock": True}
    opts = showdown_world_optimal_lineups(
        W, pool, n_sample=120, seed=5, salary_cap=50000, min_salary=0,
        strategy_map=strategy, max_k=2, max_dst=1)
    _check("world optimals exist", len(opts) > 0)
    by_id = {str(r["ID"]): r for _, r in pool.iterrows()}
    ok = True
    for o in opts:
        ids = set(o["player_ids"])
        if "111" in ids or "112" not in ids:
            ok = False
        if o["cpt_id"] not in ("100", "109", "103"):
            ok = False
        sal = int(by_id[o["cpt_id"]]["CaptainSalary"]) + sum(
            int(by_id[p]["FlexSalary"]) for p in ids if p != o["cpt_id"])
        if sal > 50000:
            ok = False
        teams = {str(by_id[p]["Team"]) for p in ids}
        if len(teams) < 2:
            ok = False
        for t in ("DAL", "TB"):
            catchers = [p for p in ids
                        if str(by_id[p]["Team"]) == t
                        and bool(by_id[p]["is_passcatcher"])]
            qb = [p for p in ids if str(by_id[p]["Team"]) == t
                  and bool(by_id[p]["is_QB"])]
            if len(catchers) >= 3 and not qb:
                ok = False
    _check("world optimals obey excludes/locks/captain pool/salary/"
           "both-teams/linkage", ok)


def test_selection_order():
    pool = _pool()
    rows = [
        _row(pool, "103", ("100", "110", "112", "114", "105")),  # chalk
        _row(pool, "102", ("100", "103", "104", "111", "114")),  # punt CPT
        _row(pool, "100", ("103", "101", "112", "113", "115")),  # QB CPT
    ]
    cands = add_showdown_ratings(pd.DataFrame(rows), 0.3)
    # The generator hands selection a rating-sorted pool; mirror
    # that contract, then attach world metrics per captain.
    cands = cands.sort_values("Rating Score", ascending=False,
                              kind="mergesort").reset_index(drop=True)
    legacy_first = str(cands.iloc[0]["CPT_ID"])
    # World metrics invert the order: the punt-captain lineup wins
    # the worlds; the QB-captain chalk wins none.
    world_by_cpt = {
        "103": {"Sim Mean": 70.0, "Ceiling P90": 90.0,
                "Best-of-pool %": 2.0, "Top-decile %": 10.0},
        "102": {"Sim Mean": 68.0, "Ceiling P90": 95.0,
                "Best-of-pool %": 30.0, "Top-decile %": 40.0},
        "100": {"Sim Mean": 75.0, "Ceiling P90": 99.0,
                "Best-of-pool %": 5.0, "Top-decile %": 12.0}}
    metrics = pd.DataFrame([world_by_cpt[str(r["CPT_ID"])]
                            for _, r in cands.iterrows()])
    scores = showdown_world_scores(metrics)
    ordered = [float(s) for s in scores]
    res_legacy, _ = select_showdown_portfolio(cands, pool, {}, 1)
    res_world, _ = select_showdown_portfolio(
        cands, pool, {}, 1, order_scores=ordered)
    _check("legacy selection takes the rating leader",
           str(res_legacy.iloc[0]["CPT_ID"]) == legacy_first)
    _check("world selection takes the world leader (punt CPT)",
           str(res_world.iloc[0]["CPT_ID"]) == "102")
    _check("the two orders actually differ",
           legacy_first != "102")


def test_world_rating():
    frame = pd.DataFrame([
        {"Projection": 90.0, "Salary Left": 0, "Correlation Raw": 10.0,
         "Total Own": 30.0, "Dup Raw": 5.0, "User Fit Raw": 5.0,
         "Coherence Score": 90.0, "Best-of-pool %": 1.0,
         "Top-decile %": 5.0, "Ceiling P90": 80.0, "Sim Mean": 60.0},
        {"Projection": 70.0, "Salary Left": 0, "Correlation Raw": 2.0,
         "Total Own": 30.0, "Dup Raw": 5.0, "User Fit Raw": 5.0,
         "Coherence Score": 90.0, "Best-of-pool %": 40.0,
         "Top-decile %": 50.0, "Ceiling P90": 85.0, "Sim Mean": 62.0},
    ])
    legacy = add_showdown_ratings(frame.copy(), 0.3)
    _check("legacy rating crowns the projection leader",
           legacy["Rating Score"].iloc[0] > legacy["Rating Score"].iloc[1])
    world = add_showdown_ratings(frame.copy(), 0.3,
                                 use_world_metrics=True)
    _check("world rating crowns the world winner",
           world["Rating Score"].iloc[1] > world["Rating Score"].iloc[0])
    bare = frame.drop(columns=["Best-of-pool %", "Top-decile %",
                               "Ceiling P90", "Sim Mean"])
    fallback = add_showdown_ratings(bare.copy(), 0.3,
                                    use_world_metrics=True)
    _check("world flag without world columns = legacy composite",
           list(fallback["Rating Score"]) == list(legacy["Rating Score"]))


if __name__ == "__main__":
    test_world_metrics_exact()
    test_world_optimal_legality()
    test_selection_order()
    test_world_rating()
    print("ALL PASS")
