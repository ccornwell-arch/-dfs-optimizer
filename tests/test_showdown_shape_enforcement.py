"""Build Shape enforcement on the WS26 supply, selection, and audit
(Oct 28, 2026).

The incident: Chris built CAR@GB with construction weights
{"3-3": 1, "4-2": 1, "5-1": 0} — 5-1 unchecked — and 18 of the 20
delivered lineups were 5-1. The projection MILP honored his zeros,
but the world-authored candidate supply never received the weights
(its only team constraint was "both teams represented", which
legalizes 5-1), the portfolio selection never checked shapes, and
the audit's challengers were solved under the same loose [1, 5]
team rows. Selection walked world-score order, so the 5-1-heavy
world candidates crowded out the shape-legal supply.

Invariants pinned here:
  - world optimals under a restricted weight set contain ZERO
    disallowed splits and non-zero supply; with all shapes allowed
    (or weights None) the supply is byte-identical to before, 5-1
    splits included;
  - selection with order_scores rigged to favor 5-1 candidates
    still delivers ZERO disallowed shapes, and names the Build
    Shape exclusion when the legal supply cannot meet the count;
  - selection with weights=None (or all shapes allowed) delivers
    the 5-1 leaders exactly as before — no phantom filtering;
  - the audit's Tier-2 re-solve (_solve_challenger) returns only
    allowed splits under restricted weights, including for a
    non-adjacent allowed set ({5-1} only), and is unconstrained
    without them.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from dfs_lab.audit import _solve_challenger
from dfs_lab.showdown import (
    select_showdown_portfolio,
    showdown_candidate_row,
    showdown_world_optimal_lineups,
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


def _shape_of_ids(by_id, ids):
    per_team = {}
    for pid in ids:
        t = str(by_id[str(pid)]["Team"])
        per_team[t] = per_team.get(t, 0) + 1
    cs = sorted(per_team.values(), reverse=True)
    return f"{cs[0]}-{cs[1]}"


def _shape_of_row(by_id, row):
    ids = [str(row[c]) for c in
           ["CPT_ID"] + [f"FLEX{j}_ID" for j in range(1, 6)]]
    return _shape_of_ids(by_id, ids)


WEIGHTS_NO_51 = {"3-3": 1, "4-2": 1, "5-1": 0}
WEIGHTS_ALL = {"3-3": 1, "4-2": 1, "5-1": 1}


def test_world_optimals_respect_shape_weights():
    pool = _pool()
    by_id = {str(r["ID"]): r for _, r in pool.iterrows()}
    rng = np.random.default_rng(3)
    W = rng.gamma(shape=4.0, scale=1.0,
                  size=(400, len(pool))) * (pool["My Proj"].to_numpy()
                                            / 4.0)
    # Skew the worlds hard toward DAL: unconstrained, the optimal
    # legal lineup of most worlds is a 5-1 DAL stack — exactly the
    # slate shape that produced the incident.
    dal = pool["Team"].to_numpy() == "DAL"
    W[:, dal] *= 3.0
    W[:, ~dal] *= 0.6
    kw = dict(n_sample=120, seed=5, salary_cap=50000, min_salary=0,
              strategy_map={})
    open_opts = showdown_world_optimal_lineups(W, pool, **kw)
    open_shapes = [_shape_of_ids(by_id, o["player_ids"])
                   for o in open_opts]
    _check("skewed worlds: unrestricted supply exists",
           len(open_opts) > 0)
    _check("skewed worlds: unrestricted supply includes 5-1",
           "5-1" in open_shapes)
    all_opts = showdown_world_optimal_lineups(
        W, pool, construction_weights=WEIGHTS_ALL, **kw)
    _check("all shapes allowed = byte-identical supply",
           [(o["cpt_id"], o["player_ids"], o["wins"])
            for o in all_opts]
           == [(o["cpt_id"], o["player_ids"], o["wins"])
               for o in open_opts])
    shut_opts = showdown_world_optimal_lineups(
        W, pool, construction_weights=WEIGHTS_NO_51, **kw)
    shut_shapes = [_shape_of_ids(by_id, o["player_ids"])
                   for o in shut_opts]
    _check("5-1 unchecked: world supply is non-zero",
           len(shut_opts) > 0)
    _check("5-1 unchecked: ZERO 5-1 world candidates",
           all(s in ("3-3", "4-2") for s in shut_shapes))


def _mixed_candidates(pool):
    rows = [
        # Two 5-1 DAL stacks (the world-score leaders below).
        _row(pool, "103", ("100", "101", "104", "106", "115")),
        _row(pool, "100", ("103", "104", "107", "108", "115")),
        # One 4-2 and one 3-3 (the only shape-legal supply).
        _row(pool, "103", ("100", "106", "105", "111", "114")),
        _row(pool, "109", ("110", "113", "101", "102", "108")),
    ]
    return pd.DataFrame(rows), [100.0, 90.0, 50.0, 40.0]


def test_selection_shape_hard_filter():
    pool = _pool()
    by_id = {str(r["ID"]): r for _, r in pool.iterrows()}
    cands, scores = _mixed_candidates(pool)
    res, _ = select_showdown_portfolio(
        cands, pool, {}, 2, order_scores=scores,
        construction_weights=WEIGHTS_NO_51)
    shapes = [_shape_of_row(by_id, r) for _, r in res.iterrows()]
    _check("selection delivers the full count from legal supply",
           len(res) == 2)
    _check("selection delivers ZERO disallowed shapes despite "
           "5-1-favoring world scores",
           all(s in ("3-3", "4-2") for s in shapes))
    res3, notes3 = select_showdown_portfolio(
        cands, pool, {}, 3, order_scores=scores,
        construction_weights=WEIGHTS_NO_51)
    _check("shortfall: only the legal supply is delivered",
           len(res3) == 2)
    _check("shortfall: the generic shortfall note fires",
           any(n.startswith("Shortfall:") for n in notes3))
    _check("shortfall: the Build Shape cause is named",
           any(n.startswith("Build Shape:") and "5-1" in n
               and "built 2 of 3" in n for n in notes3))


def test_selection_weights_none_unchanged():
    pool = _pool()
    by_id = {str(r["ID"]): r for _, r in pool.iterrows()}
    cands, scores = _mixed_candidates(pool)
    for label, weights in (("weights None", None),
                           ("all shapes allowed", WEIGHTS_ALL)):
        res, _ = select_showdown_portfolio(
            cands, pool, {}, 2, order_scores=scores,
            construction_weights=weights)
        shapes = [_shape_of_row(by_id, r) for _, r in res.iterrows()]
        _check(f"selection unchanged with {label}: the 5-1 world "
               f"leaders are delivered", shapes == ["5-1", "5-1"])


def test_audit_challenger_respects_shape_weights():
    pool = _pool()
    by_id = {str(r["ID"]): r for _, r in pool.iterrows()}
    # Flat worlds where every DAL player crushes every TB player:
    # the best-legal construction is a 5-1 DAL stack unless the
    # user's Build Shape weights forbid it.
    W = np.tile(np.where(pool["Team"].to_numpy() == "DAL", 30.0, 2.0),
                (8, 1))
    profile = {"cpt_id": "103",
               "ids": ["103", "100", "101", "104", "106", "115"]}
    base = {"min_salary": 0, "max_salary": 60000,
            "max_k": 2, "max_dst": 2}

    def _solve(settings):
        return _solve_challenger(
            pool, W, np.ones(8, bool), {}, profile, settings=settings,
            exempt=set(), no_role=set(), by_id=by_id, core_ids=[],
            captain_free=True, time_left=10.0)

    chal = _solve(dict(base))
    _check("audit challenger unconstrained without weights",
           chal is not None
           and _shape_of_ids(by_id, chal[1]) == "5-1")
    chal = _solve(dict(base, construction_weights=WEIGHTS_NO_51))
    _check("audit challenger obeys the unchecked 5-1 (best legal "
           "is 4-2)", chal is not None
           and _shape_of_ids(by_id, chal[1]) == "4-2")
    chal = _solve(dict(base, construction_weights={
        "3-3": 0, "4-2": 0, "5-1": 1}))
    _check("audit challenger: non-adjacent allowed set ({5-1} "
           "only) is honored exactly",
           chal is not None
           and _shape_of_ids(by_id, chal[1]) == "5-1")


if __name__ == "__main__":
    test_world_optimals_respect_shape_weights()
    test_selection_shape_hard_filter()
    test_selection_weights_none_unchanged()
    test_audit_challenger_respects_shape_weights()
    print("ALL PASS")
