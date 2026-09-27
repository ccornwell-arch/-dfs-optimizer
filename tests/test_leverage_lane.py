"""Leverage Lane feature tests (Ship It Nation strategy package, Sep 2026).

Covers, on synthetic data only:
  1. Mobile-QB tag: rushing-share threshold, position gate, missing-signal fallback.
  2. _dk_rushing_points_from_stats mirrors the rushing parts of the DK formula.
  3. Relaxed stack minimum: a locked mobile QB with no eligible teammates can
     still build (naked rushing-QB build); a pocket QB in the same spot is
     infeasible and never appears.
  4. Coherence: naked mobile-QB builds are not penalized and are labeled;
     naked pocket-QB builds still take the -18 flag.
  5. Chalk-cutting RB+DST: threshold, pair detection, leverage-path kicker
     (stacks on, never duplicates, the flat correlation reward).
  6. Leverage Lane pick: low-owned edge play wins, QBs/DSTs excluded,
     deterministic, no positive edge -> None.
  7. If-chalk-fails: conditional beneficiaries from synthetic sim worlds.

Run from repo root:  python3 tests/test_leverage_lane.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import io
import numpy as np
import pandas as pd

from dfs_lab.data import prepare_player_pool, _dk_rushing_points_from_stats
from dfs_lab.leverage import (
    compute_mobile_qb_tags, rb_chalk_threshold, chalk_cut_pairs,
    leverage_lane_pick, chalk_bust_beneficiaries,
    MOBILE_QB_RUSH_SHARE, CHALK_CUT_KICKER,
)
from dfs_lab.classic import (
    generate_lineups, classic_lineup_coherence, lineup_details, add_ratings,
)

import dfs_lab.data as _data_mod
# Synthetic players ("AAron Starter", ...) have no nflverse roster record. Stub
# live availability as unreachable so the guard fails open (its documented
# path) instead of excluding every synthetic player as teamless.
_EMPTY_DF = pd.DataFrame()
_data_mod._load_live_nfl_availability = lambda season=2026: (_EMPTY_DF, _EMPTY_DF)

CHECKS = []
def _check(name, cond, detail=""):
    CHECKS.append(name)
    if not cond:
        raise AssertionError(f"FAIL {name} {detail}")
    print(f"  ok: {name}")

SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]

# (ID, Name, Position, Roster Position, Team, Salary, Proj, Own)
PLAYERS = [
    (1001, "AAron Starter", "QB", "QB", "AAA", 7000, 20.0, 10.0),
    (1002, "AAron Backup", "QB", "QB", "AAA", 5000, 12.0, 2.0),
    (1003, "AAron Runner", "RB", "RB", "AAA", 6500, 16.0, 12.0),
    (1004, "AAron Change", "RB", "RB", "AAA", 4500, 9.0, 5.0),
    (1005, "AAron Wide1", "WR", "WR", "AAA", 6000, 14.0, 10.0),
    (1006, "AAron Wide2", "WR", "WR", "AAA", 5200, 11.0, 7.0),
    (1007, "AAron Wide3", "WR", "WR", "AAA", 4200, 8.0, 4.0),
    (1008, "AAron Tight", "TE", "TE", "AAA", 4500, 10.0, 8.0),
    (1009, "AAron Defense", "DST", "DST", "AAA", 3000, 7.0, 6.0),
    (2001, "Baker Healthy", "QB", "QB", "BBB", 5200, 14.0, 6.0),
    (2002, "Baker Runner", "RB", "RB", "BBB", 6200, 15.0, 11.0),
    (2003, "Baker Wide1", "WR", "WR", "BBB", 5800, 13.0, 9.0),
    (2004, "Baker Wide2", "WR", "WR", "BBB", 5000, 10.0, 6.0),
    (2005, "Baker Tight", "TE", "TE", "BBB", 4300, 9.0, 7.0),
    (2006, "Baker Defense", "DST", "DST", "BBB", 2800, 6.0, 5.0),
    (3001, "Caleb Thrower", "QB", "QB", "CCC", 6800, 18.0, 9.0),
    (3002, "Caleb Star", "RB", "RB", "CCC", 8000, 22.0, 25.0),
    (3003, "Caleb Runner", "RB", "RB", "CCC", 5500, 12.0, 8.0),
    (3004, "Caleb Wide1", "WR", "WR", "CCC", 6300, 15.0, 11.0),
    (3005, "Caleb Wide2", "WR", "WR", "CCC", 4800, 10.0, 6.0),
    (3006, "Caleb Tight", "TE", "TE", "CCC", 4000, 8.0, 5.0),
    (3007, "Caleb Defense", "DST", "DST", "CCC", 3200, 8.0, 7.0),
]
GAMES = {"AAA": "AAA@BBB 09/28/2026 01:00PM ET", "BBB": "AAA@BBB 09/28/2026 01:00PM ET",
         "CCC": "CCC@DDD 09/28/2026 04:05PM ET", "DDD": "CCC@DDD 09/28/2026 04:05PM ET"}


def _dk_file():
    lines = ["Position,Name + ID,Salary,Name,ID,Roster Position,Game Info,TeamAbbrev,AvgPointsPerGame"]
    for pid, name, pos, rp, team, sal, proj, own in PLAYERS:
        rp_out = {"RB": "RB/FLEX", "WR": "WR/FLEX", "TE": "TE/FLEX"}.get(rp, rp)
        lines.append(f"{pos},{name} ({pid}),{sal},{name},{pid},{rp_out},{GAMES[team]},{team},{proj}")
    return io.BytesIO("\n".join(lines).encode("utf-8"))


def _ss_file():
    lines = ["Name,My Proj,My Own"]
    for pid, name, pos, rp, team, sal, proj, own in PLAYERS:
        lines.append(f"{name},{proj},{own}")
    return io.BytesIO("\n".join(lines).encode("utf-8"))


def _pool(mobile_names=()):
    """SaberSim-path pool with injected rushing-share tags (live availability
    is irrelevant: all names are fictional, so no live guard can verify them —
    the tag path is what is under test)."""
    df = prepare_player_pool(_dk_file(), _ss_file())
    df["Rush Share"] = 0.05
    df.loc[df["Name"].isin(list(mobile_names)), "Rush Share"] = 0.35
    df["Mobile QB"] = compute_mobile_qb_tags(df)
    return df


def test_mobile_tag_logic():
    df = _pool(mobile_names=["AAron Starter"])
    tags = df.set_index("Name")["Mobile QB"]
    _check("mobile QB tagged", bool(tags["AAron Starter"]))
    _check("pocket QB not tagged", not bool(tags["Baker Healthy"]))
    _check("pocket QB not tagged 2", not bool(tags["Caleb Thrower"]))
    # A non-QB with a huge rushing share is still not a mobile *QB*.
    df.loc[df["Name"] == "AAron Runner", "Rush Share"] = 0.60
    df["Mobile QB"] = compute_mobile_qb_tags(df)
    _check("high-share non-QB not tagged",
            not bool(df.set_index("Name").loc["AAron Runner", "Mobile QB"]))
    # Threshold boundary.
    df2 = _pool()
    df2.loc[df2["Name"] == "AAron Starter", "Rush Share"] = MOBILE_QB_RUSH_SHARE - 0.001
    _check("below threshold not tagged",
            not bool(compute_mobile_qb_tags(df2).loc[df2["Name"] == "AAron Starter"].iloc[0]))
    # Missing signal -> no tags, never fabricated.
    df3 = _pool().drop(columns=["Rush Share"])
    _check("missing rush share tags nobody", not compute_mobile_qb_tags(df3).any())


def test_rushing_points_formula():
    st = pd.DataFrame([{
        "passing_yards": 0, "passing_tds": 0, "interceptions": 0,
        "rushing_yards": 100, "rushing_tds": 1,
        "receptions": 0, "receiving_yards": 0, "receiving_tds": 0,
        "rushing_fumbles_lost": 0, "receiving_fumbles_lost": 0, "sack_fumbles_lost": 0,
    }, {
        "passing_yards": 300, "passing_tds": 3, "interceptions": 1,
        "rushing_yards": 0, "rushing_tds": 0,
        "receptions": 0, "receiving_yards": 0, "receiving_tds": 0,
        "rushing_fumbles_lost": 0, "receiving_fumbles_lost": 0, "sack_fumbles_lost": 0,
    }])
    pts = _dk_rushing_points_from_stats(st)
    _check("100-yd 1-TD rush game = 19.0", abs(float(pts.iloc[0]) - 19.0) < 1e-9, f"got {pts.iloc[0]}")
    _check("pass-only game = 0 rushing pts", abs(float(pts.iloc[1])) < 1e-9)


def _naked_setup(mobile_names):
    df = _pool(mobile_names=mobile_names)
    # Lock AAron Starter; deactivate all his team's pass catchers so the only
    # legal stack for him is a naked one.
    df.loc[df["Name"].isin(["AAron Wide1", "AAron Wide2", "AAron Wide3", "AAron Tight"]), "ActiveForBuild"] = False
    sm = {"1001": {"Lock": True}}
    return df, sm


def _qb_lineups(res, name):
    out = []
    for _, r in res.iterrows():
        if r["QB"] == name:
            out.append(r)
    return out


def test_naked_mobile_build_allowed():
    df, sm = _naked_setup(["AAron Starter"])
    res = generate_lineups(df, 500, "GPP / top-heavy", 6, 200, 40000, 1,
                           "Optional", [], sm, {}, False, False, 7)
    got = _qb_lineups(res, "AAron Starter")
    _check("mobile QB builds despite no teammates", len(got) >= 1, f"found {len(got)}")
    naked = [r for r in got if int(r["QB Stack"]) == 0]
    _check("naked mobile-QB lineup exists", len(naked) >= 1)
    r = naked[0]
    _check("QB Mobile flag set", int(r["QB Mobile"]) == 1)
    _check("stack summary notes naked rushing-QB build",
            "naked rushing-QB build" in str(r["Stack Summary"]))
    _check("no naked-QB warning for mobile build",
            "Naked QB" not in str(r["Coherence Flags"]))


def test_pocket_qb_still_requires_stack():
    # Control: identical setup, but the QB is a pocket passer. The naked build
    # is infeasible, so he must not appear at all.
    df, sm = _naked_setup([])
    res = generate_lineups(df, 500, "GPP / top-heavy", 6, 200, 40000, 1,
                           "Optional", [], sm, {}, False, False, 7)
    got = _qb_lineups(res, "AAron Starter")
    _check("pocket QB never appears without a legal stack", len(got) == 0, f"found {len(got)}")


def _lineup_df():
    # Minimal 9-man pool for lineup_details / coherence unit tests.
    rows = [
        # Name, Pos, Team, Opp, Salary, Proj, Own, Rush
        ("Mob QB", "QB", "AAA", "BBB", 7000, 20.0, 10.0, 0.35),
        ("Pocket QB", "QB", "CCC", "DDD", 6800, 18.0, 9.0, 0.05),
        ("Chalk RB", "RB", "AAA", "BBB", 8000, 22.0, 25.0, 0.02),
        ("Mid RB", "RB", "BBB", "AAA", 6000, 13.0, 8.0, 0.02),
        ("Low RB", "RB", "CCC", "DDD", 4500, 9.0, 4.0, 0.02),
        ("WR A", "WR", "BBB", "AAA", 6000, 14.0, 10.0, 0.0),
        ("WR B", "WR", "CCC", "DDD", 5900, 13.0, 9.0, 0.0),
        ("WR C", "WR", "DDD", "CCC", 5100, 11.0, 6.0, 0.0),
        ("TE A", "TE", "DDD", "CCC", 4400, 9.0, 6.0, 0.0),
        ("WR D", "WR", "DDD", "CCC", 4800, 10.0, 5.0, 0.0),
        ("Flex RB", "RB", "DDD", "CCC", 5500, 12.0, 8.0, 0.02),
        ("AAA DST", "DST", "AAA", "BBB", 3000, 7.0, 6.0, 0.0),
        ("CCC DST", "DST", "CCC", "DDD", 3200, 8.0, 7.0, 0.0),
    ]
    df = pd.DataFrame(rows, columns=["Name", "Position", "Team", "Opponent", "Salary", "My Proj", "My Own", "Rush Share"])
    df["ID"] = [9000 + i for i in range(len(df))]
    _game_of = {"AAA": "AAA@BBB", "BBB": "AAA@BBB", "CCC": "CCC@DDD", "DDD": "CCC@DDD"}
    df["Matchup"] = df["Team"].map(_game_of)
    df["ActiveForBuild"] = True
    df["is_QB"] = df["Position"].eq("QB")
    df["is_RB"] = df["Position"].eq("RB")
    df["is_WR"] = df["Position"].eq("WR")
    df["is_TE"] = df["Position"].eq("TE")
    df["is_DST"] = df["Position"].eq("DST")
    df["is_FLEX"] = df["Position"].isin(["RB", "WR", "TE"])
    df["Mobile QB"] = compute_mobile_qb_tags(df)
    return df


def test_chalk_cut_detection_and_kicker():
    df = _lineup_df()
    thr = rb_chalk_threshold(df)
    _check("chalk threshold = max(15, q75)", abs(thr - 15.0) < 1e-9, f"got {thr}")
    # Lineup 1: chalk RB (AAA, 25% own) + AAA DST -> 1 chalk-cut pair.
    c1 = [("QB", 0), ("RB1", 2), ("RB2", 3), ("WR1", 5), ("WR2", 6), ("WR3", 7),
          ("TE", 8), ("FLEX", 10), ("DST", 11)]
    pairs = chalk_cut_pairs(df, c1)
    _check("chalk RB+DST pair detected", pairs == [("Chalk RB", "AAA DST")], f"got {pairs}")
    d1 = lineup_details(df, c1, {}, [])
    _check("Chalk Cut = 1 for chalk pair", int(d1["Chalk Cut"]) == 1)
    _check("RB+DST counted", int(d1["RB+DST"]) == 1)
    # Lineup 2: non-chalk RB (BBB, 8% own) + AAA DST -> no pair, no kicker.
    c2 = [("QB", 0), ("RB1", 3), ("RB2", 10), ("WR1", 5), ("WR2", 6), ("WR3", 7),
          ("TE", 8), ("FLEX", 4), ("DST", 11)]
    _check("non-chalk RB+DST -> no pair", chalk_cut_pairs(df, c2) == [])
    d2 = lineup_details(df, c2, {}, [])
    _check("Chalk Cut = 0 for non-chalk pair", int(d2["Chalk Cut"]) == 0)
    # Same composition, chalk flag flipped off: correlation must be identical,
    # proving the kicker is NOT double-counted in the correlation path.
    df_low = df.copy()
    df_low.loc[df_low["Name"] == "Chalk RB", "My Own"] = 8.0
    d3 = lineup_details(df_low, c1, {}, [])
    _check("below-threshold RB loses the pair", int(d3["Chalk Cut"]) == 0)
    _check("kicker not duplicated in correlation_raw",
            abs(float(d3["Correlation Raw"]) - float(d1["Correlation Raw"])) < 1e-9)

    # add_ratings: identical lineups except Chalk Cut -> kicker wins via leverage.
    base = {"Projection": 150.0, "Correlation Raw": 5.0, "User Fit Raw": 0.0,
            "Avg Own": 10.0, "Salary Left": 0, "Coherence Score": 90.0}
    rows = []
    for i in range(4):
        r = dict(base)
        r["Chalk Cut"] = 1 if i < 2 else 0
        rows.append(r)
    out = add_ratings(pd.DataFrame(rows), 1.0)
    chalk_scores = out.loc[out["Chalk Cut"] == 1, "Rating Score"]
    plain_scores = out.loc[out["Chalk Cut"] == 0, "Rating Score"]
    _check("chalk-cut kicker lifts rating via leverage",
            float(chalk_scores.min()) > float(plain_scores.max()))
    # Backwards compatibility: frames without the column still rate.
    out2 = add_ratings(pd.DataFrame([{k: v for k, v in base.items()}] * 3), 1.0)
    _check("add_ratings without Chalk Cut column", len(out2) == 3)


def test_coherence_naked_waiver():
    df = _lineup_df()
    # Naked mobile-QB lineup: Mob QB + no AAA pass catchers. AAA DST faces two
    # BBB players (-22) so the naked-QB component is isolated by comparison.
    c = [("QB", 0), ("RB1", 3), ("RB2", 10), ("WR1", 5), ("WR2", 7), ("WR3", 9),
         ("TE", 8), ("FLEX", 4), ("DST", 11)]
    coh = classic_lineup_coherence(df, c)
    _check("naked mobile-QB lineup accepted", bool(coh["Accept"]))
    _check("story labels naked rushing-QB build",
            "naked rushing-QB build" in str(coh["Lineup Story"]))
    _check("no naked-QB warning for mobile build",
            "Naked QB" not in str(coh["Coherence Flags"]))
    # Same shape with a pocket QB and no CCC pass catchers: the -18 naked
    # flag still applies. Score differs from the mobile build by exactly 18
    # (rb_dst equalized at 0 in both so the naked component is isolated).
    c2 = [("QB", 1), ("RB1", 4), ("RB2", 3), ("WR1", 5), ("WR2", 7), ("WR3", 9),
          ("TE", 8), ("FLEX", 10), ("DST", 11)]
    coh2 = classic_lineup_coherence(df, c2)
    _check("pocket naked QB still flagged",
            "Naked QB" in str(coh2["Coherence Flags"]))
    _check("pocket naked QB takes exactly -18 vs mobile",
            abs((float(coh["Coherence Score"]) - float(coh2["Coherence Score"])) - 18.0) < 1e-9,
            f"mobile={coh['Coherence Score']} pocket={coh2['Coherence Score']}")


def test_leverage_lane_pick():
    rows = [
        ("Chalk Stud", "RB", "AAA", 5000, 20.0, 25.0),
        ("Quiet Edge", "WR", "BBB", 4000, 15.0, 3.0),
        ("Trap", "TE", "CCC", 6000, 9.0, 2.0),
        ("Star QB", "QB", "AAA", 6000, 25.0, 5.0),
        ("Any DST", "DST", "BBB", 3000, 9.0, 4.0),
    ]
    df = pd.DataFrame(rows, columns=["Name", "Position", "Team", "Salary", "My Proj", "My Own"])
    df["ActiveForBuild"] = True
    df["Own Estimated"] = True
    p1 = leverage_lane_pick(df)
    # Quiet Edge: edge 15-6.2=8.8, score 8.8/3=2.93 beats Chalk Stud 12.25/25=0.49.
    _check("low-owned edge play wins", p1["name"] == "Quiet Edge", f"got {p1['name']}")
    _check("QB excluded from leverage lane", p1["position"] != "QB")
    _check("ownership labeled estimated", p1["own_estimated"] is True)
    _check("deterministic", leverage_lane_pick(df)["name"] == p1["name"])
    # No positive edge anywhere -> None.
    df2 = df.copy()
    df2["My Proj"] = 1.0
    _check("no edge -> None", leverage_lane_pick(df2) is None)
    # Inactive players are not candidates.
    df3 = df.copy()
    df3.loc[df3["Name"] == "Quiet Edge", "ActiveForBuild"] = False
    _check("inactive excluded", leverage_lane_pick(df3)["name"] == "Chalk Stud")


def test_chalk_bust_beneficiaries():
    rng = np.random.default_rng(11)
    n_worlds = 2000
    chalk = np.where(np.arange(n_worlds) < 500, 5.0, 20.0) + rng.normal(0, 1, n_worlds)
    bene = np.where(np.arange(n_worlds) < 500, 30.0, 10.0) + rng.normal(0, 1, n_worlds)
    fill1 = 12.0 + rng.normal(0, 2, n_worlds)
    fill2 = 11.0 + rng.normal(0, 2, n_worlds)
    worlds = pd.DataFrame({"c0": chalk, "c1": bene, "c2": fill1, "c3": fill2})
    df = pd.DataFrame([
        ("Chalk RB", "RB", "AAA", 25.0),
        ("Slate Breaker", "WR", "BBB", 5.0),
        ("Filler One", "RB", "CCC", 4.0),
        ("Filler Two", "TE", "DDD", 3.0),
    ], columns=["Name", "Position", "Team", "My Own"])
    df["Salary"] = 5000
    df["My Proj"] = 12.0
    df["ActiveForBuild"] = True
    rows = chalk_bust_beneficiaries(worlds, df, top_chalk=2, per_chalk=2)
    _check("chalk entries returned", len(rows) == 2, f"got {len(rows)}")
    top = rows[0]
    _check("most-owned is first chalk", top["chalk"] == "Chalk RB")
    bens = [b["name"] for b in top["beneficiaries"]]
    _check("conditional beneficiary found", "Slate Breaker" in bens, f"got {bens}")
    sb = [b for b in top["beneficiaries"] if b["name"] == "Slate Breaker"][0]
    _check("cond mean reflects bust worlds", abs(sb["cond_mean"] - 30.0) < 2.0, f"got {sb['cond_mean']}")
    _check("gain positive", sb["gain"] > 5.0, f"got {sb['gain']}")
    # Column/name misalignment -> safe empty, never raises.
    bad = pd.DataFrame(np.zeros((10, 2)), columns=["x", "y"])
    _check("misaligned worlds -> []", chalk_bust_beneficiaries(bad, df) == [])
    _check("empty worlds -> []", chalk_bust_beneficiaries(pd.DataFrame(), df) == [])
    # Deterministic.
    again = chalk_bust_beneficiaries(worlds, df, top_chalk=2, per_chalk=2)
    _check("deterministic", [b["name"] for b in again[0]["beneficiaries"]] == bens)


if __name__ == "__main__":
    test_mobile_tag_logic()
    test_rushing_points_formula()
    test_naked_mobile_build_allowed()
    test_pocket_qb_still_requires_stack()
    test_chalk_cut_detection_and_kicker()
    test_coherence_naked_waiver()
    test_leverage_lane_pick()
    test_chalk_bust_beneficiaries()
    print(f"ALL LEVERAGE-LANE TESTS PASSED ({len(CHECKS)} checks)")