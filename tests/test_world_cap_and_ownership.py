"""Tests for the Oct 7, 2026 build: showdown ownership feed + game-world cap.

1. Ownership feed: prepare_showdown_pool() left My Own / CPT Own at 0.0 when no
   SaberSim file was uploaded, which silently disabled all leverage/duplication
   math (grades showed "Unavailable"). The estimator existed but was never
   called on the showdown path. Now DK-only builds get a directional estimate
   (600 = 6 showdown slots x 100%), with contest-aware dispersion.
2. World cap: no single game world may exceed its share of the portfolio.
   Smart default keys off lineups built (not contest max): <=5 -> 100%,
   6-20 -> 60%, 21-50 -> 40%, 50+ -> 30%.

Run from repo root:  python3 tests/test_world_cap_and_ownership.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from dfs_lab.data import estimate_ownership, _estimate_showdown_ownership
import dfs_lab.showdown as sd


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def _own_df():
    # Small showdown-style pool: 2 QBs, 3 RB, 4 WR, 2 TE, 2 K, 2 DST, 1 inactive.
    # (DSTs included so the classic position-group path is fully exercised.)
    return pd.DataFrame({
        "Name": ["Dak Prescott", "Jalon Daniels", "Javonte Williams", "Bucky Irving",
                 "Sean Tucker", "CeeDee Lamb", "George Pickens", "Chris Godwin",
                 "Emeka Egbuka", "Jake Ferguson", "Cade Otton",
                 "Brandon Aubrey", "Chase McLaughlin", "DAL DST", "TB DST",
                 "Tyler Goodson"],
        "Position": ["QB", "QB", "RB", "RB", "RB", "WR", "WR", "WR",
                     "WR", "TE", "TE", "K", "K", "DST", "DST", "RB"],
        "Team": ["DAL", "TB", "DAL", "TB", "TB", "DAL", "DAL", "TB",
                 "TB", "DAL", "TB", "DAL", "TB", "DAL", "TB", "DAL"],
        "FlexSalary": [10400, 6000, 8200, 7800, 4200, 9800, 8800, 7200,
                       6800, 5200, 4800, 5000, 4600, 4200, 4000, 3000],
        "My Proj": [19.5, 8.0, 13.2, 12.8, 6.1, 18.9, 14.4, 11.7,
                    10.9, 8.8, 7.9, 8.0, 7.5, 7.0, 6.5, 0.0],
    })


def test_showdown_ownership_sums_to_600():
    own = estimate_ownership(_own_df(), salary_col="FlexSalary", total=600.0, by_position=False)
    _check("showdown estimate sums to ~600", abs(float(own.sum()) - 600.0) < 5.0)
    _check("all non-negative", bool((own >= 0).all()))


def test_showdown_ownership_kickers_nonzero_and_inactive_zero():
    df = _own_df()
    own = estimate_ownership(df, salary_col="FlexSalary", total=600.0, by_position=False)
    k = own[df["Position"].eq("K")]
    _check("kickers get nonzero ownership", bool((k > 0).all()))
    _check("inactive (proj 0) gets zero", float(own.iloc[-1]) == 0.0)
    _check("top projector is Dak or Lamb", df.loc[own.idxmax(), "Name"] in ("Dak Prescott", "CeeDee Lamb"))


def test_classic_ownership_still_900():
    df = _own_df()
    own = estimate_ownership(df, salary_col="FlexSalary")  # classic defaults
    _check("classic default still sums to ~900", abs(float(own.sum()) - 900.0) < 5.0)


def test_dispersion_flattens_distribution():
    df = _own_df()
    sharp = estimate_ownership(df, salary_col="FlexSalary", total=600.0, by_position=False, dispersion=0.0)
    flat = estimate_ownership(df, salary_col="FlexSalary", total=600.0, by_position=False, dispersion=0.30)
    _check("dispersion lowers the max (flatter)", float(flat.max()) < float(sharp.max()))
    _check("dispersion preserves the total", abs(float(flat.sum()) - 600.0) < 5.0)


def test_fill_estimates_when_sabersim_missing():
    df = _own_df()
    df["My Own"] = 0.0
    df["CPT Own"] = 0.0
    was_estimated = _estimate_showdown_ownership(df, entry_format="20-Max")
    _check("returns True when estimated", was_estimated is True)
    _check("My Own populated", bool((df["My Own"] > 0).any()))
    _check("CPT Own populated (~18% of flex)", bool((df["CPT Own"] > 0).any()))
    _check("Own Estimated flag set", bool(df["Own Estimated"].iloc[0]) is True)
    # CPT own should be well below flex own on average (captain is one slot).
    _check("CPT own << flex own on average", float(df["CPT Own"].mean()) < float(df["My Own"].mean()))


def test_fill_keeps_sabersim_ownership():
    df = _own_df()
    n = len(df)
    df["My Own"] = [30.0, 5.0, 22.0, 20.0, 8.0, 35.0, 25.0, 18.0, 15.0, 10.0, 9.0, 12.0, 11.0, 9.5, 8.5, 0.0][:n]
    df["CPT Own"] = [20.0, 1.0, 8.0, 7.0, 2.0, 22.0, 12.0, 6.0, 5.0, 3.0, 2.5, 0.5, 0.5, 0.4, 0.4, 0.0][:n]
    was_estimated = _estimate_showdown_ownership(df, entry_format="20-Max")
    _check("returns False when SaberSim present", was_estimated is False)
    _check("SaberSim values untouched", float(df["My Own"].iloc[0]) == 30.0)
    _check("Own Estimated flag clear", bool(df["Own Estimated"].iloc[0]) is False)


def test_default_world_max_share():
    _check("5 lineups -> 100%", sd.default_world_max_share(5) == 1.0)
    _check("1 lineup -> 100%", sd.default_world_max_share(1) == 1.0)
    _check("6-20 -> 60%", sd.default_world_max_share(20) == 0.60)
    _check("21-50 -> 40%", sd.default_world_max_share(50) == 0.40)
    _check("50+ -> 30%", sd.default_world_max_share(100) == 0.30)


def test_world_share_capped():
    counts = {"A": 8}
    _check("under cap -> False", sd.world_share_capped(counts, "A", 20, 0.60) is False)
    counts = {"A": 12}
    _check("at cap -> True", sd.world_share_capped(counts, "A", 20, 0.60) is True)
    _check("cap 1.0 never caps", sd.world_share_capped({"A": 99}, "A", 20, 1.0) is False)
    _check("cap None never caps", sd.world_share_capped({"A": 99}, "A", 20, None) is False)
    _check("missing world counts as zero", sd.world_share_capped({}, "B", 20, 0.60) is False)


def _build_pool():
    # Two-team synthetic slate, enough depth for a 12-lineup portfolio.
    players = []
    specs = [
        ("DAL", "QB", [("Dak Prescott", 10400, 19.5)]),
        ("DAL", "RB", [("Javonte Williams", 8200, 13.2), ("Hunter Luepke", 3600, 5.2)]),
        ("DAL", "WR", [("CeeDee Lamb", 9800, 18.9), ("George Pickens", 8800, 14.4),
                        ("Ryan Flournoy", 5200, 8.1), ("Ted Hurst", 3000, 4.5)]),
        ("DAL", "TE", [("Jake Ferguson", 5200, 8.8), ("Brevyn Spann-Ford", 2800, 3.9)]),
        ("DAL", "K", [("Brandon Aubrey", 5000, 8.0)]),
        ("TB", "QB", [("Jalon Daniels", 6000, 8.0)]),
        ("TB", "RB", [("Bucky Irving", 7800, 12.8), ("Sean Tucker", 4200, 6.1),
                       ("Kenny Gainwell", 3800, 5.5)]),
        ("TB", "WR", [("Chris Godwin", 7200, 11.7), ("Emeka Egbuka", 6800, 10.9),
                       ("Tez Johnson", 3400, 5.0)]),
        ("TB", "TE", [("Cade Otton", 4800, 7.9), ("Payne Durham", 2600, 3.2)]),
        ("TB", "K", [("Chase McLaughlin", 4600, 7.5)]),
    ]
    pid = 0
    for team, pos, plist in specs:
        for name, sal, proj in plist:
            pid += 1
            players.append((f"p{pid}", name, pos, team, sal, proj))
    df = pd.DataFrame(players, columns=["ID", "Name", "Position", "Team", "FlexSalary", "My Proj"])
    pos = df["Position"].str.upper()
    df["is_QB"] = pos.eq("QB")
    df["is_RB"] = pos.eq("RB")
    df["is_WR"] = pos.eq("WR")
    df["is_TE"] = pos.eq("TE")
    df["is_DST"] = False
    df["is_K"] = pos.eq("K")
    df["is_passcatcher"] = df["is_WR"] | df["is_TE"]
    df["My Own"] = 20.0
    df["CPT Own"] = 4.0
    df["DFS Lab Proj"] = df["My Proj"]
    df["CaptainSalary"] = (df["FlexSalary"] * 1.5).round(-2).astype(int)
    df["Opponent"] = df["Team"].map({"DAL": "TB", "TB": "DAL"})
    df["ActiveForBuild"] = True
    df["History Games"] = 10
    df["Current Games"] = 4
    df["AvgPointsPerGame"] = df["My Proj"]
    df["Game Info"] = "DAL@TB"
    df["CPT_NameID"] = df["Name"] + " (" + df["ID"] + ")"
    df["FLEX_NameID"] = df["Name"] + " (" + df["ID"] + ")"
    return df


def test_world_cap_enforced_end_to_end():
    df = _build_pool()
    sd.configure_game_worlds(df)
    n_worlds = len(sd.GAME_WORLDS)
    _check("worlds configured", n_worlds >= 8)
    strategy_map = {pid: {"CPT Eligible": True} for pid in df["ID"]}
    res = sd.generate_showdown_lineups(
        df, 20000, "GPP", 12, 200, 40000, 50000,
        {"3-3": 1.0, "4-2": 1.0, "5-1": 0.0}, "Neutral", "",
        strategy_map, "20-Max", 0, 0, 0, 2, 1, 0, 20261007,
        show_progress=False, world_max_share=0.25,
    )
    _check("built all 12 lineups", res is not None and len(res) == 12)
    wc = res["Game World"].value_counts()
    _check("no world exceeds 25% (3 of 12)", bool((wc <= 3).all()))
    _check("cap forced real spread (>=4 worlds)", int((wc > 0).sum()) >= 4)


def test_small_portfolio_allows_single_theory():
    df = _build_pool()
    sd.configure_game_worlds(df)
    strategy_map = {pid: {"CPT Eligible": True} for pid in df["ID"]}
    res = sd.generate_showdown_lineups(
        df, 20000, "GPP", 5, 120, 40000, 50000,
        {"3-3": 1.0, "4-2": 1.0, "5-1": 0.0}, "Neutral", "",
        strategy_map, "20-Max", 0, 0, 0, 2, 1, 0, 777,
        show_progress=False,  # smart default: 5 lineups -> cap 100%
    )
    _check("built all 5 lineups", res is not None and len(res) == 5)


if __name__ == "__main__":
    test_showdown_ownership_sums_to_600()
    test_showdown_ownership_kickers_nonzero_and_inactive_zero()
    test_classic_ownership_still_900()
    test_dispersion_flattens_distribution()
    test_fill_estimates_when_sabersim_missing()
    test_fill_keeps_sabersim_ownership()
    test_default_world_max_share()
    test_world_share_capped()
    test_world_cap_enforced_end_to_end()
    test_small_portfolio_allows_single_theory()
    print("ALL PASS")
