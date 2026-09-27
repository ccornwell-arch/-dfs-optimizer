"""Regression tests for the Results Command Center (build-step results UI).

Exercises the pure helpers in dfs_lab/ui/results.py against a synthetic
4-team slate (KC@MIA, BUF@DET), plus the bring-back worthiness gating invariant:
no bring-back may come from an offense below league-average scoring.

Run from repo root:  python3 tests/test_results_command_center.py
"""
import io
import sys

sys.path.insert(0, ".")

import pandas as pd

import dfs_lab.data as _data_mod
from dfs_lab.data import prepare_player_pool

# These suites use synthetic players ("KC Quarterback", ...) with no nflverse
# roster record. Stub live availability as unreachable so the guard takes its
# documented fail-open path instead of excluding every synthetic player as
# teamless. (Also covers test_lab_agent_llm and test_lineups_first, which
# import build_pipeline from this module.)
_EMPTY_DF = pd.DataFrame()
_data_mod._load_live_nfl_availability = lambda season=2026: (_EMPTY_DF, _EMPTY_DF)
from dfs_lab.classic import (
    generate_lineups, topdown_simulate_slate, lineup_sim_equity,
    bringback_worthy_teams,
)
from dfs_lab.ui.results import (
    parse_stack_summary, short_story, lineup_title, hero_stats, sort_lineups,
    world_rows, player_table_for_lineup, swap_suggestions, SORT_OPTIONS,
)

# (ID, Name, Position, Roster Position, Team, Salary, Proj, Own)
PLAYERS = [
    # KC (strong) @ MIA (weak) — 1pm game
    (101, "KC Quarterback", "QB", "QB", "KC", 7200, 24.0, 14.0),
    (102, "KC Runner1", "RB", "RB", "KC", 6800, 17.0, 13.0),
    (103, "KC Runner2", "RB", "RB", "KC", 4600, 10.0, 6.0),
    (104, "KC Wide1", "WR", "WR", "KC", 6400, 16.0, 12.0),
    (105, "KC Wide2", "WR", "WR", "KC", 5400, 13.0, 9.0),
    (106, "KC Wide3", "WR", "WR", "KC", 4300, 10.0, 5.0),
    (107, "KC Tight", "TE", "TE", "KC", 4700, 11.0, 8.0),
    (108, "KC Defense", "DST", "DST", "KC", 3100, 7.5, 7.0),
    (201, "MIA Quarterback", "QB", "QB", "MIA", 6400, 15.0, 8.0),
    (202, "MIA Runner1", "RB", "RB", "MIA", 5900, 12.0, 9.0),
    (203, "MIA Runner2", "RB", "RB", "MIA", 4200, 8.0, 4.0),
    (204, "MIA Wide1", "WR", "WR", "MIA", 5600, 11.0, 8.0),
    (205, "MIA Wide2", "WR", "WR", "MIA", 4800, 9.0, 5.0),
    (206, "MIA Wide3", "WR", "WR", "MIA", 4000, 7.0, 3.0),
    (207, "MIA Tight", "TE", "TE", "MIA", 4100, 8.0, 5.0),
    (208, "MIA Defense", "DST", "DST", "MIA", 2700, 6.0, 5.0),
    # BUF (strong) @ DET (strong) — 4pm game
    (301, "BUF Quarterback", "QB", "QB", "BUF", 7000, 23.0, 13.0),
    (302, "BUF Runner1", "RB", "RB", "BUF", 6600, 16.0, 12.0),
    (303, "BUF Runner2", "RB", "RB", "BUF", 4400, 9.0, 5.0),
    (304, "BUF Wide1", "WR", "WR", "BUF", 6200, 15.0, 11.0),
    (305, "BUF Wide2", "WR", "WR", "BUF", 5200, 12.0, 8.0),
    (306, "BUF Wide3", "WR", "WR", "BUF", 4100, 9.0, 4.0),
    (307, "BUF Tight", "TE", "TE", "BUF", 4500, 10.0, 7.0),
    (308, "BUF Defense", "DST", "DST", "BUF", 3000, 7.0, 6.0),
    (401, "DET Quarterback", "QB", "QB", "DET", 6800, 20.0, 11.0),
    (402, "DET Runner1", "RB", "RB", "DET", 6300, 15.0, 11.0),
    (403, "DET Runner2", "RB", "RB", "DET", 4300, 9.0, 5.0),
    (404, "DET Wide1", "WR", "WR", "DET", 6000, 14.0, 10.0),
    (405, "DET Wide2", "WR", "WR", "DET", 5000, 11.0, 7.0),
    (406, "DET Wide3", "WR", "WR", "DET", 4000, 8.0, 4.0),
    (407, "DET Tight", "TE", "TE", "DET", 4300, 9.0, 6.0),
    (408, "DET Defense", "DST", "DST", "DET", 2900, 6.5, 5.0),
]

GAMES = {"KC": "KC@MIA 09/28/2026 01:00PM ET", "MIA": "KC@MIA 09/28/2026 01:00PM ET",
         "BUF": "BUF@DET 09/28/2026 04:05PM ET", "DET": "BUF@DET 09/28/2026 04:05PM ET"}

SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]


def dk_file():
    lines = ["Position,Name + ID,Salary,Name,ID,Roster Position,Game Info,TeamAbbrev,AvgPointsPerGame"]
    for pid, name, pos, rp, team, sal, proj, own in PLAYERS:
        rp_out = {"RB": "RB/FLEX", "WR": "WR/FLEX", "TE": "TE/FLEX"}.get(rp, rp)
        lines.append(f"{pos},{name} ({pid}),{sal},{name},{pid},{rp_out},{GAMES[team]},{team},{proj}")
    return io.BytesIO("\n".join(lines).encode("utf-8"))


def ss_file():
    lines = ["Name,My Proj,My Own"]
    for pid, name, pos, rp, team, sal, proj, own in PLAYERS:
        lines.append(f"{name},{proj},{own}")
    return io.BytesIO("\n".join(lines).encode("utf-8"))


def build_pipeline():
    df = prepare_player_pool(dk_file(), ss_file())
    bb_worthy = bringback_worthy_teams(df)
    sim = topdown_simulate_slate(df, sims=10000, seed=42)
    res = generate_lineups(
        df, 50000, "GPP / top-heavy", 8, 400, 40000, 2, "Required",
        [], {}, {}, True, False, 42, bringback_worthy=bb_worthy,
    )
    assert not res.empty, "synthetic build produced no lineups"
    eq = lineup_sim_equity(res, sim["worlds"], df["Name"].astype(str).tolist())
    return df, bb_worthy, sim, res, eq


def test_stack_summary_parsing():
    p = parse_stack_summary("KC: KC Quarterback + KC Wide1 | MIA bring-back: none")
    assert p["qb_team"] == "KC" and p["qb_name"] == "KC Quarterback", p
    assert p["opp"] == "MIA" and p["bb_names"] == [], p
    p2 = parse_stack_summary("BUF: BUF Quarterback + BUF Wide1 + BUF Tight | DET bring-back: DET Wide1")
    assert p2["bb_names"] == ["DET Wide1"], p2
    assert parse_stack_summary("")["qb_team"] == ""
    print("PASS parsing")


def test_short_story():
    row = {"Stack Summary": "KC: KC Quarterback + KC Wide1 | MIA bring-back: none"}
    assert short_story(row) == "KC leads / no MIA answer", short_story(row)
    row2 = {"Stack Summary": "BUF: BUF Quarterback + BUF Wide1 | DET bring-back: DET Wide1"}
    assert short_story(row2) == "BUF leads / DET comeback", short_story(row2)
    print("PASS short_story")


def test_hero_and_sort(res, eq):
    hs = hero_stats(res, eq)
    assert hs["grade"] in ("A+", "A", "A-", "B+", "B", "B-", "C+", "C"), hs
    assert hs["lineups"] == len(res) and hs["top_projection"] > 0
    assert hs["best_ceiling"] > 0 and hs["avg_break"] >= 0
    disp = res.copy()
    for c in ["Sim Mean", "Ceiling P90", "Break Slate %"]:
        if c in eq.columns:
            disp[c] = eq[c].to_numpy()
    for label in SORT_OPTIONS:
        ordered = sort_lineups(disp, label)
        assert len(ordered) == len(disp)
        col = SORT_OPTIONS[label]
        vals = pd.to_numeric(ordered[col], errors="coerce").tolist()
        assert vals == sorted(vals, reverse=True), f"not sorted desc: {label}"
    print("PASS hero_stats + sort_lineups")


def test_world_rows(sim):
    rows = world_rows(sim["game_table"])
    assert len(rows) == 2, rows
    probs = [r["prob"] for r in rows]
    assert probs == sorted(probs, reverse=True), "worlds not sorted desc"
    assert all(r["archetype"] in ("Shootout", "Balanced scoring", "Defensive grind") for r in rows)
    assert abs(sum(probs) - 100.0) < 1.0, f"ceiling shares should ~sum to 100, got {sum(probs)}"
    print("PASS world_rows:", [(r["game"], r["archetype"], r["prob"]) for r in rows])


def test_explorer_helpers(df, res):
    row = res.iloc[0]
    pt = player_table_for_lineup(row, df)
    assert list(pt.columns) == ["Slot", "Player", "Pos", "Team", "Salary", "Proj", "Own %", "Value", "Leverage"]
    assert len(pt) == 9 and pt["Slot"].tolist() == SLOTS
    assert set(pt["Leverage"].unique()) <= {"High", "Med", "Low", "—"}
    alts = swap_suggestions(df, row, "WR1", 40000)
    assert isinstance(alts, list)
    for a in alts:
        assert set(a) == {"Player", "Team", "Salary", "Proj", "Own %", "dProj", "dOwn", "dSalary", "Tradeoff"}, a
        assert a["Tradeoff"], "empty tradeoff note"
    # lineup_title never empty
    assert lineup_title(row), "empty lineup title"
    print(f"PASS explorer helpers ({len(alts)} WR1 swaps suggested)")


def test_bringback_worthiness_gating(df, bb_worthy, res):
    weak = [t for t in ("KC", "MIA", "BUF", "DET") if t not in bb_worthy]
    print(f"  worthy={sorted(bb_worthy)} weak={weak}")
    name_team = {str(r["Name"]): str(r["Team"]) for _, r in df.iterrows()}
    for _, row in res.iterrows():
        parts = parse_stack_summary(row.get("Stack Summary", ""))
        for bb in parts["bb_names"]:
            assert name_team.get(bb, "") in bb_worthy, \
                f"bring-back {bb} from weak offense {name_team.get(bb)}"
    # With Required mode, lineups whose QB faces a weak offense get NO bring-back.
    for _, row in res.iterrows():
        parts = parse_stack_summary(row.get("Stack Summary", ""))
        if parts["opp"] in weak:
            assert parts["bb_names"] == [], f"weak-offense bring-back leaked: {row.get('Stack Summary')}"
    print("PASS bring-back worthiness gating")


if __name__ == "__main__":
    test_stack_summary_parsing()
    test_short_story()
    df, bb_worthy, sim, res, eq = build_pipeline()
    print(f"  built {len(res)} lineups on synthetic KC@MIA + BUF@DET slate")
    test_hero_and_sort(res, eq)
    test_world_rows(sim)
    test_explorer_helpers(df, res)
    test_bringback_worthiness_gating(df, bb_worthy, res)
    print("\nALL RESULTS-COMMAND-CENTER TESTS PASSED")
