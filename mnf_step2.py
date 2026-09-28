"""Headless MNF Showdown build: PHI @ CHI, 2026-09-28.

Inactives/overrides applied (verified 2026-09-28 news):
  OUT: Caleb Williams (hamstring), Dallas Goedert (knee), Marquise "Hollywood" Brown (ankle)
  Keenum: named CHI starter -> forced active as CHI primary QB, proj 11.0
  Ertz: expected elevation from practice squad -> forced active, proj 7.5
  Barkley: off injury report, full practice -> no adjustment
"""
import sys

sys.path.insert(0, "/home/hatch/workspace/dfs-lab")

import pandas as pd  # noqa: E402

from dfs_lab.data import apply_projection_overrides  # noqa: E402
from dfs_lab.showdown import (  # noqa: E402
    apply_context_engine,
    apply_showdown_scenario,
    audit_showdown_portfolio,
    configure_game_worlds,
    generate_showdown_lineups,
    showdown_upload_csv,
)

POOL = "/home/hatch/workspace/dfs-lab/mnf_pool.pkl"
OUT_DIR = "/home/hatch/workspace/dfs-lab/mnf_out"

KEENUM_ID = "44282409"
ERTZ_ID = "44282430"
OUT_IDS = {"44282405", "44282461",   # Caleb Williams (CPT/FLEX rows collapse to one)
           "44282413", "44282469",   # Dallas Goedert
           "44282429", "44282485"}   # Hollywood Brown


def default_strategy(pid):
    return {"Lock": False, "CPT Lock": False, "Exclude": False,
            "CPT Eligible": True, "Priority": "Neutral",
            "Min Exposure": 0, "Max Exposure": 100,
            "CPT Min": 0, "CPT Max": 100}


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_pickle(POOL)

    # --- Inactives: drop OUT players entirely ---
    df = df[~df["ID"].astype(str).isin(OUT_IDS)].reset_index(drop=True)
    assert not df["Name"].str.contains("Caleb Williams|Dallas Goedert|Hollywood Brown").any(), \
        "OUT player still in pool!"

    # --- Keenum: CHI starter. Fix primary-QB flag and re-activate ---
    df.loc[df["Name"].eq("Case Keenum"), "Primary QB"] = True
    df.loc[df["Team"].eq("CHI") & df["Name"].ne("Case Keenum"), "Primary QB"] = False
    for pid in (KEENUM_ID, ERTZ_ID):
        m = df["ID"].astype(str).eq(pid)
        assert m.any(), f"override target {pid} missing from pool"
        df.loc[m, "ActiveForBuild"] = True
        df.loc[m, "Auto Excluded Reason"] = ""

    # --- Scenario + context (Neutral script, no context takes) ---
    build_df = apply_showdown_scenario(df, "Neutral", "", False, None, 50)
    build_df = apply_context_engine(build_df, {}, "Standard")
    build_df = apply_projection_overrides(build_df, {KEENUM_ID: 11.0, ERTZ_ID: 7.5})

    # Final safety gate (mirrors UI): zero-proj players are unavailable.
    zero = pd.to_numeric(build_df["DFS Lab Proj"], errors="coerce").fillna(0.0) <= 0.01
    build_df.loc[zero, "ActiveForBuild"] = False

    active = build_df[build_df["ActiveForBuild"].astype(bool)]
    print(f"Active pool: {len(active)} players")
    print(active[["Name", "Position", "Team", "FlexSalary", "CaptainSalary",
                  "DFS Lab Proj"]].sort_values("DFS Lab Proj", ascending=False)
          .head(18).to_string(index=False))

    configure_game_worlds(build_df)
    strategy_map = {str(pid): default_strategy(pid) for pid in build_df["ID"].astype(str)}

    result = generate_showdown_lineups(
        build_df,
        field_size=500, payout_style="Winner take all", count=20,
        attempts=120, min_salary=47500, max_salary=50000,
        construction_weights={"3-3": 1, "4-2": 1, "5-1": 1},
        script="Neutral", script_team="",
        strategy_map=strategy_map, entry_format="Single Entry",
        cpt_qb_passcatchers=0, wrte_cpt_qb_pair_pct=80, rb_cpt_dst_k_pct=35,
        max_k=2, max_dst=1, min_unique=1, seed=20260928,
        relationship_rules=[], world_influence=50, show_progress=False,
    )
    assert result is not None and not result.empty, "Build produced no lineups!"
    print(f"\nBuilt {len(result)} lineups")

    id_to_name = dict(zip(build_df["ID"].astype(str), build_df["Name"]))
    violations = audit_showdown_portfolio(result, strategy_map, id_to_name)
    print("Audit violations:", violations if violations else "none")

    cols = ["Rank", "CPT", "FLEX", "FLEX.1", "FLEX.2", "FLEX.3", "FLEX.4",
            "Projection", "Rating Score", "Salary", "Lineup Story"]
    have = [c for c in cols if c in result.columns]
    print("\n=== TOP 5 ===")
    print(result[have].head(5).to_string(index=False))

    result.to_pickle(f"{OUT_DIR}/mnf_result.pkl")
    with open(f"{OUT_DIR}/mnf_dk_upload.csv", "w") as fh:
        fh.write(showdown_upload_csv(result))
    print(f"\nSaved: {OUT_DIR}/mnf_result.pkl, mnf_dk_upload.csv")


if __name__ == "__main__":
    main()
