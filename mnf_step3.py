"""Headless MNF Showdown rebuild WITH SaberSim ownership + comparison.

Same inactives/overrides as step 2, except Keenum nudged 11.0 -> 12.0
(SaberSim has him at 12.81 as the confirmed starter; 12.0 stays conservative).
Ertz stays at 7.5 (LAB model 8.14 on his real history; SaberSim's 4.76 is the
outlier and likely doesn't price the elevation).
"""
import os
import sys

sys.path.insert(0, "/home/hatch/workspace/dfs-lab")

import pandas as pd  # noqa: E402

from dfs_lab.data import prepare_showdown_pool, apply_projection_overrides  # noqa: E402
from dfs_lab.showdown import (  # noqa: E402
    apply_context_engine,
    apply_showdown_scenario,
    audit_showdown_portfolio,
    configure_game_worlds,
    generate_showdown_lineups,
    showdown_upload_csv,
)

DK = "/home/hatch/workspace/user/files/DKSalaries_15.csv"
SS = "/home/hatch/workspace/user/files/NFL_2026-09-28-515pm_DK_SHOWDOWN_PHI-_-CHI.csv"
OUT_DIR = "/home/hatch/workspace/dfs-lab/mnf_out"

KEENUM_ID = "44282409"
ERTZ_ID = "44282430"
OUT_NAMES = {"Caleb Williams", "Dallas Goedert", "Hollywood Brown",
             # SaberSim zeroes him (0.00/0.00 = not expected active); the LAB's
             # history-only 3.97 is stale. Only zeroed player the guard left active.
             "Elijah Moore"}


class _F:
    def __init__(self, path, name):
        with open(path, "rb") as fh:
            self._b = fh.read()
        self.name = name

    def getvalue(self):
        return self._b


def default_strategy(pid):
    return {"Lock": False, "CPT Lock": False, "Exclude": False,
            "CPT Eligible": True, "Priority": "Neutral",
            "Min Exposure": 0, "Max Exposure": 100,
            "CPT Min": 0, "CPT Max": 100}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = prepare_showdown_pool(_F(DK, "DKSalaries_15.csv"), SS)
    print(f"Pool: {len(df)} players; CPT Own estimated: "
          f"{int(df['CPT Own Estimated'].sum())} (rest from SaberSim CPT rows)")

    # Inactives out (SaberSim also has them at 0/0 — belt and suspenders).
    df = df[~df["Name"].isin(OUT_NAMES)].reset_index(drop=True)
    assert not df["Name"].isin(OUT_NAMES).any()

    # Keenum: starter; Ertz: expected elevation.
    df.loc[df["Name"].eq("Case Keenum"), "Primary QB"] = True
    df.loc[df["Team"].eq("CHI") & df["Name"].ne("Case Keenum"), "Primary QB"] = False
    for pid in (KEENUM_ID, ERTZ_ID):
        m = df["ID"].astype(str).eq(pid)
        assert m.any(), pid
        df.loc[m, "ActiveForBuild"] = True
        df.loc[m, "Auto Excluded Reason"] = ""

    build_df = apply_showdown_scenario(df, "Neutral", "", False, None, 50)
    build_df = apply_context_engine(build_df, {}, "Standard")
    build_df = apply_projection_overrides(build_df, {KEENUM_ID: 12.0, ERTZ_ID: 7.5})
    zero = pd.to_numeric(build_df["DFS Lab Proj"], errors="coerce").fillna(0.0) <= 0.01
    build_df.loc[zero, "ActiveForBuild"] = False

    # Projection comparison: LAB vs SaberSim (flex basis).
    comp = build_df[["Name", "Position", "Team", "FlexSalary",
                      "DFS Lab Proj", "SaberSim Proj", "My Own", "CPT Own"]].copy()
    comp["Delta"] = comp["DFS Lab Proj"] - comp["SaberSim Proj"]
    print("\n=== LAB vs SaberSim (top deltas) ===")
    print(comp.sort_values("Delta", ascending=False).head(8).to_string(index=False))
    print(comp.sort_values("Delta").head(8).to_string(index=False))

    configure_game_worlds(build_df)
    strategy_map = {str(pid): default_strategy(pid) for pid in build_df["ID"].astype(str)}

    result = generate_showdown_lineups(
        build_df, field_size=500, payout_style="Winner take all", count=20,
        attempts=120, min_salary=47500, max_salary=50000,
        construction_weights={"3-3": 1, "4-2": 1, "5-1": 1},
        script="Neutral", script_team="", strategy_map=strategy_map,
        entry_format="Single Entry", cpt_qb_passcatchers=0,
        wrte_cpt_qb_pair_pct=80, rb_cpt_dst_k_pct=35,
        max_k=2, max_dst=1, min_unique=1, seed=20260928,
        relationship_rules=[], world_influence=50, show_progress=False,
    )
    assert result is not None and not result.empty
    print(f"\nBuilt {len(result)} lineups")
    id_to_name = dict(zip(build_df["ID"].astype(str), build_df["Name"]))
    v = audit_showdown_portfolio(result, strategy_map, id_to_name)
    print("Audit violations:", v if v else "none")

    print("\n=== Captain distribution ===")
    print(result["CPT"].value_counts().to_string())

    pool_sal = dict(zip(build_df["Name"],
                        zip(build_df["FlexSalary"], build_df["CaptainSalary"])))
    pool_proj = dict(zip(build_df["Name"], build_df["DFS Lab Proj"]))
    pool_own = dict(zip(build_df["Name"],
                        zip(build_df["My Own"], build_df["CPT Own"])))
    for i in range(6):
        row = result.iloc[i]
        print(f"\nRank {int(row['Rank'])} | {row['CPT']} CPT | Proj {row['Projection']:.1f} | "
              f"Rating {row['Rating Score']:.1f} | Sal {int(row['Salary'])} | {row['Construction']} | "
              f"{row['Game World']} | Lev {row.get('Leverage Grade','')} | Dup {row.get('Dup Risk','')}")
        for n, s in enumerate(["CPT"] + [f"FLEX{k}" for k in range(1, 6)]):
            nm = row[s]
            fs, cs = pool_sal.get(nm, (0, 0))
            p = pool_proj.get(nm, 0)
            fo, co = pool_own.get(nm, (0, 0))
            tag = "CPT" if n == 0 else "FLX"
            cost = cs if n == 0 else fs
            pts = p * 1.5 if n == 0 else p
            own = co if n == 0 else fo
            print(f"   {tag} {nm:20s} ${cost:<6d} proj {pts:5.1f} own {own:5.1f}%")

    result.to_pickle(f"{OUT_DIR}/mnf_result_ss.pkl")
    with open(f"{OUT_DIR}/mnf_dk_upload_ss.csv", "w") as fh:
        fh.write(showdown_upload_csv(result))
    print(f"\nSaved: {OUT_DIR}/mnf_result_ss.pkl, mnf_dk_upload_ss.csv")


if __name__ == "__main__":
    main()
