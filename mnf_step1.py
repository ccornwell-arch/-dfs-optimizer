"""Headless MNF Showdown build: PHI @ CHI, 2026-09-28.

Step 1: parse the DK slate, build Aytia projections, print the sanity table.
Step 2 (after review): apply inactives + overrides, generate the portfolio.
"""
import io
import sys

sys.path.insert(0, "/home/hatch/workspace/dfs-lab")

import pandas as pd  # noqa: E402

from dfs_lab.data import prepare_showdown_pool  # noqa: E402

CSV = "/home/hatch/workspace/user/files/DKSalaries_15.csv"


def main():
    with open(CSV, "rb") as fh:
        raw = fh.read()
    f = io.BytesIO(raw)
    f.name = "DKSalaries_15.csv"
    df = prepare_showdown_pool(f)
    print(f"Pool players: {len(df)}")
    print(f"Teams: {sorted(df['Team'].dropna().unique().tolist())}")
    cols = ["Name", "Position", "Team", "FlexSalary", "CaptainSalary",
            "My Proj", "AvgPointsPerGame", "ActiveForBuild",
            "Auto Excluded Reason" if "Auto Excluded Reason" in df.columns else "My Proj"]
    view = df[[c for c in cols if c in df.columns]].copy()
    view = view.sort_values("My Proj", ascending=False)
    print(view.head(35).to_string(index=False))
    print("\n--- Auto-excluded ---")
    excl = view[~view["ActiveForBuild"].astype(bool)] if "ActiveForBuild" in view.columns else view.iloc[0:0]
    print(excl[["Name", "Position", "Team", "FlexSalary", "My Proj"]].to_string(index=False)
          if len(excl) else "(none)")
    df.to_pickle("/home/hatch/workspace/dfs-lab/mnf_pool.pkl")
    print("\nSaved pool -> mnf_pool.pkl")


if __name__ == "__main__":
    main()
