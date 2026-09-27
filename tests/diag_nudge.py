"""Diagnose the Optional bring-back nudge on a realistic-scale synthetic slate.

Measures, across seeds and nudge magnitudes:
  - fire rate: fraction of lineups where the nudge vector was produced
  - flip rate: fraction where the re-solve actually added a bring-back
  - final bring-back rate
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import io
import numpy as np
import pandas as pd

import dfs_lab.data as _data_mod
_EMPTY_DF = pd.DataFrame()
_data_mod._load_live_nfl_availability = lambda season=2026: (_EMPTY_DF, _EMPTY_DF)

from dfs_lab.data import prepare_player_pool
from dfs_lab.leverage import compute_mobile_qb_tags
from dfs_lab.classic import generate_lineups
import dfs_lab.classic as classic_mod
import dfs_lab.config as config_mod

rng = np.random.default_rng(0)
TEAMS = [f"T{i:02d}" for i in range(20)]  # 10 games
GAMES = {}
for g in range(10):
    a, b = TEAMS[2*g], TEAMS[2*g+1]
    gi = f"{a}@{b} 09/28/2026 01:00PM ET"
    GAMES[a] = gi; GAMES[b] = gi

# Team offensive strength varies; worthiness gate uses real nflverse ratings,
# but for the diagnostic we pass an explicit worthy set (the 10 strongest).
strength = {t: rng.normal(0, 1) for t in TEAMS}
WORTHY = {t for t in TEAMS if strength[t] > np.median(list(strength.values()))}

players = []
pid = 1000
for t in TEAMS:
    s = strength[t]
    def add(name, pos, rp, sal, proj, own):
        global pid
        pid += 1
        players.append((pid, name, pos, rp, t, sal, proj, own))
    qb_proj = 17 + 2.2*s + rng.normal(0, 1.2)
    add(f"{t} QB", "QB", "QB", int(6000+400*s+rng.normal(0,300)), round(qb_proj,1), round(6+2*s+rng.normal(0,2),1))
    for r in range(2):
        p = 13 + 1.8*s + rng.normal(0, 2.0)
        add(f"{t} RB{r+1}", "RB", "RB/FLEX", int(4500+500*s+rng.normal(0,400)), round(max(p,3),1), round(5+2*s+rng.normal(0,2.5),1))
    for w in range(4):
        p = 11 + 1.6*s - w*1.5 + rng.normal(0, 1.8)
        add(f"{t} WR{w+1}", "WR", "WR/FLEX", int(4000+450*s-w*400+rng.normal(0,300)), round(max(p,2.5),1), round(4+1.5*s+rng.normal(0,2),1))
    p = 8 + 1.2*s + rng.normal(0, 1.5)
    add(f"{t} TE1", "TE", "TE/FLEX", int(3500+300*s+rng.normal(0,300)), round(max(p,2.5),1), round(3+1.2*s+rng.normal(0,1.5),1))
    add(f"{t} DST", "DST", "DST", int(2200+200*s+rng.normal(0,200)), round(6+rng.normal(0,1.5),1), round(4+rng.normal(0,2),1))

def _pool():
    lines = ["Position,Name + ID,Salary,Name,ID,Roster Position,Game Info,TeamAbbrev,AvgPointsPerGame"]
    for pid_, name, pos, rp, team, sal, proj, own in players:
        lines.append(f"{pos},{name} ({pid_}),{sal},{name},{pid_},{rp},{GAMES[team]},{team},{proj}")
    dk = io.BytesIO("\n".join(lines).encode())
    ss = io.BytesIO(("Name,My Proj,My Own\n" + "\n".join(
        f"{name},{proj},{max(own,0.5)}" for _, name, _, _, _, _, proj, own in players)).encode())
    df = prepare_player_pool(dk, ss)
    df["Rush Share"] = 0.05
    df["Mobile QB"] = compute_mobile_qb_tags(df)
    return df

# Instrument the nudge: count fires and flips.
stats = {"fires": 0, "flips": 0, "total": 0}
_orig = classic_mod.bringback_nudge_vector
def _counting(df, randomized, chosen, mode, worthy):
    v = _orig(df, randomized, chosen, mode, worthy)
    if v is not None:
        stats["fires"] += 1
        # a flip happens iff the second solve's lineup differs in bring-back status;
        # we detect it after the fact via the final rate vs first-pass rate instead.
    return v
classic_mod.bringback_nudge_vector = _counting

def _bb_rate(res, df):
    # bring-back = opposing skill player of the lineup's QB, worthy-gated like the app
    id2row = {str(r["ID"]): r for _, r in df.iterrows()}
    n = bb = 0
    for _, r in res.iterrows():
        qb_id = str(r["QB_ID"]); q = id2row.get(qb_id)
        if q is None: continue
        n += 1
        opp = str(q["Opponent"]).upper()
        if opp not in WORTHY: continue
        ids = {str(r[c]) for c in ["RB1_ID","RB2_ID","WR1_ID","WR2_ID","WR3_ID","TE_ID","FLEX_ID"]}
        for pid_ in ids:
            pr = id2row.get(pid_)
            if pr is not None and str(pr["Team"]).upper() == opp and (pr["is_RB"] or pr["is_WR"] or pr["is_TE"]):
                bb += 1; break
    return bb / n if n else 0.0

for bonus in [1.2]:
    config_mod.BRINGBACK_NUDGE = bonus
    classic_mod.BRINGBACK_NUDGE = bonus
    # classic.py imported the name directly, so patch the module attr AND the
    # reference used inside bringback_nudge_vector via module lookup.
    import importlib
    src_rates, fires, totals = [], [], []
    for seed in [7, 11, 21, 42]:
        stats.update(fires=0, flips=0, total=0)
        df = _pool()
        res = generate_lineups(df, 500, "GPP / top-heavy", 20, 800, 40000, 1,
                               "Optional", [], {}, {}, False, False, seed,
                               bringback_worthy=WORTHY)
        totals.append(len(res)); fires.append(stats["fires"])
        src_rates.append(_bb_rate(res, df))
    print(f"bonus={bonus:4.1f}  bring-back rate={np.mean(src_rates):.2f}  "
          f"fire rate={np.sum(fires)/max(np.sum(totals),1):.2f}  n={np.sum(totals)}")
