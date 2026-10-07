"""Leverage Lane: ownership-aware leverage constructions for Aytia Classic.

Ideas sourced from Ship It Nation strategy content (Sep 2026, "Leverage Lane"
and "Sunday Shippers" episodes), implemented as data-driven rules — never
hardcoded player names:

1. Mobile-QB stack relaxation. Rushing QBs (the Lamar/Allen/Daniels archetype)
   can be played naked or with a skinny stack because their rushing IS the
   correlation. A QB is tagged mobile when rushing fantasy is >=
   MOBILE_QB_RUSH_SHARE of his total fantasy (four-season weighted nflverse
   weekly stats, computed by the projection engine). Mobile QBs need one fewer
   same-team pass catcher (floor 0), count as one stack leg in the correlation
   score, and are never penalized for a naked build.

2. Chalk-cutting RB+DST bonus. Pairing a CHALK running back with his own DST
   is a differentiated way to play popular RBs (if the RB's team leads, the
   DST gets sack/turnover opportunity). The app already gives a flat
   correlation reward for any same-team RB+DST pairing; this adds an
   ownership-aware kicker on top, routed through the leverage weight path,
   only when the paired RB is chalk.

3. Leverage Lane callout. One highest-leverage play per slate for Slate
   Intel: projection edge over salary-implied expectation, divided by
   estimated ownership.

4. "If chalk fails" beneficiaries. From the top-down sim worlds: for each of
   the most-owned players, who posts the best conditional outcomes in the
   worlds where that chalk busts (bottom quartile of his own world
   distribution).

Ownership is always Aytia's estimate unless a SaberSim file supplied it;
callers must keep the "estimated" label wherever it is displayed.
"""
import numpy as np
import pandas as pd

# QB tagged mobile when rushing fantasy >= 25% of total fantasy. Four-season
# weighted, so Lamar/Allen/Daniels/Hurts archetypes clear it (~30-45%) while
# pocket passers (Mahomes ~12%, Burrow ~5%) do not.
MOBILE_QB_RUSH_SHARE = 0.25

# An RB is chalk for the RB+DST kicker when his estimated ownership clears
# both an absolute floor (15%) and the 75th percentile of active-RB ownership.
CHALK_RB_OWN_FLOOR = 15.0

# Leverage-raw bonus per chalk RB+DST pair, in average-ownership percentage
# points. It shifts the leverage component only; the flat RB+DST correlation
# reward is untouched, so nothing is double-counted.
CHALK_CUT_KICKER = 2.0

# Salary-implied fantasy points per $1K: the same rate the projection engine
# uses as its salary prior for thin-history players.
SALARY_PTS_PER_1K = 1.55

# Ownership floor for the leverage-lane ratio so a ~0% estimate can't
# produce an infinite score.
LEVERAGE_OWN_FLOOR = 2.0

# Bust = at or below the 25th percentile of the player's own sim-world
# distribution.
BUST_QUANTILE = 0.25


def _active(df):
    if "ActiveForBuild" in df.columns:
        return df["ActiveForBuild"].fillna(False).astype(bool)
    return pd.Series(True, index=df.index)


def compute_mobile_qb_tags(df):
    """Boolean Series: True for QBs whose rushing share clears the threshold.

    Falls back to all-False when the pool has no rushing-share signal (e.g.
    SaberSim path or nflverse evidence unavailable) — a missing signal means
    no relaxation, never a fabricated tag.
    """
    is_qb = df["Position"].astype(str).str.upper().eq("QB") if "Position" in df.columns else df.get("is_QB", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    if "Rush Share" not in df.columns:
        return pd.Series(False, index=df.index)
    share = pd.to_numeric(df["Rush Share"], errors="coerce").fillna(0.0)
    return (is_qb & (share >= MOBILE_QB_RUSH_SHARE)).fillna(False).astype(bool)


def rb_chalk_threshold(df):
    """Estimated-ownership cutoff for a chalk RB on this slate."""
    m = _active(df) & df["Position"].astype(str).str.upper().eq("RB") if "Position" in df.columns else _active(df)
    own = pd.to_numeric(df.loc[m, "My Own"], errors="coerce").fillna(0.0)
    if own.empty:
        return CHALK_RB_OWN_FLOOR
    return float(max(CHALK_RB_OWN_FLOOR, own.quantile(0.75)))


def chalk_cut_pairs(df, chosen):
    """Same-team (chalk RB, DST) pairs among the chosen lineup indexes.

    Returns a list of (rb_name, dst_name) where the RB clears the chalk
    threshold. Pure function of the pool + lineup; no side effects.
    """
    idxs = [i for _, i in chosen]
    p = df.loc[idxs]
    thr = rb_chalk_threshold(df)
    rb_teams = set(p.loc[p["is_RB"].fillna(False).astype(bool), "Team"].astype(str))
    dst_teams = set(p.loc[p["is_DST"].fillna(False).astype(bool), "Team"].astype(str))
    pairs = []
    for t in rb_teams & dst_teams:
        rbs = p[(p["is_RB"].fillna(False).astype(bool)) & (p["Team"].astype(str) == t)]
        dsts = p[(p["is_DST"].fillna(False).astype(bool)) & (p["Team"].astype(str) == t)]
        for _, rb in rbs.iterrows():
            if float(rb.get("My Own", 0) or 0) >= thr:
                for _, dst in dsts.iterrows():
                    pairs.append((str(rb["Name"]), str(dst["Name"])))
    return pairs


def leverage_lane_pick(df):
    """One highest-leverage skill-position play for Slate Intel.

    leverage = (My Proj - salary-implied points) / max(est. own, floor).
    Requires a positive projection edge. Deterministic: score desc, then
    edge desc, then name asc. Returns None when nothing qualifies.
    """
    m = _active(df)
    pos = df["Position"].astype(str).str.upper()
    cand = df[m & pos.isin(["RB", "WR", "TE"])].copy()
    if cand.empty:
        return None
    proj = pd.to_numeric(cand["My Proj"], errors="coerce").fillna(0.0)
    sal = pd.to_numeric(cand["Salary"], errors="coerce").fillna(0.0)
    own = pd.to_numeric(cand["My Own"], errors="coerce").fillna(0.0)
    implied = sal / 1000.0 * SALARY_PTS_PER_1K
    edge = proj - implied
    cand = cand[edge > 0]
    if cand.empty:
        return None
    edge = edge[edge > 0]
    own = own.loc[cand.index]
    score = edge / own.clip(lower=LEVERAGE_OWN_FLOOR)
    order = sorted(cand.index, key=lambda i: (-float(score.loc[i]), -float(edge.loc[i]), str(cand.loc[i, "Name"])))
    top = cand.loc[order[0]]
    est = True
    if "Own Estimated" in df.columns:
        try:
            est = bool(df["Own Estimated"].fillna(True).iloc[0])
        except Exception:
            est = True
    return {
        "name": str(top["Name"]),
        "position": str(top["Position"]).upper(),
        "team": str(top["Team"]),
        "salary": int(sal.loc[order[0]]),
        "proj": round(float(proj.loc[order[0]]), 1),
        "own": round(float(own.loc[order[0]]), 1),
        "edge": round(float(edge.loc[order[0]]), 1),
        "score": round(float(score.loc[order[0]]), 3),
        "own_estimated": est,
    }


def chalk_bust_beneficiaries(sim_worlds, df, top_chalk=5, per_chalk=3):
    """Who benefits when chalk busts, from top-down sim worlds.

    For each of the most-owned active players: bust worlds = worlds where he
    scores at/below the 25th percentile of his own distribution. Beneficiaries
    are the other active players with the highest conditional mean in those
    worlds, filtered to players whose conditional mean exceeds their
    unconditional mean (they genuinely benefit, not just good players).
    Deterministic ordering: conditional mean desc, then name asc.
    """
    if sim_worlds is None or getattr(sim_worlds, "empty", True) or df is None or df.empty:
        return []
    names = df["Name"].astype(str).tolist()
    if len(names) != sim_worlds.shape[1]:
        return []
    col_of = {nm: c for c, nm in zip(sim_worlds.columns, names)}
    active = _active(df)
    own = pd.to_numeric(df["My Own"], errors="coerce").fillna(0.0)
    chalk_idx = own[active].sort_values(ascending=False).head(max(1, int(top_chalk))).index.tolist()
    out = []
    for i in chalk_idx:
        nm = str(df.loc[i, "Name"])
        col = col_of.get(nm)
        if col is None:
            continue
        dist = pd.to_numeric(sim_worlds[col], errors="coerce").fillna(0.0)
        thr = float(dist.quantile(BUST_QUANTILE))
        bust = dist <= thr
        if int(bust.sum()) < 50:
            continue
        rows = []
        for j in df.index:
            if j == i or not bool(active.loc[j]):
                continue
            onm = str(df.loc[j, "Name"])
            ocol = col_of.get(onm)
            if ocol is None:
                continue
            od = pd.to_numeric(sim_worlds[ocol], errors="coerce").fillna(0.0)
            cond = float(od[bust].mean())
            uncond = float(od.mean())
            if cond > uncond:
                rows.append({
                    "name": onm,
                    "team": str(df.loc[j, "Team"]),
                    "position": str(df.loc[j, "Position"]).upper(),
                    "cond_mean": round(cond, 1),
                    "gain": round(cond - uncond, 1),
                })
        rows.sort(key=lambda r: (-r["cond_mean"], r["name"]))
        out.append({
            "chalk": nm,
            "chalk_team": str(df.loc[i, "Team"]),
            "chalk_pos": str(df.loc[i, "Position"]).upper(),
            "chalk_own": round(float(own.loc[i]), 1),
            "beneficiaries": rows[: max(0, int(per_chalk))],
        })
    return out
