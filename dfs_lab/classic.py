"""Moved verbatim from streamlit_app.py (refactor/modularize). No logic changes."""

import csv
import math
import io
import os
import json
import re
import difflib
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


from dfs_lab.common import _first_existing, contest_aggression, percentile_label
from dfs_lab.leverage import (CHALK_CUT_KICKER, chalk_cut_pairs, compute_mobile_qb_tags,
                              rb_chalk_threshold)
from dfs_lab.data import (_name_col, _opp_col, _season_col,
                          _dk_fantasy_points_from_stats, _load_nflverse_projection_inputs)
from dfs_lab.config import ROSTER_SLOTS, PRIORITY_BONUS, TEAM_PRIORITY_BONUS, BRINGBACK_NUDGE

def letter_grade(score):
    if score >= 93: return "A+"
    if score >= 89: return "A"
    if score >= 85: return "A-"
    if score >= 81: return "B+"
    if score >= 77: return "B"
    if score >= 73: return "B-"
    if score >= 68: return "C+"
    if score >= 63: return "C"
    return "C-"

def add_ratings(out, aggr):
    if out.empty:
        return out

    # Component percentiles are relative to the lineups generated in THIS build.
    proj_pct = out["Projection"].rank(pct=True)
    corr_pct = out["Correlation Raw"].rank(pct=True)
    fit_pct = out["User Fit Raw"].rank(pct=True)
    # Chalk-cutting RB+DST pairs earn leverage credit through the leverage
    # weight path. The kicker is additive to a separate component from the
    # flat RB+DST correlation reward, so nothing is double-counted.
    _chalk = pd.to_numeric(out["Chalk Cut"], errors="coerce").fillna(0.0) if "Chalk Cut" in out.columns else 0.0
    leverage_raw = -out["Avg Own"] + CHALK_CUT_KICKER * _chalk
    lev_pct = leverage_raw.rank(pct=True)
    unique_pct = out["Salary Left"].rank(pct=True)
    coherence_pct = out["Coherence Score"].rank(pct=True) if "Coherence Score" in out.columns else pd.Series(0.5,index=out.index)

    # Contest-aware weights. Coherence is intentionally meaningful: a lineup that
    # barely passes the football-sense gate should not outrank a similarly projected
    # lineup with a cleaner first-place story.
    w_proj = 0.42 - 0.10 * aggr
    w_corr = 0.22 + 0.05 * aggr
    w_fit = 0.18
    w_lev = 0.08 + 0.11 * aggr
    w_unique = 0.00 + 0.04 * aggr
    w_coh = 0.12
    total = w_proj + w_corr + w_fit + w_lev + w_unique + w_coh

    composite = (
        w_proj * proj_pct
        + w_corr * corr_pct
        + w_fit * fit_pct
        + w_lev * lev_pct
        + w_unique * unique_pct
        + w_coh * coherence_pct
    ) / total

    # V2.1 fix:
    # The old version treated the weighted percentile itself as a school-style
    # 0-100 score, which made even strong lineups show as C/C+.
    # Now the letter grade is based on where the lineup ranks versus this build.
    rel_pct = composite.rank(pct=True, method="average")

    def relative_grade(p):
        if p >= 0.95: return "A+"
        if p >= 0.85: return "A"
        if p >= 0.70: return "A-"
        if p >= 0.50: return "B+"
        if p >= 0.30: return "B"
        if p >= 0.15: return "B-"
        if p >= 0.05: return "C+"
        return "C"

    # Display score is intentionally relative, not fake precision.
    # 68–99 keeps it readable while the letter grade is the main signal.
    rating_score = 68 + 31 * rel_pct

    out["Rating Score"] = rating_score.round(1)
    out["Rating"] = [relative_grade(x) for x in rel_pct]
    out["Projection Grade"] = [percentile_label(x, out["Projection"]) for x in out["Projection"]]
    out["Correlation Grade"] = [percentile_label(x, out["Correlation Raw"]) for x in out["Correlation Raw"]]
    out["Leverage Grade"] = [percentile_label(x, leverage_raw) for x in leverage_raw]

    # If every lineup has the same user-fit score, don't misleadingly label them all Excellent.
    if out["User Fit Raw"].nunique() <= 1:
        out["User Fit Grade"] = "Neutral"
    else:
        out["User Fit Grade"] = [percentile_label(x, out["User Fit Raw"]) for x in out["User Fit Raw"]]

    return out

def slot_eligibility(df):
    return {
        "QB": df["is_QB"].to_numpy(bool),
        "RB1": df["is_RB"].to_numpy(bool),
        "RB2": df["is_RB"].to_numpy(bool),
        "WR1": df["is_WR"].to_numpy(bool),
        "WR2": df["is_WR"].to_numpy(bool),
        "WR3": df["is_WR"].to_numpy(bool),
        "TE": df["is_TE"].to_numpy(bool),
        "FLEX": df["is_FLEX"].to_numpy(bool) & ~df["is_QB"].to_numpy(bool) & ~df["is_DST"].to_numpy(bool),
        "DST": df["is_DST"].to_numpy(bool),
    }

def player_objective(df, aggr, strategy_map, preferred_stack_teams, team_strategy_map, exposure_state=None, built_count=0):
    proj_col = "Aytia Proj" if "Aytia Proj" in df.columns else ("Script Proj" if "Script Proj" in df.columns else "My Proj")
    proj = df[proj_col].to_numpy(float)
    own = np.clip(df["My Own"].to_numpy(float), 0.05, None)

    # Projection remains the backbone.
    objective = proj.copy()

    # Contest-aware, controlled leverage.
    leverage = np.log((proj + 2.0) / (own + 2.0))
    objective += (0.6 + 1.8 * aggr) * leverage

    # User opinions matter explicitly.
    for i, r in df.iterrows():
        strat = strategy_map.get(str(r["ID"]), {})
        pr = strat.get("Priority", "Neutral")
        objective[i] += PRIORITY_BONUS.get(pr, 0.0)
        # Portfolio exposure steering:
        # If a player is below the user's desired minimum, boost him.
        # If he is already near/over the target max, reduce him before the hard max filter.
        if exposure_state is not None and built_count > 0:
            current_exp = 100.0 * exposure_state.get(str(r["ID"]), 0) / built_count
            min_exp = float(strat.get("Min Exposure", 0))
            max_exp = float(strat.get("Max Exposure", 100))
            if current_exp < min_exp:
                deficit = min_exp - current_exp
                objective[i] += min(5.0, 0.10 * deficit)
            if current_exp > max_exp - 5:
                objective[i] -= min(4.0, 0.10 * max(0.0, current_exp - (max_exp - 5)))

        team_pr = team_strategy_map.get(r["Team"], "Neutral")
        objective[i] += TEAM_PRIORITY_BONUS.get(team_pr, 0.0)
        if r["Team"] in preferred_stack_teams and r["is_QB"]:
            objective[i] += 1.1
        elif r["Team"] in preferred_stack_teams and (r["is_WR"] or r["is_TE"]):
            objective[i] += 0.45

    return objective

def solve_one(
    df, aggr, rng, strategy_map, preferred_stack_teams, team_strategy_map,
    min_salary, qb_stack_min, bringback_mode,
    exposure_state=None, built_count=0,
    no_dst_from_qb_game=True, no_offense_vs_dst=False,
    noise_scale=0.16, max_players_team=9, max_players_game=9,
    max_te=3, allow_qb_with_rb=True, allowed_qb_ids=None, flex_position=None,
    bringback_worthy=None,
):
    n = len(df)
    s = len(ROSTER_SLOTS)
    total_vars = n * s
    elig = slot_eligibility(df)
    active = df["ActiveForBuild"].to_numpy(bool)

    base = player_objective(df, aggr, strategy_map, preferred_stack_teams, team_strategy_map, exposure_state, built_count)
    randomized = base * np.exp(rng.normal(0, noise_scale, size=n))

    c = np.zeros(total_vars)
    integrality = np.ones(total_vars)
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)

    def vidx(i, j): return i * s + j

    # Build variable bounds. The objective vector is filled per solve pass
    # below; bounds and constraints do not depend on it.
    for i in range(n):
        player_id = str(df.loc[i, "ID"])
        strat = strategy_map.get(player_id, {})
        excluded = bool(strat.get("Exclude", False)) or strat.get("Priority") == "Exclude"
        locked = bool(strat.get("Lock", False)) and not excluded
        if allowed_qb_ids and bool(df.loc[i,"is_QB"]) and player_id not in set(str(x) for x in allowed_qb_ids):
            excluded = True

        for j, slot in enumerate(ROSTER_SLOTS):
            if (not active[i]) or excluded or (not elig[slot][i]):
                ub[vidx(i, j)] = 0
            elif slot=="FLEX" and flex_position in {"RB","WR","TE"}:
                pos_ok = (
                    (flex_position=="RB" and bool(df.loc[i,"is_RB"])) or
                    (flex_position=="WR" and bool(df.loc[i,"is_WR"])) or
                    (flex_position=="TE" and bool(df.loc[i,"is_TE"]))
                )
                if not pos_ok:
                    ub[vidx(i, j)] = 0

    rows, lows, highs = [], [], []

    # Exactly one per slot.
    for j in range(s):
        rows.append({vidx(i, j): 1.0 for i in range(n)})
        lows.append(1.0); highs.append(1.0)

    # No duplicate player.
    for i in range(n):
        rows.append({vidx(i, j): 1.0 for j in range(s)})
        lows.append(0.0); highs.append(1.0)

    # Locks.
    for i in range(n):
        player_id = str(df.loc[i, "ID"])
        strat = strategy_map.get(player_id, {})
        if bool(strat.get("Lock", False)) and not bool(strat.get("Exclude", False)):
            coeff = {vidx(i, j): 1.0 for j in range(s)}
            rows.append(coeff); lows.append(1.0); highs.append(1.0)

    # Salary.
    coeff = {}
    for i in range(n):
        for j in range(s):
            coeff[vidx(i, j)] = float(df.loc[i, "Salary"])
    rows.append(coeff); lows.append(float(min_salary)); highs.append(50000.0)

    # If preferred stack teams are chosen, QB must be from one of them.
    if preferred_stack_teams:
        coeff = {}
        for i in df.index[df["is_QB"]]:
            if df.loc[i, "Team"] in preferred_stack_teams:
                for j in range(s):
                    if elig[ROSTER_SLOTS[j]][i]:
                        coeff[vidx(i, j)] = 1.0
        if coeff:
            rows.append(coeff); lows.append(1.0); highs.append(1.0)

    # QB stack / bringback.
    # Mobile QBs (rushing is the correlation) need one fewer same-team pass
    # catcher, floor 0 — a naked rushing-QB build is a designed construction,
    # not a constraint violation.
    _mob = compute_mobile_qb_tags(df)
    for q in df.index[df["is_QB"] & df["ActiveForBuild"]]:
        team = df.loc[q, "Team"]
        opp = df.loc[q, "Opponent"]
        qmin = qb_stack_min - (1 if bool(_mob.loc[q]) else 0)
        qmin = max(0, qmin)

        receivers = df.index[
            df["ActiveForBuild"] & (df["Team"] == team) & (df["is_WR"] | df["is_TE"])
        ].tolist()

        coeff = {}
        for i in receivers:
            for j in range(s):
                if elig[ROSTER_SLOTS[j]][i]:
                    coeff[vidx(i, j)] = coeff.get(vidx(i, j), 0.0) + 1.0
        for j in range(s):
            if elig[ROSTER_SLOTS[j]][q]:
                coeff[vidx(q, j)] = coeff.get(vidx(q, j), 0.0) - float(qmin)
        rows.append(coeff); lows.append(0.0); highs.append(np.inf)

        # A forced bring-back from a weak offense is bad process: the "shootout"
        # game script is a blowout instead. Only enforce it against worthy offenses.
        if bringback_mode == "Required" and (bringback_worthy is None or str(opp).upper() in bringback_worthy):
            opp_skill = df.index[
                df["ActiveForBuild"] & (df["Team"] == opp)
                & (df["is_RB"] | df["is_WR"] | df["is_TE"])
            ].tolist()
            coeff = {}
            for i in opp_skill:
                for j in range(s):
                    if elig[ROSTER_SLOTS[j]][i]:
                        coeff[vidx(i, j)] = coeff.get(vidx(i, j), 0.0) + 1.0
            for j in range(s):
                if elig[ROSTER_SLOTS[j]][q]:
                    coeff[vidx(q, j)] = coeff.get(vidx(q, j), 0.0) - 1.0
            rows.append(coeff); lows.append(0.0); highs.append(np.inf)

        elif bringback_mode == "None":
            opp_skill = df.index[
                df["ActiveForBuild"] & (df["Team"] == opp)
                & (df["is_RB"] | df["is_WR"] | df["is_TE"])
            ].tolist()
            coeff = {}
            for i in opp_skill:
                for j in range(s):
                    if elig[ROSTER_SLOTS[j]][i]:
                        coeff[vidx(i, j)] = coeff.get(vidx(i, j), 0.0) + 1.0
            # If QB selected, opponent skill count must be 0.
            # Big-M: opp_skill_count + 8*QB <= 8
            for j in range(s):
                if elig[ROSTER_SLOTS[j]][q]:
                    coeff[vidx(q, j)] = coeff.get(vidx(q, j), 0.0) + 8.0
            rows.append(coeff); lows.append(0.0); highs.append(8.0)

    # Conditional rule: if a QB stack is used, do not roster either DST from that game.
    if no_dst_from_qb_game:
        for q in df.index[df["is_QB"] & df["ActiveForBuild"]]:
            q_team = df.loc[q, "Team"]
            q_opp = df.loc[q, "Opponent"]
            game_dsts = df.index[df["is_DST"] & df["Team"].isin([q_team, q_opp])].tolist()
            for d in game_dsts:
                coeff = {}
                for j in range(s):
                    if elig[ROSTER_SLOTS[j]][q]:
                        coeff[vidx(q, j)] = coeff.get(vidx(q, j), 0.0) + 1.0
                    if elig[ROSTER_SLOTS[j]][d]:
                        coeff[vidx(d, j)] = coeff.get(vidx(d, j), 0.0) + 1.0
                rows.append(coeff); lows.append(0.0); highs.append(1.0)

    # Optional broader rule: do not roster offensive players against your DST.
    if no_offense_vs_dst:
        for d in df.index[df["is_DST"] & df["ActiveForBuild"]]:
            d_team = df.loc[d, "Team"]
            d_opp = df.loc[d, "Opponent"]
            opp_offense = df.index[
                df["ActiveForBuild"] & (df["Team"] == d_opp) & ~df["is_DST"]
            ].tolist()
            for o in opp_offense:
                coeff = {}
                for j in range(s):
                    if elig[ROSTER_SLOTS[j]][d]:
                        coeff[vidx(d, j)] = coeff.get(vidx(d, j), 0.0) + 1.0
                    if elig[ROSTER_SLOTS[j]][o]:
                        coeff[vidx(o, j)] = coeff.get(vidx(o, j), 0.0) + 1.0
                rows.append(coeff); lows.append(0.0); highs.append(1.0)

    # Avoid QB vs opposing DST.
    for q in df.index[df["is_QB"] & df["ActiveForBuild"]]:
        opp = df.loc[q, "Opponent"]
        opp_dst = df.index[df["is_DST"] & (df["Team"] == opp)].tolist()
        for d in opp_dst:
            coeff = {}
            for j in range(s):
                if elig[ROSTER_SLOTS[j]][q]:
                    coeff[vidx(q, j)] = coeff.get(vidx(q, j), 0.0) + 1.0
                if elig[ROSTER_SLOTS[j]][d]:
                    coeff[vidx(d, j)] = coeff.get(vidx(d, j), 0.0) + 1.0
            rows.append(coeff); lows.append(0.0); highs.append(1.0)

    # Classic portfolio structure controls.
    if int(max_players_team) < 9:
        for team in df["Team"].dropna().unique().tolist():
            coeff={}
            for i in df.index[df["Team"].eq(team)]:
                for j in range(s):
                    if elig[ROSTER_SLOTS[j]][i]:
                        coeff[vidx(i,j)] = 1.0
            if coeff:
                rows.append(coeff); lows.append(0.0); highs.append(float(max_players_team))

    if int(max_players_game) < 9 and "Matchup" in df.columns:
        for matchup in [m for m in df["Matchup"].dropna().unique().tolist() if str(m).strip()]:
            coeff={}
            for i in df.index[df["Matchup"].eq(matchup)]:
                for j in range(s):
                    if elig[ROSTER_SLOTS[j]][i]:
                        coeff[vidx(i,j)] = 1.0
            if coeff:
                rows.append(coeff); lows.append(0.0); highs.append(float(max_players_game))

    if int(max_te) < 3:
        coeff={}
        for i in df.index[df["is_TE"]]:
            for j in range(s):
                if elig[ROSTER_SLOTS[j]][i]:
                    coeff[vidx(i,j)] = 1.0
        if coeff:
            rows.append(coeff); lows.append(0.0); highs.append(float(max_te))

    if not bool(allow_qb_with_rb):
        for q in df.index[df["is_QB"] & df["ActiveForBuild"]]:
            qteam=df.loc[q,"Team"]
            for rb in df.index[df["is_RB"] & df["ActiveForBuild"] & df["Team"].eq(qteam)]:
                coeff={}
                for j in range(s):
                    if elig[ROSTER_SLOTS[j]][q]:
                        coeff[vidx(q,j)] = coeff.get(vidx(q,j),0.0)+1.0
                    if elig[ROSTER_SLOTS[j]][rb]:
                        coeff[vidx(rb,j)] = coeff.get(vidx(rb,j),0.0)+1.0
                rows.append(coeff); lows.append(0.0); highs.append(1.0)

    A = lil_matrix((len(rows), total_vars), dtype=float)
    for r, coeff in enumerate(rows):
        for col, val in coeff.items():
            A[r, col] = val
    _cons = LinearConstraint(A.tocsr(), np.array(lows), np.array(highs))

    def _run(obj_vec):
        c = np.zeros(total_vars)
        for i in range(n):
            v = -obj_vec[i]
            for j in range(s):
                c[vidx(i, j)] = v
        result = milp(
            c=c,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            constraints=_cons,
            options={"time_limit": 8.0},
        )
        if not result.success or result.x is None:
            return None
        chosen = []
        for j, slot in enumerate(ROSTER_SLOTS):
            vals = [(result.x[vidx(i, j)], i) for i in range(n)]
            _, i = max(vals)
            chosen.append((slot, int(i)))
        return chosen

    chosen = _run(randomized)

    # Optional-mode bring-back nudge. The per-player objective has no
    # correlation term, so without this "Optional" builds a bring-back only
    # by accident. "None" still forbids bring-backs; "Required" still forces
    # them by constraint.
    nudged = bringback_nudge_vector(df, randomized, chosen, bringback_mode, bringback_worthy)
    if nudged is not None:
        second = _run(nudged)
        if second:
            chosen = second
    return chosen


def bringback_nudge_vector(df, randomized, chosen, bringback_mode, bringback_worthy):
    """Second-pass bring-back incentive for bringback_mode="Optional".

    Returns a copy of the (already noise-drawn) objective vector with
    BRINGBACK_NUDGE added to the chosen QB's active opposing skill players,
    when the first-pass lineup rides a QB whose opposing offense is worthy
    of a shootout but carries no bring-back. Returns None when no second
    pass is warranted (wrong mode, unknown/unworthy opponent, or the lineup
    already has a bring-back). The caller keeps the first-pass lineup if the
    re-solve fails.
    """
    if not (chosen and bringback_mode == "Optional" and bringback_worthy):
        return None
    qb_idx = next((i for _, i in chosen if bool(df.loc[i, "is_QB"])), None)
    if qb_idx is None:
        return None
    opp = str(df.loc[qb_idx, "Opponent"]).upper()
    if opp not in bringback_worthy:
        return None
    if any(
        str(df.loc[i, "Team"]).upper() == opp
        and (bool(df.loc[i, "is_RB"]) or bool(df.loc[i, "is_WR"]) or bool(df.loc[i, "is_TE"]))
        for _, i in chosen
    ):
        return None
    nudged = randomized.copy()
    team_up = df["Team"].astype(str).str.upper().to_numpy()
    skill = (df["is_RB"] | df["is_WR"] | df["is_TE"]).to_numpy(bool)
    nudged[df["ActiveForBuild"].to_numpy(bool) & (team_up == opp) & skill] += BRINGBACK_NUDGE
    return nudged

def classic_lineup_coherence(df, chosen):
    """Football-sense validator for DraftKings Classic lineups.

    Hard rejection is reserved for combinations whose paths to first place directly
    fight each other. Softer concerns lower the coherence score but remain available
    when there is a credible tournament explanation.
    """
    idxs=[i for _,i in chosen]
    p=df.loc[idxs].copy()
    hard=[]; warnings=[]; score=100.0

    # No auto-excluded/backup player may sneak through even if another rule changes.
    if "Role Confidence" in p.columns:
        bad=p[p["Role Confidence"].astype(str).isin(["Backup QB","Inactive / no usable projection"])]
        if not bad.empty:
            hard.append("Contains a player Aytia marked unavailable for the current football role.")

    qbs=p[p["is_QB"]]
    if len(qbs)!=1:
        hard.append("Classic lineup must contain exactly one usable primary QB.")
        qb=None
    else:
        qb=qbs.iloc[0]

    # DST/offense contradictions. One opponent can occasionally be defensible; an
    # opposing QB stack attacking the same DST is not a coherent first-place story.
    for _,dst in p[p["is_DST"]].iterrows():
        opp=str(dst.get("Opponent",""))
        opp_off=p[(p["Team"].astype(str).eq(opp)) & (~p["is_DST"])]
        opp_qb=opp_off[opp_off["is_QB"]]
        opp_pc=opp_off[opp_off["is_WR"]|opp_off["is_TE"]]
        if len(opp_qb)>=1 and len(opp_pc)>=1:
            hard.append(f"{dst['Name']} is paired against an opposing QB stack.")
        elif len(opp_off)>=3:
            hard.append(f"{dst['Name']} is paired against three or more opposing offensive players.")
        elif len(opp_off)==2:
            score-=22; warnings.append(f"{dst['Name']} faces two players in this lineup")
        elif len(opp_off)==1:
            score-=8; warnings.append(f"{dst['Name']} faces one player in this lineup")

    stack_names=[]; bb_names=[]
    qb_mobile = bool(qb.get("Mobile QB", False)) if qb is not None else False
    if qb is not None:
        qteam=str(qb["Team"]); opp=str(qb.get("Opponent",""))
        pcs=p[(p["Team"].astype(str).eq(qteam)) & (p["is_WR"]|p["is_TE"])]
        bring=p[(p["Team"].astype(str).eq(opp)) & (p["is_RB"]|p["is_WR"]|p["is_TE"])]
        stack_names=pcs["Name"].astype(str).tolist()
        bb_names=bring["Name"].astype(str).tolist()
        if len(pcs)==0:
            if qb_mobile:
                # A naked rushing-QB build is a designed construction: the
                # rushing equity IS the correlation, so no penalty.
                pass
            else:
                # Naked QB can be viable for rushing/TD concentration, so flag it rather
                # than banning it without a player-specific rushing model.
                score-=18; warnings.append("Naked QB requires the quarterback to create ceiling without a receiver stack")
        elif len(pcs)>=2:
            score+=4
        if len(bring)>=1:
            score+=3

    # Reward secondary same-game or RB+DST correlation without requiring it.
    rb_dst=len(set(p.loc[p["is_RB"],"Team"].astype(str)) & set(p.loc[p["is_DST"],"Team"].astype(str)))
    score+=min(4,2*rb_dst)
    score=float(np.clip(score,0,100))

    # Chalk-cutting pairs: a chalk RB rostered with his own DST is a
    # differentiated way to play popular RBs (leverage kicker, not just
    # the flat correlation reward above).
    chalk_pairs=chalk_cut_pairs(df, chosen)

    if qb is not None:
        qname=str(qb["Name"]); qteam=str(qb["Team"]); opp=str(qb.get("Opponent",""))
        if stack_names:
            stack_txt=" + ".join(stack_names)
        else:
            stack_txt="naked rushing-QB build" if qb_mobile else "naked QB ceiling"
        bb_txt=(" with "+opp+" run-back "+", ".join(bb_names)) if bb_names else ""
        story=f"{qteam} passing/rushing ceiling through {qname} + {stack_txt}{bb_txt}"
    else:
        story="No coherent QB-led Classic story available"

    if rb_dst:
        story += f"; {rb_dst} RB+DST correlation" if rb_dst==1 else f"; {rb_dst} RB+DST correlations"
    for rb_name, dst_name in chalk_pairs[:2]:
        story += f"; chalk-cutting {rb_name} + {dst_name}"

    return {
        "Accept":len(hard)==0,
        "Coherence Score":round(score,1),
        "Lineup Story":story,
        "Coherence Flags":"; ".join(hard+warnings) if (hard or warnings) else "No major football contradictions"
    }

def lineup_details(df, chosen, strategy_map, preferred_stack_teams, bringback_worthy=None):
    idxs = [i for _, i in chosen]
    p = df.loc[idxs].copy()

    projection = float(p["My Proj"].sum())
    salary = int(p["Salary"].sum())
    total_own = float(p["My Own"].sum())
    avg_own = total_own / 9.0

    qb = p[p["is_QB"]].iloc[0]
    qb_team = qb["Team"]
    opp = qb["Opponent"]
    matchup = qb["Matchup"]
    qb_mobile = bool(qb.get("Mobile QB", False))

    pass_catchers = p[(p["Team"] == qb_team) & (p["is_WR"] | p["is_TE"])]
    # A player from a weak opposing offense is salary filler, not a game-script
    # call — don't frame or score him as a bring-back.
    _bb_worthy = bringback_worthy is None or str(opp).upper() in bringback_worthy
    if _bb_worthy:
        bringbacks = p[(p["Team"] == opp) & (p["is_RB"] | p["is_WR"] | p["is_TE"])]
    else:
        bringbacks = p.iloc[0:0]

    rb_dst = len(set(p.loc[p["is_RB"], "Team"]) & set(p.loc[p["is_DST"], "Team"]))
    qb_game_players = int((p["Matchup"] == matchup).sum())

    # User fit rewards intentional preferences, penalizes fades.
    fit = 0.0
    fit_notes = []
    for i in idxs:
        pid = str(df.loc[i, "ID"])
        pr = strategy_map.get(pid, {}).get("Priority", "Neutral")
        if pr == "Core":
            fit += 2.2; fit_notes.append(f"{df.loc[i, 'Name']} Core")
        elif pr == "Like":
            fit += 1.0; fit_notes.append(f"{df.loc[i, 'Name']} Like")
        elif pr == "Fade":
            fit -= 1.3; fit_notes.append(f"{df.loc[i, 'Name']} Fade")

    if qb_team in preferred_stack_teams:
        fit += 1.8
        fit_notes.append(f"{qb_team} preferred stack")

    # No correlation credit for a bring-back from an offense too weak to shoot out.
    # A mobile QB's rushing counts as one stack leg: the rushing equity IS the
    # correlation, so a skinny/naked mobile-QB build is not scored as uncorrelated.
    bb_worthy = bringback_worthy is None or str(opp).upper() in bringback_worthy
    stack_legs = min(len(pass_catchers) + (1 if qb_mobile else 0), 2)
    correlation_raw = (
        2.0 * stack_legs
        + (0.9 * min(len(bringbacks), 1) if bb_worthy else 0.0)
        + 0.5 * rb_dst
        + 0.35 * max(0, qb_game_players - 2)
    )
    chalk_pairs = chalk_cut_pairs(df, chosen)

    stack_names = " + ".join(pass_catchers["Name"].tolist()) if len(pass_catchers) else "none"
    bb_names = " + ".join(bringbacks["Name"].tolist()) if len(bringbacks) else "none"
    stack_summary = f"{qb_team}: {qb['Name']} + {stack_names} | {opp} bring-back: {bb_names}"
    if qb_mobile and len(pass_catchers) == 0:
        stack_summary += " · naked rushing-QB build"

    return {
        "Projection": round(projection, 2),
        "Salary": salary,
        "Salary Left": 50000 - salary,
        "Total Own": round(total_own, 1),
        "Avg Own": round(avg_own, 1),
        "QB Stack": len(pass_catchers),
        "Bring-backs": len(bringbacks),
        "QB Game Players": qb_game_players,
        "RB+DST": rb_dst,
        "Chalk Cut": len(chalk_pairs),
        "QB Mobile": 1 if qb_mobile else 0,
        "Correlation Raw": round(correlation_raw, 2),
        "User Fit Raw": round(fit, 2),
        "Stack Summary": stack_summary,
        "Strategy Notes": ", ".join(fit_notes) if fit_notes else "Neutral build",
    }

def generate_lineups(
    df, field_size, payout_style, count, attempts, min_salary,
    qb_stack_min, bringback_mode, preferred_stack_teams,
    strategy_map, team_strategy_map, no_dst_from_qb_game, no_offense_vs_dst, seed,
    max_players_team=9, max_players_game=9, max_te=3, allow_qb_with_rb=True,
    allowed_qb_ids=None, flex_mix=None, bringback_worthy=None,
):
    aggr = contest_aggression(field_size, payout_style)
    rng = np.random.default_rng(seed)
    seen = set()
    rows = []
    exposure_counts = defaultdict(int)

    # Portfolio-level FLEX mix. Convert percentages to exact lineup counts with
    # largest-remainder rounding, then shuffle the requested FLEX positions so
    # one position is not systematically favored early in generation.
    flex_schedule=[]
    if flex_mix:
        raw={p:max(0.0,float(flex_mix.get(p,0) or 0)) for p in ["RB","WR","TE"]}
        total=sum(raw.values())
        if total>0:
            norm={p:100.0*raw[p]/total for p in raw}
            exact={p:count*norm[p]/100.0 for p in norm}
            quota={p:int(np.floor(exact[p])) for p in exact}
            left=max(0,int(count)-sum(quota.values()))
            order=sorted(exact,key=lambda p:(exact[p]-quota[p]),reverse=True)
            for p in order[:left]:
                quota[p]+=1
            for p in ["RB","WR","TE"]:
                flex_schedule.extend([p]*quota[p])
            rng.shuffle(flex_schedule)

    prog = st.progress(0, text="Generating lineups...")

    for attempt in range(attempts):
        if len(rows) >= count:
            break

        chosen = solve_one(
            df, aggr, rng, strategy_map, preferred_stack_teams, team_strategy_map,
            min_salary, qb_stack_min, bringback_mode,
            exposure_state=exposure_counts,
            built_count=len(rows),
            no_dst_from_qb_game=no_dst_from_qb_game,
            no_offense_vs_dst=no_offense_vs_dst,
            noise_scale=0.13 + 0.11 * aggr,
            max_players_team=max_players_team,
            max_players_game=max_players_game,
            max_te=max_te,
            allow_qb_with_rb=allow_qb_with_rb,
            allowed_qb_ids=allowed_qb_ids,
            flex_position=(flex_schedule[len(rows)] if len(rows)<len(flex_schedule) else None),
            bringback_worthy=bringback_worthy,
        )
        if not chosen:
            continue

        ids = tuple(sorted(str(df.loc[i, "ID"]) for _, i in chosen))
        if ids in seen:
            continue

        # Soft max exposure filter during pool generation.
        reject = False
        for _, i in chosen:
            pid = str(df.loc[i, "ID"])
            max_exp = float(strategy_map.get(pid, {}).get("Max Exposure", 100))
            if len(rows) >= 10:
                projected_exp = 100 * (exposure_counts[pid] + 1) / (len(rows) + 1)
                if projected_exp > max_exp + 3.0:
                    reject = True
                    break
        if reject:
            continue

        coherence=classic_lineup_coherence(df,chosen)
        if not coherence["Accept"]:
            continue

        seen.add(ids)
        detail = lineup_details(df, chosen, strategy_map, preferred_stack_teams, bringback_worthy)
        detail.update({
            "Coherence Score":coherence["Coherence Score"],
            "Lineup Story":coherence["Lineup Story"],
            "Coherence Flags":coherence["Coherence Flags"]
        })
        row = dict(detail)
        for slot, i in chosen:
            row[slot] = df.loc[i, "Name"]
            row[slot + "_ID"] = str(df.loc[i, "ID"])
        rows.append(row)

        for _, i in chosen:
            exposure_counts[str(df.loc[i, "ID"])] += 1

        if attempt % 10 == 0:
            prog.progress(min(1.0, len(rows) / max(1, count)), text=f"Generated {len(rows)} / {count}")

    prog.empty()

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = add_ratings(out, aggr)
    out = out.sort_values(["Rating Score", "Projection"], ascending=[False, False]).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    return out

def _classic_time_bucket(game_info):
    import re
    s=str(game_info or "")
    m=re.search(r"(\d{1,2}):(\d{2})\s*(AM|PM)",s,re.I)
    if m:
        h=int(m.group(1))%12
        if m.group(3).upper()=="PM": h+=12
        return "Night" if h>=18 else "Day"
    # Some DK exports use a 24-hour clock without AM/PM.
    m24=re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b",s)
    if m24:
        return "Night" if int(m24.group(1))>=18 else "Day"
    return "Time unknown"

@st.cache_data(ttl=21600, show_spinner=False)
def classic_context_evidence(intel_df):
    """Historical + opponent-vs-position evidence for Classic Slate Intel."""
    base=intel_df[["Name","Position","Team","Opponent","Game Info","My Proj"]].copy()
    base["Hist FPPG"]=np.nan; base["Recent 6"]=np.nan; base["Hist Games"]=0
    base["DVP Adj %"]=0.0; base["Current Window"]="Unavailable"
    base["Time"]=base["Game Info"].map(_classic_time_bucket)
    try:
        season=2026
        stats=_load_nflverse_projection_inputs(season).copy()
        nc=_name_col(stats); sc=_season_col(stats); oc=_opp_col(stats)
        if not nc or not sc:
            return base
        stats["Name"]=stats[nc].astype(str).str.strip()
        stats["Season"]=pd.to_numeric(stats[sc],errors="coerce")
        stats["_fp"]=_dk_fantasy_points_from_stats(stats)
        week_col=_first_existing(stats.columns,["week","Week"])
        pos_col=_first_existing(stats.columns,["position","position_group","Pos"])
        stats["_week"]=pd.to_numeric(stats[week_col],errors="coerce").fillna(0) if week_col else 0
        hist=stats.groupby("Name")["_fp"].agg(["mean","count"]).reset_index().rename(columns={"mean":"Hist FPPG","count":"Hist Games"})
        recent=(stats.sort_values(["Season","_week"]).groupby("Name").tail(6).groupby("Name")["_fp"].mean().reset_index().rename(columns={"_fp":"Recent 6"}))
        base=base.drop(columns=["Hist FPPG","Recent 6","Hist Games"]).merge(hist,on="Name",how="left").merge(recent,on="Name",how="left")
        base["Current Window"]="2023–2026 + recent 6"
        if oc and pos_col:
            stats["Opp"]=stats[oc].astype(str).str.strip()
            stats["Pos"]=stats[pos_col].astype(str).str.upper().replace({"HB":"RB","FB":"RB"})
            use=stats[stats["Pos"].isin(["QB","RB","WR","TE"])]
            league=use.groupby("Pos")["_fp"].mean().to_dict()
            allowed=use.groupby(["Opp","Pos"],as_index=False).agg(Allowed=("_fp","mean"),N=("_fp","size"))
            dvp={}
            for _,r in allowed.iterrows():
                denom=max(float(league.get(r["Pos"],0)),1.0)
                raw=float(r["Allowed"])/denom-1.0
                shrink=float(r["N"])/(float(r["N"])+24.0)
                dvp[(str(r["Opp"]),str(r["Pos"]))]=float(np.clip(raw*shrink,-0.12,0.12))*100
            base["DVP Adj %"]=[round(dvp.get((str(r["Opponent"]),str(r["Position"]).upper()),0.0),1) for _,r in base.iterrows()]
    except Exception:
        pass
    for col in ["Hist FPPG","Recent 6","DVP Adj %"]:
        base[col]=pd.to_numeric(base[col],errors="coerce")
    base["Hist Games"]=pd.to_numeric(base["Hist Games"],errors="coerce").fillna(0).astype(int)
    return base

@st.cache_data(show_spinner=False)
def _nflverse_team_ratings(season):
    """Offense (points scored) and defense (points allowed) ratings per team.

    Four-season weighted means with the same recency/reliability treatment as
    the projection engine. Returns (offense, defense, league_avg).
    """
    from dfs_lab.data import _load_nflverse_inputs
    _, sched = _load_nflverse_inputs(season)
    s = sched.copy()
    tg = pd.concat([
        pd.DataFrame({"team": s["home_team"].astype(str).str.upper(),
                      "season": pd.to_numeric(s["season"], errors="coerce"),
                      "scored": pd.to_numeric(s["home_score"], errors="coerce"),
                      "allowed": pd.to_numeric(s["away_score"], errors="coerce")}),
        pd.DataFrame({"team": s["away_team"].astype(str).str.upper(),
                      "season": pd.to_numeric(s["season"], errors="coerce"),
                      "scored": pd.to_numeric(s["away_score"], errors="coerce"),
                      "allowed": pd.to_numeric(s["home_score"], errors="coerce")}),
    ], ignore_index=True).dropna(subset=["scored", "allowed"])
    w = {season: 0.34, season - 1: 0.38, season - 2: 0.19, season - 3: 0.09}
    off, deff = {}, {}
    for t, g in tg.groupby("team"):
        on = od = dn = dd = 0.0
        for yr, gg in g.groupby("season"):
            wt = w.get(int(yr), 0.0)
            if wt <= 0:
                continue
            n = len(gg)
            rel = min(1.0, max(0.20, n / 8.0)) if int(yr) == season else min(1.0, n / 8.0)
            ew = wt * rel
            on += ew * gg["scored"].mean(); od += ew
            dn += ew * gg["allowed"].mean(); dd += ew
        if od:
            off[t] = on / od
        if dd:
            deff[t] = dn / dd
    return off, deff, float(tg["scored"].mean())

def _team_fantasy_calibration(season):
    """Empirical skill-position fantasy points per real point scored (team-week).

    Returns (ratio, emp_team_fp): the league ratio and each team's historical
    mean team-week skill fantasy, used to size usage-share coverage.
    """
    from dfs_lab.data import _load_nflverse_inputs, _dk_fantasy_points_from_stats
    stats, sched = _load_nflverse_inputs(season)
    stx = stats.copy()
    stx["_fp"] = _dk_fantasy_points_from_stats(stx)
    posc = _first_existing(stx.columns, ["position", "position_group", "Pos"])
    stx["_pos"] = stx[posc].astype(str).str.upper().replace({"HB": "RB", "FB": "RB"}) if posc else ""
    stx["_tm"] = stx["team"].astype(str).str.upper()
    stx["_sn"] = pd.to_numeric(stx["season"], errors="coerce")
    stx["_wk"] = pd.to_numeric(stx["week"], errors="coerce")
    tw = stx[stx["_pos"].isin(["QB", "RB", "WR", "TE"])].groupby(
        ["_tm", "_sn", "_wk"], as_index=False).agg(skill_fp=("_fp", "sum"))
    s = sched.copy()
    pts = pd.concat([
        pd.DataFrame({"_tm": s["home_team"].astype(str).str.upper(),
                      "_sn": pd.to_numeric(s["season"], errors="coerce"),
                      "_wk": pd.to_numeric(s["week"], errors="coerce"),
                      "pts": pd.to_numeric(s["home_score"], errors="coerce")}),
        pd.DataFrame({"_tm": s["away_team"].astype(str).str.upper(),
                      "_sn": pd.to_numeric(s["season"], errors="coerce"),
                      "_wk": pd.to_numeric(s["week"], errors="coerce"),
                      "pts": pd.to_numeric(s["away_score"], errors="coerce")}),
    ], ignore_index=True)
    m = tw.merge(pts, on=["_tm", "_sn", "_wk"], how="inner")
    m = m[m["pts"] > 0]
    ratio = float((m["skill_fp"] / m["pts"]).mean()) if len(m) else 3.2
    emp = m.groupby("_tm")["skill_fp"].mean().to_dict()
    return ratio, emp

def bringback_worthy_teams(df):
    """Offenses good enough to merit a bring-back.

    A bring-back only pays when the opposing offense can actually score and
    keep a shootout alive. Worthy = four-season weighted points-scored rating
    at or above league average. Returns uppercase team abbreviations.
    """
    from dfs_lab.data import _slate_season
    season = _slate_season(df)
    off, _, lg = _nflverse_team_ratings(season)
    return {str(t).upper() for t, r in off.items() if r >= lg}

def topdown_simulate_slate(df, sims=10000, seed=42):
    """Top-down game simulation: team scores -> team fantasy -> player shares.

    Each game is simulated `sims` times from team offense/defense ratings
    (nflverse, four-season weighted) with a home-field edge and
    pace-correlated team noise. Team fantasy points are dealt to players by
    projection-implied usage shares with individual lognormal noise, so stacks
    and bring-backs emerge from correlated game scripts instead of ad-hoc
    shock terms. Defenses move against their opponent's simulated score via
    the points-allowed bracket.

    Returns a dict with:
      player_stats: Name, Position, Team, Sim Mean, Sim P50, Sim P75, Sim P90,
                    Sim Std, P(3x) % (probability of 3x salary value)
      game_table:   Game, Mean DFS env, P75, P90, Volatility, Slate ceiling %
                    (same shape as the old bottom-up sim table)
      worlds:       DataFrame (sims rows x players) of simulated fantasy
                    points, columns are df integer positions.
    """
    from dfs_lab.data import _slate_season, _dst_points_allowed_fantasy
    d = df.reset_index(drop=True).copy()
    n = len(d)
    empty = {"player_stats": pd.DataFrame(), "game_table": pd.DataFrame(), "worlds": pd.DataFrame()}
    if n == 0:
        return empty
    season = _slate_season(d)
    rng = np.random.default_rng(int(seed))
    sims = int(max(1000, sims))

    off, deff, lg = _nflverse_team_ratings(season)
    ratio, emp_team_fp = _team_fantasy_calibration(season)
    HFA, SCORE_SD, PACE_CORR = 1.8, 10.0, 0.2

    proj = pd.to_numeric(d["My Proj"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()
    pos = d["Position"].astype(str).str.upper().to_numpy()
    teams = d["Team"].astype(str).str.upper().to_numpy()
    is_dst = pos == "DST"

    away = d["Away"].astype(str).str.upper() if "Away" in d.columns else pd.Series([""] * n)
    home = d["Home"].astype(str).str.upper() if "Home" in d.columns else pd.Series([""] * n)
    games, seen = [], set()
    for a, h in zip(away, home):
        if a and h and (a, h) not in seen:
            seen.add((a, h)); games.append((a, h))
    if not games:
        return empty

    team_list = sorted(set(teams.tolist()) | {t for g in games for t in g})
    tix = {t: i for i, t in enumerate(team_list)}
    team_pts = np.zeros((sims, len(team_list)))
    simmed = np.zeros(len(team_list), dtype=bool)
    cov = [[SCORE_SD ** 2, PACE_CORR * SCORE_SD ** 2],
           [PACE_CORR * SCORE_SD ** 2, SCORE_SD ** 2]]
    for (a, h) in games:
        ea = lg * (off.get(a, lg) / lg) * (deff.get(h, lg) / lg)
        eh = lg * (off.get(h, lg) / lg) * (deff.get(a, lg) / lg) + HFA
        draw = rng.multivariate_normal([ea, eh], cov, size=sims)
        team_pts[:, tix[a]] = np.clip(draw[:, 0], 0, None)
        team_pts[:, tix[h]] = np.clip(draw[:, 1], 0, None)
        simmed[tix[a]] = simmed[tix[h]] = True
    team_pts[:, ~simmed] = np.clip(rng.normal(lg, SCORE_SD, size=(sims, (~simmed).sum())), 0, None)
    skill_fp = ratio * team_pts  # sims x teams

    # Usage shares from projections, scaled by historical coverage so backups'
    # fantasy is not dealt to pool players.
    share = np.zeros(n)
    for t in team_list:
        m = (teams == t) & (~is_dst)
        ps = proj[m].sum()
        if ps <= 0:
            continue
        covr = min(1.0, max(0.05, ps / max(emp_team_fp.get(t, ps), 1.0)))
        share[m] = proj[m] / ps * covr

    # Individual noise: part of weekly variance is team-driven, the rest is the
    # player's own. 0.65 keeps total variance near the player's Sim Vol.
    if "Sim Vol" in d.columns:
        pvol = pd.to_numeric(d["Sim Vol"], errors="coerce").fillna(0.45).to_numpy()
    else:
        pvol = np.array([{"QB": 0.26, "RB": 0.42, "WR": 0.50, "TE": 0.48, "DST": 0.62}.get(p, 0.45) for p in pos])
    ind = 0.65 * np.clip(pvol, 0.12, 0.95)
    noise = np.exp(rng.normal(0, 1, size=(sims, n)) * ind[None, :] - 0.5 * ind[None, :] ** 2)

    team_col = np.array([tix[t] for t in teams])
    worlds = np.zeros((sims, n))
    sk = ~is_dst
    worlds[:, sk] = skill_fp[:, team_col[sk]] * share[sk][None, :] * noise[:, sk]

    # Defenses ride their projection, adjusted by the opponent's simulated
    # score through the points-allowed bracket.
    opp_of = {}
    for (a, h) in games:
        opp_of[a] = h; opp_of[h] = a
    for j in np.where(is_dst)[0]:
        t = teams[j]
        opp = opp_of.get(t)
        base = max(float(proj[j]), 0.5)
        adj = (np.vectorize(_dst_points_allowed_fantasy)(team_pts[:, tix[opp]]) - 1.0) if opp else 0.0
        dv = min(0.9, max(0.2, float(pvol[j])))
        worlds[:, j] = np.clip(base + adj + rng.normal(0, dv * base, sims), 0, None)

    sal = pd.to_numeric(d["Salary"], errors="coerce").fillna(0.0).to_numpy()
    thr3x = 3.0 * sal / 1000.0
    player_stats = pd.DataFrame({
        "Name": d["Name"], "Position": d["Position"], "Team": d["Team"],
        "Sim Mean": np.round(worlds.mean(axis=0), 2),
        "Sim P50": np.round(np.median(worlds, axis=0), 2),
        "Sim P75": np.round(np.percentile(worlds, 75, axis=0), 2),
        "Sim P90": np.round(np.percentile(worlds, 90, axis=0), 2),
        "Sim Std": np.round(worlds.std(axis=0), 2),
        "P(3x) %": np.round(100 * (worlds > thr3x[None, :]).mean(axis=0), 1),
    })

    grows, gmats = [], []
    for (a, h) in games:
        gfp = skill_fp[:, tix[a]] + skill_fp[:, tix[h]]
        for t in (a, h):
            dm = np.where(is_dst & (teams == t))[0]
            if len(dm):
                gfp = gfp + worlds[:, dm].sum(axis=1)
        gmats.append(gfp)
        grows.append({"Game": f"{a}@{h}", "Mean DFS env": round(float(gfp.mean()), 1),
                      "P75": round(float(np.percentile(gfp, 75)), 1),
                      "P90": round(float(np.percentile(gfp, 90)), 1),
                      "Volatility": round(float(gfp.std()), 1)})
    game_table = pd.DataFrame(grows)
    mat = np.vstack(gmats)
    winners = np.argmax(mat, axis=0)
    counts = np.bincount(winners, minlength=len(grows))
    game_table["Slate ceiling %"] = np.round(100 * counts / sims, 1)
    game_table = game_table.sort_values(["Slate ceiling %", "P90"], ascending=False).reset_index(drop=True)

    worlds_df = pd.DataFrame(worlds, columns=d.index)
    return {"player_stats": player_stats, "game_table": game_table, "worlds": worlds_df}

def lineup_sim_equity(lineups, worlds_df, names):
    """Per-lineup tournament equity from top-down worlds.

    Ceiling P90 is the lineup's 90th-percentile total; Break Slate % is the
    share of worlds where the lineup beats the 99.5th percentile of all
    built-lineup totals (a slate-breaking score). Positional: row i of the
    return matches row i of `lineups`. `names` is the player-name series in
    the same order as the worlds columns.
    """
    if lineups is None or lineups.empty or worlds_df is None or worlds_df.empty:
        return pd.DataFrame()
    name_to_col = {}
    for c, nm in zip(worlds_df.columns, names):
        name_to_col[str(nm)] = c
    slot_cols = [c for c in ROSTER_SLOTS if c in lineups.columns]
    totals = []
    for _, r in lineups.iterrows():
        cols = [name_to_col.get(str(r[c])) for c in slot_cols]
        cols = [c for c in cols if c is not None]
        totals.append(worlds_df[cols].to_numpy().sum(axis=1) if cols else np.zeros(len(worlds_df)))
    totals = np.vstack(totals)
    nut = float(np.percentile(totals, 99.5))
    return pd.DataFrame({
        "Sim Mean": np.round(totals.mean(axis=1), 1),
        "Ceiling P90": np.round(np.percentile(totals, 90, axis=1), 1),
        "Break Slate %": np.round(100 * (totals > nut).mean(axis=1), 2),
    })

def classic_strategy_theses(df, intel, sim_table, field_size, payout_style, entry_format):
    """Find evidence-backed ways to attack the slate. A thesis may start with a game, QB, receiver, or RB."""
    x=df[["ID","Name","Position","Team","Opponent","Matchup","Salary","My Proj","My Own"]].copy()
    if intel is not None and not intel.empty:
        keep=[z for z in ["Name","Hist FPPG","Recent 6","DVP Adj %"] if z in intel.columns]
        x=x.merge(intel[keep].drop_duplicates("Name"),on="Name",how="left")
    for col in ["Hist FPPG","Recent 6","DVP Adj %"]:
        if col not in x.columns: x[col]=0.0
        x[col]=pd.to_numeric(x[col],errors="coerce").fillna(0.0)
    x["My Proj"]=pd.to_numeric(x["My Proj"],errors="coerce").fillna(0.0)
    x["My Own"]=pd.to_numeric(x["My Own"],errors="coerce").fillna(0.0)
    x["Salary"]=pd.to_numeric(x["Salary"],errors="coerce").fillna(0.0)

    def pct(s):
        s=pd.to_numeric(s,errors="coerce").fillna(0.0)
        if len(s)<=1 or float(s.max())==float(s.min()): return pd.Series(0.5,index=s.index)
        return s.rank(pct=True,method="average")

    x["_proj"]=pct(x["My Proj"])
    x["_recent"]=pct(x["Recent 6"])
    x["_dvp"]=pct(x["DVP Adj %"])
    x["_lev"]=pct(x["My Proj"]/(x["My Own"]+4.0))
    x["_value"]=pct(x["My Proj"]/(x["Salary"]/1000.0+1.0))

    game_share={}
    game_p90={}
    if sim_table is not None and not sim_table.empty:
        game_share={str(r["Game"]):float(r.get("Slate ceiling %",0)) for _,r in sim_table.iterrows()}
        game_p90={str(r["Game"]):float(r.get("P90",0)) for _,r in sim_table.iterrows()}
    if game_share:
        vals=pd.Series(list(game_share.values()))
        lo=float(vals.min()); hi=float(vals.max())
        game_norm={g:(v-lo)/(hi-lo) if hi>lo else 0.5 for g,v in game_share.items()}
    else:
        game_norm={}

    theses=[]
    def add(kind,title,score,team="",game="",player="",paired_qb="",why=""):
        theses.append({"Type":kind,"Thesis":title,"Score":float(score),"Team":str(team),"Game":str(game),
                       "Player":str(player),"Paired QB":str(paired_qb),"Why":why})

    # Game-led theses.
    if sim_table is not None and not sim_table.empty:
        p90s=pd.to_numeric(sim_table["P90"],errors="coerce").fillna(0.0)
        p90pct=p90s.rank(pct=True,method="average")
        for ix,r in sim_table.iterrows():
            score=0.62*(float(r.get("Slate ceiling %",0))/max(float(sim_table["Slate ceiling %"].max()),1.0))+0.38*float(p90pct.loc[ix])
            add("Game",f"{r['Game']} game environment",score,game=r["Game"],
                why=f"{float(r.get('Slate ceiling %',0)):.1f}% slate-ceiling share · P90 {float(r.get('P90',0)):.1f}")

    # Player-led theses. Receiver/RB routes can choose the QB rather than starting from him.
    qbs=x[x["Position"].astype(str).str.upper().eq("QB")].copy()
    for _,r in x.iterrows():
        pos=str(r["Position"]).upper()
        if pos not in ["QB","RB","WR","TE"]: continue
        g=float(game_norm.get(str(r["Matchup"]),0.5))
        if pos=="QB":
            score=.34*r["_proj"]+.22*g+.16*r["_recent"]+.12*r["_dvp"]+.16*r["_lev"]
            add("QB",f"{r['Name']} passing/rushing ceiling",score,team=r["Team"],game=r["Matchup"],player=r["Name"],
                paired_qb=r["Name"],why=f"Proj {r['My Proj']:.1f} · DVP {r['DVP Adj %']:+.1f}% · own {r['My Own']:.1f}%")
        elif pos in ["WR","TE"]:
            same=qbs[qbs["Team"].astype(str).eq(str(r["Team"]))].sort_values("My Proj",ascending=False)
            pq=str(same.iloc[0]["Name"]) if not same.empty else ""
            score=.30*r["_proj"]+.24*g+.17*r["_recent"]+.13*r["_dvp"]+.16*r["_lev"]
            add("Receiver",f"{r['Name']} receiving ceiling",score,team=r["Team"],game=r["Matchup"],player=r["Name"],
                paired_qb=pq,why=f"Proj {r['My Proj']:.1f} · recent {r['Recent 6']:.1f} · DVP {r['DVP Adj %']:+.1f}% · own {r['My Own']:.1f}%")
        else:
            score=.30*r["_proj"]+.20*g+.16*r["_recent"]+.10*r["_dvp"]+.14*r["_lev"]+.10*r["_value"]
            add("RB",f"{r['Name']} RB-led script",score,team=r["Team"],game=r["Matchup"],player=r["Name"],
                why=f"Proj {r['My Proj']:.1f} · recent {r['Recent 6']:.1f} · value/ownership support")

    if not theses:
        return pd.DataFrame(),{"label":"Open slate","reason":"No usable thesis evidence was available."}

    t=pd.DataFrame(theses).sort_values("Score",ascending=False).reset_index(drop=True)
    # Entry format changes how sharply we CONCENTRATE on evidence; it does not choose a fixed number of QBs.
    sharp={"Single Entry":3.0,"3-Max":2.35,"20-Max":1.45,"150-Max":0.90}.get(entry_format,1.5)
    scores=t["Score"].to_numpy(float)
    z=np.exp((scores-scores.max())*sharp*3.0)
    t["Attention %"]=100.0*z/max(z.sum(),1e-9)
    t["Attention %"]=t["Attention %"].round(1)

    top=float(t.iloc[0]["Attention %"])
    top3=float(t.head(3)["Attention %"].sum())
    gap=float(t.iloc[0]["Score"]-t.iloc[1]["Score"]) if len(t)>1 else 1.0
    if top>=34 or gap>=0.11:
        label="Clear stand"
    elif top3>=48:
        label="Strong cluster"
    else:
        label="Open slate"
    reason=(f"{entry_format} rewards {'concentration' if entry_format in ['Single Entry','3-Max'] else 'coverage'}, "
            f"but the slate evidence decides how many paths deserve attention. Top thesis attention {top:.1f}%; top three {top3:.1f}%.")
    return t,{"label":label,"reason":reason,"top_attention":round(top,1),"top3_attention":round(top3,1)}

def classic_qb_concentration_plan(df, thesis_table, entry_format):
    """Choose how concentrated the Classic QB pool should be from slate evidence.

    WHY THIS EXISTS:
    In Single Entry and 3-Max, the user is not trying to cover every plausible outcome.
    The portfolio should express a smaller number of strongest slate theses. A 50-lineup
    candidate set using 15-20 QBs is useful for exploration, but it is the wrong behavior
    for choosing one or three actual entries because weak, nearly interchangeable QB paths
    survive simply through optimizer randomness.

    IMPORTANT: this is NOT "Single Entry = N quarterbacks." The number is earned by the
    slate. QB support is aggregated from direct QB theses, receiver/TE theses that imply a
    paired QB, and game-environment theses. We then keep quarterbacks while their support
    remains meaningfully close to the best paths. An open slate can therefore keep many;
    a slate with real separation can narrow to one or two.
    """
    qbs=df[df["is_QB"] & df["ActiveForBuild"]][["ID","Name","Team","Matchup","My Proj","My Own"]].copy()
    if qbs.empty:
        return [],pd.DataFrame(),{"label":"No QB pool","reason":"No active quarterbacks were available."}

    support={str(r["Name"]):0.0 for _,r in qbs.iterrows()}
    reasons={str(r["Name"]):[] for _,r in qbs.iterrows()}

    if thesis_table is not None and not thesis_table.empty:
        for _,t in thesis_table.iterrows():
            att=float(t.get("Attention %",0.0))
            typ=str(t.get("Type",""))
            paired=str(t.get("Paired QB","")).strip()
            game=str(t.get("Game","")).strip()
            if paired in support:
                weight=1.0 if typ=="QB" else 0.85
                support[paired]+=att*weight
                reasons[paired].append(f"{typ.lower()} thesis {att:.1f}")
            if typ=="Game" and game:
                game_qbs=qbs[qbs["Matchup"].astype(str).eq(game)]
                if not game_qbs.empty:
                    split=att/len(game_qbs)
                    for nm in game_qbs["Name"].astype(str):
                        support[nm]+=split*0.60
                        reasons[nm].append(f"game thesis {att:.1f}")

    # Projection and leverage provide a floor so a strong QB is not discarded merely
    # because another player on his team generated the named thesis.
    proj=qbs["My Proj"].rank(pct=True,method="average")
    lev=(qbs["My Proj"]/(pd.to_numeric(qbs["My Own"],errors="coerce").fillna(0)+4.0)).rank(pct=True,method="average")
    for idx,r in qbs.iterrows():
        nm=str(r["Name"])
        support[nm]+=12.0*float(proj.loc[idx])+5.0*float(lev.loc[idx])

    rows=[]
    for _,r in qbs.iterrows():
        nm=str(r["Name"])
        rows.append({"ID":str(r["ID"]),"QB":nm,"Team":str(r["Team"]),"Game":str(r["Matchup"]),
                     "Support":float(support[nm]),"Why":" · ".join(reasons[nm][:3]) or "projection/leverage support"})
    tab=pd.DataFrame(rows).sort_values("Support",ascending=False).reset_index(drop=True)
    if tab.empty:
        return [],tab,{"label":"No QB pool","reason":"No quarterback support scores were available."}

    top=float(tab.iloc[0]["Support"])
    tab["Relative %"]=(100.0*tab["Support"]/max(top,1e-9)).round(1)

    # Entry format changes how far down the evidence curve we are willing to go.
    # These are evidence cutoffs, not quarterback-count targets.
    rel_floor={"Single Entry":58.0,"3-Max":48.0,"20-Max":30.0,"150-Max":14.0}.get(entry_format,30.0)
    kept=tab[tab["Relative %"]>=rel_floor].copy()

    # If the slate is extremely flat, avoid a false sense of precision: keep every QB
    # that is effectively tied with the last qualifying path.
    if not kept.empty:
        boundary=float(kept.iloc[-1]["Support"])
        tied=tab[tab["Support"]>=boundary*0.97]
        kept=tied.copy()

    ids=kept["ID"].astype(str).tolist()
    names=kept["QB"].astype(str).tolist()
    label=f"{len(ids)} QB path" if len(ids)==1 else f"{len(ids)} QB paths"
    reason=(f"{entry_format}: Aytia is keeping {label} because each retained QB has at least "
            f"{rel_floor:.0f}% of the top evidence score. The cutoff is based on thesis support, "
            "projection, game environment and leverage—not a preset QB count.")
    return ids,tab,{"label":label,"reason":reason,"names":names,"relative_floor":rel_floor}

def classic_apply_qb_exclusions(qb_ids, excluded_ids):
    """Remove user-excluded QBs from the build pool.

    qb_ids is rank-ordered (best evidence first). excluded_ids is a set of ID
    strings from the Slate Intel QB checkboxes. Never returns an empty pool
    when qb_ids is non-empty: if every QB is excluded, the top-ranked QB is
    kept so the build stays feasible.
    """
    base=[x for x in (qb_ids or [])]
    if not base:
        return []
    excl=set(str(x) for x in (excluded_ids or set()))
    kept=[x for x in base if str(x) not in excl]
    if kept:
        return kept
    return [base[0]]

def classic_apply_qb_cap(qb_ids, qb_table, qb_plan, cap):
    """Apply a user-requested maximum QB pool size without inventing a fixed default."""
    try:
        cap=int(cap or 0)
    except Exception:
        cap=0
    if cap<=0 or qb_table is None or qb_table.empty:
        return qb_ids,qb_table,qb_plan
    ranked=qb_table.copy().reset_index(drop=True)
    ranked_ids=ranked["ID"].astype(str).tolist()
    base_ids=set(str(x) for x in (qb_ids or ranked_ids))
    kept=[pid for pid in ranked_ids if pid in base_ids][:cap]
    if not kept:
        kept=ranked_ids[:cap]
    names=ranked[ranked["ID"].astype(str).isin(set(kept))]["QB"].astype(str).tolist()
    plan=dict(qb_plan or {})
    plan["names"]=names
    plan["user_cap"]=cap
    plan["label"]=f"{len(kept)} QB path" if len(kept)==1 else f"{len(kept)} QB paths"
    plan["reason"]=(f"User instruction: use no more than {cap} quarterbacks. Aytia kept the highest-ranked "
                    f"evidence-backed QB paths inside the existing slate model; this cap stays active until cleared.")
    ranked["In build pool"]=ranked["ID"].astype(str).isin(set(kept))
    return kept,ranked,plan

def classic_contest_recommendations(field_size, payout_style, entry_format, sim_table):
    aggr=contest_aggression(field_size,payout_style)
    aggr=float(np.clip(aggr+{"Single Entry":-0.12,"3-Max":-0.04,"20-Max":0.06,"150-Max":0.14}.get(entry_format,0),0.03,1.0))
    top_share=float(sim_table["Slate ceiling %"].max()) if sim_table is not None and not sim_table.empty else 0.0
    concentrated=top_share>=22.0
    qb_stack=2 if (aggr>=0.28 or concentrated) else 1
    if entry_format=="Single Entry" and field_size<=1000:
        bringback="Required" if concentrated else "Optional"
    elif aggr>=0.68:
        bringback="Optional"
    else:
        bringback="Required"
    min_salary=49000 if aggr<0.22 else 48500 if aggr<0.48 else 47500 if aggr<0.72 else 46500
    return {"aggression":round(aggr,2),"qb_stack":qb_stack,"bringback":bringback,"min_salary":min_salary,
            "max_team":4 if aggr<0.25 else 5,"max_game":5 if concentrated or aggr>=0.45 else 4,"max_te":2,
            "no_dst":True,"no_off":False,"allow_qb_rb":True,"top_game_share":round(top_share,1)}

def classic_portfolio_intelligence(df, result):
    """Deterministic portfolio diagnostics for the conversational DFS strategist."""
    if result is None or not isinstance(result, pd.DataFrame) or result.empty:
        return {"built": False, "lineups": 0, "note": "No Classic portfolio has been generated yet."}

    lineup_cols=[x for x in ROSTER_SLOTS if x in result.columns]
    n=int(len(result))
    name_meta={}
    for _,p in df.iterrows():
        name_meta[str(p["Name"])]={
            "position":str(p.get("Position","")),
            "team":str(p.get("Team","")),
            "game":str(p.get("Matchup","")),
            "field_own":float(pd.to_numeric(pd.Series([p.get("My Own",0)]),errors="coerce").fillna(0).iloc[0]),
            "projection":float(pd.to_numeric(pd.Series([p.get("My Proj",0)]),errors="coerce").fillna(0).iloc[0]),
        }

    counts=defaultdict(int); pair_counts=defaultdict(int); triple_counts=defaultdict(int)
    game_stack_counts=defaultdict(int); qb_counts=defaultdict(int)
    for _,lr in result.iterrows():
        names=[str(lr.get(col,"")) for col in lineup_cols if str(lr.get(col,"")).strip() and str(lr.get(col,""))!="nan"]
        for nm in names: counts[nm]+=1
        if "QB" in result.columns and str(lr.get("QB","")).strip():
            qb_counts[str(lr.get("QB"))]+=1
        sn=sorted(set(names))
        for a in range(len(sn)):
            for b in range(a+1,len(sn)):
                pair_counts[(sn[a],sn[b])]+=1
                for d in range(b+1,len(sn)):
                    triple_counts[(sn[a],sn[b],sn[d])]+=1
        games=defaultdict(int)
        for nm in names:
            g=name_meta.get(nm,{}).get("game","")
            if g: games[g]+=1
        for g,k in games.items():
            if k>=3: game_stack_counts[g]+=1

    exposures=[]
    for nm,cnt in counts.items():
        meta=name_meta.get(nm,{})
        my=100.0*cnt/max(n,1); field=float(meta.get("field_own",0.0))
        exposures.append({
            "player":nm,"position":meta.get("position",""),"team":meta.get("team",""),
            "lineups":int(cnt),"exposure_pct":round(my,1),"field_own_pct":round(field,1),
            "leverage_pct":round(my-field,1),"projection":round(float(meta.get("projection",0.0)),2)
        })
    exposures=sorted(exposures,key=lambda x:(x["exposure_pct"],x["projection"]),reverse=True)
    overweight=sorted(exposures,key=lambda x:x["leverage_pct"],reverse=True)[:8]
    underweight=sorted(exposures,key=lambda x:x["leverage_pct"])[:8]
    qbs=[{"qb":k,"lineups":int(v),"exposure_pct":round(100.0*v/max(n,1),1)} for k,v in sorted(qb_counts.items(),key=lambda z:z[1],reverse=True)]

    stack_mix={}
    if "QB Stack" in result.columns:
        for k,v in result["QB Stack"].value_counts().sort_index().items(): stack_mix[str(int(k))]=int(v)
    # Mobile-QB lineups are designed to run skinny stacks; the stack-structure
    # coach finding below excludes them from the skinny-stack denominator.
    mobile_qb_lineups=int(pd.to_numeric(result["QB Mobile"],errors="coerce").fillna(0).sum()) if "QB Mobile" in result.columns else 0
    bringback_mix={}
    if "Bring-backs" in result.columns:
        for k,v in result["Bring-backs"].value_counts().sort_index().items(): bringback_mix[str(int(k))]=int(v)

    def top_cores(source,size,limit=8):
        out=[]
        for core,cnt in sorted(source.items(),key=lambda z:z[1],reverse=True)[:limit]:
            out.append({"players":list(core),"lineups":int(cnt),"portfolio_pct":round(100.0*cnt/max(n,1),1)})
        return out

    salary_left=pd.to_numeric(result.get("Salary Left",pd.Series(dtype=float)),errors="coerce").dropna()
    avg_own=pd.to_numeric(result.get("Avg Own",pd.Series(dtype=float)),errors="coerce").dropna()
    proj=pd.to_numeric(result.get("Projection",pd.Series(dtype=float)),errors="coerce").dropna()

    top_lineups=[]
    for _,lr in result.head(8).iterrows():
        top_lineups.append({
            "rank":int(lr.get("Rank",0) or 0),"rating":str(lr.get("Rating","")),
            "projection":round(float(lr.get("Projection",0) or 0),2),
            "salary":int(lr.get("Salary",0) or 0),"salary_left":int(lr.get("Salary Left",0) or 0),
            "avg_player_own_pct":round(float(lr.get("Avg Own",0) or 0),1),
            "stack":str(lr.get("Stack Summary","")),
            "players":[str(lr.get(col,"")) for col in lineup_cols]
        })

    flex_mix={}
    if "FLEX" in result.columns:
        for nm in result["FLEX"].dropna().astype(str):
            pos=name_meta.get(nm,{}).get("position","")
            if pos:
                flex_mix[pos]=flex_mix.get(pos,0)+1
    flex_mix_pct={k:round(100.0*v/max(n,1),1) for k,v in flex_mix.items()}

    return {
        "built":True,"lineups":n,"unique_qbs":len(qbs),"qb_usage":qbs,
        "stack_mix":stack_mix,"bringback_mix":bringback_mix,"flex_mix_pct":flex_mix_pct,"mobile_qb_lineups":mobile_qb_lineups,
        "salary_left":{"mean":round(float(salary_left.mean()),1) if len(salary_left) else None,
                       "median":round(float(salary_left.median()),1) if len(salary_left) else None,
                       "max":int(salary_left.max()) if len(salary_left) else None},
        "projection":{"mean":round(float(proj.mean()),2) if len(proj) else None,
                      "max":round(float(proj.max()),2) if len(proj) else None,
                      "min":round(float(proj.min()),2) if len(proj) else None},
        "avg_player_ownership":{"mean":round(float(avg_own.mean()),1) if len(avg_own) else None},
        "top_exposures":exposures[:15],"most_overweight":overweight,"most_underweight":underweight,
        "top_three_plus_game_exposure":[{"game":g,"lineups":int(v),"portfolio_pct":round(100.0*v/max(n,1),1)}
                                        for g,v in sorted(game_stack_counts.items(),key=lambda z:z[1],reverse=True)[:10]],
        "repeated_pairs":top_cores(pair_counts,2),"repeated_triples":top_cores(triple_counts,3),
        "top_lineups":top_lineups,
    }

def classic_postbuild_report(packet):
    """Deterministic post-build coach: concise contest-aware findings from the actual portfolio."""
    p=packet.get("portfolio",{}) or {}
    contest=packet.get("contest",{}) or {}
    rules=packet.get("active_rules",{}) or {}
    if not p.get("built"):
        return []
    entry=str(contest.get("entry_format",""))
    field=int(contest.get("field_size",0) or 0)
    payout=str(contest.get("payout",""))
    findings=[]

    # QB concentration.
    uq=int(p.get("unique_qbs",0) or 0)
    n=int(p.get("lineups",0) or 0)
    if entry=="Single Entry" and uq>4:
        findings.append(("QB concentration",
            f"The candidate portfolio uses {uq} QBs across {n} lineups. For Single Entry, this is research breadth rather than a recommendation to spread your one final entry across many QB ideas."))
    elif entry=="20-Max" and uq>10:
        findings.append(("QB concentration",
            f"{uq} QBs across {n} lineups is broad for 20-Max. The portfolio may be spreading conviction too thin unless the slate is unusually flat."))

    # Stack structure. Mobile-QB lineups run skinny stacks by design (rushing
    # is the correlation), so the skinny-stack check excludes them.
    sm={str(k):int(v) for k,v in (p.get("stack_mix",{}) or {}).items()}
    singles=sm.get("1",0); doubles=sm.get("2",0); triples=sm.get("3",0)
    n_mobile=int(p.get("mobile_qb_lineups",0) or 0)
    skinny_eligible=max(n-n_mobile,0)
    if skinny_eligible:
        if doubles/skinny_eligible < .20:
            _mob_note=f" ({n_mobile} mobile-QB lineups run skinny by design and are excluded)" if n_mobile else ""
            _dbl_txt=f"None of {skinny_eligible}" if doubles==0 else f"Only {doubles} of {skinny_eligible}"
            findings.append(("Stack structure",
                f"{_dbl_txt} non-mobile-QB lineups are QB+2 builds{_mob_note}. That is not automatically wrong, but Aytia should verify that skinny stacks are being chosen because the second pass catcher is weak—not just because the optimizer prefers median projection."))
        elif doubles/n > .70:
            findings.append(("Stack structure",
                f"{doubles} of {n} lineups are QB+2 builds. That is a concentrated construction bet; make sure the slate actually has enough condensed passing offenses to justify it."))

    # Bringbacks. Skip the note entirely when the user turned bring-backs off —
    # scolding the portfolio for following an explicit "None" setting is noise.
    bm={str(k):int(v) for k,v in (p.get("bringback_mix",{}) or {}).items()}
    no_bb=bm.get("0",0)
    if str(rules.get("bringback",""))!="None" and n and no_bb/n>.65:
        findings.append(("Bring-backs",
            f"{no_bb} of {n} lineups have no bring-back. That can be correct on a soft-pricing slate, but Aytia should make sure those games can still reach their stack ceiling without being pushed."))

    # Repeated cores.
    pairs=p.get("repeated_pairs",[]) or []
    if pairs:
        top=pairs[0]
        pct=float(top.get("portfolio_pct",0) or 0)
        if pct>=25:
            findings.append(("Repeated core",
                f"{' + '.join(top.get('players',[]))} appears together in {pct:.0f}% of the portfolio. That is meaningful concentration and should represent a deliberate slate thesis, not an accidental optimizer habit."))

    # FLEX construction.
    fm=p.get("flex_mix_pct",{}) or {}
    if fm:
        wr=float(fm.get("WR",0) or 0); rb=float(fm.get("RB",0) or 0); te=float(fm.get("TE",0) or 0)
        findings.append(("FLEX construction",
            f"FLEX mix is WR {wr:.0f}% / RB {rb:.0f}% / TE {te:.0f}%. Construction leverage matters here: the strongest choice is the one with tournament ceiling, not simply the lowest-owned position."))

    # Contest lens.
    if entry=="Single Entry" and "winner" in payout.lower():
        findings.append(("Winner-take-all lens",
            f"In a {field:,}-entry Single Entry winner-take-all, the goal is one coherent first-place path. You can eat strong chalk, but the final lineup should have at least one meaningful source of leverage through its exact stack, direct leverage, or roster construction—not random low ownership."))

    return findings[:6]

_DFS_PRO_PLAYBOOK = """DFS PRO PLAYBOOK:
- Contest size changes strategy. A 200-person Single Entry should not be built like a 150,000-entry GPP.
- Low-entry contests should take stands when evidence separates. Large portfolios can spread more.
- In 20-Max, do not scatter across 10-12+ quarterbacks just because they are viable. Concentrate enough
  that each QB path gets multiple correlated combinations around it. The exact QB count should be earned
  by the slate, not fixed in advance.
- Quarterback ownership is usually naturally spread. Do not fade a QB you like solely because of ownership
  unless the ownership is truly extreme relative to alternatives.
- Game stacks matter most when offensive production is concentrated among a few players. A high-scoring
  game with points spread across many players can still disappoint for DFS stacking.
- Popular games can still be played, but if using the most popular pieces together, find leverage elsewhere
  or use a less common construction within that game.
- Run-backs are contextual, not mandatory. Some stacks should be brought back; others can stand alone if
  the opposing piece is weak or the offense can score without being pushed.
- Cheap chalk is not automatically bad. If a cheap popular player unlocks multiple true ceiling plays,
  the construction can still be strong; judge the whole lineup, not the isolated ownership.
- One-off plays are allowed. Do not force every player into a correlation rule if a strong standalone play
  improves ceiling and lineup construction.
- Consider team tendencies and game scripts: some offenses keep throwing with a lead, others may shut down
  and lean on the run. Use that to judge whether a stack needs an opponent bring-back.
- Manual projection changes are a valid way to express conviction when the user believes one player should
  project materially better or worse than the model.
- In very large fields, identify where the lineup is different. Playing a popular core is fine if another
  part of the construction creates enough leverage or uniqueness.
- Millie Maker / massive-field, top-heavy contests require more ceiling and uniqueness than Single Entry or 3-Max.
  In those fields, avoid sacrificing too much ceiling just to be different; every roster spot should have a credible path
  to a slate-winning score.
- Balanced builds can be strong when pricing is soft because they avoid weak punts while preserving ceiling at every spot.
  Stars-and-scrubs is not automatically superior just because cheap value exists.
- Cheap chalk should be judged by what it unlocks. A cheap tight end who scores only 6-8 points can still be part of a
  tournament-winning lineup if the salary savings create extra 30-point ceiling pieces elsewhere.
- Do not force uniqueness by leaving salary unused on large Classic slates. Salary left is a secondary concern; lineup
  quality, ceiling, correlation and ownership structure matter more. Showdown is different.
- Prefer concentrated workloads at running back when comparing similar projections: backs with 70-90% roles have stronger
  ceiling/floor cases than similarly priced backs in true committees.
- When a popular player is a very strong projection, it can be correct to eat the chalk and find differentiation elsewhere.
  Fading a strong chalk play should have an actual game-script or ownership-based reason, not contrarianism for its own sake.
- If fading the most obvious player from a high-powered offense, consider leveraging that stance with other pieces from the
  same offense whose success directly benefits from the chalk player's failure.
- On large fields, stack decisions should consider whether a double stack requires an unusually large QB ceiling. Expensive
  double stacks can be less attractive if both pass catchers need a near-perfect quarterback outcome to pay off.
- Flex construction is slate-dependent. Three-RB and four-WR builds can both be correct; do not hard-code one roster shape
  across every slate.
- For very low-owned elite quarterbacks, talent and ceiling can justify overweight exposure even in difficult matchups,
  especially in 150-Max. Low ownership does not automatically mean good leverage, though; compare projected ownership with
  simulated optimal or ceiling rates when available.
- Defense can be eaten as chalk if the matchup, pressure/turnover environment and salary make it clearly superior, but
  defensive variance is high, so overexposure should still be deliberate.
- Do not confuse 'popular game' with 'must fade.' A popular game can still deserve exposure if it is condensed and has
  multiple slate-breaking pieces; use different combinations, lower-owned secondary pieces, or alternate constructions
  rather than reflexively avoiding it.
- When late injury news creates mispriced backups, distinguish between median value and tournament ceiling. A cheap player
  who projects well but lacks a plausible 20+ point path may be better for cash than for a massive-field GPP.
- When two strong plays are mutually dependent, assess whether their success stories conflict. Example: a rushing QB paired
  with a non-pass-catching RB can be negatively correlated and may deserve a rule or reduced pairing rate.
- Inputs drive outputs. Treat projections, ownership, player-pool exclusions, exposure targets and stack rules
  as the user's actual DFS opinions; do not assume the optimizer can rescue weak or incoherent inputs.
- Portfolio diversification should be based on independent paths to first place, not a mechanical minimum-uniques rule.
  Two lineups can differ by two players yet tell the same story; conversely, similar cores can still represent meaningfully
  different game environments or leverage points.
- On soft-pricing slates, assume many lineups will look superficially strong. Evaluate whether each lineup has enough actual
  ceiling and a coherent path to beating other strong-looking lineups rather than rewarding raw projection alone.
- In large fields, low-owned plays should have genuine slate-breaking upside. A player projected for a decent median but
  with little chance to reach roughly 18-25+ DraftKings points may not be a useful tournament differentiator.
- Secondary correlations are valuable and do not require a quarterback. Examples include RB+opposing WR/TE, RB+DEF,
  or skill-player pairings from the same game when their paths to ceiling are compatible.
- A chalk player can be used differently by changing the players around him. Do not evaluate ownership one player at a time;
  judge the ownership and correlation of the full combination.
- Overstacking a high-total game can be a legitimate large-field strategy when the lineup is explicitly betting on that game
  materially exceeding expectations. Do not reject 5-6 player game environments solely because they are unconventional.
- Embrace uncertainty when the payoff is asymmetric. Week 1 roles, rookies and changing depth charts can create low-owned
  players whose true workload is wider than the projection assumes; uncertainty itself can create tournament leverage.
- Early-season slates deserve wider scenario ranges. In Weeks 1-3, preseason usage, new coordinators, rookie roles and
  incomplete depth-chart information make median projections less trustworthy than later in the year; increase scenario breadth rather
  than pretending the inputs are equally certain.
- Coaching continuity is a real contextual input. Offenses returning the same quarterback, line, skill core and system can deserve a
  narrower uncertainty band than teams installing new schemes, changing play callers or replacing multiple starters.
- New coaching staffs can create both risk and leverage. When the market is anchored to last year's usage, test alternate target shares,
  pace, pass rate and red-zone roles that fit the new coach's historical tendencies instead of assuming last year's distribution persists.
- Offensive-line quality and specific line mismatches should influence DFS ceilings. Strong pass rush versus a weak or injured offensive
  line can lower a quarterback stack's clean-pocket ceiling while increasing opposing DST upside, sack/turnover paths and short-field scoring.
- Treat major offensive-line injuries as team-level changes, not just small player projection downgrades. They can alter pace, route depth,
  pressure rate, rushing efficiency and play-calling, changing the viability of an entire game stack.
- When injuries or trades remove receiving options, explicitly reallocate target concentration before merely boosting every remaining player.
  Fewer available weapons can make an offense more DFS-friendly if volume funnels toward one or two players.
- Distinguish uncertainty about talent from uncertainty about opportunity. A player can be hard to evaluate as an NFL talent but still be
  attractive in DFS if his role, salary and concentration are favorable; conversely, a talented player in a diffuse role may remain a poor tournament bet.
- Do not import betting heuristics or historical trend slogans directly into DFS. Home/road, division, prime-time and similar labels matter only
  through mechanisms already affecting projection, ownership, pace, weather, matchup or role; avoid double-counting narratives already embedded in inputs.
- Market lines are useful context, not standalone DFS answers. Use spread, total and line movement to shape likely game scripts, but require a
  football explanation for how that script creates fantasy production and which players capture it.
- Separate real-football team strength from DFS usefulness. A team can be a strong favorite yet produce a poor tournament stack if scoring is
  diffuse, pace is slow or the opponent cannot push; an underdog can still be DFS-viable when concentrated volume and pass-heavy comeback paths create ceiling.
- When a team is expected to trail, model whether the quarterback and receivers benefit from increased dropbacks rather than automatically
  downgrading the entire offense. Conversely, large favorites may increase RB and DST correlation while reducing full-game stack appeal.
- Avoid overconfidence in tiny samples from preseason or the prior season. Use them as evidence about role and scheme, but weight current
  depth chart, coaching changes, injuries and market expectations more heavily than one or two recent games.
- Week 2 is a recency-bias trap. Do not automatically crown Week 1 breakouts or bury Week 1 failures. Separate what changed in role
  (snaps, routes, targets, carries, red-zone work, designed QB runs) from what was merely one-game scoring variance.
- Role evidence should update faster than box-score efficiency. A player who earned a large snap share, route share or target share but failed
  can be a stronger tournament buy-back than a player who scored efficiently on limited opportunity.
- When a starter is removed, identify the actual beneficiary by role. Reallocate routes, targets, carries, goal-line work and pass-protection snaps
  separately instead of assuming one backup inherits the entire workload.
- Running-back snap share alone can mislead. Distinguish early-down rushing work, third-down/pass-protection work, targets and goal-line usage;
  a 50/50 snap split can still hide a much more valuable fantasy role for one back.
- Pass protection can cap a running back's routes and two-minute work even when he leads carries. Treat protection trust as a workload constraint,
  especially for young or recently activated backs.
- Cheap quarterbacks create lineup-construction leverage only when they retain a realistic tournament ceiling. Salary savings are valuable because
  they can buy additional ceiling elsewhere, but a cheap QB projecting 18-20 points is not automatically superior to a premium QB with 30-point upside.
- Buy-back candidates are especially useful when the field overreacts to one bad game but the underlying role, matchup mechanism and ceiling remain intact.
  Do not fade a strong player merely because he disappointed at high ownership the previous week.
- When quarterback ownership is broadly distributed, low ownership by itself is weak leverage. Prefer quarterbacks whose exact stacks are under-owned,
  whose rushing creates standalone ceiling, or whose salary unlocks materially stronger roster construction.
- Evaluate quarterback ceiling through touchdown pathways. When an offense loses or lacks a dominant goal-line running back, passing-touchdown share can rise;
  when an elite rushing QB owns red-zone equity, his ceiling can remain high even if pass-catcher concentration is uncertain.
- Exact stack correlation matters more than generic team exposure. Some receivers' ceiling games are tightly tied to a quarterback eruption, while high-volume
  receivers can post strong scores even when the QB has only an ordinary fantasy day. Use route depth, target concentration and touchdown dependence to distinguish them.
- WR3/secondary-receiver exposure becomes more interesting when explicitly betting on a game exceeding expectations. In a true shootout, ancillary players can
  capture the extra touchdowns that make the game environment beat its median; do not use them merely because they are cheap.
- When ownership is flat across a position, the slate may offer more freedom to prioritize projection, role and correlation instead of forcing low-owned plays.
  Game theory matters most where the field is actually concentrated.
- Defensive selection should emphasize opponent pressure rate, sack susceptibility, turnover risk, offensive-line weakness and likely negative game script.
  Repeatedly targeting mistake-prone quarterbacks can be rational, but do not convert a small sample into an unconditional rule.
- Leverage is not simply low ownership. Start by identifying the field's most common roster shell: which positions absorb salary, which chalk pieces are commonly paired,
  and which roster constructions repeat. Attack the shell when a different but football-coherent construction has comparable ceiling.
- Direct leverage is strongest when one player's success naturally reduces a popular player's ceiling. Examples: a WR/TE capturing touchdowns instead of a chalk RB,
  a passing stack succeeding instead of a popular same-team runner, or an alternative skill player inheriting concentrated targets. Prefer this to unrelated low-owned darts.
- Do not judge a fade only by whether the pivot scores well. Ask whether the pivot can beat the chalk even when the chalk has a strong-but-not-perfect game.
  Expensive elite chalk may remain difficult to fade if a 28-30 point outcome still beats most alternatives.
- Treat combinatorial ownership as more important than isolated player ownership. A moderately popular QB and moderately popular WR can still form an uncommon double stack;
  conversely several individually acceptable chalk pieces can create a very duplicated lineup shell when combined.
- Roster-construction leverage can be cleaner than player-level leverage. If the field is concentrated on RB or TE in FLEX, a high-ceiling WR FLEX can be a strong tournament
  construction when the slate supports it. Do not force the opposite of the field; require ceiling and a plausible football story.
- Tight-end leverage is especially powerful when the field is overconfident in cheap median outcomes. A chalk TE scoring 8-12 points may not hurt directly, but a lower-owned
  TE reaching 20-25 can create real positional separation. Compare ceiling, price savings and what each construction unlocks.
- Soft early-season pricing changes how aggressively correlation should be forced. Preserve strong stacks, but do not force a weak bring-back or secondary player when a
  materially better one-off is available. The question is whether the omitted opponent piece is necessary for the game script to keep producing.
- Double stacks are useful because they increase the chance of capturing the pass catcher who posts the difference-making score; both receivers do not need to hit their ceiling.
  This can be particularly valuable in smaller fields where a 10-15 point secondary piece can survive beside one slate-breaking teammate.
- For large-field MME, diversify stack structures rather than applying one correlation template to every lineup. It can be correct to remain overweight on double-stack/bring-back
  constructions while allowing a minority of strong lineups to use skinny stacks or no bring-back when the specific scenario supports it.
- Week 2 leverage should explicitly test the field's Week 1 overconfidence. Carry forward role evidence quickly, but be skeptical that touchdowns, efficiency, team scoring,
  defensive performance or last week's winning roster construction will repeat unchanged.
- When evaluating FLEX, compare median projection with ceiling and field construction. Running backs may lead median projections while wide receivers close the gap substantially
  in ceiling; tournament decisions should reward the latter when the field overweights median outcomes.
- Build leverage lineups around a sentence: what popular assumption is wrong, what succeeds instead, and how does this lineup get paid if that scenario occurs?
  A contrarian lineup without that causal story is not automatically a good tournament lineup.
- Historical split trends such as home tight-end performance may be used as a weak prior only. Do not hard-code them without a football mechanism, sufficient sample,
  and current role/matchup support; avoid double-counting trends already reflected in projections or market inputs.
- Contest simulation output is only as reliable as its projections, ownership assumptions, field-generation model and correlation logic. Treat simulated EV as one
  diagnostic rather than truth, and compare it with simpler lineup-level measures such as projection, ceiling, ownership concentration, correlation and duplication risk.
- For Classic GPPs, median projection is not the target. Treat projection as the center of a player's range and explicitly estimate ceiling, tail probability and the game conditions
  that create those ceiling outcomes. A 12-point median with a credible 28-point branch can be more tournament-relevant than a higher-median player with a narrow range.
- Upside should be modeled as conditional and clustered, not independent. When a receiver reaches a slate-breaking outcome, the quarterback, opposing pass catcher, teammate or game environment
  that enabled it often has an elevated chance to hit too. Build around correlated stories rather than collecting unrelated ceiling players.
- Each Classic lineup should be explainable as one version of Sunday: which game(s) exceed expectation, which chalk fails or succeeds, where salary value emerges, and which correlated pieces
  benefit together. If Aytia cannot tell that story, the lineup is probably an optimizer artifact rather than an intentional tournament build.
- Stacking is a consequence of the game script, not an end in itself. QB+1, QB+2, bringbacks, mini-correlations and even occasional naked rushing-QB lineups should appear at rates supported by
  scenario outcomes rather than being forced uniformly across the portfolio.
- Field ownership is the price of an outcome, not a reason to fade it automatically. A popular player can still be correct when his ceiling probability justifies the ownership; a lower-owned
  player is valuable only when his upside and the game story behind it can actually move the lineup toward first place.
- Track leverage explicitly as portfolio exposure minus projected field ownership, but interpret it in context. Positive leverage on a weak ceiling play is not useful; negative leverage on
  elite chalk can be acceptable when the rest of the lineup differentiates intelligently.
- Portfolio diversification should spread entries across genuinely different versions of Sunday, not just different player combinations. A portfolio that has 20 unique lineups but depends on
  the same two games and the same chalk assumptions is still fragile.
- User opinions should be translated into portfolio-level instructions. If the user says 'less of this chalk,' 'more Cincinnati stacks,' or 'I think this game shoots out,' Aytia should
  rebuild all affected lineups coherently instead of requiring the user to micromanage every replacement player.
- When the user supplies a partial lineup or locks two, three or four players, treat that as a thesis seed. Complete the remaining roster spots using the best correlated and contest-appropriate
  complements for that specific story rather than simply choosing the highest remaining projections.
- Classic lineup evaluation should include ceiling, correlation, leverage, repeated-core concentration, salary efficiency and game-story coherence in addition to median projection.
  High projected points alone should never be sufficient for an A-grade tournament lineup.
- Late news is part of the slate, not an exception. When an inactive or role change appears after early games lock, preserve already-locked players and existing correlations while rebuilding
  unlocked roster spots around the new information, using any newly freed salary intelligently.
- Late-swap logic should distinguish between survival and tournament leverage. Removing a zero is mandatory; beyond that, a trailing lineup may rationally become more contrarian while a lineup
  already performing well may preserve projection and correlation rather than taking unnecessary risk.
- Post-slate review should separate process from outcome. The one Sunday that happened is only one branch; evaluate whether the pre-lock portfolio had strong ceiling coverage, sensible ownership
  leverage, coherent game scripts and reasonable exposure concentration even when variance produced a losing result.
- When actual contest ownership, duplication and winning constructions are available after lock, compare them with Aytia's projections. Use those errors to recalibrate field modeling,
  ownership assumptions, leverage estimates and portfolio construction instead of overreacting to individual player results.
- Main-slate scenario generation should eventually simulate complete correlated slate worlds rather than independently perturbing player projections. The target architecture is: simulate each
  game's scoring/volume paths, combine games into full-slate worlds, solve the best lineup for each world, then diversify the final portfolio across those distinct worlds.
- Contest-simulation ROI should remain a secondary model output rather than an oracle. Its usefulness depends on the quality of the field model and ownership assumptions; always show the user
  simpler evidence alongside it so Aytia can explain why a lineup is attractive without hiding behind one EV number.
- Tournament profitability is dominated by rare top-end finishes, so Aytia should never judge a process by how often it min-cashes or by the result of one slate. Favor first-place equity
  and repeatable decision quality over lineups engineered merely to finish above the cash line.
- Treat variance as a normal property of DFS, not evidence that a sound process is broken. When reviewing a losing slate, ask whether the portfolio created enough credible top-1% paths before
  asking whether a specific player or rule should be changed.
- Contest context determines the correct tradeoff among three competing levers: projection, ownership and correlation. Projection is the base; correlation amplifies ceiling when a story is right;
  ownership determines how much relative movement that success creates against the field.
- Do not optimize those three levers with one universal formula. Larger and more top-heavy contests justify sacrificing more median projection for stronger correlation, lower-owned ceiling and
  uniqueness, while smaller or flatter contests should generally preserve more projection and require less aggressive differentiation.
- Whenever Aytia recommends a lineup or portfolio change, explain the price being paid across those three levers. For example: 'This lowers median projection by 4 points, improves field leverage
  by 18 percentage points and creates a stronger two-player game correlation.' Make the tradeoff visible instead of hiding it in a grade.
- A portfolio should be treated as a set of investments in different slate outcomes. One hundred fifty technically unique lineups built around the same core thesis are not diversified; they are
  one concentrated bet expressed 150 ways.
- Avoid defaulting to a rigid core-player philosophy for large portfolios. Concentration is appropriate when evidence strongly supports it, especially in low-entry formats, but 20-Max and 150-Max
  builds should usually cover a wider set of high-quality slate stories rather than rotating peripheral players around the same small core.
- Diversification should happen after candidate lineup quality is established, not by crippling the optimizer during lineup creation. First generate a large pool of strong, coherent candidates;
  then select a diversified subset from that pool for the requested number of entries.
- Min-unique rules are a portfolio-selection tool, not a primary lineup-quality engine. Applying aggressive uniques while lineups are being generated can force later lineups into bad combinations;
  prefer scoring a deep candidate pool first and then selecting entries that maximize both quality and meaningful scenario separation.
- Exposure caps are also blunt instruments. Use them when the user has a real conviction or risk limit, but do not rely on arbitrary global caps as the main way to create diversity. Prefer
  scenario-aware diversification, conditional correlations and post-generation portfolio selection.
- When selecting a final portfolio from a candidate pool, measure similarity at multiple levels: shared players, shared quarterback/game thesis, repeated two- and three-player cores, ownership
  profile and scenario dependence. Two lineups that differ by three players can still be the same bet if they require the same game to shoot out and the same chalk to fail.
- Portfolio quality should include a concentration-risk report. Surface how much of the portfolio dies if a heavily used player fails, if one game disappoints, or if one ownership assumption is
  badly wrong. This gives the user an investment-style view of where their entries are fragile.
- Aytia should distinguish low-entry conviction from MME diversification. Single Entry and 3-Max can intentionally concentrate around the strongest thesis; 20-Max and especially 150-Max should
  generally widen scenario coverage while preserving lineup quality.
- The assistant should reinforce realistic outcome expectations without using short-term wins or losses as proof of skill. A good answer should separate process metrics from bankroll outcomes and
  avoid encouraging the user to chase losses, increase stakes after a bad slate, or overreact to one result.
- If bankroll features are ever added, model them as risk controls rather than lineup-selection signals. Contest volume and stake sizing should be handled separately from player and lineup quality.
- Showdown is a game-script problem before it is a six-player optimization problem. Each lineup should correspond to a coherent version of how the single game unfolds:
  shootout, favorite blowout, underdog comeback, low-scoring grind, rushing control, defensive/field-goal game, or another explicit scenario.
- Do not force generic Showdown rules into 100% of lineups. QB-captain plus pass-catcher, WR-captain plus QB, no two same-team RBs, 3-3, 4-2 and 5-1 can all be correct in some
  game states and wrong in others. Let the game script determine when a construction belongs rather than assuming a universal rule.
- In Showdown, the objective is not merely to find a high-projection lineup. The lineup must have a plausible path to being the actual optimal six for a specific game outcome.
- Captain decisions should be evaluated by conditional game stories. Ask what has to happen for this player to be the optimal captain, which teammates benefit from that same story,
  and which expensive teammates may fail to justify salary in that branch.
- Showdown portfolio construction should diversify across independent game scripts rather than mechanically changing one or two players. Twenty lineups that all require the same
  shootout path are not meaningfully diversified even if every lineup is technically unique.
- Duplication is materially more important in Showdown than on large Classic slates because the player pool is tiny and top prizes can be split many ways. Evaluate expected payout
  using both the chance a lineup is optimal and the likelihood that many opponents reach the same construction.
- Being correct about the game but duplicated hundreds of times can still be a poor tournament result. Salary left, unusual captain choices and less common player combinations can
  matter when they reduce duplication without destroying the lineup's football story.
- Do not chase uniqueness for its own sake. A unique lineup still needs a realistic route to first place. Prefer differentiation that is caused by a coherent alternative game script,
  role assumption, touchdown allocation or captain thesis.
- When changing a captain exposure or fading a popular captain, rebuild the rest of the portfolio around that opinion instead of swapping the captain into otherwise unrelated lineups.
  The surrounding five players should be the best complements to the new captain thesis.
- User takes should act like head-coach directions, not manual chores. If the user says 'less Puka captain, more Adams captain' or 'I think this is a defensive grind,' Aytia should
  translate that into the correlated lineup branches, exposure shifts and construction changes that logically follow.
- When the user specifies two, three or four players they want together in Showdown, treat them as the beginning of a game story and autocomplete the remaining spots with the strongest
  correlated complements rather than merely filling salary by projection.
- Showdown feedback should quantify the cost of a user's take. Compare the adjusted portfolio with the baseline in projection, ceiling, leverage, duplication risk, construction mix and
  scenario coverage so the user can decide whether the conviction is worth the tradeoff.
- Evaluate 3-3, 4-2 and 5-1 as outcomes, not quotas. A blowout can naturally create 5-1; a competitive shootout may favor 3-3; a controlled favorite win may create 4-2. Construction mix
  should emerge from the scenario distribution and user beliefs rather than fixed target percentages whenever the simulator has enough evidence.
- Kicker and DST inclusion should also be scenario-driven. Low totals, stalled drives, sacks, turnovers and field-position games increase their optimal paths; aggressive shootouts and
  concentrated touchdown environments reduce them.
- A QB does not always need his highest-priced receiver in Showdown. Touchdowns can distribute across secondary receivers, tight ends or rushing scores, so conditional simulations should
  preserve those lower-frequency but valid branches instead of deleting them with hard pairing rules.
- Process evaluation must be separated from short-term bankroll results. Showdown wins are rare and top-heavy; review whether lineups had strong simulated/estimated optimality, payout-adjusted
  value, duplication profile and scenario coverage even when the actual game did not cooperate.
- Post-slate analysis is strongest when actual contest ownership and duplication are available. Compare projected field behavior with the real field, identify where Aytia misestimated
  ownership or duplication, and use that error to recalibrate future Showdown builds rather than judging only by finishing position.
- When a player's role is uncertain, favor ceiling-aware scenario analysis over blindly trusting a single median projection.
  Ask what happens if the player earns 70-80% of the work rather than the market's assumed 50-60%, and compare that outcome
  with ownership.
- Narrowing a QB pool can be more valuable than spreading across every viable passer. Quarterback is naturally distributed,
  so portfolio conviction can come from concentrating on a smaller number of strong QB theses while diversifying their stacks.
- Weather should be interpreted by mechanism. Rain alone is not an automatic passing-game downgrade; wind, field conditions,
  timing and whether the weather persists through the game matter more. Do not overreact to generic precipitation labels.
- Player props can be used as market evidence, especially for uncertain workloads, but they are not ceilings. A modest median
  rushing or receiving prop can coexist with a tournament-winning tail outcome.
- Defense plus running back is a useful positive-correlation construction when the lineup is betting on a team controlling
  the game, creating sacks/turnovers and producing rushing volume. It should be available as a portfolio thesis rather than
  forced universally.
- When a popular RB can fail because his team's scoring shifts through the air, the QB/pass-catcher stack can be direct leverage.
  The inverse is also true: a lower-owned RB can leverage popular passing stacks from the same offense.
- Roster construction itself is a leverage point. When the field is concentrated on the same cheap RBs, deliberately test
  alternate constructions such as one cheap RB, two pay-up RBs, WR/TE in FLEX, or balanced midrange builds when they still preserve ceiling.
- Do not confuse projected team points with DFS usefulness. A high team total is most actionable when scoring is concentrated enough to
  identify where the touchdowns and volume are likely to land; diffuse offenses can be strong real-life spots but poor stacking targets.
- When a team carries an unusually high implied total, ask where those points come from. If fading the obvious chalk piece, create
  correlated alternatives that capture the same team scoring through the quarterback, pass catchers, or complementary RB/DEF paths.
- Cheap volume can be strong chalk. A low-salary starter with a near-workhorse role and strong touchdown equity can remain viable even
  at heavy ownership because the opportunity-cost penalty for failure is smaller than with an expensive chalk player.
- Evaluate cheap RBs by workload certainty, not salary alone. Distinguish true workhorse injury replacements from committee backs whose
  pass protection, goal-line role, or backup involvement can cap their ceiling.
- On slates with many viable offenses, a concentrated offense can be more valuable than a higher-total but highly distributed offense.
  Prefer stacks where the likely fantasy production can be captured through a manageable number of players.
- If an offense is difficult to stack because targets are distributed, a naked rushing quarterback can be viable when his own rushing
  equity is a major part of the ceiling and no pass catcher is required for him to post a slate-winning score.
- A game can be attractive because it is cheap, not only because its median total is high. Cheap correlated stacks can unlock elite
  one-offs or secondary stacks elsewhere and create a stronger whole-lineup ceiling profile.
- When multiple lineups are available, allocate enough combinations to a high-upside offense to cover its meaningful scoring branches.
  If a team can score through WR1, WR2, TE and RB, one token lineup may be insufficient; either commit enough portfolio resources to
  explore the branches or reduce exposure to the offense.
- In small-entry formats, avoid pretending to cover every attractive game. When many spots look good, concentration should come from
  choosing the few game environments or offenses with the clearest combination of ceiling, concentration and leverage.
- Ownership at quarterback should be evaluated through the exact stack, not the QB alone. A 5% QB paired with the same two obvious
  receivers as the field may still produce a common construction; a more unusual correlated combination can create the real leverage.
- Opposing-player stacks do not always need the quarterback from the same game. A premium QB can be paired with one or two opposing
  pass catchers if those players are the most likely pieces to force the shootout while the QB can score through rushing or dispersed passing.
- Late-window allocation is not mandatory. Do not force afternoon players solely to preserve a sweat, but when projected value is close,
  prefer constructions that preserve meaningful late-swap optionality.
- After early games begin, ownership information itself becomes actionable. Compare actual early ownership with projections before
  choosing late pivots; a later star whose early-position alternative came in more popular than expected can become more valuable leverage.
- Paying up at defense can be correct on slates with several strong defensive mismatches and weak cheap alternatives. Defense should still
  be judged by pressure, turnover potential, opponent protection and game script rather than salary prestige.
- A low-owned expensive RB can be useful leverage against concentrated cheap-RB roster construction when his workload and touchdown ceiling
  remain elite. Compare what the spend-up sacrifices elsewhere with how much unique ceiling it adds.
- When a popular cheap player fails, the most valuable leverage may be the correlated teammate who benefits from that failure, not a random
  low-owned replacement from another game.
- Use market movement as evidence, not gospel. Material moves in game total, team total, props or availability can strengthen a thesis,
  but the program should still explain the football mechanism behind the move.
- Separate prediction from game theory. A player can look mediocre by film or recent box scores yet remain a valid DFS play if salary,
  opportunity, ownership and scoring environment create positive tournament value.
- Late-swap flexibility has value before games begin. When practical, place later-starting players in FLEX and preserve salary/
  positional paths so underperforming early lineups can pivot to lower-owned ceiling plays without unnecessary dead ends.
- Late swap matters when live results are available: lineups doing well can move toward safer/chalkier paths;
  lineups behind can pivot toward lower-owned ceiling outcomes.

"""


def classic_postbuild_llm_answer(question, packet, lineups_ctx, history=None):
    """Conversational post-build coach backed by the OpenAI model.

    Returns the model's answer, or None when no API key is configured or the
    call fails. The caller falls back to the local template responder on None,
    so this function never leaves the user without an answer.
    """
    q = str(question or "").strip()
    if not q:
        return None
    try:
        from openai import OpenAI
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        instructions = (
            "You are Aytia's post-build coach, an NFL DraftKings tournament strategy assistant. "
            "The build step is done: you are reviewing a portfolio of lineups Aytia actually built. "
            "Answer the user's EXACT question first, in direct conversational language. "
            "Ground EVERY claim in the supplied PORTFOLIO DATA (lineup ranks, rosters, projections, ceilings, "
            "game-script stories, grades, game worlds, exposures). Never invent a lineup, player, number, or game. "
            "Refer to lineups by their rank (#1, #2, ...). When asked 'which lineup', name the rank and explain why "
            "using its stack, game story, ceiling sources, and leverage. "
            "Be willing to critique the build when the data supports it — do not cheerlead. "
            "Keep answers tight: a few short paragraphs max. Use specific names and numbers, not generic advice.\n\n"
            + _DFS_PRO_PLAYBOOK
        )
        recent_history = (history or [])[-8:]
        history_text = "\n".join(
            f"USER: {x[0]}\nAytia: {x[1]}"
            for x in recent_history if isinstance(x, (list, tuple)) and len(x) >= 2
        )
        p = packet or {}
        portfolio = p.get("portfolio", {}) or {}
        contest = p.get("contest", {}) or {}
        slim = {
            "contest": contest,
            "portfolio": {k: portfolio.get(k) for k in (
                "lineups", "unique_qbs", "qb_usage", "stack_mix", "bringback_mix",
                "flex_mix_pct", "salary_left", "avg_player_ownership",
            )},
            "top_exposures": (portfolio.get("top_exposures", []) or [])[:15],
            "most_overweight": (portfolio.get("most_overweight", []) or [])[:8],
            "most_underweight": (portfolio.get("most_underweight", []) or [])[:8],
            "repeated_pairs": (portfolio.get("repeated_pairs", []) or [])[:5],
        }
        ctx = (lineups_ctx or "").strip() or "(No detailed lineup data supplied; answer from the portfolio packet.)"
        prompt = (
            f"{instructions}\n\nPORTFOLIO DATA (built lineups):\n{ctx}\n\n"
            f"GAME WORLDS / EXPOSURES / CONTEST:\n{json.dumps(slim, default=str)}\n\n"
            f"RECENT CONVERSATION:\n{history_text}\n\nUSER QUESTION:\n{q}"
        )
        try:
            model_name = st.secrets.get("OPENAI_MODEL", None)
        except Exception:
            model_name = None
        model_name = model_name or os.getenv("OPENAI_MODEL") or "gpt-5.6-sol"
        resp = OpenAI(api_key=api_key).responses.create(
            model=model_name, reasoning={"effort": "medium"}, input=prompt, max_output_tokens=1200)
        st.session_state["classic_ai_model"] = model_name
        if resp.output_text:
            st.session_state.pop("classic_ai_error", None)
            return resp.output_text
    except Exception as e:
        try:
            st.session_state["classic_ai_error"] = f"{type(e).__name__}: {str(e)[:500]}"
        except Exception:
            pass
    return None


def classic_postbuild_answer(question, packet, history=None, lineups_ctx=None):
    """Reliable evidence-backed Q&A about the portfolio that was actually built."""
    q=str(question or "").strip()
    ql=q.lower()
    p=packet.get("portfolio",{}) or {}
    contest=packet.get("contest",{}) or {}
    players=packet.get("context_players",[]) or []
    if not p.get("built"):
        return "Build lineups first. Post-Build Coach only answers from the portfolio Aytia actually created."

    # Conversational LLM coach first when an API key is configured; the local
    # evidence-backed templates below are the fallback.
    if q:
        try:
            llm = classic_postbuild_llm_answer(q, packet, lineups_ctx, history)
        except Exception:
            llm = None
        if llm:
            return llm

    entry=str(contest.get("entry_format","this contest"))
    field=int(contest.get("field_size",0) or 0)
    payout=str(contest.get("payout",""))
    n=int(p.get("lineups",0) or 0)

    # Resolve player by full name or unique last name.
    player=None
    norm=re.sub(r"[^a-z0-9 ]+"," ",ql)
    for x in players:
        nm=str(x.get("Name","")).strip()
        if nm and re.sub(r"[^a-z0-9 ]+"," ",nm.lower()) in norm:
            player=x; break
    if player is None:
        words=set(norm.split()); hits=[]
        for x in players:
            nm=str(x.get("Name","")).strip()
            parts=[z for z in re.sub(r"[^a-z0-9 ]+"," ",nm.lower()).split() if len(z)>=3]
            if parts and parts[-1] in words: hits.append(x)
        if len(hits)==1: player=hits[0]

    if player is not None:
        nm=str(player.get("Name",""))
        exp=next((x for x in p.get("top_exposures",[]) if str(x.get("player",""))==nm),None)
        if exp is None:
            # Search full overweight/underweight lists too.
            exp=next((x for x in (p.get("most_overweight",[])+p.get("most_underweight",[])) if str(x.get("player",""))==nm),None)
        field_own=float(player.get("My Own",0) or 0)
        proj=float(player.get("My Proj",0) or 0)
        sal=int(player.get("Salary",0) or 0)
        if exp:
            my=float(exp.get("exposure_pct",0) or 0); lev=float(exp.get("leverage_pct",my-field_own) or 0)
            return (f"**{nm} is in {my:.0f}% of this portfolio versus {field_own:.1f}% projected field ownership** ({lev:+.1f} pts of leverage). "
                    f"Aytia is getting there from a {proj:.1f}-point projection at ${sal:,}, plus the lineup combinations he fits. "
                    "The key question is whether that exposure is supported by ceiling/correlation or is simply being repeated because he fits salary. "
                    "If you ask 'too much?' I would judge that against the repeated cores and game stories he appears in.")
        return (f"**{nm}** is projected for {proj:.1f} points at ${sal:,} with {field_own:.1f}% projected ownership, "
                "but he is not among the portfolio's highest exposures. That means Aytia is not leaning heavily on him in the current build.")

    # Winner-take-all / how different.
    if any(x in ql for x in ["winner take all","winner-take-all","how different","different do i","unique do i","first place"]):
        pairs=p.get("repeated_pairs",[]) or []
        top_pair=pairs[0] if pairs else None
        core=(f" Your most repeated two-player core is **{' + '.join(top_pair['players'])} at {top_pair['portfolio_pct']:.0f}%**." if top_pair else "")
        return (f"For **{entry} in a {field:,}-entry {payout} contest**, you do **not** need nine low-owned players. "
                "You need one lineup with a credible first-place ceiling and at least one meaningful way it differs from the common field shell. "
                "That difference can come from the exact stack combination, WR/RB/TE FLEX construction, direct leverage against chalk, or a lower-owned ceiling play."
                +core+
                " Strong chalk is fine if the rest of the lineup tells a different story. Random contrarianism is not the goal.")

    if "qb" in ql or "quarterback" in ql:
        qbs=p.get("qb_usage",[]) or []
        txt=", ".join(f"{x['qb']} {x['exposure_pct']:.0f}%" for x in qbs[:8]) or "none"
        return (f"This {n}-lineup candidate portfolio uses **{p.get('unique_qbs',0)} QBs**: {txt}. "
                f"For {entry}, that spread is useful for exploring alternatives, but your final entry should express one QB/game thesis. "
                "The reason to keep a QB should be his ceiling plus the quality and ownership of his exact stack—not QB ownership by itself.")

    if "stack" in ql or "pass catcher" in ql:
        sm=p.get("stack_mix",{}) or {}
        return (f"Current QB-stack mix is **{sm}**. The number is descriptive, not automatically optimal. "
                "QB+2 should be favored when the passing offense is condensed or when the second receiver increases your chance of capturing the slate-breaking pass catcher. "
                "QB+1 is preferable when the second teammate is weak and forcing him costs a materially better one-off.")

    if "bring" in ql or "run back" in ql or "runback" in ql:
        bm=p.get("bringback_mix",{}) or {}
        return (f"Current bring-back mix is **{bm}**. A bring-back should exist because that opponent helps the stack keep scoring, not because a rule says every stack needs one. "
                "On soft-pricing slates, Aytia should be willing to omit a weak opponent piece when a much stronger one-off preserves more ceiling.")

    if "flex" in ql or "construction" in ql:
        fm=p.get("flex_mix_pct",{}) or {}
        return (f"Your current FLEX construction is **{fm}**. This is one of the cleanest places to look for leverage because roster shape can be different even when individual players are not. "
                "Aytia should compare median projection with ceiling here; a WR can trail an RB in median while still offering the better tournament-separating outcome.")

    if any(x in ql for x in ["chalk","chalky","ownership","leverage","different"]):
        ow=p.get("most_overweight",[])[:4]; uw=p.get("most_underweight",[])[:4]
        owtxt=", ".join(f"{x['player']} {x['exposure_pct']:.0f}% vs {x['field_own_pct']:.0f}%" for x in ow) or "none"
        uwtxt=", ".join(f"{x['player']} {x['exposure_pct']:.0f}% vs {x['field_own_pct']:.0f}%" for x in uw) or "none"
        return (f"Your biggest current overweights are **{owtxt}**. Biggest underweights are **{uwtxt}**. "
                "Those differences only matter if the overweight players have real ceiling or direct leverage. I would not lower exposure just to make the lineup look contrarian.")

    if any(x in ql for x in ["core","repeat","same","concentrated","spread"]):
        pairs=p.get("repeated_pairs",[])[:4]
        txt="; ".join(f"{' + '.join(x['players'])} {x['portfolio_pct']:.0f}%" for x in pairs) or "none"
        return (f"The most repeated two-player cores are **{txt}**. Repetition is fine when it represents conviction, but if several lineups share the same core *and* the same game thesis, "
                "they are less diversified than their unique-player count suggests.")

    if any(x in ql for x in ["salary","left over","leftover"]):
        sl=p.get("salary_left",{}) or {}
        return (f"Average salary left is **${float(sl.get('mean') or 0):,.0f}** with a median of **${float(sl.get('median') or 0):,.0f}**. "
                "On Classic slates, unused salary is not leverage by itself. It only matters if leaving salary produces a stronger, less common construction without sacrificing too much ceiling.")

    if any(x in ql for x in ["critique","change","wrong","risk","like","dislike","good"]):
        findings=classic_postbuild_report(packet)
        if findings:
            return "\n\n".join(f"**{title}:** {body}" for title,body in findings[:5])

    findings=classic_postbuild_report(packet)
    return ("Here is what I can defend from the current build:\n\n"+
            "\n\n".join(f"**{title}:** {body}" for title,body in findings[:4])+
            "\n\nAsk me **why** about a player, QB concentration, stack mix, bring-backs, FLEX construction, ownership/leverage, repeated cores, or winner-take-all strategy.")

def classic_ai_slate_answer(question, packet, history=None):
    """Contest-aware Aytia assistant.

    Prefer the OpenAI model when available. If the API is unavailable, the local fallback
    still answers the user's ACTUAL question from Slate Intel instead of repeating a canned
    recommendation block.
    """
    q=str(question or "").strip()
    ql=q.lower()
    try:
        from openai import OpenAI
        try: api_key=st.secrets.get("OPENAI_API_KEY",None)
        except Exception: api_key=os.getenv("OPENAI_API_KEY")
        if api_key:
            instructions=("""You are Aytia Classic Slate Intel, an NFL DraftKings strategy assistant.
Answer the user's exact question first. Do not dump generic recommendations unless they are relevant.
Use the supplied slate packet and contest context. Be willing to disagree with Aytia's default settings
when the evidence supports it. Explain tradeoffs rather than pretending there is one correct DFS answer.

""" + _DFS_PRO_PLAYBOOK + """For Single Entry and 3-Max, discuss concentration and taking stands when evidence separates. For 20-Max
and 150-Max, discuss portfolio coverage and diversification. A thesis may originate from a QB, receiver,
RB, or game environment. Distinguish field ownership from user exposure. Never invent injuries, Vegas,
weather, travel, or facts missing from the packet. When the user challenges a number (for example, '24 QBs
is too many'), directly evaluate that number using the QB concentration evidence and recommend what the
program should change or what evidence would justify keeping it. When evaluating lineups, think in terms
of the STORY the lineup tells, whether that story is coherent, whether ownership is concentrated in the
same obvious places as the field, and whether the portfolio gives enough combinations to its strongest
theses.""")
            recent_history=(history or [])[-8:]
            history_text="\n".join([f"USER: {x[0]}\nAytia: {x[1]}" for x in recent_history if isinstance(x,(list,tuple)) and len(x)>=2])
            prompt=f"{instructions}\n\nSLATE + PORTFOLIO PACKET:\n{json.dumps(packet,default=str)}\n\nRECENT CONVERSATION:\n{history_text}\n\nUSER QUESTION:\n{q}"
            try: model_name=st.secrets.get("OPENAI_MODEL",None)
            except Exception: model_name=None
            model_name=model_name or os.getenv("OPENAI_MODEL") or "gpt-5.6-sol"
            resp=OpenAI(api_key=api_key).responses.create(model=model_name,reasoning={"effort":"medium"},input=prompt,max_output_tokens=1800)
            st.session_state["classic_ai_model"]=model_name
            if resp.output_text:
                st.session_state.pop("classic_ai_error",None)
                return resp.output_text
    except Exception as e:
        st.session_state["classic_ai_error"]=f"{type(e).__name__}: {str(e)[:500]}"

    # Local evidence-aware fallback. This should still be conversational and answer intent.
    rec=packet.get("recommendations",{})
    contest=packet.get("contest",{})
    sims=packet.get("simulations",[]) or []
    qb=packet.get("qb_concentration",{}) or {}
    qbc=packet.get("qb_candidates",[]) or []
    theses=packet.get("theses",[]) or []
    portfolio=packet.get("portfolio",{}) or {}
    context_players=packet.get("context_players",[]) or []
    entry=str(contest.get("entry_format","this contest"))
    field=int(contest.get("field_size",0) or 0)
    recent_history=(history or [])[-8:]
    prior_user=" ".join(str(x[0]) for x in recent_history if isinstance(x,(list,tuple)) and len(x)>=2).lower()

    # Build transparent game-level leverage diagnostics from the packet.
    # "ownership load" is a sum of player ownership, NOT projected stack ownership.
    sim_by_game={str(x.get("Game","")):float(x.get("Slate ceiling %",0) or 0) for x in sims if str(x.get("Game",""))}
    game_players=defaultdict(list)
    for p in context_players:
        g=str(p.get("Matchup",p.get("Game","")) or "")
        if not g: continue
        game_players[g].append({
            "name":str(p.get("Name","")),
            "own":float(pd.to_numeric(pd.Series([p.get("My Own",p.get("Own",0))]),errors="coerce").fillna(0).iloc[0]),
            "proj":float(pd.to_numeric(pd.Series([p.get("My Proj",p.get("Proj",0))]),errors="coerce").fillna(0).iloc[0]),
            "pos":str(p.get("Position",""))
        })
    game_diag=[]
    for g,plist in game_players.items():
        ranked=sorted(plist,key=lambda z:(z["own"],z["proj"]),reverse=True)
        # Top-six load captures how much popular salary/ownership is clustering in the game
        # without pretending this equals combination ownership.
        own_load=sum(x["own"] for x in ranked[:6])
        ceiling=float(sim_by_game.get(g,0.0))
        game_diag.append({"game":g,"ceiling":ceiling,"own_load":own_load,"top":ranked[:5]})
    game_diag=sorted(game_diag,key=lambda z:(z["ceiling"],-z["own_load"]),reverse=True)

    # Resolve a specifically named player before broad intent checks such as "why".
    # This keeps "Why is Chris Olave highly owned?" from falling into the generic game branch.
    player_match=None
    q_norm=re.sub(r"[^a-z0-9 ]+"," ",ql)
    for p in context_players:
        nm=str(p.get("Name","")).strip()
        if nm and re.sub(r"[^a-z0-9 ]+"," ",nm.lower()) in q_norm:
            player_match=p
            break
    if player_match is None:
        q_words=set(q_norm.split())
        candidates=[]
        for p in context_players:
            nm=str(p.get("Name","")).strip()
            parts=[x for x in re.sub(r"[^a-z0-9 ]+"," ",nm.lower()).split() if len(x)>=3]
            if parts and parts[-1] in q_words:
                candidates.append(p)
        if len(candidates)==1:
            player_match=candidates[0]

    # QB-pool / concentration questions.
    if any(k in ql for k in ["qb","quarterback"]) and any(k in ql for k in ["too many","too big","field","pool","narrow","fewer","many"]):
        kept=[x for x in qbc if bool(x.get("In build pool",False))]
        # Older packet rows may not include the boolean; fall back to concentration names.
        names=list(qb.get("names",[]) or [])
        n=len(names) if names else len(kept)
        if n==0 and qbc:
            floor=float(qb.get("relative_floor",0) or 0)
            n=sum(1 for x in qbc if float(x.get("Relative %",0) or 0)>=floor)
        top=qbc[:6]
        evidence=", ".join(f"{x.get('QB','?')} {float(x.get('Relative %',0) or 0):.0f}%" for x in top)
        if entry in ["Single Entry","3-Max"]:
            stance=("I agree that **"+str(n)+" QBs is too broad for "+entry+"** unless the slate is exceptionally flat. "
                    "With only a few actual entries, candidate generation should concentrate on the strongest evidence-backed routes, "
                    "not preserve nearly every viable quarterback just because each can make a legal lineup.")
        elif entry=="20-Max":
            stance=("For **20-Max**, "+str(n)+" QBs is probably too broad if most of them sit well below the top evidence tier. "
                    "Twenty entries still benefit from taking stands; diversification should come from multiple correlated constructions, "
                    "not automatically from using almost every quarterback.")
        else:
            stance=("For **"+entry+"**, a wider QB pool can make sense, but it should still be earned by the evidence rather than by randomness.")
        return (stance+"\n\nThe current QB evidence is: **"+evidence+"**. "
                "I would tighten the QB evidence band until the remaining quarterbacks are meaningfully competitive with the top routes, "
                "while preserving tied/near-tied options. That means the number of QBs can change slate to slate instead of using a fixed cap.")

    # Stack questions.
    if any(k in ql for k in ["pass catcher","double stack","single stack","stack"]):
        return (f"For **{entry}** in a {field:,}-entry field, Aytia currently recommends **{rec.get('qb_stack',1)} QB pass catcher(s)**. "
                "That should be treated as a slate-driven starting point, not a universal rule. The better implementation is to let the "
                "simulation/thesis evidence determine a mix of single and double stacks, then concentrate that mix more aggressively in "
                "Single Entry/3-Max and spread it more in larger portfolios.")

    # Specific player ownership / exposure questions take priority over broad "why" intent.
    if player_match is not None and any(k in ql for k in ["ownership","owned","chalk","exposure","overweight","underweight"]):
        nm=str(player_match.get("Name","this player"))
        own=float(pd.to_numeric(pd.Series([player_match.get("My Own",player_match.get("Own",0))]),errors="coerce").fillna(0).iloc[0])
        proj=float(pd.to_numeric(pd.Series([player_match.get("Proj",player_match.get("My Proj",0))]),errors="coerce").fillna(0).iloc[0])
        salary=float(pd.to_numeric(pd.Series([player_match.get("Salary",0)]),errors="coerce").fillna(0).iloc[0])
        hist=float(pd.to_numeric(pd.Series([player_match.get("Hist FPPG",0)]),errors="coerce").fillna(0).iloc[0])
        recent=float(pd.to_numeric(pd.Series([player_match.get("Recent 6",0)]),errors="coerce").fillna(0).iloc[0])
        dvp=float(pd.to_numeric(pd.Series([player_match.get("DVP Adj %",0)]),errors="coerce").fillna(0).iloc[0])
        value=(proj/(salary/1000.0)) if salary>0 else 0.0
        if "why" in ql or "ownership" in ql or "owned" in ql or "chalk" in ql:
            facts=[]
            if proj>0: facts.append(f"**{proj:.1f} DK projection**")
            if salary>0: facts.append(f"**${salary:,.0f} salary**")
            if value>0: facts.append(f"**{value:.2f}x median value**")
            if recent>0: facts.append(f"**{recent:.1f} recent-six FPPG**")
            if hist>0: facts.append(f"**{hist:.1f} historical FPPG**")
            if abs(dvp)>=0.1: facts.append(f"**{dvp:+.1f}% opponent-vs-position adjustment**")
            evidence=", ".join(facts) if facts else "the projection and salary inputs in the uploaded slate"
            return (f"**{nm} is projected for {own:.1f}% field ownership.** In the data Aytia currently has, the visible reasons are {evidence}. "
                    "That explains why the field can gravitate to him, but field ownership is still a projection, not your exposure. "
                    "Aytia does not currently have a live injury/news feed, so I will not invent a news or role explanation that is not in the slate packet.")
        if portfolio.get("built"):
            cur=next((x for x in portfolio.get("top_exposures",[]) if str(x.get("player",""))==nm),None)
            if cur:
                return (f"**{nm}: your exposure is {float(cur.get('exposure_pct',0)):.1f}% versus {own:.1f}% projected field ownership** "
                        f"({float(cur.get('leverage_pct',0)):+.1f} percentage points). Tell me a max or minimum exposure if you want Aytia to change the build rule.")
        return (f"**{nm} is projected for {own:.1f}% field ownership.** There is no current portfolio exposure to compare until lineups are built.")

    # Game ownership / leverage questions. Use actual player ownership evidence and conversation context.
    _game_ref=None
    for gd in game_diag:
        g=gd["game"]
        if g and (g.lower() in ql or g.lower() in prior_user):
            _game_ref=gd
            if g.lower() in ql: break

    if any(k in ql for k in ["chalk","chalky","owned","ownership","popular"]) and (_game_ref is not None or "game" in ql):
        gd=_game_ref or (game_diag[0] if game_diag else None)
        if gd:
            topbits=", ".join(f"{x['name']} {x['own']:.0f}%" for x in gd["top"][:4])
            alternatives=[x for x in game_diag if x["game"]!=gd["game"] and x["ceiling"]>0][:4]
            alt_txt="; ".join(
                f"**{x['game']}** — {x['ceiling']:.1f}% ceiling share, top-six player-own load {x['own_load']:.0f}"
                for x in alternatives[:3]
            )
            return (f"**{gd['game']} can be strong without automatically being too chalky.** It leads the current simulations at "
                    f"**{gd['ceiling']:.1f}%** of slate-leading environments. The most-owned pieces in that game are **{topbits}**. "
                    f"I would not call the game itself 'too chalky' from those individual numbers alone because player ownership does not equal stack ownership. "
                    f"The better question is whether your exact combination is common and whether you're also using the same RB/TE/FLEX shell as the field. "
                    +(f"Lower-owned game alternatives with meaningful simulated ceiling are {alt_txt}. " if alt_txt else "")
                    +"For Single Entry, I would prefer one strong game thesis with a differentiated combination over fading the best environment just because it is popular.")

    # Contextual follow-up such as "what are other alternatives?" after discussing a game.
    if any(k in ql for k in ["alternative","alternatives","other games","what else","else can","other options"]):
        base=_game_ref["game"] if _game_ref else (sims[0].get("Game","") if sims else "")
        alts=[x for x in game_diag if x["game"]!=base and x["ceiling"]>0][:5]
        if alts:
            parts=[]
            for x in alts[:4]:
                topnames=", ".join(f"{p['name']} {p['own']:.0f}%" for p in x["top"][:3])
                parts.append(f"**{x['game']}** — {x['ceiling']:.1f}% simulated ceiling share; top ownership pieces: {topnames}")
            return ("If you're looking for alternatives to **"+str(base)+"**, I would start with these game environments rather than just grabbing random low-owned players:\n\n"
                    +"\n\n".join(parts)
                    +"\n\nThe best alternative is the one that gives you a coherent stack with a lower-owned **combination**, not simply the lowest-owned game. In Single Entry I would usually choose one of these as the primary alternate thesis rather than trying to cover all of them.")

    # General game / thesis questions.
    if any(k in ql for k in ["game","thesis","why","disappoint"]):
        top=sims[0] if sims else {}
        tg=top.get("Game","the leading game")
        share=float(top.get("Slate ceiling %",0) or 0)
        runner=float(sims[1].get("Slate ceiling %",0) or 0) if len(sims)>1 else 0
        return (f"**{tg}** is the current leading simulated game environment at **{share:.1f}%** of slate-leading outcomes"
                +(f", versus **{runner:.1f}%** for the next game" if runner else "")+
                ". That is a lead, not domination. I would treat it as a lean, not a lock. If you want, ask whether that game is too chalky or ask for alternatives and I will compare ceiling with the actual ownership of the pieces.")

    # Exposure / ownership questions.
    if "exposure" in ql or "ownership" in ql or "owned" in ql or "overweight" in ql or "underweight" in ql:
        if portfolio.get("built"):
            ow=portfolio.get("most_overweight",[])[:4]; uw=portfolio.get("most_underweight",[])[:4]
            ow_txt=", ".join(f"{x['player']} {x['exposure_pct']:.0f}% vs {x['field_own_pct']:.0f}% ({x['leverage_pct']:+.0f})" for x in ow)
            uw_txt=", ".join(f"{x['player']} {x['exposure_pct']:.0f}% vs {x['field_own_pct']:.0f}% ({x['leverage_pct']:+.0f})" for x in uw)
            return (f"Your current {portfolio.get('lineups',0)}-lineup portfolio is most overweight on **{ow_txt}**. "
                    f"The largest underweights are **{uw_txt}**. Those deltas are exposure minus projected field ownership; "
                    "they are not automatically good or bad. I would judge each against ceiling, role, correlation and the contest.")
        return ("There is no built portfolio yet, so I can discuss projected field ownership but not your actual exposure. "
                "Generate lineups first and I can compare your exposure with the field player by player.")

    if portfolio.get("built") and any(k in ql for k in ["portfolio","critique","spread","concentrated","chalky","change","risk"]):
        qbs=portfolio.get("qb_usage",[])
        pairs=portfolio.get("repeated_pairs",[])
        qtxt=", ".join(f"{x['qb']} {x['exposure_pct']:.0f}%" for x in qbs[:5]) or "none"
        ptxt=", ".join(f"{' + '.join(x['players'])} {x['portfolio_pct']:.0f}%" for x in pairs[:3]) or "none"
        return (f"Your current portfolio has **{portfolio.get('lineups',0)} lineups and {portfolio.get('unique_qbs',0)} QBs**. "
                f"Top QB usage: **{qtxt}**. The most repeated two-player cores are **{ptxt}**. "
                f"Stack mix is {portfolio.get('stack_mix',{})}; bring-back mix is {portfolio.get('bringback_mix',{})}. "
                "That is the right place to start the critique: whether those concentrations represent independent first-place stories or the same bet repeated.")

    # General fallback uses recent conversation instead of resetting to a canned slate summary.
    if recent_history:
        lastq=str(recent_history[-1][0]) if isinstance(recent_history[-1],(list,tuple)) and len(recent_history[-1])>=2 else ""
        return (f"I’m treating this as a follow-up to **{lastq}**. I don’t have enough local evidence to answer that specific follow-up cleanly yet, "
                "so I’d rather say that than reset to the generic slate summary. Try naming the game/player once and I’ll use the current slate evidence.")
    top=sims[0]["Game"] if sims else "the top simulated game"
    return (f"For **{entry}** in a {field:,}-entry field, the strongest current game environment is **{top}**. "
            f"The current QB concentration plan is **{qb.get('label','open')}**.")

def classic_ai_action_plan(question, packet, history=None):
    """Language-model intent/action router for natural DFS conversation."""
    q=str(question or "").strip()
    if not q:
        return {"mode":"question","actions":[],"needs_clarification":False,"clarification":None}
    try:
        from openai import OpenAI
        try: api_key=st.secrets.get("OPENAI_API_KEY",None)
        except Exception: api_key=os.getenv("OPENAI_API_KEY")
        if not api_key: return None
        try: model_name=st.secrets.get("OPENAI_MODEL",None)
        except Exception: model_name=None
        model_name=model_name or os.getenv("OPENAI_MODEL") or "gpt-5.6-sol"
        players=[{"id":str(x.get("ID","")),"name":str(x.get("Name","")),"position":str(x.get("Position","")),
                  "team":str(x.get("Team","")),"field_own":x.get("My Own",0),"projection":x.get("My Proj",0)}
                 for x in (packet.get("context_players",[]) or [])]
        ctx={"contest":packet.get("contest",{}),"active_rules":packet.get("active_rules",{}),
             "qb_concentration":packet.get("qb_concentration",{}),"players":players,
             "portfolio":packet.get("portfolio",{})}
        recent=(history or [])[-8:]
        router=f"""You are the action router for Aytia, an NFL DFS strategist.
Interpret natural language dynamically, using context and recent conversation.

Classify the user's message as a question/challenge, a clear build instruction, or both.
A concern is not automatically an instruction: "Olave seems too high" means analyze it.
A directive is an instruction: "Cut Olave to 15%" means change the build.
Resolve pronouns from recent conversation only when unambiguous. If ambiguous, ask a clarification.

Allowed actions:
set_player_max_exposure(player_id,player_name,value 0-100)
set_player_min_exposure(player_id,player_name,value 0-100)
exclude_player(player_id,player_name)
include_player(player_id,player_name)
set_player_priority(player_id,player_name,value Core|Like|Neutral|Fade)
set_qb_cap(value 1-32)
clear_qb_cap
set_qb_stack_min(value 1-3)
set_bringback(value Optional|Required|None)
set_min_salary(value 44000-50000)
set_max_game(value 4-9)
set_max_team(value 4-9)
set_max_te(value 1-3)
set_allow_qb_rb(value boolean)
set_no_dst_from_qb_game(value boolean)
set_no_offense_vs_dst(value boolean)
set_team_priority(team,value Core|Like|Neutral|Fade|Exclude)

Return ONLY JSON:
{{"mode":"question|instruction|mixed","actions":[{{"type":"..."}}],"needs_clarification":false,"clarification":null}}

Never invent player IDs. Do not change projected FIELD ownership; exposure actions change only the user's portfolio.
Do not create a numeric cap unless the user clearly stated or confirmed it.

CONTEXT:
{json.dumps(ctx,default=str)}
RECENT:
{json.dumps(recent,default=str)}
USER:
{q}"""
        resp=OpenAI(api_key=api_key).responses.create(model=model_name,reasoning={"effort":"medium"},input=router,max_output_tokens=900)
        raw=(resp.output_text or "").strip()
        raw=re.sub(r"^\x60\x60\x60(?:json)?\s*|\s*\x60\x60\x60$","",raw,flags=re.I|re.S).strip()
        plan=json.loads(raw)
        if not isinstance(plan,dict): return None
        plan.setdefault("actions",[]); plan.setdefault("needs_clarification",False); plan.setdefault("clarification",None)
        st.session_state["classic_ai_model"]=model_name
        return plan
    except Exception as e:
        st.session_state["classic_ai_error"]=f"{type(e).__name__}: {str(e)[:500]}"
        return None

def classic_apply_ai_actions(actions, df):
    """Apply validated structured actions from the AI router."""
    applied=[]
    valid_ids=set(df["ID"].astype(str).tolist()); valid_teams=set(df["Team"].astype(str).tolist())
    for a in (actions or []):
        if not isinstance(a,dict): continue
        typ=str(a.get("type","")).strip()
        if typ.startswith("set_player_") or typ in {"exclude_player","include_player"}:
            pid=str(a.get("player_id",""))
            if pid not in valid_ids: continue
            row=df[df["ID"].astype(str).eq(pid)].iloc[0]; name=str(row["Name"])
            cur=dict(st.session_state["strategy_master"].get(pid,{}) or {})
            cur.setdefault("Priority","Neutral"); cur.setdefault("Min Exposure",0.0); cur.setdefault("Max Exposure",100.0)
            cur.setdefault("Lock",False); cur.setdefault("Exclude",False)
            if typ=="set_player_max_exposure":
                v=max(0.0,min(100.0,float(a.get("value",100)))); cur["Max Exposure"]=v
                if float(cur.get("Min Exposure",0))>v: cur["Min Exposure"]=v
                applied.append(f"{name} max exposure {v:.0f}%")
            elif typ=="set_player_min_exposure":
                v=max(0.0,min(100.0,float(a.get("value",0)))); cur["Min Exposure"]=v
                if float(cur.get("Max Exposure",100))<v: cur["Max Exposure"]=v
                applied.append(f"{name} min exposure {v:.0f}%")
            elif typ=="exclude_player":
                cur["Exclude"]=True; cur["Lock"]=False; cur["Priority"]="Exclude"; applied.append(f"exclude {name}")
            elif typ=="include_player":
                cur["Exclude"]=False
                if cur.get("Priority")=="Exclude": cur["Priority"]="Neutral"
                applied.append(f"include {name}")
            elif typ=="set_player_priority":
                v=str(a.get("value","Neutral"))
                if v in {"Core","Like","Neutral","Fade"}:
                    cur["Priority"]=v; cur["Exclude"]=False; applied.append(f"{name} lean {v}")
            st.session_state["strategy_master"][pid]=cur
        elif typ=="set_qb_cap":
            v=max(1,min(32,int(a.get("value",1)))); st.session_state["classic_qb_cap"]=v; applied.append(f"QB pool cap {v}")
        elif typ=="clear_qb_cap":
            st.session_state["classic_qb_cap"]=0; applied.append("clear QB pool cap")
        elif typ=="set_qb_stack_min":
            v=max(1,min(3,int(a.get("value",1)))); st.session_state["classic_qb_stack"]=v; applied.append(f"QB + {v} pass catcher minimum")
        elif typ=="set_bringback":
            v=str(a.get("value","Optional"))
            if v in {"Optional","Required","None"}: st.session_state["classic_bringback"]=v; applied.append(f"bring-back {v}")
        elif typ=="set_min_salary":
            v=max(44000,min(50000,int(a.get("value",48500)))); st.session_state["classic_min_salary"]=v; applied.append(f"minimum salary ${v:,}")
        elif typ=="set_max_game":
            v=max(4,min(9,int(a.get("value",5)))); st.session_state["classic_max_game"]=v; applied.append(f"max {v} from one game")
        elif typ=="set_max_team":
            v=max(4,min(9,int(a.get("value",5)))); st.session_state["classic_max_team"]=v; applied.append(f"max {v} from one team")
        elif typ=="set_max_te":
            v=max(1,min(3,int(a.get("value",2)))); st.session_state["classic_max_te"]=v; applied.append(f"max {v} tight ends")
        elif typ=="set_allow_qb_rb":
            v=bool(a.get("value",True)); st.session_state["classic_allow_qb_rb"]=v; applied.append("allow QB + same-team RB" if v else "block QB + same-team RB")
        elif typ=="set_no_dst_from_qb_game":
            v=bool(a.get("value",True)); st.session_state["classic_no_dst"]=v; applied.append("block DST from QB game" if v else "allow DST from QB game")
        elif typ=="set_no_offense_vs_dst":
            v=bool(a.get("value",False)); st.session_state["classic_no_off"]=v; applied.append("block offense vs DST" if v else "allow offense vs DST")
        elif typ=="set_team_priority":
            team=str(a.get("team","")); v=str(a.get("value","Neutral"))
            if team in valid_teams and v in {"Core","Like","Neutral","Fade","Exclude"}:
                st.session_state["team_strategy_master"][team]=v; applied.append(f"{team} team lean {v}")
    return applied

def calculate_exposure_table(df, result, strategy_map):
    if result is None or result.empty:
        return pd.DataFrame()

    counts = defaultdict(int)
    lineup_cols = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
    for col in lineup_cols:
        for name in result[col].dropna():
            counts[name] += 1

    n = len(result)
    rows = []
    for _, p in df.iterrows():
        name = p["Name"]
        pid = str(p["ID"])
        strat = strategy_map.get(pid, {})
        count = counts.get(name, 0)
        rows.append({
            "ID": pid,
            "Name": name,
            "Pos": p["Position"],
            "Team": p["Team"],
            "Opponent": p.get("Opponent",""),
            "Salary": int(p["Salary"]),
            "Proj": round(float(p["My Proj"]), 2),
            "Proj Own": round(float(p["My Own"]), 1),
            "Actual Exp %": round(100.0 * count / n, 1),
            "Lineups": count,
            "Min Target %": float(strat.get("Min Exposure", 0)),
            "Max Target %": float(strat.get("Max Exposure", 100)),
        })
    return pd.DataFrame(rows).sort_values(["Actual Exp %", "Proj"], ascending=[False, False]).reset_index(drop=True)
