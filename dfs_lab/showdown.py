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


from dfs_lab.common import contest_aggression, percentile_label
from dfs_lab.config import (SHOWDOWN_SLOTS, GAME_WORLDS, CONTEXT_FACTOR_WEIGHTS,
                            CONTEXT_RATING_LABELS, PRIORITY_BONUS)

def showdown_aggression(field_size, payout_style, entry_format="Single Entry"):
    aggr = contest_aggression(field_size, payout_style)
    entry_bump={"Single Entry":-0.14,"3-Max":-0.05,"20-Max":0.05,"150-Max":0.14}.get(entry_format,0.0)
    return float(np.clip(aggr + 0.08 + entry_bump, 0.05, 1.0))

def configure_game_worlds(df):
    global GAME_WORLDS
    teams=[str(t) for t in df["Team"].dropna().unique().tolist() if str(t)]
    if len(teams)<2:
        GAME_WORLDS={"Balanced game":{"family":"shootout","desc":"Both sides produce usable fantasy scoring."}}
        return GAME_WORLDS

    # Preserve away/home ordering when Game Info gives it to us.
    away=[str(x) for x in df.get("Away",pd.Series(dtype=str)).dropna().unique().tolist() if str(x)]
    home=[str(x) for x in df.get("Home",pd.Series(dtype=str)).dropna().unique().tolist() if str(x)]
    ordered=[]
    for t in away+home+teams:
        if t in teams and t not in ordered: ordered.append(t)
    a,b=ordered[:2]

    def primary_qb(team):
        q=df[df["Team"].eq(team)&df["is_QB"]].copy()
        if q.empty: return ""
        # Never name a world after a quarterback who is not playing: the
        # depth-chart "Primary QB" flag can still point at an inactive player
        # (e.g. an OUT starter), which produced nonsense like a
        # "Caleb Williams ceiling" world for a game he is not playing in.
        if "ActiveForBuild" in q.columns:
            q_active=q[q["ActiveForBuild"].fillna(True).astype(bool)]
            if q_active.empty: return ""
            q=q_active
        marked=q[q.get("Primary QB",False).astype(bool)] if "Primary QB" in q.columns else pd.DataFrame()
        if not marked.empty: return str(marked.iloc[0]["Name"])
        q=q.sort_values(["FlexSalary","AvgPointsPerGame","My Proj"],ascending=False)
        return str(q.iloc[0]["Name"])

    qa,qb=primary_qb(a),primary_qb(b)
    worlds={
        "Balanced shootout":{"family":"shootout","desc":f"{a} and {b} both produce; scoring is spread across primary pieces."},
        f"{a} passing ceiling":{"family":"pass_ceiling","team":a,"desc":f"{a} scoring concentrates through its primary quarterback and pass catchers."},
        f"{b} passing ceiling":{"family":"pass_ceiling","team":b,"desc":f"{b} scoring concentrates through its primary quarterback and pass catchers."},
        f"{a} RB-led win":{"family":"rb_control","team":a,"desc":f"{a} controls enough of the game for its running backs and control pieces to matter."},
        f"{b} RB-led win":{"family":"rb_control","team":b,"desc":f"{b} controls enough of the game for its running backs and control pieces to matter."},
        f"{a} leads / {b} comeback":{"family":"comeback","lead":a,"trail":b,"desc":f"{a} plays from ahead while {b} answers with elevated passing volume."},
        f"{b} leads / {a} comeback":{"family":"comeback","lead":b,"trail":a,"desc":f"{b} plays from ahead while {a} answers with elevated passing volume."},
        "Low-scoring game":{"family":"low","desc":"Scoring disappoints; kickers, defenses and concentrated touchdown paths gain importance."},
    }
    if qa:
        worlds[f"{qa} ceiling"]={"family":"qb_ceiling","team":a,"player":qa,"desc":f"{qa} captures an outsized share of {a}'s fantasy production."}
    if qb:
        worlds[f"{qb} ceiling"]={"family":"qb_ceiling","team":b,"player":qb,"desc":f"{qb} captures an outsized share of {b}'s fantasy production."}
    GAME_WORLDS=worlds
    return GAME_WORLDS

def contest_world_weights(entry_format, field_size, script="Neutral"):
    # Portfolio exploration weights, generated from the CURRENT uploaded matchup.
    # Game Worlds are portfolio alternatives, not the same thing as the user's
    # selected game thesis. A GB-dominant thesis can still contain a balanced
    # shootout world; the thesis shapes projections/construction inside that world.
    if not GAME_WORLDS:
        return {"Balanced shootout":1.0}
    family_base={
        "Single Entry":{"shootout":36,"pass_ceiling":15,"qb_ceiling":8,"rb_control":7,"comeback":5,"low":2},
        "3-Max":{"shootout":27,"pass_ceiling":16,"qb_ceiling":10,"rb_control":8,"comeback":6,"low":3},
        "20-Max":{"shootout":20,"pass_ceiling":16,"qb_ceiling":11,"rb_control":9,"comeback":8,"low":3},
        "150-Max":{"shootout":13,"pass_ceiling":15,"qb_ceiling":12,"rb_control":10,"comeback":10,"low":5},
    }
    base=family_base.get(entry_format,family_base["20-Max"])
    w={name:float(base.get(cfg.get("family"),5)) for name,cfg in GAME_WORLDS.items()}

    if script in ["Shootout","Pass-heavy shootout"]:
        for name,cfg in GAME_WORLDS.items():
            if cfg.get("family") in ["shootout","pass_ceiling","comeback"]: w[name]*=1.35
            elif cfg.get("family")=="low": w[name]*=0.35
    elif script in ["Low-scoring game","Defensive / field-goal battle","Ground-and-pound"]:
        for name,cfg in GAME_WORLDS.items():
            if cfg.get("family")=="low": w[name]*=3.0
            elif cfg.get("family")=="rb_control": w[name]*=1.35
            elif cfg.get("family")=="shootout": w[name]*=0.45
            # A low-scoring thesis contradicts passing-ceiling worlds too: keep
            # them as rare alternatives, not full-weight draws.
            elif cfg.get("family")=="pass_ceiling": w[name]*=0.5
            elif cfg.get("family")=="qb_ceiling": w[name]*=0.6
            elif cfg.get("family")=="comeback": w[name]*=0.7

    if field_size>=50000 and entry_format=="150-Max":
        for name,cfg in GAME_WORLDS.items():
            if cfg.get("family")=="shootout": w[name]*=0.8
            elif cfg.get("family") in ["low","qb_ceiling"]: w[name]*=1.25

    # A directional game thesis should dominate the world mix, not sit inside
    # it as a small side bet. Lift the thesis family to a ~45% plurality; the
    # remaining worlds stay as genuine alternatives instead of contradicting
    # the user's pick most of the time.
    _THESIS_FAMILY={
        "Low-scoring game":"low",
        "Defensive / field-goal battle":"low",
        "Ground-and-pound":"rb_control",
        "Shootout":"shootout",
        "Pass-heavy shootout":"pass_ceiling",
    }
    fam=_THESIS_FAMILY.get(script)
    if fam:
        fam_names=[n for n,cfg in GAME_WORLDS.items() if cfg.get("family")==fam]
        fam_total=sum(w[n] for n in fam_names)
        rest_total=sum(w.values())-fam_total
        if fam_total>0 and rest_total>0:
            scale=(0.45/0.55)*(rest_total/fam_total)
            for n in fam_names: w[n]*=scale
    return w

def choose_game_world(rng, entry_format, field_size, script):
    weights=contest_world_weights(entry_format, field_size, script)
    names=list(weights); probs=np.array([weights[n] for n in names],dtype=float); probs/=probs.sum()
    return rng.choice(names,p=probs)

def game_world_bonus(row, world_name, influence=50):
    cfg=GAME_WORLDS.get(world_name,{}); fam=cfg.get("family"); scale=float(np.clip(influence,0,100))/100.0
    team=row["Team"]; b=0.0
    if fam=="shootout":
        if row["is_QB"]: b+=2.0
        if row["is_passcatcher"]: b+=1.5
        if row["is_DST"]: b-=1.2
    elif fam=="pass_ceiling":
        same=team==cfg.get("team")
        if same and row["is_QB"]: b+=3.2
        if same and row["is_passcatcher"]: b+=2.4
        if same and row["is_RB"]: b-=0.5
        if not same and (row["is_QB"] or row["is_passcatcher"]): b+=0.8
    elif fam=="qb_ceiling":
        if str(row["Name"])==cfg.get("player"): b+=4.2
        elif team==cfg.get("team") and row["is_passcatcher"]: b+=0.7
        elif team==cfg.get("team") and row["is_RB"]: b-=0.6
    elif fam=="rb_control":
        same=team==cfg.get("team")
        if same and row["is_RB"]: b+=3.0
        if same and (row["is_DST"] or row["is_K"]): b+=1.6
        if same and row["is_passcatcher"]: b-=0.5
        if not same and (row["is_QB"] or row["is_passcatcher"]): b+=0.8
    elif fam=="comeback":
        if team==cfg.get("trail") and row["is_QB"]: b+=2.5
        if team==cfg.get("trail") and row["is_passcatcher"]: b+=2.0
        if team==cfg.get("lead") and row["is_RB"]: b+=1.8
        if team==cfg.get("lead") and row["is_K"]: b+=0.7
    elif fam=="low":
        if row["is_DST"]: b+=2.8
        if row["is_K"]: b+=2.2
        if row["is_RB"]: b+=1.0
        if row["is_QB"] or row["is_passcatcher"]: b-=1.0
    return b*(0.45+0.85*scale)

def world_construction_weights(base_weights, world_name, entry_format):
    allowed={k:float(base_weights.get(k,0))>0 for k in ["3-3","4-2","5-1"]}
    fam=GAME_WORLDS.get(world_name,{}).get("family")
    if fam in ["shootout","comeback"]: desired={"3-3":58,"4-2":38,"5-1":4}
    elif fam in ["pass_ceiling","qb_ceiling"]: desired={"3-3":42,"4-2":50,"5-1":8}
    elif fam=="rb_control": desired={"3-3":30,"4-2":55,"5-1":15}
    else: desired={"3-3":34,"4-2":50,"5-1":16}
    if entry_format=="Single Entry": desired["5-1"]*=0.35; desired["3-3"]*=1.18
    elif entry_format=="150-Max": desired["5-1"]*=1.35
    return {k:(desired[k] if allowed[k] else 0.0) for k in desired}

def showdown_script_bonus(row, script, script_team):
    team = row["Team"]
    opp = row["Opponent"]
    same = team == script_team if script_team else False
    bonus = 0.0
    if script in ["Shootout", "Pass-heavy shootout"]:
        if row["is_QB"]: bonus += 1.8
        if row["is_passcatcher"]: bonus += 1.2
        if row["is_RB"]: bonus += 0.35
        if row["is_DST"]: bonus -= 1.1
    elif script in ["Low-scoring game", "Defensive / field-goal battle", "Ground-and-pound"]:
        if row["is_RB"]: bonus += 1.0
        if row["is_K"]: bonus += 1.1
        if row["is_DST"]: bonus += 1.2
        if row["is_QB"] or row["is_passcatcher"]: bonus -= 0.25
    elif script in ["Team dominates", "Team plays from ahead"]:
        if same and row["is_RB"]: bonus += 1.4
        if same and row["is_DST"]: bonus += 1.3
        if same and row["is_K"]: bonus += 0.8
        if same and row["is_QB"]: bonus += 0.35
        if (not same) and row["is_QB"]: bonus += 0.6
        if (not same) and row["is_passcatcher"]: bonus += 0.7
    elif script == "Team wins close":
        if same: bonus += 0.35
        if row["is_QB"] or row["is_passcatcher"] or row["is_RB"]: bonus += 0.25
    elif script == "Team passing comeback":
        if same and row["is_QB"]: bonus += 1.6
        if same and row["is_passcatcher"]: bonus += 1.15
        if same and row["is_RB"]: bonus -= 0.35
        if (not same) and row["is_RB"]: bonus += 0.8
        if (not same) and row["is_DST"]: bonus += 0.25
    return bonus

def infer_score_script(team_scores):
    """Translate a predicted final score into a broad Showdown game environment."""
    if not team_scores or len(team_scores) < 2:
        return "Neutral", "", {"total": 0, "margin": 0, "winner": "", "loser": ""}
    ordered = sorted(team_scores.items(), key=lambda kv: kv[1], reverse=True)
    winner, win_pts = ordered[0]
    loser, lose_pts = ordered[1]
    total = float(win_pts + lose_pts)
    margin = float(win_pts - lose_pts)
    if total <= 41:
        script = "Low-scoring game"
    elif total >= 55 and margin <= 10:
        script = "Pass-heavy shootout"
    elif margin >= 17:
        script = "Team dominates"
    elif margin >= 8:
        script = "Team plays from ahead"
    else:
        script = "Team wins close"
    return script, winner, {"total": total, "margin": margin, "winner": winner, "loser": loser}

def _scenario_intensity_scale(level):
    
    if isinstance(level, (int, float, np.integer, np.floating)):
        # 0 = no scenario effect, 50 = standard, 100 = strongest bounded effect.
        return float(np.clip(level, 0, 100)) / 50.0
    return {"Conservative": 0.65, "Standard": 1.0, "Aggressive": 1.30}.get(level, 1.0)

def score_projection_multiplier(row, team_scores, intensity="Standard"):
    """Conservative, transparent heuristic for translating a predicted score into role-based projection movement.
    It is intentionally bounded; the goal is to tilt a baseline projection, not replace a projection model.
    """
    if not team_scores or row["Team"] not in team_scores or row["Opponent"] not in team_scores:
        return 1.0
    team_pts = float(team_scores[row["Team"]])
    opp_pts = float(team_scores[row["Opponent"]])
    total = team_pts + opp_pts
    margin = team_pts - opp_pts
    scale = _scenario_intensity_scale(intensity)

    # Baselines roughly represent an ordinary NFL scoring environment. Effects are kept modest.
    team_env = np.clip((team_pts - 23.5) / 55.0, -0.12, 0.12)
    total_env = np.clip((total - 47.0) / 85.0, -0.09, 0.09)
    lead = np.clip(margin / 70.0, -0.12, 0.12)
    trail = np.clip((-margin) / 70.0, -0.12, 0.12)

    delta = 0.0
    if row["is_QB"]:
        delta += 0.75 * team_env + 0.75 * total_env + 0.45 * max(0.0, trail) - 0.20 * max(0.0, lead)
    elif row["is_passcatcher"]:
        delta += 0.75 * team_env + 0.90 * total_env + 0.65 * max(0.0, trail) - 0.18 * max(0.0, lead)
    elif row["is_RB"]:
        delta += 0.70 * team_env + 0.20 * total_env + 0.90 * max(0.0, lead) - 0.65 * max(0.0, trail)
    elif row["is_K"]:
        delta += 0.45 * team_env
        if total <= 44: delta += 0.045
        if 16 <= team_pts <= 29: delta += 0.030
        if team_pts < 13: delta -= 0.070
    elif row["is_DST"]:
        # Defense is driven more by opponent suppression than own-team scoring.
        delta += np.clip((20.5 - opp_pts) / 75.0, -0.13, 0.13)
        if total <= 42: delta += 0.045
        if margin >= 7: delta += 0.035

    delta = float(np.clip(delta * scale, -0.20, 0.20))
    return 1.0 + delta

def named_script_projection_multiplier(row, script, script_team, intensity="Standard"):
    """Small projection tilts for a user's football story. These stack with score-driven tilts."""
    scale = _scenario_intensity_scale(intensity)
    same = bool(script_team) and row["Team"] == script_team
    d = 0.0
    if script in ["Shootout", "Pass-heavy shootout"]:
        if row["is_QB"]: d += 0.055
        if row["is_passcatcher"]: d += 0.065
        if row["is_RB"]: d += 0.010
        if row["is_DST"]: d -= 0.065
    elif script in ["Low-scoring game", "Defensive / field-goal battle"]:
        if row["is_RB"]: d += 0.035
        if row["is_K"]: d += 0.060
        if row["is_DST"]: d += 0.075
        if row["is_QB"] or row["is_passcatcher"]: d -= 0.035
    elif script == "Ground-and-pound":
        if row["is_RB"]: d += 0.070
        if row["is_K"] or row["is_DST"]: d += 0.035
        if row["is_QB"] or row["is_passcatcher"]: d -= 0.025
    elif script in ["Team dominates", "Team plays from ahead"]:
        if same and row["is_RB"]: d += 0.070
        if same and row["is_DST"]: d += 0.070
        if same and row["is_K"]: d += 0.035
        if (not same) and row["is_QB"]: d += 0.035
        if (not same) and row["is_passcatcher"]: d += 0.045
        if same and row["is_passcatcher"] and script == "Team dominates": d -= 0.015
    elif script == "Team wins close":
        if same: d += 0.018
        if row["is_QB"] or row["is_passcatcher"] or row["is_RB"]: d += 0.012
    elif script == "Team passing comeback":
        if same and row["is_QB"]: d += 0.075
        if same and row["is_passcatcher"]: d += 0.065
        if same and row["is_RB"]: d -= 0.040
        if (not same) and row["is_RB"]: d += 0.045
    return 1.0 + float(np.clip(d * scale, -0.12, 0.12))

def apply_showdown_scenario(df, script, script_team, use_score=False, team_scores=None, intensity="Standard"):
    out = df.copy()
    multipliers = []
    for _, r in out.iterrows():
        m = named_script_projection_multiplier(r, script, script_team, intensity)
        if use_score:
            m *= score_projection_multiplier(r, team_scores or {}, intensity)
        multipliers.append(float(np.clip(m, 0.75, 1.25)))
    out["Scenario Mult"] = multipliers
    out["Script Proj"] = (out["My Proj"].astype(float) * out["Scenario Mult"]).round(3)
    out["Proj Change %"] = ((out["Scenario Mult"] - 1.0) * 100).round(1)
    return out

def script_build_adjustments(base_weights, script, script_team, use_score=False, team_scores=None, auto_shape=True):
    """Tilt only the construction shapes the user has allowed.
    4-2 and 5-1 are orientation-free; directional scripts choose the heavy team later.
    """
    allowed={k: float(base_weights.get(k,0)) > 0 for k in ["3-3","4-2","5-1"]}
    weights={k:(1.0 if allowed[k] else 0.0) for k in allowed}
    if not auto_shape:
        return weights, None

    margin=0.0; total=0.0
    if use_score and team_scores and len(team_scores)>=2:
        vals=sorted([float(v) for v in team_scores.values()], reverse=True)
        margin=vals[0]-vals[1]; total=sum(vals)

    if script in ["Shootout","Pass-heavy shootout"] or total>=55:
        desired={"3-3":62,"4-2":34,"5-1":4}; corr={"qb_pc":2,"wrte_qb":90,"rb_ctrl":30}
    elif script in ["Low-scoring game","Defensive / field-goal battle","Ground-and-pound"] or (use_score and total and total<=41):
        desired={"3-3":46,"4-2":46,"5-1":8}; corr={"qb_pc":1,"wrte_qb":70,"rb_ctrl":75}
    elif script=="Team dominates" or margin>=17:
        desired={"3-3":18,"4-2":56,"5-1":26}; corr={"qb_pc":1,"wrte_qb":70,"rb_ctrl":80}
    elif script=="Team plays from ahead" or margin>=8:
        desired={"3-3":34,"4-2":54,"5-1":12}; corr={"qb_pc":1,"wrte_qb":75,"rb_ctrl":70}
    elif script=="Team passing comeback":
        desired={"3-3":56,"4-2":40,"5-1":4}; corr={"qb_pc":2,"wrte_qb":90,"rb_ctrl":40}
    elif script=="Team wins close" or (use_score and margin<=6):
        desired={"3-3":62,"4-2":34,"5-1":4}; corr={"qb_pc":2,"wrte_qb":85,"rb_ctrl":50}
    else:
        desired={"3-3":50,"4-2":45,"5-1":5}; corr=None
    weights={k:(desired[k] if allowed[k] else 0.0) for k in desired}
    if sum(weights.values())<=0: weights={"3-3":1.0,"4-2":0.0,"5-1":0.0}
    return weights,corr

def apply_context_engine(df, context_map=None, strength="Standard"):
    """Explainable context layer. Ratings are deliberately bounded and confidence-shrunk.
    This is the V5.2 foundation for future automated defensive/usage/travel/split feeds.
    """
    out = df.copy()
    context_map = context_map or {}
    strength_scale = {"Conservative":0.65, "Standard":1.0, "Aggressive":1.25}.get(strength,1.0)
    adjs=[]; reasons=[]
    for _, r in out.iterrows():
        cfg=context_map.get(str(r["ID"]),{})
        confidence=float(cfg.get("Confidence",50))/100.0
        total=0.0; parts=[]
        for key,w in CONTEXT_FACTOR_WEIGHTS.items():
            rating=int(cfg.get(key,0) or 0)
            delta=rating*w*confidence*strength_scale
            total += delta
            if rating:
                parts.append(f"{key}: {CONTEXT_RATING_LABELS.get(rating,str(rating))} ({delta*100:+.1f}%)")
        # Prevent a collection of small/contextual splits from overwhelming the baseline model.
        total=float(np.clip(total,-0.18,0.18))
        adjs.append(total)
        note=str(cfg.get("Note","") or "").strip()
        reason=" • ".join(parts) if parts else "No context adjustment"
        if note: reason += f" • {note}"
        reasons.append(reason)
    out["Context Adj %"]=(np.array(adjs)*100).round(1)
    base_col="Script Proj" if "Script Proj" in out.columns else "My Proj"
    out["DFS Lab Proj"]=(out[base_col].astype(float)*(1.0+np.array(adjs))).round(3)
    out["Context Why"]=reasons
    return out

def showdown_base_objective(df, aggr, strategy_map, script, script_team, exposure_state=None, cpt_exposure_state=None, built_count=0, game_world=None, world_influence=50):
    proj_col = "DFS Lab Proj" if "DFS Lab Proj" in df.columns else ("Script Proj" if "Script Proj" in df.columns else "My Proj")
    proj = df[proj_col].to_numpy(float)
    own = np.clip(df["My Own"].to_numpy(float), 0.1, None)
    objective = proj.copy()
    ownership_available = bool(pd.to_numeric(df.get("My Own",0),errors="coerce").fillna(0).max() > 0.01)
    if ownership_available:
        leverage = np.log((proj + 2.0) / (own + 2.0))
        objective += (0.65 + 2.0 * aggr) * leverage

    # Contest-aware role confidence. Single-entry builds pay a larger price for
    # thin, low-projection salary punts; large-field builds may accept them when
    # they unlock a coherent ceiling construction. This is a soft penalty, never a ban.
    hist = pd.to_numeric(df.get("History Games", pd.Series(0,index=df.index)),errors="coerce").fillna(0).to_numpy(float)
    curg = pd.to_numeric(df.get("Current Games", pd.Series(0,index=df.index)),errors="coerce").fillna(0).to_numpy(float)
    sal = pd.to_numeric(df.get("FlexSalary",0),errors="coerce").fillna(0).to_numpy(float)
    role_conf = np.clip((hist/24.0)*0.65 + (curg/6.0)*0.35, 0, 1)
    punt = (sal <= 2200) & (proj < 4.0)
    objective -= punt.astype(float) * (1.0-role_conf) * (2.8 - 1.9*aggr)

    for i, r in df.iterrows():
        strat = strategy_map.get(str(r["ID"]), {})
        objective[i] += PRIORITY_BONUS.get(strat.get("Priority", "Neutral"), 0.0)
        objective[i] += showdown_script_bonus(r, script, script_team)
        if game_world: objective[i] += game_world_bonus(r, game_world, world_influence)
        if exposure_state is not None and built_count > 0:
            current = 100.0 * exposure_state.get(str(r["ID"]), 0) / built_count
            mn = float(strat.get("Min Exposure", 0)); mx = float(strat.get("Max Exposure", 100))
            if current < mn: objective[i] += min(5.5, 0.11 * (mn-current))
            if current > mx - 5: objective[i] -= min(5.0, 0.12 * max(0, current-(mx-5)))
    return objective

def _add_constraint(rows, lows, highs, coeff, low, high):
    rows.append(coeff); lows.append(low); highs.append(high)

def solve_showdown_one(
    df, aggr, rng, strategy_map, min_salary, max_salary, construction_target,
    script, script_team, cpt_qb_passcatchers, wrte_cpt_qb_pair_pct, rb_cpt_dst_k_pct,
    max_k, max_dst, min_unique, previous_lineups,
    exposure_state=None, cpt_exposure_state=None, built_count=0, noise_scale=0.18,
    relationship_rules=None, forced_overall_ids=None, forced_cpt_ids=None, game_world=None, world_influence=50
):
    n = len(df); s = len(SHOWDOWN_SLOTS); total_vars = n*s
    c = np.zeros(total_vars); lb = np.zeros(total_vars); ub = np.ones(total_vars); integrality = np.ones(total_vars)
    active = df["ActiveForBuild"].to_numpy(bool)
    base = showdown_base_objective(df, aggr, strategy_map, script, script_team, exposure_state, cpt_exposure_state, built_count, game_world, world_influence)
    noise = np.exp(rng.normal(0, noise_scale, size=n))

    def vidx(i,j): return i*s+j

    # Objective and bounds. Captain objective includes 1.5 scoring plus captain-specific leverage.
    for i, r in df.iterrows():
        strat = strategy_map.get(str(r["ID"]), {})
        excluded = bool(strat.get("Exclude", False)) or strat.get("Priority") == "Exclude"
        # A captain-locked player is by definition captain-eligible: the lock is
        # the stronger, more explicit intent ("WILL captain" beats "may captain").
        # Without this, CPT Lock=True + CPT Eligible=False forces the player into
        # the CPT slot (hard constraint below) while the slot's upper bound is
        # 0 -- an infeasible model.
        cpt_ok = (bool(strat.get("CPT Eligible", True)) or bool(strat.get("CPT Lock", False))) and not excluded
        for j, slot in enumerate(SHOWDOWN_SLOTS):
            if not active[i] or excluded:
                ub[vidx(i,j)] = 0
                continue
            if slot == "CPT" and not cpt_ok:
                ub[vidx(i,j)] = 0
            if slot == "CPT":
                cpt_own = max(float(r["CPT Own"]), 0.1)
                cpt_proj = float(r["DFS Lab Proj"] if "DFS Lab Proj" in r.index else (r["Script Proj"] if "Script Proj" in r.index else r["My Proj"]))
                cpt_lev = math.log((1.5*cpt_proj+2)/(cpt_own+1.5))
                val = 1.5*base[i] + (0.5 + 1.5*aggr)*cpt_lev
                # Position priors are soft, never hard rules.
                if r["is_WR"]: val += 0.55
                if r["is_TE"]: val += 0.25
                if r["is_QB"]: val += 0.10
                if r["is_K"] or r["is_DST"]: val -= 0.35
                # Captain exposure steering.
                if cpt_exposure_state is not None and built_count > 0:
                    curr = 100*cpt_exposure_state.get(str(r["ID"]),0)/built_count
                    mn = float(strat.get("CPT Min",0)); mx = float(strat.get("CPT Max",100))
                    if curr < mn: val += min(6.0, .13*(mn-curr))
                    if curr > mx-4: val -= min(6.0, .14*max(0,curr-(mx-4)))
                c[vidx(i,j)] = -val*noise[i]
            else:
                c[vidx(i,j)] = -base[i]*noise[i]

    rows=[]; lows=[]; highs=[]
    # One player per slot.
    for j in range(s):
        _add_constraint(rows,lows,highs,{vidx(i,j):1.0 for i in range(n)},1,1)
    # Player at most once. Group by DraftKings ID as a second safety net in case
    # an upstream file ever contains duplicate player rows.
    for pid, idxs in df.groupby(df["ID"].astype(str)).groups.items():
        coeff={}
        for i in idxs:
            for j in range(s):
                coeff[vidx(int(i),j)] = 1.0
        _add_constraint(rows,lows,highs,coeff,0,1)

    # Overall locks and captain locks.
    for i,r in df.iterrows():
        strat = strategy_map.get(str(r["ID"]), {})
        if bool(strat.get("CPT Lock", False)) and not bool(strat.get("Exclude", False)):
            _add_constraint(rows,lows,highs,{vidx(i,0):1.0},1,1)
        elif bool(strat.get("Lock", False)) and not bool(strat.get("Exclude", False)):
            _add_constraint(rows,lows,highs,{vidx(i,j):1.0 for j in range(s)},1,1)

    # Portfolio minimum exposure enforcement. When a target becomes mathematically
    # due, the player is forced into the current solve so minimums are real targets,
    # not just preference boosts.
    forced_overall_ids=set(forced_overall_ids or [])
    forced_cpt_ids=set(forced_cpt_ids or [])
    for i,r in df.iterrows():
        pid=str(r["ID"])
        if pid in forced_cpt_ids:
            _add_constraint(rows,lows,highs,{vidx(i,0):1.0},1,1)
        elif pid in forced_overall_ids:
            _add_constraint(rows,lows,highs,{vidx(i,j):1.0 for j in range(s)},1,1)

    # Relationship Rules Engine. A/B selectors can represent a player or a
    # team-position group. Hard rules are translated directly into MILP constraints.
    def selected_coeff(indices):
        out={}
        for ii in indices:
            for jj in range(s): out[vidx(int(ii),jj)]=1.0
        return out
    def resolve_side(side):
        if not side: return []
        if side.get("kind")=="Player":
            return df.index[df["ID"].astype(str).eq(str(side.get("id","")))].tolist()
        team=side.get("team","Any"); pos=side.get("position","Any")
        mask=df["ActiveForBuild"].copy()
        if team!="Any": mask &= df["Team"].eq(team)
        if pos!="Any": mask &= df["Position"].eq(pos)
        return df.index[mask].tolist()
    for rule in (relationship_rules or []):
        if not rule.get("enabled",True): continue
        aidx=resolve_side(rule.get("a")); bidx=resolve_side(rule.get("b"))
        if not aidx or not bidx: continue
        rtype=rule.get("rule","Never Together")
        if rtype=="Never Together":
            # Pairwise prevents any selected A from appearing with any selected B.
            for ai in aidx:
                for bi in bidx:
                    if ai==bi: continue
                    coeff={}
                    for jj in range(s):
                        coeff[vidx(int(ai),jj)]=coeff.get(vidx(int(ai),jj),0)+1
                        coeff[vidx(int(bi),jj)]=coeff.get(vidx(int(bi),jj),0)+1
                    _add_constraint(rows,lows,highs,coeff,0,1)
        elif rtype=="Require B when A used":
            bcoeff=selected_coeff(bidx)
            for ai in aidx:
                coeff=dict(bcoeff)
                for jj in range(s): coeff[vidx(int(ai),jj)]=coeff.get(vidx(int(ai),jj),0)-1
                _add_constraint(rows,lows,highs,coeff,0,np.inf)

    # Salary by slot.
    salary_coeff={}
    for i,r in df.iterrows():
        salary_coeff[vidx(i,0)] = float(r["CaptainSalary"])
        for j in range(1,s): salary_coeff[vidx(i,j)] = float(r["FlexSalary"])
    _add_constraint(rows,lows,highs,salary_coeff,float(min_salary),float(max_salary))

    # At least one from each team and optional exact construction.
    teams = [t for t in df["Team"].dropna().unique().tolist() if t]
    if len(teams) >= 2:
        for t in teams[:2]:
            coeff={}
            for i in df.index[df["Team"].eq(t)]:
                for j in range(s): coeff[vidx(i,j)] = 1.0
            _add_constraint(rows,lows,highs,coeff,1,5)
        if construction_target:
            team0, count0 = construction_target
            coeff={}
            for i in df.index[df["Team"].eq(team0)]:
                for j in range(s): coeff[vidx(i,j)] = 1.0
            _add_constraint(rows,lows,highs,coeff,count0,count0)

    # Position caps.
    for mask,maxn in [(df["is_K"],max_k),(df["is_DST"],max_dst)]:
        coeff={}
        for i in df.index[mask]:
            for j in range(s): coeff[vidx(i,j)] = 1.0
        if coeff: _add_constraint(rows,lows,highs,coeff,0,float(maxn))

    # Captain-specific correlation rules. Use randomized enforcement for percentage-based rules.
    for cpt in df.index[df["ActiveForBuild"]]:
        r=df.loc[cpt]
        # QB captain -> optional minimum/maximum same-team pass-catcher rule.
        if r["is_QB"] and cpt_qb_passcatchers != 0:
            pcs=df.index[df["ActiveForBuild"] & df["Team"].eq(r["Team"]) & df["is_passcatcher"]].tolist()
            n_pc=abs(int(cpt_qb_passcatchers))
            coeff={}
            for i in pcs:
                for j in range(1,s): coeff[vidx(i,j)] = coeff.get(vidx(i,j),0)+1
            if cpt_qb_passcatchers > 0:
                # Minimum N: when this QB is Captain, require at least N same-team pass catchers.
                coeff[vidx(cpt,0)] = coeff.get(vidx(cpt,0),0)-float(n_pc)
                _add_constraint(rows,lows,highs,coeff,0,np.inf)
            else:
                # No more than N: when this QB is Captain, cap same-team pass catchers at N.
                max_flex=float(s-1)
                coeff[vidx(cpt,0)] = coeff.get(vidx(cpt,0),0)+(max_flex-float(n_pc))
                _add_constraint(rows,lows,highs,coeff,-np.inf,max_flex)

        # WR/TE captain -> same-team QB.
        # Reliability rule: "Sometimes" and "Usually" are portfolio preferences, NOT hard
        # MILP constraints. Only "Always" may make a lineup infeasible. Random hard rules
        # were causing entire Showdown builds to fail depending on which captain branches
        # happened to be sampled.
        if r["is_passcatcher"] and float(wrte_cpt_qb_pair_pct) >= 99.5:
            qbs=df.index[df["ActiveForBuild"] & df["Team"].eq(r["Team"]) & df["is_QB"]].tolist()
            if qbs:
                coeff={}
                for q in qbs:
                    for j in range(1,s): coeff[vidx(q,j)] = coeff.get(vidx(q,j),0)+1
                coeff[vidx(cpt,0)] = coeff.get(vidx(cpt,0),0)-1
                _add_constraint(rows,lows,highs,coeff,0,np.inf)
            else:
                # If no active same-team QB exists, this player simply cannot be Captain
                # under an explicit Always rule.
                _add_constraint(rows,lows,highs,{vidx(cpt,0):1.0},0,0)

        # RB captain -> same-team DST/K. As above, only "Always" is a hard constraint.
        if r["is_RB"] and float(rb_cpt_dst_k_pct) >= 99.5:
            partners=df.index[df["ActiveForBuild"] & df["Team"].eq(r["Team"]) & (df["is_DST"]|df["is_K"])].tolist()
            if partners:
                coeff={}
                for p in partners:
                    for j in range(1,s): coeff[vidx(p,j)] = coeff.get(vidx(p,j),0)+1
                coeff[vidx(cpt,0)] = coeff.get(vidx(cpt,0),0)-1
                _add_constraint(rows,lows,highs,coeff,0,np.inf)
            else:
                _add_constraint(rows,lows,highs,{vidx(cpt,0):1.0},0,0)

    # Portfolio uniqueness relative to previously accepted lineups.
    if min_unique > 0:
        for prev in previous_lineups:
            coeff={}
            for i in prev:
                for j in range(s): coeff[vidx(i,j)] = 1.0
            _add_constraint(rows,lows,highs,coeff,0,6-min_unique)

    A=lil_matrix((len(rows),total_vars),dtype=float)
    for rr,coeff in enumerate(rows):
        for col,val in coeff.items(): A[rr,col]=val
    # Reliability guard: a single impossible/over-constrained Showdown branch must not
    # freeze the whole Streamlit session. HiGHS gets a short solve window; the portfolio
    # loop then tries the other allowed constructions and game worlds.
    result=milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb,ub),
        constraints=LinearConstraint(A.tocsr(),np.array(lows),np.array(highs)),
        options={"time_limit": 2.5, "mip_rel_gap": 0.02, "presolve": True},
    )
    # A short Showdown solve can time out without an incumbent and look identical to
    # "infeasible" to the caller. Captain-pairing constraints add enough rows that this
    # can happen on otherwise legal slates. Retry the SAME rules with a longer window
    # before declaring the branch impossible; no user setting is relaxed here.
    if result.x is None:
        result=milp(
            c=c,
            integrality=integrality,
            bounds=Bounds(lb,ub),
            constraints=LinearConstraint(A.tocsr(),np.array(lows),np.array(highs)),
            options={"time_limit": 8.0, "mip_rel_gap": 0.05, "presolve": True},
        )
    if result.x is None:
        # Last-resort FEASIBILITY solve using the exact same bounds and constraints.
        # This intentionally drops the optimization objective but changes NO user rule.
        # A zero-objective MILP is much easier for HiGHS to satisfy and prevents a hard
        # but legal Showdown branch from being mislabeled as "no legal lineup."
        result=milp(
            c=np.zeros(total_vars,dtype=float),
            integrality=integrality,
            bounds=Bounds(lb,ub),
            constraints=LinearConstraint(A.tocsr(),np.array(lows),np.array(highs)),
            options={"time_limit": 6.0, "mip_rel_gap": 0.10, "presolve": True},
        )
    if result.x is None:
        return None

    # HiGHS may hit the short time limit after already finding a valid incumbent.
    # Do not throw that lineup away merely because optimality was not proven.
    # Instead round the incumbent and verify EVERY hard MILP constraint ourselves.
    xbin=(np.asarray(result.x,dtype=float)>=0.5).astype(float)
    if np.any(xbin < lb-1e-9) or np.any(xbin > ub+1e-9):
        return None
    lhs=np.asarray(A.tocsr().dot(xbin)).reshape(-1)
    low_arr=np.asarray(lows,dtype=float); high_arr=np.asarray(highs,dtype=float)
    if np.any(lhs < low_arr-1e-7) or np.any(lhs > high_arr+1e-7):
        return None

    chosen=[]
    for j,slot in enumerate(SHOWDOWN_SLOTS):
        slot_hits=[i for i in range(n) if xbin[vidx(i,j)]>0.5]
        if len(slot_hits)!=1:
            return None
        chosen.append((slot,int(slot_hits[0])))

    # Final safety validation before a lineup is ever shown to the user.
    chosen_ids=[str(df.loc[i,"ID"]) for _,i in chosen]
    if len(chosen_ids) != len(SHOWDOWN_SLOTS) or len(set(chosen_ids)) != len(SHOWDOWN_SLOTS):
        return None
    cpt_i=[i for slot,i in chosen if slot=="CPT"][0]
    flex_i=[i for slot,i in chosen if slot!="CPT"]
    salary=int(df.loc[cpt_i,"CaptainSalary"])+int(df.loc[flex_i,"FlexSalary"].sum())
    if salary < int(min_salary) or salary > int(max_salary):
        return None
    return chosen

def showdown_lineup_coherence(df, chosen, game_world=None):
    """Reject Showdown constructions that are legal on DK but contradict their own football story."""
    slot_map={slot:i for slot,i in chosen}
    idxs=[i for _,i in chosen]
    p=df.loc[idxs].copy()
    cpt=df.loc[slot_map["CPT"]]
    hard=[]; warnings=[]; score=100.0

    if "Role Confidence" in p.columns:
        bad=p[p["Role Confidence"].astype(str).isin(["Backup QB","Inactive / no usable projection"])]
        if not bad.empty:
            hard.append("Contains a player DFS LAB marked unavailable for the current football role.")

    # Defense cannot reasonably be a ceiling piece while the lineup also needs a
    # full opposing passing stack to smash.
    for _,dst in p[p["is_DST"]].iterrows():
        opp=str(dst.get("Opponent",""))
        opp_off=p[(p["Team"].astype(str).eq(opp)) & (~p["is_DST"])]
        opp_qb=opp_off[opp_off["is_QB"]]
        opp_pc=opp_off[opp_off["is_WR"]|opp_off["is_TE"]]
        if len(opp_qb)>=1 and len(opp_pc)>=2:
            hard.append(f"{dst['Name']} conflicts with an opposing QB plus two pass catchers.")
        elif bool(cpt["is_DST"]) and str(cpt["Name"])==str(dst["Name"]) and len(opp_off)>=3:
            hard.append(f"{dst['Name']} at Captain conflicts with three or more opposing offensive players.")
        elif len(opp_off)>=3:
            score-=18; warnings.append(f"{dst['Name']} needs to survive three opposing offensive pieces")

    cfg=GAME_WORLDS.get(game_world,{})
    fam=cfg.get("family")
    if fam=="pass_ceiling":
        team=str(cfg.get("team",""))
        relevant=p[(p["Team"].astype(str).eq(team)) & (p["is_QB"]|p["is_passcatcher"])]
        if relevant.empty:
            score-=24; warnings.append(f"{game_world} is only a weak match because it lacks a {team} passing piece")
        else:
            score+=3
    elif fam=="rb_control":
        team=str(cfg.get("team",""))
        if p[(p["Team"].astype(str).eq(team)) & p["is_RB"]].empty:
            score-=20; warnings.append(f"{game_world} is only a weak match because it lacks a {team} running back")
        else:
            score+=3
    elif fam=="comeback":
        trail=str(cfg.get("trail",""))
        if p[(p["Team"].astype(str).eq(trail)) & (p["is_QB"]|p["is_passcatcher"])].empty:
            score-=20; warnings.append(f"{game_world} is only a weak match because it lacks a {trail} passing piece")
        else:
            score+=3
    elif fam=="qb_ceiling":
        player=str(cfg.get("player",""))
        if player and not p["Name"].astype(str).eq(player).any():
            score-=24; warnings.append(f"{game_world} is only a weak match because it does not contain {player}")
        elif player:
            score+=4

    # Captain-specific sense checks. These are soft because leverage can justify
    # unconventional but still plausible structures.
    cteam=str(cpt["Team"])
    if cpt["is_QB"]:
        pcs=p[(p["Team"].astype(str).eq(cteam)) & p["is_passcatcher"]]
        if pcs.empty:
            score-=15; warnings.append("QB Captain needs a concentrated rushing/TD path without a same-team pass catcher")
        else:
            score+=min(5,2*len(pcs))
    elif cpt["is_passcatcher"]:
        has_qb=bool(((p["Team"].astype(str).eq(cteam)) & p["is_QB"]).any())
        if not has_qb:
            score-=10; warnings.append("WR/TE Captain omits his quarterback and needs concentrated receiving leverage")
    elif cpt["is_RB"]:
        control=bool(((p["Team"].astype(str).eq(cteam)) & (p["is_DST"]|p["is_K"])).any())
        if control: score+=3

    score=float(np.clip(score,0,100))
    world_txt=GAME_WORLDS.get(game_world,{}).get("desc","A plausible single-game scoring path")
    story=f"{game_world or 'Showdown game'}: {world_txt} Captain {cpt['Name']} is the ceiling engine."
    return {
        "Accept":len(hard)==0,
        "Coherence Score":round(score,1),
        "Lineup Story":story,
        "Coherence Flags":"; ".join(hard+warnings) if (hard or warnings) else "No major football contradictions"
    }

def showdown_lineup_details(df, chosen, strategy_map, script, script_team, game_world=None):
    slot_map={slot:i for slot,i in chosen}
    idxs=[i for _,i in chosen]
    cpt_i=slot_map["CPT"]
    cpt=df.loc[cpt_i]
    flex_idxs=[i for slot,i in chosen if slot!="CPT"]
    p=df.loc[idxs]
    proj_col = "DFS Lab Proj" if "DFS Lab Proj" in df.columns else ("Script Proj" if "Script Proj" in df.columns else "My Proj")
    projection=1.5*float(cpt[proj_col])+float(df.loc[flex_idxs,proj_col].sum())
    base_projection=1.5*float(cpt["My Proj"])+float(df.loc[flex_idxs,"My Proj"].sum())
    salary=int(cpt["CaptainSalary"])+int(df.loc[flex_idxs,"FlexSalary"].sum())
    total_own=float(p["My Own"].sum())

    # Popularity / duplication proxy: log joint ownership + salary usage. Relative ranking is used later.
    cpt_prob=max(float(cpt["CPT Own"]),0.1)/100.0
    flex_probs=[max(float(df.loc[i,"My Own"]),0.1)/100.0 for i in flex_idxs]
    log_pop=math.log(cpt_prob)+sum(math.log(x) for x in flex_probs)
    salary_left=50000-salary
    dup_raw=log_pop - 0.00022*salary_left

    corr=0.0; notes=[]
    cpt_team=cpt["Team"]
    if cpt["is_QB"]:
        n_pc=int(((p["Team"]==cpt_team)&p["is_passcatcher"]).sum())
        corr += 1.7*min(n_pc,3); notes.append(f"QB CPT + {n_pc} pass catcher(s)")
    elif cpt["is_passcatcher"]:
        has_qb=bool(((p["Team"]==cpt_team)&p["is_QB"]).any())
        corr += 2.0 if has_qb else 0.4
        notes.append("CPT paired with QB" if has_qb else "WR/TE CPT without QB leverage")
    elif cpt["is_RB"]:
        has_control=bool(((p["Team"]==cpt_team)&(p["is_DST"]|p["is_K"])).any())
        corr += 1.4 if has_control else 0.5
        if has_control: notes.append("RB CPT + team control piece")
    if script_team:
        script_count=int((p["Team"]==script_team).sum())
        corr += 0.25*script_count
    if salary_left>=500: corr += min(1.2,salary_left/2500)

    fit=0.0
    for i in idxs:
        pr=strategy_map.get(str(df.loc[i,"ID"]),{}).get("Priority","Neutral")
        if pr=="Core": fit+=2.2
        elif pr=="Like": fit+=1.0
        elif pr=="Fade": fit-=1.3
    cpt_pr=strategy_map.get(str(cpt["ID"]),{}).get("Priority","Neutral")
    if cpt_pr in ["Core","Like"]: fit += 0.8

    teams=p["Team"].value_counts().to_dict()
    construction="-".join(str(v) for v in sorted(teams.values(), reverse=True)) if teams else ""
    story=f"{script}"
    if script_team: story += f" • {script_team}"
    story += f" • {cpt['Name']} CPT • {construction}"

    return {
        "Projection":round(projection,2),"Base Projection":round(base_projection,2),"Scenario Delta":round(projection-base_projection,2),"Salary":salary,"Salary Left":salary_left,
        "Total Own":round(total_own,1),"CPT Own":round(float(cpt["CPT Own"]),1),
        "Correlation Raw":round(corr,2),"User Fit Raw":round(fit,2),
        "Dup Raw":dup_raw,"Captain":cpt["Name"],"Captain Pos":cpt["Position"],
        "Construction":construction,"Story":story,"Game World":game_world or script,"World Thesis":GAME_WORLDS.get(game_world,{}).get("desc", "Game-script build"),"Strategy Notes":"; ".join(notes) if notes else "Game-script build"
    }

def add_showdown_ratings(out, aggr):
    if out.empty: return out
    proj=out["Projection"].rank(pct=True)
    captain=(out["Projection"] - 0.5*out["Salary Left"]/1000).rank(pct=True)
    corr=out["Correlation Raw"].rank(pct=True)
    ownership_available=bool(pd.to_numeric(out["Total Own"],errors="coerce").fillna(0).max()>0.01)
    lev=(-out["Total Own"]).rank(pct=True) if ownership_available else pd.Series(0.5,index=out.index)
    dup=(-out["Dup Raw"]).rank(pct=True) if ownership_available else pd.Series(0.5,index=out.index)
    fit=out["User Fit Raw"].rank(pct=True)
    coh=out["Coherence Score"].rank(pct=True) if "Coherence Score" in out.columns else pd.Series(0.5,index=out.index)
    w_proj=0.30-0.04*aggr; w_cpt=.17; w_corr=.18; w_lev=(.10+.04*aggr) if ownership_available else 0.0; w_dup=(.09+.05*aggr) if ownership_available else 0.0; w_fit=.06; w_coh=.12
    comp=(w_proj*proj+w_cpt*captain+w_corr*corr+w_lev*lev+w_dup*dup+w_fit*fit+w_coh*coh)/(w_proj+w_cpt+w_corr+w_lev+w_dup+w_fit+w_coh)
    rel=comp.rank(pct=True,method="average")
    def grade(p):
        if p>=.95:return "A+"
        if p>=.85:return "A"
        if p>=.70:return "A-"
        if p>=.50:return "B+"
        if p>=.30:return "B"
        if p>=.15:return "B-"
        if p>=.05:return "C+"
        return "C"
    out["Rating Score"]=(68+31*rel).round(1); out["Rating"]=[grade(x) for x in rel]
    out["Projection Grade"]=[percentile_label(x,out["Projection"]) for x in out["Projection"]]
    out["Captain Grade"]=[percentile_label(x,out["Projection"]-0.5*out["Salary Left"]/1000) for x in out["Projection"]-0.5*out["Salary Left"]/1000]
    out["Correlation Grade"]=[percentile_label(x,out["Correlation Raw"]) for x in out["Correlation Raw"]]
    out["Leverage Grade"]=[percentile_label(-x,-out["Total Own"]) for x in out["Total Own"]] if ownership_available else ["Unavailable"]*len(out)
    out["Duplication Grade"]=[percentile_label(-x,-out["Dup Raw"]) for x in out["Dup Raw"]] if ownership_available else ["Unavailable"]*len(out)
    q1=out["Dup Raw"].quantile(.33); q2=out["Dup Raw"].quantile(.67)
    out["Dup Risk"]=["Low" if x<=q1 else "Medium" if x<=q2 else "High" for x in out["Dup Raw"]]
    return out

def choose_construction_target(rng, teams, weights, script, script_team):
    # 4-2 means four players from either team; 5-1 means five from either team.
    if len(teams)!=2: return None
    labels=["3-3","4-2","5-1"]
    probs=np.array([max(0,float(weights.get(k,0))) for k in labels],dtype=float)
    if probs.sum()<=0: probs=np.array([1,0,0],dtype=float)
    probs=probs/probs.sum()
    label=rng.choice(labels,p=probs)
    a,b=map(int,label.split("-"))
    if a==b: return (teams[0],a)
    directional=script in ["Team dominates","Team plays from ahead","Team wins close"] and script_team in teams
    heavy_team=script_team if directional else (teams[0] if rng.random()<0.5 else teams[1])
    return (heavy_team,max(a,b))

def generate_showdown_lineups(df, field_size, payout_style, count, attempts, min_salary, max_salary,
                              construction_weights, script, script_team, strategy_map,
                              entry_format,
                              cpt_qb_passcatchers, wrte_cpt_qb_pair_pct, rb_cpt_dst_k_pct,
                              max_k, max_dst, min_unique, seed, relationship_rules=None, world_influence=50,
                              show_progress=True):
    """Generate a Showdown portfolio without wasting attempts on one randomly chosen build shape.

    V6.5 reliability change: every attempt now tries every user-allowed team construction
    (and both 4-2 / 5-1 orientations) before declaring that attempt infeasible. This keeps
    Game Worlds as preference/ordering, but a single impossible sampled construction can no
    longer make DFS LAB report "no legal lineup" when another allowed construction is legal.
    """
    aggr=showdown_aggression(field_size,payout_style,entry_format); rng=np.random.default_rng(seed)
    teams=[t for t in df["Team"].dropna().unique().tolist() if t]
    rows=[]; exposure=defaultdict(int); cpt_exp=defaultdict(int); previous=[]; seen=set()

    def _allowed_targets():
        if len(teams)!=2:
            return [None]
        out=[]
        if float(construction_weights.get("3-3",0))>0:
            # 3-3 has no meaningful orientation.
            out.append((teams[0],3))
        if float(construction_weights.get("4-2",0))>0:
            out.extend([(teams[0],4),(teams[1],4)])
        if float(construction_weights.get("5-1",0))>0:
            out.extend([(teams[0],5),(teams[1],5)])
        return out or [(teams[0],3)]

    legal_targets=_allowed_targets()
    prog=st.progress(0,text="Building Showdown lineups...") if show_progress else None
    last_built_attempt=-1
    for attempt in range(attempts):
        if len(rows)>=count: break
        # Stop a dead build instead of leaving the iPad on an endless progress bar.
        # Forty consecutive attempts without accepting a lineup is enough evidence that
        # the current hard-rule combination has stalled.
        if attempt - last_built_attempt > 40:
            break

        game_world=choose_game_world(rng,entry_format,field_size,script)
        world_weights=world_construction_weights(construction_weights,game_world,entry_format)
        preferred=choose_construction_target(rng,teams,world_weights,script,script_team)

        # Try the Game World's preferred shape first, then every other shape the user
        # explicitly allowed. This is deterministic feasibility fallback, not a relaxation.
        candidates=[]
        if preferred in legal_targets:
            candidates.append(preferred)
        rest=[x for x in legal_targets if x not in candidates]
        if len(rest)>1:
            order=rng.permutation(len(rest))
            rest=[rest[int(i)] for i in order]
        candidates.extend(rest)

        # Exact minimum-exposure scheduling: only force a minimum when all remaining
        # accepted lineups are needed to reach it.
        remaining=count-len(rows)
        forced_overall=[]; forced_cpt=[]
        for pid,strat in strategy_map.items():
            need=max(0, int(math.ceil(float(strat.get("Min Exposure",0))*count/100.0))-exposure[pid])
            cneed=max(0, int(math.ceil(float(strat.get("CPT Min",0))*count/100.0))-cpt_exp[pid])
            if need>=remaining and need>0: forced_overall.append(pid)
            if cneed>=remaining and cneed>0: forced_cpt.append(pid)

        chosen=None
        for target in candidates:
            chosen=solve_showdown_one(
                df,aggr,rng,strategy_map,min_salary,max_salary,target,script,script_team,
                cpt_qb_passcatchers,wrte_cpt_qb_pair_pct,rb_cpt_dst_k_pct,max_k,max_dst,
                min_unique,previous,exposure,cpt_exp,len(rows),noise_scale=.14+.13*aggr,
                relationship_rules=relationship_rules,forced_overall_ids=forced_overall,
                forced_cpt_ids=forced_cpt,game_world=game_world,world_influence=world_influence
            )
            if chosen:
                break
        if not chosen:
            continue

        ids=tuple(sorted(str(df.loc[i,"ID"]) for _,i in chosen))
        cpt_id=str(df.loc[[i for slot,i in chosen if slot=="CPT"][0],"ID"])
        key=(cpt_id,ids)
        if key in seen: continue

        # Exposure hard-ish caps.
        reject=False
        if len(rows)>=8:
            for slot,i in chosen:
                pid=str(df.loc[i,"ID"]); strat=strategy_map.get(pid,{})
                overall=100*(exposure[pid]+1)/(len(rows)+1)
                if overall>float(strat.get("Max Exposure",100))+3: reject=True; break
                if slot=="CPT":
                    cexp=100*(cpt_exp[pid]+1)/(len(rows)+1)
                    if cexp>float(strat.get("CPT Max",100))+3: reject=True; break
        if reject: continue

        coherence=showdown_lineup_coherence(df,chosen,game_world)
        if not coherence["Accept"]:
            continue

        seen.add(key)
        detail=showdown_lineup_details(df,chosen,strategy_map,script,script_team,game_world)
        detail.update({
            "Coherence Score":coherence["Coherence Score"],
            "Lineup Story":coherence["Lineup Story"],
            "Coherence Flags":coherence["Coherence Flags"]
        })
        row=dict(detail)
        for slot,i in chosen:
            row[slot]=df.loc[i,"Name"]; row[slot+"_ID"]=str(df.loc[i,"ID"])
            row[slot+"_NameID"] = df.loc[i,"CPT_NameID"] if slot=="CPT" else df.loc[i,"FLEX_NameID"]
        rows.append(row); previous.append([i for _,i in chosen])
        last_built_attempt=attempt
        for slot,i in chosen:
            pid=str(df.loc[i,"ID"]); exposure[pid]+=1
            if slot=="CPT": cpt_exp[pid]+=1
        if prog is not None:
            prog.progress(min(1.0,len(rows)/max(1,count)), text=f"Built {len(rows)} / {count} · attempt {attempt+1}")

    if prog is not None:
        prog.empty()
    out=pd.DataFrame(rows)
    if out.empty:return out
    out=add_showdown_ratings(out,aggr)
    out["Contest Format"]=entry_format
    out["Field Size"]=int(field_size)
    out["Contest Aggression"] = round(aggr*100,1)
    out=out.sort_values(["Rating Score","Projection"],ascending=[False,False]).reset_index(drop=True)
    out.insert(0,"Rank",np.arange(1,len(out)+1))
    return out

def showdown_exposure_table(df,result,strategy_map):
    if result is None or result.empty:return pd.DataFrame()
    total=defaultdict(int); cpt=defaultdict(int); n=len(result)
    for _,r in result.iterrows():
        cpt[r["CPT"]]+=1; total[r["CPT"]]+=1
        for col in ["FLEX1","FLEX2","FLEX3","FLEX4","FLEX5"]: total[r[col]]+=1
    rows=[]
    for _,p in df.iterrows():
        strat=strategy_map.get(str(p["ID"]),{})
        rows.append({"ID":str(p["ID"]),"Name":p["Name"],"Pos":p["Position"],"Team":p["Team"],
                     "Flex $":int(p["FlexSalary"]),"Proj":round(float(p["My Proj"]),2),"Proj Own":round(float(p["My Own"]),1),
                     "CPT Own":round(float(p["CPT Own"]),1),"Actual %":round(100*total[p["Name"]]/n,1),
                     "CPT Actual %":round(100*cpt[p["Name"]]/n,1),"Lineups":total[p["Name"]],
                     "Min %":float(strat.get("Min Exposure",0)),"Max %":float(strat.get("Max Exposure",100)),
                     "CPT Min %":float(strat.get("CPT Min",0)),"CPT Max %":float(strat.get("CPT Max",100))})
    return pd.DataFrame(rows).sort_values(["Actual %","Proj"],ascending=[False,False]).reset_index(drop=True)

def audit_min_exposure(result, df, strategy_map):
    """Check minimum-exposure targets against actuals, with a reason for each miss.

    A missed target is silent in the UI unless something says why. Reasons:
    the player is marked Out/excluded, is inactive (not in the build pool), or
    the target simply was not in force when the build ran (edited but never
    applied, or clobbered by a stale editor). Pure.
    Returns a list of dicts: {name, target, actual, reason}.
    """
    if result is None or result.empty:
        return []
    n = len(result)
    total = defaultdict(int)
    for _, r in result.iterrows():
        total[str(r["CPT"])] += 1
        for col in ("FLEX1", "FLEX2", "FLEX3", "FLEX4", "FLEX5"):
            total[str(r[col])] += 1
    id_to_name = {}
    id_to_active = {}
    for _, r in df.iterrows():
        pid = str(r["ID"])
        id_to_name[pid] = str(r["Name"])
        id_to_active[pid] = bool(r.get("ActiveForBuild", True))
    misses = []
    for pid, strat in (strategy_map or {}).items():
        tmin = float((strat or {}).get("Min Exposure", 0) or 0)
        if tmin <= 0:
            continue
        pid = str(pid)
        name = id_to_name.get(pid, pid)
        actual = 100.0 * total.get(name, 0) / n
        if actual >= tmin - 0.51:
            continue
        if pid not in id_to_name:
            reason = "not in this slate's player pool"
        elif not id_to_active.get(pid, True):
            reason = "inactive — not in the build pool (check the Players tab Live column)"
        elif bool((strat or {}).get("Exclude", False)) or str((strat or {}).get("Priority", "")) == "Exclude":
            reason = "marked Out/excluded in the Players tab — uncheck Out so the optimizer can pick them"
        else:
            reason = "target was not in force for the last build — re-apply it in the Exposure Lab, then Generate"
        misses.append({"name": name, "target": tmin, "actual": round(actual, 1), "reason": reason})
    return misses


def showdown_upload_csv(result):
    cols=["CPT","FLEX1","FLEX2","FLEX3","FLEX4","FLEX5"]
    rows=[]
    for _,r in result.iterrows():
        rows.append([r.get(c+"_NameID",r[c]) for c in cols])
    return pd.DataFrame(rows,columns=["CPT","FLEX","FLEX","FLEX","FLEX","FLEX"]).to_csv(index=False)

def captain_pool_ids(df, strategy_map):
    """IDs the optimizer may actually put at Captain.

    Mirrors the MILP's own semantics (showdown build): a player must be active
    for the build, not excluded, and CPT-eligible. A player with no strategy
    entry at all defaults to eligible (first-run behavior); an explicit
    CPT Eligible=False (CPT? unchecked + applied) removes them from the pool.

    A captain-locked player (CPT Lock checked) is always eligible: the lock is
    the stronger intent ("WILL captain"), so checking the CPT column alone is
    enough to put a player in the pool.
    """
    ids = []
    active = (df["ActiveForBuild"].to_numpy(bool) if "ActiveForBuild" in df.columns
              else [True] * len(df))
    for pos, (_, r) in enumerate(df.iterrows()):
        strat = strategy_map.get(str(r["ID"]), {})
        excluded = bool(strat.get("Exclude", False)) or strat.get("Priority") == "Exclude"
        eligible = bool(strat.get("CPT Eligible", True)) or bool(strat.get("CPT Lock", False))
        if active[pos] and not excluded and eligible:
            ids.append(str(r["ID"]))
    return ids


def audit_showdown_portfolio(result, strategy_map, id_to_name=None):
    """Post-build hard-rule audit for a Showdown portfolio.

    Verifies every built lineup against the user's hard player rules:
      - nobody captains with CPT? unchecked (CPT Eligible=False)
      - no excluded player appears in any slot
      - every CPT-locked player is the Captain
      - every locked player appears somewhere in the lineup

    Returns a list of human-readable violation strings (empty when clean).
    This is a safety net: the MILP already enforces these as hard bounds, so
    any violation here means the strategy map and the built lineups disagree
    (e.g. a UI edit landed on the wrong player) and the user must see it.
    """
    violations = []
    if result is None or getattr(result, "empty", True):
        return violations
    id_to_name = id_to_name or {}
    slot_id_cols = ["CPT_ID", "FLEX1_ID", "FLEX2_ID", "FLEX3_ID", "FLEX4_ID", "FLEX5_ID"]
    slot_name_cols = ["CPT", "FLEX1", "FLEX2", "FLEX3", "FLEX4", "FLEX5"]

    def _nm(pid):
        return id_to_name.get(str(pid), str(pid))

    for idx, row in result.iterrows():
        label = f"Lineup {row.get('Rank', idx + 1)}"
        lineup_ids = [str(row.get(c, "")) for c in slot_id_cols if str(row.get(c, ""))]
        cap_id = str(row.get("CPT_ID", ""))
        cap_name = str(row.get("CPT", _nm(cap_id)))
        cap_strat = strategy_map.get(cap_id, {})
        # A captain-locked player is eligible by definition (lock implies eligibility).
        if cap_id and not (bool(cap_strat.get("CPT Eligible", True)) or bool(cap_strat.get("CPT Lock", False))):
            violations.append(
                f"{label}: {cap_name} is Captain but CPT? is unchecked for them."
            )
        for c, ncol in zip(slot_id_cols, slot_name_cols):
            pid = str(row.get(c, ""))
            if not pid:
                continue
            st_ = strategy_map.get(pid, {})
            if bool(st_.get("Exclude", False)) or st_.get("Priority") == "Exclude":
                violations.append(
                    f"{label}: {_nm(pid)} ({str(row.get(ncol, ''))}) is marked Out but is in the lineup."
                )
        for pid, st_ in strategy_map.items():
            pid = str(pid)
            if bool(st_.get("CPT Lock", False)) and not bool(st_.get("Exclude", False)):
                if cap_id != pid:
                    violations.append(
                        f"{label}: {_nm(pid)} is captain-locked but is not the Captain."
                    )
            if bool(st_.get("Lock", False)) and not bool(st_.get("Exclude", False)):
                if pid not in lineup_ids:
                    violations.append(
                        f"{label}: {_nm(pid)} is locked but is missing from the lineup."
                    )
    return violations
