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

    # One ceiling world per team, named for the team. A separate "{QB} ceiling"
    # world duplicated the team's passing-ceiling story (same QB, same outcome)
    # and could go stale when a starter was ruled out mid-week, so QB-named
    # worlds were removed entirely.
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
        "Single Entry":{"shootout":36,"pass_ceiling":15,"rb_control":7,"comeback":5,"low":2},
        "3-Max":{"shootout":27,"pass_ceiling":16,"rb_control":8,"comeback":6,"low":3},
        "20-Max":{"shootout":20,"pass_ceiling":16,"rb_control":9,"comeback":8,"low":3},
        "150-Max":{"shootout":13,"pass_ceiling":15,"rb_control":10,"comeback":10,"low":5},
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
            elif cfg.get("family")=="comeback": w[name]*=0.7

    if field_size>=50000 and entry_format=="150-Max":
        for name,cfg in GAME_WORLDS.items():
            if cfg.get("family")=="shootout": w[name]*=0.8
            elif cfg.get("family")=="low": w[name]*=1.25

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
    elif fam=="pass_ceiling": desired={"3-3":42,"4-2":50,"5-1":8}
    elif fam=="rb_control": desired={"3-3":30,"4-2":55,"5-1":15}
    else: desired={"3-3":34,"4-2":50,"5-1":16}
    if entry_format=="Single Entry": desired["5-1"]*=0.35; desired["3-3"]*=1.18
    elif entry_format=="150-Max": desired["5-1"]*=1.35
    return {k:(desired[k] if allowed[k] else 0.0) for k in desired}

def _allowed_showdown_shapes(construction_weights):
    """The user's allowed Build Shapes as a set of shape keys
    (e.g. {"3-3", "4-2"}), or None when the weights carry no
    restriction worth enforcing.

    Oct 28, 2026 — Chris built CAR@GB with Build Shape weights
    {"3-3": 1, "4-2": 1, "5-1": 0} (5-1 unchecked) and 18 of the 20
    delivered lineups were 5-1: the projection MILP honored his
    zeros (see _allowed_targets in generate_showdown_lineups), but
    the WS26 world-authored candidate supply and the portfolio
    selection never looked at the shapes at all, so the naturally
    5-1-heavy world candidates crowded out the shape-legal supply.
    Every layer that can author, select, or substitute a lineup —
    showdown_world_optimal_lineups, select_showdown_portfolio, and
    the audit's challengers — now derives the allowed set from the
    RAW user weights through this one helper. Returns None (no
    shape enforcement, the pre-fix behavior) when the weights are
    missing, allow every shape, or allow none: an all-zero dict
    names no legal shape to enforce toward, and generation's own
    fallback already handles that case.
    """
    if not construction_weights:
        return None
    vals = {}
    for k in ("3-3", "4-2", "5-1"):
        try:
            vals[k] = float(construction_weights.get(k, 0) or 0)
        except (TypeError, ValueError):
            vals[k] = 0.0
    if not any(v > 0 for v in vals.values()):
        return None
    if all(v > 0 for v in vals.values()):
        return None
    return {k for k, v in vals.items() if v > 0}

def _showdown_shape_of(per_team_counts):
    """Orientation-free shape key ("4-2") for a 6-player lineup's
    per-team counts (an iterable of exactly two counts)."""
    cs = sorted((int(c) for c in per_team_counts), reverse=True)
    return f"{cs[0]}-{cs[1]}" if len(cs) == 2 else None

def _allowed_shape_team_counts(allowed_shapes):
    """Per-team player counts an allowed-shape set permits: shape
    "a-b" contributes both a and b ({3-3, 4-2} -> {2, 3, 4})."""
    counts = set()
    for shape in allowed_shapes:
        a, b = shape.split("-")
        counts.add(int(a)); counts.add(int(b))
    return counts

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

def infer_score_script(team_scores, market_total=None):
    """Translate a predicted final score into a broad Showdown game environment.

    When `market_total` (the game's market total) is given, the
    low-scoring / shootout bands are judged RELATIVE to it — a script is
    low-scoring when its line centers clearly under the total (<= total
    - 9) and a shootout when it clears the total (>= total + 4.5) in a
    close game — so a 44-point line in a 50.5 game is not mislabeled
    and a 48 in a 38.5 game can be a shootout. Without a market total
    the long-standing absolute bands apply (low <= 41, shootout >= 55).
    """
    if not team_scores or len(team_scores) < 2:
        return "Neutral", "", {"total": 0, "margin": 0, "winner": "", "loser": ""}
    ordered = sorted(team_scores.items(), key=lambda kv: kv[1], reverse=True)
    winner, win_pts = ordered[0]
    loser, lose_pts = ordered[1]
    total = float(win_pts + lose_pts)
    margin = float(win_pts - lose_pts)
    if market_total:
        low_cut = float(market_total) - 9.0
        shoot_cut = float(market_total) + 4.5
    else:
        low_cut, shoot_cut = 41.0, 55.0
    if total <= low_cut:
        script = "Low-scoring game"
    elif total >= shoot_cut and margin <= 10:
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
    out["Aytia Proj"]=(out[base_col].astype(float)*(1.0+np.array(adjs))).round(3)
    out["Context Why"]=reasons
    return out

def showdown_base_objective(df, aggr, strategy_map, script, script_team, exposure_state=None, cpt_exposure_state=None, built_count=0, game_world=None, world_influence=50):
    proj_col = "Aytia Proj" if "Aytia Proj" in df.columns else ("Script Proj" if "Script Proj" in df.columns else "My Proj")
    proj = df[proj_col].to_numpy(float)
    own = np.clip(df["My Own"].to_numpy(float), 0.1, None)
    objective = proj.copy()
    ownership_available = bool(pd.to_numeric(df.get("My Own",0),errors="coerce").fillna(0).max() > 0.01)
    if "Own Estimated" in df.columns:
        # Estimated ownership is a formula, not field data: it must not
        # steer the build. Real ownership (SaberSim file) is unaffected.
        try: ownership_available = ownership_available and not bool(df["Own Estimated"].fillna(True).any())
        except Exception: ownership_available = False
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
    relationship_rules=None, forced_overall_ids=None, forced_cpt_ids=None, game_world=None, world_influence=50,
    qb_exact=None, blocked_ids=None, cpt_blocked_ids=None,
    qb_linkage_exempt_teams=None
):
    n = len(df); s = len(SHOWDOWN_SLOTS); total_vars = n*s
    c = np.zeros(total_vars); lb = np.zeros(total_vars); ub = np.ones(total_vars); integrality = np.ones(total_vars)
    active = df["ActiveForBuild"].to_numpy(bool)
    base = showdown_base_objective(df, aggr, strategy_map, script, script_team, exposure_state, cpt_exposure_state, built_count, game_world, world_influence)
    noise = np.exp(rng.normal(0, noise_scale, size=n))

    def vidx(i,j): return i*s+j

    # Objective and bounds. Captain objective includes 1.5 scoring plus captain-specific leverage.
    blocked_ids = blocked_ids or set()
    cpt_blocked_ids = cpt_blocked_ids or set()
    for i, r in df.iterrows():
        strat = strategy_map.get(str(r["ID"]), {})
        excluded = bool(strat.get("Exclude", False)) or strat.get("Priority") == "Exclude"
        # A captain-locked player is by definition captain-eligible: the lock is
        # the stronger, more explicit intent ("WILL captain" beats "may captain").
        # Without this, CPT Lock=True + CPT Eligible=False forces the player into
        # the CPT slot (hard constraint below) while the slot's upper bound is
        # 0 -- an infeasible model.
        cpt_ok = (bool(strat.get("CPT Eligible", True)) or bool(strat.get("CPT Lock", False))) and not excluded
        # Portfolio exposure ceilings, enforced as hard bounds: a player
        # whose overall (or captain) count has reached the exact max for
        # the requested portfolio size is simply unselectable in this
        # solve, so a Max can never be overshot by construction.
        pid_blocked = str(r["ID"]) in blocked_ids
        pid_cpt_blocked = str(r["ID"]) in cpt_blocked_ids
        for j, slot in enumerate(SHOWDOWN_SLOTS):
            if not active[i] or excluded or pid_blocked:
                ub[vidx(i,j)] = 0
                continue
            if slot == "CPT" and (not cpt_ok or pid_cpt_blocked):
                ub[vidx(i,j)] = 0
            if slot == "CPT":
                cpt_own = max(float(r["CPT Own"]), 0.1)
                cpt_proj = float(r["Aytia Proj"] if "Aytia Proj" in r.index else (r["Script Proj"] if "Script Proj" in r.index else r["My Proj"]))
                # CPT leverage only on real (SaberSim) captain ownership;
                # an estimated CPT Own must not steer the build.
                cpt_lev = 0.0 if bool(r.get("CPT Own Estimated", False)) else math.log((1.5*cpt_proj+2)/(cpt_own+1.5))
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
        elif rtype=="Not B when A is Captain":
            # If any selected A is at CPT (slot 0), no selected B may appear
            # in any slot. A at FLEX is unrestricted.
            for ai in aidx:
                coeff={vidx(int(ai),0):1}
                for bi in bidx:
                    for jj in range(s):
                        coeff[vidx(int(bi),jj)]=coeff.get(vidx(int(bi),jj),0)+1
                _add_constraint(rows,lows,highs,coeff,0,1)

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
    # Optional exact-QB rule (e.g. "exactly one QB per lineup").
    if qb_exact is not None:
        coeff={}
        for i in df.index[df["is_QB"]]:
            for j in range(s): coeff[vidx(i,j)] = 1.0
        if coeff: _add_constraint(rows,lows,highs,coeff,float(qb_exact),float(qb_exact))

    # Flex QB-linkage guard (Oct 28, 2026), enforced in-solve so the
    # solver never spends attempts on the shape coherence would
    # reject at delivery: without his QB, a team's pass catchers cap
    # at 2 in this lineup. Linear form per team: catchers − 4·QBs ≤ 2
    # (a rostered QB makes the bound 6 — non-binding). Teams the
    # user's own settings exempt are skipped; his locks construct
    # their shape and are delivered with the override named instead.
    _link_exempt = {str(t) for t in (qb_linkage_exempt_teams or ())}
    for t in teams:
        if str(t) in _link_exempt: continue
        coeff={}
        for i in df.index[df["Team"].eq(t) & df["is_passcatcher"]]:
            for j in range(s): coeff[vidx(i,j)] = coeff.get(vidx(i,j),0)+1.0
        for i in df.index[df["Team"].eq(t) & df["is_QB"]]:
            for j in range(s): coeff[vidx(i,j)] = coeff.get(vidx(i,j),0)-4.0
        if coeff: _add_constraint(rows,lows,highs,coeff,-np.inf,2.0)

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

def showdown_lineup_coherence(df, chosen, game_world=None, exempt_teams=None):
    """Reject Showdown constructions that are legal on DK but contradict their own football story."""
    slot_map={slot:i for slot,i in chosen}
    idxs=[i for _,i in chosen]
    p=df.loc[idxs].copy()
    cpt=df.loc[slot_map["CPT"]]
    hard=[]; warnings=[]; score=100.0

    if "Role Confidence" in p.columns:
        bad=p[p["Role Confidence"].astype(str).isin(["Backup QB","Inactive / no usable projection"])]
        if not bad.empty:
            hard.append("Contains a player Aytia marked unavailable for the current football role.")

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

    # Flex-level QB linkage (Oct 28, 2026): the captain-slot checks
    # below never looked at FLEX pass catchers, so a lineup could
    # carry three Carolina receivers with no Bryce Young and still
    # freeze as "Coherence 100.0 — no contradictions." A team's pass
    # catchers can't all get there without the quarterback who feeds
    # them: 3+ catchers from one team with no QB from that team
    # anywhere in the lineup is a hard contradiction. Exactly 2 is a
    # legitimate construction — it costs a named penalty only. Teams
    # the user's own settings exempt (his locks / exclusions
    # construct the shape — see _showdown_qb_linkage_exemptions)
    # are delivered with the reason named, never silently blocked
    # and never scored as clean.
    _exempt = exempt_teams or {}
    for team in sorted({str(t) for t in p["Team"].astype(str)}):
        tp = p[p["Team"].astype(str).eq(team)]
        n_pc = int(tp["is_passcatcher"].sum())
        if n_pc < 2 or bool(tp["is_QB"].any()):
            continue
        qb_name = _showdown_team_qb_name(df, team)
        qb_txt = qb_name or "their quarterback"
        if n_pc >= 3:
            if team in _exempt:
                score -= 40
                warnings.append(
                    f"{n_pc} {team} pass catchers without {qb_txt} — "
                    f"delivered because {_exempt[team]}")
            else:
                score -= 40
                hard.append(
                    f"{n_pc} {team} pass catchers without {qb_txt} — "
                    f"a team's pass catchers can't all get there "
                    f"without their quarterback")
        else:
            score -= 8
            warnings.append(f"2 {team} pass catchers without {qb_txt}")

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
    proj_col = "Aytia Proj" if "Aytia Proj" in df.columns else ("Script Proj" if "Script Proj" in df.columns else "My Proj")
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

def add_showdown_ratings(out, aggr, use_world_metrics=False):
    if out.empty: return out
    proj=out["Projection"].rank(pct=True)
    captain=(out["Projection"] - 0.5*out["Salary Left"]/1000).rank(pct=True)
    corr=out["Correlation Raw"].rank(pct=True)
    ownership_available=bool(pd.to_numeric(out["Total Own"],errors="coerce").fillna(0).max()>0.01)
    lev=(-out["Total Own"]).rank(pct=True) if ownership_available else pd.Series(0.5,index=out.index)
    dup=(-out["Dup Raw"]).rank(pct=True) if ownership_available else pd.Series(0.5,index=out.index)
    fit=out["User Fit Raw"].rank(pct=True)
    coh=out["Coherence Score"].rank(pct=True) if "Coherence Score" in out.columns else pd.Series(0.5,index=out.index)
    # WS26 (trust release, Piece 1): when the caller supplies the sim
    # world metrics (see showdown_lineup_world_metrics), the composite
    # is anchored on the worlds — Best-of-pool % and Top-decile
    # presence (the win terms), Ceiling P90 and Sim Mean — plus the
    # Coherence and User Fit percentiles at their standing weights and
    # the ownership field terms as-is. Raw adjusted projection and the
    # correlation heuristic LEAVE the rating: a lineup's standing comes
    # from the worlds it wins, not from a percentile table of the
    # projection column inside its own delivered set (the Oct 28
    # failure: a lineup 9th of its own 20 by sim mean rated #2).
    _world_cols = ("Best-of-pool %", "Top-decile %", "Ceiling P90",
                   "Sim Mean")
    if use_world_metrics and all(c in out.columns for c in _world_cols):
        w_best=.24; w_top=.24; w_p90=.10; w_mean=.08
        w_lev=(.10+.04*aggr) if ownership_available else 0.0
        w_dup=(.09+.05*aggr) if ownership_available else 0.0
        w_fit=.06; w_coh=.12
        comp=(w_best*out["Best-of-pool %"].rank(pct=True)
              +w_top*out["Top-decile %"].rank(pct=True)
              +w_p90*out["Ceiling P90"].rank(pct=True)
              +w_mean*out["Sim Mean"].rank(pct=True)
              +w_lev*lev+w_dup*dup+w_fit*fit+w_coh*coh) \
            /(w_best+w_top+w_p90+w_mean+w_lev+w_dup+w_fit+w_coh)
    else:
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

def default_world_max_share(count):
    """Smart default for the game-world diversification cap, keyed off how many
    lineups the user is actually building (not the contest max).

    A 5-lineup portfolio should be allowed to ride one theory; a 50-lineup
    portfolio paying 50 entry fees should cover the outcome space instead of
    making the same correlated bet 50 times.
    """
    count=int(count)
    if count<=5: return 1.0
    if count<=20: return 0.60
    if count<=50: return 0.40
    return 0.30

def world_share_capped(world_counts, world, count, cap):
    """True when `world` has already reached its allowed share of `count` lineups."""
    if cap is None or cap>=1.0: return False
    limit=max(1,int(math.ceil(float(cap)*int(count))))
    return int(world_counts.get(world,0))>=limit

def _pct_label(x):
    """Format a percentage setting the way the user typed it (50, not 50.0)."""
    f = float(x)
    return str(int(f)) if f == int(f) else str(f)


def _showdown_exposure_targets(df, strategy_map, count):
    """Exact per-player appearance targets for a portfolio of `count`
    lineups, converted from the strategy map's percentage settings.

    Returns (targets, notes). targets maps every pool player ID (str) to
    {name, min, max, cpt_min, cpt_max, min_pct, max_pct, cpt_min_pct,
    cpt_max_pct} in LINEUP COUNTS: a Max is a hard ceiling
    floor(pct * count / 100) — never to be exceeded; a Min is a floor
    ceil(pct * count / 100) the assembly must reach or report.

    Settings that cannot be honored as set are adjusted here, ONCE, and
    every adjustment is named in `notes` (player, setting, achievable
    range) — the build proceeds, but nothing is silently relaxed:
      - a Min/CPT Min on an excluded or inactive player (achievable 0);
      - a CPT Min on a player who is not captain-eligible (achievable 0);
      - a Min above its own Max (the build targets the Max and says so);
      - a Lock / CPT Lock combined with a Max below 100% (the lock is
        the stronger intent and wins; the cap is named as unholdable);
      - aggregate infeasibility: the caps together leave fewer player
        slots (or captain slots) than the portfolio needs, so a
        shortfall is guaranteed before a single solve runs.
    """
    count = int(count)
    targets = {}
    notes = []
    active_col = (df["ActiveForBuild"].to_numpy(bool)
                  if "ActiveForBuild" in df.columns else [True] * len(df))
    slot_capacity = 0
    cpt_capacity = 0
    binding = []
    for pos, (_, r) in enumerate(df.iterrows()):
        pid = str(r["ID"])
        name = str(r["Name"])
        strat = (strategy_map or {}).get(pid, {}) or {}
        excluded = bool(strat.get("Exclude", False)) \
            or strat.get("Priority") == "Exclude"
        active = bool(active_col[pos])
        lock = bool(strat.get("Lock", False)) and not excluded
        cpt_lock = bool(strat.get("CPT Lock", False)) and not excluded
        cpt_ok = (bool(strat.get("CPT Eligible", True)) or cpt_lock) \
            and not excluded
        mn = float(strat.get("Min Exposure", 0) or 0)
        mx = float(strat.get("Max Exposure", 100) or 0)
        cmn = float(strat.get("CPT Min", 0) or 0)
        cmx = float(strat.get("CPT Max", 100) or 0)
        mn_c = int(math.ceil(min(max(mn, 0.0), 100.0) * count / 100.0))
        mx_c = int(math.floor(min(max(mx, 0.0), 100.0) * count / 100.0))
        cmn_c = int(math.ceil(min(max(cmn, 0.0), 100.0) * count / 100.0))
        cmx_c = int(math.floor(min(max(cmx, 0.0), 100.0) * count / 100.0))
        usable = active and not excluded
        if not usable:
            why = "excluded by your rules" if excluded else "inactive"
            if mn_c > 0:
                notes.append(
                    f"Min Exposure for {name} ({_pct_label(mn)}%) cannot "
                    f"be honored — {name} is {why}, so the achievable "
                    f"range is 0 of {count} lineups.")
            if cmn_c > 0:
                notes.append(
                    f"CPT Min for {name} ({_pct_label(cmn)}%) cannot be "
                    f"honored — {name} is {why}, so the achievable range "
                    f"is 0 of {count} captain slots.")
            mn_c = cmn_c = 0
            mx_c = cmx_c = 0
        else:
            if lock and mx_c < count:
                notes.append(
                    f"{name} is locked, so Max Exposure "
                    f"{_pct_label(mx)}% cannot hold — a lock plays in "
                    f"every lineup (achievable max is 100%).")
                mx_c = count
            if cpt_lock and cmx_c < count:
                notes.append(
                    f"{name} is captain-locked, so CPT Max "
                    f"{_pct_label(cmx)}% cannot hold — a captain lock "
                    f"captains every lineup (achievable max is 100%).")
                cmx_c = count
            if not cpt_ok:
                if cmn_c > 0:
                    notes.append(
                        f"CPT Min for {name} ({_pct_label(cmn)}%) cannot "
                        f"be honored — {name} is not captain-eligible in "
                        f"your rules, so the achievable range is 0 of "
                        f"{count} captain slots.")
                cmn_c = 0
                cmx_c = 0
            if mn_c > mx_c:
                notes.append(
                    f"{name}: Min Exposure {_pct_label(mn)}% ({mn_c} of "
                    f"{count}) exceeds Max Exposure {_pct_label(mx)}% "
                    f"({mx_c}) — building to the max; the achievable "
                    f"range is 0–{mx_c} lineups.")
                mn_c = mx_c
            if cmn_c > cmx_c:
                notes.append(
                    f"{name}: CPT Min {_pct_label(cmn)}% ({cmn_c} of "
                    f"{count}) exceeds CPT Max {_pct_label(cmx)}% "
                    f"({cmx_c}) — building to the max; the achievable "
                    f"range is 0–{cmx_c} captain slots.")
                cmn_c = cmx_c
            slot_capacity += mx_c
            if cpt_ok:
                cpt_capacity += cmx_c
            if mx_c < count:
                binding.append((mx_c, name, mx))
        targets[pid] = {"name": name, "min": mn_c, "max": mx_c,
                        "cpt_min": cmn_c, "cpt_max": cmx_c,
                        "min_pct": mn, "max_pct": mx,
                        "cpt_min_pct": cmn, "cpt_max_pct": cmx,
                        "usable": usable}
    if count > 0 and slot_capacity < 6 * count:
        binding.sort()
        named = ", ".join(f"{nm} (max {_pct_label(p)}%)"
                          for _, nm, p in binding[:4])
        notes.append(
            f"Exposure caps leave room for only {slot_capacity} of the "
            f"{6 * count} player slots a {count}-lineup portfolio needs "
            f"— the build will fall short unless caps loosen. Binding "
            f"caps: {named}.")
    if count > 0 and cpt_capacity < count:
        notes.append(
            f"Captain caps leave room for only {cpt_capacity} of the "
            f"{count} captain slots a {count}-lineup portfolio needs — "
            f"the build will fall short unless CPT Max values loosen.")
    return targets, notes


def _showdown_no_role_players(df, strategy_map):
    """Punt-eligibility floor (Chris's standing rule, Oct 25 2026): a
    punt is a player with a real route/target role, not the cheapest
    body in the pool — a zero-target blocker or a long snapper listed
    at TE is a mannequin, not value (the Kallerup/Winchester lesson).

    Returns {pid: {"name", "proj", "opp"}} for ACTIVE skill-position
    players whose role is effectively zero: expected opportunities
    (Exp Tgt + Exp Car, + Exp Att for QBs — the role columns the
    projection engine writes) under 1.0 per game AND a standing
    projection under 2.0. When the pool carries no Exp columns
    (SaberSim-fed frames), the projection alone decides, at a tighter
    under-1.0 bar. Kickers and defenses are exempt — their role is the
    position itself. The floor judges "My Proj" (the standing
    projection), never the scenario-tilted one.

    A player the user explicitly endorsed is NEVER in the set — his
    call beats the rule: Lock / CPT Lock, Priority Like or Core, or a
    Min Exposure / CPT Min above zero.
    """
    info = {}
    if df is None or len(df) == 0:
        return info
    have_exp = all(c in df.columns for c in ("Exp Tgt", "Exp Car"))
    active_col = (df["ActiveForBuild"].to_numpy(bool)
                  if "ActiveForBuild" in df.columns else [True] * len(df))
    for pos, (_, r) in enumerate(df.iterrows()):
        if not bool(active_col[pos]):
            continue
        if str(r.get("Position", "")).upper() in ("K", "DST"):
            continue
        pid = str(r["ID"])
        strat = (strategy_map or {}).get(pid, {}) or {}
        if bool(strat.get("Lock", False)) or bool(strat.get("CPT Lock", False)):
            continue
        if str(strat.get("Priority", "Neutral")) in ("Like", "Core"):
            continue
        if float(strat.get("Min Exposure", 0) or 0) > 0:
            continue
        if float(strat.get("CPT Min", 0) or 0) > 0:
            continue
        try:
            proj = float(r.get("My Proj"))
        except (TypeError, ValueError):
            continue
        opp = None
        if have_exp:
            def _exp(col):
                try:
                    return float(r.get(col) or 0)
                except (TypeError, ValueError):
                    return 0.0
            opp = _exp("Exp Tgt") + _exp("Exp Car")
            if str(r.get("Position", "")).upper() == "QB":
                opp += _exp("Exp Att")
            no_role = opp < 1.0 and proj < 2.0
        else:
            no_role = proj < 1.0
        if no_role:
            info[pid] = {"name": str(r.get("Name", pid)), "proj": proj,
                         "opp": opp}
    return info


def _punt_floor_notes(no_role, counts, delivered_n):
    """One named warning per no-role player a delivered set actually
    used — the floor was breached only because the requested set could
    not be filled without him."""
    notes = []
    for pid, meta in no_role.items():
        used = int(counts.get(pid, 0))
        if used <= 0:
            continue
        if meta["opp"] is None:
            detail = f"projection {meta['proj']:.2f}"
        else:
            detail = (f"projection {meta['proj']:.2f}, "
                      f"~{meta['opp']:.1f} opportunities/game")
        notes.append(
            f"Punt floor breached: {meta['name']} has no measurable "
            f"role ({detail}) and was used in {used} of {delivered_n} "
            f"lineups because the requested set could not be filled "
            f"without him.")
    return notes


def _showdown_team_qb_name(df, team):
    """The name the QB-linkage guard speaks for a team's quarterback:
    the highest-projected QB of that team in the pool."""
    qbs = df[df["Team"].astype(str).eq(str(team)) & df["is_QB"]]
    if qbs.empty:
        return None
    if "My Proj" in qbs.columns:
        qbs = qbs.sort_values("My Proj", ascending=False, kind="mergesort")
    return str(qbs.iloc[0]["Name"])


def _showdown_qb_linkage_exemptions(df, strategy_map):
    """Teams exempt from the flex QB-linkage guard because the USER's
    own explicit settings construct the no-QB pass-catcher shape
    (Chris's standing rule: his settings outrank the guard, exactly
    as his endorsements outrank the punt floor).

    Returns {team: reason}, where reason completes the sentence
    "...delivered because <reason>" in the coherence flag and the
    build warning. A team is exempt when:
      - 3+ of its pass catchers are Locked / CPT Locked — his locks
        alone put 3+ of that team's catchers into every lineup; or
      - every QB on the team is excluded in his rules (or inactive)
        while at least one of its pass catchers carries his
        endorsement (Lock / CPT Lock / Like / Core / a minimum) —
        he benched the quarterback and backed the receivers himself.
    A bare QB exclusion with NO catcher endorsement is not an
    exemption: the guard still caps that team's catchers at 2 per
    lineup, which fights nothing he asked for.
    """
    exempt = {}
    if df is None or len(df) == 0:
        return exempt
    smap = strategy_map or {}
    active_col = (df["ActiveForBuild"].to_numpy(bool)
                  if "ActiveForBuild" in df.columns else [True] * len(df))
    df = df.assign(_active=list(active_col))

    def _strat(pid):
        return smap.get(str(pid), {}) or {}

    def _excluded(pid):
        st = _strat(pid)
        return bool(st.get("Exclude", False)) \
            or st.get("Priority") == "Exclude"

    def _locked(pid):
        st = _strat(pid)
        return (bool(st.get("Lock", False))
                or bool(st.get("CPT Lock", False))) and not _excluded(pid)

    def _endorsed(pid):
        st = _strat(pid)
        if _excluded(pid):
            return False
        return (_locked(pid) or st.get("Priority") in ("Like", "Core")
                or float(st.get("Min Exposure", 0) or 0) > 0
                or float(st.get("CPT Min", 0) or 0) > 0)

    for team in sorted({str(t) for t in df["Team"].dropna().unique()}):
        tdf = df[df["Team"].astype(str).eq(team)]
        pcs = tdf[tdf["is_passcatcher"]]
        if pcs.empty:
            continue
        locked_pcs = [str(r["Name"]) for _, r in pcs.iterrows()
                      if _locked(r["ID"])]
        if len(locked_pcs) >= 3:
            named = ", ".join(locked_pcs[:3])
            if len(locked_pcs) > 3:
                named += f" and {len(locked_pcs) - 3} more"
            exempt[team] = (f"you locked {len(locked_pcs)} {team} pass "
                            f"catchers ({named})")
            continue
        qbs = tdf[tdf["is_QB"]]
        if qbs.empty:
            continue
        qb_usable = any(bool(r["_active"]) and not _excluded(r["ID"])
                        for _, r in qbs.iterrows())
        qb_ruled_out = any(_excluded(r["ID"]) or not bool(r["_active"])
                           for _, r in qbs.iterrows())
        endorsed_pcs = [str(r["Name"]) for _, r in pcs.iterrows()
                        if _endorsed(r["ID"])]
        if not qb_usable and qb_ruled_out and endorsed_pcs:
            qb_name = _showdown_team_qb_name(df, team) or "the quarterback"
            exempt[team] = (f"{qb_name} is ruled out in your settings "
                            f"while you backed {team} pass catchers "
                            f"({', '.join(endorsed_pcs[:3])})")
    return exempt


def _showdown_linkage_breaches(df, ids, skip_teams=()):
    """[(team, n_catchers, qb_name)] for one lineup given as player
    IDs: the guarded shape — 3+ pass catchers from one team with no
    QB from that team anywhere in the lineup (the captain counts
    toward his team like any other slot). Teams in skip_teams (the
    user's own settings construct their shape) are not reported."""
    if df is None or len(df) == 0:
        return []
    team_of, pc_of, qb_of = {}, {}, {}
    for _, r in df.iterrows():
        pid = str(r["ID"])
        team_of[pid] = str(r["Team"])
        pc_of[pid] = bool(r["is_passcatcher"])
        qb_of[pid] = bool(r["is_QB"])
    skip = {str(t) for t in (skip_teams or ())}
    per_team = {}
    for pid in ids:
        t = team_of.get(str(pid))
        if t is None or t in skip:
            continue
        cell = per_team.setdefault(t, [0, False])
        if pc_of.get(str(pid)):
            cell[0] += 1
        if qb_of.get(str(pid)):
            cell[1] = True
    out = []
    for team in sorted(per_team):
        n_pc, has_qb = per_team[team]
        if n_pc >= 3 and not has_qb:
            out.append((team, n_pc,
                        _showdown_team_qb_name(df, team)))
    return out


def select_showdown_portfolio(candidates, df, strategy_map, count,
                              min_unique=2, order_scores=None,
                              construction_weights=None):
    """Assemble the final `count` lineups from generated candidate rows
    under EXACT exposure accounting against `count`.

    The generator enforces Min/Max/CPT targets against the count IT was
    asked for; callers that over-generate (the web API asks for ~3x and
    delivers the best `count`) must re-enforce against the delivered
    count, or the delivered slice can overshoot a cap the generated
    portfolio honored (Oct 26, 2026: Pierce Max 50 read 70% of the
    delivered 20). Selection order:
      A. reserve — highest-rated candidates that satisfy each unmet
         Min / CPT Min first (a minimum the rating order would crowd
         out is a promise, not a preference);
      B. fill — rating order, subject to the exact caps and the
         portfolio uniqueness rule;
      C. repair — swap a selected lineup for an unselected candidate
         carrying a still-deficient player when caps/uniqueness allow;
      D. fill again with the uniqueness rule stepped down (recorded);
    then any target still unmet, and any shortfall, is reported by name
    in the returned notes. Max caps are never relaxed in any phase.
    Returns (selected_df, notes); with no exposure settings set the
    result is exactly the first `count` candidates in rating order.

    WS26 (trust release): `order_scores`, when given, is a per-candidate
    preference score aligned with `candidates` rows (the sim world
    score — see showdown_world_scores). Every phase then prefers
    candidates in world-score order instead of candidate (rating)
    order, and the delivered frame is sorted by it. The exposure
    accounting itself — reserves, caps, repairs, named shortfalls —
    is identical either way; only the preference order changes.

    `construction_weights` (Oct 28, 2026), when given, is the user's
    RAW Build Shape weights: a candidate whose team split is a shape
    the user switched off is hard-ineligible in every phase below
    (see _allowed_showdown_shapes). Selection previously re-enforced
    exposures, the punt floor, and the QB-linkage guard but never
    shapes — the hole that delivered 18-of-20 5-1 lineups against
    an unchecked 5-1. None (the default) = no shape enforcement,
    the pre-fix behavior.
    """
    count = int(count)
    notes = []
    if candidates is None or len(candidates) == 0:
        if count > 0:
            notes.append(
                f"Shortfall: built 0 of {count} requested lineups — "
                f"the generator returned no candidates under your rules.")
        return candidates, notes
    targets, target_notes = _showdown_exposure_targets(
        df, strategy_map, count)
    notes.extend(target_notes)
    # Preference order (WS26): identity = the candidates' own order
    # (rating order, the pre-release behavior); with order_scores the
    # phases walk candidates best-world-score first and "lowest-rated
    # selected" in Phase C means lowest world score.
    _n_cand = len(candidates)
    if order_scores is not None and len(order_scores) == _n_cand:
        _scores = [float(x) for x in order_scores]
        pref = sorted(range(_n_cand), key=lambda i: (-_scores[i], i))
    else:
        _scores = None
        pref = list(range(_n_cand))
    rankpos = [0] * _n_cand
    for _pos, _idx in enumerate(pref):
        rankpos[_idx] = _pos
    slot_cols = ["CPT_ID"] + [f"FLEX{j}_ID" for j in range(1, 6)]
    cand_sets = []
    cand_cpts = []
    for _, row in candidates.iterrows():
        cand_sets.append([str(row[c]) for c in slot_cols])
        cand_cpts.append(str(row["CPT_ID"]))

    # Build Shape hard filter (Oct 28, 2026): with the user's raw
    # construction weights in hand, a candidate whose team split is
    # a disallowed shape is INELIGIBLE in every phase — enforced in
    # _fits at the same hard tier as a malformed row, never demoted
    # to the last-resort "dirty" tier, so no fill, repair, or
    # dirty-pass can deliver it. The incident: Chris unchecked 5-1
    # on CAR@GB and selection — which never received the weights —
    # walked world-score order straight past the generator's
    # shape-legal candidates into the 5-1-heavy world supply.
    shape_banned = set()
    shape_banned_shapes = set()
    _allowed_shapes = _allowed_showdown_shapes(construction_weights)
    if _allowed_shapes is not None and df is not None \
            and "Team" in df.columns and "ID" in df.columns:
        _pool_teams = sorted({str(t) for t in df["Team"].dropna()
                              .unique() if str(t)})
        if len(_pool_teams) == 2:
            _team_of = {str(r["ID"]): str(r["Team"])
                        for _, r in df.iterrows()}
            for _i, _ids in enumerate(cand_sets):
                _per_team = [sum(1 for _p in _ids
                                 if _team_of.get(_p) == _t)
                             for _t in _pool_teams]
                if sum(_per_team) != 6:
                    continue
                _shape = _showdown_shape_of(_per_team)
                if _shape not in _allowed_shapes:
                    shape_banned.add(_i)
                    shape_banned_shapes.add(_shape)
    exposure = defaultdict(int)
    cpt_exp = defaultdict(int)
    selected = []
    used = set()

    # Punt floor at selection (see _showdown_no_role_players): a
    # candidate carrying a no-role player is last-resort — every phase
    # fills from clean candidates first, and the dirty pass runs only
    # when the requested count or a minimum cannot be reached without
    # them. Breaches in the DELIVERED set are named after the tally.
    no_role = _showdown_no_role_players(df, strategy_map)
    dirty = set()
    if no_role:
        _nr = set(no_role)
        dirty = {i for i, ids in enumerate(cand_sets) if _nr & set(ids)}

    # QB-linkage guard at selection (Oct 28, 2026): a candidate
    # carrying 3+ pass catchers from one team without that team's QB
    # is last-resort on the same terms as a punt-floor breach. The
    # generator already rejects the shape in-solve and at coherence,
    # so this only bites for candidates that arrived by another
    # route; any delivered anyway are named after the tally. Teams
    # the user's own settings exempt are not breaches here — their
    # override is named by the generator's build notes.
    guard_breach = {}
    _guard_exempt_sel = _showdown_qb_linkage_exemptions(df, strategy_map)
    for _i, _ids in enumerate(cand_sets):
        _b = _showdown_linkage_breaches(
            df, _ids, skip_teams=set(_guard_exempt_sel))
        if _b:
            guard_breach[_i] = _b
            dirty.add(_i)

    def _fits(idx, uniq_cap, allow_dirty=False):
        ids = cand_sets[idx]
        if len(set(ids)) != 6:
            return False
        # A user-disallowed Build Shape is never admissible, in any
        # phase, on any pass (Oct 28, 2026 — see shape_banned above).
        if idx in shape_banned:
            return False
        if idx in dirty and not allow_dirty:
            return False
        for pid in ids:
            tgt = targets.get(pid)
            if tgt is not None and exposure[pid] + 1 > tgt["max"]:
                return False
        tgt = targets.get(cand_cpts[idx])
        if tgt is not None and cpt_exp[cand_cpts[idx]] + 1 > tgt["cpt_max"]:
            return False
        new_set = set(ids)
        for prev in selected:
            if len(new_set & set(cand_sets[prev])) > uniq_cap:
                return False
        return True

    def _take(idx):
        used.add(idx)
        selected.append(idx)
        for pid in cand_sets[idx]:
            exposure[pid] += 1
        cpt_exp[cand_cpts[idx]] += 1

    def _drop(idx):
        selected.remove(idx)
        used.discard(idx)
        for pid in cand_sets[idx]:
            exposure[pid] -= 1
        cpt_exp[cand_cpts[idx]] -= 1

    uniq_cap = 6 - int(min_unique)
    # Phase A — reserve minimums (largest deficit first, recomputed as
    # picks land; a candidate may serve two minimums at once).
    def _phase_a(allow_dirty):
        while len(selected) < count:
            deficits = []
            for pid, tgt in targets.items():
                if tgt["min"] > exposure[pid]:
                    deficits.append((tgt["min"] - exposure[pid], pid, False))
                if tgt["cpt_min"] > cpt_exp[pid]:
                    deficits.append((tgt["cpt_min"] - cpt_exp[pid], pid, True))
            if not deficits:
                break
            deficits.sort(key=lambda d: (-d[0], targets[d[1]]["name"]))
            progressed = False
            for _, pid, is_cpt in deficits:
                if len(selected) >= count:
                    break
                want = (targets[pid]["cpt_min"] if is_cpt
                        else targets[pid]["min"])
                have = cpt_exp[pid] if is_cpt else exposure[pid]
                while have < want and len(selected) < count:
                    pick = None
                    for idx in pref:
                        if idx in used:
                            continue
                        if is_cpt and cand_cpts[idx] != pid:
                            continue
                        if not is_cpt and pid not in cand_sets[idx]:
                            continue
                        if _fits(idx, uniq_cap, allow_dirty):
                            pick = idx
                            break
                    if pick is None:
                        break
                    _take(pick)
                    progressed = True
                    have = cpt_exp[pid] if is_cpt else exposure[pid]
            if not progressed:
                break

    def _mins_unmet():
        return any(tgt["min"] > exposure[pid]
                   or tgt["cpt_min"] > cpt_exp[pid]
                   for pid, tgt in targets.items())

    _phase_a(False)
    if dirty and len(selected) < count and _mins_unmet():
        # An explicit minimum outranks the punt floor (same principle
        # as an endorsement): admit dirty candidates to reserve it.
        _phase_a(True)
    # Phases B–D run clean first; if the portfolio is still short or
    # a minimum is still unmet afterwards, they run once more with
    # dirty (no-role-carrying) candidates admitted — the punt floor
    # yields only to an otherwise-unfillable request, and the breach
    # is named after the tally below.
    def _fill_repair(allow_dirty):
        # Phase B — fill in preference order under the caps.
        for idx in pref:
            if len(selected) >= count:
                break
            if idx not in used and _fits(idx, uniq_cap, allow_dirty):
                _take(idx)
        # Phase C — repair unmet minimums by swapping whole candidates.
        for _ in range(60):
            deficits = [(pid, False) for pid, tgt in targets.items()
                        if tgt["min"] > exposure[pid]]
            deficits += [(pid, True) for pid, tgt in targets.items()
                         if tgt["cpt_min"] > cpt_exp[pid]]
            if not deficits or not selected:
                break
            swapped = False
            for pid, is_cpt in deficits:
                for idx in pref:
                    if idx in used:
                        continue
                    if idx in dirty and not allow_dirty:
                        continue
                    if is_cpt and cand_cpts[idx] != pid:
                        continue
                    if not is_cpt and pid not in cand_sets[idx]:
                        continue
                    # Try replacing the lowest-preference selected lineup
                    # whose removal breaks no minimum of its own.
                    for prev in sorted(selected,
                                       key=lambda p: rankpos[p],
                                       reverse=True):
                        if not is_cpt and pid in cand_sets[prev]:
                            continue
                        removed = cand_sets[prev]
                        ok = True
                        for q in set(removed):
                            if q == pid:
                                continue
                            qt = targets.get(q)
                            if qt is not None \
                                    and exposure[q] - 1 < qt["min"]:
                                ok = False
                                break
                        if ok:
                            qc = cand_cpts[prev]
                            qt = targets.get(qc)
                            if qt is not None \
                                    and cpt_exp[qc] - 1 < qt["cpt_min"]:
                                ok = False
                        if not ok:
                            continue
                        _drop(prev)
                        if _fits(idx, uniq_cap, allow_dirty):
                            _take(idx)
                            swapped = True
                            break
                        _take(prev)
                    if swapped:
                        break
                if swapped:
                    break
            if not swapped:
                break
        # Phase D — uniqueness step-down fill (announced), caps untouched.
        if len(selected) < count and int(min_unique) > 0:
            for mu in range(int(min_unique) - 1, -1, -1):
                cap = 6 - mu
                added = False
                for idx in pref:
                    if len(selected) >= count:
                        break
                    if idx not in used and _fits(idx, cap, allow_dirty):
                        _take(idx)
                        added = True
                if added:
                    notes.append(
                        f"Uniqueness relaxed from {int(min_unique)} to {mu} "
                        f"while assembling the portfolio to reach {count} "
                        f"lineups under your exposure rules.")
                if len(selected) >= count:
                    break

    _fill_repair(False)
    if dirty and (len(selected) < count or _mins_unmet()):
        _fill_repair(True)
    for pid, tgt in targets.items():
        if tgt["min"] > exposure[pid]:
            notes.append(
                f"Min Exposure: {tgt['name']} targeted "
                f"{_pct_label(tgt['min_pct'])}% ({tgt['min']} of "
                f"{count} lineups), reached {exposure[pid]} — the "
                f"generated candidates under your other rules ran out.")
        if tgt["cpt_min"] > cpt_exp[pid]:
            notes.append(
                f"CPT Min: {tgt['name']} targeted "
                f"{_pct_label(tgt['cpt_min_pct'])}% ({tgt['cpt_min']} of "
                f"{count}) at captain, reached {cpt_exp[pid]} — the "
                f"generated candidates under your other rules ran out.")
    if no_role:
        notes.extend(_punt_floor_notes(no_role, exposure, len(selected)))
    if guard_breach:
        _gb = {}
        for _idx in selected:
            for _t, _n, _qn in guard_breach.get(_idx, []):
                _gb[(_t, _qn)] = _gb.get((_t, _qn), 0) + 1
        for (_t, _qn), _k in sorted(_gb.items()):
            notes.append(
                f"QB-linkage guard breached: {_k} of {len(selected)} "
                f"delivered lineups carry 3 or more {_t} pass catchers "
                f"without {_qn or 'their quarterback'} because the "
                f"requested set could not be filled without them.")
    if len(selected) < count:
        _cpt_lock_name = None
        for _pid, _st in (strategy_map or {}).items():
            if isinstance(_st, dict) and _st.get("CPT Lock") \
                    and not _st.get("Exclude"):
                _tgt = targets.get(str(_pid))
                _cpt_lock_name = (_tgt or {}).get("name") or str(_pid)
                break
        capped = sorted(
            (tgt for tgt in targets.values()
             if tgt["usable"] and tgt["max"] < count),
            key=lambda t: (t["max"], t["name"]))
        usable_n = sum(1 for t in targets.values() if t["usable"])
        if capped and len(capped) >= usable_n:
            cause = (f"every usable player is capped (tightest: "
                     f"{capped[0]['name']} at "
                     f"{_pct_label(capped[0]['max_pct'])}%) and the "
                     f"candidate supply under those caps ran out")
        elif _cpt_lock_name:
            cause = (f"with {_cpt_lock_name} locked at captain, the "
                     f"legal-lineup supply under your other rules "
                     f"ran out")
        elif capped:
            cause = (f"exposure caps bind on {len(capped)} players "
                     f"(tightest: {capped[0]['name']} at "
                     f"{_pct_label(capped[0]['max_pct'])}%) and the "
                     f"candidate supply under those caps ran out")
        else:
            cause = "the candidate supply ran out"
        notes.append(
            f"Shortfall: assembled {len(selected)} of {count} "
            f"requested lineups from the generated candidates — "
            f"{cause}.")
    if shape_banned and len(selected) < count:
        # Name the shape filter as a cause of the shortfall: the
        # generic "candidate supply ran out" above would otherwise
        # hide that the missing lineups were the user's own rule
        # working as set (Oct 28, 2026).
        notes.append(
            f"Build Shape: {len(shape_banned)} candidates excluded "
            f"as shapes you did not allow "
            f"({', '.join(sorted(shape_banned_shapes))}); "
            f"built {len(selected)} of {count}.")
    out = candidates.iloc[selected]
    if _scores is not None:
        out = out.assign(_ws26_order=[_scores[i] for i in selected])
        out = out.sort_values("_ws26_order", ascending=False,
                              kind="mergesort")
        out = out.drop(columns=["_ws26_order"])
    elif "Rating Score" in out.columns:
        out = out.sort_values("Rating Score", ascending=False,
                              kind="mergesort")
    return out.reset_index(drop=True), list(dict.fromkeys(notes))


def generate_showdown_lineups(df, field_size, payout_style, count, attempts, min_salary, max_salary,
                              construction_weights, script, script_team, strategy_map,
                              entry_format,
                              cpt_qb_passcatchers, wrte_cpt_qb_pair_pct, rb_cpt_dst_k_pct,
                              max_k, max_dst, min_unique, seed, relationship_rules=None, world_influence=50,
                              show_progress=True, world_max_share=None, qb_exact=None,
                              progress_cb=None):
    """Generate a Showdown portfolio without wasting attempts on one randomly chosen build shape.

    V6.5 reliability change: every attempt now tries every user-allowed team construction
    (and both 4-2 / 5-1 orientations) before declaring that attempt infeasible. This keeps
    Game Worlds as preference/ordering, but a single impossible sampled construction can no
    longer make Aytia report "no legal lineup" when another allowed construction is legal.

    world_max_share: max fraction of the portfolio any single game world may take
        (0-1). None = smart default from lineup count via default_world_max_share().
        The cap can never make a build infeasible: it is floored at 1/#worlds.
    """
    aggr=showdown_aggression(field_size,payout_style,entry_format); rng=np.random.default_rng(seed)
    _wcap=float(world_max_share) if world_max_share else default_world_max_share(count)
    _n_worlds=len(GAME_WORLDS) if GAME_WORLDS else 1
    _wcap=min(1.0,max(_wcap,1.0/max(1,_n_worlds)))
    world_counts=defaultdict(int)
    teams=[t for t in df["Team"].dropna().unique().tolist() if t]
    rows=[]; exposure=defaultdict(int); cpt_exp=defaultdict(int); previous=[]; seen=set()
    # Exposure targets as exact counts against the requested portfolio
    # size, with every unhonorable setting named up front (Oct 27, 2026
    # — the settings-stick fix for Showdown; see
    # _showdown_exposure_targets). Max/CPT Max bind as hard solve
    # bounds below; Min/CPT Min bind via tail forcing + repair, and any
    # target still unmet at the end is reported in
    # out.attrs["showdown_notes"] instead of passing silently.
    targets, build_notes = _showdown_exposure_targets(df, strategy_map, count)

    # Punt-eligibility floor (Oct 27, 2026): no-role players are
    # blocked in-solve everywhere below; only the dedicated breach
    # fills after the uniqueness ladder may admit them, and every
    # no-role player the delivered rows actually use is named.
    no_role = _showdown_no_role_players(df, strategy_map)
    no_role_ids = set(no_role)

    # Flex QB-linkage guard (Oct 28, 2026): 3+ pass catchers from one
    # team without that team's QB is a hard incoherence — blocked
    # in-solve and rejected by coherence at delivery. Teams the
    # user's own settings exempt (his locks / QB exclusions
    # construct the shape) skip both enforcements; delivered exempt
    # shapes are named in the build notes after the tally.
    guard_exempt = _showdown_qb_linkage_exemptions(df, strategy_map)

    # Locked-captain candidate supply (Oct 27, 2026): with a captain
    # locked, the per-world solves converge on the same sim-optimal
    # flex cores and the distinct-lineup supply starves long before
    # the legal supply runs out (break-it T10 follow-up: a $5,400
    # captain lock yielded 10 of 60 candidates against 585 legal
    # sets). When a build with a CPT Lock stalls, solves escalate —
    # hotter objective noise first, then small random bans on the
    # flex players the accepted lineups lean on hardest — until
    # generation unsticks. Locks, the locked captain, and players
    # with tail-due minimums are never banned. Non-locked builds
    # never perturb: this fires only in supply mode, only on stall.
    _smap = strategy_map or {}
    cpt_locked_ids = {str(pid) for pid, st in _smap.items()
                      if isinstance(st, dict) and bool(st.get("CPT Lock"))}
    locked_ids = {str(pid) for pid, st in _smap.items()
                  if isinstance(st, dict) and bool(st.get("Lock"))}
    supply_mode = bool(cpt_locked_ids)
    base_noise = .14 + .13 * aggr

    _band_cache = []

    def _flex_band():
        """(cpt_salary, [(pid, flex_salary) desc]) for the locked
        captain, computed once — the salary-band guard for bans."""
        if _band_cache:
            return _band_cache
        cpt_sal = 0
        band = []
        _act = (df["ActiveForBuild"].to_numpy(bool)
                if "ActiveForBuild" in df.columns else [True] * len(df))
        for _pos, (_, _r) in enumerate(df.iterrows()):
            if not bool(_act[_pos]):
                continue
            _pid = str(_r["ID"])
            if _pid in cpt_locked_ids:
                cpt_sal = max(cpt_sal, int(_r["CaptainSalary"]))
                continue
            band.append((_pid, int(_r["FlexSalary"])))
        band.sort(key=lambda t: -t[1])
        _band_cache.extend([cpt_sal, band])
        return _band_cache

    def _top5_without(banned):
        _, band = _flex_band()
        tot = 0
        taken = 0
        for pid2, s2 in band:
            if pid2 in banned:
                continue
            tot += s2
            taken += 1
            if taken == 5:
                break
        return tot if taken == 5 else -1

    def _supply_perturb(stall_now, base_blocked):
        if not supply_mode or stall_now <= 4:
            return base_blocked, base_noise
        noise = base_noise + min(0.5, 0.1 * (stall_now - 4))
        if stall_now <= 6 or not rows:
            return base_blocked, noise
        usage = {}
        for r in rows:
            for i in range(1, 6):
                fid = str(r.get(f"FLEX{i}_ID", ""))
                if fid:
                    usage[fid] = usage.get(fid, 0) + 1
        protected = set(locked_ids) | set(cpt_locked_ids)
        remaining = count - len(rows)
        if remaining > 0:
            for pid, tgt in targets.items():
                if tgt["min"] - exposure[pid] >= remaining:
                    protected.add(pid)
                if tgt["cpt_min"] - cpt_exp[pid] >= remaining:
                    protected.add(pid)
        # Salary-band guard: never ban a player the salary floor
        # makes essential, individually or via the ban set jointly.
        # In a thin band (a $300 captain under a $47,500 floor) the
        # chalk flexes ARE the only legal flexes — banning them turns
        # every solve infeasible instead of diversifying it.
        cpt_sal, _ = _flex_band()
        cand = [(pid, c) for pid, c in usage.items()
                if pid not in protected
                and cpt_sal + _top5_without({pid}) >= int(min_salary)]
        if not cand:
            return base_blocked, noise
        ids = [c[0] for c in cand]
        wts = np.array([c[1] for c in cand], dtype=float)
        wts = wts / wts.sum()
        k = min(len(ids), 1 + int(stall_now >= 10) + int(stall_now >= 16))
        bans = list(rng.choice(ids, size=k, replace=False,
                               p=wts).tolist())
        while bans and cpt_sal + _top5_without(set(bans)) < int(min_salary):
            bans.pop()
        return base_blocked | set(bans), noise

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
        # World diversification cap: no single game world may exceed its share of
        # the portfolio. Resample a few times; if the cap still binds, skip the
        # attempt (the cap is floored at 1/#worlds, so it can never deadlock).
        _wtries=0
        while world_share_capped(world_counts,game_world,count,_wcap) and _wtries<12:
            game_world=choose_game_world(rng,entry_format,field_size,script); _wtries+=1
        if world_share_capped(world_counts,game_world,count,_wcap):
            continue
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
        # accepted lineups are needed to reach it. Targets come from
        # _showdown_exposure_targets, so a minimum the pre-flight already
        # named as unhonorable (excluded / ineligible / above its own
        # max) is clamped there and never forced into an impossible solve.
        remaining=count-len(rows)
        forced_overall=[]; forced_cpt=[]
        for pid,tgt in targets.items():
            need=tgt["min"]-exposure[pid]
            cneed=tgt["cpt_min"]-cpt_exp[pid]
            if need>=remaining and need>0: forced_overall.append(pid)
            if cneed>=remaining and cneed>0: forced_cpt.append(pid)
        # Hard exposure ceilings for this solve: players at their exact
        # max are unselectable (overall) or un-captainable (CPT).
        blocked={pid for pid,tgt in targets.items()
                 if tgt["max"]<count and exposure[pid]>=tgt["max"]}
        cpt_blocked={pid for pid,tgt in targets.items()
                     if tgt["cpt_max"]<count and cpt_exp[pid]>=tgt["cpt_max"]}

        stall = attempt - last_built_attempt
        solve_blocked, solve_noise = _supply_perturb(
            stall, blocked | no_role_ids)
        chosen=None
        for target in candidates:
            chosen=solve_showdown_one(
                df,aggr,rng,strategy_map,min_salary,max_salary,target,script,script_team,
                cpt_qb_passcatchers,wrte_cpt_qb_pair_pct,rb_cpt_dst_k_pct,max_k,max_dst,
                min_unique,previous,exposure,cpt_exp,len(rows),noise_scale=solve_noise,
                relationship_rules=relationship_rules,forced_overall_ids=forced_overall,
                forced_cpt_ids=forced_cpt,game_world=game_world,world_influence=world_influence,
                qb_exact=qb_exact,blocked_ids=solve_blocked,cpt_blocked_ids=cpt_blocked,
                qb_linkage_exempt_teams=set(guard_exempt)
            )
            if chosen:
                break
        if not chosen:
            continue

        ids=tuple(sorted(str(df.loc[i,"ID"]) for _,i in chosen))
        cpt_id=str(df.loc[[i for slot,i in chosen if slot=="CPT"][0],"ID"])
        key=(cpt_id,ids)
        if key in seen: continue

        # Exposure ceilings, exact against the final portfolio size.
        # The in-solve bounds already make an overshoot impossible; this
        # accept check is the belt to that suspenders (it replaces the
        # old running-percentage check, which exempted the first 8
        # lineups, allowed +3 slack, and still let a Max 50 land at 70%).
        reject=False
        for slot,i in chosen:
            pid=str(df.loc[i,"ID"]); tgt=targets.get(pid)
            if tgt is None: continue
            if exposure[pid]+1>tgt["max"]: reject=True; break
            if slot=="CPT" and cpt_exp[pid]+1>tgt["cpt_max"]: reject=True; break
        if reject: continue

        coherence=showdown_lineup_coherence(df,chosen,game_world,
                                            exempt_teams=guard_exempt)
        if not coherence["Accept"]:
            continue

        seen.add(key)
        world_counts[game_world]+=1
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
        if progress_cb is not None:
            # Per-accepted-lineup progress for background-job callers
            # (the web API): (rows built, rows requested). Default
            # None = historical behavior, byte-identical output.
            progress_cb(len(rows), count)
        last_built_attempt=attempt
        for slot,i in chosen:
            pid=str(df.loc[i,"ID"]); exposure[pid]+=1
            if slot=="CPT": cpt_exp[pid]+=1
        if prog is not None:
            prog.progress(min(1.0,len(rows)/max(1,count)), text=f"Built {len(rows)} / {count} · attempt {attempt+1}")

    # ---- Count guarantee + minimum repair (Oct 27, 2026) ----
    # The main loop can stall short of `count` (its 40-attempt dead-build
    # break) or land the count with a minimum unmet. What follows never
    # relaxes a user rule: exposure caps stay hard in every solve, and
    # the only preference ever stepped down is uniqueness — announced
    # in build_notes when it happens. Anything still unmet afterwards
    # is reported by name below, never silently.
    fill_stall = [0]

    def _fill_attempt(mu, allow_no_role=False):
        game_world=choose_game_world(rng,entry_format,field_size,script)
        _wtries=0
        while world_share_capped(world_counts,game_world,count,_wcap) and _wtries<12:
            game_world=choose_game_world(rng,entry_format,field_size,script); _wtries+=1
        if world_share_capped(world_counts,game_world,count,_wcap):
            return False
        world_weights=world_construction_weights(construction_weights,game_world,entry_format)
        preferred=choose_construction_target(rng,teams,world_weights,script,script_team)
        cand_targets=[]
        if preferred in legal_targets:
            cand_targets.append(preferred)
        rest=[x for x in legal_targets if x not in cand_targets]
        if len(rest)>1:
            order=rng.permutation(len(rest))
            rest=[rest[int(i)] for i in order]
        cand_targets.extend(rest)
        remaining=count-len(rows)
        f_overall=[]; f_cpt=[]
        for pid,tgt in targets.items():
            need=tgt["min"]-exposure[pid]
            cneed=tgt["cpt_min"]-cpt_exp[pid]
            if need>=remaining and need>0: f_overall.append(pid)
            if cneed>=remaining and cneed>0: f_cpt.append(pid)
        blocked={pid for pid,tgt in targets.items()
                 if tgt["max"]<count and exposure[pid]>=tgt["max"]}
        cpt_blocked={pid for pid,tgt in targets.items()
                     if tgt["cpt_max"]<count and cpt_exp[pid]>=tgt["cpt_max"]}
        f_blocked = blocked if allow_no_role else (blocked | no_role_ids)
        f_blocked, f_noise = _supply_perturb(fill_stall[0], f_blocked)
        chosen=None
        for target in cand_targets:
            chosen=solve_showdown_one(
                df,aggr,rng,strategy_map,min_salary,max_salary,target,script,script_team,
                cpt_qb_passcatchers,wrte_cpt_qb_pair_pct,rb_cpt_dst_k_pct,max_k,max_dst,
                mu,previous,exposure,cpt_exp,len(rows),noise_scale=f_noise,
                relationship_rules=relationship_rules,forced_overall_ids=f_overall,
                forced_cpt_ids=f_cpt,game_world=game_world,world_influence=world_influence,
                qb_exact=qb_exact,blocked_ids=f_blocked,cpt_blocked_ids=cpt_blocked,
                qb_linkage_exempt_teams=set(guard_exempt)
            )
            if chosen:
                break
        if not chosen:
            return False
        ids=tuple(sorted(str(df.loc[i,"ID"]) for _,i in chosen))
        cpt_id=str(df.loc[[i for slot,i in chosen if slot=="CPT"][0],"ID"])
        key=(cpt_id,ids)
        if key in seen: return False
        for slot,i in chosen:
            pid=str(df.loc[i,"ID"]); tgt=targets.get(pid)
            if tgt is None: continue
            if exposure[pid]+1>tgt["max"]: return False
            if slot=="CPT" and cpt_exp[pid]+1>tgt["cpt_max"]: return False
        coherence=showdown_lineup_coherence(df,chosen,game_world,
                                            exempt_teams=guard_exempt)
        if not coherence["Accept"]:
            return False
        seen.add(key)
        world_counts[game_world]+=1
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
        if progress_cb is not None:
            progress_cb(len(rows), count)
        for slot,i in chosen:
            pid=str(df.loc[i,"ID"]); exposure[pid]+=1
            if slot=="CPT": cpt_exp[pid]+=1
        return True

    if len(rows)<count:
        mu_ladder=[]
        for mu in (min_unique, max(0,min_unique-1), 0):
            if mu not in mu_ladder: mu_ladder.append(mu)
        for mu in mu_ladder:
            if len(rows)>=count: break
            added_before=len(rows); fails=0
            for _ in range(60):
                if len(rows)>=count or fails>=25: break
                if _fill_attempt(mu):
                    fails=0; fill_stall[0]=0
                else:
                    fails+=1; fill_stall[0]+=1
            if mu<min_unique and len(rows)>added_before:
                build_notes.append(
                    f"Uniqueness relaxed from {min_unique} to {mu} to "
                    f"reach the requested {count} lineups.")

    # Punt-floor breach fills: the set is still short and every clean
    # avenue above has run — only now may no-role players enter, one
    # lineup at a time, and each one used is named after repair below.
    if len(rows)<count and no_role_ids:
        fails=0
        for _ in range(60):
            if len(rows)>=count or fails>=25: break
            if _fill_attempt(0, allow_no_role=True):
                fails=0; fill_stall[0]=0
            else:
                fails+=1; fill_stall[0]+=1

    # Minimum repair by in-place swaps. A swap may only move a player
    # INTO a lineup when every hard rule still holds afterwards —
    # salary, team split, K/DST caps, QB count, the captain-pairing
    # rules for the resulting captain, portfolio uniqueness, coherence,
    # and every other player's own min/max. Relationship-rule builds
    # skip repair (pair rules are not re-validated here); their unmet
    # minimums are reported by name below instead of risked.
    idx_by_id={}
    for i in df.index:
        idx_by_id.setdefault(str(df.loc[i,"ID"]), i)
    id_by_index={i: str(df.loc[i,"ID"]) for i in df.index}
    allowed_max={c for _, c in legal_targets} if len(teams)==2 else set()

    def _validate_change(k, new_indices):
        new_ids=[id_by_index[i] for i in new_indices]
        if len(set(new_ids))!=6: return False
        cpt_i=new_indices[0]; flex_i=new_indices[1:]
        salary=int(df.loc[cpt_i,"CaptainSalary"])+int(df.loc[flex_i,"FlexSalary"].sum())
        if salary<int(min_salary) or salary>int(max_salary): return False
        if len(teams)==2:
            split=defaultdict(int)
            for i in new_indices: split[str(df.loc[i,"Team"])]+=1
            if len(split)<2 or max(split.values()) not in allowed_max: return False
        if int(df.loc[new_indices,"is_K"].sum())>int(max_k): return False
        if int(df.loc[new_indices,"is_DST"].sum())>int(max_dst): return False
        if qb_exact is not None and int(df.loc[new_indices,"is_QB"].sum())!=int(qb_exact): return False
        # Exposure deltas: nobody drops below their own minimum, nobody
        # rises above their own maximum.
        old_ids=[id_by_index[i] for i in previous[k]]
        delta=defaultdict(int)
        for pid in new_ids: delta[pid]+=1
        for pid in old_ids: delta[pid]-=1
        for pid,d in delta.items():
            if d==0: continue
            tgt=targets.get(pid)
            if tgt is None: continue
            if d>0 and exposure[pid]+d>tgt["max"]: return False
            if d<0 and exposure[pid]+d<tgt["min"]: return False
        if new_ids[0]!=old_ids[0]:
            tgt=targets.get(new_ids[0])
            if tgt is not None and cpt_exp[new_ids[0]]+1>tgt["cpt_max"]: return False
            tgt=targets.get(old_ids[0])
            if tgt is not None and cpt_exp[old_ids[0]]-1<tgt["cpt_min"]: return False
        # Uniqueness against every other lineup (player-set overlap).
        new_set=set(new_ids)
        for m in range(len(previous)):
            if m==k: continue
            other={id_by_index[i] for i in previous[m]}
            if len(new_set & other)>6-min_unique: return False
        # Captain-pairing hard rules for the resulting captain.
        cap_row=df.loc[cpt_i]; others=new_indices[1:]
        if cap_row["is_QB"] and cpt_qb_passcatchers!=0:
            n_pc=int(sum(1 for i in others
                         if str(df.loc[i,"Team"])==str(cap_row["Team"])
                         and bool(df.loc[i,"is_passcatcher"])))
            if cpt_qb_passcatchers>0 and n_pc<int(cpt_qb_passcatchers): return False
            if cpt_qb_passcatchers<0 and n_pc>abs(int(cpt_qb_passcatchers)): return False
        if cap_row["is_passcatcher"] and float(wrte_cpt_qb_pair_pct)>=99.5:
            if not any(str(df.loc[i,"Team"])==str(cap_row["Team"]) and bool(df.loc[i,"is_QB"]) for i in others):
                return False
        if cap_row["is_RB"] and float(rb_cpt_dst_k_pct)>=99.5:
            if not any(str(df.loc[i,"Team"])==str(cap_row["Team"]) and (bool(df.loc[i,"is_DST"]) or bool(df.loc[i,"is_K"])) for i in others):
                return False
        return True

    def _apply_change(k, new_indices):
        chosen=[(slot,new_indices[j]) for j,slot in enumerate(SHOWDOWN_SLOTS)]
        world=rows[k].get("Game World")
        coherence=showdown_lineup_coherence(df,chosen,world,
                                            exempt_teams=guard_exempt)
        if not coherence["Accept"]:
            return False
        old_ids=[id_by_index[i] for i in previous[k]]
        old_key=(old_ids[0],tuple(sorted(old_ids)))
        new_ids=[id_by_index[i] for i in new_indices]
        detail=showdown_lineup_details(df,chosen,strategy_map,script,script_team,world)
        detail.update({
            "Coherence Score":coherence["Coherence Score"],
            "Lineup Story":coherence["Lineup Story"],
            "Coherence Flags":coherence["Coherence Flags"]
        })
        row=dict(detail)
        for slot,i in chosen:
            row[slot]=df.loc[i,"Name"]; row[slot+"_ID"]=str(df.loc[i,"ID"])
            row[slot+"_NameID"] = df.loc[i,"CPT_NameID"] if slot=="CPT" else df.loc[i,"FLEX_NameID"]
        for pid in old_ids: exposure[pid]-=1
        if old_ids[0]: cpt_exp[old_ids[0]]-=1
        for pid in new_ids: exposure[pid]+=1
        cpt_exp[new_ids[0]]+=1
        seen.discard(old_key)
        seen.add((new_ids[0],tuple(sorted(new_ids))))
        previous[k]=list(new_indices)
        rows[k]=row
        return True

    def _repair_mins():
        if relationship_rules and any(r.get("enabled",True) for r in relationship_rules):
            return
        ops=0
        while ops<80:
            deficit=[pid for pid,tgt in targets.items() if exposure[pid]<tgt["min"]]
            cpt_deficit=[pid for pid,tgt in targets.items() if cpt_exp[pid]<tgt["cpt_min"]]
            if not deficit and not cpt_deficit:
                break
            progressed=False
            for pid in deficit:
                p_idx=idx_by_id.get(pid)
                if p_idx is None: continue
                done=False
                for k in range(len(rows)-1,-1,-1):
                    if p_idx in previous[k]: continue
                    for j in range(1,6):
                        new=list(previous[k]); new[j]=p_idx
                        if _validate_change(k,new) and _apply_change(k,new):
                            done=True; break
                    if done: break
                if done:
                    progressed=True; ops+=1; break
            if progressed: continue
            for pid in cpt_deficit:
                p_idx=idx_by_id.get(pid)
                if p_idx is None: continue
                done=False
                for k in range(len(rows)-1,-1,-1):
                    cur=previous[k]
                    if p_idx not in cur[1:]: continue
                    j=cur.index(p_idx)
                    new=list(cur); new[0]=p_idx; new[j]=cur[0]
                    if _validate_change(k,new) and _apply_change(k,new):
                        done=True; break
                if done:
                    progressed=True; ops+=1; break
            if progressed: continue
            for pid in cpt_deficit:
                p_idx=idx_by_id.get(pid)
                if p_idx is None: continue
                done=False
                for k in range(len(rows)-1,-1,-1):
                    cur=previous[k]
                    if p_idx in cur: continue
                    for j in range(1,6):
                        # Donor at j leaves; the old captain drops to
                        # that flex slot; P enters as captain.
                        new=list(cur); new[0]=p_idx; new[j]=cur[0]
                        if _validate_change(k,new) and _apply_change(k,new):
                            done=True; break
                    if done: break
                if done:
                    progressed=True; ops+=1; break
            if not progressed:
                break

    if rows:
        _repair_mins()
    if no_role and rows:
        _used_nr = {}
        for r in rows:
            for _col in ("CPT_ID", "FLEX1_ID", "FLEX2_ID", "FLEX3_ID",
                         "FLEX4_ID", "FLEX5_ID"):
                _pid = str(r.get(_col, ""))
                if _pid in no_role:
                    _used_nr[_pid] = _used_nr.get(_pid, 0) + 1
        build_notes.extend(
            _punt_floor_notes(no_role, _used_nr, len(rows)))
    if guard_exempt and rows:
        # His settings outrank the QB-linkage guard — but an exempt
        # shape that actually shipped is named, never silent.
        _ovr = {}
        for r in rows:
            _ids = [str(r.get(_c, "")) for _c in
                    ("CPT_ID", "FLEX1_ID", "FLEX2_ID", "FLEX3_ID",
                     "FLEX4_ID", "FLEX5_ID")]
            for _t, _n, _qn in _showdown_linkage_breaches(df, _ids):
                if _t in guard_exempt:
                    _ovr[_t] = _ovr.get(_t, 0) + 1
        for _t in sorted(_ovr):
            _qn = _showdown_team_qb_name(df, _t) or "their quarterback"
            build_notes.append(
                f"QB-linkage guard overridden by your settings: "
                f"{_ovr[_t]} of {len(rows)} lineups carry 3+ {_t} "
                f"pass catchers without {_qn} — delivered because "
                f"{guard_exempt[_t]}.")
    for pid,tgt in targets.items():
        if tgt["min"]>0 and exposure[pid]<tgt["min"]:
            build_notes.append(
                f"Min Exposure: {tgt['name']} targeted "
                f"{_pct_label(tgt['min_pct'])}% ({tgt['min']} of {count} "
                f"lineups), reached {exposure[pid]} — the legal-lineup "
                f"supply under your other rules ran out.")
        if tgt["cpt_min"]>0 and cpt_exp[pid]<tgt["cpt_min"]:
            build_notes.append(
                f"CPT Min: {tgt['name']} targeted "
                f"{_pct_label(tgt['cpt_min_pct'])}% ({tgt['cpt_min']} of "
                f"{count}) at captain, reached {cpt_exp[pid]} — the "
                f"legal-lineup supply under your other rules ran out.")
    if len(rows)<count:
        if any("leave room for only" in n for n in build_notes):
            cause="your exposure caps leave too little room (see the cap note above)"
        elif any(tgt["min"]>exposure[pid] or tgt["cpt_min"]>cpt_exp[pid]
                 for pid,tgt in targets.items()):
            cause="the minimum-exposure targets could not all be satisfied at once"
        elif cpt_locked_ids:
            _nm=(targets.get(next(iter(cpt_locked_ids)), {})
                 or {}).get("name", "your locked captain")
            cause=(f"the legal-lineup supply with {_nm} locked at "
                   f"captain ran out under your other rules")
        else:
            cause="the legal-lineup supply under your rules ran out"
        build_notes.append(
            f"Shortfall: built {len(rows)} of {count} requested lineups "
            f"— {cause}.")

    if prog is not None:
        prog.empty()
    out=pd.DataFrame(rows)
    if out.empty:
        out.attrs["showdown_notes"]=list(dict.fromkeys(build_notes))
        return out
    out=add_showdown_ratings(out,aggr)
    out["Contest Format"]=entry_format
    out["Field Size"]=int(field_size)
    out["Contest Aggression"] = round(aggr*100,1)
    out=out.sort_values(["Rating Score","Projection"],ascending=[False,False]).reset_index(drop=True)
    out.insert(0,"Rank",np.arange(1,len(out)+1))
    out.attrs["showdown_notes"]=list(dict.fromkeys(build_notes))
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


def showdown_simulate_game(df, sims=10000, seed=42):
    """Single-game top-down simulation for showdown: team scores -> team
    fantasy -> player shares.

    Mirrors classic.topdown_simulate_slate but for one game: both teams'
    scores are drawn from nflverse four-season-weighted offense/defense
    ratings (current season heaviest), blended with the market's implied
    team totals when nflverse carries a line for the game
    (classic._anchored_team_means), with a home-field edge in the
    ratings prior and pace-correlated noise, then each
    team's fantasy is dealt to its players by projection-implied usage
    shares with individual lognormal noise. Defenses ride the
    points-allowed bracket off the opponent's simulated score.

    All player scores are FLEX (1x) scoring — captain 1.5x is applied at
    lineup time, not here.

    Returns a dict with:
      player_stats: Name, Position, Team, Sim Mean, Sim P10, Sim P25,
                    Sim P50, Sim P75, Sim P90, Sim Std, P(3x) %
      worlds:       DataFrame (sims rows x players) of simulated FLEX
                    fantasy points, columns are df integer positions.
      game_env:     dict with Game, Mean DFS env, P75, P90 for the game.
    """
    from dfs_lab.data import _slate_season, _dst_points_allowed_fantasy
    from dfs_lab.classic import (_anchored_team_means,
                                 _nflverse_team_ratings,
                                 _team_fantasy_calibration)
    d = df.reset_index(drop=True).copy()
    n = len(d)
    empty = {"player_stats": pd.DataFrame(), "worlds": pd.DataFrame(), "game_env": {}}
    if n == 0:
        return empty
    season = _slate_season(d)
    rng = np.random.default_rng(int(seed))
    sims = int(max(1000, sims))

    away = str(d["Away"].iloc[0]).upper() if "Away" in d.columns else ""
    home = str(d["Home"].iloc[0]).upper() if "Home" in d.columns else ""
    if not away or not home or away == "NAN" or home == "NAN":
        return empty

    off, deff, lg = _nflverse_team_ratings(season)
    ratio, emp_team_fp = _team_fantasy_calibration(season)
    HFA, SCORE_SD, PACE_CORR = 1.8, 10.0, 0.2

    proj = pd.to_numeric(d["My Proj"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy()
    pos = d["Position"].astype(str).str.upper().to_numpy()
    teams = d["Team"].astype(str).str.upper().to_numpy()
    is_dst = np.isin(pos, ["DST", "D/ST"])

    ea, eh = _anchored_team_means(away, home, season, off, deff, lg)
    cov = [[SCORE_SD ** 2, PACE_CORR * SCORE_SD ** 2],
           [PACE_CORR * SCORE_SD ** 2, SCORE_SD ** 2]]
    draw = rng.multivariate_normal([ea, eh], cov, size=sims)
    pts_a = np.clip(draw[:, 0], 0, None)
    pts_h = np.clip(draw[:, 1], 0, None)
    skill_fp = {away: ratio * pts_a, home: ratio * pts_h}

    # Usage shares from projections, scaled by historical coverage.
    share = np.zeros(n)
    for t in (away, home):
        m = (teams == t) & (~is_dst)
        ps = proj[m].sum()
        if ps <= 0:
            continue
        covr = min(1.0, max(0.05, ps / max(emp_team_fp.get(t, ps), 1.0)))
        share[m] = proj[m] / ps * covr

    if "Sim Vol" in d.columns:
        pvol = pd.to_numeric(d["Sim Vol"], errors="coerce").fillna(0.45).to_numpy()
    else:
        pvol = np.array([{"QB": 0.26, "RB": 0.42, "WR": 0.50, "TE": 0.48,
                          "K": 0.55, "PK": 0.55, "DST": 0.62}.get(p, 0.45) for p in pos])
    ind = 0.65 * np.clip(pvol, 0.12, 0.95)
    noise = np.exp(rng.normal(0, 1, size=(sims, n)) * ind[None, :] - 0.5 * ind[None, :] ** 2)

    team_fp = np.array([skill_fp.get(t, np.zeros(sims)) for t in teams]).T  # sims x n
    worlds = np.zeros((sims, n))
    sk = ~is_dst
    worlds[:, sk] = team_fp[:, sk] * share[sk][None, :] * noise[:, sk]

    # Defenses ride their projection, adjusted by the opponent's simulated
    # score through the points-allowed bracket.
    opp_pts = {away: pts_h, home: pts_a}
    for j in np.where(is_dst)[0]:
        t = teams[j]
        base = max(float(proj[j]), 0.5)
        adj = (np.vectorize(_dst_points_allowed_fantasy)(opp_pts.get(t, np.zeros(sims))) - 1.0)
        dv = min(0.9, max(0.2, float(pvol[j])))
        worlds[:, j] = np.clip(base + adj + rng.normal(0, dv * base, sims), 0, None)

    sal = pd.to_numeric(d.get("FlexSalary", d.get("Salary", 0)), errors="coerce").fillna(0.0).to_numpy()
    thr3x = 3.0 * sal / 1000.0
    player_stats = pd.DataFrame({
        "Name": d["Name"], "Position": d["Position"], "Team": d["Team"],
        "Sim Mean": np.round(worlds.mean(axis=0), 2),
        "Sim P10": np.round(np.percentile(worlds, 10, axis=0), 2),
        "Sim P25": np.round(np.percentile(worlds, 25, axis=0), 2),
        "Sim P50": np.round(np.median(worlds, axis=0), 2),
        "Sim P75": np.round(np.percentile(worlds, 75, axis=0), 2),
        "Sim P90": np.round(np.percentile(worlds, 90, axis=0), 2),
        "Sim Std": np.round(worlds.std(axis=0), 2),
        "P(3x) %": np.round(100 * (worlds > thr3x[None, :]).mean(axis=0), 1),
    })

    gfp = skill_fp[away] + skill_fp[home]
    for j in np.where(is_dst)[0]:
        gfp = gfp + worlds[:, j]
    game_env = {"Game": f"{away}@{home}",
                "Mean DFS env": round(float(gfp.mean()), 1),
                "P75": round(float(np.percentile(gfp, 75)), 1),
                "P90": round(float(np.percentile(gfp, 90)), 1)}

    worlds_df = pd.DataFrame(worlds, columns=d.index)
    return {"player_stats": player_stats, "worlds": worlds_df, "game_env": game_env}


def showdown_optimal_rates(worlds_df, df, n_sample=500, seed=7, salary_cap=50000,
                           progress_cb=None):
    """Optimal % and CPT % per player for showdown.

    Solves one MILP per sampled world: 1 captain (1.5x points,
    CaptainSalary) + 5 FLEX (FlexSalary), salary cap, CPT-eligible
    captains only. Optimal % = share of worlds in the optimal lineup at
    either slot; CPT % = share as the optimal captain.
    Returns DataFrame with columns Name, "Optimal %", "CPT %".
    """
    cols_out = pd.DataFrame(columns=["Name", "Optimal %", "CPT %"])
    if worlds_df is None or worlds_df.empty or df is None or df.empty:
        return cols_out
    d = df.reset_index(drop=True)
    if "ActiveForBuild" in d.columns:
        d = d[d["ActiveForBuild"]].reset_index(drop=True)
    if d.empty or "FlexSalary" not in d.columns:
        return cols_out

    names = d["Name"].astype(str).tolist()
    flex_sal = pd.to_numeric(d["FlexSalary"], errors="coerce").fillna(0).to_numpy(float)
    cpt_sal = pd.to_numeric(d.get("CaptainSalary", d["FlexSalary"] * 1.5),
                            errors="coerce").fillna(0).to_numpy(float)
    cpt_ok = d["CPT?"].astype(bool).to_numpy() if "CPT?" in d.columns else np.ones(len(d), bool)
    n = len(d)

    # Variables: x_i (FLEX) for i in 0..n-1, c_i (captain) for i in n..2n-1.
    total = 2 * n
    rows, lows, highs = [], [], []
    rows.append({i: 1.0 for i in range(n)}); lows.append(5.0); highs.append(5.0)
    rows.append({n + i: 1.0 for i in range(n) if cpt_ok[i]}); lows.append(1.0); highs.append(1.0)
    for i in range(n):
        rows.append({i: 1.0, n + i: 1.0}); lows.append(0.0); highs.append(1.0)
    rows.append({i: flex_sal[i] for i in range(n)} | {n + i: cpt_sal[i] for i in range(n)})
    lows.append(0.0); highs.append(float(salary_cap))
    A = lil_matrix((len(rows), total), dtype=float)
    for r, coeff in enumerate(rows):
        for col, val in coeff.items():
            A[r, col] = val
    cons = LinearConstraint(A.tocsr(), np.array(lows), np.array(highs))
    integrality = np.ones(total)
    bounds = Bounds(np.zeros(total), np.ones(total))

    # World column k corresponds to row k of the full (pre-filter) pool.
    full = df.reset_index(drop=True)
    name_to_k = {str(nm): k for k, nm in enumerate(full["Name"].astype(str))}
    keep = [name_to_k.get(nm) for nm in names]
    if any(k is None for k in keep):
        return cols_out
    mat_all = worlds_df.to_numpy(dtype=float)
    if max(keep) >= mat_all.shape[1]:
        return cols_out
    mat = mat_all[:, keep]

    rng = np.random.default_rng(int(seed))
    n_worlds = mat.shape[0]
    idx = rng.choice(n_worlds, size=min(int(n_sample), n_worlds), replace=False)
    opt_hits = np.zeros(n)
    cpt_hits = np.zeros(n)
    total_done = 0
    for t, w in enumerate(idx):
        pts = mat[w]
        c = np.zeros(total)
        c[:n] = -pts
        c[n:] = -1.5 * pts
        try:
            res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
        except Exception:
            continue
        if res.status != 0 or res.x is None:
            continue
        x = (res.x[:n] > 0.5)
        cc = (res.x[n:] > 0.5)
        opt_hits[x | cc] += 1
        cpt_hits[cc] += 1
        total_done += 1
        if progress_cb and (t + 1) % 200 == 0:
            progress_cb(t + 1, len(idx))
    if progress_cb:
        progress_cb(len(idx), len(idx))
    denom = max(total_done, 1)
    return pd.DataFrame({"Name": names,
                         "Optimal %": np.round(100 * opt_hits / denom, 1),
                         "CPT %": np.round(100 * cpt_hits / denom, 1)})


# =====================================================================
# WS26 — Trust release, Piece 1: sim-driven showdown selection.
#
# Until Oct 28, 2026 the simulated worlds never entered showdown
# generation, selection, or rating: candidates came from a MILP on one
# adjusted-projection vector, selection delivered them in Rating Score
# order, and the rating was a percentile composite INSIDE the delivered
# set — so a lineup could rate #2 of 20 while ranking 9th of its own
# 20 by sim mean and winning almost no worlds (Chris's CAR@GB Lineup
# 2). The helpers below make the worlds an author and the judge:
# per-world optimal lineups join the candidate pool, every candidate
# is scored against the full world set, selection prefers the world
# score (select_showdown_portfolio's order_scores), and the rating is
# anchored on world metrics (add_showdown_ratings' use_world_metrics).
# =====================================================================

#: Enrichment sample: per-world optimal solves drawn from the cached
#: worlds. A named constant, tuned by measurement (trust-release spec
#: §1.4): the battery measures the end-to-end build time with this
#: sample; if the droplet budget (20 lineups ≤ 45 s) fails, the sample
#: steps down — it is never a hidden behavior.
SHOWDOWN_WORLD_SAMPLE = 300


def showdown_align_worlds(worlds_df, sim_frame, df):
    """Align a sim worlds frame to another frame's row order.

    `worlds_df` columns are the row positions of `sim_frame` (the
    frame showdown_simulate_game ran on). Returns an (n_worlds x
    len(df)) float array whose column j is df row j's simulated FLEX
    points, matched by player ID (Name fallback), or None when the
    worlds cannot be aligned (missing players, empty inputs)."""
    if worlds_df is None or sim_frame is None or df is None:
        return None
    if len(worlds_df) == 0 or len(df) == 0 or len(sim_frame) == 0:
        return None
    mat = worlds_df.to_numpy(dtype=float)
    if mat.shape[1] != len(sim_frame):
        return None
    if "ID" in sim_frame.columns and "ID" in df.columns:
        key_sim = [str(x) for x in sim_frame["ID"]]
        key_df = [str(x) for x in df["ID"]]
    else:
        key_sim = [str(x) for x in sim_frame["Name"]]
        key_df = [str(x) for x in df["Name"]]
    col_of = {}
    for k, key in enumerate(key_sim):
        col_of.setdefault(key, k)
    keep = []
    for key in key_df:
        k = col_of.get(key)
        if k is None:
            return None
        keep.append(k)
    return mat[:, keep]


def showdown_world_optimal_lineups(worlds_mat, df, n_sample=SHOWDOWN_WORLD_SAMPLE,
                                   seed=7, salary_cap=50000, min_salary=0,
                                   strategy_map=None, max_k=None,
                                   max_dst=None, construction_weights=None):
    """World-authored candidates: the optimal legal lineup of sampled
    simulated worlds, as distinct player sets with win counts.

    The solve is showdown_optimal_rates' problem (1 CPT at 1.5x + 5
    FLEX, salary band) hardened with the build's own legality, the way
    Classic's _world_optimal_lineups feeds its sim-first generator:
    the user's excludes and captain pool bind pre-solve, locks bind
    in-solve, the punt floor's no-role players are unselectable, and
    the QB-linkage guard's constraint (catchers - 4*QB <= 2 per team,
    exempt teams skipped) holds — a world may not author a candidate
    the delivery rules would reject. The user's Build Shape weights
    bind the same way when passed as `construction_weights` (the
    allowed team splits only; Oct 28, 2026). Returns a list of
    {"cpt_id", "player_ids" (sorted tuple), "wins"} sorted by wins
    desc; empty when the inputs cannot support it."""
    out = []
    if worlds_mat is None or df is None or len(df) == 0:
        return out
    d = df.reset_index(drop=True)
    smap = strategy_map or {}
    # Build Shape enforcement (Oct 28, 2026): this supply ignored
    # construction shapes entirely when it shipped — its only team
    # constraint was "both teams represented" (1..5 per team), which
    # legalizes the 5-1 splits Chris had unchecked on CAR@GB, and
    # the API merged the candidates in unfiltered. With the user's
    # raw weights the allowed per-team counts derive from the
    # allowed shapes ({3-3, 4-2} -> {2, 3, 4}); the solve's
    # per-team rows tighten to that count range below, and the
    # returned list is post-filtered to exact allowed splits,
    # because a bare range leaks non-adjacent mixes (allowed
    # {3-3, 5-1} -> counts {1, 3, 5} would still admit 4-2 inside
    # [1, 5]). Weights None or all-allowed -> _shape_counts stays
    # None and behavior is byte-identical to before.
    _allowed_shapes = _allowed_showdown_shapes(construction_weights)
    _shape_counts = (_allowed_shape_team_counts(_allowed_shapes)
                     if _allowed_shapes is not None else None)
    active_col = (d["ActiveForBuild"].to_numpy(bool)
                  if "ActiveForBuild" in d.columns
                  else np.ones(len(d), bool))

    def _excluded(pid):
        st = smap.get(str(pid), {}) or {}
        return bool(st.get("Exclude", False)) \
            or st.get("Priority") == "Exclude"

    no_role = set(_showdown_no_role_players(d, smap))
    usable = []
    for pos in range(len(d)):
        pid = str(d["ID"].iloc[pos])
        if not bool(active_col[pos]) or _excluded(pid) or pid in no_role:
            continue
        usable.append(pos)
    if len(usable) < 6:
        return out
    sub = d.iloc[usable].reset_index(drop=True)
    n = len(sub)
    ids = [str(x) for x in sub["ID"]]
    flex_sal = pd.to_numeric(sub["FlexSalary"], errors="coerce") \
        .fillna(0).to_numpy(float)
    cpt_sal = pd.to_numeric(sub.get("CaptainSalary", sub["FlexSalary"] * 1.5),
                            errors="coerce").fillna(0).to_numpy(float)
    if strategy_map is not None:
        pool = set(captain_pool_ids(d, smap))
        cpt_ok = np.array([pid in pool for pid in ids])
    elif "CPT?" in sub.columns:
        cpt_ok = sub["CPT?"].astype(bool).to_numpy()
    else:
        cpt_ok = np.ones(n, bool)
    if not cpt_ok.any():
        return out

    total = 2 * n
    rows, lows, highs = [], [], []
    rows.append({i: 1.0 for i in range(n)}); lows.append(5.0); highs.append(5.0)
    rows.append({n + i: 1.0 for i in range(n) if cpt_ok[i]})
    lows.append(1.0); highs.append(1.0)
    for i in range(n):
        rows.append({i: 1.0, n + i: 1.0}); lows.append(0.0); highs.append(1.0)
    rows.append({i: flex_sal[i] for i in range(n)}
                | {n + i: cpt_sal[i] for i in range(n)})
    lows.append(float(min_salary)); highs.append(float(salary_cap))
    # Locks bind in-solve (Classic sim-first semantics): a plain Lock
    # must appear somewhere; a CPT Lock must captain.
    for i, pid in enumerate(ids):
        st = smap.get(pid, {}) or {}
        if bool(st.get("CPT Lock", False)) and not _excluded(pid):
            rows.append({n + i: 1.0}); lows.append(1.0); highs.append(1.0)
        elif bool(st.get("Lock", False)) and not _excluded(pid):
            rows.append({i: 1.0, n + i: 1.0}); lows.append(1.0); highs.append(1.0)
    # Both teams represented (the generator's own legality). With
    # Build Shape weights in hand the per-team range tightens from
    # [1, 5] to the allowed count range (Oct 28, 2026 — see the
    # _shape_counts note at the top of this function); the exact
    # split is enforced by the post-filter on the returned list.
    teams = [str(t) for t in sub["Team"].dropna().unique() if str(t)]
    if len(teams) == 2:
        _team_lo, _team_hi = 1.0, 5.0
        if _shape_counts:
            _team_lo = float(min(_shape_counts))
            _team_hi = float(max(_shape_counts))
        for t in teams:
            coeff = {}
            for i in np.where(sub["Team"].astype(str).eq(t).to_numpy())[0]:
                coeff[int(i)] = 1.0
                coeff[n + int(i)] = 1.0
            if coeff:
                rows.append(coeff)
                lows.append(_team_lo); highs.append(_team_hi)
    # Position caps when the build carries them.
    for mask_col, cap in (("is_K", max_k), ("is_DST", max_dst)):
        if cap is None or mask_col not in sub.columns:
            continue
        coeff = {}
        for i in np.where(sub[mask_col].to_numpy(bool))[0]:
            coeff[int(i)] = 1.0
            coeff[n + int(i)] = 1.0
        if coeff:
            rows.append(coeff); lows.append(0.0); highs.append(float(cap))
    # QB-linkage guard in-solve, exempt teams skipped (the guard's own
    # semantics — see solve_showdown_one).
    exempt = set(_showdown_qb_linkage_exemptions(d, smap))
    if "is_passcatcher" in sub.columns and "is_QB" in sub.columns:
        for t in teams:
            if t in exempt:
                continue
            coeff = {}
            tm = sub["Team"].astype(str).eq(t).to_numpy()
            for i in np.where(tm & sub["is_passcatcher"].to_numpy(bool))[0]:
                coeff[int(i)] = coeff.get(int(i), 0.0) + 1.0
                coeff[n + int(i)] = coeff.get(n + int(i), 0.0) + 1.0
            for i in np.where(tm & sub["is_QB"].to_numpy(bool))[0]:
                coeff[int(i)] = coeff.get(int(i), 0.0) - 4.0
                coeff[n + int(i)] = coeff.get(n + int(i), 0.0) - 4.0
            if coeff:
                rows.append(coeff); lows.append(-np.inf); highs.append(2.0)
    A = lil_matrix((len(rows), total), dtype=float)
    for r, coeff in enumerate(rows):
        for col, val in coeff.items():
            A[r, col] = val
    cons = LinearConstraint(A.tocsr(), np.array(lows), np.array(highs))
    integrality = np.ones(total)
    # Captain variables outside the captain pool must be hard-zero:
    # the captain-sum row only spans eligible columns, so without
    # this the objective parks every ineligible captain variable at
    # 1 and the extraction check rejects every solve.
    ub = np.ones(total)
    ub[n:] = np.where(cpt_ok, 1.0, 0.0)
    bounds = Bounds(np.zeros(total), ub)

    mat = np.asarray(worlds_mat, dtype=float)
    if mat.shape[1] != len(d):
        return out
    sub_mat = mat[:, usable]
    rng = np.random.default_rng(int(seed))
    n_worlds = sub_mat.shape[0]
    idx = rng.choice(n_worlds, size=min(int(n_sample), n_worlds),
                     replace=False)
    wins = {}
    for w in idx:
        pts = sub_mat[w]
        c = np.zeros(total)
        c[:n] = -pts
        c[n:] = -1.5 * pts
        try:
            res = milp(c=c, constraints=cons, integrality=integrality,
                       bounds=bounds,
                       options={"time_limit": 2.0, "presolve": True})
        except Exception:
            continue
        if res.status != 0 or res.x is None:
            continue
        x = res.x[:n] > 0.5
        cc = res.x[n:] > 0.5
        if x.sum() != 5 or cc.sum() != 1:
            continue
        chosen = [ids[i] for i in np.where(x | cc)[0]]
        cpt_id = ids[int(np.where(cc)[0][0])]
        key = (cpt_id, tuple(sorted(chosen)))
        wins[key] = wins.get(key, 0) + 1
    out = [{"cpt_id": cpt, "player_ids": pids, "wins": int(k)}
           for (cpt, pids), k in wins.items()]
    out.sort(key=lambda r: (-r["wins"], r["cpt_id"], r["player_ids"]))
    if _allowed_shapes is not None and out:
        # Exact-split post-filter (Oct 28, 2026): the tightened
        # in-solve range cannot express non-adjacent allowed sets,
        # so any split the range leaked is dropped here — a
        # world-authored candidate the user's Build Shape switched
        # off never reaches the merge.
        _team_of = {str(r["ID"]): str(r["Team"])
                    for _, r in d.iterrows()}
        _kept = []
        for _o in out:
            _per_team = {}
            for _p in _o["player_ids"]:
                _t = _team_of.get(str(_p))
                if _t is not None:
                    _per_team[_t] = _per_team.get(_t, 0) + 1
            if len(_per_team) == 2 and _showdown_shape_of(
                    _per_team.values()) in _allowed_shapes:
                _kept.append(_o)
        out = _kept
    return out


def showdown_lineup_totals(worlds_mat, df, lineups):
    """Per-world totals for a set of lineups: (len(lineups) x
    n_worlds) float array, captain at 1.5x. `lineups` is an iterable
    of (cpt_id, player_ids) pairs or a result/candidate DataFrame
    with CPT_ID + FLEX1..5_ID columns."""
    if worlds_mat is None or df is None or len(df) == 0:
        return np.zeros((0, 0))
    d = df.reset_index(drop=True)
    pos_of = {str(pid): i for i, pid in enumerate(d["ID"])}
    pairs = _lineup_pairs(lineups)
    mat = np.asarray(worlds_mat, dtype=float)
    if mat.shape[1] != len(d) or not pairs:
        return np.zeros((0, mat.shape[0] if mat.ndim == 2 else 0))
    totals = np.zeros((len(pairs), mat.shape[0]))
    for r, (cpt_id, pids) in enumerate(pairs):
        try:
            flex = [pos_of[str(p)] for p in pids if str(p) != str(cpt_id)]
            cpt_pos = pos_of[str(cpt_id)]
        except KeyError:
            continue
        if len(flex) != 5:
            continue
        totals[r] = 1.5 * mat[:, cpt_pos] + mat[:, flex].sum(axis=1)
    return totals


def _lineup_pairs(lineups):
    """Normalize lineup references to [(cpt_id, ids_tuple)]."""
    pairs = []
    if lineups is None:
        return pairs
    if isinstance(lineups, pd.DataFrame):
        if "CPT_ID" not in lineups.columns:
            return pairs
        cols = ["CPT_ID"] + [f"FLEX{j}_ID" for j in range(1, 6)]
        for _, row in lineups.iterrows():
            ids = tuple(str(row[c]) for c in cols)
            pairs.append((ids[0], ids))
        return pairs
    for item in lineups:
        if isinstance(item, dict):
            cpt = str(item["cpt_id"])
            ids = tuple(str(p) for p in item["player_ids"])
        else:
            cpt, ids = str(item[0]), tuple(str(p) for p in item[1])
        if cpt not in ids:
            ids = (cpt,) + ids
        pairs.append((cpt, ids))
    return pairs


def showdown_lineup_world_metrics(worlds_mat, df, lineups):
    """World metrics per lineup, scored against the pool of lineups
    given (Best-of-pool % is pool-relative by definition — callers
    pass the full candidate pool for selection/rating, the delivered
    set + challengers for the audit).

    Columns: Sim Mean, Ceiling P90, Best-of-pool % (share of worlds
    where the lineup is the pool's highest scorer), Top-decile %
    (share of worlds where it lands in the pool's top decile that
    world). All vectorized; sub-second at candidate-pool scale."""
    pairs = _lineup_pairs(lineups)
    cols = ["Sim Mean", "Ceiling P90", "Best-of-pool %", "Top-decile %"]
    if not pairs:
        return pd.DataFrame(columns=cols)
    totals = showdown_lineup_totals(worlds_mat, df, pairs)
    if totals.size == 0 or totals.shape[1] == 0:
        return pd.DataFrame({c: [0.0] * len(pairs) for c in cols})
    best = totals.max(axis=0)
    decile = np.percentile(totals, 90, axis=0)
    return pd.DataFrame({
        "Sim Mean": np.round(totals.mean(axis=1), 2),
        "Ceiling P90": np.round(np.percentile(totals, 90, axis=1), 2),
        "Best-of-pool %": np.round(
            100.0 * (totals >= best[None, :] - 1e-9).mean(axis=1), 2),
        "Top-decile %": np.round(
            100.0 * (totals >= decile[None, :] - 1e-9).mean(axis=1), 2),
    })


def showdown_world_scores(metrics):
    """The selection preference score from a metrics frame (spec
    §1.3): a fixed composite of the two win terms — Best-of-pool %
    and Top-decile presence, percentile-ranked within the pool —
    with Sim Mean as the tie-break stratum, never the driver."""
    if metrics is None or len(metrics) == 0:
        return np.zeros(0)
    best = metrics["Best-of-pool %"].rank(pct=True).to_numpy(float)
    top = metrics["Top-decile %"].rank(pct=True).to_numpy(float)
    mean = metrics["Sim Mean"].rank(pct=True).to_numpy(float)
    return 0.5 * best + 0.5 * top + 0.01 * mean


def showdown_candidate_row(df, chosen, strategy_map, script, script_team,
                           game_world=None, exempt_teams=None):
    """Assemble one full candidate row (the generator's row shape:
    details + coherence + slot names/IDs) for a chosen lineup given
    as [(slot, df_position)] over a reset frame — the same assembly
    generate_showdown_lineups performs inline, exposed for the
    world-enrichment and audit paths. Returns None when coherence
    hard-rejects the construction."""
    d = df.reset_index(drop=True)
    chosen = [(slot, int(i)) for slot, i in chosen]
    coherence = showdown_lineup_coherence(
        d, chosen, game_world, exempt_teams=exempt_teams)
    if not coherence["Accept"]:
        return None
    detail = showdown_lineup_details(d, chosen, strategy_map, script,
                                     script_team, game_world)
    detail.update({
        "Coherence Score": coherence["Coherence Score"],
        "Lineup Story": coherence["Lineup Story"],
        "Coherence Flags": coherence["Coherence Flags"],
    })
    row = dict(detail)
    for slot, i in chosen:
        row[slot] = d.loc[i, "Name"]
        row[slot + "_ID"] = str(d.loc[i, "ID"])
        row[slot + "_NameID"] = (d.loc[i, "CPT_NameID"] if slot == "CPT"
                                 else d.loc[i, "FLEX_NameID"])
    return row
