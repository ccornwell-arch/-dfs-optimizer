
import csv
import math
import io
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

st.set_page_config(page_title="DFS LAB", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* ---------- Global ---------- */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}
html, body, [class*="css"]  {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 15% 0%, rgba(46, 204, 113, 0.08), transparent 28%),
        radial-gradient(circle at 100% 20%, rgba(0, 184, 255, 0.06), transparent 30%);
}
h1, h2, h3 {
    letter-spacing: -0.02em;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(17,24,39,0.98), rgba(15,23,42,0.98));
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * {
    color: #f8fafc;
}
[data-testid="stSidebar"] label {
    font-weight: 600;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.12) !important;
}

/* ---------- Hero ---------- */
.hero {
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 22px;
    padding: 22px 24px;
    margin-bottom: 18px;
    background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,41,59,0.92));
    box-shadow: 0 10px 30px rgba(0,0,0,0.10);
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0;
}
.hero-sub {
    color: #cbd5e1;
    margin-top: 4px;
}
.hero-chip {
    display: inline-block;
    margin-top: 12px;
    padding: 5px 10px;
    border-radius: 999px;
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.35);
    color: #86efac;
    font-size: 0.82rem;
    font-weight: 700;
}

/* ---------- Section cards ---------- */
.section-card {
    border: 1px solid rgba(148,163,184,0.20);
    border-radius: 18px;
    padding: 18px;
    background: rgba(255,255,255,0.82);
    box-shadow: 0 8px 22px rgba(15,23,42,0.05);
    margin-bottom: 14px;
}
@media (prefers-color-scheme: dark) {
    .section-card {
        background: rgba(15,23,42,0.55);
    }
}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    border: 1px solid rgba(148,163,184,0.20);
    border-radius: 16px;
    padding: 12px 14px;
    background: rgba(255,255,255,0.70);
}
[data-testid="stMetricValue"] {
    font-weight: 800;
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    padding: 0.65rem 1rem;
    border: 1px solid rgba(148,163,184,0.25);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #16a34a, #22c55e);
    border: 0;
    color: white;
}

/* High-contrast download buttons, including iPad Safari */
[data-testid="stDownloadButton"] > button {
    background: #2563eb !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
}
[data-testid="stDownloadButton"] > button * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* ---------- Dataframes ---------- */
[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(148,163,184,0.20);
}

/* ---------- Tabs ---------- */
button[data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    font-weight: 700;
}

/* ---------- Badges ---------- */
.badge {
    display:inline-block;
    padding:4px 8px;
    border-radius:999px;
    font-size:0.78rem;
    font-weight:800;
}
.badge-a {background:#dcfce7;color:#166534;}
.badge-b {background:#dbeafe;color:#1d4ed8;}
.badge-c {background:#fef3c7;color:#92400e;}
.badge-x {background:#fee2e2;color:#991b1b;}

/* ---------- Small helper ---------- */
.muted { color:#64748b; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* V2.6 iPad/sidebar readability fix */
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="input"] *,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] input {
    background: #ffffff !important;
    color: #111827 !important;
    border-color: rgba(148,163,184,0.35) !important;
}

/* Keep labels and section headings light on dark sidebar */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p {
    color: #f8fafc !important;
}

/* Slider/number value text */
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    color: #111827 !important;
    background: #ffffff !important;
}

/* Select dropdown selected value */
[data-testid="stSidebar"] [role="combobox"] {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}
</style>
""", unsafe_allow_html=True)

ROSTER_SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
PRIORITY_OPTIONS = ["Core", "Like", "Neutral", "Fade", "Exclude"]
PRIORITY_BONUS = {"Core": 2.8, "Like": 1.35, "Neutral": 0.0, "Fade": -1.5, "Exclude": -100.0}
TEAM_PRIORITY_BONUS = {"Core": 1.8, "Like": 0.9, "Neutral": 0.0, "Fade": -0.8, "Exclude": -100.0}

# -----------------------------
# File loading
# -----------------------------

def load_dk_template(uploaded_file):
    raw = uploaded_file.getvalue().decode("utf-8-sig", errors="ignore").splitlines()
    reader = csv.reader(raw)
    header = None
    rows = []
    for row in reader:
        if "Position" in row and "Name + ID" in row and "Salary" in row:
            header = row
            break
    if header is None:
        raise ValueError("Could not locate DraftKings player header.")

    idx = {name: i for i, name in enumerate(header)}
    for row in reader:
        if not row or len(row) <= max(idx.values()):
            continue
        name = row[idx["Name"]].strip()
        if not name:
            continue
        try:
            salary = int(float(row[idx["Salary"]]))
        except Exception:
            continue
        rows.append({
            "Position": row[idx["Position"]].strip(),
            "Name + ID": row[idx["Name + ID"]].strip(),
            "Name": name,
            "ID": str(row[idx["ID"]].strip()),
            "Roster Position": row[idx["Roster Position"]].strip(),
            "Salary": salary,
            "Game Info": row[idx["Game Info"]].strip(),
            "Team": row[idx["TeamAbbrev"]].strip(),
            "AvgPointsPerGame": float(row[idx["AvgPointsPerGame"]] or 0),
        })
    return pd.DataFrame(rows)


def parse_matchup(game_info):
    matchup = str(game_info).split()[0].strip()
    if "@" not in matchup:
        return "", "", matchup
    away, home = matchup.split("@", 1)
    return away.strip(), home.strip(), matchup


def prepare_player_pool(dk_file, ss_file):
    dk = load_dk_template(dk_file)
    ss = pd.read_csv(ss_file)

    needed = {"Name", "My Proj", "My Own"}
    missing = needed - set(ss.columns)
    if missing:
        raise ValueError(f"SaberSim file is missing columns: {sorted(missing)}")

    ss = ss[["Name", "My Proj", "My Own"]].copy()
    ss["My Proj"] = pd.to_numeric(ss["My Proj"], errors="coerce").fillna(0.0)
    ss["My Own"] = pd.to_numeric(ss["My Own"], errors="coerce").fillna(0.0)

    df = dk.merge(ss, on="Name", how="left")
    df["My Proj"] = df["My Proj"].fillna(0.0)
    df["My Own"] = df["My Own"].fillna(0.0)

    away, home, matchup = [], [], []
    for g in df["Game Info"]:
        a, h, m = parse_matchup(g)
        away.append(a); home.append(h); matchup.append(m)
    df["Away"] = away
    df["Home"] = home
    df["Matchup"] = matchup
    df["Opponent"] = np.where(df["Team"] == df["Away"], df["Home"], df["Away"])

    df["ActiveForBuild"] = (df["My Proj"] > 0.05) & (df["Salary"] > 0)
    df["is_QB"] = df["Roster Position"].str.contains(r"\bQB\b", regex=True)
    df["is_RB"] = df["Roster Position"].str.contains(r"\bRB\b", regex=True)
    df["is_WR"] = df["Roster Position"].str.contains(r"\bWR\b", regex=True)
    df["is_TE"] = df["Roster Position"].str.contains(r"\bTE\b", regex=True)
    df["is_DST"] = df["Position"].eq("DST")
    df["is_FLEX"] = df["Roster Position"].str.contains(r"\bFLEX\b", regex=True)

    return df.reset_index(drop=True)

# -----------------------------
# Contest / rating logic
# -----------------------------

def contest_aggression(field_size, payout_style):
    fs = max(2, int(field_size))
    raw = (math.log10(fs) - 2.0) / 4.0
    raw = min(1.0, max(0.08, raw))
    if payout_style == "Winner take all":
        raw = min(1.0, raw + 0.14)
    elif payout_style == "Flatter payouts":
        raw = max(0.04, raw - 0.12)
    return raw


def percentile_label(x, series):
    if len(series) < 2:
        return "Average"
    pct = float((series <= x).mean())
    if pct >= 0.90: return "Excellent"
    if pct >= 0.70: return "Very good"
    if pct >= 0.45: return "Good"
    if pct >= 0.25: return "Average"
    return "Weak"


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
    lev_pct = (-out["Avg Own"]).rank(pct=True)
    unique_pct = out["Salary Left"].rank(pct=True)

    # Contest-aware weights.
    w_proj = 0.46 - 0.12 * aggr
    w_corr = 0.24 + 0.06 * aggr
    w_fit = 0.22
    w_lev = 0.08 + 0.12 * aggr
    w_unique = 0.00 + 0.04 * aggr
    total = w_proj + w_corr + w_fit + w_lev + w_unique

    composite = (
        w_proj * proj_pct
        + w_corr * corr_pct
        + w_fit * fit_pct
        + w_lev * lev_pct
        + w_unique * unique_pct
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
    out["Leverage Grade"] = [percentile_label(-x, -out["Avg Own"]) for x in out["Avg Own"]]

    # If every lineup has the same user-fit score, don't misleadingly label them all Excellent.
    if out["User Fit Raw"].nunique() <= 1:
        out["User Fit Grade"] = "Neutral"
    else:
        out["User Fit Grade"] = [percentile_label(x, out["User Fit Raw"]) for x in out["User Fit Raw"]]

    return out

# -----------------------------
# Optimizer
# -----------------------------

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
    proj_col = "DFS Lab Proj" if "DFS Lab Proj" in df.columns else ("Script Proj" if "Script Proj" in df.columns else "My Proj")
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
    noise_scale=0.16
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

    # Build variable bounds / objective.
    for i in range(n):
        player_id = str(df.loc[i, "ID"])
        strat = strategy_map.get(player_id, {})
        excluded = bool(strat.get("Exclude", False)) or strat.get("Priority") == "Exclude"
        locked = bool(strat.get("Lock", False)) and not excluded

        for j, slot in enumerate(ROSTER_SLOTS):
            c[vidx(i, j)] = -randomized[i]
            if (not active[i]) or excluded or (not elig[slot][i]):
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
    for q in df.index[df["is_QB"] & df["ActiveForBuild"]]:
        team = df.loc[q, "Team"]
        opp = df.loc[q, "Opponent"]

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
                coeff[vidx(q, j)] = coeff.get(vidx(q, j), 0.0) - float(qb_stack_min)
        rows.append(coeff); lows.append(0.0); highs.append(np.inf)

        if bringback_mode == "Required":
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

    A = lil_matrix((len(rows), total_vars), dtype=float)
    for r, coeff in enumerate(rows):
        for col, val in coeff.items():
            A[r, col] = val

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(A.tocsr(), np.array(lows), np.array(highs)),
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


def lineup_details(df, chosen, strategy_map, preferred_stack_teams):
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

    pass_catchers = p[(p["Team"] == qb_team) & (p["is_WR"] | p["is_TE"])]
    bringbacks = p[(p["Team"] == opp) & (p["is_RB"] | p["is_WR"] | p["is_TE"])]

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

    correlation_raw = (
        2.0 * min(len(pass_catchers), 2)
        + 0.9 * min(len(bringbacks), 1)
        + 0.5 * rb_dst
        + 0.35 * max(0, qb_game_players - 2)
    )

    stack_names = " + ".join(pass_catchers["Name"].tolist()) if len(pass_catchers) else "none"
    bb_names = " + ".join(bringbacks["Name"].tolist()) if len(bringbacks) else "none"
    stack_summary = f"{qb_team}: {qb['Name']} + {stack_names} | {opp} bring-back: {bb_names}"

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
        "Correlation Raw": round(correlation_raw, 2),
        "User Fit Raw": round(fit, 2),
        "Stack Summary": stack_summary,
        "Strategy Notes": ", ".join(fit_notes) if fit_notes else "Neutral build",
    }


def generate_lineups(
    df, field_size, payout_style, count, attempts, min_salary,
    qb_stack_min, bringback_mode, preferred_stack_teams,
    strategy_map, team_strategy_map, no_dst_from_qb_game, no_offense_vs_dst, seed
):
    aggr = contest_aggression(field_size, payout_style)
    rng = np.random.default_rng(seed)
    seen = set()
    rows = []
    exposure_counts = defaultdict(int)

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
            noise_scale=0.13 + 0.11 * aggr
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

        seen.add(ids)
        detail = lineup_details(df, chosen, strategy_map, preferred_stack_teams)
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
            "Salary": int(p["Salary"]),
            "Proj": round(float(p["My Proj"]), 2),
            "Proj Own": round(float(p["My Own"]), 1),
            "Actual Exp %": round(100.0 * count / n, 1),
            "Lineups": count,
            "Min Target %": float(strat.get("Min Exposure", 0)),
            "Max Target %": float(strat.get("Max Exposure", 100)),
        })
    return pd.DataFrame(rows).sort_values(["Actual Exp %", "Proj"], ascending=[False, False]).reset_index(drop=True)


# -----------------------------


# -----------------------------
# Showdown engine
# -----------------------------
SHOWDOWN_SLOTS = ["CPT", "FLEX1", "FLEX2", "FLEX3", "FLEX4", "FLEX5"]


def _first_existing(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


def load_showdown_dk_template(uploaded_file):
    """Load a DraftKings Showdown CSV/template and preserve CPT/FLEX identifiers when present."""
    raw = uploaded_file.getvalue().decode("utf-8-sig", errors="ignore").splitlines()
    reader = csv.reader(raw)
    header = None
    for row in reader:
        if "Position" in row and "Name" in row and "Salary" in row:
            header = row
            break
    if header is None:
        raise ValueError("Could not locate the DraftKings player header.")

    idx = {name: i for i, name in enumerate(header)}
    required = ["Position", "Name", "Salary", "TeamAbbrev"]
    for c in required:
        if c not in idx:
            raise ValueError(f"DraftKings file is missing {c}.")

    rows = []
    for row in reader:
        if not row or len(row) <= max(idx.values()):
            continue
        name = row[idx["Name"]].strip()
        if not name:
            continue
        try:
            salary = int(float(row[idx["Salary"]]))
        except Exception:
            continue
        rp = row[idx.get("Roster Position", idx["Position"])].strip() if ("Roster Position" in idx or "Position" in idx) else ""
        pid = row[idx["ID"]].strip() if "ID" in idx else name
        name_id = row[idx["Name + ID"]].strip() if "Name + ID" in idx else name
        gi = row[idx["Game Info"]].strip() if "Game Info" in idx else ""
        avg = 0.0
        if "AvgPointsPerGame" in idx:
            try:
                avg = float(row[idx["AvgPointsPerGame"]] or 0)
            except Exception:
                avg = 0.0
        rows.append({
            "Position": row[idx["Position"]].strip(),
            "Name": name,
            "RawID": str(pid),
            "Name + ID": name_id,
            "Roster Position": rp,
            "RawSalary": salary,
            "Game Info": gi,
            "Team": row[idx["TeamAbbrev"]].strip(),
            "AvgPointsPerGame": avg,
        })
    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        raise ValueError("No Showdown players were found in the DraftKings file.")

    # DK files vary: some expose CPT/FLEX as separate rows, others expose one row with CPT/FLEX eligibility.
    # Collapse to one player while retaining the exact identifier for each roster position when possible.
    collapsed = []
    for (name, team), g in raw_df.groupby(["Name", "Team"], sort=False):
        g = g.copy()
        cpt_rows = g[g["Roster Position"].str.contains("CPT", case=False, na=False)]
        flex_rows = g[g["Roster Position"].str.contains("FLEX", case=False, na=False)]

        # If CPT-specific row has the larger salary, use it. FLEX base salary is the smallest observed salary.
        flex_row = (flex_rows.sort_values("RawSalary").iloc[0] if not flex_rows.empty else g.sort_values("RawSalary").iloc[0])
        cpt_row = (cpt_rows.sort_values("RawSalary", ascending=False).iloc[0] if not cpt_rows.empty else None)
        flex_salary = int(g["RawSalary"].min())
        cpt_salary = int(cpt_row["RawSalary"]) if cpt_row is not None and int(cpt_row["RawSalary"]) > flex_salary else int(round(flex_salary * 1.5))

        collapsed.append({
            "Position": str(flex_row["Position"]),
            "Name": name,
            "ID": str(flex_row["RawID"]),
            "FLEX_ID": str(flex_row["RawID"]),
            "FLEX_NameID": str(flex_row["Name + ID"]),
            "CPT_ID": str(cpt_row["RawID"]) if cpt_row is not None else str(flex_row["RawID"]),
            "CPT_NameID": str(cpt_row["Name + ID"]) if cpt_row is not None else str(flex_row["Name + ID"]),
            "FlexSalary": flex_salary,
            "CaptainSalary": cpt_salary,
            "Game Info": str(flex_row["Game Info"]),
            "Team": team,
            "AvgPointsPerGame": float(flex_row["AvgPointsPerGame"]),
        })
    return pd.DataFrame(collapsed).reset_index(drop=True)



def _dk_fantasy_points_from_stats(stats):
    """DraftKings-style fantasy points from nflverse weekly player stats."""
    def col(name):
        return pd.to_numeric(stats[name], errors="coerce").fillna(0.0) if name in stats.columns else pd.Series(0.0, index=stats.index)
    pts = (col("passing_yards") * 0.04 + col("passing_tds") * 4 - col("interceptions")
           + col("rushing_yards") * 0.10 + col("rushing_tds") * 6
           + col("receptions") * 1.0 + col("receiving_yards") * 0.10 + col("receiving_tds") * 6
           - col("rushing_fumbles_lost") - col("receiving_fumbles_lost") - col("sack_fumbles_lost"))
    # DK 300-yard passing and 100-yard rushing/receiving bonuses.
    pts += (col("passing_yards") >= 300).astype(float) * 3
    pts += (col("rushing_yards") >= 100).astype(float) * 3
    pts += (col("receiving_yards") >= 100).astype(float) * 3
    return pts

@st.cache_data(ttl=21600, show_spinner=False)
def _load_nflverse_projection_inputs(season):
    """Load a multi-year evidence window. Current season is included, but never allowed to dominate early."""
    import nflreadpy as nfl
    seasons=[int(season)-3,int(season)-2,int(season)-1,int(season)]
    stats=nfl.load_player_stats(seasons, summary_level="week").to_pandas()
    return stats

def _name_col(df):
    return _first_existing(df.columns,["player_display_name","player_name","Name"])

def _opp_col(df):
    return _first_existing(df.columns,["opponent_team","opponent","opp_team","Opp"])

def _season_col(df):
    return _first_existing(df.columns,["season","Season"])

def _role_points_from_stats(x):
    """Opportunity signal: volume moves faster than TD-driven fantasy scoring."""
    def c(n):
        return pd.to_numeric(x[n],errors="coerce").fillna(0.0) if n in x.columns else pd.Series(0.0,index=x.index)
    return c("carries")*0.52 + c("targets")*0.78 + c("receptions")*0.18 + c("passing_attempts")*0.08

def dfs_lab_projection_engine(dk):
    """V6.1 independent projection baseline.

    Uses four seasons of nflverse weekly evidence, sample-size shrinkage, opportunity/role,
    and opponent-vs-position history when the source exposes opponent_team. DK AvgPointsPerGame
    is only a weak fallback/prior. No SaberSim projection is used.
    """
    out=dk.copy(); season=2026
    try:
        import re
        m=re.search(r"(20\d{2})", " ".join(out["Game Info"].astype(str).tolist()))
        if m: season=int(m.group(1))
    except Exception: pass
    out["DFS Lab Data"]="DK prior fallback"
    out["DFS Lab Base Proj"]=pd.to_numeric(out.get("AvgPointsPerGame",0),errors="coerce").fillna(0.0)
    out["History Games"]=0; out["Current Games"]=0; out["Role Signal"]=0.0; out["Matchup Adj %"]=0.0
    out["Projection Why"]="DK slate prior fallback"
    try:
        stx=_load_nflverse_projection_inputs(season).copy()
        nc=_name_col(stx); sc=_season_col(stx); oc=_opp_col(stx)
        if not nc or not sc: raise ValueError("nflverse player name/season columns unavailable")
        stx["Name"]=stx[nc].astype(str).str.strip(); stx["Season"]=pd.to_numeric(stx[sc],errors="coerce")
        stx["_fp"]=_dk_fantasy_points_from_stats(stx); stx["_role"]=_role_points_from_stats(stx)
        # Ignore placeholder rows with no statistical activity.
        activity=[]
        for c in ["passing_attempts","carries","targets","receptions","field_goals_made","extra_points_made"]:
            if c in stx.columns: activity.append(pd.to_numeric(stx[c],errors="coerce").fillna(0.0))
        if activity:
            active=sum(activity)>0
            stx=stx[active | (stx["_fp"].abs()>0)].copy()

        # Per-player per-season summaries. Four-year weights favor recency without letting one game take over.
        ss=stx.groupby(["Name","Season"],as_index=False).agg(FPPG=("_fp","mean"),Role=("_role","mean"),Games=("_fp","size"))
        season_weights={season:0.34,season-1:0.38,season-2:0.19,season-3:0.09}
        rows=[]
        for name,g in ss.groupby("Name"):
            hist_num=hist_den=role_num=role_den=0.0; hist_games=cur_games=0
            for _,r in g.iterrows():
                yr=int(r["Season"]); games=int(r["Games"]); w=season_weights.get(yr,0.0)
                if w<=0: continue
                # Current-year reliability ramps from 20% after one game toward full weight after eight.
                reliability=min(1.0,max(0.20,games/8.0)) if yr==season else min(1.0,games/8.0)
                ew=w*reliability
                hist_num += ew*float(r["FPPG"]); hist_den += ew
                role_num += ew*float(r["Role"]); role_den += ew
                hist_games += games
                if yr==season: cur_games=games
            rows.append({"Name":name,"HistProj":hist_num/hist_den if hist_den else 0.0,"RoleSignal":role_num/role_den if role_den else 0.0,"HistoryGames":hist_games,"CurrentGames":cur_games})
        ps=pd.DataFrame(rows)
        out=out.merge(ps,on="Name",how="left")
        for c in ["HistProj","RoleSignal","HistoryGames","CurrentGames"]: out[c]=pd.to_numeric(out[c],errors="coerce").fillna(0.0)

        # Opponent-vs-position fantasy allowance: three prior seasons + current season, shrunk toward neutral.
        matchup={}
        if oc:
            stx["Opp"]=stx[oc].astype(str).str.strip()
            posc=_first_existing(stx.columns,["position","position_group","Pos"])
            if posc:
                stx["Pos"]=stx[posc].astype(str).str.upper().replace({"HB":"RB","FB":"RB"})
                allowed=stx[stx["Pos"].isin(["QB","RB","WR","TE"])].groupby(["Opp","Pos"],as_index=False).agg(Allowed=("_fp","mean"),N=("_fp","size"))
                league=stx[stx["Pos"].isin(["QB","RB","WR","TE"])].groupby("Pos")["_fp"].mean().to_dict()
                for _,r in allowed.iterrows():
                    base=max(float(league.get(r["Pos"],0)),1.0); n=float(r["N"])
                    raw=float(r["Allowed"])/base-1.0; shrink=n/(n+24.0)
                    matchup[(str(r["Opp"]),str(r["Pos"]))]=float(np.clip(raw*shrink,-0.10,0.10))

        dkavg=pd.to_numeric(out["AvgPointsPerGame"],errors="coerce").fillna(0.0)
        sal=pd.to_numeric(out["FlexSalary"],errors="coerce").fillna(0.0); pos=out["Position"].astype(str).str.upper()
        vals=[]; reasons=[]; madjs=[]
        for _,r in out.iterrows():
            hist=float(r.get("HistProj",0)); games=int(r.get("HistoryGames",0)); curg=int(r.get("CurrentGames",0)); prior=float(r.get("AvgPointsPerGame",0) or 0)
            # Historical evidence dominates established players; DK average is a weak stabilizer/fallback.
            evidence=min(0.88, games/(games+8.0))
            base=(evidence*hist + (1-evidence)*prior) if hist>0 else prior
            # Role signal is used only as a modest stabilizer, not converted directly to fantasy points.
            role=float(r.get("RoleSignal",0)); role_adj=0.0
            if role>0 and base>0:
                # Keeps TD spikes from dominating while rewarding sustained opportunity.
                role_adj=float(np.clip((role/12.0)-0.5,-0.04,0.05))
            opp=""
            try:
                teams=[t for t in out["Team"].dropna().unique().tolist() if t]
                if len(teams)==2: opp=teams[1] if r["Team"]==teams[0] else teams[0]
            except Exception: pass
            m=float(matchup.get((str(opp),str(r["Position"]).upper()),0.0)); madjs.append(m*100)
            # Matchup and role are bounded; they refine the baseline rather than rewrite it.
            model=max(0.0,base*(1.0+role_adj+m))
            # Salary prior only for thin-history skill players.
            if games<5 and str(r["Position"]).upper() in ["QB","RB","WR","TE"]:
                sp=max(0.3,float(r["FlexSalary"])/1000.0*1.55)
                model=0.90*model+0.10*sp
            vals.append(round(model,3))
            reasons.append(f"4-year history {hist:.2f} over {games} games; current season {curg} game(s) is sample-shrunk; DK prior {prior:.2f}; role {role_adj*100:+.1f}%; {opp or 'opponent'} matchup {m*100:+.1f}%")
        out["DFS Lab Base Proj"]=vals; out["History Games"]=out["HistoryGames"].astype(int); out["Current Games"]=out["CurrentGames"].astype(int)
        out["Role Signal"]=out["RoleSignal"].round(2); out["Matchup Adj %"]=np.round(madjs,1); out["Projection Why"]=reasons
        out["DFS Lab Data"]="2023-2026 history + role + matchup"
    except Exception as e:
        out.attrs["projection_warning"]=f"Live nflverse evidence could not load ({e}). DFS Lab used the DK slate prior for this run."
    return out

def apply_projection_overrides(df, override_map=None):
    out=df.copy(); override_map=override_map or {}
    out["Model Proj"]=pd.to_numeric(out.get("DFS Lab Proj",out.get("My Proj",0)),errors="coerce").fillna(0.0)
    finals=[]; flags=[]
    for _,r in out.iterrows():
        val=override_map.get(str(r["ID"]),None)
        if val is None or float(val)<0: finals.append(float(r["Model Proj"])); flags.append(False)
        else: finals.append(float(val)); flags.append(True)
    out["DFS Lab Proj"]=np.array(finals).round(3); out["Projection Override"]=flags
    return out

def prepare_showdown_pool(dk_file, ss_file=None):
    """Create the Showdown pool. DFS Lab projections work with DK alone; SaberSim is optional comparison data."""
    dk=load_showdown_dk_template(dk_file)
    dk=dk.drop_duplicates(subset=["ID"],keep="first").drop_duplicates(subset=["Name","Team"],keep="first")
    df=dfs_lab_projection_engine(dk)
    df["SaberSim Proj"]=np.nan; df["My Own"]=0.0; df["CPT Own"]=0.0; df["CPT Own Estimated"]=True
    if ss_file is not None:
        ss_raw=pd.read_csv(ss_file)
        name_col=_first_existing(ss_raw.columns,["Name","Player","Player Name"]); proj_col=_first_existing(ss_raw.columns,["My Proj","Projection","Proj"])
        own_col=_first_existing(ss_raw.columns,["My Own","Ownership","Own","Projected Ownership"])
        roster_col=_first_existing(ss_raw.columns,["Roster Position","Roster Pos","Slot","Lineup Position","Position Type"])
        if name_col and proj_col:
            x=ss_raw.copy(); x["Name"]=x[name_col].astype(str).str.strip(); x["_proj"]=pd.to_numeric(x[proj_col],errors="coerce").fillna(0.0)
            x["_own"]=pd.to_numeric(x[own_col],errors="coerce").fillna(0.0) if own_col else 0.0
            rows=[]
            for name,g in x.groupby("Name",sort=False):
                g=g.copy()
                flex=None; cpt=None
                if roster_col:
                    slot=g[roster_col].astype(str).str.upper(); fg=g[slot.str.contains("FLEX",na=False)]; cg=g[slot.str.contains("CPT|CAPTAIN",regex=True,na=False)]
                    if not fg.empty:flex=fg.iloc[0]
                    if not cg.empty:cpt=cg.iloc[0]
                if flex is None: flex=g.sort_values("_proj",ascending=True).iloc[0]
                if cpt is None and len(g)>1: cpt=g.sort_values("_proj",ascending=False).iloc[0]
                rows.append({"Name":name,"SaberSim Proj":float(flex["_proj"]),"My Own":float(flex["_own"]),"CPT Own":float(cpt["_own"]) if cpt is not None else max(.1,float(flex["_own"])*.18),"CPT Own Estimated":cpt is None})
            ss=pd.DataFrame(rows).drop_duplicates("Name")
            base_cols=[c for c in df.columns if c not in ["SaberSim Proj","My Own","CPT Own","CPT Own Estimated"]]
            df=df[base_cols].merge(ss,on="Name",how="left")
            for c in ["SaberSim Proj","My Own","CPT Own"]: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0)
            df["CPT Own Estimated"]=df["CPT Own Estimated"].fillna(True).astype(bool)
    # Compatibility: My Proj is now DFS Lab's independent baseline, not SaberSim.
    df["My Proj"]=pd.to_numeric(df["DFS Lab Base Proj"],errors="coerce").fillna(0.0)
    away=[];home=[];matchup=[]
    for g in df["Game Info"]:
        a,h,m=parse_matchup(g);away.append(a);home.append(h);matchup.append(m)
    df["Away"]=away;df["Home"]=home;df["Matchup"]=matchup
    teams=[t for t in df["Team"].dropna().unique().tolist() if t]
    if len(teams)==2:
        opp={teams[0]:teams[1],teams[1]:teams[0]};df["Opponent"]=df["Team"].map(opp).fillna("")
    else: df["Opponent"]=np.where(df["Team"]==df["Away"],df["Home"],df["Away"])
    pos=df["Position"].astype(str).str.upper();df["is_QB"]=pos.eq("QB");df["is_RB"]=pos.eq("RB");df["is_WR"]=pos.eq("WR");df["is_TE"]=pos.eq("TE");df["is_DST"]=pos.isin(["DST","D/ST"]);df["is_K"]=pos.isin(["K","PK"]);df["is_passcatcher"]=df["is_WR"]|df["is_TE"]
    df["ActiveForBuild"]=(df["My Proj"]>0.01)&(df["FlexSalary"]>0)
    return df.reset_index(drop=True)

def showdown_aggression(field_size, payout_style, entry_format="Single Entry"):
    aggr = contest_aggression(field_size, payout_style)
    entry_bump={"Single Entry":-0.14,"3-Max":-0.05,"20-Max":0.05,"150-Max":0.14}.get(entry_format,0.0)
    return float(np.clip(aggr + 0.08 + entry_bump, 0.05, 1.0))




# --- V6.3 Game Worlds -------------------------------------------------------
GAME_WORLDS = {
    "Balanced shootout": {"family":"shootout", "desc":"Both offenses succeed; scoring is spread across primary pieces."},
    "BUF passing ceiling": {"family":"pass_ceiling", "team":"BUF", "desc":"Buffalo scoring concentrates through its quarterback and pass catchers."},
    "DET passing ceiling": {"family":"pass_ceiling", "team":"DET", "desc":"Detroit scoring concentrates through its quarterback and pass catchers."},
    "Allen rushing ceiling": {"family":"qb_rush", "team":"BUF", "player":"Josh Allen", "desc":"Josh Allen captures an outsized share of Buffalo touchdowns with rushing production."},
    "DET RB-led win": {"family":"rb_control", "team":"DET", "desc":"Detroit controls enough of the game for its running backs and control pieces to matter."},
    "BUF RB-led win": {"family":"rb_control", "team":"BUF", "desc":"Buffalo controls enough of the game for its running backs and control pieces to matter."},
    "BUF leads / DET comeback": {"family":"comeback", "lead":"BUF", "trail":"DET", "desc":"Buffalo scores first and Detroit answers with elevated second-half passing volume."},
    "DET leads / BUF comeback": {"family":"comeback", "lead":"DET", "trail":"BUF", "desc":"Detroit scores first and Buffalo answers with elevated passing volume."},
    "Low-scoring upset path": {"family":"low", "desc":"Scoring disappoints; kickers, defenses and concentrated touchdown paths gain importance."},
}

def contest_world_weights(entry_format, field_size, script="Neutral"):
    # Probabilities are portfolio exploration weights, not claims about real-world game odds.
    if entry_format == "Single Entry":
        w={"Balanced shootout":36,"BUF passing ceiling":15,"DET passing ceiling":15,"Allen rushing ceiling":8,"DET RB-led win":7,"BUF RB-led win":7,"BUF leads / DET comeback":5,"DET leads / BUF comeback":5,"Low-scoring upset path":2}
    elif entry_format == "3-Max":
        w={"Balanced shootout":27,"BUF passing ceiling":16,"DET passing ceiling":16,"Allen rushing ceiling":10,"DET RB-led win":8,"BUF RB-led win":8,"BUF leads / DET comeback":6,"DET leads / BUF comeback":6,"Low-scoring upset path":3}
    elif entry_format == "20-Max":
        w={"Balanced shootout":20,"BUF passing ceiling":16,"DET passing ceiling":16,"Allen rushing ceiling":11,"DET RB-led win":9,"BUF RB-led win":9,"BUF leads / DET comeback":8,"DET leads / BUF comeback":8,"Low-scoring upset path":3}
    else:
        w={"Balanced shootout":13,"BUF passing ceiling":15,"DET passing ceiling":15,"Allen rushing ceiling":12,"DET RB-led win":10,"BUF RB-led win":10,"BUF leads / DET comeback":10,"DET leads / BUF comeback":10,"Low-scoring upset path":5}
    if script in ["Shootout","Pass-heavy shootout"]:
        for k in ["Balanced shootout","BUF passing ceiling","DET passing ceiling","BUF leads / DET comeback","DET leads / BUF comeback"]: w[k]*=1.35
        w["Low-scoring upset path"]*=0.35
    elif script in ["Low-scoring game","Defensive / field-goal battle","Ground-and-pound"]:
        w["Low-scoring upset path"]*=3.0; w["DET RB-led win"]*=1.35; w["BUF RB-led win"]*=1.35; w["Balanced shootout"]*=0.45
    # Very large fields explore tails a little more.
    if field_size >= 50000 and entry_format == "150-Max":
        w["Balanced shootout"]*=0.8; w["Low-scoring upset path"]*=1.35; w["Allen rushing ceiling"]*=1.15
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
    elif fam=="qb_rush":
        if str(row["Name"])==cfg.get("player"): b+=4.2
        elif team==cfg.get("team") and row["is_passcatcher"]: b+=0.7
        elif team==cfg.get("team") and row["is_RB"]: b-=0.8
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
    elif fam in ["pass_ceiling","qb_rush"]: desired={"3-3":42,"4-2":50,"5-1":8}
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

CONTEXT_FACTOR_WEIGHTS = {
    "Defense": 0.045,
    "Usage": 0.050,
    "Home/Rest": 0.020,
    "Travel": 0.015,
    "Time/Split": 0.010,
}
CONTEXT_RATING_LABELS = {
    -3: "Strong negative", -2: "Negative", -1: "Slight negative", 0: "Neutral",
    1: "Slight positive", 2: "Positive", 3: "Strong positive",
}

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
        cpt_ok = bool(strat.get("CPT Eligible", True)) and not excluded
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
        # QB captain -> require same-team pass catchers.
        if r["is_QB"] and cpt_qb_passcatchers > 0:
            pcs=df.index[df["ActiveForBuild"] & df["Team"].eq(r["Team"]) & df["is_passcatcher"]].tolist()
            coeff={}
            for i in pcs:
                for j in range(1,s): coeff[vidx(i,j)] = coeff.get(vidx(i,j),0)+1
            coeff[vidx(cpt,0)] = coeff.get(vidx(cpt,0),0)-float(cpt_qb_passcatchers)
            _add_constraint(rows,lows,highs,coeff,0,np.inf)

        # WR/TE captain -> pair same-team QB in chosen percentage of solves.
        if r["is_passcatcher"] and rng.random() < wrte_cpt_qb_pair_pct/100.0:
            qbs=df.index[df["ActiveForBuild"] & df["Team"].eq(r["Team"]) & df["is_QB"]].tolist()
            if qbs:
                coeff={}
                for q in qbs:
                    for j in range(1,s): coeff[vidx(q,j)] = coeff.get(vidx(q,j),0)+1
                coeff[vidx(cpt,0)] = coeff.get(vidx(cpt,0),0)-1
                _add_constraint(rows,lows,highs,coeff,0,np.inf)

        # RB captain -> same team DST or K in chosen percentage of solves.
        if r["is_RB"] and rng.random() < rb_cpt_dst_k_pct/100.0:
            partners=df.index[df["ActiveForBuild"] & df["Team"].eq(r["Team"]) & (df["is_DST"]|df["is_K"])].tolist()
            if partners:
                coeff={}
                for p in partners:
                    for j in range(1,s): coeff[vidx(p,j)] = coeff.get(vidx(p,j),0)+1
                coeff[vidx(cpt,0)] = coeff.get(vidx(cpt,0),0)-1
                _add_constraint(rows,lows,highs,coeff,0,np.inf)

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
    result=milp(c=c, integrality=integrality, bounds=Bounds(lb,ub),
                constraints=LinearConstraint(A.tocsr(),np.array(lows),np.array(highs)),
                options={"time_limit":8.0})
    if not result.success or result.x is None: return None
    chosen=[]
    for j,slot in enumerate(SHOWDOWN_SLOTS):
        vals=[(result.x[vidx(i,j)],i) for i in range(n)]
        _,i=max(vals); chosen.append((slot,int(i)))
    return chosen


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
    w_proj=0.34-0.05*aggr; w_cpt=.18; w_corr=.20; w_lev=(.11+.04*aggr) if ownership_available else 0.0; w_dup=(.10+.05*aggr) if ownership_available else 0.0; w_fit=.07
    comp=(w_proj*proj+w_cpt*captain+w_corr*corr+w_lev*lev+w_dup*dup+w_fit*fit)/(w_proj+w_cpt+w_corr+w_lev+w_dup+w_fit)
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
                              max_k, max_dst, min_unique, seed, relationship_rules=None, world_influence=50):
    aggr=showdown_aggression(field_size,payout_style,entry_format); rng=np.random.default_rng(seed)
    teams=[t for t in df["Team"].dropna().unique().tolist() if t]
    rows=[]; exposure=defaultdict(int); cpt_exp=defaultdict(int); previous=[]; seen=set()
    prog=st.progress(0,text="Building Showdown lineups...")
    for attempt in range(attempts):
        if len(rows)>=count: break
        game_world=choose_game_world(rng,entry_format,field_size,script)
        world_weights=world_construction_weights(construction_weights,game_world,entry_format)
        target=choose_construction_target(rng,teams,world_weights,script,script_team)
        # Exact minimum-exposure scheduling: only force a minimum when all remaining
        # accepted lineups are needed to reach it. Existing objective steering tries
        # to satisfy the target earlier and avoid a last-lineup pileup.
        remaining=count-len(rows)
        forced_overall=[]; forced_cpt=[]
        for pid,strat in strategy_map.items():
            need=max(0, int(math.ceil(float(strat.get("Min Exposure",0))*count/100.0))-exposure[pid])
            cneed=max(0, int(math.ceil(float(strat.get("CPT Min",0))*count/100.0))-cpt_exp[pid])
            if need>=remaining and need>0: forced_overall.append(pid)
            if cneed>=remaining and cneed>0: forced_cpt.append(pid)
        chosen=solve_showdown_one(df,aggr,rng,strategy_map,min_salary,max_salary,target,script,script_team,
                                  cpt_qb_passcatchers,wrte_cpt_qb_pair_pct,rb_cpt_dst_k_pct,max_k,max_dst,
                                  min_unique,previous,exposure,cpt_exp,len(rows),noise_scale=.14+.13*aggr,
                                  relationship_rules=relationship_rules,forced_overall_ids=forced_overall,forced_cpt_ids=forced_cpt,game_world=game_world,world_influence=world_influence)
        if not chosen: continue
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
        seen.add(key)
        detail=showdown_lineup_details(df,chosen,strategy_map,script,script_team,game_world)
        row=dict(detail)
        for slot,i in chosen:
            row[slot]=df.loc[i,"Name"]; row[slot+"_ID"]=str(df.loc[i,"ID"])
            row[slot+"_NameID"] = df.loc[i,"CPT_NameID"] if slot=="CPT" else df.loc[i,"FLEX_NameID"]
        rows.append(row); previous.append([i for _,i in chosen])
        for slot,i in chosen:
            pid=str(df.loc[i,"ID"]); exposure[pid]+=1
            if slot=="CPT": cpt_exp[pid]+=1
        if attempt%10==0: prog.progress(min(1.0,len(rows)/max(1,count)),text=f"Built {len(rows)} / {count}")
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


def showdown_upload_csv(result):
    cols=["CPT","FLEX1","FLEX2","FLEX3","FLEX4","FLEX5"]
    rows=[]
    for _,r in result.iterrows():
        rows.append([r.get(c+"_NameID",r[c]) for c in cols])
    return pd.DataFrame(rows,columns=["CPT","FLEX","FLEX","FLEX","FLEX","FLEX"]).to_csv(index=False)


# -----------------------------
# V4 Apple-style UI
# -----------------------------
st.markdown("""
<style>
:root{--ink:#111827;--muted:#6b7280;--line:rgba(17,24,39,.09);--surface:rgba(255,255,255,.82);--accent:#0071e3;--good:#15803d;--warn:#a16207;}
html,body,[class*="css"]{font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Segoe UI",sans-serif;}
[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 5% -10%,rgba(0,113,227,.08),transparent 28%),radial-gradient(circle at 95% 10%,rgba(99,102,241,.05),transparent 26%),#f5f5f7;}
.block-container{max-width:1480px;padding-top:1.05rem;padding-bottom:3rem;}
[data-testid="stSidebar"]{background:rgba(255,255,255,.95);border-right:1px solid var(--line);}
[data-testid="stSidebar"] *{color:var(--ink)!important;}
[data-testid="stSidebar"] [data-baseweb="select"]>div,[data-testid="stSidebar"] input{background:#fff!important;color:var(--ink)!important;border-color:rgba(17,24,39,.12)!important;border-radius:12px!important;}
.apple-hero{background:linear-gradient(135deg,rgba(255,255,255,.94),rgba(255,255,255,.72));border:1px solid rgba(255,255,255,.85);border-radius:28px;padding:28px 30px;box-shadow:0 16px 50px rgba(17,24,39,.08);backdrop-filter:blur(20px);margin-bottom:16px;}
.apple-eyebrow{font-size:.79rem;font-weight:800;color:var(--accent);text-transform:uppercase;letter-spacing:.09em;}.apple-title{font-size:2.3rem;line-height:1.03;font-weight:800;color:var(--ink);letter-spacing:-.05em;margin-top:5px;}.apple-sub{font-size:1rem;color:var(--muted);margin-top:8px;max-width:820px;}
.pill{display:inline-block;padding:5px 10px;border-radius:999px;background:#eef6ff;color:#0066cc;font-size:.78rem;font-weight:750;margin-top:13px;}
.card-title{color:var(--ink);font-size:1.08rem;font-weight:780;letter-spacing:-.02em;margin-top:5px;}.card-sub{color:var(--muted);font-size:.92rem;margin-bottom:10px;}
[data-testid="stMetric"]{background:rgba(255,255,255,.75);border:1px solid rgba(17,24,39,.07);border-radius:18px;padding:12px 14px;box-shadow:0 7px 24px rgba(17,24,39,.04);}
[data-testid="stDataFrame"]{border-radius:18px;overflow:hidden;border:1px solid rgba(17,24,39,.08);}
.stButton>button{border-radius:13px;font-weight:750;padding:.65rem 1rem}.stButton>button[kind="primary"]{background:#0071e3;color:#fff;border:0;}
button[data-baseweb="tab"]{font-weight:750;}
hr{border-color:rgba(17,24,39,.07)!important;}

/* V6.1 iPad sidebar fix.
   IMPORTANT: let Streamlit own the sidebar transform/width so its native << button
   can actually collapse it. The main canvas then expands into the released space. */
[data-testid="stMain"], [data-testid="stMainBlockContainer"], .block-container{max-width:100%!important;width:100%!important;}
section[data-testid="stSidebar"]{box-shadow:14px 0 38px rgba(0,0,0,.18);}
section[data-testid="stSidebar"][aria-expanded="false"]{box-shadow:none!important;}
</style>
""",unsafe_allow_html=True)

st.markdown("""
<style>
/* V4.3 iPad readability + scenario engine */
[data-testid="stSidebar"] {
    background: #f5f5f7 !important;
    border-right: 1px solid rgba(17,24,39,.08) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label *,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] * {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [role="combobox"] {
    background: #ffffff !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    border-color: rgba(17,24,39,.14) !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] * {
    -webkit-text-fill-color: initial !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color:#111827 !important;
    -webkit-text-fill-color:#111827 !important;
}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label * {
    opacity: 1 !important;
    color:#374151 !important;
    -webkit-text-fill-color:#374151 !important;
    font-weight:600 !important;
}

/* Make segmented control choices readable on the light sidebar. */
[data-testid="stSidebar"] [data-baseweb="button-group"] button,
[data-testid="stSidebar"] [data-baseweb="button-group"] button * {
    color:#111827 !important;
    -webkit-text-fill-color:#111827 !important;
}
/* Do not force sidebar width on iPad. A fixed width prevents Streamlit's
   collapse transform from taking effect in Safari. */
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* DFS Lab V5 — dark iPad control deck */
:root{--v5-bg:#f5f5f7;--v5-panel:#ffffff;--v5-line:rgba(0,0,0,.10);--v5-text:#1d1d1f;--v5-muted:#6e6e73;}
[data-testid="stAppViewContainer"]{background:linear-gradient(180deg,#fbfbfd 0%,#f5f5f7 52%,#f2f2f4 100%)!important;color:var(--v5-text)!important;}
[data-testid="stHeader"]{background:transparent!important}.block-container{max-width:1480px;padding-top:1rem}
.apple-hero{background:rgba(255,255,255,.88)!important;border:1px solid rgba(0,0,0,.08)!important;box-shadow:0 18px 50px rgba(0,0,0,.07)!important}
.apple-title,.card-title,h1,h2,h3,h4{color:var(--v5-text)!important}.apple-sub,.card-sub,.muted{color:var(--v5-muted)!important}.pill{background:rgba(47,129,247,.15)!important;color:#8ec5ff!important}
[data-testid="stMetric"]{background:#fff!important;border:1px solid var(--v5-line)!important;box-shadow:0 8px 24px rgba(0,0,0,.04)!important}[data-testid="stMetricLabel"],[data-testid="stMetricValue"]{color:var(--v5-text)!important}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,#1769d2,#2f81f7)!important;box-shadow:0 8px 24px rgba(47,129,247,.22)}
.lineup-card{background:#fff;border:1px solid var(--v5-line);border-radius:18px;padding:16px 18px;margin:10px 0;box-shadow:0 10px 30px rgba(0,0,0,.05)}.lineup-rank{font-size:.78rem;font-weight:800;color:#0071e3}.lineup-rank span{background:rgba(0,113,227,.10);padding:3px 7px;border-radius:999px}.lineup-cpt{font-size:1.15rem;font-weight:800;color:#1d1d1f;margin-top:7px}.lineup-flex{font-size:.92rem;color:#424245;margin-top:5px}.lineup-meta{font-size:.80rem;color:#6e6e73;margin-top:9px}
@media(max-width:800px){.apple-title{font-size:1.85rem!important}.apple-hero{padding:21px!important}.block-container{padding-left:.8rem!important;padding-right:.8rem!important}.lineup-card{padding:14px}}
</style>
""",unsafe_allow_html=True)


st.markdown("""<style>
[data-testid="stAppViewContainer"] label {color:#1d1d1f !important;}
.stCaption, [data-testid="stCaptionContainer"], .card-sub {color:#6e6e73 !important;}
[data-testid="stTabs"] button p {color:#6e6e73 !important;}
[data-testid="stTabs"] button[aria-selected="true"] p {color:#1d1d1f !important;}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {color:#303033;}
[data-testid="stSidebarCollapseButton"] button{background:#1d1d1f!important;color:white!important;border-radius:999px!important;min-width:42px!important;min-height:42px!important;box-shadow:0 4px 16px rgba(0,0,0,.18)!important;}
[data-testid="stSidebarCollapseButton"] svg{fill:white!important;color:white!important;}
</style>""",unsafe_allow_html=True)
st.markdown(r"""
<style>
/* DFS LAB V6.3.2 — product design system */
:root{--lab-bg:#e9edf3;--lab-canvas:#f2f4f7;--lab-panel:#ffffff;--lab-sidebar:#e3e8ef;--lab-ink:#101828;--lab-muted:#667085;--lab-line:#d6dce5;--lab-blue:#1267d6;--lab-blue2:#2f7eea;}
[data-testid="stAppViewContainer"]{background:linear-gradient(135deg,#e6ebf2 0%,#f4f6f9 46%,#e9eef5 100%)!important;color:var(--lab-ink)!important;}
[data-testid="stHeader"]{background:rgba(242,244,247,.82)!important;backdrop-filter:blur(18px)!important;border-bottom:1px solid rgba(16,24,40,.06)!important;}
.block-container{max-width:1500px!important;padding-top:1.25rem!important;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#e0e6ee,#edf0f5)!important;border-right:1px solid #cbd3df!important;box-shadow:10px 0 32px rgba(25,39,62,.08)!important;}
section[data-testid="stSidebar"] *{color:var(--lab-ink)!important;-webkit-text-fill-color:initial!important;}
section[data-testid="stSidebar"] [data-baseweb="select"]>div,section[data-testid="stSidebar"] input{background:rgba(255,255,255,.82)!important;border:1px solid #cbd3df!important;box-shadow:none!important;}
.apple-hero{background:linear-gradient(120deg,#111b2d 0%,#182944 62%,#173760 100%)!important;border:1px solid rgba(255,255,255,.08)!important;box-shadow:0 20px 50px rgba(26,45,75,.16)!important;border-radius:24px!important;padding:28px 32px!important;}
.apple-hero .apple-eyebrow{color:#79b8ff!important}.apple-hero .apple-title{color:#fff!important;font-size:2.55rem!important}.apple-hero .apple-sub{color:#c9d4e3!important}.apple-hero .pill{background:rgba(58,139,253,.16)!important;color:#9ac8ff!important;border:1px solid rgba(121,184,255,.22)!important;}
[data-testid="stMetric"]{background:rgba(255,255,255,.88)!important;border:1px solid var(--lab-line)!important;border-radius:16px!important;box-shadow:0 8px 24px rgba(31,50,81,.06)!important;}
[data-testid="stDataFrame"]{background:#fff!important;border:1px solid var(--lab-line)!important;border-radius:16px!important;box-shadow:0 8px 26px rgba(31,50,81,.05)!important;}
[data-testid="stTabs"]{background:transparent!important;}
button[data-baseweb="tab"]{padding-top:.7rem!important;padding-bottom:.7rem!important;}
.stButton>button{border-radius:12px!important;min-height:44px!important;font-weight:750!important;}
.stButton>button[kind="primary"],[data-testid="stDownloadButton"]>button{background:linear-gradient(90deg,var(--lab-blue),var(--lab-blue2))!important;color:#fff!important;-webkit-text-fill-color:#fff!important;border:0!important;box-shadow:0 8px 20px rgba(18,103,214,.20)!important;}
.stButton>button[kind="primary"] *,[data-testid="stDownloadButton"]>button *{color:#fff!important;-webkit-text-fill-color:#fff!important;opacity:1!important;}
/* Make Streamlit's native collapse control look like a product control, not a black orb. */
[data-testid="stSidebarCollapseButton"] button{background:var(--lab-blue)!important;color:#fff!important;border:1px solid rgba(255,255,255,.5)!important;border-radius:10px!important;min-width:54px!important;min-height:40px!important;box-shadow:0 6px 18px rgba(18,103,214,.22)!important;}
[data-testid="stSidebarCollapseButton"] svg{fill:#fff!important;color:#fff!important;stroke:#fff!important;}
[data-testid="stSidebarCollapseButton"] button:hover{background:#0b5fc8!important;}
.card-title,h1,h2,h3,h4{color:var(--lab-ink)!important}.card-sub,.muted,[data-testid="stCaptionContainer"]{color:var(--lab-muted)!important;}
.lineup-card{background:#fff!important;border:1px solid var(--lab-line)!important;box-shadow:0 10px 28px rgba(31,50,81,.06)!important;}
@media(max-width:900px){.apple-hero{padding:22px 24px!important}.apple-hero .apple-title{font-size:2.1rem!important}.block-container{padding-left:1rem!important;padding-right:1rem!important;}}
</style>
""",unsafe_allow_html=True)

st.markdown('''<div class="apple-hero"><div class="apple-eyebrow">DFS LAB</div><div class="apple-title">DFS LAB</div><div class="apple-sub">Build lineups around how you think the game will happen.</div><span class="pill">V6.3.2 • Design System</span></div>''',unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### DFS LAB Controls")
    st.caption("Use the blue Hide control at the top edge to collapse this panel. It stays available as Controls when collapsed.")
    st.markdown("### Contest")
    mode=st.segmented_control("Mode",["Classic","Showdown"],default="Showdown")
    preset=st.selectbox("Contest preset",["Large GPP","Small-field GPP","Single Entry","Winner Take All","Cash-ish"])
    defaults={"Large GPP":(50000,"GPP / top-heavy","150-Max"),"Small-field GPP":(500,"GPP / top-heavy","3-Max"),"Single Entry":(300,"Flatter payouts","Single Entry"),"Winner Take All":(500,"Winner take all","Single Entry"),"Cash-ish":(100,"Flatter payouts","Single Entry")}
    dfield,dpayout,dentry=defaults[preset]
    field_size=st.number_input("Field size",min_value=2,value=int(dfield),step=1)
    entry_format=st.selectbox("Entry format",["Single Entry","3-Max","20-Max","150-Max"],index=["Single Entry","3-Max","20-Max","150-Max"].index(dentry))
    payout_style=st.selectbox("Payout",["GPP / top-heavy","Winner take all","Flatter payouts"],index=["GPP / top-heavy","Winner take all","Flatter payouts"].index(dpayout))
    lineup_count=st.slider("Lineup pool",25,500,100,25)
    with st.expander("Advanced"):
        seed=st.number_input("Random seed",min_value=1,value=42,step=1)
        st.caption("Change this only when you want a different randomized batch.")

st.markdown('<div class="card-title">Slate files</div><div class="card-sub">DraftKings is the only required file. DFS Lab can create its own baseline projection; SaberSim is now optional and used only as a comparison source.</div>',unsafe_allow_html=True)
u1,u2=st.columns(2)
with u1: dk_file=st.file_uploader("DraftKings salaries/template · required",type=["csv"],key="dfs_lab_dk_upload")
with u2: ss_file=st.file_uploader("SaberSim · optional comparison",type=["csv"],key="dfs_lab_ss_upload")

# Keep a working copy of uploaded bytes during ordinary Streamlit reruns. This prevents
# widget refreshes from forcing the user to remove/re-add the same slate.
if dk_file is not None:
    st.session_state["dfs_lab_dk_bytes"]=dk_file.getvalue(); st.session_state["dfs_lab_dk_name"]=getattr(dk_file,"name","DKSalaries.csv")
elif st.session_state.get("dfs_lab_dk_bytes"):
    dk_file=io.BytesIO(st.session_state["dfs_lab_dk_bytes"]); dk_file.name=st.session_state.get("dfs_lab_dk_name","DKSalaries.csv")
if ss_file is not None:
    st.session_state["dfs_lab_ss_bytes"]=ss_file.getvalue(); st.session_state["dfs_lab_ss_name"]=getattr(ss_file,"name","SaberSim.csv")
elif st.session_state.get("dfs_lab_ss_bytes"):
    ss_file=io.BytesIO(st.session_state["dfs_lab_ss_bytes"]); ss_file.name=st.session_state.get("dfs_lab_ss_name","SaberSim.csv")

if dk_file is None:
    st.info("Upload the DraftKings slate to open DFS LAB.")
    st.stop()
if mode=="Classic" and not ss_file:
    st.info("DFS Lab-only projections are enabled for Showdown first. Classic still needs the projection file in this V6 test build.")
    st.stop()

if mode=="Classic":
    # Preserve the proven V3 Classic engine with a compact V4 shell.
    try:
        df=prepare_player_pool(dk_file,ss_file); teams=sorted(df["Team"].dropna().unique().tolist())
        st.session_state.setdefault("strategy_master",{}); st.session_state.setdefault("team_strategy_master",{})
        q1,q2,q3,q4=st.columns(4); q1.metric("Players",len(df)); q2.metric("Teams",len(teams)); q3.metric("Field",f"{int(field_size):,}"); q4.metric("Pool",lineup_count)
        min_salary=st.sidebar.slider("Min salary",44000,50000,49000,100)
        qb_stack=st.sidebar.selectbox("QB pass catchers",[1,2],index=1)
        bringback_mode=st.sidebar.selectbox("Bring-back",["Optional","Required","None"])
        tabs=st.tabs(["Build","Players","Rules","Lineups","Exposure"])
        with tabs[0]:
            preferred_stack_teams=st.multiselect("Preferred QB stack teams",teams)
            team_df=pd.DataFrame({"Team":teams,"Priority":[st.session_state["team_strategy_master"].get(t,"Neutral") for t in teams]})
            team_edit=st.data_editor(team_df,hide_index=True,use_container_width=True,disabled=["Team"],column_config={"Priority":st.column_config.SelectboxColumn("Lean",options=["Core","Like","Neutral","Fade","Exclude"])},key="v4_classic_team")
            for _,r in team_edit.iterrows(): st.session_state["team_strategy_master"][r["Team"]]=r["Priority"]
            build_btn=st.button("Generate rated lineups",type="primary",use_container_width=True,key="v4_classic_build")
        with tabs[1]:
            view=df.copy(); team_filter=st.multiselect("Teams",teams,key="v4_cteam"); pos_filter=st.multiselect("Positions",["QB","RB","WR","TE","DST"],key="v4_cpos")
            if team_filter:view=view[view["Team"].isin(team_filter)]
            if pos_filter:view=view[view["Position"].isin(pos_filter)]
            ed=pd.DataFrame({"ID":view["ID"].astype(str),"Name":view["Name"],"Pos":view["Position"],"Team":view["Team"],"Salary":view["Salary"],"Proj":view["My Proj"].round(2),"Own":view["My Own"].round(1),"Lock":False,"Exclude":False,"Priority":"Neutral","Min Exposure":0,"Max Exposure":100})
            for x,r in ed.iterrows():
                e=st.session_state["strategy_master"].get(str(r["ID"]),{})
                for c,k,d in [("Lock","Lock",False),("Exclude","Exclude",False),("Priority","Priority","Neutral"),("Min Exposure","Min Exposure",0),("Max Exposure","Max Exposure",100)]: ed.at[x,c]=e.get(k,d)
            st.caption("Make all of your player/Captain changes, then tap Apply changes once. This prevents the screen from dimming after every checkbox.")
            with st.form("showdown_player_editor_form", clear_on_submit=False):
                edited=st.data_editor(
                    ed,
                hide_index=True,
                use_container_width=True,
                height=620,
                disabled=["ID","Name","Pos","Team","Salary","Proj","Own"],
                column_order=["Name","Pos","Team","Salary","Proj","Own","Lock","Exclude","Priority","Min Exposure","Max Exposure"],
                column_config={
                    "Name":st.column_config.TextColumn("Player",width=190,pinned=True),
                    "Pos":st.column_config.TextColumn("Pos",width=60),
                    "Team":st.column_config.TextColumn("Team",width=70),
                    "Salary":st.column_config.NumberColumn("Salary",width=85,format="$%d"),
                    "Proj":st.column_config.NumberColumn("Proj",width=75,format="%.2f"),
                    "Own":st.column_config.NumberColumn("Own",width=70,format="%.1f"),
                    "Priority":st.column_config.SelectboxColumn("Lean",options=PRIORITY_OPTIONS,width=95),
                    "Lock":st.column_config.CheckboxColumn("Lock",width=65),
                    "Exclude":st.column_config.CheckboxColumn("Out",width=60),
                    "Min Exposure":st.column_config.NumberColumn("Min %",min_value=0,max_value=100,step=5,width=75),
                    "Max Exposure":st.column_config.NumberColumn("Max %",min_value=0,max_value=100,step=5,width=75),
                },
                key="v4_classic_players",
            )
            for _,r in edited.iterrows():
                ex=bool(r["Exclude"]); st.session_state["strategy_master"][str(r["ID"]) ]={"Lock":bool(r["Lock"]) and not ex,"Exclude":ex,"Priority":"Exclude" if ex else str(r["Priority"]),"Min Exposure":float(r["Min Exposure"]),"Max Exposure":float(r["Max Exposure"])}
        with tabs[2]:
            no_dst=st.toggle("No defense from my stacked game",value=True,key="v4_no_dst"); no_off=st.toggle("No offense against my DST",value=False,key="v4_no_off")
        if build_btn:
            res=generate_lineups(df,field_size,payout_style,lineup_count,max(300,lineup_count*12),min_salary,qb_stack,bringback_mode,preferred_stack_teams,st.session_state["strategy_master"],st.session_state["team_strategy_master"],no_dst,no_off,seed)
            st.session_state["classic_result_v4"]=res
        res=st.session_state.get("classic_result_v4")
        with tabs[3]:
            if res is None or res.empty: st.info("Generate lineups from Build.")
            else:
                show_cols=["Rank","Rating","Rating Score","Projection","Base Projection","Scenario Delta","Salary","Salary Left","Avg Own","Stack Summary"]+ROSTER_SLOTS
                st.dataframe(res[[c for c in show_cols if c in res.columns]],hide_index=True,use_container_width=True,height=590)
                st.download_button("Download lineup analysis CSV",res.to_csv(index=False),"classic_lineups_v4.csv","text/csv",use_container_width=True)
        with tabs[4]:
            if res is None or res.empty: st.info("Generate lineups first.")
            else: st.dataframe(calculate_exposure_table(df,res,st.session_state["strategy_master"]),hide_index=True,use_container_width=True,height=620)
    except Exception as e:
        st.error(f"Classic build error: {e}")
else:
    try:
        df=prepare_showdown_pool(dk_file,ss_file); teams=[t for t in df["Team"].dropna().unique().tolist() if t]
        if len(teams)!=2: st.warning(f"Showdown normally has two teams. I found {len(teams)}: {', '.join(teams)}")
        nonzero_proj=int((pd.to_numeric(df["My Proj"],errors="coerce").fillna(0)>0.05).sum())
        nonzero_own=int((pd.to_numeric(df["My Own"],errors="coerce").fillna(0)>0).sum())
        if nonzero_proj==0:
            st.error("DFS Lab could not create usable projections for this slate.")
            st.stop()
        elif nonzero_proj < 6: st.warning(f"Projection check · Only {nonzero_proj} players have usable projections. The slate may be incomplete.")
        else: st.success(f"DFS Lab projection engine ready · {nonzero_proj} players have usable projections." + (" · SaberSim comparison loaded." if ss_file else " · No SaberSim file used."))
        if nonzero_own==0: st.warning("Ownership not populated yet · Lineups can be built, but leverage and duplication ratings that depend on ownership are provisional.")
        st.session_state.setdefault("showdown_strategy",{})
        st.session_state.setdefault("showdown_context",{})
        st.session_state.setdefault("showdown_relationships",[])
        st.session_state.setdefault("projection_overrides",{})
        st.session_state.setdefault("context_strength","Standard")
        st.session_state.setdefault("sd_use_score", False)
        st.session_state.setdefault("sd_script", "Neutral")
        st.session_state.setdefault("sd_script_team", "None")
        # V6.2.1 migration: older sessions stored Conservative/Standard/Aggressive text.
        _old_intensity = st.session_state.get("sd_intensity", 50)
        if isinstance(_old_intensity, str):
            _old_intensity = {"Conservative": 30, "Standard": 50, "Aggressive": 75}.get(_old_intensity, 50)
        try:
            _old_intensity = int(float(_old_intensity))
        except Exception:
            _old_intensity = 50
        st.session_state["sd_intensity"] = max(0, min(100, _old_intensity))
        st.session_state.setdefault("sd_auto_shape", True)
        if len(teams) >= 2:
            st.session_state.setdefault("sd_score_0", 24)
            st.session_state.setdefault("sd_score_1", 21)
        q1,q2,q3,q4=st.columns(4); q1.metric("Players",len(df)); q2.metric("Teams",len(teams)); q3.metric("Field",f"{int(field_size):,}"); q4.metric("Pool",lineup_count)
        if bool(df["CPT Own Estimated"].any()): st.warning("Your SaberSim file does not appear to include Captain ownership. DFS Lab is using a neutral fallback for CPT leverage. Overall ownership is still used normally.")

        # Showdown-specific sidebar controls.
        salary_style=st.sidebar.selectbox("Salary strategy",["Optimal","Balanced","GPP","Unique","Custom"],index=2)
        salary_defaults={"Optimal":49500,"Balanced":48500,"GPP":47500,"Unique":46000,"Custom":47000}
        if salary_style=="Custom": min_salary=st.sidebar.slider("Minimum salary",40000,50000,47000,100)
        else: min_salary=salary_defaults[salary_style]; st.sidebar.caption(f"Minimum salary: ${min_salary:,}")
        max_salary=50000
        min_unique=st.sidebar.selectbox("Minimum unique players",[1,2,3],index=0)

        tabs=st.tabs(["Build","Players","Relationships","Game View","Scripts","Lineup Lab","Exposure"])
        with tabs[0]:
            st.markdown('<div class="card-title">Showdown Build</div><div class="card-sub">Control how the six-man portfolio is shaped before the optimizer starts solving.</div>',unsafe_allow_html=True)
            st.markdown("#### Allowed team builds")
            st.caption("Choose what is allowed — no percentages. 4-2 means four players from either team; 5-1 means five from either team.")
            c1,c2,c3=st.columns(3)
            with c1:allow_33=st.checkbox("3-3",value=True,key="sd_allow_33")
            with c2:allow_42=st.checkbox("4-2",value=True,key="sd_allow_42")
            with c3:allow_51=st.checkbox("5-1",value=False,key="sd_allow_51")
            if not any([allow_33,allow_42,allow_51]):
                st.warning("Choose at least one build. 3-3 will be used until you select one.")
                allow_33=True
            construction_weights={"3-3":1 if allow_33 else 0,"4-2":1 if allow_42 else 0,"5-1":1 if allow_51 else 0}
            st.caption("The Scenario Engine decides how often to use each allowed build based on your score and game script.")
            st.markdown("#### Captain pairing")
            st.caption("These settings apply only when that position is Captain — they do not control how often the position becomes Captain.")
            a,b,c=st.columns(3)
            with a:cpt_qb_pc=st.selectbox("When QB is CPT · pass catchers",[0,1,2,3],index=2,help="Number of same-team WR/TE pass catchers to pair with a QB Captain.")
            pair_map={"Never":0,"Sometimes":35,"Usually":80,"Always":100}
            with b: wrte_pair=st.selectbox("When WR/TE is CPT · pair QB",list(pair_map),index=2)
            with c: rb_pair=st.selectbox("When RB is CPT · pair DST/K",list(pair_map),index=1)
            wrte_qb=pair_map[wrte_pair]; rb_ctrl=pair_map[rb_pair]
            d,e=st.columns(2)
            with d:max_k=st.selectbox("Max kickers",[0,1,2],index=2)
            with e:max_dst=st.selectbox("Max defenses",[0,1,2],index=1)
            build_btn=st.button("Generate 100 Lineups",type="primary",use_container_width=True,key="v4_sd_build")

        with tabs[1]:
            st.markdown('<div class="card-title">Player + Captain Exposure</div><div class="card-sub">Overall exposure and Captain exposure are controlled separately.</div>',unsafe_allow_html=True)
            team_filter=st.multiselect("Teams",teams,key="v4_sdteam")
            pos_filter=st.multiselect("Positions",sorted(df["Position"].dropna().unique().tolist()),key="v4_sdpos")
            current_script=st.session_state.get("sd_script","Neutral")
            current_team=st.session_state.get("sd_script_team","None")
            if current_team=="None": current_team=""
            current_use_score=bool(st.session_state.get("sd_use_score",False))
            current_scores={}
            if len(teams)>=2:
                current_scores={teams[0]:float(st.session_state.get("sd_score_0",24)),teams[1]:float(st.session_state.get("sd_score_1",21))}
            if current_script=="Auto from score" and current_use_score:
                current_script,current_team,_=infer_score_script(current_scores)
            scenario_df=apply_showdown_scenario(df,current_script,current_team,current_use_score,current_scores,st.session_state.get("sd_intensity",50))
            scenario_df=apply_context_engine(scenario_df,st.session_state.get("showdown_context",{}),st.session_state.get("context_strength","Standard"))
            scenario_df=apply_projection_overrides(scenario_df,st.session_state.get("projection_overrides",{}))
            view=scenario_df.copy()
            if team_filter:view=view[view["Team"].isin(team_filter)]
            if pos_filter:view=view[view["Position"].isin(pos_filter)]

            st.markdown("##### Quick lock")
            quick_names=view["Name"].astype(str).tolist()
            if quick_names:
                q1,q2,q3,q4,q5=st.columns([2.4,1,1,1,1])
                with q1: quick_player=st.selectbox("Quick player",quick_names,key="sd_quick_player",label_visibility="collapsed")
                qrow=view[view["Name"].astype(str)==str(quick_player)].iloc[0]; qid=str(qrow["ID"])
                if qid not in st.session_state["showdown_strategy"]: st.session_state["showdown_strategy"][qid]={}
                with q2:
                    if st.button("Lock",use_container_width=True,key="sd_quick_lock"):
                        st.session_state["showdown_strategy"][qid].update({"Lock":True,"CPT Lock":False,"Exclude":False}); st.rerun()
                with q3:
                    if st.button("CPT",use_container_width=True,key="sd_quick_cpt"):
                        st.session_state["showdown_strategy"][qid].update({"Lock":False,"CPT Lock":True,"Exclude":False,"CPT Eligible":True}); st.rerun()
                with q4:
                    if st.button("Out",use_container_width=True,key="sd_quick_out"):
                        st.session_state["showdown_strategy"][qid].update({"Lock":False,"CPT Lock":False,"Exclude":True,"CPT Eligible":False,"Priority":"Exclude"}); st.rerun()
                with q5:
                    if st.button("Clear",use_container_width=True,key="sd_quick_clear"):
                        st.session_state["showdown_strategy"].pop(qid,None); st.rerun()
                st.caption("Lock = every lineup • CPT = Captain every lineup • Out = never use")

            ed=pd.DataFrame({"ID":view["ID"].astype(str),"Name":view["Name"],"Pos":view["Position"],"Team":view["Team"],"Flex $":view["FlexSalary"],"DFS Base":view["My Proj"].round(2),"Hist G":view.get("History Games",pd.Series(0,index=view.index)),"Matchup %":view.get("Matchup Adj %",pd.Series(0.0,index=view.index)),"Model":view["Model Proj"].round(2),"Your Proj":view["DFS Lab Proj"].round(2),"Δ%":view["Proj Change %"].round(1),"Own":view["My Own"].round(1),"CPT Own":view["CPT Own"].round(1),"Lock":False,"CPT Lock":False,"Exclude":False,"CPT Eligible":True,"Priority":"Neutral","Min Exposure":0,"Max Exposure":100,"CPT Min":0,"CPT Max":100})
            for x,r in ed.iterrows():
                e=st.session_state["showdown_strategy"].get(str(r["ID"]),{})
                for c,k,d in [("Lock","Lock",False),("CPT Lock","CPT Lock",False),("Exclude","Exclude",False),("CPT Eligible","CPT Eligible",True),("Priority","Priority","Neutral"),("Min Exposure","Min Exposure",0),("Max Exposure","Max Exposure",100),("CPT Min","CPT Min",0),("CPT Max","CPT Max",100)]: ed.at[x,c]=e.get(k,d)
            st.caption("Make all of your player/Captain changes, then tap Apply changes once. This prevents the screen from dimming after every checkbox.")
            with st.form("showdown_player_editor_form", clear_on_submit=False):
                edited=st.data_editor(
                    ed,
                hide_index=True,
                use_container_width=True,
                height=650,
                disabled=["ID","Name","Pos","Team","Flex $","DFS Base","Hist G","Matchup %","Model","Δ%","Own","CPT Own"],
                column_order=["Name","Lock","CPT Lock","Exclude","CPT Eligible","Priority","Pos","Team","Flex $","DFS Base","Hist G","Matchup %","Model","Your Proj","Δ%","Own","CPT Own","Min Exposure","Max Exposure","CPT Min","CPT Max"],
                column_config={
                    "ID":None,
                    "Name":st.column_config.TextColumn("Player",width=190,pinned=True),
                    "Pos":st.column_config.TextColumn("Pos",width=58),
                    "Team":st.column_config.TextColumn("Team",width=68),
                    "Flex $":st.column_config.NumberColumn("Flex $",width=78,format="$%d"),
                    "Proj":st.column_config.NumberColumn("Base",width=68,format="%.2f"),
                    "Script Proj":st.column_config.NumberColumn("Scenario",width=78,format="%.2f"),
                    "DFS Lab":st.column_config.NumberColumn("DFS Lab",width=78,format="%.2f"),
                    "Δ%":st.column_config.NumberColumn("Δ%",width=58,format="%.1f"),
                    "Own":st.column_config.NumberColumn("Own",width=64,format="%.1f"),
                    "CPT Own":st.column_config.NumberColumn("CPT Own",width=78,format="%.1f"),
                    "Lock":st.column_config.CheckboxColumn("Lock",width=62,pinned=True),
                    "CPT Lock":st.column_config.CheckboxColumn("CPT",width=62,pinned=True),
                    "Exclude":st.column_config.CheckboxColumn("Out",width=58,pinned=True),
                    "CPT Eligible":st.column_config.CheckboxColumn("CPT?",width=60),
                    "Priority":st.column_config.SelectboxColumn("Lean",options=PRIORITY_OPTIONS,width=92),
                    "Min Exposure":st.column_config.NumberColumn("Min %",min_value=0,max_value=100,step=5,width=70),
                    "Max Exposure":st.column_config.NumberColumn("Max %",min_value=0,max_value=100,step=5,width=70),
                    "CPT Min":st.column_config.NumberColumn("CPT Min",min_value=0,max_value=100,step=5,width=74),
                    "CPT Max":st.column_config.NumberColumn("CPT Max",min_value=0,max_value=100,step=5,width=74),
                },
                    key="v4_sdplayers",
                )
                apply_player_changes=st.form_submit_button("Apply player changes",type="primary",use_container_width=True)
            if apply_player_changes:
                for _,r in edited.iterrows():
                    pid=str(r["ID"]); model_val=float(view.loc[view["ID"].astype(str).eq(pid),"Model Proj"].iloc[0]) if not view.loc[view["ID"].astype(str).eq(pid)].empty else float(r["Your Proj"])
                    user_val=float(r["Your Proj"])
                    if abs(user_val-model_val)>0.01: st.session_state["projection_overrides"][pid]=user_val
                    else: st.session_state["projection_overrides"].pop(pid,None)
                    ex=bool(r["Exclude"]); cptlock=bool(r["CPT Lock"]) and not ex
                    st.session_state["showdown_strategy"][str(r["ID"]) ]={"Lock":bool(r["Lock"]) and not ex and not cptlock,"CPT Lock":cptlock,"Exclude":ex,"CPT Eligible":bool(r["CPT Eligible"]) and not ex,"Priority":"Exclude" if ex else str(r["Priority"]),"Min Exposure":float(r["Min Exposure"]),"Max Exposure":float(r["Max Exposure"]),"CPT Min":float(r["CPT Min"]),"CPT Max":float(r["CPT Max"])}
                st.success("Player settings applied.")
        with tabs[2]:
            st.markdown('<div class="card-title">Relationships</div><div class="card-sub">Teach DFS Lab which players, positions and team roles belong together — or should never appear together.</div>',unsafe_allow_html=True)
            st.caption("Hard relationship rules are enforced by the optimizer. Use Player for a specific matchup or Team + Position for broader football logic.")
            player_options={f"{r['Name']} · {r['Team']} {r['Position']}":str(r['ID']) for _,r in df.sort_values(['Team','Position','Name']).iterrows()}
            positions=["Any"]+sorted([p for p in df["Position"].dropna().astype(str).unique().tolist() if p])
            team_opts=["Any"]+teams
            with st.expander("+ Add relationship",expanded=True):
                r1,r2=st.columns(2)
                with r1:
                    a_kind=st.segmented_control("Side A",["Player","Team + Position"],default="Player",key="rel_a_kind")
                    if a_kind=="Player":
                        a_label=st.selectbox("A player",list(player_options),key="rel_a_player")
                        a={"kind":"Player","id":player_options[a_label],"label":a_label}
                    else:
                        at=st.selectbox("A team",team_opts,key="rel_a_team"); ap=st.selectbox("A position",positions,key="rel_a_pos")
                        a={"kind":"Team + Position","team":at,"position":ap,"label":f"{at} {ap}"}
                with r2:
                    b_kind=st.segmented_control("Side B",["Player","Team + Position"],default="Player",key="rel_b_kind")
                    if b_kind=="Player":
                        b_label=st.selectbox("B player",list(player_options),key="rel_b_player")
                        b={"kind":"Player","id":player_options[b_label],"label":b_label}
                    else:
                        bt=st.selectbox("B team",team_opts,key="rel_b_team"); bp=st.selectbox("B position",positions,key="rel_b_pos")
                        b={"kind":"Team + Position","team":bt,"position":bp,"label":f"{bt} {bp}"}
                rule_type=st.selectbox("Relationship",["Never Together","Require B when A used"],help="Never Together blocks every A/B pairing. Require B means any lineup using A must contain at least one B.")
                if st.button("Add relationship rule",type="primary",use_container_width=True,key="add_relationship"):
                    if a==b:
                        st.warning("Choose two different sides.")
                    else:
                        st.session_state["showdown_relationships"].append({"a":a,"b":b,"rule":rule_type,"enabled":True})
                        st.rerun()
            rules=st.session_state["showdown_relationships"]
            if not rules:
                st.info("No custom relationships yet. Example: Jahmyr Gibbs ↔ Jacob Saylors → Never Together, or BUF DST ↔ DET QB → Never Together.")
            else:
                for ri,rule in enumerate(list(rules)):
                    c1,c2,c3=st.columns([5,1,1])
                    with c1: st.markdown(f"**{rule['a'].get('label','A')}**  →  **{rule['rule']}**  →  **{rule['b'].get('label','B')}**")
                    with c2:
                        enabled=st.toggle("On",value=rule.get("enabled",True),key=f"rel_on_{ri}")
                        st.session_state["showdown_relationships"][ri]["enabled"]=enabled
                    with c3:
                        if st.button("Remove",key=f"rel_rm_{ri}",use_container_width=True):
                            st.session_state["showdown_relationships"].pop(ri); st.rerun()
            st.markdown("#### Quick football rules")
            st.caption("These create broad restrictions without naming individual players.")
            max_one_rb=st.toggle("Max 1 RB from the same team",value=st.session_state.get("max_one_rb_team",False),key="max_one_rb_team",help="Useful when two same-team RBs are direct alternatives. Leave off when a backfield can realistically support two players together.")

        with tabs[3]:
            st.markdown('<div class="card-title">Game View</div><div class="card-sub">The football signals DFS LAB is using for this slate. Neutral inputs stay out of the way; open Model details only when you want to audit them.</div>',unsafe_allow_html=True)
            st.info("Ratings are confidence-shrunk and capped. Defense and current usage carry more weight than travel or primetime splits.")
            context_strength=st.select_slider("Context influence",options=["Conservative","Standard","Aggressive"],key="context_strength")
            rating_opts=[-3,-2,-1,0,1,2,3]
            ced=pd.DataFrame({
                "ID":df["ID"].astype(str),"Player":df["Name"],"Pos":df["Position"],"Team":df["Team"],
                "Defense":0,"Usage":0,"Home/Rest":0,"Travel":0,"Time/Split":0,"Confidence":50,"Note":""
            })
            for x,r in ced.iterrows():
                cfg=st.session_state["showdown_context"].get(str(r["ID"]),{})
                for col in ["Defense","Usage","Home/Rest","Travel","Time/Split","Confidence","Note"]:
                    ced.at[x,col]=cfg.get(col,ced.at[x,col])
            cedited=st.data_editor(ced,hide_index=True,use_container_width=True,height=560,
                disabled=["ID","Player","Pos","Team"],
                column_order=["Player","Pos","Team","Defense","Usage","Home/Rest","Travel","Time/Split","Confidence","Note"],
                column_config={
                    "ID":None,"Player":st.column_config.TextColumn("Player",pinned=True,width=185),
                    "Pos":st.column_config.TextColumn("Pos",width=55),"Team":st.column_config.TextColumn("Team",width=62),
                    "Defense":st.column_config.SelectboxColumn("Defense",options=rating_opts,width=78,help="-3 strong negative to +3 strong positive defensive matchup."),
                    "Usage":st.column_config.SelectboxColumn("Usage",options=rating_opts,width=72,help="Current role: routes, targets, carries, red-zone work and injury-driven opportunity."),
                    "Home/Rest":st.column_config.SelectboxColumn("Home/Rest",options=rating_opts,width=90),
                    "Travel":st.column_config.SelectboxColumn("Travel",options=rating_opts,width=70),
                    "Time/Split":st.column_config.SelectboxColumn("Time/Split",options=rating_opts,width=90,help="Day/night/Thursday/Monday split. Intentionally low weight."),
                    "Confidence":st.column_config.NumberColumn("Conf %",min_value=0,max_value=100,step=5,width=78),
                    "Note":st.column_config.TextColumn("Why / source note",width=220),
                },key="v51_context_editor")
            for _,r in cedited.iterrows():
                st.session_state["showdown_context"][str(r["ID"])]= {k:(int(r[k]) if k not in ["Note"] else str(r[k])) for k in ["Defense","Usage","Home/Rest","Travel","Time/Split","Confidence","Note"]}

            # Preview context on top of the current scenario state.
            ctx_script=st.session_state.get("sd_script","Neutral"); ctx_team=st.session_state.get("sd_script_team","None")
            if ctx_team=="None": ctx_team=""
            ctx_use_score=bool(st.session_state.get("sd_use_score",False)); ctx_scores={}
            if len(teams)>=2: ctx_scores={teams[0]:float(st.session_state.get("sd_score_0",24)),teams[1]:float(st.session_state.get("sd_score_1",21))}
            if ctx_script=="Auto from score" and ctx_use_score: ctx_script,ctx_team,_=infer_score_script(ctx_scores)
            ctx_base=apply_showdown_scenario(df,ctx_script,ctx_team,ctx_use_score,ctx_scores,st.session_state.get("sd_intensity",50))
            ctx_df=apply_context_engine(ctx_base,st.session_state["showdown_context"],context_strength)
            st.markdown("#### Market vs. model foundation")
            st.caption("Base is now DFS Lab's independent nflverse/DK-prior projection. Scenario + context create the model projection; Your Proj can override the final optimizer input.")
            cp_cols=["Name","My Proj","Script Proj","Context Adj %","DFS Lab Proj"] + (["SaberSim Proj"] if "SaberSim Proj" in ctx_df.columns else [])
            cp=ctx_df[cp_cols].copy()
            cp.columns=["Player","Base","Scenario","Context %","DFS Lab"] + (["SaberSim"] if "SaberSim Proj" in ctx_df.columns else [])
            cp=cp.sort_values("Context %",key=lambda x:x.abs(),ascending=False)
            st.dataframe(cp,hide_index=True,use_container_width=True,height=390,column_config={"Player":st.column_config.TextColumn("Player",pinned=True,width=185),"Base":st.column_config.NumberColumn("Base",format="%.2f"),"Scenario":st.column_config.NumberColumn("Scenario",format="%.2f"),"Context %":st.column_config.NumberColumn("Context %",format="%.1f"),"DFS Lab":st.column_config.NumberColumn("DFS Lab",format="%.2f")})
            st.caption("Neutral means DFS LAB found no reason to move the projection. Detailed model inputs remain available above for auditing; Lineup Lab explains what matters for each lineup.")

        with tabs[4]:
            st.markdown('<div class="card-title">Scenario Engine</div><div class="card-sub">Use a football story, a predicted score, or both. Your game thesis changes projections, correlation and lineup construction. Use the 0–100 influence control to decide how strongly DFS Lab should commit to it.</div>',unsafe_allow_html=True)
            script_options=["Neutral","Auto from score","Shootout","Pass-heavy shootout","Low-scoring game","Defensive / field-goal battle","Ground-and-pound","Team wins close","Team dominates","Team plays from ahead","Team passing comeback"]
            script=st.selectbox("Game script",script_options,key="sd_script")
            use_score=st.toggle("Use predicted score to adjust projections",key="sd_use_score")
            team_scores={}
            score_profile={"total":0,"margin":0,"winner":"","loser":""}
            if len(teams)>=2:
                sc1,sc2=st.columns(2)
                with sc1: score0=st.number_input(f"{teams[0]} score",0,70,key="sd_score_0",disabled=not use_score)
                with sc2: score1=st.number_input(f"{teams[1]} score",0,70,key="sd_score_1",disabled=not use_score)
                team_scores={teams[0]:float(score0),teams[1]:float(score1)}
            intensity=st.slider("Scenario influence",min_value=0,max_value=100,value=50,step=5,key="sd_intensity",help="0 = ignore the game thesis. 50 = standard influence. 100 = strongest bounded scenario influence.")
            st.caption(f"Game-thesis influence: {intensity}%")
            directional=script in ["Team wins close","Team dominates","Team plays from ahead","Team passing comeback"]
            script_team=st.selectbox("Script team",["None"]+teams,disabled=(not directional) or script=="Auto from score",key="sd_script_team")
            if script_team=="None": script_team=""

            effective_script=script
            effective_team=script_team
            if use_score and team_scores:
                auto_script,auto_team,score_profile=infer_score_script(team_scores)
                if script=="Auto from score":
                    effective_script=auto_script; effective_team=auto_team
                st.markdown(f"**Score read:** {int(score_profile['total'])} total • {int(score_profile['margin'])}-point margin • **{auto_team}** projected winner • auto shape: **{auto_script}**")
            elif script=="Auto from score":
                effective_script="Neutral"; effective_team=""
                st.warning("Turn on predicted score to use Auto from score.")

            script_text={
                "Neutral":"No directional football-story tilt. Baseline projection, leverage and your player takes drive the build.",
                "Shootout":"Boost both passing games and reduce defense preference.",
                "Pass-heavy shootout":"Stronger QB/WR/TE emphasis, more 3-3 builds and tighter QB/pass-catcher correlation.",
                "Low-scoring game":"Raise RB/K/DST slightly and reduce passing-game projections.",
                "Defensive / field-goal battle":"Strongest K/DST environment; keeps skill-player adjustments conservative.",
                "Ground-and-pound":"Raises RBs and control pieces while reducing pass-game emphasis.",
                "Team wins close":"Mostly balanced 3-3 builds with a mild lean toward the selected team.",
                "Team dominates":"More 4-2/5-1 winner-heavy constructions; winner RB/DST/K rise and trailing pass volume rises.",
                "Team plays from ahead":"Winner rushing/control pieces rise while the opponent QB/WR/TE get catch-up volume.",
                "Team passing comeback":"Selected-team QB/WR/TE rise; opponent RB gets a closing-game boost.",
                "Auto from score":"The entered score chooses the broad game shape automatically."
            }
            st.info(script_text.get(script,script_text["Neutral"]))

            auto_shape=st.toggle("Let the scenario shape lineup construction + correlation",key="sd_auto_shape")
            scenario_df=apply_showdown_scenario(df,effective_script,effective_team,use_score,team_scores,intensity)
            effective_weights,corr_overrides=script_build_adjustments(construction_weights,effective_script,effective_team,use_score,team_scores,auto_shape)
            if auto_shape:
                mix_txt=" • ".join(f"{k} {int(v)}%" for k,v in effective_weights.items())
                st.caption(f"Scenario build mix: {mix_txt}")
                if corr_overrides:
                    st.caption(f"Scenario correlation: QB CPT + {corr_overrides['qb_pc']} pass catcher(s) • WR/TE CPT + QB {corr_overrides['wrte_qb']}% • RB CPT + DST/K {corr_overrides['rb_ctrl']}%")

            st.markdown("#### Projection movement")
            preview=scenario_df[["Name","Position","Team","My Proj","Script Proj","Proj Change %"]].copy()
            preview.columns=["Player","Pos","Team","Base","Scenario","Change %"]
            preview=preview.sort_values("Change %",key=lambda x:x.abs(),ascending=False).head(14)
            st.dataframe(preview,hide_index=True,use_container_width=True,height=410,column_config={"Player":st.column_config.TextColumn("Player",pinned=True,width=190),"Base":st.column_config.NumberColumn("Base",format="%.2f"),"Scenario":st.column_config.NumberColumn("Scenario",format="%.2f"),"Change %":st.column_config.NumberColumn("Change %",format="%.1f")})
            st.caption("These are bounded scenario tilts applied to the DFS Lab baseline projections. They are not a claim that a final score can precisely predict individual fantasy points.")
            st.markdown("#### How V6.3 grades Showdown")
            st.caption("Projection 29–34% • Captain quality 18% • correlation 20% • leverage 11–15% • duplication proxy 10–15% • your takes 7%. The exact weights move with contest size/payout.")
            st.caption("The scenario engine changes the projection and construction inputs before the lineup is graded; it does not simply add points to the final grade.")

        # Defaults exist even before the user opens tabs because Streamlit executes all tab bodies.
        strategy_map=st.session_state["showdown_strategy"]
        # Apply the scenario to the actual build, not just the preview.
        build_df=apply_showdown_scenario(df,effective_script,effective_team,use_score,team_scores,intensity)
        build_df=apply_context_engine(build_df,st.session_state.get("showdown_context",{}),st.session_state.get("context_strength","Standard"))
        build_df=apply_projection_overrides(build_df,st.session_state.get("projection_overrides",{}))
        build_weights,corr_overrides=script_build_adjustments(construction_weights,effective_script,effective_team,use_score,team_scores,auto_shape)
        eff_qb_pc=cpt_qb_pc; eff_wrte_qb=wrte_qb; eff_rb_ctrl=rb_ctrl
        if corr_overrides:
            eff_qb_pc=corr_overrides["qb_pc"]; eff_wrte_qb=corr_overrides["wrte_qb"]; eff_rb_ctrl=corr_overrides["rb_ctrl"]
        active_relationships=list(st.session_state.get("showdown_relationships",[]))
        if st.session_state.get("max_one_rb_team",False):
            for t in teams:
                # Two identical team/RB sides would be ignored by pairwise self matches,
                # so create pair-specific rules for every RB combination on that team.
                rb_rows=build_df[build_df["Team"].eq(t) & build_df["Position"].eq("RB")]
                rb_ids=rb_rows["ID"].astype(str).tolist()
                for x in range(len(rb_ids)):
                    for y in range(x+1,len(rb_ids)):
                        active_relationships.append({"enabled":True,"rule":"Never Together","a":{"kind":"Player","id":rb_ids[x]},"b":{"kind":"Player","id":rb_ids[y]}})
        if build_btn:
            result=generate_showdown_lineups(build_df,field_size,payout_style,lineup_count,max(350,lineup_count*15),min_salary,max_salary,build_weights,effective_script,effective_team,strategy_map,entry_format,eff_qb_pc,eff_wrte_qb,eff_rb_ctrl,max_k,max_dst,min_unique,seed,relationship_rules=active_relationships,world_influence=intensity)
            st.session_state["showdown_result_v4"]=result
            if result is None or result.empty:
                st.error("No legal lineup found. Check minimum salary, locks/outs, Captain eligibility, allowed team builds, and Captain-pairing rules. Try one change at a time; DFS Lab will preserve your player settings.")
            elif len(result) < lineup_count:
                st.warning(f"Built {len(result)} of {lineup_count} requested lineups. The current salary, uniqueness, locks or exposure limits are restricting the pool.")
        result=st.session_state.get("showdown_result_v4")

        with tabs[5]:
            st.markdown('<div class="card-title">Rated Lineups</div><div class="card-sub">The grade is portfolio-relative. A+ means one of the strongest lineups in this build — not a guarantee of outcome.</div>',unsafe_allow_html=True)
            if result is None or result.empty: st.info("Set your build, player takes and script, then generate lineups.")
            else:
                m1,m2,m3,m4=st.columns(4); m1.metric("A / A+",int(result["Rating"].isin(["A","A+"]).sum())); m2.metric("Top projection",f"{result['Projection'].max():.1f}"); m3.metric("Avg salary left",f"${int(result['Salary Left'].mean()):,}"); m4.metric("Built",len(result))
                st.markdown("#### Explore game worlds")
                world_counts=result["Game World"].value_counts().rename_axis("World").reset_index(name="Lineups") if "Game World" in result.columns else pd.DataFrame()
                world_filter="All worlds"
                if not world_counts.empty:
                    world_counts["Share %"]=(100*world_counts["Lineups"]/len(result)).round(1)
                    world_options=["All worlds"]+[f"{r['World']} · {int(r['Lineups'])} lineups" for _,r in world_counts.iterrows()]
                    world_pick=st.selectbox("Highlight a world",world_options,key="world_explorer_pick")
                    if world_pick!="All worlds": world_filter=world_pick.rsplit(" · ",1)[0]
                    wc=world_counts if world_filter=="All worlds" else world_counts[world_counts["World"].eq(world_filter)]
                    st.dataframe(wc,hide_index=True,use_container_width=True,height=min(330,42+35*len(wc)))
                filtered_result=result if world_filter=="All worlds" else result[result["Game World"].astype(str).eq(world_filter)]
                if world_filter!="All worlds" and not filtered_result.empty:
                    w1,w2,w3=st.columns(3); w1.metric("Lineups in world",len(filtered_result)); w2.metric("Top projection",f"{filtered_result['Projection'].max():.1f}"); w3.metric("Avg salary left",f"${int(filtered_result['Salary Left'].mean()):,}")
                    st.caption(str(filtered_result.iloc[0].get("World Thesis","")))
                st.markdown("#### Lineup Explorer")
                st.caption("Highlight a lineup and DFS LAB will analyze all six players together — not just the Captain.")
                lineup_choices=[f"#{int(r['Rank'])} · {r['Captain']} CPT · {r['Construction']} · {r['Projection']:.1f} pts" for _,r in filtered_result.iterrows()]
                pick=st.selectbox("Analyze lineup",lineup_choices,key="lineup_lab_pick")
                li=lineup_choices.index(pick); lr=filtered_result.iloc[li]
                risk = "lower" if entry_format=="Single Entry" else ("moderate" if entry_format in ["3-Max","20-Max"] else "higher")
                contest_reason=(f"{entry_format} with {int(field_size):,} entries. DFS Lab uses {risk} tolerance for fragile salary-relief plays and weights tournament ceiling/correlation accordingly.")
                st.markdown(f"**Why this lineup exists**  \n**Contest:** {entry_format} · {int(field_size):,} entries · {payout_style}  \n**Game thesis:** {effective_script} · influence {int(intensity)}%  \n**Game world:** {lr.get('Game World',effective_script)}  \n**Construction:** {lr['Construction']} · **Captain:** {lr['Captain']}  \n**Projection:** {lr['Projection']:.2f} · **Salary left:** ${int(lr['Salary Left']):,}")
                st.write(contest_reason)
                st.write(f"**World thesis:** {lr.get('World Thesis','')}")
                roster_names=[str(lr['Captain'])]+[str(lr.get('FLEX'+str(i),'')) for i in range(1,6)]
                roster_names=[x for x in roster_names if x]
                detail_rows=[]
                for j,nm in enumerate(roster_names):
                    pr=build_df[build_df['Name'].astype(str).eq(nm)]
                    if pr.empty: continue
                    pr=pr.iloc[0]; role="Captain / ceiling engine" if j==0 else ("Primary projection piece" if float(pr.get('DFS Lab Proj',0))>=12 else "Salary relief / secondary path")
                    detail_rows.append({"Slot":"CPT" if j==0 else f"FLEX {j}","Player":nm,"Pos":pr.get('Position',''),"Team":pr.get('Team',''),"DFS Lab":round(float(pr.get('DFS Lab Proj',0)),2),"Salary":int(pr.get('FlexSalary',0)) if j else int(pr.get('CPTSalary',0)),"Purpose":role})
                if detail_rows:
                    st.dataframe(pd.DataFrame(detail_rows),hide_index=True,use_container_width=True,column_config={"Player":st.column_config.TextColumn("Player",pinned=True),"Salary":st.column_config.NumberColumn("Salary",format="$%d")})
                    low=min(detail_rows,key=lambda x:x["DFS Lab"])
                    st.write(f"**Weakest projection link:** {low['Player']} ({low['DFS Lab']:.2f}). DFS LAB is using this spot as {low['Purpose'].lower()} within the six-player construction.")
                st.markdown("##### Ask DFS LAB")
                ai_q=st.text_input("Ask about this lineup or world",placeholder="Why is this player here? What kills this lineup? What if BUF wins by 14?",key="dfs_lab_question")
                if ai_q:
                    q=ai_q.lower()
                    if "kill" in q or "fail" in q or "break" in q:
                        st.info(f"This lineup is most vulnerable if the {lr.get('Game World',effective_script)} thesis fails, if {lr['Captain']} does not reach a Captain-level ceiling, or if its lowest-projection salary-relief piece contributes almost nothing.")
                    elif "why" in q and ("lineup" in q or "built" in q):
                        st.info(f"DFS LAB built this six-player unit for {lr.get('Game World',effective_script)}: {lr.get('World Thesis','')} It preserves a {lr['Construction']} construction with {lr['Captain']} at CPT while balancing projection, salary, correlation and the {entry_format} contest profile.")
                    elif "world" in q:
                        st.info(f"This lineup belongs to {lr.get('Game World',effective_script)}. {lr.get('World Thesis','')} The current portfolio contains {len(filtered_result) if world_filter!='All worlds' else int((result['Game World']==lr.get('Game World')).sum())} lineup(s) in that world.")
                    else:
                        st.info("DFS LAB has the active lineup, world, contest and player pool in context. In this first mini-AI layer, use questions about why the lineup exists, what could make it fail, or its game world. V6.4 will add simulation evidence and natural-language re-simulation.")
                st.write(f"DFS Lab selected **{lr['Captain']} at Captain** while preserving the {lr['Construction']} game construction because this combination ranked strongly under the current projection, correlation, salary and contest-risk settings. {lr.get('Strategy Notes','')}")
                if float(lr.get('Scenario Delta',0))!=0:
                    st.write(f"Your game thesis moved this lineup by **{float(lr['Scenario Delta']):+.2f} projected DK points** versus the unadjusted baseline.")
                st.markdown("##### Challenge this lineup")
                st.caption("Choose a player you would rather use. DFS Lab will show the salary/projection contrast. Full conversational AI control comes after the optimizer evidence layer is validated.")
                out_player=st.selectbox("Replace",roster_names,key="lab_out")
                pool_names=[x for x in build_df['Name'].astype(str).tolist() if x not in roster_names]
                in_player=st.selectbox("With",pool_names,key="lab_in")
                if out_player and in_player:
                    po=build_df[build_df['Name'].astype(str).eq(out_player)].iloc[0]; pi=build_df[build_df['Name'].astype(str).eq(in_player)].iloc[0]
                    dproj=float(pi['DFS Lab Proj'])-float(po['DFS Lab Proj']); dsal=int(pi['FlexSalary'])-int(po['FlexSalary'])
                    legal = int(lr['Salary']) + dsal <= 50000
                    st.write(f"**Direct swap:** projection {dproj:+.2f} · salary {dsal:+,} · {'salary-cap legal' if legal else 'over the salary cap — a second change would be required'}. This is a direct contrast, not yet a full re-optimization.")
                st.markdown("#### Top lineup cards")
                for _,lr in result.head(5).iterrows():
                    flex_names=" · ".join(str(lr.get("FLEX"+str(i),"")) for i in range(1,6))
                    card_html=("<div class='lineup-card'><div class='lineup-rank'>#%s &nbsp; <span>%s</span></div>" % (int(lr["Rank"]),lr["Rating"]) +
                               "<div class='lineup-cpt'>CPT · %s</div>" % lr["Captain"] +
                               "<div class='lineup-flex'>%s</div>" % flex_names +
                               "<div class='lineup-meta'>Proj %.1f &nbsp; • &nbsp; $%s &nbsp; • &nbsp; %s &nbsp; • &nbsp; %s duplication risk</div></div>" % (lr["Projection"],f"{int(lr['Salary']):,}",lr["Construction"],lr["Dup Risk"]))
                    st.markdown(card_html,unsafe_allow_html=True)
                st.markdown("#### Full lineup table")
                cols=["Rank","Rating","Rating Score","Projection","Salary","Salary Left","Captain","Captain Pos","CPT Own","Construction","Dup Risk","Projection Grade","Captain Grade","Correlation Grade","Leverage Grade","Duplication Grade","Story","CPT","FLEX1","FLEX2","FLEX3","FLEX4","FLEX5"]
                st.dataframe(result[[c for c in cols if c in result.columns]],hide_index=True,use_container_width=True,height=610)
                d1,d2=st.columns(2)
                with d1: st.download_button("Download analysis CSV",result.to_csv(index=False),"showdown_lineups_v6_3_1.csv","text/csv",use_container_width=True)
                with d2: st.download_button("Download DK-format lineup CSV",showdown_upload_csv(result),"showdown_dk_upload_v6_3_1.csv","text/csv",use_container_width=True)
                pick=st.number_input("Inspect lineup rank",min_value=1,max_value=len(result),value=1,step=1)
                r=result.iloc[int(pick)-1]
                st.write(f"**{r['Rating']} ({r['Rating Score']})** — {r['Story']}")
                st.caption(f"Projection {r['Projection']} • Salary ${int(r['Salary']):,} • ${int(r['Salary Left']):,} left • Duplication risk {r['Dup Risk']} • {r['Strategy Notes']}")

        with tabs[6]:
            st.markdown('<div class="card-title">Exposure Lab</div><div class="card-sub">See overall and Captain exposure side by side.</div>',unsafe_allow_html=True)
            if result is None or result.empty: st.info("Generate lineups first.")
            else:
                exp=showdown_exposure_table(df,result,strategy_map)
                st.markdown("#### Set player exposure")
                st.caption("iPad-friendly controls. Type a percentage or use +/−. Minimums are enforced across the requested portfolio when feasible.")
                exp_names=exp["Name"].astype(str).tolist()
                ep=st.selectbox("Player",exp_names,key="exposure_player_select")
                erow=df[df["Name"].astype(str).eq(ep)].iloc[0]; epid=str(erow["ID"])
                estrat=st.session_state["showdown_strategy"].setdefault(epid,{})
                actual=float(exp.loc[exp["Name"].eq(ep),"Actual %"].iloc[0]); cactual=float(exp.loc[exp["Name"].eq(ep),"CPT Actual %"].iloc[0])
                st.caption(f"Current portfolio: {actual:.0f}% overall • {cactual:.0f}% Captain")
                e1,e2,e3,e4=st.columns(4)
                with e1: emin=st.number_input("Min %",0,100,int(estrat.get("Min Exposure",0)),5,key=f"emin_{epid}")
                with e2: emax=st.number_input("Max %",0,100,int(estrat.get("Max Exposure",100)),5,key=f"emax_{epid}")
                with e3: cmin=st.number_input("CPT Min %",0,100,int(estrat.get("CPT Min",0)),5,key=f"ecmin_{epid}")
                with e4: cmax=st.number_input("CPT Max %",0,100,int(estrat.get("CPT Max",100)),5,key=f"ecmax_{epid}")
                if emin>emax or cmin>cmax:
                    st.warning("Minimum exposure cannot be higher than maximum exposure.")
                else:
                    estrat.update({"Min Exposure":float(emin),"Max Exposure":float(emax),"CPT Min":float(cmin),"CPT Max":float(cmax)})
                st.markdown("#### Portfolio exposure")
                st.dataframe(exp,hide_index=True,use_container_width=True,height=560,column_config={"Name":st.column_config.TextColumn("Player",pinned=True,width=190)})
                st.download_button("Download exposure CSV",exp.to_csv(index=False),"showdown_exposure_v6_1.csv","text/csv",use_container_width=True)
    except Exception as e:
        st.error(f"Showdown build error: {e}")

st.caption("DFS LAB • V6.3.2 • Design System • Game Worlds • Lineup Explorer")
