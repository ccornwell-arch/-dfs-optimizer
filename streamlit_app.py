
import csv
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

st.set_page_config(page_title="DFS Tournament Builder V2.2", page_icon="🏈", layout="wide")

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


def player_objective(df, aggr, strategy_map, preferred_stack_teams, team_strategy_map):
    proj = df["My Proj"].to_numpy(float)
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
    no_dst_from_qb_game=True, no_offense_vs_dst=False,
    noise_scale=0.16
):
    n = len(df)
    s = len(ROSTER_SLOTS)
    total_vars = n * s
    elig = slot_eligibility(df)
    active = df["ActiveForBuild"].to_numpy(bool)

    base = player_objective(df, aggr, strategy_map, preferred_stack_teams, team_strategy_map)
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
        excluded = strat.get("Priority") == "Exclude"
        locked = bool(strat.get("Lock", False))

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
        if bool(strat.get("Lock", False)):
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


# -----------------------------
# UI
# -----------------------------

st.title("🏈 DFS Tournament Builder V2.2")
st.caption("Your football opinions first. The optimizer builds around them, then rates the lineups.")

with st.sidebar:
    st.header("Contest")
    preset = st.selectbox(
        "Contest preset",
        ["Small-field WTA", "Small-field single entry", "Medium-field GPP", "Large-field GPP", "Custom"]
    )

    preset_defaults = {
        "Small-field WTA": (500, "Winner take all", 48000, 2),
        "Small-field single entry": (1500, "GPP / top-heavy", 48000, 2),
        "Medium-field GPP": (10000, "GPP / top-heavy", 47500, 2),
        "Large-field GPP": (250000, "GPP / top-heavy", 46500, 2),
        "Custom": (6500, "GPP / top-heavy", 47500, 2),
    }
    d_field, d_payout, d_salary, d_stack = preset_defaults[preset]

    field_size = st.number_input("Field size", min_value=2, value=int(d_field), step=1)
    payout_style = st.selectbox(
        "Payout style",
        ["GPP / top-heavy", "Winner take all", "Flatter payouts"],
        index=["GPP / top-heavy", "Winner take all", "Flatter payouts"].index(d_payout)
    )
    min_salary = st.slider("Minimum salary", 44000, 50000, int(d_salary), 100)
    qb_stack = st.selectbox("QB pass catchers", [1, 2], index=1 if d_stack == 2 else 0)
    bringback_mode = st.selectbox("Opponent bring-back", ["Optional", "Required", "None"])
    lineup_count = st.slider("Lineups to generate", 25, 500, 100, 25)
    seed = st.number_input("Random seed", min_value=1, value=42, step=1)

st.subheader("1. Upload this week's files")
c1, c2 = st.columns(2)
with c1:
    dk_file = st.file_uploader("DraftKings salary CSV", type=["csv"], key="dk")
with c2:
    ss_file = st.file_uploader("SaberSim projections + ownership CSV", type=["csv"], key="ss")

if dk_file and ss_file:
    try:
        df = prepare_player_pool(dk_file, ss_file)

        st.subheader("2. Choose the stacks you want")
        teams = sorted(df["Team"].dropna().unique().tolist())
        preferred_stack_teams = st.multiselect(
            "Preferred QB stack teams",
            teams,
            help="If you select teams here, generated lineups will use a QB from one of these teams."
        )

        st.markdown("**Team priorities**")
        team_priority_df = pd.DataFrame({
            "Team": teams,
            "Priority": ["Neutral"] * len(teams),
        })
        if "team_strategy_master" not in st.session_state:
            st.session_state["team_strategy_master"] = {}
        for idx, r in team_priority_df.iterrows():
            team_priority_df.at[idx, "Priority"] = st.session_state["team_strategy_master"].get(r["Team"], "Neutral")

        team_priority_edit = st.data_editor(
            team_priority_df,
            hide_index=True,
            use_container_width=True,
            disabled=["Team"],
            column_config={
                "Priority": st.column_config.SelectboxColumn(
                    "Priority",
                    options=["Core", "Like", "Neutral", "Fade", "Exclude"]
                )
            },
            key="team_priority_editor",
        )
        for _, r in team_priority_edit.iterrows():
            st.session_state["team_strategy_master"][r["Team"]] = r["Priority"]
        team_strategy_map = st.session_state["team_strategy_master"]

        st.markdown("**Conditional rules**")
        no_dst_from_qb_game = st.checkbox(
            "If I stack a game, do not use either defense from that game",
            value=True,
            help="Example: Herbert + LAC pass catchers means no Chargers DST and no Cardinals DST."
        )
        no_offense_vs_dst = st.checkbox(
            "Do not use offensive players against my selected DST",
            value=False,
            help="Stronger rule. If Jets DST is used, no Titans offensive player can appear."
        )

        st.subheader("3. Set your player opinions")
        st.caption("Lock = must use. Core/Like increases priority. Fade reduces priority. Exclude removes the player.")

        f1, f2 = st.columns([1, 1])
        with f1:
            team_filter = st.multiselect("Filter teams", teams)
        with f2:
            pos_filter = st.multiselect("Filter positions", ["QB", "RB", "WR", "TE", "DST"])

        view = df.copy()
        if team_filter:
            view = view[view["Team"].isin(team_filter)]
        if pos_filter:
            view = view[view["Position"].isin(pos_filter)]

        base_strategy = pd.DataFrame({
            "ID": view["ID"].astype(str),
            "Name": view["Name"],
            "Pos": view["Position"],
            "Team": view["Team"],
            "Salary": view["Salary"],
            "Proj": view["My Proj"].round(2),
            "Own": view["My Own"].round(1),
            "Lock": False,
            "Priority": "Neutral",
            "Max Exposure": 100,
        })

        # Persist edits across reruns by merging existing session strategy.
        if "strategy_master" not in st.session_state:
            st.session_state["strategy_master"] = {}

        for idx, r in base_strategy.iterrows():
            existing = st.session_state["strategy_master"].get(str(r["ID"]))
            if existing:
                base_strategy.at[idx, "Lock"] = existing.get("Lock", False)
                base_strategy.at[idx, "Priority"] = existing.get("Priority", "Neutral")
                base_strategy.at[idx, "Max Exposure"] = existing.get("Max Exposure", 100)

        edited = st.data_editor(
            base_strategy,
            hide_index=True,
            use_container_width=True,
            height=520,
            disabled=["ID", "Name", "Pos", "Team", "Salary", "Proj", "Own"],
            column_config={
                "Priority": st.column_config.SelectboxColumn("Priority", options=PRIORITY_OPTIONS),
                "Lock": st.column_config.CheckboxColumn("Lock"),
                "Max Exposure": st.column_config.NumberColumn("Max %", min_value=0, max_value=100, step=5),
            },
            key="player_strategy_editor",
        )

        # Save visible edits into master map.
        for _, r in edited.iterrows():
            st.session_state["strategy_master"][str(r["ID"])] = {
                "Lock": bool(r["Lock"]),
                "Priority": str(r["Priority"]),
                "Max Exposure": float(r["Max Exposure"]),
            }

        strategy_map = st.session_state["strategy_master"]

        # Quick summary
        counts = defaultdict(int)
        locks = 0
        for v in strategy_map.values():
            counts[v.get("Priority", "Neutral")] += 1
            locks += int(bool(v.get("Lock", False)))
        st.caption(
            f"Locks: {locks} | Core: {counts['Core']} | Like: {counts['Like']} | "
            f"Fade: {counts['Fade']} | Exclude: {counts['Exclude']}"
        )

        st.subheader("4. Generate and rate lineups")
        if st.button("Generate Rated Lineups", type="primary", use_container_width=True):
            with st.spinner("Building around your player and stack preferences..."):
                result = generate_lineups(
                    df=df,
                    field_size=field_size,
                    payout_style=payout_style,
                    count=lineup_count,
                    attempts=lineup_count * 10,
                    min_salary=min_salary,
                    qb_stack_min=qb_stack,
                    bringback_mode=bringback_mode,
                    preferred_stack_teams=preferred_stack_teams,
                    strategy_map=strategy_map,
                    team_strategy_map=team_strategy_map,
                    no_dst_from_qb_game=no_dst_from_qb_game,
                    no_offense_vs_dst=no_offense_vs_dst,
                    seed=seed,
                )

            if result.empty:
                st.error("No valid lineups were produced. Loosen locks, exclusions, salary, stack or exposure rules.")
            else:
                st.session_state["v2_result"] = result
                st.success(f"Generated {len(result)} unique lineups.")

    except Exception as e:
        st.error(str(e))

if "v2_result" in st.session_state:
    result = st.session_state["v2_result"]

    st.subheader("5. Your rated lineup pool")
    st.caption("Ratings are relative to the lineups this build generated. They balance projection, correlation, leverage and your own preferences.")

    display_cols = [
        "Rank", "Rating", "Projection Grade", "Correlation Grade", "Leverage Grade", "User Fit Grade",
        "Projection", "Salary", "Salary Left", "Total Own",
        "Stack Summary", "Strategy Notes",
        "QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"
    ]
    st.dataframe(result[display_cols], use_container_width=True, height=620)

    st.subheader("Inspect a lineup")
    rank_choice = st.selectbox("Choose rank", result["Rank"].tolist())
    r = result[result["Rank"] == rank_choice].iloc[0]

    a, b, c, d = st.columns(4)
    a.metric("Overall", r["Rating"])
    b.metric("Projection", r["Projection"])
    c.metric("Total own", f"{r['Total Own']}%")
    d.metric("Salary left", f"${int(r['Salary Left']):,}")

    st.write(f"**Stack:** {r['Stack Summary']}")
    st.write(
        f"**Grades:** Projection — {r['Projection Grade']} | "
        f"Correlation — {r['Correlation Grade']} | "
        f"Leverage — {r['Leverage Grade']} | "
        f"Your strategy — {r['User Fit Grade']}"
    )

    st.download_button(
        "Download rated lineups CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="rated_lineups_v2.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()
st.caption(
    "V2 uses your stack choices, player priorities, locks/excludes, projections, ownership and contest size. "
    "It still does not claim to know true ceiling or duplication until those data sources are added."
)
