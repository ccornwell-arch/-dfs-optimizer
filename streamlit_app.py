
import io
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

st.set_page_config(page_title="DFS Tournament Optimizer", page_icon="🏈", layout="wide")

ROSTER_SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]

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
        raise ValueError("Could not locate the DraftKings player header.")

    idx = {name: i for i, name in enumerate(header)}
    for row in reader:
        if not row or len(row) <= max(idx.values()):
            continue
        name = row[idx["Name"]].strip()
        if not name:
            continue
        try:
            salary = int(float(row[idx["Salary"]]))
        except:
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

def parse_opponent(game_info, team):
    matchup = str(game_info).split()[0]
    if "@" not in matchup:
        return ""
    away, home = matchup.split("@", 1)
    return home if team == away else away

def prepare_player_pool(dk_file, ss_file):
    dk = load_dk_template(dk_file)
    ss = pd.read_csv(ss_file)

    needed = {"Name", "My Proj", "My Own"}
    missing = needed - set(ss.columns)
    if missing:
        raise ValueError(f"SaberSim file is missing columns: {sorted(missing)}")

    ss = ss[["Name", "My Proj", "My Own"]].copy()
    ss["My Proj"] = pd.to_numeric(ss["My Proj"], errors="coerce").fillna(0)
    ss["My Own"] = pd.to_numeric(ss["My Own"], errors="coerce").fillna(0)

    df = dk.merge(ss, on="Name", how="left")
    df["My Proj"] = df["My Proj"].fillna(0)
    df["My Own"] = df["My Own"].fillna(0)
    df["Opponent"] = [parse_opponent(g, t) for g, t in zip(df["Game Info"], df["Team"])]
    df["ActiveForBuild"] = (df["My Proj"] > 0.05) & (df["Salary"] > 0)

    df["is_QB"] = df["Roster Position"].str.contains(r"\bQB\b", regex=True)
    df["is_RB"] = df["Roster Position"].str.contains(r"\bRB\b", regex=True)
    df["is_WR"] = df["Roster Position"].str.contains(r"\bWR\b", regex=True)
    df["is_TE"] = df["Roster Position"].str.contains(r"\bTE\b", regex=True)
    df["is_DST"] = df["Position"].eq("DST")
    df["is_FLEX"] = df["Roster Position"].str.contains(r"\bFLEX\b", regex=True)

    return df.reset_index(drop=True)

def aggression(field_size, payout_style):
    fs = max(2, int(field_size))
    raw = (math.log10(fs) - 2.0) / 4.0
    raw = min(1.0, max(0.10, raw))
    if payout_style == "Winner take all":
        raw = min(1.0, raw + 0.12)
    elif payout_style == "Flatter payouts":
        raw = max(0.05, raw - 0.10)
    return raw

def player_base_score(df, aggr):
    proj = df["My Proj"].to_numpy(float)
    own = np.clip(df["My Own"].to_numpy(float), 0.05, None)
    leverage = np.log((proj + 2.0) / (own + 2.0))
    return proj + (0.75 + 2.25 * aggr) * leverage

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

def solve_one(df, aggr, rng, min_salary, qb_stack_min, require_bringback, noise_scale):
    n = len(df)
    s = len(ROSTER_SLOTS)
    total_vars = n * s
    elig = slot_eligibility(df)
    active = df["ActiveForBuild"].to_numpy(bool)

    base = player_base_score(df, aggr)
    noise = rng.normal(0, noise_scale, size=n)
    randomized = base * np.exp(noise)

    c = np.zeros(total_vars)
    integrality = np.ones(total_vars)
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)

    def vidx(i, j):
        return i * s + j

    for i in range(n):
        for j, slot in enumerate(ROSTER_SLOTS):
            c[vidx(i, j)] = -randomized[i]
            if (not active[i]) or (not elig[slot][i]):
                ub[vidx(i, j)] = 0

    rows, lows, highs = [], [], []

    for j in range(s):
        coeff = {vidx(i, j): 1.0 for i in range(n)}
        rows.append(coeff); lows.append(1.0); highs.append(1.0)

    for i in range(n):
        coeff = {vidx(i, j): 1.0 for j in range(s)}
        rows.append(coeff); lows.append(0.0); highs.append(1.0)

    coeff = {}
    for i in range(n):
        for j in range(s):
            coeff[vidx(i, j)] = float(df.loc[i, "Salary"])
    rows.append(coeff); lows.append(float(min_salary)); highs.append(50000.0)

    for q in df.index[df["is_QB"] & df["ActiveForBuild"]]:
        team = df.loc[q, "Team"]
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

        if require_bringback:
            opp = df.loc[q, "Opponent"]
            opp_skill = df.index[
                df["ActiveForBuild"] & (df["Team"] == opp) &
                (df["is_RB"] | df["is_WR"] | df["is_TE"])
            ].tolist()
            if opp_skill:
                coeff = {}
                for i in opp_skill:
                    for j in range(s):
                        if elig[ROSTER_SLOTS[j]][i]:
                            coeff[vidx(i, j)] = coeff.get(vidx(i, j), 0.0) + 1.0
                for j in range(s):
                    if elig[ROSTER_SLOTS[j]][q]:
                        coeff[vidx(q, j)] = coeff.get(vidx(q, j), 0.0) - 1.0
                rows.append(coeff); lows.append(0.0); highs.append(np.inf)

    for q in df.index[df["is_QB"] & df["ActiveForBuild"]]:
        opp = df.loc[q, "Opponent"]
        for d in df.index[df["is_DST"] & (df["Team"] == opp)]:
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

def lineup_metrics(df, chosen, aggr):
    idxs = [i for _, i in chosen]
    p = df.loc[idxs].copy()
    projection = float(p["My Proj"].sum())
    salary = int(p["Salary"].sum())
    total_own = float(p["My Own"].sum())
    avg_own = total_own / 9

    qb_row = p[p["is_QB"]].iloc[0]
    qb_team = qb_row["Team"]
    qb_opp = qb_row["Opponent"]
    qb_game = qb_row["Game Info"].split()[0]

    stack_count = len(p[(p["Team"] == qb_team) & (p["is_WR"] | p["is_TE"])])
    bringback_count = len(p[(p["Team"] == qb_opp) & (p["is_RB"] | p["is_WR"] | p["is_TE"])])
    rb_dst = len(set(p.loc[p["is_RB"], "Team"]) & set(p.loc[p["is_DST"], "Team"]))
    game_players = int(p["Game Info"].str.startswith(qb_game).sum())

    correlation_bonus = (
        1.2 * min(stack_count, 2)
        + 0.7 * min(bringback_count, 1)
        + 0.45 * rb_dst
        + 0.25 * max(0, game_players - 2)
    )
    leverage_bonus = aggr * max(0.0, (18.0 - avg_own)) * 0.22
    salary_left = 50000 - salary
    salary_uniqueness = aggr * min(salary_left, 2500) / 1000.0 * 0.18

    tournament_score = projection + correlation_bonus + leverage_bonus + salary_uniqueness

    return {
        "Projection": round(projection, 2),
        "Salary": salary,
        "Salary Left": salary_left,
        "Total Own": round(total_own, 1),
        "Avg Own": round(avg_own, 1),
        "QB Stack": stack_count,
        "Bring-backs": bringback_count,
        "QB Game Players": game_players,
        "RB+DST": rb_dst,
        "Tournament Score": round(tournament_score, 2),
    }

def generate(df, field_size, payout_style, count, attempts, min_salary, qb_stack_min, bringback, seed):
    aggr = aggression(field_size, payout_style)
    rng = np.random.default_rng(seed)
    seen = set()
    rows = []

    progress = st.progress(0, text="Generating lineups...")
    for attempt in range(attempts):
        if len(rows) >= count:
            break
        chosen = solve_one(
            df, aggr, rng, min_salary, qb_stack_min, bringback,
            noise_scale=0.14 + 0.10 * aggr
        )
        if not chosen:
            continue

        ids = tuple(sorted(str(df.loc[i, "ID"]) for _, i in chosen))
        if ids in seen:
            continue
        seen.add(ids)

        metrics = lineup_metrics(df, chosen, aggr)
        row = dict(metrics)
        for slot, i in chosen:
            row[slot] = df.loc[i, "Name"]
            row[slot + "_ID"] = str(df.loc[i, "ID"])
        rows.append(row)

        if attempt % 10 == 0:
            progress.progress(min(1.0, len(rows) / max(1, count)), text=f"Generated {len(rows)} / {count}")

    progress.empty()

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["Tournament Score", "Projection"], ascending=[False, False]).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    return out

st.title("🏈 DFS Tournament Optimizer")
st.caption("DraftKings NFL Classic — Version 1")

with st.sidebar:
    st.header("Contest settings")
    field_size = st.number_input("Field size", min_value=2, value=6507, step=1)
    payout_style = st.selectbox("Payout style", ["GPP / top-heavy", "Winner take all", "Flatter payouts"])
    min_salary = st.slider("Minimum salary", 44000, 50000, 47500, 100)
    qb_stack = st.selectbox("QB stack", [1, 2], index=1, help="Number of same-team WR/TE players required with the QB")
    bringback = st.checkbox("Require opponent bring-back", value=False)
    lineup_count = st.slider("Lineups to generate", 25, 500, 100, 25)
    seed = st.number_input("Random seed", min_value=1, value=42, step=1)

st.subheader("1. Upload your files")
c1, c2 = st.columns(2)
with c1:
    dk_file = st.file_uploader("DraftKings salary CSV", type=["csv"], key="dk")
with c2:
    ss_file = st.file_uploader("SaberSim projections CSV", type=["csv"], key="ss")

if dk_file and ss_file:
    try:
        df = prepare_player_pool(dk_file, ss_file)
        matched = (df["My Proj"] > 0).sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("DraftKings players", len(df))
        m2.metric("Players with projections", matched)
        m3.metric("Contest aggression", f"{aggression(field_size, payout_style):.2f}")

        with st.expander("Preview player pool"):
            st.dataframe(
                df[["Name", "Position", "Team", "Opponent", "Salary", "My Proj", "My Own"]]
                .sort_values("My Proj", ascending=False)
                .head(50),
                use_container_width=True
            )

        st.subheader("2. Generate lineups")
        if st.button("Generate Lineups", type="primary", use_container_width=True):
            with st.spinner("Running optimizer..."):
                result = generate(
                    df=df,
                    field_size=field_size,
                    payout_style=payout_style,
                    count=lineup_count,
                    attempts=lineup_count * 8,
                    min_salary=min_salary,
                    qb_stack_min=qb_stack,
                    bringback=bringback,
                    seed=seed,
                )

            if result.empty:
                st.error("No lineups were generated. Try lowering minimum salary or loosening stack rules.")
            else:
                st.session_state["result"] = result
                st.success(f"Generated {len(result)} unique lineups.")

    except Exception as e:
        st.error(str(e))

if "result" in st.session_state:
    result = st.session_state["result"]
    st.subheader("3. Ranked lineups")

    display_cols = [
        "Rank", "Tournament Score", "Projection", "Salary", "Salary Left",
        "Total Own", "QB Stack", "Bring-backs",
        "QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"
    ]
    st.dataframe(result[display_cols], use_container_width=True, height=560)

    csv_bytes = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download ranked lineups CSV",
        data=csv_bytes,
        file_name="ranked_lineups.csv",
        mime="text/csv",
        use_container_width=True,
    )

    upload_rows = []
    for _, r in result.head(20).iterrows():
        upload_rows.append({
            "QB": r["QB_ID"],
            "RB": r["RB1_ID"],
            "RB.1": r["RB2_ID"],
            "WR": r["WR1_ID"],
            "WR.1": r["WR2_ID"],
            "WR.2": r["WR3_ID"],
            "TE": r["TE_ID"],
            "FLEX": r["FLEX_ID"],
            "DST": r["DST_ID"],
        })
    upload_df = pd.DataFrame(upload_rows)
    st.download_button(
        "Download DraftKings-ID file (top 20)",
        data=upload_df.to_csv(index=False).encode("utf-8"),
        file_name="upload_ready_top20.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()
st.caption("V1 uses projection, ownership, salary, stacking and contest size. It does not yet include real ceiling, Vegas, weather, injury or coverage data.")
