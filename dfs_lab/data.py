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


from dfs_lab.common import _first_existing
from dfs_lab.leverage import compute_mobile_qb_tags
from dfs_lab.showdown import configure_game_worlds

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

def apply_football_reality_guard(df, salary_col, projection_col, active_col="ActiveForBuild"):
    """Shared NFL reality layer for Classic and Showdown.

    The optimizer should not treat every technically eligible DK row as equally real.
    For quarterbacks, keep only the most likely primary QB for each team active by
    default. This prevents backup QBs from entering lineups solely because historical
    production or salary relief gives them a mathematical score.

    This is deliberately conservative: non-QBs are labeled for role confidence but are
    not automatically removed without a stronger news/depth-chart source.
    """
    out=df.copy()
    if active_col not in out.columns:
        out[active_col]=True
    out["Primary QB"]=False
    out["Role Confidence"]="Rotation / uncertain"
    out["Auto Excluded Reason"]=""

    pos=out["Position"].astype(str).str.upper()
    qb_mask=pos.eq("QB")
    teams=[str(t) for t in out["Team"].dropna().unique().tolist() if str(t)]

    # Basic confidence labels for explanation/UI. These are not injury/news claims.
    proj=pd.to_numeric(out.get(projection_col,0),errors="coerce").fillna(0.0)
    sal=pd.to_numeric(out.get(salary_col,0),errors="coerce").fillna(0.0)
    out.loc[proj<=0.01,"Role Confidence"]="Inactive / no usable projection"
    out.loc[(proj>0.01)&(sal>0),"Role Confidence"]="Active pool"

    # A player with no usable projection is out before the build. The label and
    # the flag must agree, so deactivate here instead of labeling only.
    _no_proj=proj<=0.01
    out.loc[_no_proj,active_col]=False
    _no_proj_unexplained=_no_proj&(out["Auto Excluded Reason"].astype(str)=="")
    out.loc[_no_proj_unexplained,"Auto Excluded Reason"]="No usable projection — auto-excluded before the build"

    for team in teams:
        qidx=out.index[out["Team"].astype(str).eq(team)&qb_mask].tolist()
        if not qidx:
            continue
        # A QB with no usable projection can never be the primary: an injured
        # "starter" must not keep the primary slot over a healthy backup.
        # Among QBs with usable projections, salary remains the strongest
        # slate-specific market signal; projection/APG only break ties.
        primary=max(
            qidx,
            key=lambda i:(
                1.0 if float(pd.to_numeric(pd.Series([out.loc[i,projection_col]]),errors="coerce").fillna(0).iloc[0])>0.01 else 0.0,
                float(pd.to_numeric(pd.Series([out.loc[i,salary_col]]),errors="coerce").fillna(0).iloc[0]),
                float(pd.to_numeric(pd.Series([out.loc[i,projection_col]]),errors="coerce").fillna(0).iloc[0]),
                float(pd.to_numeric(pd.Series([out.loc[i,"AvgPointsPerGame"] if "AvgPointsPerGame" in out.columns else 0]),errors="coerce").fillna(0).iloc[0])
            )
        )
        if bool(out.loc[primary,active_col]):
            out.loc[primary,"Primary QB"]=True
            out.loc[primary,"Role Confidence"]="Primary QB"
        for i in qidx:
            if i==primary:
                continue
            if not bool(out.loc[i,active_col]):
                # Already out (e.g. no usable projection) — keep that reason
                # instead of mislabeling an unavailable QB as a healthy backup.
                continue
            out.loc[i,active_col]=False
            out.loc[i,"Role Confidence"]="Backup QB"
            out.loc[i,"Auto Excluded Reason"]="Backup QB — DFS LAB keeps only the primary QB active by default"

    # Anything already inactive without an explanation (e.g. a pool pre-filter
    # band) gets one, so the pre-build auto-excluded list is complete.
    _unexplained=(~out[active_col].astype(bool))&(out["Auto Excluded Reason"].astype(str)=="")
    out.loc[_unexplained,"Auto Excluded Reason"]="Excluded before the build — projection or salary below the usable minimum"

    return out

def apply_post_edit_availability_gate(df, projection_col="My Proj", active_col="ActiveForBuild", threshold=0.05):
    """Final safety gate for projection edits made AFTER pool creation.

    User projection overrides (Classic Players tab) and scenario tilts change
    numbers without rebuilding the pool, so a projection edited to zero must
    take the player out of the build here. Without this, ActiveForBuild stays
    True from the original upload and a zeroed player can still be selected.
    """
    out=df.copy()
    if active_col not in out.columns:
        out[active_col]=True
    for _col, _default in (("Auto Excluded Reason",""),("Role Confidence","")):
        if _col not in out.columns:
            out[_col]=_default
    proj=pd.to_numeric(out.get(projection_col,0),errors="coerce").fillna(0.0)
    zeroed=(proj<=threshold)&out[active_col].astype(bool)
    out.loc[zeroed,active_col]=False
    out.loc[zeroed,"Role Confidence"]="Unavailable — zero final projection"
    _unexplained=zeroed&(out["Auto Excluded Reason"].astype(str)=="")
    out.loc[_unexplained,"Auto Excluded Reason"]="Projection edited to 0 after pool creation — auto-excluded before the build"
    return out

@st.cache_data(ttl=1800, show_spinner=False)
def _load_live_nfl_availability(season=2026):
    """Current roster + injury availability from nflverse.

    Roster status catches non-injury absences (commissioner exempt, suspension,
    reserve/PUP, practice squad, inactive). Injury reports add official OUT status.
    Cached only 30 minutes because availability is time-sensitive on game day.
    """
    import nflreadpy as nfl
    roster=nfl.load_rosters_weekly(int(season)).to_pandas()
    injuries=nfl.load_injuries(int(season)).to_pandas()
    return roster,injuries

def _dfs_name_key(v):
    import re, unicodedata
    s=unicodedata.normalize("NFKD",str(v or "")).encode("ascii","ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]","",s)

def apply_live_availability_guard(df, active_col="ActiveForBuild", season=2026):
    """Auto-remove players with reliable current evidence they cannot play.

    Fail-open only when live data cannot load: if nflverse is unreachable,
    nobody is excluded. But a player with no roster record on any team this
    season is not on an NFL roster (e.g. a free agent the slate file still
    lists) and cannot score fantasy points, so they are excluded with a clear
    reason surfaced in the pre-build auto-excluded panel. Team defenses are
    exempt (they are not in player roster data). Questionable/doubtful players
    are surfaced but not automatically removed; official OUT and non-active
    roster statuses are removed.
    """
    out=df.copy()
    if active_col not in out.columns: out[active_col]=True
    out["Live Status"]="Not verified"
    out["Live Status Source"]=""
    out["Headshot URL"]=""
    try:
        roster,inj=_load_live_nfl_availability(int(season))
        # Latest weekly roster row per player NAME (any team). A player with no
        # roster row on any team this season is not on an NFL roster — e.g. a
        # free agent the slate file still lists — and cannot score fantasy
        # points, so they are excluded rather than failed open.
        nmap = None
        if not roster.empty:
            rn=_first_existing(roster.columns,["full_name","football_name","display_name"])
            rt=_first_existing(roster.columns,["team","Team"])
            rw=_first_existing(roster.columns,["week","Week"])
            rs=_first_existing(roster.columns,["status","status_description_abbr"])
            rh=_first_existing(roster.columns,["headshot_url","headshot","Headshot URL"])
            if rn and rt and rs:
                rr=roster.copy()
                rr["_key"]=rr[rn].map(_dfs_name_key); rr["_team"]=rr[rt].astype(str)
                rr["_week"]=pd.to_numeric(rr[rw],errors="coerce").fillna(0) if rw else 0
                rr=rr.sort_values("_week").drop_duplicates(["_key"],keep="last")
                nmap={r["_key"]:(str(r[rs]),str(r["_team"]),int(r["_week"]),str(r[rh]) if rh and pd.notna(r[rh]) else "") for _,r in rr.iterrows()}
        if nmap is not None:
            block_codes={"EXE","INA","PUP","RES","SUS","DEV","CUT","RET","UFA","NWT","RSN","RFA","TRC","TRD","TRT"}
            _pos=out["Position"].astype(str).str.upper() if "Position" in out.columns else pd.Series("",index=out.index)
            _is_dst=_pos.isin(["DST","D/ST"])
            for i,r in out.iterrows():
                if bool(_is_dst.loc[i]):
                    continue  # team defenses are not in player roster data
                hit=nmap.get(_dfs_name_key(r["Name"]))
                if not hit:
                    out.at[i,active_col]=False
                    out.at[i,"Role Confidence"]="Unavailable — not on a roster"
                    out.at[i,"Auto Excluded Reason"]=f"No {int(season)} NFL roster record — player is not on a team"
                    out.at[i,"Live Status"]="No roster record"
                    out.at[i,"Live Status Source"]=f"nflverse roster {int(season)}"
                    continue
                status,team,wk,headshot=hit; su=status.upper().strip()
                out.at[i,"Live Status"]=status
                out.at[i,"Live Status Source"]=f"nflverse roster W{wk}"
                if headshot and headshot.lower() not in ["nan","none",""]:
                    out.at[i,"Headshot URL"]=headshot
                if (not team) or (team.strip().upper() in ("","NAN","NONE","NULL")):
                    out.at[i,active_col]=False
                    out.at[i,"Role Confidence"]="Unavailable — not on a roster"
                    out.at[i,"Auto Excluded Reason"]="Roster record has no team — player is not on a team"
                    out.at[i,"Live Status"]="No team on roster record"
                    continue
                text=su.replace("."," ")
                blocked=(su in block_codes or any(x in text for x in [
                    "EX/COMM","COMMISSIONER","PRACTICE SQUAD","R/INJURED","RESERVE",
                    "SUSP","INACTIVE","WAIVER","RETIRED","R/PUP","PHYSICALLY UNABLE"
                ]))
                # Explicit Active/ACT is playable; unknown statuses are fail-open.
                if blocked:
                    out.at[i,active_col]=False
                    out.at[i,"Role Confidence"]="Unavailable — live roster"
                    out.at[i,"Auto Excluded Reason"]=f"Live roster status: {status}"

        # Latest official injury report. OUT is hard; doubtful/questionable are labels only.
        if not inj.empty:
            nn=_first_existing(inj.columns,["full_name","player_name","Name"])
            nt=_first_existing(inj.columns,["team","Team"])
            nw=_first_existing(inj.columns,["week","Week"])
            ns=_first_existing(inj.columns,["report_status","game_status","Status"])
            if nn and nt and ns:
                ii=inj.copy(); ii["_key"]=ii[nn].map(_dfs_name_key); ii["_team"]=ii[nt].astype(str)
                ii["_week"]=pd.to_numeric(ii[nw],errors="coerce").fillna(0) if nw else 0
                ii=ii.sort_values("_week").drop_duplicates(["_key","_team"],keep="last")
                imap={(r["_key"],r["_team"]):(str(r[ns]),int(r["_week"])) for _,r in ii.iterrows()}
                for i,r in out.iterrows():
                    hit=imap.get((_dfs_name_key(r["Name"]),str(r["Team"])))
                    if not hit: continue
                    status,wk=hit
                    if status and status.lower() not in ["nan","none",""]:
                        out.at[i,"Live Status"]=status
                        out.at[i,"Live Status Source"]=f"nflverse injury W{wk}"
                    if str(status).strip().upper()=="OUT":
                        out.at[i,active_col]=False
                        out.at[i,"Role Confidence"]="Unavailable — OUT"
                        out.at[i,"Auto Excluded Reason"]="Official injury report: OUT"
    except Exception as e:
        out.attrs["availability_warning"]=f"Live availability could not be verified: {e}"
    return out

def _slate_season(df, fallback=2026):
    import re
    try:
        m=re.search(r"(20\d{2})"," ".join(df.get("Game Info",pd.Series(dtype=str)).astype(str).tolist()))
        return int(m.group(1)) if m else int(fallback)
    except Exception:
        return int(fallback)

def estimate_ownership(df, proj_col="My Proj", salary_col="Salary", pos_col="Position",
                       total=900.0, dispersion=0.0, by_position=True):
    """V1 estimated field ownership.

    Combines projection, salary-implied value, and position baselines into a
    self-consistent ownership estimate.

    total: accounting target for the pool. 900 = classic 9 roster spots x 100%,
        the identity that the field's total ownership must equal. 600 = showdown
        6 roster spots x 100%.
    by_position: when True, enforce per-position-group slot accounting
        (classic: QB/DST groups + RB/WR/TE skill pool). When False, normalize
        globally across the pool (showdown, where any position can fill FLEX,
        including kickers).
    dispersion: 0 = as-estimated. Higher blends toward uniform, modeling the
        flatter field ownership of high-max-entry contests (a 150-Max field
        owns the "contrarian" plays more than a single-entry field does).

    This is a directional estimate for leverage, NOT a proprietary projection.
    Callers should label it as estimated wherever it is displayed.
    """
    proj=pd.to_numeric(df[proj_col],errors="coerce").fillna(0.0).clip(lower=0.0)
    sal=pd.to_numeric(df[salary_col],errors="coerce").fillna(0.0)
    pos=df[pos_col].astype(str).str.upper()
    value=proj/(sal/1000.0+0.5)
    pos_base={"QB":1.55,"RB":1.0,"WR":1.0,"TE":0.72,"DST":0.85}
    base=pos.map(pos_base).fillna(0.9)
    vp=value.rank(pct=True); pp=proj.rank(pct=True)
    raw=base*(0.25+0.75*vp)*(0.45+0.55*pp)
    active=proj>0.05
    raw=raw*active
    own=pd.Series(0.0,index=df.index)
    if by_position:
        groups={"QB":("QB",total/9.0),"DST":("DST",total/9.0)}
        for grp,target in groups.values():
            m=pos.eq(grp); tot=float(raw[m].sum())
            if tot>0: own[m]=target*raw[m]/tot
        skill=pos.isin(["RB","WR","TE"]); stot=float(raw[skill].sum())
        if stot>0: own[skill]=(total*7.0/9.0)*raw[skill]/stot
    else:
        tot=float(raw.sum())
        if tot>0: own=total*raw/tot
    if dispersion>0 and bool(active.any()):
        uniform=float(total)/float(active.sum())
        own=own*(1.0-dispersion)+dispersion*uniform*active.astype(float)
    return own.round(1)

def prepare_player_pool(dk_file, ss_file=None):
    dk = load_dk_template(dk_file)
    own_estimated=False
    if ss_file is not None:
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
    else:
        # DFS Lab independent projections: no SaberSim file required.
        # The engine output preserves DK row order (left merge), but merge on
        # Name explicitly so alignment never depends on that assumption.
        proj_df = dfs_lab_projection_engine(dk)
        proj_cols=["Name","DFS Lab Base Proj","Projection Why","Sim Vol","Matchup Adj %","Rush Share"]
        df = dk.merge(proj_df[[c for c in proj_cols if c in proj_df.columns]], on="Name", how="left")
        df["My Proj"] = pd.to_numeric(df["DFS Lab Base Proj"], errors="coerce").fillna(0.0)
        df["My Own"] = estimate_ownership(df)
        df["Projection Why"] = df["Projection Why"].fillna("DK slate prior fallback")
        df["Sim Vol"] = pd.to_numeric(df["Sim Vol"], errors="coerce").fillna(0.45)
        own_estimated=True

    # Mobile-QB tag: rushing share of fantasy from nflverse history. Absent on
    # the SaberSim path or when live evidence failed — then no QB is tagged,
    # never a fabricated one.
    if "Rush Share" not in df.columns:
        df["Rush Share"] = 0.0
    df["Rush Share"] = pd.to_numeric(df["Rush Share"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["Mobile QB"] = compute_mobile_qb_tags(df)

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

    # The same Football Reality layer used by Showdown also governs the Main Slate.
    # A backup QB should never require a manual exclusion just because DK listed him.
    df=apply_football_reality_guard(df,"Salary","My Proj","ActiveForBuild")
    df=apply_live_availability_guard(df,"ActiveForBuild",_slate_season(df))
    df["Own Estimated"]=bool(own_estimated)

    return df.reset_index(drop=True)

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

def _dk_rushing_points_from_stats(stats):
    """Rushing-only DraftKings fantasy points from nflverse weekly player stats.

    Used to measure how much of a player's (especially a QB's) fantasy value
    comes from rushing, without any hardcoded player list. Mirrors the rushing
    components of _dk_fantasy_points_from_stats exactly.
    """
    def col(name):
        return pd.to_numeric(stats[name], errors="coerce").fillna(0.0) if name in stats.columns else pd.Series(0.0, index=stats.index)
    pts = (col("rushing_yards") * 0.10 + col("rushing_tds") * 6
           - col("rushing_fumbles_lost"))
    # DK 100-yard rushing bonus.
    pts += (col("rushing_yards") >= 100).astype(float) * 3
    return pts

@st.cache_data(ttl=21600, show_spinner=False)
def _load_nflverse_inputs(season):
    """Multi-year evidence window: weekly player stats + schedules.

    Current season is included, but never allowed to dominate early.
    Schedules power the DST model (points allowed) and DST matchup table.
    """
    import nflreadpy as nfl
    seasons=[int(season)-3,int(season)-2,int(season)-1,int(season)]
    stats=nfl.load_player_stats(seasons, summary_level="week").to_pandas()
    sched=nfl.load_schedules(seasons).to_pandas()
    return stats, sched

def _load_nflverse_projection_inputs(season):
    stats, _ = _load_nflverse_inputs(season)
    return stats

def _opp_from_gameinfo(game_info, team):
    """Opponent abbreviation from a DK 'AWY@HOM' Game Info string."""
    try:
        m=str(game_info).split()[0]
        if "@" in m:
            a,h=[x.strip().upper() for x in m.split("@",1)]
            t=str(team).strip().upper()
            if t==a: return h
            if t==h: return a
    except Exception:
        pass
    return ""

def _dst_points_allowed_fantasy(pa):
    """DraftKings DST points-allowed bracket."""
    try: pa=float(pa)
    except Exception: return 0.0
    if pa<=0: return 10.0
    if pa<=6: return 7.0
    if pa<=13: return 4.0
    if pa<=20: return 1.0
    if pa<=27: return 0.0
    if pa<=34: return -1.0
    return -4.0

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

    Also emits per-player volatility ("Sim Vol") and handles DST rows via a team-defense
    model (sacks, turnovers, points allowed) so Classic can run without any SaberSim file.
    """
    out=dk.copy(); season=2026
    if "FlexSalary" not in out.columns and "Salary" in out.columns:
        out["FlexSalary"]=pd.to_numeric(out["Salary"],errors="coerce")
    try:
        import re
        m=re.search(r"(20\d{2})", " ".join(out["Game Info"].astype(str).tolist()))
        if m: season=int(m.group(1))
    except Exception: pass
    out["DFS Lab Data"]="DK prior fallback"
    out["DFS Lab Base Proj"]=pd.to_numeric(out.get("AvgPointsPerGame",0),errors="coerce").fillna(0.0)
    out["History Games"]=0; out["Current Games"]=0; out["Role Signal"]=0.0; out["Matchup Adj %"]=0.0
    out["Projection Why"]="DK slate prior fallback"
    out["Sim Vol"]=0.45
    try:
        stx=_load_nflverse_projection_inputs(season).copy()
        nc=_name_col(stx); sc=_season_col(stx); oc=_opp_col(stx)
        if not nc or not sc: raise ValueError("nflverse player name/season columns unavailable")
        stx["Name"]=stx[nc].astype(str).str.strip(); stx["Season"]=pd.to_numeric(stx[sc],errors="coerce")
        stx["_fp"]=_dk_fantasy_points_from_stats(stx); stx["_role"]=_role_points_from_stats(stx); stx["_rush_fp"]=_dk_rushing_points_from_stats(stx)
        # Ignore placeholder rows with no statistical activity.
        activity=[]
        for c in ["passing_attempts","carries","targets","receptions","field_goals_made","extra_points_made"]:
            if c in stx.columns: activity.append(pd.to_numeric(stx[c],errors="coerce").fillna(0.0))
        if activity:
            active=sum(activity)>0
            stx=stx[active | (stx["_fp"].abs()>0)].copy()

        # Per-player per-season summaries. Four-year weights favor recency without letting one game take over.
        ss=stx.groupby(["Name","Season"],as_index=False).agg(FPPG=("_fp","mean"),RushFPPG=("_rush_fp","mean"),Role=("_role","mean"),Games=("_fp","size"),Std=("_fp","std"))
        season_weights={season:0.34,season-1:0.38,season-2:0.19,season-3:0.09}
        rows=[]
        for name,g in ss.groupby("Name"):
            hist_num=hist_den=role_num=role_den=vol_num=vol_den=rush_num=rush_den=0.0; hist_games=cur_games=0
            for _,r in g.iterrows():
                yr=int(r["Season"]); games=int(r["Games"]); w=season_weights.get(yr,0.0)
                if w<=0: continue
                # Current-year reliability ramps from 20% after one game toward full weight after eight.
                reliability=min(1.0,max(0.20,games/8.0)) if yr==season else min(1.0,games/8.0)
                ew=w*reliability
                hist_num += ew*float(r["FPPG"]); hist_den += ew
                rush_num += ew*float(r["RushFPPG"]); rush_den += ew
                role_num += ew*float(r["Role"]); role_den += ew
                _sd=float(r["Std"]) if pd.notna(r["Std"]) else 0.0
                vol_num += ew*_sd; vol_den += ew
                hist_games += games
                if yr==season: cur_games=games
            rows.append({"Name":name,"HistProj":hist_num/hist_den if hist_den else 0.0,"RushHist":rush_num/rush_den if rush_den else 0.0,"RoleSignal":role_num/role_den if role_den else 0.0,"HistoryGames":hist_games,"CurrentGames":cur_games,"Vol":vol_num/vol_den if vol_den else 0.0})
        ps=pd.DataFrame(rows)
        out=out.merge(ps,on="Name",how="left")
        for c in ["HistProj","RoleSignal","HistoryGames","CurrentGames","Vol"]: out[c]=pd.to_numeric(out[c],errors="coerce").fillna(0.0)
        # Rushing share of fantasy: the mobile-QB signal. 0 when there is no
        # usable history rather than a fabricated value.
        _rh=pd.to_numeric(out["RushHist"],errors="coerce").fillna(0.0) if "RushHist" in out.columns else 0.0
        _hp=pd.to_numeric(out["HistProj"],errors="coerce").fillna(0.0)
        out["Rush Share"]=pd.Series(np.where(_hp>1.0,(_rh/_hp).clip(0.0,1.0),0.0),index=out.index).round(3)

        # ---- DST model: team-week defensive fantasy scores in DK scoring ----
        # Fumble recoveries aren't split out in the weekly feed, so recoveries are
        # estimated at half of forced fumbles (empirically ~50% are recovered by the defense).
        dst_matchup={}
        try:
            _dstats,_sched=_load_nflverse_inputs(season)
            _d=_dstats.copy()
            def _dc(n):
                return pd.to_numeric(_d[n],errors="coerce").fillna(0.0) if n in _d.columns else pd.Series(0.0,index=_d.index)
            _d["_dteam"]=_d["team"].astype(str).str.strip().str.upper()
            _d["_sn"]=pd.to_numeric(_d["season"],errors="coerce"); _d["_wk"]=pd.to_numeric(_d["week"],errors="coerce")
            _d["_dst_pts"]=(_dc("def_sacks")*1.0+_dc("def_interceptions")*2.0+_dc("def_fumbles_forced")*0.5*2.0
                +_dc("def_tds")*6.0+_dc("def_safeties")*2.0
                +(_dc("def_fg_blocks")+_dc("def_punt_blocks")+_dc("def_pat_blocks"))*2.0)
            _tw=_d.groupby(["_dteam","_sn","_wk"],as_index=False).agg(dst_pts=("_dst_pts","sum"))
            _s=_sched.copy()
            _hs=pd.to_numeric(_s["home_score"],errors="coerce"); _aws=pd.to_numeric(_s["away_score"],errors="coerce")
            _pa=pd.concat([
                pd.DataFrame({"_dteam":_s["home_team"].astype(str).str.upper(),"_sn":pd.to_numeric(_s["season"],errors="coerce"),"_wk":pd.to_numeric(_s["week"],errors="coerce"),"pa":_aws}),
                pd.DataFrame({"_dteam":_s["away_team"].astype(str).str.upper(),"_sn":pd.to_numeric(_s["season"],errors="coerce"),"_wk":pd.to_numeric(_s["week"],errors="coerce"),"pa":_hs}),
            ],ignore_index=True)
            _tw=_tw.merge(_pa,on=["_dteam","_sn","_wk"],how="left")
            _tw["dst_pts"]=_tw["dst_pts"]+_tw["pa"].apply(_dst_points_allowed_fantasy)
            _ss=_tw.groupby(["_dteam","_sn"],as_index=False).agg(DSTFPG=("dst_pts","mean"),DSTGames=("dst_pts","size"),DSTStd=("dst_pts","std"))
            _drows=[]
            for _t,_g in _ss.groupby("_dteam"):
                hn=hd=vn=vd=0.0; hg=cg=0
                for _,_r in _g.iterrows():
                    _yr=int(_r["_sn"]); _gm=int(_r["DSTGames"]); _wt=season_weights.get(_yr,0.0)
                    if _wt<=0: continue
                    _rel=min(1.0,max(0.20,_gm/8.0)) if _yr==season else min(1.0,_gm/8.0)
                    _ew=_wt*_rel
                    hn+=_ew*float(_r["DSTFPG"]); hd+=_ew
                    _sdd=float(_r["DSTStd"]) if pd.notna(_r["DSTStd"]) else 0.0
                    vn+=_ew*_sdd; vd+=_ew
                    hg+=_gm
                    if _yr==season: cg=_gm
                _drows.append({"Team":_t,"DSTHist":hn/hd if hd else 0.0,"DSTGames":hg,"DSTCur":cg,"DSTVol":vn/vd if vd else 0.0})
            _dh=pd.DataFrame(_drows)
            if not _dh.empty:
                out=out.merge(_dh,on="Team",how="left")
                # Matchup: fantasy scored by defenses facing each offense.
                _opp_map={}
                for _,_r in _s.iterrows():
                    try:
                        _ht=str(_r["home_team"]).upper(); _at=str(_r["away_team"]).upper()
                        _sn2=int(_r["season"]); _wk2=int(_r["week"])
                        _opp_map[(_ht,_sn2,_wk2)]=_at; _opp_map[(_at,_sn2,_wk2)]=_ht
                    except Exception: pass
                _tw["_off_opp"]=_tw.apply(lambda _r:_opp_map.get((_r["_dteam"],int(_r["_sn"]),int(_r["_wk"])),""),axis=1)
                _al=_tw[_tw["_off_opp"]!=""].groupby("_off_opp",as_index=False).agg(Allowed=("dst_pts","mean"),N=("dst_pts","size"))
                _lg=float(_tw["dst_pts"].mean())
                for _,_r in _al.iterrows():
                    _b=max(_lg,1.0); _n=float(_r["N"])
                    _raw=float(_r["Allowed"])/_b-1.0; _sh=_n/(_n+24.0)
                    dst_matchup[(str(_r["_off_opp"]),"DST")]=float(np.clip(_raw*_sh,-0.10,0.10))
        except Exception:
            dst_matchup={}
        for c in ["DSTHist","DSTGames","DSTCur","DSTVol"]:
            if c in out.columns: out[c]=pd.to_numeric(out[c],errors="coerce").fillna(0.0)

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
        vals=[]; reasons=[]; madjs=[]; sigs=[]
        _pos_default_vol={"QB":0.26,"RB":0.42,"WR":0.50,"TE":0.48,"DST":0.62}
        import math as _math
        for _,r in out.iterrows():
            pos_u=str(r["Position"]).upper()
            is_dst=(pos_u=="DST")
            if is_dst and float(r.get("DSTHist",0))>0:
                hist=float(r.get("DSTHist",0)); games=int(r.get("DSTGames",0)); curg=int(r.get("DSTCur",0)); prior=float(r.get("AvgPointsPerGame",0) or 0)
                emp_vol=float(r.get("DSTVol",0)); why_src="DST history"
            else:
                hist=float(r.get("HistProj",0)); games=int(r.get("HistoryGames",0)); curg=int(r.get("CurrentGames",0)); prior=float(r.get("AvgPointsPerGame",0) or 0)
                emp_vol=float(r.get("Vol",0)); why_src="4-year history"
            # Historical evidence dominates established players; DK average is a weak stabilizer/fallback.
            evidence=min(0.88, games/(games+8.0))
            base=(evidence*hist + (1-evidence)*prior) if hist>0 else prior
            # Role signal is used only as a modest stabilizer, not converted directly to fantasy points.
            role=float(r.get("RoleSignal",0)); role_adj=0.0
            if role>0 and base>0 and not is_dst:
                # Keeps TD spikes from dominating while rewarding sustained opportunity.
                role_adj=float(np.clip((role/12.0)-0.5,-0.04,0.05))
            opp=""
            try:
                teams=[t for t in out["Team"].dropna().unique().tolist() if t]
                if len(teams)==2: opp=teams[1] if r["Team"]==teams[0] else teams[0]
                else: opp=_opp_from_gameinfo(r.get("Game Info",""),r["Team"])
            except Exception: pass
            _mtable=dst_matchup if is_dst else matchup
            m=float(_mtable.get((str(opp),pos_u),0.0)); madjs.append(m*100)
            # Matchup and role are bounded; they refine the baseline rather than rewrite it.
            model=max(0.0,base*(1.0+role_adj+m))
            # Salary prior only for thin-history skill players.
            if games<5 and pos_u in ["QB","RB","WR","TE"]:
                sp=max(0.3,float(r["FlexSalary"])/1000.0*1.55)
                model=0.90*model+0.10*sp
            vals.append(round(model,3))
            reasons.append(f"{why_src} {hist:.2f} over {games} games; current season {curg} game(s) is sample-shrunk; DK prior {prior:.2f}; role {role_adj*100:+.1f}%; {opp or 'opponent'} matchup {m*100:+.1f}%")
            # Per-player volatility: empirical weekly std shrunk toward the position prior.
            _sig=0.45
            if emp_vol>0 and base>0:
                _cv=emp_vol/max(base,1.5)
                _sig=float(min(0.9,max(0.15,_math.sqrt(_math.log1p(_cv*_cv)))))
            sigs.append(round(0.5*_sig+0.5*_pos_default_vol.get(pos_u,0.45),3))
        out["DFS Lab Base Proj"]=vals; out["History Games"]=out["HistoryGames"].astype(int); out["Current Games"]=out["CurrentGames"].astype(int)
        out["Role Signal"]=out["RoleSignal"].round(2); out["Matchup Adj %"]=np.round(madjs,1); out["Projection Why"]=reasons
        out["Sim Vol"]=sigs
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

def prepare_showdown_pool(dk_file, ss_file=None, entry_format=None):
    """Create the Showdown pool. DFS Lab projections work with DK alone; SaberSim is optional comparison data.

    entry_format: contest entry format ("Single Entry", "3-Max", "20-Max", "150-Max").
        Used only to set the dispersion of the estimated ownership (flatter field
        ownership in high-max-entry contests). SaberSim ownership, when present,
        always wins and is never overwritten.
    """
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
    _estimate_showdown_ownership(df, entry_format)
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

    # Shared with Classic: apply the same football-reality gate before optimization.
    df=apply_football_reality_guard(df,"FlexSalary","My Proj","ActiveForBuild")
    df=apply_live_availability_guard(df,"ActiveForBuild",_slate_season(df))

    configure_game_worlds(df)
    return df.reset_index(drop=True)

def _estimate_showdown_ownership(df, entry_format=None):
    """Fill My Own / CPT Own for a showdown pool when no SaberSim ownership exists.

    SaberSim values always win when present. Otherwise estimate directionally:
    showdown normalizes to 600 (6 roster spots x 100%), globally across the pool
    (any position can fill FLEX, including kickers). CPT ownership falls back to
    ~18% of FLEX, the same convention used when SaberSim lacks captain data.
    Dispersion models flatter field ownership in high-max-entry contests.
    Returns True when values were estimated.
    """
    _dispersion={"Single Entry":0.0,"3-Max":0.10,"20-Max":0.20,"150-Max":0.30}.get(entry_format,0.0)
    _has_own=bool(pd.to_numeric(df["My Own"],errors="coerce").fillna(0.0).max()>0.01)
    if not _has_own:
        df["My Own"]=estimate_ownership(df,proj_col="My Proj",salary_col="FlexSalary",
                                        pos_col="Position",total=600.0,
                                        dispersion=_dispersion,by_position=False)
        df["CPT Own"]=(pd.to_numeric(df["My Own"],errors="coerce").fillna(0.0)*0.18).clip(lower=0.1).round(1)
    df["Own Estimated"]=bool(not _has_own)
    return bool(not _has_own)
