"""Auto-fill Game Intel from nflverse data.

The vision: DFS LAB does the research, the user only overrides. Every
suggestion carries a moderate Confidence (60 -- a data heuristic, not a take)
and a Note starting with "Auto:" so the audit trail shows what the machine
believed. The UI never overwrites a cell the user has touched (see
_auto_intel_adopt_manual in dfs_lab/ui/main.py).

Five factors, all from nflverse weekly player stats + schedules:
  Defense    - fantasy points the opponent's defense allows to the player's
               position over the last 4 weeks (positional DvP), percentile
               ranked across all 32 teams.
  Usage      - trend in the player's opportunities (targets + carries): last
               3 games vs earlier this season. Plus an injury bump: when a
               same-team, same-position mate is OUT, the next man up gets +2
               (this automates the Ertz-for-Goedert case).
  Home/Rest  - home team +1; a 3+ day rest edge over the opponent is +1 for
               the rested side / -1 for the tired side (from schedule dates).
  Travel     - away team crossing 2+ time zones (or a 1500+ mile trip) -1.
  Time/Split - left at 0 in v1: no honest data edge found (primetime is not
               inherently positive). The column stays user-only for now.

All suggestion builders are pure (dataframes in, dicts out) so they are
unit-testable without network. auto_fill_game_intel() is the thin wrapper
that loads nflverse and calls suggest_game_intel().
"""
import math

import pandas as pd
import streamlit as st

from dfs_lab.data import _dk_fantasy_points_from_stats

AUTO_CONFIDENCE = 60
_SKILL_POS = ("QB", "RB", "WR", "TE")

# Stadium coordinates + standard-time UTC offset. Offsets only ever appear
# as differences, so DST observance does not matter.
TEAM_GEO = {
    "ARI": (33.53, -112.26, -7), "ATL": (33.76, -84.40, -5),
    "BAL": (39.28, -76.62, -5), "BUF": (42.77, -78.79, -5),
    "CAR": (35.23, -80.85, -5), "CHI": (41.86, -87.62, -6),
    "CIN": (39.10, -84.51, -5), "CLE": (41.51, -81.70, -5),
    "DAL": (32.75, -97.09, -6), "DEN": (39.74, -105.02, -7),
    "DET": (42.34, -83.05, -5), "GB": (44.50, -88.06, -6),
    "HOU": (29.68, -95.41, -6), "IND": (39.76, -86.16, -5),
    "JAX": (30.32, -81.64, -5), "KC": (39.05, -94.48, -6),
    "LV": (36.09, -115.18, -8), "LAC": (32.78, -117.12, -8),
    "LAR": (33.95, -118.34, -8), "MIA": (25.96, -80.24, -5),
    "MIN": (44.97, -93.26, -6), "NE": (42.09, -71.26, -5),
    "NO": (29.95, -90.08, -6), "NYG": (40.81, -74.07, -5),
    "NYJ": (40.81, -74.07, -5), "PHI": (39.90, -75.17, -5),
    "PIT": (40.45, -80.02, -5), "SF": (37.40, -121.97, -8),
    "SEA": (47.60, -122.33, -8), "TB": (27.98, -82.50, -5),
    "TEN": (36.17, -86.77, -6), "WAS": (38.91, -76.86, -5),
}


def _name_key(name):
    """Normalize a player name for cross-source matching."""
    import re
    k = str(name or "").lower().replace("'", "").replace(".", "").replace("-", " ")
    k = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", k.strip())
    return re.sub(r"\s+", " ", k)


def _season_week_stats(player_stats, season):
    s = player_stats
    if "season" in s.columns:
        s = s[s["season"].astype(str).eq(str(season))]
    return s


def slate_game_from_schedule(schedules, season, teams):
    """Find this slate's game in the nflverse schedule. Pure.

    Returns the schedule row (week, home/away, gameday...) or None when the
    slate's matchup cannot be matched -- auto-fill then reports honestly
    instead of guessing.
    """
    tset = {str(t) for t in teams if str(t) and str(t).lower() != "nan"}
    if len(tset) != 2:
        return None
    s = schedules
    if "season" in s.columns:
        s = s[s["season"].astype(str).eq(str(season))]
    a, b = sorted(tset)
    m = s[((s["home_team"].eq(a)) & (s["away_team"].eq(b))) |
          ((s["home_team"].eq(b)) & (s["away_team"].eq(a)))]
    if m.empty:
        return None
    return m.sort_values("week").iloc[-1]


def defense_dvp(player_stats, season, week, lookback=4):
    """{(defense_team, position): avg DK pts allowed per game}. Pure.

    Window: the `lookback` completed weeks before `week`, current season.
    """
    s = _season_week_stats(player_stats, season)
    s = s[(s["week"] >= week - lookback) & (s["week"] < week)].copy()
    s = s[s["position"].isin(_SKILL_POS)]
    if s.empty:
        return {}
    s["_pts"] = _dk_fantasy_points_from_stats(s).to_numpy()
    total = s.groupby(["opponent_team", "position"])["_pts"].sum()
    games = s.groupby("opponent_team")["week"].nunique()
    out = {}
    for (dteam, pos), pts in total.items():
        g = games.get(dteam, 0)
        if g:
            out[(dteam, pos)] = float(pts) / float(g)
    return out


def defense_rating(allowed, league_values):
    """Map a defense's allowed-points figure to -3..+3 by percentile. Pure."""
    vals = sorted(v for v in league_values if v is not None)
    if not vals or allowed is None:
        return 0, ""
    # Percentile of `allowed` within the league distribution (high = soft defense).
    pct = sum(1 for v in vals if v < allowed) / max(1, len(vals) - 1) if len(vals) > 1 else 0.5
    if pct >= 0.90:
        return 3, "softest"
    if pct >= 0.75:
        return 2, "soft"
    if pct >= 0.60:
        return 1, "below average"
    if pct <= 0.10:
        return -3, "elite"
    if pct <= 0.25:
        return -2, "tough"
    if pct <= 0.40:
        return -1, "above average"
    return 0, ""


def usage_trend(player_stats, season, week, namekey, recent_n=3):
    """(rating, note) from opportunity trend: last `recent_n` games vs earlier. Pure."""
    s = _season_week_stats(player_stats, season)
    s = s[(s["week"] < week)].copy()
    if s.empty or "player_name" not in s.columns:
        return 0, ""
    s["_key"] = s["player_name"].map(_name_key)
    s = s[s["_key"].eq(namekey)]
    if s.empty:
        return 0, ""
    opp = (pd.to_numeric(s.get("targets", 0), errors="coerce").fillna(0)
           + pd.to_numeric(s.get("carries", 0), errors="coerce").fillna(0))
    s = s.assign(_opp=opp).sort_values("week")
    recent = s[s["week"] >= week - recent_n]["_opp"]
    base = s[s["week"] < week - recent_n]["_opp"]
    if len(recent) < 2 or len(base) < 2 or base.mean() <= 0:
        return 0, ""
    pct = (recent.mean() - base.mean()) / base.mean()
    if pct >= 0.40:
        return 2, f"opportunities up {pct:.0%} L{recent_n}G ({recent.mean():.1f} vs {base.mean():.1f}/g)"
    if pct >= 0.20:
        return 1, f"opportunities up {pct:.0%} L{recent_n}G ({recent.mean():.1f} vs {base.mean():.1f}/g)"
    if pct <= -0.30:
        return -2, f"opportunities down {abs(pct):.0%} L{recent_n}G ({recent.mean():.1f} vs {base.mean():.1f}/g)"
    if pct <= -0.15:
        return -1, f"opportunities down {abs(pct):.0%} L{recent_n}G ({recent.mean():.1f} vs {base.mean():.1f}/g)"
    return 0, ""


def _season_avg_points(player_stats, season, week):
    """{_name_key: avg DK pts per game this season (weeks before `week`)}. Pure."""
    s = _season_week_stats(player_stats, season)
    s = s[s["week"] < week].copy()
    if s.empty or "player_name" not in s.columns:
        return {}
    s["_pts"] = _dk_fantasy_points_from_stats(s).to_numpy()
    s["_key"] = s["player_name"].map(_name_key)
    return s.groupby("_key")["_pts"].mean().to_dict()


def injury_usage_bumps(df, star_avg=None):
    """{player_id: (rating, note)} for next-man-up after a teammate's absence. Pure.

    When a same-team, same-position RB/WR/TE is inactive for the build and was
    a significant player, the top active replacement gets Usage +2 and the
    second +1. "Significant" means a meaningful projection OR a strong season
    average -- an inactive star's projection is usually zeroed (e.g. Goedert
    OUT at 0.0), so the projection alone cannot be the test. QBs are excluded:
    the backup-QB projection is handled manually elsewhere and a second nudge
    would double-count.
    """
    out = {}
    if "ActiveForBuild" not in df.columns:
        return out
    star_avg = star_avg or {}
    d = df.copy()
    active = d["ActiveForBuild"].astype(bool)
    proj = pd.to_numeric(d.get("My Proj", 0), errors="coerce").fillna(0)
    d["_star"] = d["Name"].map(_name_key).map(star_avg).fillna(0.0)
    inactive = d[(~active) & d["Position"].isin(("RB", "WR", "TE"))
                 & ((proj >= 5) | (d["_star"] >= 6))]
    for (team, pos), grp in inactive.groupby(["Team", "Position"]):
        mates = d[(d["Team"].eq(team)) & (d["Position"].eq(pos)) & active].copy()
        if mates.empty:
            continue
        mates["_p"] = pd.to_numeric(mates.get("My Proj", 0), errors="coerce").fillna(0)
        mates = mates.sort_values("_p", ascending=False)
        for i, (_, m) in enumerate(mates.head(2).iterrows()):
            out_name = str(grp.iloc[0]["Name"])
            bump = 2 if i == 0 else 1
            pid = str(m["ID"])
            prev = out.get(pid)
            if prev is None or bump > prev[0]:
                out[pid] = (bump, f"{out_name} OUT -- next {pos} up")
    return out


def _haversine_miles(a, b):
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 3959 * math.asin(math.sqrt(h))


def rest_travel_for_game(schedules, season, game_row):
    """{team: {"Home/Rest": (rating, note), "Travel": (rating, note)}}. Pure."""
    home, away = str(game_row["home_team"]), str(game_row["away_team"])
    gameday = pd.to_datetime(game_row["gameday"])
    s = schedules
    if "season" in s.columns:
        s = s[s["season"].astype(str).eq(str(season))]
    rest = {}
    for t in (home, away):
        past = s[((s["home_team"].eq(t)) | (s["away_team"].eq(t))) & (s["week"] < int(game_row["week"]))]
        if past.empty:
            rest[t] = 7
        else:
            last = pd.to_datetime(past.sort_values("week").iloc[-1]["gameday"])
            rest[t] = max(1, int((gameday - last).days))
    out = {home: {"Home/Rest": (1, "home game"), "Travel": (0, "")},
           away: {"Home/Rest": (0, ""), "Travel": (0, "")}}
    edge = rest[home] - rest[away]
    if edge >= 3:
        r, n = out[home]["Home/Rest"]
        out[home]["Home/Rest"] = (min(2, r + 1), f"home game, +{edge}d rest edge")
        r2, _ = out[away]["Home/Rest"]
        out[away]["Home/Rest"] = (r2 - 1, f"-{edge}d rest disadvantage")
    elif edge <= -3:
        r, n = out[away]["Home/Rest"]
        out[away]["Home/Rest"] = (r + 1, f"+{abs(edge)}d rest edge")
        r2, _ = out[home]["Home/Rest"]
        out[home]["Home/Rest"] = (r2 - 1, f"home game, -{abs(edge)}d rest disadvantage")
    if home in TEAM_GEO and away in TEAM_GEO:
        tz = abs(TEAM_GEO[away][2] - TEAM_GEO[home][2])
        mi = _haversine_miles(TEAM_GEO[away], TEAM_GEO[home])
        if tz >= 2:
            out[away]["Travel"] = (-1, f"{tz}-timezone trip to {home} ({mi:,.0f} mi)")
        elif mi >= 1500:
            out[away]["Travel"] = (-1, f"{mi:,.0f}-mi trip to {home}")
    return out


def suggest_game_intel(df, player_stats, schedules, season):
    """{player_id: {Defense, Usage, Home/Rest, Travel, Time/Split, Confidence, Note}}.

    Pure core of auto-fill. Returns ({}, message) when the slate cannot be
    matched to the schedule or there is no usable data -- the caller reports
    the message honestly instead of filling junk.
    """
    teams = [str(t) for t in df["Team"].dropna().unique().tolist()]
    game = slate_game_from_schedule(schedules, season, teams)
    if game is None:
        return {}, "Could not match this slate to the nflverse schedule, so no auto-fill was applied."
    week = int(game["week"])
    home, away = str(game["home_team"]), str(game["away_team"])
    dvp = defense_dvp(player_stats, season, week)
    rt = rest_travel_for_game(schedules, season, game)
    star_avg = _season_avg_points(player_stats, season, week)
    bumps = injury_usage_bumps(df, star_avg)
    suggestions = {}
    for _, r in df.iterrows():
        pid, pos, team = str(r["ID"]), str(r["Position"]), str(r["Team"])
        opp = away if team == home else home
        s = {"Defense": 0, "Usage": 0, "Home/Rest": 0, "Travel": 0,
             "Time/Split": 0, "Confidence": AUTO_CONFIDENCE, "Note": ""}
        notes = []
        if pos in _SKILL_POS:
            key = (opp, pos)
            if key in dvp:
                league = [v for (t, p), v in dvp.items() if p == pos]
                rating, word = defense_rating(dvp[key], league)
                if rating:
                    s["Defense"] = rating
                    rank = sum(1 for v in league if v >= dvp[key])
                    notes.append(f"{opp} allowed {dvp[key]:.1f} DK pts/g to {pos}s L4W ({word}, {rank}/{len(league)})")
        tr, tnote = usage_trend(player_stats, season, week, _name_key(r["Name"]))
        br, bnote = bumps.get(pid, (0, ""))
        if br >= tr and br:
            s["Usage"], unote = br, bnote
        elif tr:
            s["Usage"], unote = tr, tnote
        else:
            unote = ""
        if unote:
            notes.append(unote)
        hr, hnote = rt.get(team, {}).get("Home/Rest", (0, ""))
        if hr:
            s["Home/Rest"] = hr
            notes.append(hnote)
        tv, tnote2 = rt.get(team, {}).get("Travel", (0, ""))
        if tv:
            s["Travel"] = tv
            notes.append(tnote2)
        if any(s[c] for c in ("Defense", "Usage", "Home/Rest", "Travel", "Time/Split")):
            s["Note"] = "Auto: " + " • ".join(notes)
            suggestions[pid] = s
    msg = f"DFS LAB researched {len(suggestions)} players from nflverse (wk {week} {away}@ {home})."
    return suggestions, msg


@st.cache_data(ttl=21600, show_spinner=False)
def _cached_auto_intel_inputs(season):
    import nflreadpy as nfl
    stats = nfl.load_player_stats([int(season)], summary_level="week").to_pandas()
    sched = nfl.load_schedules([int(season)]).to_pandas()
    return stats, sched


def auto_fill_game_intel(df, season):
    """Load nflverse (cached) and suggest Game Intel for this slate.

    Returns (suggestions, message). Never raises: on any data failure it
    returns ({}, <honest reason>) so the UI can say so plainly.
    """
    try:
        stats, sched = _cached_auto_intel_inputs(int(season))
    except Exception as e:
        return {}, f"Auto-fill couldn't reach nflverse ({e}). Your table is unchanged."
    try:
        return suggest_game_intel(df, stats, sched, int(season))
    except Exception as e:
        return {}, f"Auto-fill hit a data problem ({e}). Your table is unchanged."
