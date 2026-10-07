"""Results Command Center — post-build review UI for the Classic build step.

SCOPE: this is the BUILD step. It appears after the optimizer generates lineups,
for reviewing and understanding the completed build. It is NOT the lineup-choosing
step (picking which lineups to actually enter contests) — that gets its own
simpler design later. The Lineups / Game Worlds / Lab+Agent views belong here only.

All helpers are pure (no Streamlit) so they can be unit-tested headlessly;
the render_* functions below are thin Streamlit wrappers around them.
"""

import html
import os

import numpy as np
import pandas as pd
import streamlit as st

from dfs_lab.classic import lineup_sim_equity, classic_postbuild_answer
from dfs_lab.config import ROSTER_SLOTS
from dfs_lab.leverage import chalk_bust_beneficiaries

SALARY_CAP = 50000


def _esc(s):
    """Escape uploaded-data strings before they enter unsafe_allow_html HTML."""
    return html.escape("" if s is None else str(s))

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def parse_stack_summary(summary):
    """Parse 'KC: Mahomes + Kelce | MIA bring-back: Hill' into parts.

    Returns dict(qb_team, qb_name, stack_names, opp, bb_names). Never raises.
    """
    out = {"qb_team": "", "qb_name": "", "stack_names": [], "opp": "", "bb_names": []}
    try:
        s = str(summary or "")
        if "|" in s:
            left, right = s.split("|", 1)
        else:
            left, right = s, ""
        if ":" in left:
            team, rest = left.split(":", 1)
            out["qb_team"] = team.strip()
            pieces = [p.strip() for p in rest.split("+")]
            if pieces:
                out["qb_name"] = pieces[0]
                out["stack_names"] = [p for p in pieces[1:] if p and p.lower() != "none"]
        if "bring-back:" in right:
            opp_tok, bb = right.split("bring-back:", 1)
            out["opp"] = opp_tok.strip()
            out["bb_names"] = [p.strip() for p in bb.split("+") if p.strip() and p.strip().lower() != "none"]
    except Exception:
        pass
    return out


def short_story(row):
    """One-line game-script label, e.g. 'KC leads / MIA comeback'.

    Derived from the stack summary, never invented. Falls back to the stored
    Lineup Story when the summary cannot be parsed.
    """
    parts = parse_stack_summary(row.get("Stack Summary", ""))
    qb_team, opp = parts["qb_team"], parts["opp"]
    if qb_team and opp:
        if parts["bb_names"]:
            return f"{qb_team} leads / {opp} comeback"
        return f"{qb_team} leads / no {opp} answer"
    story = str(row.get("Lineup Story", "") or "").strip()
    if story and story != "No coherent QB-led Classic story available":
        return story[:90]
    return "Balanced build"


def lineup_title(row):
    """Compact 'QB (TEAM +N)' label for pickers and expander labels."""
    parts = parse_stack_summary(row.get("Stack Summary", ""))
    qb = parts["qb_name"] or str(row.get("QB", ""))
    team = parts["qb_team"]
    n = len(parts["stack_names"])
    if qb and team:
        return f"{qb} ({team} +{n})"
    return qb or f"Lineup #{row.get('Rank', '?')}"


def hero_stats(res, eq):
    """Hero + tile numbers for the results shell. Pure."""
    out = {}
    if res is None or res.empty:
        return out
    best = res.iloc[0]
    out["grade"] = str(best.get("Rating", "–"))
    out["grade_score"] = float(best.get("Rating Score", 0) or 0)
    out["best_title"] = lineup_title(best)
    out["best_rank"] = int(best.get("Rank", 1) or 1)
    out["top_projection"] = float(pd.to_numeric(res["Projection"], errors="coerce").max() or 0)
    out["lineups"] = int(len(res))
    if eq is not None and not eq.empty and len(eq) == len(res):
        out["best_ceiling"] = float(pd.to_numeric(eq["Ceiling P90"], errors="coerce").max() or 0)
        out["avg_break"] = float(pd.to_numeric(eq["Break Slate %"], errors="coerce").mean() or 0)
    else:
        out["best_ceiling"] = 0.0
        out["avg_break"] = 0.0
    out["avg_salary_left"] = float(pd.to_numeric(res["Salary Left"], errors="coerce").mean() or 0)
    return out


SORT_OPTIONS = {
    "Aytia Grade": "Rating Score",
    "Projection": "Projection",
    "Ceiling P90": "Ceiling P90",
    "Break Slate %": "Break Slate %",
    "Salary Left": "Salary Left",
}


def sort_lineups(disp, sort_label):
    """Return display frame sorted by the chosen option. Pure."""
    col = SORT_OPTIONS.get(sort_label, "Rating Score")
    if col in disp.columns:
        return disp.sort_values(col, ascending=False, kind="mergesort").reset_index(drop=True)
    return disp


def game_archetype(row, p90_lo, p90_hi):
    """Classify a simulated game environment. Pure and relative to the slate."""
    p90 = float(row.get("P90", 0) or 0)
    if p90 >= p90_hi:
        return "Shootout"
    if p90 >= p90_lo:
        return "Balanced scoring"
    return "Defensive grind"


def world_rows(sim_table):
    """Game-world bars, sorted by probability descending. Pure.

    Probability = 'Slate ceiling %': share of the 10,000 simulations in which
    this game produced the slate's best DFS environment.
    """
    if sim_table is None or sim_table.empty:
        return []
    t = sim_table.copy()
    p90 = pd.to_numeric(t["P90"], errors="coerce").fillna(0)
    lo, hi = float(p90.quantile(0.4)), float(p90.quantile(0.75))
    rows = []
    for _, r in t.iterrows():
        rows.append({
            "game": str(r.get("Game", "")),
            "archetype": game_archetype(r, lo, hi),
            "prob": float(r.get("Slate ceiling %", 0) or 0),
            "p90": float(r.get("P90", 0) or 0),
            "mean_env": float(r.get("Mean DFS env", 0) or 0),
            "vol": float(r.get("Volatility", 0) or 0),
        })
    rows.sort(key=lambda x: x["prob"], reverse=True)
    return rows


def _pool_leverage_bands(df):
    """Tertile bands for proj/(own+4) leverage labels within the pool. Pure."""
    d = df.copy()
    proj = pd.to_numeric(d["My Proj"], errors="coerce").fillna(0.0)
    own = pd.to_numeric(d["My Own"], errors="coerce").fillna(0.0)
    lev = proj / (own + 4.0)
    return float(lev.quantile(0.33)), float(lev.quantile(0.66))


def leverage_label(proj, own, lo, hi):
    v = float(proj) / (float(own) + 4.0)
    if v >= hi:
        return "High"
    if v <= lo:
        return "Low"
    return "Med"


def player_table_for_lineup(lineup_row, df):
    """Player-by-player breakdown for one lineup. Pure.

    Returns DataFrame: Slot, Player, Pos, Team, Salary, Proj, Own %, Value, Leverage.
    Looks up players by {slot}_ID (reliable) with a name fallback.
    """
    lo, hi = _pool_leverage_bands(df)
    id_map = {str(r["ID"]): r for _, r in df.iterrows()}
    name_map = {str(r["Name"]): r for _, r in df.iterrows()}
    recs = []
    for slot in ROSTER_SLOTS:
        pid = str(lineup_row.get(f"{slot}_ID", "") or "")
        prow = id_map.get(pid)
        if prow is None:
            prow = name_map.get(str(lineup_row.get(slot, "")))
        if prow is None:
            recs.append({"Slot": slot, "Player": str(lineup_row.get(slot, "—")),
                         "Pos": slot.rstrip("123"), "Team": "", "Salary": 0,
                         "Proj": 0.0, "Own %": 0.0, "Value": 0.0, "Leverage": "—"})
            continue
        sal = float(prow.get("Salary", 0) or 0)
        proj = float(prow.get("My Proj", 0) or 0)
        own = float(prow.get("My Own", 0) or 0)
        recs.append({
            "Slot": slot,
            "Player": str(prow.get("Name", "")),
            "Pos": str(prow.get("Position", "")),
            "Team": str(prow.get("Team", "")),
            "Salary": int(sal),
            "Proj": round(proj, 2),
            "Own %": round(own, 1),
            "Value": round(proj / (sal / 1000.0), 2) if sal > 0 else 0.0,
            "Leverage": leverage_label(proj, own, lo, hi),
        })
    return pd.DataFrame(recs)


def swap_suggestions(df, lineup_row, slot, min_salary):
    """Up to 3 same-position alternatives for one roster slot. Pure.

    Only candidates that keep the lineup salary-legal are offered. Each carries
    its tradeoffs (dProj, dOwn, dSalary) plus a one-line read.
    """
    pid = str(lineup_row.get(f"{slot}_ID", "") or "")
    id_map = {str(r["ID"]): r for _, r in df.iterrows()}
    cur = id_map.get(pid)
    if cur is None:
        # name fallback
        nm = str(lineup_row.get(slot, ""))
        hits = df[df["Name"].astype(str) == nm]
        cur = hits.iloc[0] if not hits.empty else None
    if cur is None:
        return []
    pos = str(cur.get("Position", ""))
    active = df["ActiveForBuild"].astype(str).str.lower().isin(["true", "1"]) if "ActiveForBuild" in df.columns else pd.Series([True] * len(df))
    lineup_ids = {str(lineup_row.get(f"{s}_ID", "")) for s in ROSTER_SLOTS if f"{s}_ID" in lineup_row}
    pool = df[(df["Position"].astype(str) == pos) & active & (~df["ID"].astype(str).isin(lineup_ids))].copy()
    cur_sal = float(cur.get("Salary", 0) or 0)
    salary_left = float(lineup_row.get("Salary Left", 0) or 0)
    salary = float(lineup_row.get("Salary", SALARY_CAP) or SALARY_CAP)
    pool = pool[pd.to_numeric(pool["Salary"], errors="coerce").fillna(0) <= cur_sal + salary_left]
    pool = pool[(salary - cur_sal + pd.to_numeric(pool["Salary"], errors="coerce").fillna(0)) >= float(min_salary)]
    pool = pool.sort_values("My Proj", ascending=False, kind="mergesort").head(3)
    cur_proj = float(cur.get("My Proj", 0) or 0)
    cur_own = float(cur.get("My Own", 0) or 0)
    out = []
    for _, c in pool.iterrows():
        c_sal = float(c.get("Salary", 0) or 0)
        c_proj = float(c.get("My Proj", 0) or 0)
        c_own = float(c.get("My Own", 0) or 0)
        dp, do, ds = c_proj - cur_proj, c_own - cur_own, c_sal - cur_sal
        bits = []
        bits.append(f"{dp:+.1f} proj" if abs(dp) >= 0.05 else "same proj")
        bits.append("less chalk" if do < -1 else ("more chalk" if do > 1 else "similar own"))
        bits.append(f"saves ${abs(ds):,.0f}" if ds < -1 else (f"costs ${ds:,.0f}" if ds > 1 else "same salary"))
        out.append({
            "Player": str(c.get("Name", "")),
            "Team": str(c.get("Team", "")),
            "Salary": int(c_sal),
            "Proj": round(c_proj, 2),
            "Own %": round(c_own, 1),
            "dProj": round(dp, 2),
            "dOwn": round(do, 1),
            "dSalary": int(ds),
            "Tradeoff": " · ".join(bits),
        })
    return out


def _fnum(v):
    """Float-or-0.0 for messy dataframe cells. Never raises."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def postbuild_lineups_context(res, df, sim_table=None, n=20):
    """Compact text context of the built lineups for the LLM coach.

    One section per lineup: rank, grade, projection, ceiling P90, break-slate %,
    salary left, QB/stack summary, one-line game-script story, and compact roster
    lines (SLOT:Name Team $salary proj own%). Ends with the top game worlds.
    Pure; never raises; returns "" when there is nothing to describe.
    """
    try:
        if res is None or getattr(res, "empty", True):
            return ""
        lines = []
        frame = res.head(n)
        use_df = df is not None and not getattr(df, "empty", True)
        for _, row in frame.iterrows():
            rank = row.get("Rank", "?")
            grade = str(row.get("Rating", "-"))
            proj = _fnum(row.get("Projection"))
            ceil = _fnum(row.get("Ceiling P90"))
            brk = _fnum(row.get("Break Slate %"))
            try:
                left_s = f"${int(float(row.get('Salary Left', 0))):,}"
            except (TypeError, ValueError):
                left_s = str(row.get("Salary Left", ""))
            title = lineup_title(row)
            story = short_story(row)
            lines.append(
                f"#{rank} {title} | Grade {grade} | Proj {proj:.1f} | Ceil {ceil:.1f} | "
                f"Break {brk:.1f}% | {left_s} left | {story}"
            )
            if use_df:
                try:
                    pt = player_table_for_lineup(row, df)
                    ros = []
                    for _, pr in pt.iterrows():
                        ros.append(
                            f"{pr['Slot']}:{pr['Player']} {pr['Team']} "
                            f"${pr['Salary']} {_fnum(pr['Proj']):.0f}p {_fnum(pr['Own %']):.0f}%o"
                        )
                    lines.append("  " + " | ".join(ros))
                except Exception:
                    pass
        if sim_table is not None and not getattr(sim_table, "empty", True):
            try:
                wr = world_rows(sim_table)[:6]
                if wr:
                    wtxt = "; ".join(
                        f"{w['game']} {w['prob']:.0f}% ({w['archetype']}, P90 {w['p90']:.0f})"
                        for w in wr
                    )
                    lines.append(
                        "GAME WORLDS (share of sims where this game set the slate ceiling): " + wtxt
                    )
            except Exception:
                pass
        return "\n".join(lines)
    except Exception:
        return ""


def _lab_llm_available():
    """True when an OpenAI key is reachable via Streamlit secrets or env."""
    try:
        if st.secrets.get("OPENAI_API_KEY"):
            return True
    except Exception:
        pass
    return bool(os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Render: results shell
# ---------------------------------------------------------------------------

def _equity(res, sim_worlds, df):
    """Attach Ceiling P90 / Break Slate % / Sim Mean. Never raises."""
    disp = res.copy()
    try:
        eq = lineup_sim_equity(res, sim_worlds, df["Name"].astype(str).tolist())
        if not eq.empty and len(eq) == len(disp):
            for c in ["Sim Mean", "Ceiling P90", "Break Slate %"]:
                if c in eq.columns:
                    disp[c] = eq[c].to_numpy()
    except Exception:
        pass
    return disp


def _headshot_map(df):
    """Player name -> headshot URL from the pool's 'Headshot URL' column.

    That column is populated from nflverse roster data by
    apply_live_availability_guard (dfs_lab/data.py). Pure; never raises;
    returns {} when the column is absent.
    """
    m = {}
    try:
        if df is not None and "Headshot URL" in getattr(df, "columns", []):
            for _, r in df.iterrows():
                u = str(r.get("Headshot URL", "") or "").strip()
                if u and u.lower() not in ("nan", "none"):
                    m[str(r.get("Name", ""))] = u
    except Exception:
        pass
    return m


def _avatar_html(name, url):
    """Showdown-style avatar: photo <img> when a URL exists, initials otherwise.

    Never invents a URL — empty/missing input always yields the initials
    fallback, exactly like the Showdown lineup cards. Pure.
    """
    safe = _esc(name)
    initials = "".join(x[:1] for x in str(name).replace(".", " ").split()[:2]).upper() or "NFL"
    u = str(url or "").strip()
    if u and u.lower() not in ("nan", "none"):
        return (f"<span class='player-avatar rcc-avatar'><img src='{_esc(u)}' "
                f"alt='{safe}' loading='lazy'></span>")
    return f"<span class='player-avatar rcc-avatar avatar-fallback'>{_esc(initials)}</span>"


def _grade_cls(grade):
    g = str(grade or "")
    if g.startswith("A"):
        return "a"
    if g.startswith("B"):
        return "b"
    return "c"


def slim_banner_html(disp):
    """One-line build summary bar. Pure.

    e.g. '20 lineups built · Best: #1 Patrick Mahomes (KC +1) · Grade A+ ·
          Top proj 149.1 · Best ceiling P90 188.4'. Returns "" with no data.
    """
    hs = hero_stats(disp, disp)
    if not hs:
        return ""
    return (
        f"<div class='rcc-banner'><span>✅</span>"
        f"<span><b>{hs.get('lineups', 0)} lineups built</b> · Best: "
        f"<b>#{hs.get('best_rank', 1)} {_esc(hs.get('best_title', ''))}</b> · "
        f"Grade <b>{_esc(hs.get('grade', '–'))}</b> · "
        f"Top proj {hs.get('top_projection', 0):.1f} · "
        f"Best ceiling P90 {hs.get('best_ceiling', 0):.1f}</span></div>"
    )


def _lineup_card_html(row, df, shots):
    """One lineup as a card: header, story, FULL roster table with avatars, chips.

    The roster is visible, not collapsed — lineups come first. Pure; never raises.
    """
    try:
        rank = int(row.get("Rank", 0) or 0)
        grade = str(row.get("Rating", "–"))
        title = lineup_title(row)
        proj = float(row.get("Projection", 0) or 0)
        p90 = float(row.get("Ceiling P90", 0) or 0)
        brk = float(row.get("Break Slate %", 0) or 0)
        left = float(row.get("Salary Left", 0) or 0)
        story = short_story(row)
        pt = player_table_for_lineup(row, df)
        cells = "".join(
            f"<tr><td class='av'>{_avatar_html(r['Player'], shots.get(str(r['Player']), ''))}</td>"
            f"<td>{_esc(r['Slot'])}</td><td><b>{_esc(r['Player'])}</b></td>"
            f"<td>{_esc(r['Pos'])}</td><td>{_esc(r['Team'])}</td>"
            f"<td class='num'>${int(r['Salary']):,}</td>"
            f"<td class='num'>{float(r['Proj']):.2f}</td>"
            f"<td class='num'>{float(r['Own %']):.1f}%</td></tr>"
            for _, r in pt.iterrows()
        )
        notes = str(row.get("Strategy Notes", "") or "")
        coh = row.get("Coherence Score", "")
        chips = []
        if notes and notes != "Neutral build":
            chips.append(f"<span>{_esc(notes)}</span>")
        if coh != "":
            chips.append(f"<span>Coherence {_esc(coh)}</span>")
        chips.append(f"<span>QB stack +{int(row.get('QB Stack', 0) or 0)}</span>")
        chips.append(f"<span>Bring-backs {int(row.get('Bring-backs', 0) or 0)}</span>")
        chips.append(f"<span>Avg own {float(row.get('Avg Own', 0) or 0):.1f}%</span>")
        full_story = str(row.get("Lineup Story", "") or "")
        foot = f"<div class='rcc-card-foot'>{_esc(full_story)}</div>" if full_story else ""
        return (
            f"<div class='rcc-card'><div class='rcc-card-head'>"
            f"<span class='rcc-rank'>#{rank}</span>"
            f"<span class='rcc-grade rcc-grade-{_grade_cls(grade)} rcc-grade-sm'>{_esc(grade)}</span>"
            f"<div class='rcc-card-titlewrap'><div class='rcc-card-title'>{_esc(title)}</div>"
            f"<div class='rcc-card-meta'>Proj {proj:.1f} · P90 {p90:.1f} · "
            f"Break {brk:.2f}% · ${left:,.0f} left</div></div></div>"
            f"<div class='rcc-story'>📖 {_esc(story)}</div>"
            f"<div class='rcc-scroll-x'><table class='rcc-roster'><thead><tr><th></th>"
            f"<th>Slot</th><th>Player</th><th>Pos</th><th>Team</th>"
            f"<th class='num'>Salary</th><th class='num'>Proj</th><th class='num'>Own</th>"
            f"</tr></thead><tbody>{cells}</tbody></table></div>"
            f"<div class='rcc-chips'>{''.join(chips)}</div>{foot}</div>"
        )
    except Exception:
        return ""


def render_slim_banner(disp):
    """Compact single-bar build summary, replacing the hero + tiles."""
    out = slim_banner_html(disp)
    if out:
        st.markdown(out, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render: Lineups view
# ---------------------------------------------------------------------------

def _player_grid_html(pt, shots=None):
    """Player-by-player roster grid as accessible HTML. Pure.

    Used instead of st.dataframe in the Explorer so header size/contrast is
    fully controlled (canvas dataframe headers render tiny and dim).
    shots is an optional name->headshot-URL map for avatars.
    """
    shots = shots or {}
    cells = "".join(
        f"<tr><td class='av'>{_avatar_html(r['Player'], shots.get(str(r['Player']), ''))}</td>"
        f"<td><b>{_esc(r['Player'])}</b></td><td>{_esc(r['Slot'])}</td>"
        f"<td>{_esc(r['Pos'])}</td><td>{_esc(r['Team'])}</td>"
        f"<td class='num'>${int(r['Salary']):,}</td>"
        f"<td class='num'>{float(r['Proj']):.2f}</td>"
        f"<td class='num'>{float(r['Own %']):.1f}%</td>"
        f"<td class='num'>{float(r['Value']):.2f}</td>"
        f"<td>{_esc(r['Leverage'])}</td></tr>"
        for _, r in pt.iterrows()
    )
    if not cells:
        cells = "<tr><td colspan='10'>No players match.</td></tr>"
    return (
        "<div class='rcc-scroll-x'><table class='rcc-grid'><thead><tr>"
        "<th></th><th>Player</th><th>Slot</th><th>Pos</th><th>Team</th>"
        "<th class='num'>Salary</th><th class='num'>Proj</th><th class='num'>Own %</th>"
        "<th class='num'>Pts / $1k</th><th>Leverage</th>"
        "</tr></thead><tbody>" + cells + "</tbody></table></div>"
    )


def _swap_grid_html(alts):
    """Swap suggestions as accessible HTML. Pure.

    Used instead of st.dataframe so headers are larger and high-contrast.
    """
    cells = "".join(
        f"<tr><td><b>{_esc(a['Player'])}</b></td><td>{_esc(a['Team'])}</td>"
        f"<td class='num'>${int(a['Salary']):,}</td>"
        f"<td class='num'>{float(a['Proj']):.2f}</td>"
        f"<td class='num'>{float(a['Own %']):.1f}%</td>"
        f"<td class='num'>{float(a['dProj']):+.2f}</td>"
        f"<td class='num'>{float(a['dOwn']):+.1f}</td>"
        f"<td class='num'>{int(a['dSalary']):+d}</td>"
        f"<td class='wrap'>{_esc(a['Tradeoff'])}</td></tr>"
        for a in alts
    )
    return (
        "<div class='rcc-scroll-x'><table class='rcc-grid'><thead><tr>"
        "<th>Alternative</th><th>Team</th><th class='num'>Salary</th><th class='num'>Proj</th>"
        "<th class='num'>Own %</th><th class='num'>Δ Proj</th><th class='num'>Δ Own</th>"
        "<th class='num'>Δ Salary</th><th>Tradeoff</th>"
        "</tr></thead><tbody>" + cells + "</tbody></table></div>"
    )


def render_lineup_cards(disp, df):
    """Every lineup as a card with its full roster visible — no collapsed rows.

    Lineups come first; nothing is hidden behind a 'show all' gate. The sort
    control reorders the cards.
    """
    st.markdown("<div class='rcc-section-title'>📋 Your lineups</div>"
                "<div class='rcc-section-sub'>Full rosters, visible — no digging. "
                "A new user should see lineups here, not a preamble.</div>",
                unsafe_allow_html=True)
    sort_label = st.selectbox(
        "Sort lineups",
        list(SORT_OPTIONS.keys()),
        index=0,
        key="rcc_sort",
        help="Rank the portfolio by grade, projection, ceiling, break-slate rate, or salary left.",
    )
    ordered = sort_lineups(disp, sort_label)
    shots = _headshot_map(df)
    for _, row in ordered.iterrows():
        card = _lineup_card_html(row, df, shots)
        if card:
            st.markdown(card, unsafe_allow_html=True)


def render_data_view(disp):
    """Full optimizer table + CSV download. Dead last by design."""
    with st.expander("📊 Data view — full lineup table", expanded=False):
        st.caption("The complete optimizer output, last resort. The lineup cards above are the default view.")
        show_cols = ["Rank", "Rating", "Rating Score", "Projection", "Base Projection", "Scenario Delta",
                     "Salary", "Salary Left", "Avg Own", "Stack Summary"] + ROSTER_SLOTS
        dv = disp[[x for x in show_cols if x in disp.columns]].copy()
        st.dataframe(dv, hide_index=True, use_container_width=True, height=560,
                     column_config={"Rank": st.column_config.NumberColumn("Rank", width=60, pinned=True)})
        st.download_button("Download lineup analysis CSV", disp.to_csv(index=False),
                           "classic_lineups_v5.csv", "text/csv", use_container_width=True,
                           key="rcc_download")


# ---------------------------------------------------------------------------
# Render: Lineup Explorer (master-detail)
# ---------------------------------------------------------------------------

def render_explorer(disp, df):
    st.markdown("<div class='rcc-section-title'>🔍 Lineup Explorer</div>"
                "<div class='rcc-section-sub'>Pick a lineup, inspect every roster slot, and test swaps with honest tradeoffs.</div>",
                unsafe_allow_html=True)
    ranks = [int(r) for r in disp["Rank"].tolist()]
    c1, c2 = st.columns([1, 2])
    with c1:
        pick = st.selectbox(
            "Lineup",
            ranks,
            index=0,
            format_func=lambda r: f"#{r} · {lineup_title(disp[disp['Rank'] == r].iloc[0])}",
            key="rcc_explorer_pick",
        )
    with c2:
        q = st.text_input("Search players", placeholder="Type a name to filter the roster…",
                          key="rcc_explorer_search").strip().lower()
    row = disp[disp["Rank"] == int(pick)].iloc[0]
    pt = player_table_for_lineup(row, df)
    if q:
        pt = pt[pt["Player"].str.lower().str.contains(q, na=False)]
    st.markdown(_player_grid_html(pt, _headshot_map(df)), unsafe_allow_html=True)
    st.caption("Value is projected points per $1,000 of salary. Leverage compares projection to field ownership within this pool.")

    st.markdown("<div class='rcc-swap-head'><b>Test a swap</b></div>", unsafe_allow_html=True)
    s1, s2 = st.columns([1, 2])
    with s1:
        slot_pick = st.selectbox("Roster slot", ROSTER_SLOTS, key="rcc_swap_slot")
    with s2:
        cur_name = str(row.get(slot_pick, "—"))
        st.markdown(f"<div class='rcc-cur-player'>Currently in slot: <b>{_esc(cur_name)}</b></div>",
                    unsafe_allow_html=True)
    min_salary = int(st.session_state.get("classic_min_salary", 48500))
    alts = swap_suggestions(df, row, slot_pick, min_salary)
    if not alts:
        st.info("No salary-legal swap found for that slot in the active pool.")
    else:
        st.markdown(_swap_grid_html(alts), unsafe_allow_html=True)
        st.caption("Swaps are suggestions only — they are not applied to the portfolio. Change exposure or rebuild to act on one.")


# ---------------------------------------------------------------------------
# Render: Game Worlds view
# ---------------------------------------------------------------------------

def render_worlds_view(sim_table):
    st.markdown("<div class='rcc-section-title'>🌍 Game Worlds</div>"
                "<div class='rcc-section-sub'>How Aytia's 10,000 simulated game scripts split across the slate's games.</div>",
                unsafe_allow_html=True)
    rows = world_rows(sim_table)
    if not rows:
        st.info("No simulation data for this slate.")
        return
    # Bar widths are the raw probability percentages (not scaled to the largest),
    # so the visual length matches the labeled number.
    bars = "".join(
        f"<div class='rcc-world-row'><div class='rcc-world-label'>{_esc(r['game'])}<span>{r['archetype']}</span></div>"
        f"<div class='rcc-world-track'><div class='rcc-world-fill' style='width:{r['prob']:.1f}%'></div></div>"
        f"<div class='rcc-world-count'>{r['prob']:.1f}%</div></div>"
        for r in rows
    )
    st.markdown(f"<div class='rcc-world-board'>{bars}</div>", unsafe_allow_html=True)
    st.caption("Based on 10,000 simulations. Each bar is the share of game worlds in which that game produced the slate's best DFS environment — comparative simulations, not sportsbook probabilities.")
    with st.expander("Game environment detail", expanded=False):
        t = sim_table.copy()
        cols = ["Game", "Mean DFS env", "P75", "P90", "Volatility", "Slate ceiling %"]
        st.dataframe(t[[c for c in cols if c in t.columns]], hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Render: Lab+Agent view
# ---------------------------------------------------------------------------

def _agent_ask(question, packet, lineups_ctx=None):
    if not str(question or "").strip():
        return
    ans = classic_postbuild_answer(
        str(question).strip(), packet, st.session_state.get("classic_ai_chat", []),
        lineups_ctx=lineups_ctx,
    )
    st.session_state["classic_ai_chat"].append((str(question).strip(), ans))
    st.rerun()


def render_agent_view(packet, res, df=None, sim_table=None):
    st.markdown("<div class='rcc-section-title'>🧪 Lab + Agent</div>"
                "<div class='rcc-section-sub'>Challenge the build. The agent only answers from lineups Aytia actually built.</div>",
                unsafe_allow_html=True)
    st.session_state.setdefault("classic_ai_chat", [])
    lineups_ctx = postbuild_lineups_context(res, df, sim_table)
    if not _lab_llm_available():
        st.caption("Tip: add OPENAI_API_KEY in Streamlit secrets for the full conversational agent — it answers from your actual lineups.")
    top_qb = ""
    try:
        if res is not None and not res.empty:
            top_qb = parse_stack_summary(res.iloc[0].get("Stack Summary", "")).get("qb_name", "")
    except Exception:
        pass
    starters = [
        f"Why is {top_qb} the top QB?" if top_qb else "Why this QB spread?",
        "Which lineup has the best ceiling, and why?",
        f"What happens if I fade {top_qb}?" if top_qb else "What would you change for winner-take-all?",
        "How correlated is lineup #1?",
    ]
    cols = st.columns(len(starters))
    for i, col in enumerate(cols):
        with col:
            if st.button(starters[i], use_container_width=True, key=f"rcc_agent_starter_{i}"):
                _agent_ask(starters[i], packet, lineups_ctx)
    for uq, ar in st.session_state["classic_ai_chat"][-6:]:
        st.markdown(f"<div class='rcc-chat-user'><b>You</b><br>{uq}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='rcc-chat-ai'><b>Aytia</b><br>{ar}</div>", unsafe_allow_html=True)
    if st.session_state["classic_ai_chat"]:
        st.caption("Follow up")
        f1, f2, f3 = st.columns(3)
        with f1:
            if st.button("Explain the grade", key="rcc_fu_grade", use_container_width=True):
                _agent_ask("Explain how the Aytia Grade is calculated and what it rewards.", packet, lineups_ctx)
        with f2:
            if st.button("Compare #1 and #2", key="rcc_fu_cmp", use_container_width=True):
                _agent_ask("Compare lineup #1 and lineup #2: projection, ceiling, correlation, and who each one needs.", packet, lineups_ctx)
        with f3:
            if st.button("What would you change?", key="rcc_fu_chg", use_container_width=True):
                _agent_ask("What is the weakest assumption in this portfolio, and what would you change?", packet, lineups_ctx)
    with st.form("rcc_agent_form", clear_on_submit=True):
        q = st.text_input("Ask the Lab", placeholder="Why so much chalk at RB? Which game do I need most?")
        send = st.form_submit_button("ASK  ↗", type="primary", use_container_width=True)
    if send:
        _agent_ask(q, packet, lineups_ctx)
    if st.button("CLEAR CHAT HISTORY", use_container_width=True, key="rcc_agent_clear"):
        st.session_state["classic_ai_chat"] = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def render_chalk_bust_view(sim_worlds, df):
    """If chalk fails: conditional beneficiaries from the top-down sim worlds.

    For each of the most-owned players, bust worlds = worlds where he scores
    at/below the 25th percentile of his own simulated distribution. Shows the
    plays with the best average score in exactly those worlds. Never raises.
    """
    try:
        rows = chalk_bust_beneficiaries(sim_worlds, df)
    except Exception:
        return
    rows = [r for r in rows if r.get("beneficiaries")]
    if not rows:
        return
    st.markdown("<div class='rcc-section-title'>\U0001f329\ufe0f If chalk fails</div>", unsafe_allow_html=True)
    st.caption("From 10,000 top-down game worlds. Bust = bottom quartile of the chalk play's own simulated outcomes. Beneficiaries average the most in exactly those worlds — conditional leverage, not season averages.")
    cards = []
    for r in rows:
        bens = ", ".join(
            f"<b>{_esc(b['name'])}</b> ({_esc(b['team'])} · {b['cond_mean']:.1f} avg)"
            for b in r["beneficiaries"][:3]
        )
        cards.append(
            f"<div class='intel-card'><div class='intel-kicker'>If {_esc(r['chalk'])} "
            f"({_esc(r['chalk_team'])} · {r['chalk_own']:.1f}% est. own) busts</div>"
            f"<div class='intel-copy'>Top beneficiaries &rarr; {bens}</div></div>"
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


def render_results_command_center(res, df, sim_table, sim_worlds, packet):
    """Build-step results, lineups first.

    Stacked sections, no sub-tabs: slim banner, lineup cards (full rosters
    visible), game worlds, if-chalk-fails, Lab+Agent, explorer, data view last. The sticky
    main tab bar handles navigation.
    """
    disp = _equity(res, sim_worlds, df)
    render_slim_banner(disp)
    render_lineup_cards(disp, df)
    render_worlds_view(sim_table)
    render_chalk_bust_view(sim_worlds, df)
    render_agent_view(packet, disp, df, sim_table)
    render_explorer(disp, df)
    render_data_view(disp)
