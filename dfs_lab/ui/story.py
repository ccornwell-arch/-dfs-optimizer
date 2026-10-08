"""Story pages: belief-first lineup building for Aytia.

The Story tab is the deep end. The user tells Aytia what they believe about
the game(s); Aytia echoes its interpretation, previews the game worlds, and
"Use this story" translates the story into the real build knobs (priorities,
game script, team leans) then jumps to the Build page. The Build page itself
is untouched — this is purely additive.
"""

import streamlit as st

from dfs_lab.ui import nav as _navmod


# ---------------------------------------------------------------------------
# Shared interpretation helpers (pure)
# ---------------------------------------------------------------------------

def _interpret_showdown(winner, gtype, hs, aws, how_won, how_lost, drivers, fades, t1, t2):
    bits = []
    if winner and winner != "Coin flip":
        bits.append(f"{winner} win")
    elif winner == "Coin flip":
        bits.append("toss-up")
    if gtype:
        bits.append({"shootout": "a shootout", "normal": "a normal game",
                     "defensive": "a defensive struggle",
                     "blowout": "a blowout"}.get(gtype, gtype))
    score = ""
    if hs and aws:
        score = f" {aws}–{hs}"
    s = "Aytia's read: " + (", ".join(bits) + score if bits else "no story yet") + "."
    if how_won:
        s += f" {winner if winner != 'Coin flip' else 'The winner'} wins " + {
            "air": "through the air", "ground": "on the ground",
            "bigplay": "on big plays", "defense": "with defense and special teams"}.get(how_won, "") + "."
    if drivers:
        s += " Driven by " + ", ".join(drivers) + "."
    if fades:
        s += " Fading " + ", ".join(fades) + "."
    return s


def _worlds_for(gtype, winner, drivers):
    w = winner if winner and winner != "Coin flip" else "Winner"
    drv = ", ".join(drivers[:2]) if drivers else "the stars"
    base = {
        "shootout": [
            ("Shootout both ways", 35, "Both offenses hit; pay up for pieces of both."),
            (f"{w} pulls away", 25, "Winner's pass-catchers stack; loser chases."),
            ("Loser keeps pace", 15, "Back-and-forth; game-stack both sides."),
            ("Defensive surprise", 10, "Lower scoring than expected; kickers and DSTs matter."),
            (f"{drv} go off", 15, "Individual ceiling games decide it."),
        ],
        "normal": [
            ("Winner's script", 35, f"{w} controls the game; stack the winner."),
            ("Balanced attack", 25, "No single unit dominates; spread exposure."),
            ("Loser covers", 15, "Closer than expected; bring-backs live."),
            ("Defensive tilt", 10, "Field goals and punts; cheap defense pays."),
            (f"{drv} go off", 15, "Individual ceiling games decide it."),
        ],
        "defensive": [
            ("Field-goal battle", 40, "DSTs and kickers carry; fade expensive skill players."),
            ("One big play decides it", 25, "A single score breaks the tie."),
            ("Winner grinds it out", 20, "Run game and short passing win."),
            ("Special teams swing", 15, "Return TD or blocked kick flips it."),
        ],
        "blowout": [
            (f"{w} rolls", 45, "Winner's starters feast early; backups close it."),
            ("Garbage-time heroes", 25, "Loser's cheap pass-catchers pile up late points."),
            ("Winner's defense feasts", 15, "Turnovers and sacks pile up."),
            (f"{drv} go off", 15, "Individual ceiling games decide it."),
        ],
    }
    return base.get(gtype or "normal", base["normal"])


# ---------------------------------------------------------------------------
# Showdown
# ---------------------------------------------------------------------------

def render_showdown_story(df, teams):
    st.markdown('<div class="card-title">Tell the story</div>'
                '<div class="card-sub">Start with what you believe about the game. '
                'Aytia translates it into build settings — or skip this and build the usual way.</div>',
                unsafe_allow_html=True)
    if len(teams) < 2:
        st.info("Load a showdown slate to tell its story.")
        return
    t1, t2 = teams[0], teams[1]

    names = df["Name"].astype(str).tolist()
    id_by_name = dict(zip(df["Name"].astype(str), df["ID"].astype(str)))

    st.markdown("#### Who wins?")
    st.session_state.setdefault("story_sd_winner", "Coin flip")
    winner = st.segmented_control("Winner", [t1, t2, "Coin flip"],
                                  key="story_sd_winner")
    st.markdown("#### What kind of game?")
    st.session_state.setdefault("story_sd_gtype", "normal")
    gtype = st.segmented_control("Game type",
                                 ["shootout", "normal", "defensive", "blowout"],
                                 key="story_sd_gtype")
    st.markdown("#### Expected score")
    s1, s2 = st.columns(2)
    st.session_state.setdefault("story_sd_as", 24)
    st.session_state.setdefault("story_sd_hs", 21)
    with s1:
        aws = st.number_input(f"{t1} points", min_value=0, max_value=70,
                              key="story_sd_as")
    with s2:
        hs = st.number_input(f"{t2} points", min_value=0, max_value=70,
                             key="story_sd_hs")
    st.markdown("#### How does the winner win?")
    st.session_state.setdefault("story_sd_how_won", "air")
    how_won = st.segmented_control(
        "How won", ["air", "ground", "bigplay", "defense"],
        format_func=lambda x: {"air": "Air raid", "ground": "Ground control",
                               "bigplay": "Big plays", "defense": "Defense & ST"}[x],
        key="story_sd_how_won")
    st.markdown("#### How does the loser score?")
    st.session_state.setdefault("story_sd_how_lost", "pace")
    how_lost = st.segmented_control(
        "How lost", ["garbage", "pace", "shutdown"],
        format_func=lambda x: {"garbage": "Garbage time", "pace": "Keeps pace",
                               "shut": "Shut down", "shutdown": "Shut down"}[x],
        key="story_sd_how_lost")
    st.markdown("#### Who drives the story? (up to 3)")
    st.session_state.setdefault("story_sd_drivers", [])
    drivers = st.multiselect("Drivers", names,
                             max_selections=3, key="story_sd_drivers",
                             label_visibility="collapsed")
    st.markdown("#### Who disappoints?")
    st.session_state.setdefault("story_sd_fades", [])
    fades = st.multiselect("Disappointments", names,
                           key="story_sd_fades", label_visibility="collapsed")

    st.markdown("#### Aytia's read")
    st.info(_interpret_showdown(winner, gtype, hs, aws, how_won, how_lost,
                                drivers, fades, t1, t2))

    st.markdown("#### Game worlds")
    st.caption("Your story splits into worlds; the portfolio spreads across them. 15% stays hedged against the story.")
    for wname, pct, why in _worlds_for(gtype, winner, drivers):
        st.markdown(f"**{wname}** — {pct}%<br><span style='opacity:.7'>{why}</span>",
                    unsafe_allow_html=True)
        st.progress(pct / 100.0)

    if st.button("Use this story → build lineups", type="primary",
                 use_container_width=True, key="story_sd_apply"):
        _apply_showdown_story(df, id_by_name, winner, gtype, t1, t2,
                              aws, hs, drivers, fades)
        st.toast("Story applied to build settings")
        _navmod.goto_page("Showdown", "⚡ Build")


def _apply_showdown_story(df, id_by_name, winner, gtype, t1, t2, aws, hs, drivers, fades):
    """Translate the story into real build knobs. Only touches Neutral priors."""
    strat = st.session_state.setdefault("showdown_strategy", {})
    for nm in drivers:
        pid = id_by_name.get(nm)
        if not pid:
            continue
        cur = strat.setdefault(pid, {})
        if cur.get("Priority", "Neutral") == "Neutral":
            cur["Priority"] = "Like"
    for nm in fades:
        pid = id_by_name.get(nm)
        if not pid:
            continue
        cur = strat.setdefault(pid, {})
        if cur.get("Priority", "Neutral") == "Neutral":
            cur["Priority"] = "Fade"
    script_map = {"shootout": "Shootout", "normal": "Neutral",
                  "defensive": "Defensive / field-goal battle", "blowout": "Team dominates"}
    st.session_state["sd_script"] = script_map.get(gtype, "Neutral")
    if winner and winner != "Coin flip":
        st.session_state["sd_script_team"] = winner
        if gtype == "blowout":
            st.session_state["sd_script"] = "Team dominates"
    else:
        st.session_state["sd_script_team"] = "None"
    if aws and hs:
        st.session_state["sd_use_score"] = True
        st.session_state["sd_score_0"] = float(aws)
        st.session_state["sd_score_1"] = float(hs)


# ---------------------------------------------------------------------------
# Classic
# ---------------------------------------------------------------------------

def _classic_games(df):
    """Derive (game_label, team_a, team_b) from the pool."""
    info = df.get("Game Info", df.get("Matchup", None))
    games = {}
    if info is not None:
        for _, r in df[["Team", "Opponent"]].assign(_gi=info.astype(str)).iterrows():
            gi = str(r["_gi"]).strip()
            tm = str(r["Team"]).strip().upper()
            if not gi or gi.lower() == "nan" or not tm:
                continue
            key = tuple(sorted([tm, str(r["Opponent"]).strip().upper()]))
            games[key] = gi
    out = []
    for (a, b), gi in sorted(games.items()):
        out.append((f"{a} vs {b}", a, b))
    return out


def render_classic_story(df, teams):
    st.markdown('<div class="card-title">Tell the story</div>'
                '<div class="card-sub">Your slate thesis, then stories only for the games '
                'you have a take on. Untold games run on Aytia\'s baseline.</div>',
                unsafe_allow_html=True)
    games = _classic_games(df)
    game_labels = [g[0] for g in games] or ["No games detected"]
    names = df["Name"].astype(str).tolist()
    id_by_name = dict(zip(df["Name"].astype(str), df["ID"].astype(str)))

    st.markdown("#### My slate thesis")
    love = st.selectbox("Game I love", game_labels,
                        index=_idx(game_labels, st.session_state.get("story_cl_love")),
                        key="story_cl_love")
    bust = st.selectbox("Game I think busts", game_labels,
                        index=_idx(game_labels, st.session_state.get("story_cl_bust")),
                        key="story_cl_bust")
    lev = st.selectbox("Favorite leverage", names,
                       index=_idx(names, st.session_state.get("story_cl_leverage")),
                       key="story_cl_leverage")
    fade = st.selectbox("Fading", names,
                        index=_idx(names, st.session_state.get("story_cl_fade")),
                        key="story_cl_fade")

    st.markdown("#### Games — tell a story where you have an edge")
    told = 0
    for label, a, b in games:
        key = f"story_cl_game_{a}_{b}"
        with st.expander(label, expanded=False):
            w = st.segmented_control("Who wins?", [a, b, "Coin flip"],
                                     default="Coin flip", key=key + "_w")
            gt = st.segmented_control("Game type",
                                      ["shootout", "normal", "defensive", "blowout"],
                                      default="normal", key=key + "_t")
            gp = [n for n in names if n in df[df["Team"].astype(str).str.upper().isin([a, b])]["Name"].astype(str).tolist()]
            dr = st.multiselect("Drivers (up to 3)", gp, max_selections=3,
                                key=key + "_d", label_visibility="collapsed")
            if w != "Coin flip" or gt != "normal" or dr:
                told += 1
                st.caption(_interpret_showdown(w, gt, "", "", "air", "pace", dr, [], a, b))
    if not games:
        st.caption("Upload a slate to list its games.")
    else:
        st.caption(f"{told} storied game(s) · {len(games)-told} on Aytia baseline · 15% hedge reserved.")

    if st.button("Use my stories → build lineups", type="primary",
                 use_container_width=True, key="story_cl_apply"):
        _apply_classic_story(df, id_by_name, games, love, bust, lev, fade)
        st.toast("Stories applied to build settings")
        _navmod.goto_page("Classic", "⚡ Build")


def _idx(options, val):
    try:
        return options.index(val)
    except ValueError:
        return 0


def _apply_classic_story(df, id_by_name, games, love, bust, lev, fade):
    """Translate the thesis into real build knobs. Only touches Neutral priors."""
    strat = st.session_state.setdefault("strategy_master", {})
    tstrat = st.session_state.setdefault("team_strategy_master", {})
    glabels = {g[0]: (g[1], g[2]) for g in games}
    if love in glabels:
        for tm in glabels[love]:
            if tstrat.get(tm, "Neutral") == "Neutral":
                tstrat[tm] = "Like"
    if bust in glabels and bust != love:
        for tm in glabels[bust]:
            if tstrat.get(tm, "Neutral") == "Neutral":
                tstrat[tm] = "Fade"
    for nm, prio in ((lev, "Like"), (fade, "Fade")):
        pid = id_by_name.get(nm)
        if pid:
            cur = strat.setdefault(pid, {})
            if cur.get("Priority", "Neutral") == "Neutral":
                cur["Priority"] = prio
