"""Story detour: optional belief-first entry to building lineups.

NOT a nav tab — reached via "Tell the story" from the Build page. The user
can skip it entirely; the Build page works the same without a story.

"Apply story & build lineups" writes the story into the real build knobs,
then requests a build and jumps to the Lineups page, where render_main
executes the optimizer (same code path as the GENERATE button).
"""

import streamlit as st


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


def _pw(widget_fn, pkey, default, *args, **kwargs):
    """Persistent widget: value survives detour navigation.

    Mirrors the widget value to a non-widget session key via on_change and
    re-seeds the widget from it if Streamlit drops the unmounted widget's
    state. Same pattern as main.persistent_widget (imported lazily to avoid
    a circular import at module load).
    """
    from dfs_lab.ui.main import persistent_widget
    return persistent_widget(widget_fn, pkey, default, *args, **kwargs)


# ---------------------------------------------------------------------------
# Showdown
# ---------------------------------------------------------------------------

def render_showdown_story(df, teams):
    if st.button("← Back to Build", key="story_sd_back"):
        st.switch_page("pages/showdown_build.py")
    st.markdown('<div class="card-title">Tell the story <span style="font-size:12px;opacity:.6">(optional)</span></div>'
                '<div class="card-sub">If you have a take on this game, tell it here and Aytia will '
                'build from it. No take? Head back — the Build page works the same without a story.</div>',
                unsafe_allow_html=True)
    if len(teams) < 2:
        st.info("Load a showdown slate to tell its story.")
        return
    t1, t2 = teams[0], teams[1]

    names = df["Name"].astype(str).tolist()
    id_by_name = dict(zip(df["Name"].astype(str), df["ID"].astype(str)))

    st.markdown("#### Who wins?")
    winner = _pw(st.segmented_control, "story_sd_winner", "Coin flip",
                 "Winner", [t1, t2, "Coin flip"])
    st.markdown("#### What kind of game?")
    gtype = _pw(st.segmented_control, "story_sd_gtype", "normal",
                "Game type", ["shootout", "normal", "defensive", "blowout"])
    st.markdown("#### Expected score")
    s1, s2 = st.columns(2)
    with s1:
        aws = _pw(st.number_input, "story_sd_as", 24, f"{t1} points",
                  min_value=0, max_value=70)
    with s2:
        hs = _pw(st.number_input, "story_sd_hs", 21, f"{t2} points",
                 min_value=0, max_value=70)
    st.markdown("#### How does the winner win?")
    how_won = _pw(
        st.segmented_control, "story_sd_how_won", "air", "How won",
        ["air", "ground", "bigplay", "defense"],
        format_func=lambda x: {"air": "Air raid", "ground": "Ground control",
                               "bigplay": "Big plays", "defense": "Defense & ST"}[x])
    st.markdown("#### How does the loser score?")
    how_lost = _pw(
        st.segmented_control, "story_sd_how_lost", "pace", "How lost",
        ["garbage", "pace", "shutdown"],
        format_func=lambda x: {"garbage": "Garbage time", "pace": "Keeps pace",
                               "shutdown": "Shut down"}[x])
    st.markdown("#### Who drives the story? (up to 3)")
    drivers = _pw(st.multiselect, "story_sd_drivers", [], "Drivers", names,
                  max_selections=3, label_visibility="collapsed")
    st.markdown("#### Who disappoints?")
    fades = _pw(st.multiselect, "story_sd_fades", [], "Disappointments", names,
                label_visibility="collapsed")

    st.markdown("#### Aytia's read")
    st.info(_interpret_showdown(winner, gtype, hs, aws, how_won, how_lost,
                                drivers, fades, t1, t2))

    st.markdown("#### Game worlds")
    st.caption("Your story splits into worlds; the portfolio spreads across them. 15% stays hedged against the story.")
    for wname, pct, why in _worlds_for(gtype, winner, drivers):
        st.markdown(f"**{wname}** — {pct}%<br><span style='opacity:.7'>{why}</span>",
                    unsafe_allow_html=True)
        st.progress(pct / 100.0)

    if st.button("Apply story & build lineups", type="primary",
                 use_container_width=True, key="story_sd_build"):
        _apply_showdown_story(df, id_by_name, winner, gtype, t1, t2,
                              aws, hs, drivers, fades)
        # Request a real optimizer run; the Lineups page executes it on load
        # (same code path as the Build page's GENERATE button).
        st.session_state["story_build_requested"] = True
        st.switch_page("pages/showdown_lineups.py")


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
    if st.button("← Back to Build", key="story_cl_back"):
        st.switch_page("pages/classic_build.py")
    st.markdown('<div class="card-title">Tell the story <span style="font-size:12px;opacity:.6">(optional)</span></div>'
                '<div class="card-sub">Your slate thesis, then stories only for the games '
                'you have a take on. Untold games run on Aytia\'s baseline. No take? '
                'Head back — the Build page works the same without a story.</div>',
                unsafe_allow_html=True)
    games = _classic_games(df)
    game_labels = [g[0] for g in games] or ["No games detected"]
    names = df["Name"].astype(str).tolist()
    id_by_name = dict(zip(df["Name"].astype(str), df["ID"].astype(str)))

    st.markdown("#### My slate thesis")
    love = _pw(st.selectbox, "story_cl_love", game_labels[0],
               "Game I love", game_labels)
    bust = _pw(st.selectbox, "story_cl_bust", game_labels[0],
               "Game I think busts", game_labels)
    lev = _pw(st.selectbox, "story_cl_leverage", names[0] if names else "",
              "Favorite leverage", names)
    fade = _pw(st.selectbox, "story_cl_fade", names[0] if names else "",
               "Fading", names)

    st.markdown("#### Games — tell a story where you have an edge")
    told = 0
    for label, a, b in games:
        key = f"story_cl_game_{a}_{b}"
        with st.expander(label, expanded=False):
            w = _pw(st.segmented_control, key + "_w_p", "Coin flip",
                    "Who wins?", [a, b, "Coin flip"])
            gt = _pw(st.segmented_control, key + "_t_p", "normal",
                     "Game type", ["shootout", "normal", "defensive", "blowout"])
            gp = [n for n in names if n in df[df["Team"].astype(str).str.upper().isin([a, b])]["Name"].astype(str).tolist()]
            dr = _pw(st.multiselect, key + "_d_p", [], "Drivers", gp,
                     max_selections=3, label_visibility="collapsed")
            if w != "Coin flip" or gt != "normal" or dr:
                told += 1
                st.caption(_interpret_showdown(w, gt, "", "", "air", "pace", dr, [], a, b))
    if not games:
        st.caption("Upload a slate to list its games.")
    else:
        st.caption(f"{told} storied game(s) · {len(games)-told} on Aytia baseline · 15% hedge reserved.")

    if st.button("Apply stories & build lineups", type="primary",
                 use_container_width=True, key="story_cl_build"):
        _apply_classic_story(df, id_by_name, games, love, bust, lev, fade)
        st.session_state["story_build_requested"] = True
        st.switch_page("pages/classic_lineups.py")


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
