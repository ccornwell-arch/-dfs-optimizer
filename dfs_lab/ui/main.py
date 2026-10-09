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


from dfs_lab import styles
from dfs_lab.common import player_editor_widget_key
from dfs_lab.config import APP_BUILD, PRIORITY_OPTIONS, ROSTER_SLOTS, git_build_stamp
from dfs_lab.leverage import leverage_lane_pick, chalk_bust_beneficiaries
from dfs_lab.ui.results import render_results_command_center, postbuild_lineups_context
from dfs_lab.ui import persist
from dfs_lab.data import prepare_player_pool, prepare_showdown_pool, apply_projection_overrides, apply_override_rescue, apply_post_edit_availability_gate, availability_freshness_note
from dfs_lab.classic import (generate_lineups, classic_apply_qb_cap, classic_apply_qb_exclusions,
    classic_contest_recommendations, classic_context_evidence,
    classic_portfolio_intelligence, classic_postbuild_answer, classic_postbuild_report,
    classic_qb_concentration_plan, topdown_simulate_slate, classic_strategy_theses, bringback_worthy_teams,
    calculate_exposure_table)
from dfs_lab.showdown import (apply_context_engine, apply_showdown_scenario,
    audit_min_exposure, audit_showdown_portfolio, captain_pool_ids,
    default_world_max_share,
    generate_showdown_lineups, infer_score_script, script_build_adjustments,
    showdown_exposure_table, showdown_upload_csv)
from dfs_lab.theme import theme_css


def _sd_nav_go(tab):
    """Programmatic nav jump for the Showdown workspace.

    Must run as a button on_click callback: setting a widget's session-state
    key after that widget was already instantiated in the same run raises
    StreamlitWidgetAlreadyInstantiatedError, so the assignment has to happen
    in the pre-run callback phase, not in the script body. Now jumps to the
    real page via st.switch_page (multipage nav).
    """
    from dfs_lab.ui import nav as _navmod
    _navmod.goto_page("Showdown", tab)


def _classic_nav_go(tab):
    """Programmatic nav jump for the Classic workspace (see _sd_nav_go)."""
    from dfs_lab.ui import nav as _navmod
    _navmod.goto_page("Classic", tab)


def persistent_widget(widget_fn, persistent_key, default, *args, **kwargs):
    """Create a widget whose value survives tab navigation.

    Only the active tab's widgets render on a run. If Streamlit drops an
    unmounted widget's session state, a plain keyed widget reverts to its
    default when the user returns to that tab. This helper mirrors the value
    to a persistent (non-widget) key via on_change and re-seeds the widget
    from it on remount, so Build/Scenario settings never silently reset.

    Returns the widget's current value. Read the persistent key (not the
    widget key) in build logic.
    """
    st.session_state.setdefault(persistent_key, default)
    widget_key = f"_w_{persistent_key}"
    # Re-seed the widget from the persistent value if its state was dropped
    # (e.g. the tab was not rendered for a run).
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state[persistent_key]
    def _on_change():
        st.session_state[persistent_key] = st.session_state[widget_key]
    kwargs["key"] = widget_key
    # Chain with any caller-supplied on_change.
    _caller_on_change = kwargs.pop("on_change", None)
    _caller_args = kwargs.pop("on_change_args", ())
    _caller_kwargs = kwargs.pop("on_change_kwargs", {})
    def _chained():
        _on_change()
        if _caller_on_change:
            _caller_on_change(*_caller_args, **_caller_kwargs)
    kwargs["on_change"] = _chained
    return widget_fn(*args, **kwargs)


def resolve_saved_priority(excluded, table_priority):
    """Single source of truth for the saved Priority value.

    Never persist the contradictory state {Exclude: False, Priority: "Exclude"}:
    the status strip and the solver both treat Priority="Exclude" as an
    exclusion, so a stale "Exclude" lean kept a player out of builds after the
    user unchecked the Out box (Oct 8, 2026: Dak Prescott showed unchecked in
    the table while the strip listed him Out and the solver excluded him).
    """
    tp = str(table_priority or "Neutral")
    if excluded:
        return "Exclude"
    return "Neutral" if tp == "Exclude" else tp


def _apply_exposure_targets(ed, exp, edited_rows, strategy):
    """Apply data_editor exposure-target edits to the strategy map. Pure.

    ed: display table (Player, Target Min %, ...) indexed by row position.
    exp: showdown_exposure_table output (carries the ID column).
    edited_rows: {row_index: {column: new_value}} from the editor widget.
    Returns (applied_count, bad_players): rows whose min exceeds max are
    skipped and reported instead of being written.
    """
    bad = []
    n = 0
    for ri, chg in (edited_rows or {}).items():
        ri = int(ri)
        drow = ed.iloc[ri]
        pid = str(exp.iloc[ri]["ID"])
        tmin = float(chg.get("Target Min %", drow["Target Min %"]))
        tmax = float(chg.get("Target Max %", drow["Target Max %"]))
        cmin = float(chg.get("CPT Target Min %", drow["CPT Target Min %"]))
        cmax = float(chg.get("CPT Target Max %", drow["CPT Target Max %"]))
        if tmin > tmax or cmin > cmax:
            bad.append(str(drow["Player"]))
            continue
        strat = strategy.setdefault(pid, {})
        strat.update({"Min Exposure": tmin, "Max Exposure": tmax, "CPT Min": cmin, "CPT Max": cmax})
        n += 1
    return n, bad


def _auto_intel_adopt_manual(context, snapshot, manual):
    """Mark user-touched Game Intel cells as manual. Pure.

    A cell is manual when its current value differs from what auto-fill last
    wrote (or from the neutral default if auto-fill never touched it). The
    auto-fill pass skips manual cells, so anything the user typed survives.
    """
    _cols = ["Defense", "Usage", "Home/Rest", "Travel", "Time/Split", "Confidence", "Note"]
    _defaults = {"Defense": 0, "Usage": 0, "Home/Rest": 0, "Travel": 0,
                 "Time/Split": 0, "Confidence": 50, "Note": ""}
    for pid, cfg in (context or {}).items():
        for col in _cols:
            cur = (cfg or {}).get(col, _defaults[col])
            old = (snapshot or {}).get(str(pid), {}).get(col, _defaults[col])
            key = f"{pid}|{col}"
            if cur != old:
                manual.add(key)
            else:
                manual.discard(key)
    return manual


def _sd_lineup_card_html(lr, headshot_map):
    """Full-roster lineup card HTML for a Showdown portfolio row.

    Same look as the 'Your lineups' top cards: captain feature, all five
    FLEX players, salary/construction/dup meta and the grade strip. Used by
    the Game Worlds view so a selected world shows full lineups, not just a
    summary table. Pure: no Streamlit calls.
    """
    import html as _h

    def _avatar(name, captain=False):
        safe = _h.escape(str(name))
        url = _h.escape(headshot_map.get(str(name), ""))
        cls = "player-avatar captain-avatar" if captain else "player-avatar flex-avatar"
        if url:
            return f"<div class='{cls}'><img src='{url}' alt='{safe}'></div>"
        initials = "".join([x[:1] for x in str(name).replace(".", " ").split()[:2]]).upper() or "NFL"
        return f"<div class='{cls} avatar-fallback'>{_h.escape(initials)}</div>"

    world = str(lr.get("Game World", ""))
    captain = str(lr["Captain"])
    flexes = [str(lr.get("FLEX" + str(i), "")) for i in range(1, 6)]
    flex_people = "".join(
        "<div class='flex-person'>" + _avatar(nm, False) + "<span>" + _h.escape(nm) + "</span></div>"
        for nm in flexes if nm
    )
    card = ("<div class='lineup-card lineup-card-grid'><div class='lineup-card-head'><div class='lineup-rank'>#%s</div><div class='lineup-grade'>%s</div></div>" % (int(lr["Rank"]), lr["Rating"]) +
            "<div class='captain-feature'>" + _avatar(captain, True) + "<div><div class='lineup-points'>%.1f <small>pts</small></div>" % lr["Projection"] +
            "<div class='lineup-cpt'><span>CPT</span> %s</div></div></div>" % _h.escape(captain) +
            "<div class='lineup-flex-grid'>%s</div>" % flex_people +
            "<div class='lineup-meta'>$%s &nbsp; • &nbsp; %s &nbsp; • &nbsp; %s dup risk</div>" % (f"{int(lr['Salary']):,}", lr["Construction"], lr["Dup Risk"]) +
            "<div class='lineup-why-strip'><span>%s DUP</span><span>%s CORR</span><span>%s LEVERAGE</span></div>" % (_h.escape(str(lr.get("Dup Risk", "—")).upper()), _h.escape(str(lr.get("Correlation Grade", "—")).upper()), _h.escape(str(lr.get("Leverage Grade", "—")).upper())) +
            (("<div class='lineup-world'>◉ &nbsp; %s</div>" % _h.escape(world)) if world else "") + "</div>")
    return card


@st.cache_resource
def _dfs_lab_slate_cache():
    return {}


def render_main(settings):
    """Post-upload flow (moved verbatim from streamlit_app.py)."""
    mode = settings["mode"]
    field_size = settings["field_size"]
    entry_format = settings["entry_format"]
    payout_style = settings["payout_style"]
    lineup_count = settings["lineup_count"]
    my_entries = int(settings.get("my_entries", lineup_count))
    seed = settings["seed"]

    # House theme first: inject every style block before any UI renders, so the
    # setup screen (which stops the script before any upload) is dark from the
    # first paint — no light-background flash that later flips to dark.
    # Single color scheme: Aytia Gold house theme. (Team theming scrapped
    # Sep 2026; dfs_lab/theme.py is kept for a future revisit.)
    st.markdown(styles.MAIN_V634_SHELL_CSS, unsafe_allow_html=True)
    st.markdown(styles.MAIN_DYNAMIC_SHELL_CSS, unsafe_allow_html=True)
    st.markdown(styles.MAIN_UNIFORM_THEME_CSS, unsafe_allow_html=True)
    st.markdown(styles.MAIN_EXPANDERS_CSS, unsafe_allow_html=True)
    st.markdown(styles.MAIN_EXPANDER_HEADERS_CSS, unsafe_allow_html=True)
    st.markdown(styles.MAIN_WHY_STRIP_CSS, unsafe_allow_html=True)
    st.markdown(styles.RCC_DARK_CSS, unsafe_allow_html=True)
    st.markdown(theme_css(), unsafe_allow_html=True)

    # Keep each browser's last successfully uploaded slate alive across ordinary page refreshes.
    # The cache key is stored in the URL so a new Streamlit session can recover the same files.
    if "slate_session" not in st.query_params:
        import secrets
        st.query_params["slate_session"] = secrets.token_urlsafe(10)
    _slate_session = str(st.query_params.get("slate_session", "default"))
    _slate_cache = _dfs_lab_slate_cache()
    _cached = _slate_cache.get(_slate_session, {})
    if not st.session_state.get("dfs_lab_dk_bytes") and _cached.get("dk_bytes"):
        st.session_state["dfs_lab_dk_bytes"] = _cached["dk_bytes"]
        st.session_state["dfs_lab_dk_name"] = _cached.get("dk_name", "DKSalaries.csv")
    if not st.session_state.get("dfs_lab_ss_bytes") and _cached.get("ss_bytes"):
        st.session_state["dfs_lab_ss_bytes"] = _cached["ss_bytes"]
        st.session_state["dfs_lab_ss_name"] = _cached.get("ss_name", "SaberSim.csv")

    _has_cached_slate=bool(st.session_state.get("dfs_lab_dk_bytes"))
    with st.expander("✓ SLATE LOADED · Change files" if _has_cached_slate else "＋ LOAD SLATE FILES", expanded=not _has_cached_slate):
        st.markdown('<div class="card-sub">DraftKings is required. Without a SaberSim file, Aytia builds its own projections and estimated ownership.</div>',unsafe_allow_html=True)
        u1,u2=st.columns(2)
        with u1: dk_file=st.file_uploader("DraftKings salaries/template · required",type=["csv"],key="dfs_lab_dk_upload")
        with u2: ss_file=st.file_uploader("SaberSim · optional comparison",type=["csv"],key="dfs_lab_ss_upload")

    # Keep a working copy of uploaded bytes during ordinary Streamlit reruns. This prevents
    # widget refreshes from forcing the user to remove/re-add the same slate.
    if dk_file is not None:
        st.session_state["dfs_lab_dk_bytes"]=dk_file.getvalue(); st.session_state["dfs_lab_dk_name"]=getattr(dk_file,"name","DKSalaries.csv")
        _slate_cache[_slate_session] = {**_slate_cache.get(_slate_session, {}), "dk_bytes": st.session_state["dfs_lab_dk_bytes"], "dk_name": st.session_state["dfs_lab_dk_name"]}
    elif st.session_state.get("dfs_lab_dk_bytes"):
        dk_file=io.BytesIO(st.session_state["dfs_lab_dk_bytes"]); dk_file.name=st.session_state.get("dfs_lab_dk_name","DKSalaries.csv")
    if ss_file is not None:
        st.session_state["dfs_lab_ss_bytes"]=ss_file.getvalue(); st.session_state["dfs_lab_ss_name"]=getattr(ss_file,"name","SaberSim.csv")
        _slate_cache[_slate_session] = {**_slate_cache.get(_slate_session, {}), "ss_bytes": st.session_state["dfs_lab_ss_bytes"], "ss_name": st.session_state["dfs_lab_ss_name"]}
    elif st.session_state.get("dfs_lab_ss_bytes"):
        ss_file=io.BytesIO(st.session_state["dfs_lab_ss_bytes"]); ss_file.name=st.session_state.get("dfs_lab_ss_name","SaberSim.csv")

    if dk_file is None:
        st.info("Upload the DraftKings slate to open Aytia.")
        st.stop()

    if mode=="Classic":
        try:
            df=prepare_player_pool(dk_file,ss_file); teams=sorted(df["Team"].dropna().unique().tolist())
            if bool(df.get("Own Estimated",pd.Series([False])).any()):
                st.info("Using Aytia projections — no SaberSim file uploaded. Ownership shown is Aytia's estimate, built from projection, salary value and position baselines.")
            st.session_state.setdefault("strategy_master",{}); st.session_state.setdefault("team_strategy_master",{})
            st.session_state.setdefault("classic_projection_overrides",{})
            # 'Aytia remembers': re-apply persisted player edits after a session
            # death (iOS tab suspend). No-op when the dicts already have data
            # or the slate differs.
            persist.restore_player_state(df, "Classic")
            df["Base Proj"]=pd.to_numeric(df["My Proj"],errors="coerce").fillna(0.0)
            if st.session_state["classic_projection_overrides"]:
                for _i,_r in df.iterrows():
                    _pid=str(_r["ID"])
                    if _pid in st.session_state["classic_projection_overrides"]:
                        df.at[_i,"My Proj"]=float(st.session_state["classic_projection_overrides"][_pid])
            # Final safety gate (mirrors the Showdown path): a projection edited
            # to zero AFTER pool creation must take the player out of the build.
            # Without this, ActiveForBuild stays True from the original upload.
            df=apply_post_edit_availability_gate(df,"My Proj","ActiveForBuild",0.05)
            # Rescue path for the silent-override trust bug: a typed projection
            # on an auto-excluded player puts them back in the pool, unless the
            # player is officially OUT or the user marked them Out. The notes
            # surface in the Build tab, above the Generate button.
            df,_rescued_c,_blocked_c=apply_override_rescue(df,st.session_state.get("classic_projection_overrides",{}),st.session_state.get("strategy_master",{}))
            st.session_state["_classic_rescue_notes"]=(_rescued_c,_blocked_c)
            st.session_state.setdefault("classic_ai_chat",[])
            st.session_state.setdefault("classic_ai_model","")
            st.session_state.setdefault("classic_ai_error","")
            st.session_state.setdefault("classic_qb_cap",0)
            _classic_chat_fp=f"{entry_format}|{int(field_size)}|{payout_style}|{getattr(dk_file,'name','slate')}"
            if st.session_state.get("classic_ai_chat_fingerprint") != _classic_chat_fp:
                st.session_state["classic_ai_chat"]=[]
                st.session_state["classic_qb_cap"]=0
                st.session_state["classic_ai_chat_fingerprint"]=_classic_chat_fp
            st.session_state.setdefault("classic_qb_stack",2)
            st.session_state.setdefault("classic_bringback","Optional")
            st.session_state.setdefault("classic_min_salary",48500)
            st.session_state.setdefault("classic_max_team",5)
            st.session_state.setdefault("classic_max_game",5)
            st.session_state.setdefault("classic_max_te",2)
            st.session_state.setdefault("classic_no_dst",True)
            st.session_state.setdefault("classic_no_off",False)
            st.session_state.setdefault("classic_allow_qb_rb",True)
            st.session_state.setdefault("classic_flex_control",False)
            st.session_state.setdefault("classic_flex_rb",45)
            st.session_state.setdefault("classic_flex_wr",45)
            st.session_state.setdefault("classic_flex_te",10)
            st.session_state.setdefault("classic_thesis_applied","")
            st.session_state.setdefault("classic_pref_stack",[])

            st.markdown(styles.MAIN_TABS_CSS, unsafe_allow_html=True)

            # Story detour: the Story page sets this flag and calls render_main;
            # render only the story (no nav, no sections) and return.
            if st.session_state.pop("_story_detour", None) == "classic":
                from dfs_lab.ui.story import render_classic_story
                render_classic_story(df, teams)
                return

            # Pool summary cards live above the tabs only pre-build. Post-build the
            # Lineups tab leads with the results banner, so these cards would just
            # push lineups down the page for no reason.
            _built_now = st.session_state.get("classic_result_v4")
            _built_now = _built_now is not None and not getattr(_built_now, "empty", True)
            if not _built_now:
                q1,q2,q3,q4=st.columns(4)
                q1.metric("Players",len(df)); q2.metric("Teams",len(teams)); q3.metric("Field",f"{int(field_size):,}"); q4.metric("Pool",lineup_count)

            intel=classic_context_evidence(df[["Name","Position","Team","Opponent","Game Info","My Proj"]].copy())
            intel_map=intel.set_index("Name") if not intel.empty else pd.DataFrame()
            sim_result=topdown_simulate_slate(df, sims=10000, seed=seed)
            sim_table=sim_result["game_table"]
            player_sim=sim_result["player_stats"]
            sim_worlds=sim_result["worlds"]
            bb_worthy=bringback_worthy_teams(df)
            _slate_teams=sorted(set(df["Team"].astype(str).str.upper().tolist()))
            _bb_weak=[t for t in _slate_teams if t not in bb_worthy]
            thesis_table,thesis_state=classic_strategy_theses(df,intel,sim_table,field_size,payout_style,entry_format)
            classic_qb_ids,qb_plan_table,qb_plan=classic_qb_concentration_plan(df,thesis_table,entry_format)
            classic_qb_ids,qb_plan_table,qb_plan=classic_apply_qb_cap(classic_qb_ids,qb_plan_table,qb_plan,st.session_state.get("classic_qb_cap",0))
            # User QB-pool control: unchecking a QB in the Slate Intel table removes
            # them from the build. Exclusions are scoped to the current slate
            # fingerprint so a new upload starts with a clean pool.
            _qb_pool_fp=f"{entry_format}|{int(field_size)}|{payout_style}|{getattr(dk_file,'name','slate')}"
            if st.session_state.get("classic_qb_excluded_fp")!=_qb_pool_fp:
                st.session_state["classic_qb_excluded"]=set()
                st.session_state["classic_qb_excluded_fp"]=_qb_pool_fp
            _qb_excluded=set(str(x) for x in (st.session_state.get("classic_qb_excluded") or set()))
            _qb_base_ids=[str(x) for x in classic_qb_ids]
            # The editor key includes the slate and the eligible pool, so the widget
            # (and its edited_rows) resets whenever the pool itself changes — a stale
            # row-position edit must never exclude the wrong quarterback.
            _qb_editor_key="classic_qb_pool_editor|"+_qb_pool_fp+"|"+",".join(_qb_base_ids)
            _qb_order_ids=[]
            if qb_plan_table is not None and not qb_plan_table.empty and "ID" in qb_plan_table.columns:
                _qb_order_ids=qb_plan_table["ID"].astype(str).head(12).tolist()
            _ed_state=st.session_state.get(_qb_editor_key)
            if isinstance(_ed_state,dict):
                for _rk,_chg in ((_ed_state.get("edited_rows") or {}).items()):
                    try:_pos=int(_rk)
                    except Exception:continue
                    if 0<=_pos<len(_qb_order_ids) and isinstance(_chg,dict) and "In build pool" in _chg:
                        _qid=str(_qb_order_ids[_pos])
                        if bool(_chg["In build pool"]):_qb_excluded.discard(_qid)
                        else:_qb_excluded.add(_qid)
                st.session_state["classic_qb_excluded"]=set(_qb_excluded)
            _qb_pool_kept_one=len(_qb_base_ids)>1 and all(str(x) in _qb_excluded for x in _qb_base_ids)
            if _qb_pool_kept_one:
                # Keep the display truthful: the top QB stays in the pool, so it
                # leaves the excluded set and the editor resets to the corrected
                # state instead of showing a checked QB as unchecked.
                _qb_excluded.discard(str(_qb_base_ids[0]))
                st.session_state["classic_qb_excluded"]=set(_qb_excluded)
                st.session_state.pop(_qb_editor_key,None)
            classic_qb_ids=classic_apply_qb_exclusions(_qb_base_ids,_qb_excluded)
            if qb_plan_table is not None and not qb_plan_table.empty and "ID" in qb_plan_table.columns:
                qb_plan_table=qb_plan_table.copy()
                qb_plan_table["In build pool"]=qb_plan_table["ID"].astype(str).isin(set(str(x) for x in classic_qb_ids))
            rec=classic_contest_recommendations(field_size,payout_style,entry_format,sim_table)
            lane=leverage_lane_pick(df)


            # Post-build packet (hoisted): the Slate Intel coach and the Results
            # Command Center's Lab+Agent tab both answer from this same packet.

            _portfolio_result=st.session_state.get("classic_result_v4")
            portfolio=classic_portfolio_intelligence(df,_portfolio_result)
            packet={
                "contest":{"entry_format":entry_format,"field_size":int(field_size),"payout":payout_style,"requested_lineups":int(lineup_count)},
                "recommendations":rec,
                "active_rules":{"qb_pass_catchers":int(st.session_state["classic_qb_stack"]),"bringback":st.session_state["classic_bringback"],
                                "min_salary":int(st.session_state["classic_min_salary"]),"max_team":int(st.session_state["classic_max_team"]),
                                "max_game":int(st.session_state["classic_max_game"]),"max_te":int(st.session_state["classic_max_te"]),
                                "no_dst_from_qb_game":bool(st.session_state["classic_no_dst"]),"no_offense_vs_dst":bool(st.session_state["classic_no_off"]),
                                "allow_qb_rb":bool(st.session_state["classic_allow_qb_rb"])},
                "thesis_state":thesis_state,
                "qb_concentration":qb_plan,
                "qb_candidates":qb_plan_table.head(20).to_dict(orient="records") if qb_plan_table is not None and not qb_plan_table.empty else [],
                "theses":thesis_table.head(15).to_dict(orient="records") if thesis_table is not None and not thesis_table.empty else [],
                "simulations":sim_table.head(12).to_dict(orient="records") if sim_table is not None and not sim_table.empty else [],
                "context_players":[
                    {
                        "ID":str(_r["ID"]),"Name":str(_r["Name"]),"Position":str(_r["Position"]),"Team":str(_r["Team"]),
                        "Opponent":str(_r.get("Opponent","")),"Matchup":str(_r.get("Matchup","")),"Salary":int(_r["Salary"]),
                        "My Proj":round(float(_r["My Proj"]),2),"My Own":round(float(_r["My Own"]),2),
                        "Hist FPPG":round(float(intel_map.loc[_r["Name"],"Hist FPPG"]),2) if (not intel.empty and _r["Name"] in intel_map.index) else 0.0,
                        "Recent 6":round(float(intel_map.loc[_r["Name"],"Recent 6"]),2) if (not intel.empty and _r["Name"] in intel_map.index) else 0.0,
                        "DVP Adj %":round(float(intel_map.loc[_r["Name"],"DVP Adj %"]),2) if (not intel.empty and _r["Name"] in intel_map.index) else 0.0
                    } for _,_r in df.sort_values("My Proj",ascending=False).head(120).iterrows()
                ],
                "portfolio":portfolio,
                "limitations":["No injury/news feed in this Classic build","No weather/travel feed in this Classic build","Day/night history is not inferred when unavailable"]
            }
            _CLASSIC_TABS=["🧠 Slate Intel","⚡ Build","👤 Players","⚙ Rules","📋 Lineups","📊 Exposure","📖 Guide"]
            # Multipage nav: the segmented control switches real Streamlit pages
            # (pages/*.py) via st.switch_page, so each section is its own screen
            # instead of a tab on one long scrolling page. The bar is pinned via
            # CSS on its keyed container (.st-key-classic_nav) so it stays visible.
            st.session_state.setdefault("classic_nav","🧠 Slate Intel")
            def _classic_nav_changed():
                from dfs_lab.ui import nav as _navmod
                _navmod.goto_page("Classic", st.session_state.get("classic_nav"))
            classic_nav=st.segmented_control("Aytia workspace",_CLASSIC_TABS,key="classic_nav",label_visibility="collapsed",on_change=_classic_nav_changed)
            _setup_c1,_setup_c2=st.columns([5,1])
            with _setup_c2:
                if st.button("⚙ Setup",key="classic_goto_setup",use_container_width=True,
                             help="Contest settings and slate files"):
                    st.switch_page("streamlit_app.py")

            # --- Nav-independent build inputs (Classic) ---
            # The generate handler below runs on every run, but build_btn and
            # preferred_stack_teams live in the Build section. Derive them from
            # session state so builds and exposure rebuilds work from any tab.
            build_btn=False
            preferred_stack_teams=list(st.session_state.get("classic_pref_stack",[]))
            if classic_nav==_CLASSIC_TABS[0]:
                st.markdown('<div class="card-title">Aytia Slate Intel</div><div class="card-sub">Study the slate first. Then decide which optimizer rules deserve to be used for this contest.</div>',unsafe_allow_html=True)
                contest_desc=f"{entry_format} · {int(field_size):,} entries · {payout_style}"
                st.markdown(f"<div class='intel-card'><div class='intel-kicker'>Contest lens</div><div class='intel-big'>{contest_desc}</div><div class='intel-copy'>Aytia changes its recommendations with field size, entry format and payout shape. The same slate should not be built the same way in Single Entry and 150-Max.</div></div>",unsafe_allow_html=True)

                # Leverage Lane: one highest-leverage play per slate — projection
                # edge over salary-implied expectation, per unit of ownership.
                if lane:
                    import html as _html
                    _own_lbl=(f"{lane['own']:.1f}% estimated ownership" if lane["own_estimated"] else f"{lane['own']:.1f}% ownership")
                    st.markdown(f"<div class='intel-card'><div class='intel-kicker'>Leverage Lane — highest-leverage play</div><div class='intel-big'>{_html.escape(lane['name'])} · {lane['position']} · {lane['team']} · ${lane['salary']:,}</div><div class='intel-copy'>Projects {lane['proj']:.1f} pts (+{lane['edge']:.1f} over salary-implied) at {_own_lbl}. The most projection edge per unit of field ownership on the slate.</div></div>",unsafe_allow_html=True)

                if sim_table is not None and not sim_table.empty:
                    top=sim_table.iloc[0]
                    st.markdown(f"<div class='intel-card'><div class='intel-kicker'>10,000-simulation slate read</div><div class='intel-big'>{top['Game']} has the strongest simulated ceiling footprint</div><div class='intel-copy'>It produced the highest DFS environment in {float(top['Slate ceiling %']):.1f}% of top-down game simulations. P90 environment: {float(top['P90']):.1f}. Each game is simulated from team offense/defense ratings, so stacks and bring-backs emerge from correlated game scripts. These are comparative DFS simulations, not sportsbook game probabilities.</div></div>",unsafe_allow_html=True)
                    sim_cols=["Game","Mean DFS env","P75","P90","Volatility","Slate ceiling %"]
                    st.dataframe(sim_table[[x for x in sim_cols if x in sim_table.columns]],hide_index=True,use_container_width=True,height=min(430,70+35*len(sim_table)))
                if player_sim is not None and not player_sim.empty:
                    with st.expander("Simulation Lab — 10,000 top-down game worlds",expanded=False):
                        st.caption("Sim Mean should track My Proj (the simulation is built on it). Sim P90 and P(3x) describe ceiling: how often a player breaks the slate in a correlated game script.")
                        _ps=player_sim.sort_values("Sim P90",ascending=False)
                        st.dataframe(_ps,hide_index=True,use_container_width=True,height=420)

                st.markdown("#### Context evidence")
                st.caption("Aytia blends the uploaded slate with multi-year player results and opponent-vs-position evidence. Current day/night is shown when DraftKings Game Info exposes kickoff time. Travel, weather and injury/news are not invented when the current data feed does not supply them.")
                intel_view=intel.copy()
                if not intel_view.empty:
                    intel_view["Proj"]=intel_view["My Proj"].round(2)
                    intel_view["Hist FPPG"]=intel_view["Hist FPPG"].round(2)
                    intel_view["Recent 6"]=intel_view["Recent 6"].round(2)
                    cols=["Name","Position","Team","Opponent","Proj","Hist FPPG","Recent 6","Hist Games","DVP Adj %","Time"]
                    st.dataframe(intel_view[[x for x in cols if x in intel_view.columns]].sort_values("Proj",ascending=False).head(80),hide_index=True,use_container_width=True,height=430)

                st.markdown("#### Strategy theses")
                st.markdown(f"<div class='intel-card'><div class='intel-kicker'>Aytia stance</div><div class='intel-big'>{thesis_state['label']}</div><div class='intel-copy'>{thesis_state['reason']}</div></div>",unsafe_allow_html=True)
                st.caption("A thesis can start with a game, QB, receiver, or RB. Using one adds weight to that route — it does not lock the portfolio to that game.")
                if thesis_table is not None and not thesis_table.empty:
                    thesis_cols=["Type","Thesis","Attention %","Team","Game","Paired QB","Why"]
                    st.dataframe(thesis_table[[x for x in thesis_cols if x in thesis_table.columns]].head(10),
                        hide_index=True,use_container_width=True,height=min(430,70+35*min(10,len(thesis_table))))
                    top_thesis=thesis_table.iloc[0]
                    if st.button("USE TOP THESIS AS SOFT LEAN",type="primary",use_container_width=True,key="classic_apply_thesis"):
                        focus=[]
                        if str(top_thesis.get("Team","")).strip():
                            focus=[str(top_thesis["Team"])]
                        elif str(top_thesis.get("Game","")).strip() and "@" in str(top_thesis["Game"]):
                            focus=[x.strip() for x in str(top_thesis["Game"]).split("@") if x.strip()]

                        # A thesis should influence the build, not restrict it to one game/QB path.
                        # Preferred QB stack teams remain a separate explicit HARD control in Build.
                        st.session_state["classic_pref_stack"]=[]

                        pname=str(top_thesis.get("Player","")).strip()
                        if pname:
                            hit=df[df["Name"].astype(str).eq(pname)]
                            if not hit.empty:
                                pid=str(hit.iloc[0]["ID"])
                                cur=st.session_state["strategy_master"].get(pid,{})
                                if cur.get("Priority","Neutral")=="Neutral":
                                    cur["Priority"]="Like"
                                st.session_state["strategy_master"][pid]=cur

                        for tm in [x for x in focus if x in teams]:
                            if st.session_state["team_strategy_master"].get(tm,"Neutral")=="Neutral":
                                st.session_state["team_strategy_master"][tm]="Like"

                        st.session_state["classic_thesis_applied"]=str(top_thesis["Thesis"])
                        st.success("Top thesis added as a soft lean. Aytia can still build through other games and quarterbacks.")
                    if st.session_state.get("classic_thesis_applied"):
                        st.caption("Active thesis lean: "+str(st.session_state["classic_thesis_applied"]))

                st.markdown("#### QB concentration")
                st.markdown(f"<div class='intel-card'><div class='intel-kicker'>Contest concentration</div><div class='intel-big'>{qb_plan['label']}</div><div class='intel-copy'>{qb_plan['reason']}</div></div>",unsafe_allow_html=True)
                if qb_plan_table is not None and not qb_plan_table.empty:
                    qshow=qb_plan_table[["QB","Team","Game","Relative %","In build pool","Why"]].head(12).reset_index(drop=True)
                    st.data_editor(qshow,
                        hide_index=True,use_container_width=True,height=min(430,70+35*min(12,len(qshow))),
                        disabled=["QB","Team","Game","Relative %","Why"],
                        column_config={"In build pool":st.column_config.CheckboxColumn("In build pool",help="Uncheck to remove this QB from the build pool.")},
                        key=_qb_editor_key)
                    if _qb_pool_kept_one:
                        st.info("You unchecked every QB, so Aytia kept the top-ranked option — the pool can't be empty. Re-check any QB to widen it.")
                st.caption("Uncheck a QB to remove them from the build pool. Exclusions apply to the next build and reset when you upload a new slate. Single Entry and 3-Max intentionally narrow weak QB paths so candidate lineups express a stance. 150-Max keeps a much wider evidence band for portfolio coverage.")

                st.markdown("#### Aytia recommended setup")
                rr1,rr2,rr3,rr4=st.columns(4)
                rr1.metric("QB pass catchers",rec["qb_stack"])
                rr2.metric("Bring-back",rec["bringback"])
                rr3.metric("Salary floor","$"+f"{int(rec['min_salary']):,}")
                rr4.metric("Max from one game",rec["max_game"])
                st.caption(f"Built for {entry_format} in a {int(field_size):,}-entry {payout_style.lower()} contest. Top simulated game ceiling share: {rec['top_game_share']:.1f}%. Recommendations are staged only when you choose Apply.")
                if st.button("APPLY Aytia RECOMMENDED RULES",type="primary",use_container_width=True,key="classic_apply_intel"):
                    st.session_state["classic_qb_stack"]=int(rec["qb_stack"])
                    st.session_state["classic_bringback"]=str(rec["bringback"])
                    st.session_state["classic_min_salary"]=int(rec["min_salary"])
                    st.session_state["classic_max_team"]=int(rec["max_team"])
                    st.session_state["classic_max_game"]=int(rec["max_game"])
                    st.session_state["classic_max_te"]=int(rec["max_te"])
                    st.session_state["classic_no_dst"]=bool(rec["no_dst"])
                    st.session_state["classic_no_off"]=bool(rec["no_off"])
                    st.session_state["classic_allow_qb_rb"]=bool(rec["allow_qb_rb"])
                    st.success("Aytia recommendations staged in Rules. Review them before building.")


                if portfolio.get("built"):
                    st.markdown("#### Portfolio intelligence")
                    pi1,pi2,pi3,pi4=st.columns(4)
                    pi1.metric("Built lineups",portfolio.get("lineups",0))
                    pi2.metric("QBs used",portfolio.get("unique_qbs",0))
                    pi3.metric("Avg salary left",f"$"+f"{float(portfolio.get('salary_left',{}).get('mean') or 0):,.0f}")
                    pi4.metric("Avg player own",f"{float(portfolio.get('avg_player_ownership',{}).get('mean') or 0):.1f}%")
                    _qbtxt=" · ".join(f"{x['qb']} {x['exposure_pct']:.0f}%" for x in portfolio.get("qb_usage",[])[:6])
                    st.caption("QB portfolio · "+(_qbtxt or "No QB usage available"))

                st.markdown("#### Post-Build Coach")
                st.caption(f"Build: {APP_BUILD} · The coach only evaluates lineups Aytia actually built. It does not change rules or pretend to know evidence that is not in the portfolio.")

                if not portfolio.get("built"):
                    st.info("Generate lineups to unlock the Post-Build Coach. Slate Intel stays focused on pre-build research.")
                else:
                    findings=classic_postbuild_report(packet)
                    if findings:
                        for _title,_body in findings:
                            st.markdown(f"<div class='intel-card'><div class='intel-kicker'>{_title}</div><div class='intel-copy'>{_body}</div></div>",unsafe_allow_html=True)

                    _why_cols=st.columns(4)
                    _why_prompts=["Why this QB spread?","Why this stack mix?","How different for winner-take-all?","What would you change?"]
                    _why_ask=None
                    for _i,_qq in enumerate(_why_prompts):
                        with _why_cols[_i]:
                            if st.button(_qq,use_container_width=True,key=f"classic_coach_quick_{_i}"):
                                _why_ask=_qq

                    if st.session_state["classic_ai_chat"]:
                        for uq,ar in st.session_state["classic_ai_chat"][-6:]:
                            st.markdown(f"<div class='dfs-chat-user'><b>You</b><br>{uq}</div>",unsafe_allow_html=True)
                            st.markdown(f"<div class='dfs-chat-ai'><b>Aytia</b><br>{ar}</div>",unsafe_allow_html=True)

                    with st.form("classic_postbuild_coach_form",clear_on_submit=True):
                        _coach_q=st.text_input("Ask why",placeholder="Why so much Purdy? Why only 2 double stacks? Are these lineups too chalky?")
                        _coach_send=st.form_submit_button("ASK WHY  ↗",type="primary",use_container_width=True)

                    _coach_asked=(_coach_q.strip() if _coach_send and _coach_q.strip() else _why_ask)
                    if _coach_asked:
                        _coach_ctx=None
                        try:
                            _coach_ctx=postbuild_lineups_context(st.session_state.get("classic_result_v4"),df,sim_table)
                        except Exception:
                            _coach_ctx=None
                        _coach_answer=classic_postbuild_answer(_coach_asked,packet,st.session_state["classic_ai_chat"],lineups_ctx=_coach_ctx)
                        st.session_state["classic_ai_chat"].append((_coach_asked,_coach_answer))
                        st.rerun()

                    if st.button("CLEAR COACH HISTORY",use_container_width=True,key="classic_coach_clear"):
                        st.session_state["classic_ai_chat"]=[]
                        st.rerun()

            if classic_nav==_CLASSIC_TABS[1]:
                st.markdown('<div class="card-title">Build</div><div class="card-sub">Choose team preferences after reviewing Slate Intel, then generate the portfolio with your Rules settings.</div>',unsafe_allow_html=True)
                # Optional story detour: not a required step, just a different way in.
                with st.expander("💡 Tell the story first (optional)", expanded=False):
                    st.caption("Have a take on this slate? Tell it as a story and Aytia will translate it into build settings. Or ignore this and build the usual way — nothing here is required.")
                    if st.button("Tell the story →", key="classic_goto_story", use_container_width=True):
                        st.switch_page("pages/classic_story.py")
                preferred_stack_teams=st.multiselect("Preferred QB stack teams",teams,key="classic_pref_stack",
                    help="Hard control: if you select teams here, the optimizer must use a QB from one of them. Strategy Theses no longer populate this automatically.")
                team_df=pd.DataFrame({"Team":teams,"Priority":[st.session_state["team_strategy_master"].get(t,"Neutral") for t in teams]})
                st.caption("Your inputs, not the sim's verdict — tell Aytia which teams you want more or less of. The sim's own reads live in Slate Intel → Strategy Theses.")
                team_edit=st.data_editor(team_df,hide_index=True,use_container_width=True,disabled=["Team"],column_config={"Priority":st.column_config.SelectboxColumn("Your lean",options=["Core","Like","Neutral","Fade","Exclude"])},key="v4_classic_team")
                for _,r in team_edit.iterrows(): st.session_state["team_strategy_master"][r["Team"]]=r["Priority"]
                if st.session_state.get("classic_thesis_applied"):
                    st.markdown("**Build thesis:** "+str(st.session_state["classic_thesis_applied"]))
                st.markdown("**Automatic QB stance:** "+", ".join(qb_plan.get("names",[])) if qb_plan.get("names") else "**Automatic QB stance:** open")
                st.caption(qb_plan.get("reason",""))
                if int(st.session_state.get("classic_qb_cap",0) or 0)>0:
                    st.info("User QB cap · no more than "+str(int(st.session_state["classic_qb_cap"]))+" quarterbacks · applies to the next build")
                st.info("Current rule set · QB + "+str(st.session_state["classic_qb_stack"])+" pass catcher(s) · Bring-back "+str(st.session_state["classic_bringback"])+" · Min salary $"+f"{int(st.session_state['classic_min_salary']):,}"+" · Max "+str(st.session_state["classic_max_game"])+" from one game")
                # "How can we know this beforehand": who Aytia auto-excluded and
                # why, shown BEFORE the Generate button. Reuses the guard's own
                # Auto Excluded Reason column — no new plumbing.
                _auto=df[~df["ActiveForBuild"].astype(bool)].copy()
                _auto=_auto[_auto["Auto Excluded Reason"].astype(str).str.strip().ne("")]
                with st.expander(f"Auto-excluded by Aytia ({len(_auto)}) — who's out before you build",expanded=False):
                    if _auto.empty:
                        st.caption("Nothing auto-excluded. The full player pool is available for this build.")
                    else:
                        st.dataframe(
                            _auto[["Name","Position","Team","Salary","Auto Excluded Reason"]].rename(columns={"Auto Excluded Reason":"Why"}),
                            hide_index=True,use_container_width=True,
                            column_config={
                                "Name":st.column_config.TextColumn("Player",width=170),
                                "Position":st.column_config.TextColumn("Pos",width=55),
                                "Team":st.column_config.TextColumn("Team",width=65),
                                "Salary":st.column_config.NumberColumn("Salary",format="$%d",width=80),
                                "Why":st.column_config.TextColumn("Why auto-excluded",width=420),
                            },
                        )
                    st.caption("Backup QBs, players with no usable projection, and live OUT/inactive statuses are removed before the build — no manual exclusion needed.")
                _fresh_note = availability_freshness_note(df)
                if _fresh_note:
                    st.warning(_fresh_note)
                _rescued_c,_blocked_c=st.session_state.get("_classic_rescue_notes",([],[]))
                for _nm,_why in _rescued_c:
                    st.info(f"🔓 {_nm} is back in the pool — your projection overrode the auto-exclusion ({_why}).")
                for _nm,_why in _blocked_c:
                    st.warning(f"⚠️ Your projection for {_nm} was NOT applied — {_why}.")
                build_btn=st.button(f"⚡ GENERATE {lineup_count} RATED LINEUPS",type="primary",use_container_width=True,key="v4_classic_build")

            if classic_nav==_CLASSIC_TABS[2]:
                st.markdown('<div class="card-title">Players</div><div class="card-sub">Edit your own projection when you disagree with the model. Your projection becomes the number Aytia uses for simulations and lineup building until you reset it.</div>',unsafe_allow_html=True)
                view=df.copy(); team_filter=st.multiselect("Teams",teams,key="v4_cteam"); pos_filter=st.multiselect("Positions",["QB","RB","WR","TE","DST"],key="v4_cpos")
                if team_filter:view=view[view["Team"].isin(team_filter)]
                if pos_filter:view=view[view["Position"].isin(pos_filter)]
                s1,s2=st.columns([2,1])
                with s1:
                    sort_by=st.selectbox("Sort players by",["Salary","Projection","Ownership","Name"],index=0,key="classic_player_sort")
                with s2:
                    sort_dir=st.segmented_control("Order",["High → Low","Low → High"],default="High → Low",key="classic_player_sort_dir")
                _sort_col={"Salary":"Salary","Projection":"My Proj","Ownership":"My Own","Name":"Name"}[sort_by]
                _ascending=(sort_dir=="Low → High")
                if sort_by=="Name":
                    _ascending=(sort_dir=="Low → High")
                view=view.sort_values(_sort_col,ascending=_ascending,kind="mergesort").reset_index(drop=True)
                ed=pd.DataFrame({"ID":view["ID"].astype(str),"Name":view["Name"],"Pos":view["Position"],"Team":view["Team"],"Opponent":view["Opponent"],"Salary":view["Salary"],"Base Proj":view["Base Proj"].round(2),"Proj":view["My Proj"].round(2),"Own":pd.to_numeric(view["My Own"],errors="coerce").fillna(0.0).round(1),"Lock":False,"Exclude":False,"Priority":"Neutral","Min Exposure":0,"Max Exposure":100})
                for x,r in ed.iterrows():
                    e=st.session_state["strategy_master"].get(str(r["ID"]),{})
                    for cc,k,dv in [("Lock","Lock",False),("Exclude","Exclude",False),("Priority","Priority","Neutral"),("Min Exposure","Min Exposure",0),("Max Exposure","Max Exposure",100)]: ed.at[x,cc]=e.get(k,dv)
                # The editor's pending edits are row-positional. If the visible
                # order changes (sort/filter) before APPLY is pressed, a fixed
                # key would silently land the check on the wrong player — the
                # reported "I marked him Out but he's in my lineups" bug. Scope
                # the key to the exact visible order so pending edits reset
                # instead of misapplying, and warn when that discard happens.
                _ced_key=player_editor_widget_key("v4_classic_players",_qb_pool_fp,view["ID"].astype(str).tolist())
                _ced_prev=st.session_state.get("v4_classic_players_key")
                if _ced_prev and _ced_prev!=_ced_key:
                    _ced_old=st.session_state.get(_ced_prev)
                    if isinstance(_ced_old,dict) and _ced_old.get("edited_rows"):
                        st.warning("Player list order changed before you pressed APPLY PLAYER EDITS — unapplied Out/Lock checks were discarded (not applied to the wrong players). Please re-check them and apply.")
                    st.session_state.pop(_ced_prev,None)
                st.session_state["v4_classic_players_key"]=_ced_key
                with st.form("classic_player_editor_form",clear_on_submit=False):
                    apply_classic_players_top=st.form_submit_button("APPLY PLAYER EDITS",type="primary",use_container_width=True,key="classic_apply_top")
                    edited=st.data_editor(ed,hide_index=True,use_container_width=True,height=620,
                        disabled=["ID","Name","Pos","Team","Opponent","Salary","Base Proj","Own"],
                        # iOS Safari ignores Streamlit's pinned columns, so Out/Lock sit
                        # directly beside the player name — no horizontal scroll needed
                        # to see which row is being edited.
                        column_order=["Name","Exclude","Lock","Pos","Team","Opponent","Salary","Base Proj","Proj","Own","Priority","Min Exposure","Max Exposure"],
                        column_config={
                            "Name":st.column_config.TextColumn("Player",width=190,pinned=True),"Pos":st.column_config.TextColumn("Pos",width=60),
                            "Team":st.column_config.TextColumn("Team",width=70),"Opponent":st.column_config.TextColumn("Opponent",width=82),
                            "Salary":st.column_config.NumberColumn("Salary",width=85,format="$%d"),
                            "Base Proj":st.column_config.NumberColumn("Base Proj",width=85,format="%.2f",help="Original uploaded/model projection."),
                            "Proj":st.column_config.NumberColumn("My Proj",width=85,format="%.2f",min_value=0.0,step=0.25,help="Editable. Aytia uses this value everywhere after you apply changes."),
                            "Own":st.column_config.NumberColumn("Own %",width=70,format="%.1f"),
                            "Priority":st.column_config.SelectboxColumn("Lean",options=PRIORITY_OPTIONS,width=95),"Lock":st.column_config.CheckboxColumn("Lock",width=65),
                            "Exclude":st.column_config.CheckboxColumn("Out",width=60),"Min Exposure":st.column_config.NumberColumn("Min %",min_value=0,max_value=100,step=5,width=75),
                            "Max Exposure":st.column_config.NumberColumn("Max %",min_value=0,max_value=100,step=5,width=75),
                        },key=_ced_key)
                    pc1,pc2=st.columns(2)
                    with pc1:
                        apply_classic_players=st.form_submit_button("APPLY PLAYER EDITS",type="primary",use_container_width=True)
                    with pc2:
                        reset_classic_proj=st.form_submit_button("RESET VISIBLE PROJECTIONS",use_container_width=True)
                st.caption("⚠️ Applying edits clears the built lineups below — rebuild afterwards. If nothing changed, your lineups are kept.")
                if reset_classic_proj:
                    for _,r in edited.iterrows():
                        st.session_state["classic_projection_overrides"].pop(str(r["ID"]),None)
                    st.rerun()
                if apply_classic_players or apply_classic_players_top:
                    # The old "APPLY PLAYER CHANGES" silently wiped built lineups even
                    # when nothing was edited. Now: only clear the portfolio when an
                    # edit actually changed something.
                    changed=False
                    for _,r in edited.iterrows():
                        pid=str(r["ID"]); ex=bool(r["Exclude"])
                        new_entry={"Lock":bool(r["Lock"]) and not ex,"Exclude":ex,"Priority":resolve_saved_priority(ex, r["Priority"]),"Min Exposure":float(r["Min Exposure"]),"Max Exposure":float(r["Max Exposure"])}
                        if st.session_state["strategy_master"].get(pid,{}) != new_entry:
                            changed=True
                        st.session_state["strategy_master"][pid]=new_entry
                        base=float(r["Base Proj"]); newp=float(r["Proj"])
                        if abs(newp-base)>=0.005:
                            if st.session_state["classic_projection_overrides"].get(pid) != newp:
                                changed=True
                            st.session_state["classic_projection_overrides"][pid]=newp
                        else:
                            if pid in st.session_state["classic_projection_overrides"]:
                                changed=True
                            st.session_state["classic_projection_overrides"].pop(pid,None)
                    if changed:
                        st.session_state.pop("classic_result_v4",None)
                        st.toast("✅ Player edits applied")
                        # Pending row-positional edits are now materialized in
                        # strategy_master. Drop the widget's pending-edit state so a
                        # later sort/filter change doesn't warn about "unapplied"
                        # edits that were already applied. (Pop the key: mutating
                        # nested widget state is forbidden by Streamlit.)
                        st.session_state.pop(_ced_key,None)
                        st.rerun()
                    else:
                        st.info("No edits detected — your built lineups are untouched.")

            if classic_nav==_CLASSIC_TABS[3]:
                st.markdown('<div class="card-title">Classic Rules</div><div class="card-sub">Contest-aware structure controls. Slate Intel can recommend these, but you decide what gets enforced.</div>',unsafe_allow_html=True)
                r1,r2=st.columns(2)
                with r1:
                    min_salary=st.slider("Minimum salary",44000,50000,int(st.session_state["classic_min_salary"]),100,key="classic_min_salary")
                    qb_stack=st.selectbox("QB pass catchers",[1,2,3],key="classic_qb_stack",help="Minimum same-team WR/TE players paired with the QB. Mobile QBs (rushing is the correlation) need one fewer; naked rushing-QB builds are allowed.")
                    bringback_mode=st.selectbox("Bring-back",["Optional","Required","None"],key="classic_bringback",help="Optional nudges lineups toward one opposing skill player when the other offense is good enough to shoot out (league-average scoring or better) — a correlation lean, not a mandate. Required forces at least one opposing RB/WR/TE with the QB stack under the same worthiness gate. None forbids opposing skill players with the QB outright.")
                    _mob_names=df.loc[df["Mobile QB"].fillna(False).astype(bool),"Name"].astype(str).tolist() if "Mobile QB" in df.columns else []
                    if _mob_names:
                        st.caption("Mobile QBs need one fewer pass catcher: "+", ".join(_mob_names[:8])+(f" (+{len(_mob_names)-8} more)" if len(_mob_names)>8 else ""))
                    if _bb_weak:
                        st.caption(f"No forced bring-backs from weak offenses: {', '.join(_bb_weak)}. A bring-back only pays when the other side can score.")
                with r2:
                    max_players_team=st.selectbox("Max players from one team",[4,5,6,7,8,9],key="classic_max_team")
                    max_players_game=st.selectbox("Max players from one game",[4,5,6,7,8,9],key="classic_max_game")
                    max_te=st.selectbox("Max tight ends",[1,2,3],key="classic_max_te")
                st.markdown("#### FLEX mix")
                flex_control=st.toggle("Control FLEX position mix",key="classic_flex_control",
                    help="Applies to the portfolio as a whole. Example: 40% RB / 50% WR / 10% TE means roughly that share of generated lineups will use each position in FLEX.")
                if flex_control:
                    fa,fb,fc=st.columns(3)
                    with fa:
                        flex_rb=st.slider("RB in FLEX %",0,100,int(st.session_state["classic_flex_rb"]),5,key="classic_flex_rb")
                    with fb:
                        flex_wr=st.slider("WR in FLEX %",0,100,int(st.session_state["classic_flex_wr"]),5,key="classic_flex_wr")
                    with fc:
                        flex_te=st.slider("TE in FLEX %",0,100,int(st.session_state["classic_flex_te"]),5,key="classic_flex_te")
                    _flex_sum=int(flex_rb)+int(flex_wr)+int(flex_te)
                    if _flex_sum!=100:
                        st.error(f"FLEX percentages must total 100%. Current total: {_flex_sum}%.")
                    else:
                        _n=int(lineup_count)
                        st.caption(
                            f"Target across {_n} lineups: about "
                            f"{round(_n*int(flex_rb)/100)} RB · {round(_n*int(flex_wr)/100)} WR · {round(_n*int(flex_te)/100)} TE in FLEX."
                        )
                else:
                    st.caption("Automatic: Aytia chooses the FLEX position independently for each lineup.")

                st.markdown("#### Correlation + defense")
                no_dst=st.toggle("No defense from my QB's game",key="classic_no_dst",help="Blocks either defense from the game containing your rostered QB.")
                no_off=st.toggle("No offense against my DST",key="classic_no_off",help="If on, any DST blocks all opposing offensive players.")
                allow_qb_rb=st.toggle("Allow QB + same-team RB",key="classic_allow_qb_rb",help="Turn off if you want QB stacks to avoid same-team running backs.")
                st.caption("QB vs opposing DST is always blocked. More relationship controls can be added here without changing the core optimizer.")

            _story_build=st.session_state.pop("story_build_requested", False)
            if build_btn or _story_build:
                res=generate_lineups(
                    df,field_size,payout_style,lineup_count,max(300,lineup_count*12),
                    int(st.session_state["classic_min_salary"]),int(st.session_state["classic_qb_stack"]),st.session_state["classic_bringback"],
                    preferred_stack_teams,st.session_state["strategy_master"],st.session_state["team_strategy_master"],
                    bool(st.session_state["classic_no_dst"]),bool(st.session_state["classic_no_off"]),seed,
                    max_players_team=int(st.session_state["classic_max_team"]),
                    max_players_game=int(st.session_state["classic_max_game"]),
                    max_te=int(st.session_state["classic_max_te"]),
                    allow_qb_with_rb=bool(st.session_state["classic_allow_qb_rb"]),
                    allowed_qb_ids=classic_qb_ids,
                    flex_mix=({"RB":int(st.session_state["classic_flex_rb"]),"WR":int(st.session_state["classic_flex_wr"]),"TE":int(st.session_state["classic_flex_te"])}
                              if bool(st.session_state.get("classic_flex_control",False))
                              and int(st.session_state["classic_flex_rb"])+int(st.session_state["classic_flex_wr"])+int(st.session_state["classic_flex_te"])==100
                              else None),
                    bringback_worthy=bb_worthy
                )
                st.session_state["classic_result_v4"]=res
                # Slate Intel / Post-Build Coach renders earlier in the script than the Build tab.
                # Rerun once after a successful build so those sections immediately see the new portfolio.
                # A story-triggered build jumps straight to the Lineups page instead.
                if _story_build:
                    st.switch_page("pages/classic_lineups.py")
                st.rerun()

            res=st.session_state.get("classic_result_v4")
            if classic_nav==_CLASSIC_TABS[4]:
                if res is None or res.empty:
                    st.info("Generate lineups from Build.")
                else:
                    # Post-build quick jumps: Players / Exposure / Build are one tap away.
                    _cq1,_cq2,_cq3=st.columns(3)
                    _cq1.button("👤 Adjust Players",key="classic_jump_players",use_container_width=True,on_click=_classic_nav_go,args=(_CLASSIC_TABS[2],))
                    _cq2.button("📊 Adjust Exposure",key="classic_jump_exposure",use_container_width=True,on_click=_classic_nav_go,args=(_CLASSIC_TABS[5],))
                    _cq3.button("⚡ Build Settings",key="classic_jump_build",use_container_width=True,on_click=_classic_nav_go,args=(_CLASSIC_TABS[1],))
                    render_results_command_center(res, df, sim_table, sim_worlds, packet)
            if classic_nav==_CLASSIC_TABS[5]:
                if res is None or res.empty:
                    st.info("Generate lineups first.")
                else:
                    st.markdown('<div class="card-title">Exposure Lab</div><div class="card-sub">Review the portfolio, then tighten or loosen individual players without going back to the Players tab.</div>',unsafe_allow_html=True)
                    exp=calculate_exposure_table(df,res,st.session_state["strategy_master"]).copy()
                    st.caption("My Exposure % is what the current portfolio used. Field Own % is projected contest ownership. Change a player's target, then rebuild.")

                    # iPad-friendly quick adjustment. The Streamlit grid's in-cell editor can be
                    # awkward on touch devices, so this gives exposure changes a reliable path.
                    st.markdown("#### Quick exposure adjustment")
                    _exp_names=exp["Name"].astype(str).tolist()
                    _pick=st.selectbox("Player to adjust",_exp_names,key="classic_quick_exp_player")
                    _row=exp[exp["Name"].astype(str).eq(str(_pick))].iloc[0]
                    _pid=str(_row["ID"])
                    _cur=st.session_state["strategy_master"].get(_pid,{})
                    _default_min=int(round(float(_cur.get("Min Exposure",0))))
                    _default_max=int(round(float(_cur.get("Max Exposure",100))))
                    qa,qb,qc=st.columns([1,1,1.2])
                    with qa:
                        _qmin=st.number_input("Min exposure %",min_value=0,max_value=100,value=_default_min,step=5,key="classic_quick_exp_min")
                    with qb:
                        _qmax=st.number_input("Max exposure %",min_value=0,max_value=100,value=_default_max,step=5,key="classic_quick_exp_max")
                    with qc:
                        st.metric("Current My Exposure",f"{float(_row['Actual Exp %']):.1f}%")
                    qd,qe=st.columns(2)
                    with qd:
                        _save_quick=st.button("SAVE EXPOSURE TARGET",use_container_width=True,key="classic_quick_exp_save")
                    with qe:
                        _rebuild_quick=st.button(f"SAVE + REBUILD {lineup_count}",type="primary",use_container_width=True,key="classic_quick_exp_rebuild")
                    if _save_quick or _rebuild_quick:
                        if float(_qmin)>float(_qmax):
                            st.error("Minimum exposure cannot be higher than maximum exposure.")
                        else:
                            _cur["Min Exposure"]=float(_qmin); _cur["Max Exposure"]=float(_qmax)
                            st.session_state["strategy_master"][_pid]=_cur
                            if _rebuild_quick:
                                with st.spinner("Rebuilding with your new exposure target…"):
                                    new_res=generate_lineups(
                                        df,field_size,payout_style,lineup_count,max(300,lineup_count*12),
                                        int(st.session_state["classic_min_salary"]),int(st.session_state["classic_qb_stack"]),st.session_state["classic_bringback"],
                                        preferred_stack_teams,st.session_state["strategy_master"],st.session_state["team_strategy_master"],
                                        bool(st.session_state["classic_no_dst"]),bool(st.session_state["classic_no_off"]),seed,
                                        max_players_team=int(st.session_state["classic_max_team"]),
                                        max_players_game=int(st.session_state["classic_max_game"]),
                                        max_te=int(st.session_state["classic_max_te"]),
                                        allow_qb_with_rb=bool(st.session_state["classic_allow_qb_rb"]),
                                        allowed_qb_ids=classic_qb_ids,
                                        flex_mix=({"RB":int(st.session_state["classic_flex_rb"]),"WR":int(st.session_state["classic_flex_wr"]),"TE":int(st.session_state["classic_flex_te"])}
                                                  if bool(st.session_state.get("classic_flex_control",False))
                                                  and int(st.session_state["classic_flex_rb"])+int(st.session_state["classic_flex_wr"])+int(st.session_state["classic_flex_te"])==100
                                                  else None),
                                        bringback_worthy=bb_worthy
                                    )
                                st.session_state["classic_result_v4"]=new_res
                                st.rerun()
                            else:
                                st.success(f"{_pick}: exposure target saved at {_qmin}%–{_qmax}%.")

                    st.markdown("#### Full exposure table")
                    with st.form("classic_exposure_editor_form",clear_on_submit=False):
                        exp_edit=st.data_editor(
                            exp,
                            hide_index=True,
                            use_container_width=True,
                            height=620,
                            disabled=["ID","Name","Pos","Team","Salary","Proj","Proj Own","Actual Exp %","Lineups"],
                            column_order=["Name","Pos","Team","Opponent","Salary","Proj","Proj Own","Actual Exp %","Lineups","Min Target %","Max Target %"],
                            column_config={
                                "Name":st.column_config.TextColumn("Player",width=190,pinned=True),
                                "Pos":st.column_config.TextColumn("Pos",width=60),
                                "Team":st.column_config.TextColumn("Team",width=70),
                                "Opponent":st.column_config.TextColumn("Opponent",width=82),
                                "Salary":st.column_config.NumberColumn("Salary",format="$%d",width=85),
                                "Proj":st.column_config.NumberColumn("Proj",format="%.2f",width=75),
                                "Proj Own":st.column_config.NumberColumn("Field Own %",format="%.1f",width=85),
                                "Actual Exp %":st.column_config.NumberColumn("My Exposure %",format="%.1f",width=95),
                                "Lineups":st.column_config.NumberColumn("Lineups",width=75),
                                "Min Target %":st.column_config.NumberColumn("Min %",min_value=0,max_value=100,step=5,width=80),
                                "Max Target %":st.column_config.NumberColumn("Max %",min_value=0,max_value=100,step=5,width=80),
                            },
                            key="classic_exposure_editor"
                        )
                        ea,eb=st.columns(2)
                        with ea:
                            apply_exp=st.form_submit_button("APPLY EXPOSURE CHANGES",use_container_width=True)
                        with eb:
                            rebuild_exp=st.form_submit_button(f"APPLY + REBUILD {lineup_count} LINEUPS",type="primary",use_container_width=True)

                    if apply_exp or rebuild_exp:
                        bad=[]
                        for _,r in exp_edit.iterrows():
                            pid=str(r["ID"]); mn=float(r["Min Target %"]); mx=float(r["Max Target %"])
                            if mn>mx:
                                bad.append(str(r["Name"])); continue
                            cur=st.session_state["strategy_master"].get(pid,{})
                            cur["Min Exposure"]=mn; cur["Max Exposure"]=mx
                            st.session_state["strategy_master"][pid]=cur
                        if bad:
                            st.error("Minimum exposure cannot be higher than maximum for: "+", ".join(bad[:8]))
                        elif rebuild_exp:
                            with st.spinner("Rebuilding with your new exposure limits…"):
                                new_res=generate_lineups(
                                    df,field_size,payout_style,lineup_count,max(300,lineup_count*12),
                                    int(st.session_state["classic_min_salary"]),int(st.session_state["classic_qb_stack"]),st.session_state["classic_bringback"],
                                    preferred_stack_teams,st.session_state["strategy_master"],st.session_state["team_strategy_master"],
                                    bool(st.session_state["classic_no_dst"]),bool(st.session_state["classic_no_off"]),seed,
                                    max_players_team=int(st.session_state["classic_max_team"]),
                                    max_players_game=int(st.session_state["classic_max_game"]),
                                    max_te=int(st.session_state["classic_max_te"]),
                                    allow_qb_with_rb=bool(st.session_state["classic_allow_qb_rb"]),
                                    allowed_qb_ids=classic_qb_ids,
                                    bringback_worthy=bb_worthy
                                )
                            st.session_state["classic_result_v4"]=new_res
                            st.success("Exposure changes applied and the portfolio was rebuilt.")
                            st.rerun()
                        else:
                            st.success("Exposure targets saved. They will be used on the next build.")
            if classic_nav==_CLASSIC_TABS[6]:
                from dfs_lab.ui.guide import render_guide
                render_guide(mode="classic")
        except Exception as e:
            st.error(f"Classic build error: {e}")
    else:
        try:
            df=prepare_showdown_pool(dk_file,ss_file,entry_format=entry_format); teams=[t for t in df["Team"].dropna().unique().tolist() if t]
            if len(teams)!=2: st.warning(f"Showdown is one game (2 teams), but this file has {len(teams)}: {', '.join(teams[:8])}. Lineups will be built from the wrong player pool — re-upload the single-game DK Showdown salaries for the game you're playing.")
            nonzero_proj=int((pd.to_numeric(df["My Proj"],errors="coerce").fillna(0)>0.05).sum())
            nonzero_own=int((pd.to_numeric(df["My Own"],errors="coerce").fillna(0)>0).sum())
            if nonzero_proj==0:
                st.error("Aytia could not create usable projections for this slate.")
                st.stop()
            elif nonzero_proj < 6: st.warning(f"Projection check · Only {nonzero_proj} players have usable projections. The slate may be incomplete.")
            else: st.success(f"Aytia projection engine ready · {nonzero_proj} players have usable projections." + (" · SaberSim comparison loaded." if ss_file else " · No SaberSim file used."))
            if nonzero_own==0: st.warning("Ownership not populated yet · Lineups can be built, but leverage and duplication ratings that depend on ownership are provisional.")
            st.session_state.setdefault("showdown_strategy",{})
            st.session_state.setdefault("showdown_context",{})
            st.session_state.setdefault("showdown_relationships",[])
            st.session_state.setdefault("projection_overrides",{})
            # 'Aytia remembers': re-apply persisted player edits after a session
            # death (iOS tab suspend). No-op when the dicts already have data
            # or the slate differs. Runs before the rescue path so restored
            # overrides participate in it.
            persist.restore_player_state(df, "Showdown")
            # Rescue path for the silent-override trust bug (Jalon Daniels, TNF
            # 10/08): a typed projection on an auto-excluded player puts them
            # back in the pool, unless officially OUT or user-marked Out. Runs
            # on the pool here so build_df inherits it; notes surface in the
            # Build tab above the Generate button.
            df,_rescued_sd,_blocked_sd=apply_override_rescue(df,st.session_state.get("projection_overrides",{}),st.session_state.get("showdown_strategy",{}))
            st.session_state["_sd_rescue_notes"]=(_rescued_sd,_blocked_sd)
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
            if bool(df["CPT Own Estimated"].any()): st.warning("Your SaberSim file does not appear to include Captain ownership. Aytia is using a neutral fallback for CPT leverage. Overall ownership is still used normally.")

            # Slate-specific build controls live in the main Aytia workspace — never in Streamlit's sidebar.
            with st.expander("🏈  SLATE STRATEGY  ·  SALARY & UNIQUENESS", expanded=False):
                sc1,sc2=st.columns(2)
                with sc1: salary_style=st.selectbox("Salary strategy",["Optimal","Balanced","GPP","Unique","Custom"],index=2)
                salary_defaults={"Optimal":49500,"Balanced":48500,"GPP":47500,"Unique":46000,"Custom":47000}
                with sc2: min_unique=st.selectbox("Minimum unique players",[1,2,3],index=0)
                if salary_style=="Custom": min_salary=st.slider("Minimum salary",40000,50000,47000,100)
                else:
                    min_salary=salary_defaults[salary_style]
                    st.caption(f"{salary_style} strategy · minimum salary ${min_salary:,}")
            max_salary=50000

            st.markdown(styles.SHOWDOWN_SHELL_CSS, unsafe_allow_html=True)
            # Story detour: the Story page sets this flag and calls render_main;
            # render only the story (no nav, no sections) and return.
            if st.session_state.pop("_story_detour", None) == "showdown":
                from dfs_lab.ui.story import render_showdown_story
                render_showdown_story(df, teams)
                return
            _SD_TABS=["⚡ Build","👤 Players","🔗 Relationships","🧠 Game Intel","⚙ Rules","📋 Lineups","📊 Exposure","📖 Guide"]
            # Multipage nav: the segmented control switches real Streamlit pages
            # (pages/*.py) via st.switch_page, so each section is its own screen
            # instead of a tab on one long scrolling page. Pinned via CSS on the
            # keyed container (.st-key-sd_nav).
            st.session_state.setdefault("sd_nav","⚡ Build")
            def _sd_nav_changed():
                from dfs_lab.ui import nav as _navmod
                _navmod.goto_page("Showdown", st.session_state.get("sd_nav"))
            sd_nav=st.segmented_control("Aytia workspace",_SD_TABS,key="sd_nav",label_visibility="collapsed",on_change=_sd_nav_changed)
            _sd_setup_c1,_sd_setup_c2=st.columns([5,1])
            with _sd_setup_c2:
                if st.button("⚙ Setup",key="sd_goto_setup",use_container_width=True,
                             help="Contest settings and slate files"):
                    st.switch_page("streamlit_app.py")
            # --- Nav-independent build inputs (Showdown) ---
            # Only the active section's widgets render on a run, but every widget
            # value persists in session state under its key. The build block below
            # (and the Lineups section) run on EVERY run, so derive their inputs
            # here from session state. Sections re-derive the same values from the
            # live widgets when they render; the values agree.
            _sd_qb_rule_map={"No rule":0,"Minimum 1":1,"Minimum 2":2,"No more than 1":-1,"No more than 2":-2}
            _sd_pair_map={"Never":0,"Sometimes":35,"Usually":80,"Always":100}
            cpt_qb_pc=_sd_qb_rule_map.get(str(st.session_state.get("sd_qb_cpt_pc_rule","No rule")),0)
            wrte_qb=_sd_pair_map.get(str(st.session_state.get("sd_wrte_pair","Usually")),80)
            rb_ctrl=_sd_pair_map.get(str(st.session_state.get("sd_rb_pair","Sometimes")),35)
            max_k=int(st.session_state.get("sd_max_k",2))
            max_dst=int(st.session_state.get("sd_max_dst",1))
            construction_weights={"3-3":1 if st.session_state.get("sd_allow_33",True) else 0,
                                  "4-2":1 if st.session_state.get("sd_allow_42",True) else 0,
                                  "5-1":1 if st.session_state.get("sd_allow_51",False) else 0}
            build_btn=False
            _sd_script=str(st.session_state.get("sd_script","Neutral"))
            use_score=bool(st.session_state.get("sd_use_score",False))
            team_scores={}
            if len(teams)>=2:
                team_scores={teams[0]:float(st.session_state.get("sd_score_0",0) or 0),
                             teams[1]:float(st.session_state.get("sd_score_1",0) or 0)}
            intensity=int(st.session_state.get("sd_intensity",50) or 50)
            _sd_script_team=str(st.session_state.get("sd_script_team","None"))
            if _sd_script_team=="None": _sd_script_team=""
            auto_shape=bool(st.session_state.get("sd_auto_shape",False))
            # World cap: nav-independent (the Rules tab widget writes sd_world_max_share
            # via persistent_widget; the build block below runs on every run).
            _wcap_pct=float(st.session_state.get("sd_world_max_share",0) or 0)
            world_max_share=(_wcap_pct/100.0 if _wcap_pct>0 else None)
            effective_script=_sd_script; effective_team=_sd_script_team
            if use_score and team_scores:
                _sd_asc,_sd_atm,_sd_sp=infer_score_script(team_scores)
                if _sd_script=="Auto from score":
                    effective_script=_sd_asc; effective_team=_sd_atm
            elif _sd_script=="Auto from score":
                effective_script="Neutral"; effective_team=""
            if sd_nav==_SD_TABS[0]:
                st.markdown('<div class="card-title">Showdown Build</div><div class="card-sub">Control how the six-man portfolio is shaped before the optimizer starts solving.</div>',unsafe_allow_html=True)
                # Optional story detour: not a required step, just a different way in.
                with st.expander("💡 Tell the story first (optional)", expanded=False):
                    st.caption("Have a take on this game? Tell it as a story and Aytia will translate it into build settings. Or ignore this and build the usual way — nothing here is required.")
                    if st.button("Tell the story →", key="sd_goto_story", use_container_width=True):
                        st.switch_page("pages/showdown_story.py")
                st.markdown("#### Allowed team builds")
                st.caption("Choose what is allowed — no percentages. 4-2 means four players from either team; 5-1 means five from either team.")
                c1,c2,c3=st.columns(3)
                with c1:allow_33=persistent_widget(st.checkbox,"sd_allow_33",True,"3-3")
                with c2:allow_42=persistent_widget(st.checkbox,"sd_allow_42",True,"4-2")
                with c3:allow_51=persistent_widget(st.checkbox,"sd_allow_51",False,"5-1")
                if not any([allow_33,allow_42,allow_51]):
                    st.warning("Choose at least one build. 3-3 will be used until you select one.")
                    allow_33=True
                construction_weights={"3-3":1 if allow_33 else 0,"4-2":1 if allow_42 else 0,"5-1":1 if allow_51 else 0}
                st.caption("The Scenario Engine decides how often to use each allowed build based on your score and game script.")
                st.markdown("#### Captain pairing")
                st.caption("These settings apply only when that position is Captain — they do not control how often the position becomes Captain.")
                a,b,c=st.columns(3)
                with a:
                    qb_cpt_pc_rule = persistent_widget(
                        st.selectbox, "sd_qb_cpt_pc_rule", "No rule",
                        "When QB is CPT · pass catchers",
                        ["No rule", "Minimum 1", "Minimum 2", "No more than 1", "No more than 2"],
                        help="Minimum/maximum is enforced first. If that rule makes the entire slate impossible, Aytia keeps your other settings and falls back to its football-coherence model rather than returning zero lineups."
                    )
                    cpt_qb_pc = {"No rule":0, "Minimum 1":1, "Minimum 2":2, "No more than 1":-1, "No more than 2":-2}[qb_cpt_pc_rule]
                pair_map={"Never":0,"Sometimes":35,"Usually":80,"Always":100}
                with b: wrte_pair=persistent_widget(st.selectbox,"sd_wrte_pair","Usually","When WR/TE is CPT · pair QB",list(pair_map),
                    help="Sometimes/Usually are SOFT portfolio preferences and will not block a legal build. Always is a hard rule.")
                with c: rb_pair=persistent_widget(st.selectbox,"sd_rb_pair","Sometimes","When RB is CPT · pair DST/K",list(pair_map),
                    help="Sometimes/Usually are SOFT portfolio preferences and will not block a legal build. Always is a hard rule.")
                wrte_qb=pair_map[wrte_pair]; rb_ctrl=pair_map[rb_pair]
                d,e=st.columns(2)
                with d:max_k=persistent_widget(st.selectbox,"sd_max_k",2,"Max kickers",[0,1,2])
                with e:max_dst=persistent_widget(st.selectbox,"sd_max_dst",1,"Max defenses",[0,1,2])
                _sd_fresh_note = availability_freshness_note(df)
                if _sd_fresh_note:
                    st.warning(_sd_fresh_note)
                _rescued_sd,_blocked_sd=st.session_state.get("_sd_rescue_notes",([],[]))
                for _nm,_why in _rescued_sd:
                    st.info(f"🔓 {_nm} is back in the pool — your projection overrode the auto-exclusion ({_why}).")
                for _nm,_why in _blocked_sd:
                    st.warning(f"⚠️ Your projection for {_nm} was NOT applied — {_why}.")
                build_btn=st.button(f"⚡ GENERATE {lineup_count} LINEUPS",type="primary",use_container_width=True,key="v4_sd_build")

            if sd_nav==_SD_TABS[1]:
                st.markdown('<div class="card-title">Players</div><div class="card-sub">Overall exposure and Captain exposure are controlled separately. Check CPT? only for the players you want in the Captain pool.</div>',unsafe_allow_html=True)
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
                # Same sort controls as the Classic Players section so both
                # tables act the same. Sorting changes the visible row order,
                # which scopes the editor key (pending edits reset with a
                # warning instead of landing on the wrong player).
                _ss1,_ss2=st.columns([2,1])
                with _ss1:
                    sd_sort_by=st.selectbox("Sort players by",["Salary","Projection","Ownership","Name"],index=0,key="sd_player_sort")
                with _ss2:
                    sd_sort_dir=st.segmented_control("Order",["High → Low","Low → High"],default="High → Low",key="sd_player_sort_dir")
                _sd_sort_col={"Salary":"FlexSalary","Projection":"Aytia Proj","Ownership":"My Own","Name":"Name"}[sd_sort_by]
                view=view.sort_values(_sd_sort_col,ascending=(sd_sort_dir=="Low → High"),kind="mergesort").reset_index(drop=True)

                # Live status strip: always show who's Out/Locked from SAVED settings,
                # plus a warning if the table has unsaved edits. This is the trust
                # anchor — the user should never wonder "did my Out tap register?"
                _strat = st.session_state.get("showdown_strategy", {}) or {}
                _id_to_nm = {str(r["ID"]): str(r["Name"]) for _, r in view.iterrows()}
                _out_nms = sorted(_id_to_nm.get(pid, pid) for pid, s in _strat.items()
                                  if isinstance(s, dict) and (s.get("Exclude") or s.get("Priority") == "Exclude"))
                _lock_nms = sorted(_id_to_nm.get(pid, pid) for pid, s in _strat.items()
                                   if isinstance(s, dict) and s.get("Lock") and not s.get("Exclude"))
                _cpt_nms = sorted(_id_to_nm.get(pid, pid) for pid, s in _strat.items()
                                   if isinstance(s, dict) and s.get("CPT Lock") and not s.get("Exclude"))
                _sed_key_now = st.session_state.get("v4_sdplayers_key")
                _pending = False
                if _sed_key_now:
                    _wstate = st.session_state.get(_sed_key_now, {})
                    if isinstance(_wstate, dict) and _wstate.get("edited_rows"):
                        _pending = True
                if _out_nms or _lock_nms or _cpt_nms:
                    _bits = []
                    if _out_nms: _bits.append(f"🚫 Out ({len(_out_nms)}): " + ", ".join(_out_nms[:6]) + ("…" if len(_out_nms) > 6 else ""))
                    if _lock_nms: _bits.append(f"🔒 Locked ({len(_lock_nms)}): " + ", ".join(_lock_nms[:6]) + ("…" if len(_lock_nms) > 6 else ""))
                    if _cpt_nms: _bits.append(f"👑 CPT ({len(_cpt_nms)}): " + ", ".join(_cpt_nms[:6]) + ("…" if len(_cpt_nms) > 6 else ""))
                    st.info(" · ".join(_bits))
                else:
                    st.caption("No players marked Out or Locked. Check Out to exclude a player from every lineup.")
                if _pending:
                    st.warning("⚠️ You have unsaved table changes — tap **Apply player changes** below so the build uses them.")
                st.markdown("##### Quick lock")
                _reset_col1,_reset_col2=st.columns([1,2])
                with _reset_col1:
                    if st.button("RESET ALL PLAYER RULES",use_container_width=True,key="sd_reset_all_player_rules"):
                        st.session_state["showdown_strategy"]={}
                        st.session_state["projection_overrides"]={}
                        st.session_state["agent_projection_scenario"]={}
                        st.session_state.pop("showdown_result_v4",None)
                        st.rerun()
                with _reset_col2:
                    st.caption("Clears locks, exclusions, Captain eligibility/exposure limits and projection overrides for this browser session.")
                quick_names=view["Name"].astype(str).tolist()
                if quick_names:
                    q1,q2,q3,q4,q5=st.columns([2.4,1,1,1,1])
                    with q1: quick_player=st.selectbox("Quick player",quick_names,key="sd_quick_player",label_visibility="collapsed")
                    qrow=view[view["Name"].astype(str)==str(quick_player)].iloc[0]; qid=str(qrow["ID"])
                    if qid not in st.session_state["showdown_strategy"]: st.session_state["showdown_strategy"][qid]={}
                    with q2:
                        if st.button("Lock",use_container_width=True,key="sd_quick_lock"):
                            _qe=st.session_state["showdown_strategy"][qid]
                            _qe.update({"Lock":True,"CPT Lock":False,"Exclude":False})
                            _qe["Priority"]=resolve_saved_priority(False,_qe.get("Priority"))
                            st.toast(f"🔒 {quick_player} locked — applied"); st.rerun()
                    with q3:
                        if st.button("CPT",use_container_width=True,key="sd_quick_cpt"):
                            _qe=st.session_state["showdown_strategy"][qid]
                            _qe.update({"Lock":False,"CPT Lock":True,"Exclude":False,"CPT Eligible":True})
                            _qe["Priority"]=resolve_saved_priority(False,_qe.get("Priority"))
                            st.toast(f"👑 {quick_player} captained — applied"); st.rerun()
                    with q4:
                        if st.button("Out",use_container_width=True,key="sd_quick_out"):
                            st.session_state["showdown_strategy"][qid].update({"Lock":False,"CPT Lock":False,"Exclude":True,"CPT Eligible":False,"Priority":"Exclude"}); st.toast(f"🚫 {quick_player} marked Out — applied"); st.rerun()
                    with q5:
                        if st.button("Clear",use_container_width=True,key="sd_quick_clear"):
                            st.session_state["showdown_strategy"].pop(qid,None); st.toast(f"{quick_player} rules cleared"); st.rerun()
                    st.caption("Lock = every lineup • CPT = Captain every lineup • Out = never use — these apply instantly, no submit needed.")

                ed=pd.DataFrame({"ID":view["ID"].astype(str),"Name":view["Name"],"Pos":view["Position"],"Team":view["Team"],"Flex $":view["FlexSalary"],"DFS Base":view["My Proj"].round(2),"Availability":view.get("Live Status",pd.Series("Not verified",index=view.index)),"Hist G":view.get("History Games",pd.Series(0,index=view.index)),"Matchup %":view.get("Matchup Adj %",pd.Series(0.0,index=view.index)),"Model":view["Model Proj"].round(2),"Your Proj":view["Aytia Proj"].round(2),"Δ%":view["Proj Change %"].round(1),"Own":view["My Own"].round(1),"CPT Own":view["CPT Own"].round(1),"Lock":False,"CPT Lock":False,"Exclude":False,"CPT Eligible":False,"Priority":"Neutral","Min Exposure":0,"Max Exposure":100,"CPT Min":0,"CPT Max":100})
                for x,r in ed.iterrows():
                    e=st.session_state["showdown_strategy"].get(str(r["ID"]),{})
                    for c,k,d in [("Lock","Lock",False),("CPT Lock","CPT Lock",False),("Exclude","Exclude",False),("CPT Eligible","CPT Eligible",False),("Priority","Priority","Neutral"),("Min Exposure","Min Exposure",0),("Max Exposure","Max Exposure",100),("CPT Min","CPT Min",0),("CPT Max","CPT Max",100)]: ed.at[x,c]=e.get(k,d)
                st.caption("Make all of your player/Captain changes, then tap Apply changes once. This prevents the screen from dimming after every checkbox. The Your Proj column is editable — type your own projection and the build uses your number instead of the model. CPT? starts unchecked: check it only for the players you want in the Captain pool.")
                # Same row-positional hazard as the Classic editor: pending
                # edits are positional, so scope the widget key to the exact
                # visible order. A filter change before tapping Apply resets
                # pending edits (with a warning) instead of excluding/locking
                # the wrong player.
                _sd_fp=f"{entry_format}|{int(field_size)}|{payout_style}|{getattr(dk_file,'name','slate')}"
                _sed_key=player_editor_widget_key("v4_sdplayers",_sd_fp,view["ID"].astype(str).tolist())
                _sed_prev=st.session_state.get("v4_sdplayers_key")
                if _sed_prev and _sed_prev!=_sed_key:
                    _sed_old=st.session_state.get(_sed_prev)
                    if isinstance(_sed_old,dict) and _sed_old.get("edited_rows"):
                        st.warning("Player list order changed before you tapped Apply changes — unapplied Out/Lock checks were discarded (not applied to the wrong players). Please re-check them and apply.")
                    st.session_state.pop(_sed_prev,None)
                st.session_state["v4_sdplayers_key"]=_sed_key
                with st.form("showdown_player_editor_form", clear_on_submit=False):
                    apply_player_changes_top=st.form_submit_button("Apply player changes",type="primary",use_container_width=True,key="sd_apply_top")
                    edited=st.data_editor(
                        ed,
                    hide_index=True,
                    use_container_width=True,
                    height=650,
                    disabled=["ID","Name","Pos","Team","Flex $","DFS Base","Availability","Hist G","Matchup %","Model","Δ%","Own","CPT Own"],
                    # iOS Safari ignores Streamlit's pinned columns, so Out sits
                    # directly beside the player name — inside the first screen
                    # of the table with no horizontal scroll needed.
                    column_order=["Name","Exclude","CPT Eligible","CPT Lock","Lock","Priority","Availability","Pos","Team","Flex $","DFS Base","Hist G","Matchup %","Model","Your Proj","Δ%","Own","CPT Own","Min Exposure","Max Exposure","CPT Min","CPT Max"],
                    column_config={
                        "ID":None,
                        "Name":st.column_config.TextColumn("Player",width=190,pinned=True),
                        "Pos":st.column_config.TextColumn("Pos",width=58),
                        "Team":st.column_config.TextColumn("Team",width=68),
                        "Availability":st.column_config.TextColumn("Live",width=115,help="Latest nflverse roster/injury status. OUT/non-active roster statuses are auto-excluded."),
                        "Flex $":st.column_config.NumberColumn("Flex $",width=78,format="$%d"),
                        "Proj":st.column_config.NumberColumn("Base",width=68,format="%.2f"),
                        "Script Proj":st.column_config.NumberColumn("Scenario",width=78,format="%.2f"),
                        "Aytia":st.column_config.NumberColumn("Aytia",width=78,format="%.2f"),
                        "Δ%":st.column_config.NumberColumn("Δ%",width=58,format="%.1f"),
                        "Own":st.column_config.NumberColumn("Own",width=64,format="%.1f"),
                        "CPT Own":st.column_config.NumberColumn("CPT Own",width=78,format="%.1f"),
                        "Lock":st.column_config.CheckboxColumn("Lock",width=62,pinned=True),
                        "CPT Lock":st.column_config.CheckboxColumn("CPT",width=62,pinned=True),
                        "Exclude":st.column_config.CheckboxColumn("Out",width=58,pinned=True),
                        "CPT Eligible":st.column_config.CheckboxColumn("CPT?",width=60,pinned=True),
                        "Priority":st.column_config.SelectboxColumn("Lean",options=PRIORITY_OPTIONS,width=92),
                        "Min Exposure":st.column_config.NumberColumn("Min %",min_value=0,max_value=100,step=5,width=70),
                        "Max Exposure":st.column_config.NumberColumn("Max %",min_value=0,max_value=100,step=5,width=70),
                        "CPT Min":st.column_config.NumberColumn("CPT Min",min_value=0,max_value=100,step=5,width=74),
                        "CPT Max":st.column_config.NumberColumn("CPT Max",min_value=0,max_value=100,step=5,width=74),
                    },
                        key=_sed_key,
                    )
                    apply_player_changes=st.form_submit_button("Apply player changes",type="primary",use_container_width=True)
                if apply_player_changes or apply_player_changes_top:
                    for _,r in edited.iterrows():
                        pid=str(r["ID"]); model_val=float(view.loc[view["ID"].astype(str).eq(pid),"Model Proj"].iloc[0]) if not view.loc[view["ID"].astype(str).eq(pid)].empty else float(r["Your Proj"])
                        user_val=float(r["Your Proj"])
                        if abs(user_val-model_val)>0.01: st.session_state["projection_overrides"][pid]=user_val
                        else: st.session_state["projection_overrides"].pop(pid,None)
                        ex=bool(r["Exclude"]); cptlock=bool(r["CPT Lock"]) and not ex
                        # A captain lock implies captain eligibility: checking the CPT column
                        # alone must put the player in the captain pool (it did before the
                        # CPT? opt-in change, and the lock is the stronger intent).
                        cpt_elig=(bool(r["CPT Eligible"]) or cptlock) and not ex
                        st.session_state["showdown_strategy"][str(r["ID"]) ]={"Lock":bool(r["Lock"]) and not ex and not cptlock,"CPT Lock":cptlock,"Exclude":ex,"CPT Eligible":cpt_elig,"Priority":resolve_saved_priority(ex, r["Priority"]),"Min Exposure":float(r["Min Exposure"]),"Max Exposure":float(r["Max Exposure"]),"CPT Min":float(r["CPT Min"]),"CPT Max":float(r["CPT Max"])}
                    # Drop the widget's pending-edit state (pop the key: mutating
                    # nested widget state is forbidden by Streamlit) so a later
                    # sort/filter change doesn't warn about "unapplied" edits that
                    # were already applied. Rerun so the table immediately shows
                    # the new Your Proj values.
                    st.session_state.pop(_sed_key,None)
                    # The Exposure Lab edits these same target columns: reset its
                    # editor too, so a stale snapshot can't write old values back
                    # over what was just applied here.
                    for _k in [k for k in list(st.session_state.keys()) if str(k).startswith("sd_exp_editor_")]:
                        st.session_state.pop(_k,None)
                    _n_ov=len(st.session_state["projection_overrides"])
                    st.toast("✅ Player changes applied")
                    st.success(f"Player settings applied.{f' {_n_ov} projection override(s) active — builds use your numbers.' if _n_ov else ''}")
                    st.rerun()
                st.button("📋 Back to Lineups",key="sd_players_back_to_lineups",use_container_width=True,on_click=_sd_nav_go,args=(_SD_TABS[5],))
            if sd_nav==_SD_TABS[2]:
                st.markdown('<div class="card-title">Relationships</div><div class="card-sub">Teach Aytia which players, positions and team roles belong together — or should never appear together.</div>',unsafe_allow_html=True)
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

            if sd_nav==_SD_TABS[3]:
                st.markdown('<div class="card-title">Game View</div><div class="card-sub">The football signals Aytia is using for this slate. Neutral inputs stay out of the way; open Model details only when you want to audit them.</div>',unsafe_allow_html=True)
                st.info("Ratings are confidence-shrunk and capped. Defense and current usage carry more weight than travel or primetime splits.")
                st.session_state.setdefault("showdown_context_auto",{})
                st.session_state.setdefault("showdown_context_manual",set())
                _ai1,_ai2=st.columns(2)
                with _ai1:
                    if st.button("✨ AUTO-FILL GAME INTEL",use_container_width=True,key="auto_intel_fill",
                                 help="Aytia researches defensive matchups, usage trends, rest edges and travel from nflverse and pre-fills the ratings. Anything you've typed yourself is never overwritten."):
                        from dfs_lab.data import _slate_season
                        from dfs_lab.auto_intel import auto_fill_game_intel
                        with st.spinner("Aytia is researching matchups, usage trends, rest and travel…"):
                            _sugg,_amsg=auto_fill_game_intel(df,_slate_season(df))
                        if _sugg:
                            _auto_intel_adopt_manual(st.session_state["showdown_context"],st.session_state["showdown_context_auto"],st.session_state["showdown_context_manual"])
                            for _pid,_s in _sugg.items():
                                _cfg=st.session_state["showdown_context"].setdefault(str(_pid),{})
                                for _col,_val in _s.items():
                                    if f"{_pid}|{_col}" not in st.session_state["showdown_context_manual"]:
                                        _cfg[_col]=_val
                                st.session_state["showdown_context_auto"][str(_pid)]=dict(_s)
                            st.success(f"{_amsg} Your edits always win — auto-fill never overwrites a cell you've touched.")
                        else:
                            st.warning(_amsg or "Auto-fill found no suggestions for this slate.")
                with _ai2:
                    if st.button("Reset Game Intel",use_container_width=True,key="auto_intel_reset"):
                        st.session_state["showdown_context"]={}
                        st.session_state["showdown_context_auto"]={}
                        st.session_state["showdown_context_manual"]=set()
                        st.session_state.pop("v51_context_editor",None)
                        st.rerun()
                st.caption("Aytia does the research — defensive matchups, usage trends, rest edges, travel — and pre-fills the ratings below. Type over anything you disagree with; your cells are yours.")
                st.caption("This table is your input, not the model's output — all zeros means you are applying no adjustment. Rate players −3 to +3 wherever you have a take (e.g. Ertz's bigger role with Goedert out: Usage +2); the Context influence slider scales how strongly it moves Aytia Proj in the preview below, capped at ±18%.")
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
                # Any cell the user changed away from the last auto-fill value
                # is theirs from now on: future auto-fills won't overwrite it.
                _auto_intel_adopt_manual(st.session_state["showdown_context"],st.session_state["showdown_context_auto"],st.session_state["showdown_context_manual"])

                # Preview context on top of the current scenario state.
                ctx_script=st.session_state.get("sd_script","Neutral"); ctx_team=st.session_state.get("sd_script_team","None")
                if ctx_team=="None": ctx_team=""
                ctx_use_score=bool(st.session_state.get("sd_use_score",False)); ctx_scores={}
                if len(teams)>=2: ctx_scores={teams[0]:float(st.session_state.get("sd_score_0",24)),teams[1]:float(st.session_state.get("sd_score_1",21))}
                if ctx_script=="Auto from score" and ctx_use_score: ctx_script,ctx_team,_=infer_score_script(ctx_scores)
                ctx_base=apply_showdown_scenario(df,ctx_script,ctx_team,ctx_use_score,ctx_scores,st.session_state.get("sd_intensity",50))
                ctx_df=apply_context_engine(ctx_base,st.session_state["showdown_context"],context_strength)
                st.markdown("#### Market vs. model foundation")
                st.caption("Base is now Aytia's independent nflverse/DK-prior projection. Scenario + context create the model projection; Your Proj can override the final optimizer input.")
                cp_cols=["Name","My Proj","Script Proj","Context Adj %","Aytia Proj"] + (["SaberSim Proj"] if "SaberSim Proj" in ctx_df.columns else [])
                cp=ctx_df[cp_cols].copy()
                cp.columns=["Player","Base","Scenario","Context %","Aytia"] + (["SaberSim"] if "SaberSim Proj" in ctx_df.columns else [])
                cp=cp.sort_values("Context %",key=lambda x:x.abs(),ascending=False)
                st.dataframe(cp,hide_index=True,use_container_width=True,height=390,column_config={"Player":st.column_config.TextColumn("Player",pinned=True,width=185),"Base":st.column_config.NumberColumn("Base",format="%.2f"),"Scenario":st.column_config.NumberColumn("Scenario",format="%.2f"),"Context %":st.column_config.NumberColumn("Context %",format="%.1f"),"Aytia":st.column_config.NumberColumn("Aytia",format="%.2f")})
                st.caption("Neutral means Aytia found no reason to move the projection. Detailed model inputs remain available above for auditing; Lineup Lab explains what matters for each lineup.")

            if sd_nav==_SD_TABS[4]:
                st.markdown('<div class="card-title">Scenario Engine</div><div class="card-sub">Use a football story, a predicted score, or both. Your game thesis changes projections, correlation and lineup construction. Use the 0–100 influence control to decide how strongly Aytia should commit to it.</div>',unsafe_allow_html=True)
                script_options=["Neutral","Auto from score","Shootout","Pass-heavy shootout","Low-scoring game","Defensive / field-goal battle","Ground-and-pound","Team wins close","Team dominates","Team plays from ahead","Team passing comeback"]
                script=persistent_widget(st.selectbox,"sd_script","Neutral","Game script",script_options)
                use_score=persistent_widget(st.toggle,"sd_use_score",False,"Use predicted score to adjust projections")
                team_scores={}
                score_profile={"total":0,"margin":0,"winner":"","loser":""}
                if len(teams)>=2:
                    sc1,sc2=st.columns(2)
                    with sc1: score0=persistent_widget(st.number_input,"sd_score_0",24,f"{teams[0]} score",0,70,disabled=not use_score)
                    with sc2: score1=persistent_widget(st.number_input,"sd_score_1",21,f"{teams[1]} score",0,70,disabled=not use_score)
                    team_scores={teams[0]:float(score0),teams[1]:float(score1)}
                intensity=persistent_widget(st.slider,"sd_intensity",50,"Scenario influence",min_value=0,max_value=100,step=5,help="0 = ignore the game thesis. 50 = standard influence. 100 = strongest bounded scenario influence.")
                st.caption(f"Game-thesis influence: {intensity}%")
                _wcap_auto=int(round(default_world_max_share(lineup_count)*100))
                wcap_pct=persistent_widget(st.number_input,"sd_world_max_share",0,"Max % of lineups per game world",min_value=0,max_value=100,step=5,help=f"0 = auto ({_wcap_auto}% for {lineup_count} lineups). Caps how much of the portfolio any single game world can take, so big portfolios cover more outcomes instead of repeating one theory.")
                st.caption(f"World cap: {'auto ('+str(_wcap_auto)+'%)' if not wcap_pct else str(int(wcap_pct))+'%'}")
                directional=script in ["Team wins close","Team dominates","Team plays from ahead","Team passing comeback"]
                script_team=persistent_widget(st.selectbox,"sd_script_team","None","Script team",["None"]+teams,disabled=(not directional) or script=="Auto from score")
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

                auto_shape=persistent_widget(st.toggle,"sd_auto_shape",False,"Let the scenario shape lineup construction + correlation")
                scenario_df=apply_showdown_scenario(df,effective_script,effective_team,use_score,team_scores,intensity)
                effective_weights,corr_overrides=script_build_adjustments(construction_weights,effective_script,effective_team,use_score,team_scores,auto_shape)
                if auto_shape:
                    mix_txt=" • ".join(f"{k} {int(v)}%" for k,v in effective_weights.items())
                    st.caption(f"Scenario build mix: {mix_txt}")
                    if corr_overrides:
                        st.caption(f"Scenario correlation suggestion: QB CPT + {corr_overrides['qb_pc']} pass catcher(s) • WR/TE CPT + QB {corr_overrides['wrte_qb']}% • RB CPT + DST/K {corr_overrides['rb_ctrl']}%")
                        st.caption("Build-tab Captain pairing controls stay authoritative and are not silently overridden.")

                st.markdown("#### Projection movement")
                preview=scenario_df[["Name","Position","Team","My Proj","Script Proj","Proj Change %"]].copy()
                preview.columns=["Player","Pos","Team","Base","Scenario","Change %"]
                preview=preview.sort_values("Change %",key=lambda x:x.abs(),ascending=False).head(14)
                st.dataframe(preview,hide_index=True,use_container_width=True,height=410,column_config={"Player":st.column_config.TextColumn("Player",pinned=True,width=190),"Base":st.column_config.NumberColumn("Base",format="%.2f"),"Scenario":st.column_config.NumberColumn("Scenario",format="%.2f"),"Change %":st.column_config.NumberColumn("Change %",format="%.1f")})
                st.caption("These are bounded scenario tilts applied to the Aytia baseline projections. They are not a claim that a final score can precisely predict individual fantasy points.")
                st.markdown("#### How V6.3 grades Showdown")
                st.caption("Projection 29–34% • Captain quality 18% • correlation 20% • leverage 11–15% • duplication proxy 10–15% • your takes 7%. The exact weights move with contest size/payout.")
                st.caption("The scenario engine changes projection and construction inputs before grading. When your game thesis is directional (low-scoring, shootout, ground-and-pound…), its Game World leads the portfolio at ~45% of lineups; the other worlds remain as genuine alternatives so one thesis doesn't become twenty identical teams.")

            # Build inputs above are derived unconditionally from session state, so this
            # block is safe on every run regardless of which nav section rendered.
            strategy_map=st.session_state["showdown_strategy"]
            # Apply the scenario to the actual build, not just the preview.
            build_df=apply_showdown_scenario(df,effective_script,effective_team,use_score,team_scores,intensity)
            build_df=apply_context_engine(build_df,st.session_state.get("showdown_context",{}),st.session_state.get("context_strength","Standard"))
            build_df=apply_projection_overrides(build_df,st.session_state.get("projection_overrides",{}))
            # Agent Scenario is a reversible layer above the base/model projection. It never overwrites the model.
            st.session_state.setdefault("agent_projection_scenario",{})
            for _pid,_val in st.session_state["agent_projection_scenario"].items():
                _mask=build_df["ID"].astype(str).eq(str(_pid)) if "ID" in build_df.columns else pd.Series(False,index=build_df.index)
                if _mask.any():
                    build_df.loc[_mask,"Aytia Proj"]=float(_val)
            # Final safety gate: a projection changed to zero after pool creation must
            # become unavailable immediately. Previously ActiveForBuild could stay True.
            _zero_final=pd.to_numeric(build_df["Aytia Proj"],errors="coerce").fillna(0.0)<=0.01
            build_df.loc[_zero_final,"ActiveForBuild"]=False
            build_df.loc[_zero_final,"Role Confidence"]="Unavailable — zero final projection"
            build_df.loc[_zero_final & build_df["Auto Excluded Reason"].astype(str).eq(""),"Auto Excluded Reason"]="Final Aytia projection is 0"
            build_weights,corr_overrides=script_build_adjustments(construction_weights,effective_script,effective_team,use_score,team_scores,auto_shape)
            # User-selected Captain pairing controls are authoritative hard rules.
            # The Scenario Engine can shape construction/projections, but it must not silently
            # replace a Build-tab rule (for example changing Minimum 1 to Minimum 2).
            eff_qb_pc=cpt_qb_pc; eff_wrte_qb=wrte_qb; eff_rb_ctrl=rb_ctrl
            scenario_corr_note=None
            if corr_overrides and auto_shape:
                scenario_corr_note=(
                    f"Scenario suggestion only · QB CPT + {corr_overrides['qb_pc']} pass catcher(s) · "
                    f"WR/TE CPT + QB {corr_overrides['wrte_qb']}% · RB CPT + DST/K {corr_overrides['rb_ctrl']}%. "
                    "Your Build-tab Captain pairing settings remain the enforced rules."
                )
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
            _story_build_sd=st.session_state.pop("story_build_requested", False)
            if build_btn or _story_build_sd:
                _cpt_pool=captain_pool_ids(build_df,strategy_map)
                if not _cpt_pool:
                    st.error("No captain-eligible players — every CPT? box is unchecked (or excluded). Check **CPT?** for at least one player in 👤 Players (checking the **CPT** lock column counts too), tap **Apply player changes**, then generate again.")
                    st.session_state["showdown_result_v4"]=None
                else:
                    result=generate_showdown_lineups(build_df,field_size,payout_style,lineup_count,max(120,lineup_count*5),min_salary,max_salary,build_weights,effective_script,effective_team,strategy_map,entry_format,eff_qb_pc,eff_wrte_qb,eff_rb_ctrl,max_k,max_dst,min_unique,seed,relationship_rules=active_relationships,world_influence=intensity,world_max_share=world_max_share)
                    # If percentage-based Captain pairing happens to over-constrain every sampled
                    # path, retry once with only the deterministic user rule for QB Captain and
                    # no probabilistic WR/TE/RB pairing. This is a targeted feasibility fallback,
                    # not a silent change to locks, exposures, salary, CPT eligibility or team builds.
                    _pairing_fallback_used=False
                    if result is None or result.empty:
                        _fallback=generate_showdown_lineups(
                            build_df,field_size,payout_style,lineup_count,max(80,lineup_count*3),
                            min_salary,max_salary,build_weights,effective_script,effective_team,
                            strategy_map,entry_format,eff_qb_pc,0,0,max_k,max_dst,min_unique,seed+17,
                            relationship_rules=active_relationships,world_influence=intensity,
                            world_max_share=world_max_share
                        )
                        if _fallback is not None and not _fallback.empty:
                            result=_fallback
                            _pairing_fallback_used=True

                    # Smart feasibility fallback: a Captain-correlation preference should not
                    # brick the entire slate. If the requested QB-Captain pass-catcher rule is
                    # the blocker, keep every user lock/exclusion/exposure/team-build rule intact
                    # and allow QB Captain lineups to be judged by the coherence engine instead.
                    # This does NOT make nonsense lineups acceptable: QB-CPT without a receiver
                    # is explicitly penalized in coherence and will rank behind cleaner stories.
                    _qb_rule_fallback_used=False
                    if (result is None or result.empty) and eff_qb_pc != 0:
                        _qb_fallback=generate_showdown_lineups(
                            build_df,field_size,payout_style,lineup_count,max(100,lineup_count*4),
                            min_salary,max_salary,build_weights,effective_script,effective_team,
                            strategy_map,entry_format,0,eff_wrte_qb,eff_rb_ctrl,max_k,max_dst,min_unique,seed+29,
                            relationship_rules=active_relationships,world_influence=intensity,
                            world_max_share=world_max_share
                        )
                        if _qb_fallback is not None and not _qb_fallback.empty:
                            result=_qb_fallback
                            _qb_rule_fallback_used=True

                    st.session_state["showdown_result_v4"]=result
                    # Post-build hard-rule audit: the MILP enforces CPT eligibility,
                    # exclusions and locks as hard bounds, so any violation here means
                    # the saved strategy and the built lineups disagree (e.g. a UI edit
                    # landed on the wrong player) and the user must see it loudly.
                    if result is not None and not result.empty:
                        _id_to_name={str(r["ID"]):str(r["Name"]) for _,r in build_df.iterrows()}
                        _violations=audit_showdown_portfolio(result,strategy_map,_id_to_name)
                        if _violations:
                            _shown="\n".join(f"- {v}" for v in _violations[:10])
                            _more=f"\n…plus {len(_violations)-10} more." if len(_violations)>10 else ""
                            st.error(f"⚠️ Aytia caught a hard-rule violation in the built portfolio. Your player settings say one thing, but the lineups say another — re-check Players and rebuild:\n{_shown}{_more}")
                    if _pairing_fallback_used:
                        st.warning("Aytia built the portfolio after dropping the probabilistic WR/TE→QB and RB→DST/K pairing suggestions. Your explicit QB-Captain pass-catcher rule, salary, locks, CPT pool, exposures and team-build rules were kept.")
                    if _qb_rule_fallback_used:
                        st.info("Aytia detected that the QB-Captain pass-catcher rule blocked the slate, so it used football-coherence scoring for QB Captain builds instead of returning no lineups. All player locks, exclusions, exposures, salary and team-build settings were kept.")
                    if result is not None and not result.empty and len(result) < int(lineup_count):
                        st.warning(f"Built {len(result)} of {int(lineup_count)} requested lineups. Aytia stopped after the optimizer stalled on the current hard rules instead of hanging indefinitely. Loosen the rule called out below or build the smaller portfolio.")
                    if result is not None and not result.empty:
                        _exp_misses=audit_min_exposure(result,build_df,strategy_map)
                        if _exp_misses:
                            _lines="\n".join(f"• {_m['name']}: target min {_m['target']:.0f}%, got {_m['actual']:.1f}% — {_m['reason']}." for _m in _exp_misses[:8])
                            _more=f"\n…plus {len(_exp_misses)-8} more." if len(_exp_misses)>8 else ""
                            st.warning(f"⚠️ Minimum-exposure targets missed:{chr(10)}{_lines}{_more}")
                    if result is None or result.empty:
                        st.error("Aytia could not produce a lineup with the current rules. Running a quick feasibility check…")

                        def _diag_build(_strategy=strategy_map, _min_salary=min_salary, _weights=build_weights,
                                        _qb=eff_qb_pc, _wrte=eff_wrte_qb, _rb=eff_rb_ctrl,
                                        _rels=active_relationships, _unique=min_unique,
                                        _max_k=max_k, _max_dst=max_dst):
                            try:
                                x=generate_showdown_lineups(
                                    build_df,field_size,payout_style,1,4,_min_salary,max_salary,_weights,
                                    effective_script,effective_team,_strategy,entry_format,
                                    _qb,_wrte,_rb,_max_k,_max_dst,_unique,seed+991,
                                    relationship_rules=_rels,world_influence=intensity,show_progress=False)
                                return x is not None and not x.empty
                            except Exception:
                                return False

                        tests=[]
                        tests.append(("Relationship rules", _diag_build(_rels=[])))
                        tests.append(("QB-Captain pass-catcher rule", _diag_build(_qb=0)))
                        tests.append(("WR/TE-Captain QB rule", _diag_build(_wrte=0)))
                        tests.append(("RB-Captain DST/K rule", _diag_build(_rb=0)))
                        tests.append(("All Captain-pairing rules", _diag_build(_qb=0,_wrte=0,_rb=0)))
                        tests.append(("Allowed team builds", _diag_build(_weights={"3-3":1.0,"4-2":1.0,"5-1":1.0})))
                        tests.append((f"Minimum salary (${int(min_salary):,})", _diag_build(_min_salary=0)))
                        tests.append(("Uniqueness requirement", _diag_build(_unique=0)))
                        tests.append(("Kicker/defense caps", _diag_build(_max_k=5,_max_dst=5)))

                        all_cpt_strategy={str(k):dict(v) for k,v in strategy_map.items()}
                        for _,rr in build_df.iterrows():
                            pid=str(rr["ID"])
                            e=all_cpt_strategy.setdefault(pid,{})
                            if not bool(e.get("Exclude",False)):
                                e["CPT Eligible"]=True
                        tests.append(("Captain pool", _diag_build(_strategy=all_cpt_strategy)))

                        exposure_strategy={str(k):dict(v) for k,v in strategy_map.items()}
                        for _,e in exposure_strategy.items():
                            e["Min Exposure"]=0; e["Max Exposure"]=100
                            e["CPT Min"]=0; e["CPT Max"]=100
                        tests.append(("Exposure minimums/maximums", _diag_build(_strategy=exposure_strategy)))

                        lock_strategy={str(k):dict(v) for k,v in strategy_map.items()}
                        for _,e in lock_strategy.items():
                            e["Lock"]=False; e["CPT Lock"]=False
                        tests.append(("Player/Captain locks", _diag_build(_strategy=lock_strategy)))

                        blockers=[name for name,works in tests if works]
                        if blockers:
                            st.warning("**Feasibility diagnostic:** A test build succeeded when Aytia relaxed: **" + ", ".join(blockers) + "**. This is a clue, not proof that the setting itself is invalid. Your settings were not changed.")
                        else:
                            clean_strategy={}
                            for _,rr in build_df.iterrows():
                                clean_strategy[str(rr["ID"]) ]={"CPT Eligible":True,"Min Exposure":0,"Max Exposure":100,"CPT Min":0,"CPT Max":100,"Lock":False,"CPT Lock":False,"Exclude":False,"Priority":"Neutral"}
                            fully_relaxed=_diag_build(_strategy=clean_strategy,_min_salary=0,_weights={"3-3":1.0,"4-2":1.0,"5-1":1.0},_qb=0,_wrte=0,_rb=0,_rels=[],_unique=0,_max_k=5,_max_dst=5)
                            if fully_relaxed:
                                st.warning("**Feasibility diagnostic:** No single setting is responsible; two or more constraints conflict when combined. Your settings were not changed.")
                            else:
                                st.error("**Feasibility diagnostic:** Even a fully relaxed six-player build is infeasible. This points to a slate-data or optimizer bug, not something you selected.")

                        with st.expander("Feasibility test details"):
                            for name,works in tests:
                                st.write(("✅ Legal when relaxed: " if works else "— Still infeasible when relaxed: ") + name)
                    elif len(result) < lineup_count:
                        st.warning(f"Built {len(result)} of {lineup_count} requested lineups. The current portfolio rules are limiting the number of unique legal builds.")
                # A story-triggered build jumps straight to the Lineups page.
                if _story_build_sd:
                    st.switch_page("pages/showdown_lineups.py")
            result=st.session_state.get("showdown_result_v4")

            if sd_nav==_SD_TABS[5]:
                st.markdown('<div class="card-title">Rated Lineups</div><div class="card-sub">The grade is portfolio-relative. A+ means one of the strongest lineups in this build — not a guarantee of outcome.</div>',unsafe_allow_html=True)
                if result is None or result.empty: st.info("Set your build, player takes and script, then generate lineups.")
                else:
                    # LINEUPS FIRST: the top-3 cards lead the page. Metrics, jumps
                    # and analysis tabs follow below — no preamble scroll.
                    _headshot_map={}
                    if "Headshot URL" in build_df.columns:
                        _headshot_map={str(r["Name"]):str(r.get("Headshot URL","")) for _,r in build_df.iterrows()
                                       if str(r.get("Headshot URL","")).lower() not in ["","nan","none"]}
                    _top_cols=st.columns(3)
                    for _card_col,(_, _top_lr) in zip(_top_cols,list(result.head(3).iterrows())):
                        with _card_col:
                            st.markdown(_sd_lineup_card_html(_top_lr,_headshot_map),unsafe_allow_html=True)
                    m1,m2,m3,m4=st.columns(4); m1.metric("A / A+",int(result["Rating"].isin(["A","A+"]).sum())); m2.metric("Top projection",f"{result['Projection'].max():.1f}"); m3.metric("Avg salary left",f"${int(result['Salary Left'].mean()):,}"); m4.metric("Built",len(result))
                    _world_total=int(result["Game World"].nunique()) if "Game World" in result.columns else 0
                    # Post-build quick jumps: the user asked for a way back to Players /
                    # Exposure once lineups exist. These set the nav state via
                    # on_click callbacks (see _sd_nav_go) and the rerun lands on
                    # the chosen section.
                    _jq1,_jq2,_jq3=st.columns(3)
                    _jq1.button("👤 Adjust Players",key="sd_jump_players",use_container_width=True,on_click=_sd_nav_go,args=(_SD_TABS[1],))
                    _jq2.button("📊 Adjust Exposure",key="sd_jump_exposure",use_container_width=True,on_click=_sd_nav_go,args=(_SD_TABS[6],))
                    _jq3.button("⚡ Build Settings",key="sd_jump_build",use_container_width=True,on_click=_sd_nav_go,args=(_SD_TABS[0],))
                    result_views=st.tabs(
                        [f"🏈  LINEUPS · {len(result)}",f"🌎  GAME WORLDS · {_world_total}","🧠  LINEUP LAB + AGENT"],
                        key="sd_results_hub"
                    )

                    with result_views[0]:
                        st.markdown("<div class='results-view-title'>All lineups</div><div class='results-view-sub'>The top 3 are at the top of this page. Open the full table only when you need it.</div>",unsafe_allow_html=True)
                        import html as _html
                        with st.expander(f"See all {len(result)} lineups",expanded=False):
                            st.caption("Tap a lineup row to see why Aytia built it and what drives the grade.")
                            _quick_cols=["Rank","Rating","CPT","FLEX1","FLEX2","FLEX3","FLEX4","FLEX5","Projection","Salary","Salary Left","Construction"]
                            _quick_df=result[[x for x in _quick_cols if x in result.columns]].reset_index(drop=True)
                            try:
                                _lineup_event=st.dataframe(
                                    _quick_df,hide_index=True,use_container_width=True,
                                    height=min(560,42+35*len(result)),
                                    on_select="rerun",selection_mode="single-row",key="showdown_lineup_board"
                                )
                                _selected_rows=list(getattr(getattr(_lineup_event,"selection",None),"rows",[]) or [])
                            except TypeError:
                                st.dataframe(_quick_df,hide_index=True,use_container_width=True,height=min(560,42+35*len(result)))
                                _selected_rows=[]
                            _why_idx=int(_selected_rows[0]) if _selected_rows else 0
                            _why=result.reset_index(drop=True).iloc[_why_idx]

                            _dup=str(_why.get("Dup Risk","—"))
                            _corr=str(_why.get("Correlation Grade","—"))
                            _lev=str(_why.get("Leverage Grade","—"))
                            _coh=float(pd.to_numeric(pd.Series([_why.get("Coherence Score",0)]),errors="coerce").fillna(0).iloc[0])
                            _proj_grade=str(_why.get("Projection Grade","—"))
                            _cpt_grade=str(_why.get("Captain Grade","—"))
                            _world_why=str(_why.get("Game World","—"))
                            _story=str(_why.get("Lineup Story","") or _why.get("Story","") or "Aytia built this lineup as a distinct path to the slate ceiling.")
                            _flags=str(_why.get("Coherence Flags","")).strip()
                            _strategy_note=str(_why.get("Strategy Notes","")).strip()
                            _watch=_flags if _flags and _flags.lower() not in ["nan","none"] else (_strategy_note if _strategy_note and _strategy_note.lower() not in ["nan","none"] else "No major football-coherence warning was flagged for this build.")

                            st.markdown(
                                "<div class='why-drawer'>"
                                "<div class='why-head'><div><div class='why-kicker'>WHY THIS LINEUP?</div>"
                                "<div class='why-title'>Lineup #%s <span>· %s</span></div></div>"
                                "<div class='why-proj'>%.1f <small>pts</small></div></div>"
                                "<div class='why-chips'>"
                                "<span><b>%s</b> DUP</span><span><b>%s</b> CORRELATION</span>"
                                "<span><b>%s</b> LEVERAGE</span><span><b>%.0f</b> COHERENCE</span>"
                                "</div>"
                                "<div class='why-section'><b>Why Aytia likes it</b><p>%s</p></div>"
                                "<div class='why-grid'>"
                                "<div><small>GAME WORLD</small><strong>%s</strong></div>"
                                "<div><small>CAPTAIN</small><strong>%s · %s</strong></div>"
                                "<div><small>PROJECTION</small><strong>%s</strong></div>"
                                "<div><small>SALARY LEFT</small><strong>$%s</strong></div>"
                                "</div>"
                                "<div class='why-watch'><b>Watch-out</b><span>%s</span></div>"
                                "</div>" % (
                                    int(_why["Rank"]),_html.escape(str(_why.get("Rating","—"))),
                                    float(_why.get("Projection",0)),
                                    _html.escape(_dup.upper()),_html.escape(_corr.upper()),_html.escape(_lev.upper()),_coh,
                                    _html.escape(_story),
                                    _html.escape(_world_why),
                                    _html.escape(str(_why.get("CPT",_why.get("Captain","—")))),_html.escape(_cpt_grade),
                                    _html.escape(_proj_grade),
                                    f"{int(_why.get('Salary Left',0)):,}",
                                    _html.escape(_watch)
                                ),
                                unsafe_allow_html=True
                            )
                            _qd1,_qd2=st.columns(2)
                            with _qd1: st.download_button("Download analysis CSV",result.to_csv(index=False),"showdown_lineups_v6_4.csv","text/csv",use_container_width=True,key="quick_analysis_download")
                            with _qd2: st.download_button("Download DK-format CSV",showdown_upload_csv(result),"showdown_dk_upload_v6_4.csv","text/csv",use_container_width=True,key="quick_dk_download")
                            if my_entries < len(result):
                                _top_n = result.head(my_entries)
                                st.download_button(f"Download my {my_entries} entries (top {my_entries} by grade)",showdown_upload_csv(_top_n),f"showdown_my_{my_entries}_entries.csv","text/csv",use_container_width=True,key="quick_my_entries_download")
                                st.caption(f"Top {my_entries} by grade — review the full {len(result)} above and reorder if you prefer different ones.")
                            st.caption("Tip: on iPhone, downloading opens the Files preview — use the app switcher to come back here. Your lineups stay on this tab.")
                            st.button("📋 Back to My Lineups",key="sd_dl_back_to_lineups",use_container_width=True,on_click=_sd_nav_go,args=(_SD_TABS[5],))

                    with result_views[1]:
                        st.markdown("<div class='results-view-title'>Game Worlds</div><div class='results-view-sub'>See the different ways Aytia thinks this slate can unfold, then isolate the lineups built for any one world.</div>",unsafe_allow_html=True)
                        st.markdown("#### Game Worlds")
                        world_counts=result["Game World"].value_counts().rename_axis("World").reset_index(name="Lineups") if "Game World" in result.columns else pd.DataFrame()
                        world_filter="All worlds"
                        if not world_counts.empty:
                            world_counts["Share %"]=(100*world_counts["Lineups"]/len(result)).round(1)
                            world_options=["All worlds"]+[f"{r['World']} · {int(r['Lineups'])} lineups" for _,r in world_counts.iterrows()]
                            world_pick=st.selectbox("Highlight a world",world_options,key="world_explorer_pick")
                            if world_pick!="All worlds": world_filter=world_pick.rsplit(" · ",1)[0]
                            wc=world_counts if world_filter=="All worlds" else world_counts[world_counts["World"].eq(world_filter)]
                            _world_html="<div class='world-board'>"
                            for _,_wr in wc.iterrows():
                                _pct=float(_wr["Share %"])
                                _world_html+=("<div class='world-row'><div class='world-label'>%s</div><div class='world-track'><div class='world-fill' style='width:%s%%'></div></div><div class='world-count'>%s <span>%s%%</span></div></div>" % (_wr["World"],max(3,_pct),int(_wr["Lineups"]),f"{_pct:.0f}"))
                            _world_html+="</div>"
                            st.markdown(_world_html,unsafe_allow_html=True)
                        filtered_result=result if world_filter=="All worlds" else result[result["Game World"].astype(str).eq(world_filter)]
                        if world_filter!="All worlds" and not filtered_result.empty:
                            w1,w2,w3=st.columns(3); w1.metric("Lineups in world",len(filtered_result)); w2.metric("Top projection",f"{filtered_result['Projection'].max():.1f}"); w3.metric("Avg salary left",f"${int(filtered_result['Salary Left'].mean()):,}")
                            st.caption(str(filtered_result.iloc[0].get("World Thesis","")))
                            st.markdown(f"#### Lineups built for {world_filter}")
                            st.caption("Every lineup in this world's separated set — full rosters, not summaries. Same set the Lineup Lab explorer below is analyzing.")
                            _wl_headshots={}
                            if "Headshot URL" in build_df.columns:
                                _wl_headshots={str(r["Name"]):str(r.get("Headshot URL","")) for _,r in build_df.iterrows()
                                               if str(r.get("Headshot URL","")).lower() not in ["","nan","none"]}
                            _wl_rows=list(filtered_result.iterrows())
                            for _ci in range(0,len(_wl_rows),2):
                                _ccols=st.columns(2)
                                for _cc,(_, _wlr) in zip(_ccols,_wl_rows[_ci:_ci+2]):
                                    with _cc:
                                        st.markdown(_sd_lineup_card_html(_wlr,_wl_headshots),unsafe_allow_html=True)
                    with result_views[2]:
                        st.markdown("<div class='results-view-title'>Lineup Lab</div><div class='results-view-sub'>Pick any lineup and see the football story behind all six players — then test swaps yourself.</div>",unsafe_allow_html=True)
                        st.markdown("#### Lineup Explorer")
                        st.caption("Highlight a lineup and Aytia will analyze all six players together — not just the Captain.")
                        lineup_choices=[f"#{int(r['Rank'])} · {r['Captain']} CPT · {r['Construction']} · {r['Projection']:.1f} pts" for _,r in filtered_result.iterrows()]
                        pick=st.selectbox("Choose lineup ▾",lineup_choices,key="lineup_lab_pick",help="Tap to switch the lineup Aytia is analyzing.")
                        li=lineup_choices.index(pick); lr=filtered_result.iloc[li]
                        risk = "lower" if entry_format=="Single Entry" else ("moderate" if entry_format in ["3-Max","20-Max"] else "higher")
                        contest_reason=(f"{entry_format} with {int(field_size):,} entries. Aytia uses {risk} tolerance for fragile salary-relief plays and weights tournament ceiling/correlation accordingly.")
                        st.markdown(f"**Why this lineup exists**  \n**Contest:** {entry_format} · {int(field_size):,} entries · {payout_style}  \n**Game thesis:** {effective_script} · influence {int(intensity)}%  \n**Game world:** {lr.get('Game World',effective_script)}  \n**Construction:** {lr['Construction']} · **Captain:** {lr['Captain']}  \n**Projection:** {lr['Projection']:.2f} · **Salary left:** ${int(lr['Salary Left']):,}")
                        st.write(contest_reason)
                        st.write(f"**World thesis:** {lr.get('World Thesis','')}")
                        st.write(f"**Lineup story:** {lr.get('Lineup Story','')}")
                        st.caption(f"Football coherence {float(lr.get('Coherence Score',0)):.0f}/100 · {lr.get('Coherence Flags','')}")
                        roster_names=[str(lr['Captain'])]+[str(lr.get('FLEX'+str(i),'')) for i in range(1,6)]
                        roster_names=[x for x in roster_names if x]
                        detail_rows=[]
                        for j,nm in enumerate(roster_names):
                            pr=build_df[build_df['Name'].astype(str).eq(nm)]
                            if pr.empty: continue
                            pr=pr.iloc[0]; role="Captain / ceiling engine" if j==0 else ("Primary projection piece" if float(pr.get('Aytia Proj',0))>=12 else "Salary relief / secondary path")
                            detail_rows.append({"Slot":"CPT" if j==0 else f"FLEX {j}","Player":nm,"Pos":pr.get('Position',''),"Team":pr.get('Team',''),"Aytia":round(float(pr.get('Aytia Proj',0)),2),"Salary":int(pr.get('FlexSalary',0)) if j else int(pr.get('CaptainSalary',pr.get('CPTSalary',0))),"Purpose":role})
                        if detail_rows:
                            st.dataframe(pd.DataFrame(detail_rows),hide_index=True,use_container_width=True,column_config={"Player":st.column_config.TextColumn("Player",pinned=True),"Salary":st.column_config.NumberColumn("Salary",format="$%d")})
                            low=min(detail_rows,key=lambda x:x["Aytia"])
                            st.write(f"**Weakest projection link:** {low['Player']} ({low['Aytia']:.2f}). Aytia is using this spot as {low['Purpose'].lower()} within the six-player construction.")

                        st.write(f"Aytia selected **{lr['Captain']} at Captain** while preserving the {lr['Construction']} game construction because this combination ranked strongly under the current projection, correlation, salary and contest-risk settings. {lr.get('Strategy Notes','')}")
                        if float(lr.get('Scenario Delta',0))!=0: st.write(f"Your game thesis moved this lineup by **{float(lr['Scenario Delta']):+.2f} projected DK points** versus the unadjusted baseline.")
                        st.markdown("##### Manual swap check (optional)")
                        st.caption("Inspect a specific one-for-one swap. Only salary-cap-legal replacements are shown.")
                        out_player=st.selectbox("Replace",roster_names,key="lab_out")
                        if out_player:
                            po=build_df[build_df['Name'].astype(str).eq(out_player)].iloc[0]
                            out_is_cpt=(str(out_player)==str(lr['Captain']))
                            out_sal_col='CaptainSalary' if out_is_cpt and 'CaptainSalary' in build_df.columns else ('CPTSalary' if out_is_cpt and 'CPTSalary' in build_df.columns else 'FlexSalary')
                            old_sal=int(po.get(out_sal_col,po.get('FlexSalary',0)))
                            max_in_salary=50000-(int(lr['Salary'])-old_sal)
                            legal_pool=build_df[~build_df['Name'].astype(str).isin(roster_names)].copy()
                            legal_pool=legal_pool[pd.to_numeric(legal_pool[out_sal_col],errors='coerce').fillna(999999)<=max_in_salary]
                            legal_pool=legal_pool.sort_values('Aytia Proj',ascending=False)
                            pool_names=legal_pool['Name'].astype(str).tolist()
                            if pool_names:
                                in_player=st.selectbox("With",pool_names,key="lab_in")
                                pi=legal_pool[legal_pool['Name'].astype(str).eq(in_player)].iloc[0]
                                dproj=float(pi['Aytia Proj'])-float(po['Aytia Proj']); dsal=int(pi[out_sal_col])-old_sal
                                st.write(f"**Legal direct swap:** projection {dproj:+.2f} · salary {dsal:+,} · new salary ${int(lr['Salary'])+dsal:,}.")
                            else:
                                st.info("No one-for-one replacement fits the salary cap.")
                        with st.expander("Detailed lineup grades & diagnostics",expanded=False):
                            cols=["Rank","Rating","CPT","FLEX1","FLEX2","FLEX3","FLEX4","FLEX5","Projection","Salary","Salary Left","Construction","Coherence Score","Lineup Story","Coherence Flags","CPT Own","Dup Risk","Projection Grade","Captain Grade","Correlation Grade","Leverage Grade","Duplication Grade","Story"]
                            st.dataframe(result[[x for x in cols if x in result.columns]],hide_index=True,use_container_width=True,height=610)
                        pick=st.number_input("Inspect lineup rank",min_value=1,max_value=len(result),value=1,step=1)
                        r=result.iloc[int(pick)-1]
                        st.write(f"**{r['Rating']} ({r['Rating Score']})** — {r['Story']}")
                        st.caption(f"Projection {r['Projection']} • Salary ${int(r['Salary']):,} • ${int(r['Salary Left']):,} left • Duplication risk {r['Dup Risk']} • {r['Strategy Notes']}")

            if sd_nav==_SD_TABS[6]:
                st.markdown('<div class="card-title">Exposure Lab</div><div class="card-sub">Edit exposure targets right here — no trip to Players needed.</div>',unsafe_allow_html=True)
                if result is None or result.empty: st.info("Generate lineups first.")
                else:
                    exp=showdown_exposure_table(df,result,strategy_map)
                    # Editor key is scoped to this exact portfolio fingerprint: if the
                    # user rebuilds between editing and applying, the key changes and
                    # stale row-position edits can never land on the wrong player.
                    _fp=f"{len(result)}_{int(result['Rank'].sum())}_{float(result['Projection'].sum()):.1f}_{str(result['CPT'].iloc[0])}_{str(result['CPT'].iloc[-1])}"
                    _exp_key=f"sd_exp_editor_{_fp}"
                    st.markdown("#### Exposure targets")
                    st.caption("Edit the Target columns, then APPLY. Actuals are read-only — they describe the last build. New targets take effect on your next Generate in ⚡ Build.")
                    _ed=exp.rename(columns={"Name":"Player","Actual %":"My Exp %","Min %":"Target Min %","Max %":"Target Max %","CPT Actual %":"CPT Exp %","CPT Min %":"CPT Target Min %","CPT Max %":"CPT Target Max %"})
                    _ed_cols=["Player","Pos","Team","My Exp %","Target Min %","Target Max %","CPT Exp %","CPT Target Min %","CPT Target Max %"]
                    _ed=_ed[[c for c in _ed_cols if c in _ed.columns]]
                    st.data_editor(
                        _ed,hide_index=True,use_container_width=True,height=560,key=_exp_key,
                        disabled=[c for c in _ed.columns if "Target" not in c],
                        column_config={
                            "Player":st.column_config.TextColumn("Player",pinned=True,width=170),
                            "My Exp %":st.column_config.NumberColumn("My Exp %",format="%.1f"),
                            "Target Min %":st.column_config.NumberColumn("Min %",min_value=0,max_value=100,step=5),
                            "Target Max %":st.column_config.NumberColumn("Max %",min_value=0,max_value=100,step=5),
                            "CPT Exp %":st.column_config.NumberColumn("CPT %",format="%.1f"),
                            "CPT Target Min %":st.column_config.NumberColumn("CPT Min %",min_value=0,max_value=100,step=5),
                            "CPT Target Max %":st.column_config.NumberColumn("CPT Max %",min_value=0,max_value=100,step=5)})
                    if st.button("✅ APPLY EXPOSURE TARGETS",use_container_width=True,key=f"sd_exp_apply_{_fp}"):
                        _edited=(st.session_state.get(_exp_key,{}) or {}).get("edited_rows",{}) or {}
                        _n,_bad=_apply_exposure_targets(_ed,exp,_edited,st.session_state["showdown_strategy"])
                        if _bad:
                            st.error("Minimum above maximum for: "+", ".join(_bad)+". Those rows were skipped — fix and Apply again.")
                        else:
                            st.session_state.pop(_exp_key,None)
                            # The Players tab edits these same target columns and its
                            # Apply overwrites every row: reset its editor so a stale
                            # snapshot cannot silently write the old values back.
                            _sed_key=st.session_state.get("v4_sdplayers_key")
                            if _sed_key: st.session_state.pop(_sed_key,None)
                            _excl_warn=[]
                            _df_active={str(r["ID"]):bool(r.get("ActiveForBuild",True)) for _,r in df.iterrows()}
                            _df_name={str(r["ID"]):str(r["Name"]) for _,r in df.iterrows()}
                            for _ri in (_edited or {}):
                                try: _pid=str(exp.iloc[int(_ri)]["ID"])
                                except Exception: continue
                                _st=st.session_state["showdown_strategy"].get(_pid,{}) or {}
                                if float(_st.get("Min Exposure",0) or 0)>0 and (bool(_st.get("Exclude",False)) or not _df_active.get(_pid,True)):
                                    _excl_warn.append(_df_name.get(_pid,_pid))
                            st.success(f"Saved exposure targets for {_n} player{'s' if _n!=1 else ''}. Tap ⚡ Build → GENERATE to rebuild with them.")
                            if _excl_warn:
                                st.warning("Min exposure can't be met while a player is Out or inactive: "+", ".join(sorted(set(_excl_warn)))+". Uncheck Out in the Players tab first.")
                            st.rerun()
                    st.download_button("Download exposure CSV",exp.to_csv(index=False),"showdown_exposure_v6_1.csv","text/csv",use_container_width=True)
                    st.caption("Tip: on iPhone, downloading opens the Files preview — use the app switcher to come back here.")
                    st.button("📋 Back to Lineups",key="sd_exposure_back_to_lineups",use_container_width=True,on_click=_sd_nav_go,args=(_SD_TABS[5],))
            if sd_nav==_SD_TABS[7]:
                from dfs_lab.ui.guide import render_guide
                render_guide(mode="showdown")
        except Exception as e:
            _msg = str(e)
            # Translate common failure modes into actionable guidance.
            if "NoneType" in _msg and "subscriptable" in _msg:
                st.error("Showdown build failed: the player pool didn't load correctly. Re-upload your DK salaries file and try again. If it persists, the file may be the wrong slate format.")
            elif "captain" in _msg.lower() and "eligible" in _msg.lower():
                st.error("Showdown build failed: no captain-eligible players. Check CPT? for at least one player in 👤 Players, tap Apply, and rebuild.")
            elif "salary" in _msg.lower() or "infeasible" in _msg.lower():
                st.error("Showdown build failed: no valid lineup fits your rules. Try loosening a lock, exclusion, or the QB-captain pass-catcher rule, then rebuild.")
            else:
                st.error(f"Showdown build hit a problem: {_msg}. Try rebuilding — if it keeps happening, screenshot this and send it over.")



    st.caption(f"Aytia · Build the story. Test the lineup. Challenge the field. · build {git_build_stamp()}")
