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
from dfs_lab.config import APP_BUILD, PRIORITY_OPTIONS, ROSTER_SLOTS
from dfs_lab.data import prepare_player_pool, prepare_showdown_pool, apply_projection_overrides, apply_post_edit_availability_gate
from dfs_lab.classic import (generate_lineups, classic_apply_qb_cap,
    classic_contest_recommendations, classic_context_evidence,
    classic_portfolio_intelligence, classic_postbuild_answer, classic_postbuild_report,
    classic_qb_concentration_plan, classic_simulate_slate, classic_strategy_theses,
    calculate_exposure_table)
from dfs_lab.showdown import (apply_context_engine, apply_showdown_scenario,
    generate_showdown_lineups, infer_score_script, script_build_adjustments,
    showdown_exposure_table, showdown_upload_csv)


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
    seed = settings["seed"]

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
        st.markdown('<div class="card-sub">DraftKings is required. Without a SaberSim file, DFS Lab builds its own projections and estimated ownership.</div>',unsafe_allow_html=True)
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
        st.info("Upload the DraftKings slate to open DFS LAB.")
        st.stop()

    if mode=="Classic":
        try:
            df=prepare_player_pool(dk_file,ss_file); teams=sorted(df["Team"].dropna().unique().tolist())
            if bool(df.get("Own Estimated",pd.Series([False])).any()):
                st.info("Using DFS Lab projections — no SaberSim file uploaded. Ownership shown is DFS Lab's estimate, built from projection, salary value and position baselines.")
            st.session_state.setdefault("strategy_master",{}); st.session_state.setdefault("team_strategy_master",{})
            st.session_state.setdefault("classic_projection_overrides",{})
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

            q1,q2,q3,q4=st.columns(4)
            q1.metric("Players",len(df)); q2.metric("Teams",len(teams)); q3.metric("Field",f"{int(field_size):,}"); q4.metric("Pool",lineup_count)

            intel=classic_context_evidence(df[["Name","Position","Team","Opponent","Game Info","My Proj"]].copy())
            intel_map=intel.set_index("Name") if not intel.empty else pd.DataFrame()
            sim_input=df[["Name","Position","Team","Opponent","Matchup","My Proj"]+ (["Sim Vol"] if "Sim Vol" in df.columns else [])].copy()
            sim_input["DVP Adj %"]=sim_input["Name"].map(intel_map["DVP Adj %"] if not intel.empty else {}).fillna(0.0)
            sim_input["Sim Proj"]=pd.to_numeric(sim_input["My Proj"],errors="coerce").fillna(0.0)*(1+0.50*pd.to_numeric(sim_input["DVP Adj %"],errors="coerce").fillna(0.0)/100.0)
            _sim_cols=["Name","Position","Team","Matchup","Sim Proj"]+([ "Sim Vol"] if "Sim Vol" in sim_input.columns else [])
            sim_table=classic_simulate_slate(sim_input[_sim_cols],5000,seed)
            thesis_table,thesis_state=classic_strategy_theses(df,intel,sim_table,field_size,payout_style,entry_format)
            classic_qb_ids,qb_plan_table,qb_plan=classic_qb_concentration_plan(df,thesis_table,entry_format)
            classic_qb_ids,qb_plan_table,qb_plan=classic_apply_qb_cap(classic_qb_ids,qb_plan_table,qb_plan,st.session_state.get("classic_qb_cap",0))
            rec=classic_contest_recommendations(field_size,payout_style,entry_format,sim_table)

            tabs=st.tabs(["🧠 Slate Intel","⚡ Build","👤 Players","⚙ Rules","📋 Lineups","📊 Exposure"])

            with tabs[0]:
                st.markdown('<div class="card-title">DFS LAB Slate Intel</div><div class="card-sub">Study the slate first. Then decide which optimizer rules deserve to be used for this contest.</div>',unsafe_allow_html=True)
                contest_desc=f"{entry_format} · {int(field_size):,} entries · {payout_style}"
                st.markdown(f"<div class='intel-card'><div class='intel-kicker'>Contest lens</div><div class='intel-big'>{contest_desc}</div><div class='intel-copy'>DFS LAB changes its recommendations with field size, entry format and payout shape. The same slate should not be built the same way in Single Entry and 150-Max.</div></div>",unsafe_allow_html=True)

                if sim_table is not None and not sim_table.empty:
                    top=sim_table.iloc[0]
                    st.markdown(f"<div class='intel-card'><div class='intel-kicker'>5,000-simulation slate read</div><div class='intel-big'>{top['Game']} has the strongest simulated ceiling footprint</div><div class='intel-copy'>It produced the highest DFS environment in {float(top['Slate ceiling %']):.1f}% of projection-driven simulations. P90 environment: {float(top['P90']):.1f}. These are comparative DFS simulations, not sportsbook game probabilities.</div></div>",unsafe_allow_html=True)
                    sim_cols=["Game","Mean DFS env","P75","P90","Volatility","Slate ceiling %"]
                    st.dataframe(sim_table[[x for x in sim_cols if x in sim_table.columns]],hide_index=True,use_container_width=True,height=min(430,70+35*len(sim_table)))

                st.markdown("#### Context evidence")
                st.caption("DFS LAB blends the uploaded slate with multi-year player results and opponent-vs-position evidence. Current day/night is shown when DraftKings Game Info exposes kickoff time. Travel, weather and injury/news are not invented when the current data feed does not supply them.")
                intel_view=intel.copy()
                if not intel_view.empty:
                    intel_view["Proj"]=intel_view["My Proj"].round(2)
                    intel_view["Hist FPPG"]=intel_view["Hist FPPG"].round(2)
                    intel_view["Recent 6"]=intel_view["Recent 6"].round(2)
                    cols=["Name","Position","Team","Opponent","Proj","Hist FPPG","Recent 6","Hist Games","DVP Adj %","Time"]
                    st.dataframe(intel_view[[x for x in cols if x in intel_view.columns]].sort_values("Proj",ascending=False).head(80),hide_index=True,use_container_width=True,height=430)

                st.markdown("#### Strategy theses")
                st.markdown(f"<div class='intel-card'><div class='intel-kicker'>DFS LAB stance</div><div class='intel-big'>{thesis_state['label']}</div><div class='intel-copy'>{thesis_state['reason']}</div></div>",unsafe_allow_html=True)
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
                        st.success("Top thesis added as a soft lean. DFS LAB can still build through other games and quarterbacks.")
                    if st.session_state.get("classic_thesis_applied"):
                        st.caption("Active thesis lean: "+str(st.session_state["classic_thesis_applied"]))

                st.markdown("#### QB concentration")
                st.markdown(f"<div class='intel-card'><div class='intel-kicker'>Contest concentration</div><div class='intel-big'>{qb_plan['label']}</div><div class='intel-copy'>{qb_plan['reason']}</div></div>",unsafe_allow_html=True)
                if qb_plan_table is not None and not qb_plan_table.empty:
                    qshow=qb_plan_table.copy()
                    qshow["In build pool"]=qshow["ID"].astype(str).isin(set(str(x) for x in classic_qb_ids))
                    st.dataframe(qshow[["QB","Team","Game","Relative %","In build pool","Why"]].head(12),
                        hide_index=True,use_container_width=True,height=min(430,70+35*min(12,len(qshow))))
                st.caption("Single Entry and 3-Max intentionally narrow weak QB paths so candidate lineups express a stance. 150-Max keeps a much wider evidence band for portfolio coverage.")

                st.markdown("#### DFS LAB recommended setup")
                rr1,rr2,rr3,rr4=st.columns(4)
                rr1.metric("QB pass catchers",rec["qb_stack"])
                rr2.metric("Bring-back",rec["bringback"])
                rr3.metric("Salary floor","$"+f"{int(rec['min_salary']):,}")
                rr4.metric("Max from one game",rec["max_game"])
                st.caption(f"Built for {entry_format} in a {int(field_size):,}-entry {payout_style.lower()} contest. Top simulated game ceiling share: {rec['top_game_share']:.1f}%. Recommendations are staged only when you choose Apply.")
                if st.button("APPLY DFS LAB RECOMMENDED RULES",type="primary",use_container_width=True,key="classic_apply_intel"):
                    st.session_state["classic_qb_stack"]=int(rec["qb_stack"])
                    st.session_state["classic_bringback"]=str(rec["bringback"])
                    st.session_state["classic_min_salary"]=int(rec["min_salary"])
                    st.session_state["classic_max_team"]=int(rec["max_team"])
                    st.session_state["classic_max_game"]=int(rec["max_game"])
                    st.session_state["classic_max_te"]=int(rec["max_te"])
                    st.session_state["classic_no_dst"]=bool(rec["no_dst"])
                    st.session_state["classic_no_off"]=bool(rec["no_off"])
                    st.session_state["classic_allow_qb_rb"]=bool(rec["allow_qb_rb"])
                    st.success("DFS LAB recommendations staged in Rules. Review them before building.")

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
                st.caption(f"Build: {APP_BUILD} · The coach only evaluates lineups DFS LAB actually built. It does not change rules or pretend to know evidence that is not in the portfolio.")

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
                            st.markdown(f"<div class='dfs-chat-ai'><b>DFS LAB</b><br>{ar}</div>",unsafe_allow_html=True)

                    with st.form("classic_postbuild_coach_form",clear_on_submit=True):
                        _coach_q=st.text_input("Ask why",placeholder="Why so much Purdy? Why only 2 double stacks? Are these lineups too chalky?")
                        _coach_send=st.form_submit_button("ASK WHY  ↗",type="primary",use_container_width=True)

                    _coach_asked=(_coach_q.strip() if _coach_send and _coach_q.strip() else _why_ask)
                    if _coach_asked:
                        _coach_answer=classic_postbuild_answer(_coach_asked,packet,st.session_state["classic_ai_chat"])
                        st.session_state["classic_ai_chat"].append((_coach_asked,_coach_answer))
                        st.rerun()

                    if st.button("CLEAR COACH HISTORY",use_container_width=True,key="classic_coach_clear"):
                        st.session_state["classic_ai_chat"]=[]
                        st.rerun()

            with tabs[1]:
                st.markdown('<div class="card-title">Build</div><div class="card-sub">Choose team preferences after reviewing Slate Intel, then generate the portfolio with your Rules settings.</div>',unsafe_allow_html=True)
                preferred_stack_teams=st.multiselect("Preferred QB stack teams",teams,key="classic_pref_stack",
                    help="Hard control: if you select teams here, the optimizer must use a QB from one of them. Strategy Theses no longer populate this automatically.")
                team_df=pd.DataFrame({"Team":teams,"Priority":[st.session_state["team_strategy_master"].get(t,"Neutral") for t in teams]})
                st.caption("Your inputs, not the sim's verdict — tell DFS LAB which teams you want more or less of. The sim's own reads live in Slate Intel → Strategy Theses.")
                team_edit=st.data_editor(team_df,hide_index=True,use_container_width=True,disabled=["Team"],column_config={"Priority":st.column_config.SelectboxColumn("Your lean",options=["Core","Like","Neutral","Fade","Exclude"])},key="v4_classic_team")
                for _,r in team_edit.iterrows(): st.session_state["team_strategy_master"][r["Team"]]=r["Priority"]
                if st.session_state.get("classic_thesis_applied"):
                    st.markdown("**Build thesis:** "+str(st.session_state["classic_thesis_applied"]))
                st.markdown("**Automatic QB stance:** "+", ".join(qb_plan.get("names",[])) if qb_plan.get("names") else "**Automatic QB stance:** open")
                st.caption(qb_plan.get("reason",""))
                if int(st.session_state.get("classic_qb_cap",0) or 0)>0:
                    st.info("User QB cap · no more than "+str(int(st.session_state["classic_qb_cap"]))+" quarterbacks · applies to the next build")
                st.info("Current rule set · QB + "+str(st.session_state["classic_qb_stack"])+" pass catcher(s) · Bring-back "+str(st.session_state["classic_bringback"])+" · Min salary $"+f"{int(st.session_state['classic_min_salary']):,}"+" · Max "+str(st.session_state["classic_max_game"])+" from one game")
                # "How can we know this beforehand": who DFS LAB auto-excluded and
                # why, shown BEFORE the Generate button. Reuses the guard's own
                # Auto Excluded Reason column — no new plumbing.
                _auto=df[~df["ActiveForBuild"].astype(bool)].copy()
                _auto=_auto[_auto["Auto Excluded Reason"].astype(str).str.strip().ne("")]
                with st.expander(f"Auto-excluded by DFS LAB ({len(_auto)}) — who's out before you build",expanded=False):
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
                build_btn=st.button(f"⚡ GENERATE {lineup_count} RATED LINEUPS",type="primary",use_container_width=True,key="v4_classic_build")

            with tabs[2]:
                st.markdown('<div class="card-title">Players</div><div class="card-sub">Edit your own projection when you disagree with the model. Your projection becomes the number DFS LAB uses for simulations and lineup building until you reset it.</div>',unsafe_allow_html=True)
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
                with st.form("classic_player_editor_form",clear_on_submit=False):
                    edited=st.data_editor(ed,hide_index=True,use_container_width=True,height=620,
                        disabled=["ID","Name","Pos","Team","Opponent","Salary","Base Proj","Own"],
                        column_order=["Name","Pos","Team","Opponent","Salary","Base Proj","Proj","Own","Lock","Exclude","Priority","Min Exposure","Max Exposure"],
                        column_config={
                            "Name":st.column_config.TextColumn("Player",width=190,pinned=True),"Pos":st.column_config.TextColumn("Pos",width=60),
                            "Team":st.column_config.TextColumn("Team",width=70),"Opponent":st.column_config.TextColumn("Opponent",width=82),
                            "Salary":st.column_config.NumberColumn("Salary",width=85,format="$%d"),
                            "Base Proj":st.column_config.NumberColumn("Base Proj",width=85,format="%.2f",help="Original uploaded/model projection."),
                            "Proj":st.column_config.NumberColumn("My Proj",width=85,format="%.2f",min_value=0.0,step=0.25,help="Editable. DFS LAB uses this value everywhere after you apply changes."),
                            "Own":st.column_config.NumberColumn("Own %",width=70,format="%.1f"),
                            "Priority":st.column_config.SelectboxColumn("Lean",options=PRIORITY_OPTIONS,width=95),"Lock":st.column_config.CheckboxColumn("Lock",width=65),
                            "Exclude":st.column_config.CheckboxColumn("Out",width=60),"Min Exposure":st.column_config.NumberColumn("Min %",min_value=0,max_value=100,step=5,width=75),
                            "Max Exposure":st.column_config.NumberColumn("Max %",min_value=0,max_value=100,step=5,width=75),
                        },key="v4_classic_players")
                    pc1,pc2=st.columns(2)
                    with pc1:
                        apply_classic_players=st.form_submit_button("APPLY PLAYER CHANGES",type="primary",use_container_width=True)
                    with pc2:
                        reset_classic_proj=st.form_submit_button("RESET VISIBLE PROJECTIONS",use_container_width=True)
                if reset_classic_proj:
                    for _,r in edited.iterrows():
                        st.session_state["classic_projection_overrides"].pop(str(r["ID"]),None)
                    st.rerun()
                if apply_classic_players:
                    for _,r in edited.iterrows():
                        pid=str(r["ID"]); ex=bool(r["Exclude"])
                        st.session_state["strategy_master"][pid]={"Lock":bool(r["Lock"]) and not ex,"Exclude":ex,"Priority":"Exclude" if ex else str(r["Priority"]),"Min Exposure":float(r["Min Exposure"]),"Max Exposure":float(r["Max Exposure"])}
                        base=float(r["Base Proj"]); newp=float(r["Proj"])
                        if abs(newp-base)>=0.005:
                            st.session_state["classic_projection_overrides"][pid]=newp
                        else:
                            st.session_state["classic_projection_overrides"].pop(pid,None)
                    st.session_state.pop("classic_result_v4",None)
                    st.rerun()

            with tabs[3]:
                st.markdown('<div class="card-title">Classic Rules</div><div class="card-sub">Contest-aware structure controls. Slate Intel can recommend these, but you decide what gets enforced.</div>',unsafe_allow_html=True)
                r1,r2=st.columns(2)
                with r1:
                    min_salary=st.slider("Minimum salary",44000,50000,int(st.session_state["classic_min_salary"]),100,key="classic_min_salary")
                    qb_stack=st.selectbox("QB pass catchers",[1,2,3],key="classic_qb_stack",help="Minimum same-team WR/TE players paired with the QB.")
                    bringback_mode=st.selectbox("Bring-back",["Optional","Required","None"],key="classic_bringback",help="Required forces at least one opposing RB/WR/TE with the QB stack.")
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
                        flex_rb=st.number_input("RB in FLEX %",0,100,int(st.session_state["classic_flex_rb"]),5,key="classic_flex_rb")
                    with fb:
                        flex_wr=st.number_input("WR in FLEX %",0,100,int(st.session_state["classic_flex_wr"]),5,key="classic_flex_wr")
                    with fc:
                        flex_te=st.number_input("TE in FLEX %",0,100,int(st.session_state["classic_flex_te"]),5,key="classic_flex_te")
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
                    st.caption("Automatic: DFS LAB chooses the FLEX position independently for each lineup.")

                st.markdown("#### Correlation + defense")
                no_dst=st.toggle("No defense from my QB's game",key="classic_no_dst",help="Blocks either defense from the game containing your rostered QB.")
                no_off=st.toggle("No offense against my DST",key="classic_no_off",help="If on, any DST blocks all opposing offensive players.")
                allow_qb_rb=st.toggle("Allow QB + same-team RB",key="classic_allow_qb_rb",help="Turn off if you want QB stacks to avoid same-team running backs.")
                st.caption("QB vs opposing DST is always blocked. More relationship controls can be added here without changing the core optimizer.")

            if build_btn:
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
                              else None)
                )
                st.session_state["classic_result_v4"]=res
                # Slate Intel / Post-Build Coach renders earlier in the script than the Build tab.
                # Rerun once after a successful build so those sections immediately see the new portfolio.
                st.rerun()

            res=st.session_state.get("classic_result_v4")
            with tabs[4]:
                if res is None or res.empty:
                    st.info("Generate lineups from Build.")
                else:
                    show_cols=["Rank","Rating","Rating Score","Projection","Base Projection","Scenario Delta","Salary","Salary Left","Avg Own","Stack Summary"]+ROSTER_SLOTS
                    st.dataframe(res[[x for x in show_cols if x in res.columns]],hide_index=True,use_container_width=True,height=590,
                        column_config={
                            "Rank":st.column_config.NumberColumn("Rank",width=60,pinned=True),
                            "QB":st.column_config.TextColumn("QB",width=150,pinned=True),
                        })
                    st.download_button("Download lineup analysis CSV",res.to_csv(index=False),"classic_lineups_v5.csv","text/csv",use_container_width=True)
            with tabs[5]:
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
                                                  else None)
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
                                    allowed_qb_ids=classic_qb_ids
                                )
                            st.session_state["classic_result_v4"]=new_res
                            st.success("Exposure changes applied and the portfolio was rebuilt.")
                            st.rerun()
                        else:
                            st.success("Exposure targets saved. They will be used on the next build.")
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

            # Slate-specific build controls live in the main DFS LAB workspace — never in Streamlit's sidebar.
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
            tabs=st.tabs(["⚡ Build","👤 Players","🔗 Relationships","🧠 Game Intel","⚙ Rules","📋 Lineups","📊 Exposure"])
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
                with a:
                    qb_cpt_pc_rule = st.selectbox(
                        "When QB is CPT · pass catchers",
                        ["No rule", "Minimum 1", "Minimum 2", "No more than 1", "No more than 2"],
                        index=0,
                        help="Minimum/maximum is enforced first. If that rule makes the entire slate impossible, DFS LAB keeps your other settings and falls back to its football-coherence model rather than returning zero lineups."
                    )
                    cpt_qb_pc = {"No rule":0, "Minimum 1":1, "Minimum 2":2, "No more than 1":-1, "No more than 2":-2}[qb_cpt_pc_rule]
                pair_map={"Never":0,"Sometimes":35,"Usually":80,"Always":100}
                with b: wrte_pair=st.selectbox("When WR/TE is CPT · pair QB",list(pair_map),index=2,
                    help="Sometimes/Usually are SOFT portfolio preferences and will not block a legal build. Always is a hard rule.")
                with c: rb_pair=st.selectbox("When RB is CPT · pair DST/K",list(pair_map),index=1,
                    help="Sometimes/Usually are SOFT portfolio preferences and will not block a legal build. Always is a hard rule.")
                wrte_qb=pair_map[wrte_pair]; rb_ctrl=pair_map[rb_pair]
                d,e=st.columns(2)
                with d:max_k=st.selectbox("Max kickers",[0,1,2],index=2)
                with e:max_dst=st.selectbox("Max defenses",[0,1,2],index=1)
                build_btn=st.button(f"⚡ GENERATE {lineup_count} LINEUPS",type="primary",use_container_width=True,key="v4_sd_build")

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

                ed=pd.DataFrame({"ID":view["ID"].astype(str),"Name":view["Name"],"Pos":view["Position"],"Team":view["Team"],"Flex $":view["FlexSalary"],"DFS Base":view["My Proj"].round(2),"Availability":view.get("Live Status",pd.Series("Not verified",index=view.index)),"Hist G":view.get("History Games",pd.Series(0,index=view.index)),"Matchup %":view.get("Matchup Adj %",pd.Series(0.0,index=view.index)),"Model":view["Model Proj"].round(2),"Your Proj":view["DFS Lab Proj"].round(2),"Δ%":view["Proj Change %"].round(1),"Own":view["My Own"].round(1),"CPT Own":view["CPT Own"].round(1),"Lock":False,"CPT Lock":False,"Exclude":False,"CPT Eligible":True,"Priority":"Neutral","Min Exposure":0,"Max Exposure":100,"CPT Min":0,"CPT Max":100})
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
                    disabled=["ID","Name","Pos","Team","Flex $","DFS Base","Availability","Hist G","Matchup %","Model","Δ%","Own","CPT Own"],
                    column_order=["Name","Availability","Lock","CPT Lock","Exclude","CPT Eligible","Priority","Pos","Team","Flex $","DFS Base","Hist G","Matchup %","Model","Your Proj","Δ%","Own","CPT Own","Min Exposure","Max Exposure","CPT Min","CPT Max"],
                    column_config={
                        "ID":None,
                        "Name":st.column_config.TextColumn("Player",width=190,pinned=True),
                        "Pos":st.column_config.TextColumn("Pos",width=58),
                        "Team":st.column_config.TextColumn("Team",width=68),
                        "Availability":st.column_config.TextColumn("Live",width=115,help="Latest nflverse roster/injury status. OUT/non-active roster statuses are auto-excluded."),
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
                        st.caption(f"Scenario correlation suggestion: QB CPT + {corr_overrides['qb_pc']} pass catcher(s) • WR/TE CPT + QB {corr_overrides['wrte_qb']}% • RB CPT + DST/K {corr_overrides['rb_ctrl']}%")
                        st.caption("Build-tab Captain pairing controls stay authoritative and are not silently overridden.")

                st.markdown("#### Projection movement")
                preview=scenario_df[["Name","Position","Team","My Proj","Script Proj","Proj Change %"]].copy()
                preview.columns=["Player","Pos","Team","Base","Scenario","Change %"]
                preview=preview.sort_values("Change %",key=lambda x:x.abs(),ascending=False).head(14)
                st.dataframe(preview,hide_index=True,use_container_width=True,height=410,column_config={"Player":st.column_config.TextColumn("Player",pinned=True,width=190),"Base":st.column_config.NumberColumn("Base",format="%.2f"),"Scenario":st.column_config.NumberColumn("Scenario",format="%.2f"),"Change %":st.column_config.NumberColumn("Change %",format="%.1f")})
                st.caption("These are bounded scenario tilts applied to the DFS Lab baseline projections. They are not a claim that a final score can precisely predict individual fantasy points.")
                st.markdown("#### How V6.3 grades Showdown")
                st.caption("Projection 29–34% • Captain quality 18% • correlation 20% • leverage 11–15% • duplication proxy 10–15% • your takes 7%. The exact weights move with contest size/payout.")
                st.caption("The scenario engine changes projection and construction inputs before grading. Game Worlds remain portfolio alternatives, so a directional thesis can still include balanced or opposing worlds; those labels describe the world being explored, not a replacement for your selected thesis.")

            # Defaults exist even before the user opens tabs because Streamlit executes all tab bodies.
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
                    build_df.loc[_mask,"DFS Lab Proj"]=float(_val)
            # Final safety gate: a projection changed to zero after pool creation must
            # become unavailable immediately. Previously ActiveForBuild could stay True.
            _zero_final=pd.to_numeric(build_df["DFS Lab Proj"],errors="coerce").fillna(0.0)<=0.01
            build_df.loc[_zero_final,"ActiveForBuild"]=False
            build_df.loc[_zero_final,"Role Confidence"]="Unavailable — zero final projection"
            build_df.loc[_zero_final & build_df["Auto Excluded Reason"].astype(str).eq(""),"Auto Excluded Reason"]="Final DFS LAB projection is 0"
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
            if build_btn:
                result=generate_showdown_lineups(build_df,field_size,payout_style,lineup_count,max(120,lineup_count*5),min_salary,max_salary,build_weights,effective_script,effective_team,strategy_map,entry_format,eff_qb_pc,eff_wrte_qb,eff_rb_ctrl,max_k,max_dst,min_unique,seed,relationship_rules=active_relationships,world_influence=intensity)
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
                        relationship_rules=active_relationships,world_influence=intensity
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
                        relationship_rules=active_relationships,world_influence=intensity
                    )
                    if _qb_fallback is not None and not _qb_fallback.empty:
                        result=_qb_fallback
                        _qb_rule_fallback_used=True

                st.session_state["showdown_result_v4"]=result
                if _pairing_fallback_used:
                    st.warning("DFS LAB built the portfolio after dropping the probabilistic WR/TE→QB and RB→DST/K pairing suggestions. Your explicit QB-Captain pass-catcher rule, salary, locks, CPT pool, exposures and team-build rules were kept.")
                if _qb_rule_fallback_used:
                    st.info("DFS LAB detected that the QB-Captain pass-catcher rule blocked the slate, so it used football-coherence scoring for QB Captain builds instead of returning no lineups. All player locks, exclusions, exposures, salary and team-build settings were kept.")
                if result is not None and not result.empty and len(result) < int(lineup_count):
                    st.warning(f"Built {len(result)} of {int(lineup_count)} requested lineups. DFS LAB stopped after the optimizer stalled on the current hard rules instead of hanging indefinitely. Loosen the rule called out below or build the smaller portfolio.")
                if result is None or result.empty:
                    st.error("DFS LAB could not produce a lineup with the current rules. Running a quick feasibility check…")

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
                        st.warning("**Feasibility diagnostic:** A test build succeeded when DFS LAB relaxed: **" + ", ".join(blockers) + "**. This is a clue, not proof that the setting itself is invalid. Your settings were not changed.")
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
            result=st.session_state.get("showdown_result_v4")

            with tabs[5]:
                st.markdown('<div class="card-title">Rated Lineups</div><div class="card-sub">The grade is portfolio-relative. A+ means one of the strongest lineups in this build — not a guarantee of outcome.</div>',unsafe_allow_html=True)
                if result is None or result.empty: st.info("Set your build, player takes and script, then generate lineups.")
                else:
                    m1,m2,m3,m4=st.columns(4); m1.metric("A / A+",int(result["Rating"].isin(["A","A+"]).sum())); m2.metric("Top projection",f"{result['Projection'].max():.1f}"); m3.metric("Avg salary left",f"${int(result['Salary Left'].mean()):,}"); m4.metric("Built",len(result))
                    _world_total=int(result["Game World"].nunique()) if "Game World" in result.columns else 0
                    st.markdown(
                        f"<div class='results-hub-hero'><div class='results-hub-kicker'>RESULTS COMMAND CENTER</div>"
                        f"<div class='results-hub-title'>Results Command Center</div>"
                        f"<div class='results-hub-sub'><b>{len(result)} lineups ready.</b> Move between the portfolio, {_world_total} Game Worlds, and Lineup Lab + Agent without leaving this workspace.</div></div>",
                        unsafe_allow_html=True
                    )
                    result_views=st.tabs(
                        [f"🏈  LINEUPS · {len(result)}",f"🌎  GAME WORLDS · {_world_total}","🧠  LINEUP LAB + AGENT"],
                        key="sd_results_hub"
                    )

                    with result_views[0]:
                        st.markdown("<div class='results-view-title'>Your lineup portfolio</div><div class='results-view-sub'>The actual builds come first. Open the full table only when you need it.</div>",unsafe_allow_html=True)
                        # LINEUPS FIRST: the primary output must be visible before the analysis.
                        st.markdown("### 🏈 Your lineups")
                        st.caption("Built successfully. Start with the actual lineups; portfolio analysis and Game Worlds are below.")
                        import html as _html
                        _top_rows=list(result.head(3).iterrows())
                        _top_cols=st.columns(3)
                        _headshot_map={}
                        if "Headshot URL" in build_df.columns:
                            _headshot_map={str(r["Name"]):str(r.get("Headshot URL","")) for _,r in build_df.iterrows()
                                           if str(r.get("Headshot URL","")).lower() not in ["","nan","none"]}
                        def _avatar_html(_name,_captain=False):
                            _safe_name=_html.escape(str(_name))
                            _url=_html.escape(_headshot_map.get(str(_name),""))
                            _cls="player-avatar captain-avatar" if _captain else "player-avatar flex-avatar"
                            if _url:
                                return f"<div class='{_cls}'><img src='{_url}' alt='{_safe_name}'></div>"
                            _initials="".join([x[:1] for x in str(_name).replace("."," ").split()[:2]]).upper() or "NFL"
                            return f"<div class='{_cls} avatar-fallback'>{_html.escape(_initials)}</div>"
                        for _card_col,(_, _top_lr) in zip(_top_cols,_top_rows):
                            with _card_col:
                                _world=str(_top_lr.get("Game World",""))
                                _captain=str(_top_lr["Captain"])
                                _flex_players=[str(_top_lr.get("FLEX"+str(i),"")) for i in range(1,6)]
                                _flex_people="".join(
                                    "<div class='flex-person'>"+_avatar_html(_nm,False)+"<span>"+_html.escape(_nm)+"</span></div>"
                                    for _nm in _flex_players if _nm
                                )
                                _card_html=("<div class='lineup-card lineup-card-grid'><div class='lineup-card-head'><div class='lineup-rank'>#%s</div><div class='lineup-grade'>%s</div></div>" % (int(_top_lr["Rank"]),_top_lr["Rating"]) +
                                           "<div class='captain-feature'>"+_avatar_html(_captain,True)+"<div><div class='lineup-points'>%.1f <small>pts</small></div>" % _top_lr["Projection"] +
                                           "<div class='lineup-cpt'><span>CPT</span> %s</div></div></div>" % _html.escape(_captain) +
                                           "<div class='lineup-flex-grid'>%s</div>" % _flex_people +
                                           "<div class='lineup-meta'>$%s &nbsp; • &nbsp; %s &nbsp; • &nbsp; %s dup risk</div>" % (f"{int(_top_lr['Salary']):,}",_top_lr["Construction"],_top_lr["Dup Risk"]) +
                                           "<div class='lineup-why-strip'><span>%s DUP</span><span>%s CORR</span><span>%s LEVERAGE</span></div>" % (_html.escape(str(_top_lr.get("Dup Risk","—")).upper()),_html.escape(str(_top_lr.get("Correlation Grade","—")).upper()),_html.escape(str(_top_lr.get("Leverage Grade","—")).upper())) +
                                           (("<div class='lineup-world'>◉ &nbsp; %s</div>" % _html.escape(_world)) if _world else "")+"</div>")
                                st.markdown(_card_html,unsafe_allow_html=True)

                        with st.expander(f"See all {len(result)} lineups",expanded=False):
                            st.caption("Tap a lineup row to see why DFS LAB built it and what drives the grade.")
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
                            _story=str(_why.get("Lineup Story","") or _why.get("Story","") or "DFS LAB built this lineup as a distinct path to the slate ceiling.")
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
                                "<div class='why-section'><b>Why DFS LAB likes it</b><p>%s</p></div>"
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

                    with result_views[1]:
                        st.markdown("<div class='results-view-title'>Game Worlds</div><div class='results-view-sub'>See the different ways DFS LAB thinks this slate can unfold, then isolate the lineups built for any one world.</div>",unsafe_allow_html=True)
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
                    with result_views[2]:
                        st.markdown("<div class='results-view-title'>Lineup Lab</div><div class='results-view-sub'>Pick any lineup, see the football story behind all six players, challenge assumptions, test swaps, and talk directly to DFS LAB Agent.</div>",unsafe_allow_html=True)
                        st.markdown("#### Lineup Explorer")
                        st.caption("Highlight a lineup and DFS LAB will analyze all six players together — not just the Captain.")
                        lineup_choices=[f"#{int(r['Rank'])} · {r['Captain']} CPT · {r['Construction']} · {r['Projection']:.1f} pts" for _,r in filtered_result.iterrows()]
                        pick=st.selectbox("Choose lineup ▾",lineup_choices,key="lineup_lab_pick",help="Tap to switch the lineup DFS LAB is analyzing.")
                        li=lineup_choices.index(pick); lr=filtered_result.iloc[li]
                        risk = "lower" if entry_format=="Single Entry" else ("moderate" if entry_format in ["3-Max","20-Max"] else "higher")
                        contest_reason=(f"{entry_format} with {int(field_size):,} entries. DFS Lab uses {risk} tolerance for fragile salary-relief plays and weights tournament ceiling/correlation accordingly.")
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
                            pr=pr.iloc[0]; role="Captain / ceiling engine" if j==0 else ("Primary projection piece" if float(pr.get('DFS Lab Proj',0))>=12 else "Salary relief / secondary path")
                            detail_rows.append({"Slot":"CPT" if j==0 else f"FLEX {j}","Player":nm,"Pos":pr.get('Position',''),"Team":pr.get('Team',''),"DFS Lab":round(float(pr.get('DFS Lab Proj',0)),2),"Salary":int(pr.get('FlexSalary',0)) if j else int(pr.get('CaptainSalary',pr.get('CPTSalary',0))),"Purpose":role})
                        if detail_rows:
                            st.dataframe(pd.DataFrame(detail_rows),hide_index=True,use_container_width=True,column_config={"Player":st.column_config.TextColumn("Player",pinned=True),"Salary":st.column_config.NumberColumn("Salary",format="$%d")})
                            low=min(detail_rows,key=lambda x:x["DFS Lab"])
                            st.write(f"**Weakest projection link:** {low['Player']} ({low['DFS Lab']:.2f}). DFS LAB is using this spot as {low['Purpose'].lower()} within the six-player construction.")
                        # ---------- DFS LAB AGENT / SCENARIO CONSOLE ----------
                        st.markdown("### 🧠 DFS LAB Agent")
                        st.caption("Question the model, compare players, test assumptions, and turn a conversation into a reversible lineup scenario.")
                        st.session_state.setdefault("agent_projection_scenario",{})
                        st.session_state.setdefault("dfs_lab_chat",[])
                        st.session_state.setdefault("agent_last_players",[])

                        def _player_record(name):
                            z=build_df[build_df['Name'].astype(str).str.lower().eq(str(name).lower())]
                            return None if z.empty else z.iloc[0]

                        def _mentioned_players(qtext):
                            q=str(qtext or '').lower(); hits=[]
                            names_all=[str(x) for x in build_df['Name'].dropna().astype(str).unique()]
                            for nm in names_all:
                                last=nm.split()[-1].lower()
                                if nm.lower() in q or (len(last)>2 and re.search(r'\b'+re.escape(last)+r'\b',q)):
                                    hits.append(nm)
                            if not hits:
                                words=[w for w in re.findall(r"[a-zA-Z][a-zA-Z'\-]+",q) if len(w)>=4]
                                last_map={nm.split()[-1].lower():nm for nm in names_all}
                                full_map={nm.lower():nm for nm in names_all}
                                for w in words:
                                    m=difflib.get_close_matches(w,list(last_map.keys()),n=1,cutoff=.82)
                                    if m and last_map[m[0]] not in hits: hits.append(last_map[m[0]])
                                if not hits:
                                    m=difflib.get_close_matches(q,list(full_map.keys()),n=1,cutoff=.72)
                                    if m: hits.append(full_map[m[0]])
                            if not hits and any(x in q for x in ['he ','him ','his ','one ','them ','those ']):
                                hits=st.session_state.get('agent_last_players',[])
                            if hits: st.session_state['agent_last_players']=hits[:4]
                            return hits[:4]
                        def build_agent_packet():
                            pool_cols=[c for c in ['ID','Name','Position','Team','DFS Lab Proj','FlexSalary','CPTSalary','DFS Base','Model Proj','Script Proj','Proj Change %'] if c in build_df.columns]
                            pool=build_df[pool_cols].copy().sort_values('DFS Lab Proj',ascending=False).head(60)
                            roster=[]
                            for rr in detail_rows:
                                rec=dict(rr); pr=_player_record(rr['Player'])
                                if pr is not None:
                                    pid=str(pr.get('ID','')); rec['ID']=pid
                                    rec['Base Before Agent']=float(pr.get('Model Proj',pr.get('DFS Lab Proj',0)))
                                    rec['Agent Scenario']=st.session_state['agent_projection_scenario'].get(pid)
                                roster.append(rec)
                            return {'contest':{'entry_format':entry_format,'field_size':int(field_size),'payout':payout_style},
                                'scenario':{'script':effective_script,'team':effective_team,'influence':int(intensity),'agent_projection_overrides':st.session_state['agent_projection_scenario']},
                                'active_lineup':{'rank':int(lr['Rank']),'rating':str(lr['Rating']),'projection':float(lr['Projection']),'salary':int(lr['Salary']),'salary_left':int(lr['Salary Left']),'construction':str(lr['Construction']),'captain':str(lr['Captain']),'game_world':str(lr.get('Game World',effective_script)),'world_thesis':str(lr.get('World Thesis','')),'players':roster},
                                'portfolio':{'lineups':int(len(result)),'avg_projection':round(float(result['Projection'].mean()),2),'top_projection':round(float(result['Projection'].max()),2),'world_counts':result['Game World'].value_counts().head(10).to_dict() if 'Game World' in result.columns else {},'captain_counts':result['Captain'].value_counts().head(12).to_dict() if 'Captain' in result.columns else {}},
                                'player_pool':pool.to_dict(orient='records')}

                        def local_agent_answer(qtext, packet):
                            q=str(qtext or '').lower().strip(); names=_mentioned_players(qtext)
                            recs=[]
                            for nm in names:
                                pr=_player_record(nm)
                                if pr is not None: recs.append(pr)

                            qwords=re.findall(r"[a-z]+",q)

                            typo_neg=any(difflib.SequenceMatcher(None,w,"wont").ratio()>=0.72 for w in qwords)

                            challenge=(any(x in q for x in ["don't like","dont like","do not like","hate ","not sold","don't want","dont want","get rid","remove ","fade ","off of ","too high","overprojected","over projected","won't score","wont score","won’t score","won't get","wont get","won’t get","score that much","get that much","projected high","projection high","seems high","too much","not score","not get"]) or (typo_neg and any(x in q for x in ["score","get","project","points","much"])))
                            if recs and challenge:
                                pr=recs[0]; nm=str(pr['Name']); proj=float(pr.get('DFS Lab Proj',0))
                                rr=next((x for x in detail_rows if str(x.get('Player','')).lower()==nm.lower()),None)
                                if rr:
                                    slot=str(rr.get('Slot','FLEX')).upper(); lineup_salary=int(packet['active_lineup']['salary'])
                                    old_cost=int(pr.get('CaptainSalary',pr.get('CPTSalary',0))) if slot=='CPT' else int(pr.get('FlexSalary',0))
                                    max_cost=50000-(lineup_salary-old_cost)
                                    used={str(x.get('Player','')) for x in detail_rows}
                                    sal_col='CaptainSalary' if slot=='CPT' and 'CaptainSalary' in build_df.columns else ('CPTSalary' if slot=='CPT' and 'CPTSalary' in build_df.columns else 'FlexSalary')
                                    legal=build_df[(~build_df['Name'].astype(str).isin(used)) & (pd.to_numeric(build_df[sal_col],errors='coerce').fillna(999999)<=max_cost)].copy()
                                    legal=legal.sort_values('DFS Lab Proj',ascending=False)
                                    direct=None if legal.empty else legal.iloc[0]
                                    alt_rows=[]
                                    for _,cand_line in result.iterrows():
                                        cand_names=[str(cand_line.get('Captain',''))]+[str(cand_line.get('FLEX'+str(i),'')) for i in range(1,6)]
                                        if nm not in cand_names: alt_rows.append(cand_line)
                                    best_no_player=alt_rows[0] if alt_rows else None
                                    if best_no_player is not None:
                                        new_names=[str(best_no_player.get('Captain',''))]+[str(best_no_player.get('FLEX'+str(i),'')) for i in range(1,6)]
                                        current_names=[str(x.get('Player','')) for x in packet['active_lineup']['players']]
                                        outs=[x for x in current_names if x not in new_names]
                                        ins=[x for x in new_names if x not in current_names]
                                        changes=[]
                                        if outs: changes.append('remove ' + ', '.join(outs))
                                        if ins: changes.append('add ' + ', '.join(ins))
                                        change_text='; '.join(changes) if changes else 'use that lineup as built'
                                        proj_diff=float(best_no_player.get('Projection',0))-float(packet['active_lineup']['projection'])
                                        return f"**I’d move off {nm}.** DFS LAB has him at **{proj:.2f}**, and your concern is enough to test the best valid build without him. **My recommendation is lineup #{int(best_no_player.get('Rank',0))}: {best_no_player.get('Captain','')} at Captain**, projected **{float(best_no_player.get('Projection',0)):.2f}** ({proj_diff:+.2f} vs this lineup), salary **${int(best_no_player.get('Salary',0)):,}**. To get there: **{change_text}**. That recommendation is already salary-cap legal and comes from the generated portfolio, so you do not need to pick a replacement manually."
                                    if direct is not None:
                                        dproj=float(direct['DFS Lab Proj'])-proj
                                        return f"**I’d move off {nm}.** The best salary-cap-legal direct replacement is **{direct['Name']}** at **{float(direct['DFS Lab Proj']):.2f}** and **${int(direct[sal_col]):,}**, a projection change of **{dproj:+.2f}**."
                                    return f"**I’d fade {nm}, but there is no legal one-for-one swap in this build.** The next step should be a two-player rebuild, not forcing an illegal replacement."
                                return f"**I’d treat {nm} as a fade for the next build.** DFS LAB has him at **{proj:.2f}**, but he is not in the active lineup."

                            if len(recs)>=2:
                                a,b=recs[0],recs[1]; ap=float(a['DFS Lab Proj']); bp=float(b['DFS Lab Proj']); gap=abs(ap-bp)
                                asal=int(a.get('FlexSalary',0)); bsal=int(b.get('FlexSalary',0)); sg=abs(asal-bsal)
                                return f"**That's a real projection question.** DFS LAB has **{a['Name']} at {ap:.2f}** and **{b['Name']} at {bp:.2f}** — only **{gap:.2f} DK points apart**, with a **${sg:,} salary difference**. I wouldn't just accept that gap. We can challenge either assumption without changing the base model. **Lower {a['Name']}, lower {b['Name']}, adjust both, or investigate first.**"
                            m=re.search(r'(?:scored?|gets?|got|puts? up)\s+(-?\d+(?:\.\d+)?)',q)
                            if recs and m:
                                actual=float(m.group(1)); nm=str(recs[0]['Name']); expected=float(recs[0]['DFS Lab Proj']); delta=actual-expected
                                return f"**Scenario test:** {nm} {expected:.2f} → **{actual:.2f}** ({delta:+.2f}). I can keep that as a temporary Agent Scenario and rebuild around it without touching DFS LAB's base projection."
                            if recs:
                                pr=recs[0]; return f"**{pr['Name']} is at {float(pr['DFS Lab Proj']):.2f}.** Tell me what you dislike about the play, or I can test removing him from this build and compare the salary-feasible alternatives."
                            return "I couldn't resolve the player or request from the current slate. Rephrase it with the player's last name and I’ll evaluate that player against this exact lineup instead of giving you a generic lineup summary."
                        def build_question_evidence(qtext, packet):
                            """Deterministic calculator layer. The model reasons; DFS LAB supplies the math."""
                            names=_mentioned_players(qtext)
                            evidence={"mentioned_players":[],"comparison":None,"what_if":None,"active_lineup_test":None}
                            for nm in names:
                                pr=_player_record(nm)
                                if pr is None: continue
                                rr=next((x for x in detail_rows if str(x.get('Player','')).lower()==str(nm).lower()),None)
                                evidence["mentioned_players"].append({
                                    "name":nm,
                                    "projection":round(float(pr.get('DFS Lab Proj',0)),2),
                                    "flex_salary":int(pr.get('FlexSalary',0)),
                                    "captain_salary":int(pr.get('CaptainSalary',pr.get('CPTSalary',0))),
                                    "position":str(pr.get('Position','')),
                                    "team":str(pr.get('Team','')),
                                    "in_active_lineup":bool(rr),
                                    "slot":rr.get('Slot') if rr else None,
                                    "purpose":rr.get('Purpose') if rr else None,
                                })
                            if len(evidence["mentioned_players"])>=2:
                                a,b=evidence["mentioned_players"][:2]
                                evidence["comparison"]={
                                    "players":[a['name'],b['name']],
                                    "projection_gap":round(abs(a['projection']-b['projection']),2),
                                    "salary_gap":abs(a['flex_salary']-b['flex_salary']),
                                    "better_points_per_dollar": (a['name'] if (a['projection']/max(a['flex_salary'],1))>(b['projection']/max(b['flex_salary'],1)) else b['name'])
                                }
                            num=None
                            pats=[r'(?:score|scores|scored|gets?|got|puts? up|project(?:ed)?(?: for)?|at|to)\s*(?:about\s*)?(-?\d+(?:\.\d+)?)',r'(-?\d+(?:\.\d+)?)\s*(?:dk\s*)?points?']
                            for pat in pats:
                                m=re.search(pat,qtext,re.I)
                                if m:
                                    try: num=float(m.group(1)); break
                                    except Exception: pass
                            if num is not None and evidence["mentioned_players"]:
                                target=evidence["mentioned_players"][0]
                                current=float(target['projection']); delta=float(num-current)
                                evidence["what_if"]={"player":target['name'],"current_projection":current,"assumed_projection":num,"projection_delta":round(delta,2)}
                                if target['in_active_lineup']:
                                    mult=1.5 if str(target.get('slot','')).upper()=='CPT' else 1.0
                                    revised=float(packet['active_lineup']['projection']) + delta*mult
                                    rescored=[]
                                    for _,r in result.iterrows():
                                        names_in=[str(r.get('Captain',''))]+[str(r.get('FLEX'+str(i),'')) for i in range(1,6)]
                                        adj=float(r.get('Projection',0))
                                        if target['name'] in names_in:
                                            adj += delta*(1.5 if str(r.get('Captain',''))==target['name'] else 1.0)
                                        rescored.append((adj,int(r.get('Rank',0)),str(r.get('Captain','')),names_in))
                                    rescored.sort(key=lambda x:x[0],reverse=True)
                                    active_players=[str(x.get('Player','')) for x in packet['active_lineup']['players']]
                                    active_key=set(active_players)
                                    new_rank=None
                                    for ix,x in enumerate(rescored,1):
                                        if set(x[3])==active_key and x[2]==packet['active_lineup']['captain']:
                                            new_rank=ix; break
                                    better=sum(1 for x in rescored if x[0]>revised+1e-9)
                                    evidence["active_lineup_test"]={
                                        "original_lineup_projection":round(float(packet['active_lineup']['projection']),2),
                                        "revised_lineup_projection":round(revised,2),
                                        "lineup_projection_change":round(delta*mult,2),
                                        "existing_generated_lineups_now_above_it":int(better),
                                        "counterfactual_rank_within_existing_portfolio":new_rank,
                                        "portfolio_size":len(rescored),
                                        "note":"This re-scores the already-generated portfolio; it is not a fresh optimizer run."
                                    }
                            return evidence

                        def run_ai_agent(qtext, packet, history):
                            try:
                                from openai import OpenAI
                                try: api_key=st.secrets.get('OPENAI_API_KEY',None)
                                except Exception: api_key=os.getenv('OPENAI_API_KEY')
                                if not api_key: return None
                                hist='\n'.join([f"USER: {x[0]}\nDFS LAB: {x[1]}" for x in history[-10:]])
                                calc=build_question_evidence(qtext,packet)
                                instructions="""You are DFS LAB Agent, a sharp NFL DraftKings Showdown analyst embedded inside an optimizer. You are not a generic chatbot and you are not here to defend the optimizer.

    On every message: infer what the user actually means even with typos, fragments, shorthand or follow-ups; resolve all players and conversation references; use DFS LAB CALCULATOR EVIDENCE for math; answer the exact question first; question DFS LAB's own projections when warranted; and never invent projections, salaries, lineup ranks, ownership, injuries, news, simulations or optimizer results. If the user expresses dislike, distrust, avoidance or a fade preference for a player, treat that as a request to evaluate replacing/fading that player in the active lineup: answer that preference directly, identify what the lineup is giving up, and discuss supported alternatives. Do not respond with a generic explanation of why the existing lineup was selected.

    If the user gives a hypothetical score, analyze that exact score. If the calculator re-scores the existing portfolio, clearly call it a re-score, not a fresh optimization. Never say a lineup is still optimal unless a fresh optimizer run proves it. For comparisons, discuss both players, the projection gap, salary/value context, and what it means to this six-player build. If a user challenges a projection, you may recommend a reversible Agent Scenario. Actual changes to projections, locks, exclusions, exposures, Game Worlds or lineup generation require confirmation.

    Be conversational, concise, and useful. Sound like a strong DFS partner sitting next to the user. Avoid canned filler such as 'I can inspect...' when evidence already answers the question. If evidence is insufficient, say exactly what is missing and give the best supported observation."""
                                prompt=f"{instructions}\n\nRECENT CONVERSATION:\n{hist}\n\nCURRENT DFS LAB STATE:\n{json.dumps(packet,default=str)}\n\nDFS LAB CALCULATOR EVIDENCE FOR THIS QUESTION:\n{json.dumps(calc,default=str)}\n\nUSER MESSAGE:\n{qtext}"
                                resp=OpenAI(api_key=api_key).responses.create(
                                    model='gpt-5.6-sol',
                                    reasoning={"effort":"medium"},
                                    input=prompt,
                                    max_output_tokens=1400
                                )
                                return resp.output_text
                            except Exception as e:
                                st.session_state['dfs_agent_error']=str(e); return None

                        # Scenario strip
                        active_scen=st.session_state['agent_projection_scenario']
                        if active_scen:
                            labels=[]
                            for pid,val in active_scen.items():
                                z=build_df[build_df['ID'].astype(str).eq(str(pid))] if 'ID' in build_df.columns else pd.DataFrame()
                                if not z.empty: labels.append(f"{z.iloc[0]['Name']} {float(val):.1f}")
                            st.markdown(f"<div class='agent-scenario'><b>⚡ ACTIVE AGENT SCENARIO</b><span>{' · '.join(labels)}</span></div>",unsafe_allow_html=True)
                            if st.button("↺ RESET AGENT SCENARIO",key='reset_agent_scenario'):
                                st.session_state['agent_projection_scenario']={}; st.rerun()

                        qcols=st.columns(4)
                        quick_prompts=['Explain this build','Question these projections','Find the hidden risk','What is this lineup betting on?']
                        for qi,(qc,qp) in enumerate(zip(qcols,quick_prompts)):
                            if qc.button(qp,key=f'agent_quick_{qi}',use_container_width=True): st.session_state['dfs_agent_pending']=qp
                        with st.form('dfs_lab_agent_form',clear_on_submit=True):
                            ai_q=st.text_area('Ask DFS LAB',placeholder='Try: I don’t believe Shakir and Palmer should be this close…',height=88,label_visibility='collapsed')
                            ask_submit=st.form_submit_button('ASK DFS LAB  ↗',type='primary',use_container_width=True)
                        pending=st.session_state.pop('dfs_agent_pending',None)
                        question=(ai_q.strip() if ask_submit and ai_q.strip() else pending)
                        if question:
                            packet=build_agent_packet(); names=_mentioned_players(question)
                            # Stage explicit natural-language projection commands for confirmation.
                            staged=None
                            setm=re.search(r'(?:set|put|make|lower|raise)\\s+(.+?)\\s+(?:to|at)\\s+(\\d+(?:\\.\\d+)?)',question,re.I)
                            if setm:
                                target=setm.group(1).strip().lower(); val=float(setm.group(2))
                                matches=[nm for nm in build_df['Name'].astype(str).unique() if nm.lower() in target or nm.split()[-1].lower() in target]
                                if matches:
                                    pr=_player_record(matches[0]); staged={'id':str(pr.get('ID','')),'name':matches[0],'old':float(pr['DFS Lab Proj']),'new':val}
                                    st.session_state['agent_staged_projection']=staged
                            with st.spinner('Checking players → lineup → Game World → portfolio…'):
                                response=run_ai_agent(question,packet,st.session_state['dfs_lab_chat']) or local_agent_answer(question,packet)
                            st.session_state['dfs_lab_chat'].append((question,response))

                        staged=st.session_state.get('agent_staged_projection')
                        if staged:
                            st.markdown(f"<div class='scenario-proposal'><b>PROPOSED SCENARIO CHANGE</b><br>{staged['name']} &nbsp; {staged['old']:.2f} → <b>{staged['new']:.2f}</b><br><span>Temporary Agent Scenario · base projection stays untouched</span></div>",unsafe_allow_html=True)
                            ca,cb=st.columns(2)
                            if ca.button('✓ APPLY TO AGENT SCENARIO',type='primary',use_container_width=True,key='apply_agent_proj'):
                                st.session_state['agent_projection_scenario'][staged['id']]=float(staged['new']); st.session_state.pop('agent_staged_projection',None); st.rerun()
                            if cb.button('CANCEL',use_container_width=True,key='cancel_agent_proj'):
                                st.session_state.pop('agent_staged_projection',None); st.rerun()

                        if st.session_state['dfs_lab_chat']:
                            latest_q,latest_a=st.session_state['dfs_lab_chat'][-1]
                            st.markdown("<div class='answer-kicker'>LATEST INVESTIGATION</div>",unsafe_allow_html=True)
                            st.markdown(f"<div class='chat-user'><div class='chat-label'>YOU</div>{latest_q}</div>",unsafe_allow_html=True)
                            st.markdown("<div class='chat-agent-label'>DFS LAB AGENT</div>",unsafe_allow_html=True); st.markdown(latest_a)
                            older=st.session_state['dfs_lab_chat'][:-1]
                            if older:
                                with st.expander(f"Earlier conversation · {len(older)}",expanded=False):
                                    for uq,ar in reversed(older[-8:]): st.markdown(f"**You:** {uq}"); st.markdown(ar); st.divider()
                        if st.session_state.get('dfs_agent_error'):
                            st.warning(f"Agent status: API fallback is active. {st.session_state.get('dfs_agent_error','unknown')}")

                        if st.session_state['dfs_lab_chat']:
                            with st.form('dfs_lab_followup_form', clear_on_submit=True):
                                follow_q=st.text_input('Reply to DFS LAB', placeholder='Reply to DFS LAB…', label_visibility='collapsed')
                                follow_submit=st.form_submit_button('SEND REPLY  ↗', type='primary', use_container_width=True)
                            if follow_submit and follow_q.strip():
                                st.session_state['dfs_agent_pending']=follow_q.strip()
                                st.rerun()

                        st.write(f"DFS Lab selected **{lr['Captain']} at Captain** while preserving the {lr['Construction']} game construction because this combination ranked strongly under the current projection, correlation, salary and contest-risk settings. {lr.get('Strategy Notes','')}")
                        if float(lr.get('Scenario Delta',0))!=0: st.write(f"Your game thesis moved this lineup by **{float(lr['Scenario Delta']):+.2f} projected DK points** versus the unadjusted baseline.")
                        st.markdown("##### Manual swap check (optional)")
                        st.caption("The Agent should recommend the move first. Use this only to inspect a specific one-for-one swap. Only salary-cap-legal replacements are shown.")
                        out_player=st.selectbox("Replace",roster_names,key="lab_out")
                        if out_player:
                            po=build_df[build_df['Name'].astype(str).eq(out_player)].iloc[0]
                            out_is_cpt=(str(out_player)==str(lr['Captain']))
                            out_sal_col='CaptainSalary' if out_is_cpt and 'CaptainSalary' in build_df.columns else ('CPTSalary' if out_is_cpt and 'CPTSalary' in build_df.columns else 'FlexSalary')
                            old_sal=int(po.get(out_sal_col,po.get('FlexSalary',0)))
                            max_in_salary=50000-(int(lr['Salary'])-old_sal)
                            legal_pool=build_df[~build_df['Name'].astype(str).isin(roster_names)].copy()
                            legal_pool=legal_pool[pd.to_numeric(legal_pool[out_sal_col],errors='coerce').fillna(999999)<=max_in_salary]
                            legal_pool=legal_pool.sort_values('DFS Lab Proj',ascending=False)
                            pool_names=legal_pool['Name'].astype(str).tolist()
                            if pool_names:
                                in_player=st.selectbox("With",pool_names,key="lab_in")
                                pi=legal_pool[legal_pool['Name'].astype(str).eq(in_player)].iloc[0]
                                dproj=float(pi['DFS Lab Proj'])-float(po['DFS Lab Proj']); dsal=int(pi[out_sal_col])-old_sal
                                st.write(f"**Legal direct swap:** projection {dproj:+.2f} · salary {dsal:+,} · new salary ${int(lr['Salary'])+dsal:,}.")
                            else:
                                st.info("No one-for-one replacement fits the salary cap. Ask DFS LAB Agent for the best two-player rebuild instead.")
                        with st.expander("Detailed lineup grades & diagnostics",expanded=False):
                            cols=["Rank","Rating","CPT","FLEX1","FLEX2","FLEX3","FLEX4","FLEX5","Projection","Salary","Salary Left","Construction","Coherence Score","Lineup Story","Coherence Flags","CPT Own","Dup Risk","Projection Grade","Captain Grade","Correlation Grade","Leverage Grade","Duplication Grade","Story"]
                            st.dataframe(result[[x for x in cols if x in result.columns]],hide_index=True,use_container_width=True,height=610)
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



    st.markdown(styles.MAIN_V634_SHELL_CSS, unsafe_allow_html=True)

    st.caption("DFS LAB · Build the story. Test the lineup. Challenge the field.")


    st.markdown(styles.MAIN_DYNAMIC_SHELL_CSS, unsafe_allow_html=True)


    # Final global theme lock: keep the same visual treatment before and after lineups are generated.
    st.markdown(styles.MAIN_UNIFORM_THEME_CSS, unsafe_allow_html=True)


    # Final readability pass: one coherent blue/ink palette, readable dark bars, and lineup headshots.
    st.markdown(styles.MAIN_EXPANDERS_CSS, unsafe_allow_html=True)


    # iPad contrast hard-stop: do not use dark expander/control bars on the light workspace.
    # This deliberately makes the surface readable even if Streamlit's internal text classes change.
    st.markdown(styles.MAIN_EXPANDER_HEADERS_CSS, unsafe_allow_html=True)


    # Lineup WHY drawer: restore the explanation layer without widening the lineup table.
    st.markdown(styles.MAIN_WHY_STRIP_CSS, unsafe_allow_html=True)
