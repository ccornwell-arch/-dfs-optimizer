from pathlib import Path

p = Path('streamlit_app.py')
s = p.read_text()

old = '''        if build_btn:
            result=generate_showdown_lineups(build_df,field_size,payout_style,lineup_count,max(350,lineup_count*15),min_salary,max_salary,build_weights,effective_script,effective_team,strategy_map,entry_format,eff_qb_pc,eff_wrte_qb,eff_rb_ctrl,max_k,max_dst,min_unique,seed,relationship_rules=active_relationships,world_influence=intensity)
            st.session_state["showdown_result_v4"]=result
            if result is None or result.empty:
                st.error("No legal lineup found. Check minimum salary, locks/outs, Captain eligibility, allowed team builds, and Captain-pairing rules. Try one change at a time; DFS Lab will preserve your player settings.")
            elif len(result) < lineup_count:
                st.warning(f"Built {len(result)} of {lineup_count} requested lineups. The current salary, uniqueness, locks or exposure limits are restricting the pool.")
'''

new = '''        if build_btn:
            result=generate_showdown_lineups(build_df,field_size,payout_style,lineup_count,max(350,lineup_count*15),min_salary,max_salary,build_weights,effective_script,effective_team,strategy_map,entry_format,eff_qb_pc,eff_wrte_qb,eff_rb_ctrl,max_k,max_dst,min_unique,seed,relationship_rules=active_relationships,world_influence=intensity)
            st.session_state["showdown_result_v4"]=result
            if result is None or result.empty:
                st.error("No legal lineup found with the current rules. DFS Lab is checking the exact constraint that blocks the build…")

                def _diag_build(_strategy=strategy_map, _min_salary=min_salary, _weights=build_weights,
                                _qb=eff_qb_pc, _wrte=eff_wrte_qb, _rb=eff_rb_ctrl,
                                _rels=active_relationships, _unique=min_unique,
                                _max_k=max_k, _max_dst=max_dst):
                    try:
                        x=generate_showdown_lineups(
                            build_df,field_size,payout_style,1,120,_min_salary,max_salary,_weights,
                            effective_script,effective_team,_strategy,entry_format,
                            _qb,_wrte,_rb,_max_k,_max_dst,_unique,seed+991,
                            relationship_rules=_rels,world_influence=intensity)
                        return x is not None and not x.empty
                    except Exception:
                        return False

                tests=[]
                tests.append(("Relationship rules", _diag_build(_rels=[])))
                tests.append(("Captain-pairing rules", _diag_build(_qb=0,_wrte=0,_rb=0)))
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
                    st.warning("**Feasibility diagnostic:** A legal lineup appears as soon as DFS Lab relaxes: **" + ", ".join(blockers) + "**. Your settings were not changed.")
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
'''

if old not in s:
    raise SystemExit('Target build block not found; no changes made')

p.write_text(s.replace(old, new, 1))
print('Patched Showdown feasibility diagnostics')
