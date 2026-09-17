from pathlib import Path

p=Path('streamlit_app.py')
s=p.read_text()

old='''                    negative=any(x in q for x in ["don't like","dont like","do not like","hate ","not sold","don't want","dont want","get rid","remove ","fade ","off of "])
                    if recs and negative:
                        pr=recs[0]; nm=str(pr['Name']); proj=float(pr.get('DFS Lab Proj',0))
                        rr=next((x for x in detail_rows if str(x.get('Player','')).lower()==nm.lower()),None)
                        if rr:
                            slot=str(rr.get('Slot','FLEX')).upper(); lineup_salary=int(packet['active_lineup']['salary'])
                            old_cost=int(pr.get('CaptainSalary',pr.get('CPTSalary',0))) if slot=='CPT' else int(pr.get('FlexSalary',0))
                            max_cost=50000-(lineup_salary-old_cost)
                            used={str(x.get('Player','')) for x in detail_rows}
                            sal_col='CaptainSalary' if slot=='CPT' and 'CaptainSalary' in build_df.columns else ('CPTSalary' if slot=='CPT' and 'CPTSalary' in build_df.columns else 'FlexSalary')
                            alts=build_df[(~build_df['Name'].astype(str).isin(used)) & (pd.to_numeric(build_df[sal_col],errors='coerce').fillna(999999)<=max_cost)].copy()
                            alts=alts.sort_values('DFS Lab Proj',ascending=False).head(4)
                            cand=', '.join([f"{r['Name']} ({float(r['DFS Lab Proj']):.2f}, ${int(r[sal_col]):,})" for _,r in alts.iterrows()])
                            extra=(f" Salary-feasible one-for-one candidates are: **{cand}**." if cand else " I don't see a clean one-for-one salary-feasible alternative in the current pool.")
                            return f"**Then I would challenge {nm} in this lineup, not defend the pick.** He's in the **{slot}** slot at **{proj:.2f} projected DK points**. Your preference is a reason to test a version without him.{extra} Those are candidates, not a claim that the current lineup stays valid — a fresh rebuild is needed to verify construction/correlation and find the best replacement."
                        return f"**Then I would treat {nm} as a fade for the next build.** DFS LAB has him at **{proj:.2f}**, but he isn't in the active six-player lineup I'm evaluating right now."
'''
new='''                    challenge=any(x in q for x in ["don't like","dont like","do not like","hate ","not sold","don't want","dont want","get rid","remove ","fade ","off of ","too high","overprojected","over projected","won't score","wont score","won’t score","won't get","wont get","won’t get"])
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
'''
if old not in s: raise SystemExit('challenge block not found')
s=s.replace(old,new,1)

old2='''                st.markdown("##### Challenge this lineup")
                st.caption("Swap a player manually, or ask the Agent to challenge a projection and stage a reversible scenario.")
                out_player=st.selectbox("Replace",roster_names,key="lab_out")
                pool_names=[x for x in build_df['Name'].astype(str).tolist() if x not in roster_names]
                in_player=st.selectbox("With",pool_names,key="lab_in")
                if out_player and in_player:
                    po=build_df[build_df['Name'].astype(str).eq(out_player)].iloc[0]; pi=build_df[build_df['Name'].astype(str).eq(in_player)].iloc[0]
                    dproj=float(pi['DFS Lab Proj'])-float(po['DFS Lab Proj']); dsal=int(pi['FlexSalary'])-int(po['FlexSalary']); legal=int(lr['Salary'])+dsal<=50000
                    st.write(f"**Direct swap:** projection {dproj:+.2f} · salary {dsal:+,} · {'salary-cap legal' if legal else 'over the salary cap — a second change would be required'}.")
'''
new2='''                st.markdown("##### Manual swap check (optional)")
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
'''
if old2 not in s: raise SystemExit('manual swap block not found')
s=s.replace(old2,new2,1)
p.write_text(s)
