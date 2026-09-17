from pathlib import Path

p = Path('streamlit_app.py')
s = p.read_text()

# 1) Make QB Captain pass-catcher setting exact, not merely a minimum.
old = '''            coeff[vidx(cpt,0)] = coeff.get(vidx(cpt,0),0)-float(cpt_qb_passcatchers)\n            _add_constraint(rows,lows,highs,coeff,0,np.inf)'''
new = '''            coeff[vidx(cpt,0)] = coeff.get(vidx(cpt,0),0)-float(cpt_qb_passcatchers)\n            _add_constraint(rows,lows,highs,coeff,0,np.inf)\n\n            # Enforce the selected number exactly when this QB is Captain.\n            # If the QB is not Captain, allow the normal FLEX player mix.\n            max_flex=float(s-1)\n            upper={}\n            for i in pcs:\n                for j in range(1,s): upper[vidx(i,j)] = upper.get(vidx(i,j),0)+1\n            upper[vidx(cpt,0)] = upper.get(vidx(cpt,0),0) + (max_flex-float(cpt_qb_passcatchers))\n            _add_constraint(rows,lows,highs,upper,-np.inf,max_flex)'''
if old not in s:
    raise SystemExit('QB captain constraint target not found')
s = s.replace(old, new, 1)

# 2) Replace unreadable dark Agent status expander with a normal warning panel.
old = '''                if st.session_state.get('dfs_agent_error'):\n                    with st.expander('Agent status',expanded=False): st.caption(f\"DFS LAB Agent API fallback is active. Error: {st.session_state.get('dfs_agent_error','unknown')}\")'''
new = '''                if st.session_state.get('dfs_agent_error'):\n                    st.warning(f\"Agent status: API fallback is active. {st.session_state.get('dfs_agent_error','unknown')}\")'''
if old not in s:
    raise SystemExit('Agent status target not found')
s = s.replace(old, new, 1)

# 3) Put a reply box directly under the latest Agent answer so the conversation can continue naturally.
anchor = '''                if st.session_state.get('dfs_agent_error'):\n                    st.warning(f\"Agent status: API fallback is active. {st.session_state.get('dfs_agent_error','unknown')}\")\n                st.write(f\"DFS Lab selected **{lr['Captain']} at Captain** while preserving the {lr['Construction']} game construction because this combination ranked strongly under the current projection, correlation, salary and contest-risk settings. {lr.get('Strategy Notes','')}\")'''
replacement = '''                if st.session_state.get('dfs_agent_error'):\n                    st.warning(f\"Agent status: API fallback is active. {st.session_state.get('dfs_agent_error','unknown')}\")\n\n                if st.session_state['dfs_lab_chat']:\n                    with st.form('dfs_lab_followup_form', clear_on_submit=True):\n                        follow_q=st.text_input('Reply to DFS LAB', placeholder='Reply to DFS LAB…', label_visibility='collapsed')\n                        follow_submit=st.form_submit_button('SEND REPLY  ↗', type='primary', use_container_width=True)\n                    if follow_submit and follow_q.strip():\n                        st.session_state['dfs_agent_pending']=follow_q.strip()\n                        st.rerun()\n\n                st.write(f\"DFS Lab selected **{lr['Captain']} at Captain** while preserving the {lr['Construction']} game construction because this combination ranked strongly under the current projection, correlation, salary and contest-risk settings. {lr.get('Strategy Notes','')}\")'''
if anchor not in s:
    raise SystemExit('Follow-up insertion anchor not found')
s = s.replace(anchor, replacement, 1)

p.write_text(s)

# workflow trigger 2026-09-17 08:52
