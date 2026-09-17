from pathlib import Path
import re

p = Path('streamlit_app.py')
s = p.read_text()

# UI: make it explicit whether the number is a minimum or a maximum.
old_ui = '''            with a:cpt_qb_pc=st.selectbox("When QB is CPT · pass catchers",[0,1,2,3],index=2,help="Number of same-team WR/TE pass catchers to pair with a QB Captain.")'''
new_ui = '''            with a:
                qb_cpt_pc_rule = st.selectbox(
                    "When QB is CPT · pass catchers",
                    ["No rule", "Minimum 1", "Minimum 2", "No more than 1", "No more than 2"],
                    index=1,
                    help="Minimum means at least this many same-team WR/TE pass catchers. No more than sets a maximum."
                )
                cpt_qb_pc = {"No rule":0, "Minimum 1":1, "Minimum 2":2, "No more than 1":-1, "No more than 2":-2}[qb_cpt_pc_rule]'''
if old_ui in s:
    s = s.replace(old_ui, new_ui, 1)
elif 'qb_cpt_pc_rule = st.selectbox(' not in s:
    raise SystemExit('QB captain UI target not found')

# Solver: positive values mean minimum; negative values mean maximum.
# Replace either the original minimum-only block or the later exact-rule block.
pattern = re.compile(r'''        # QB captain -> require same-team pass catchers\.\n        if r\["is_QB"\] and cpt_qb_passcatchers > 0:\n            pcs=df\.index\[df\["ActiveForBuild"\] & df\["Team"\]\.eq\(r\["Team"\]\) & df\["is_passcatcher"\]\]\.tolist\(\)\n            coeff=\{\}\n            for i in pcs:\n                for j in range\(1,s\): coeff\[vidx\(i,j\)\] = coeff\.get\(vidx\(i,j\),0\)\+1\n            coeff\[vidx\(cpt,0\)\] = coeff\.get\(vidx\(cpt,0\),0\)-float\(cpt_qb_passcatchers\)\n            _add_constraint\(rows,lows,highs,coeff,0,np\.inf\)(?:\n\n            # Enforce the selected number exactly when this QB is Captain\.[\s\S]*?            _add_constraint\(rows,lows,highs,upper,-np\.inf,max_flex\))?''')

replacement = '''        # QB captain -> optional minimum/maximum same-team pass-catcher rule.\n        if r["is_QB"] and cpt_qb_passcatchers != 0:\n            pcs=df.index[df["ActiveForBuild"] & df["Team"].eq(r["Team"]) & df["is_passcatcher"]].tolist()\n            n=abs(int(cpt_qb_passcatchers))\n            coeff={}\n            for i in pcs:\n                for j in range(1,s): coeff[vidx(i,j)] = coeff.get(vidx(i,j),0)+1\n            if cpt_qb_passcatchers > 0:\n                # Minimum N: when this QB is Captain, require at least N same-team pass catchers.\n                coeff[vidx(cpt,0)] = coeff.get(vidx(cpt,0),0)-float(n)\n                _add_constraint(rows,lows,highs,coeff,0,np.inf)\n            else:\n                # No more than N: when this QB is Captain, cap same-team pass catchers at N.\n                max_flex=float(s-1)\n                coeff[vidx(cpt,0)] = coeff.get(vidx(cpt,0),0)+(max_flex-float(n))\n                _add_constraint(rows,lows,highs,coeff,-np.inf,max_flex)'''

s2, count = pattern.subn(replacement, s, count=1)
if count == 0:
    if 'optional minimum/maximum same-team pass-catcher rule' not in s:
        raise SystemExit('QB captain solver target not found')
else:
    s = s2

p.write_text(s)
