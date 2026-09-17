from pathlib import Path

p = Path('streamlit_app.py')
s = p.read_text()

old = '''    result=milp(c=c, integrality=integrality, bounds=Bounds(lb,ub),\n                constraints=LinearConstraint(A.tocsr(),np.array(lows),np.array(highs)),\n                options={"time_limit":8.0})\n    if not result.success or result.x is None: return None'''
new = '''    result=milp(c=c, integrality=integrality, bounds=Bounds(lb,ub),\n                constraints=LinearConstraint(A.tocsr(),np.array(lows),np.array(highs)),\n                options={"time_limit":2.0, "mip_rel_gap":0.03})\n    # HiGHS may hit the time limit after finding a perfectly usable feasible\n    # incumbent. scipy then reports success=False even though result.x is valid.\n    # Do not throw that lineup away just because optimality was not proven.\n    if result.x is None:\n        return None\n    _x=np.asarray(result.x,dtype=float)\n    if np.max(np.abs(_x-np.round(_x))) > 1e-4:\n        return None'''
if old not in s:
    if '"time_limit":2.0, "mip_rel_gap":0.03' not in s:
        raise SystemExit('solver target not found')
else:
    s = s.replace(old, new, 1)

css = '''\n\nst.markdown("""<style>\n/* Keep primary action text readable on the blue build button, including iPad Safari. */\n.stButton > button[kind="primary"],\n.stButton > button[kind="primary"] *,\n.stFormSubmitButton > button[kind="primary"],\n.stFormSubmitButton > button[kind="primary"] * {\n  color:#ffffff !important;\n  -webkit-text-fill-color:#ffffff !important;\n  opacity:1 !important;\n  font-weight:800 !important;\n}\n</style>""", unsafe_allow_html=True)\n'''
if 'Keep primary action text readable on the blue build button' not in s:
    s += css

p.write_text(s)
