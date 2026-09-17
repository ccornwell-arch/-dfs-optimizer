from pathlib import Path

p = Path('streamlit_app.py')
s = p.read_text(encoding='utf-8')

old = '''    result=milp(c=c, integrality=integrality, bounds=Bounds(lb,ub),\n                constraints=LinearConstraint(A.tocsr(),np.array(lows),np.array(highs)),\n                options={"time_limit":8.0})\n    if not result.success or result.x is None: return None\n'''
new = '''    # Let HiGHS finish the solve instead of abandoning valid/complex Showdown builds\n    # after an arbitrary per-lineup time limit. The user explicitly asked DFS LAB to\n    # finish the requested portfolio rather than stop because a solve took >8 seconds.\n    result=milp(c=c, integrality=integrality, bounds=Bounds(lb,ub),\n                constraints=LinearConstraint(A.tocsr(),np.array(lows),np.array(highs)))\n    if result.x is None: return None\n'''

if old not in s:
    raise SystemExit('Target Showdown MILP timeout block not found; no changes made.')

s = s.replace(old, new, 1)
p.write_text(s, encoding='utf-8')
print('Removed Showdown per-solve timeout.')
