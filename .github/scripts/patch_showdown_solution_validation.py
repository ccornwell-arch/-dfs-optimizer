from pathlib import Path

p = Path('streamlit_app.py')
s = p.read_text()
old = '''    if result.x is None: return None
    chosen=[]
    for j,slot in enumerate(SHOWDOWN_SLOTS):
        vals=[(result.x[vidx(i,j)],i) for i in range(n)]
        _,i=max(vals); chosen.append((slot,int(i)))
    return chosen
'''
new = '''    # Only accept a completed feasible MILP solution. A non-success result can still
    # expose a partial vector; decoding that vector can create impossible lineups
    # (for example the same player appearing in multiple slots).
    if not result.success or result.x is None:
        return None
    chosen=[]
    for j,slot in enumerate(SHOWDOWN_SLOTS):
        vals=[(result.x[vidx(i,j)],i) for i in range(n)]
        _,i=max(vals); chosen.append((slot,int(i)))

    # Final safety validation before a lineup is ever shown to the user.
    chosen_ids=[str(df.loc[i,"ID"]) for _,i in chosen]
    if len(chosen_ids) != len(SHOWDOWN_SLOTS) or len(set(chosen_ids)) != len(SHOWDOWN_SLOTS):
        return None
    cpt_i=[i for slot,i in chosen if slot=="CPT"][0]
    flex_i=[i for slot,i in chosen if slot!="CPT"]
    salary=int(df.loc[cpt_i,"CaptainSalary"])+int(df.loc[flex_i,"FlexSalary"].sum())
    if salary < int(min_salary) or salary > int(max_salary):
        return None
    return chosen
'''
if old not in s:
    raise SystemExit('target block not found')
p.write_text(s.replace(old,new,1))
