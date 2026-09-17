from pathlib import Path

p = Path('streamlit_app.py')
s = p.read_text()

anchor = '''_has_cached_slate=bool(st.session_state.get("dfs_lab_dk_bytes"))\nwith st.expander("✓ SLATE LOADED · Change files" if _has_cached_slate else "＋ LOAD SLATE FILES", expanded=not _has_cached_slate):'''
insert = '''@st.cache_resource\ndef _dfs_lab_slate_cache():\n    return {}\n\n# Keep each browser's last successfully uploaded slate alive across ordinary page refreshes.\n# The cache key is stored in the URL so a new Streamlit session can recover the same files.\nif "slate_session" not in st.query_params:\n    import secrets\n    st.query_params["slate_session"] = secrets.token_urlsafe(10)\n_slate_session = str(st.query_params.get("slate_session", "default"))\n_slate_cache = _dfs_lab_slate_cache()\n_cached = _slate_cache.get(_slate_session, {})\nif not st.session_state.get("dfs_lab_dk_bytes") and _cached.get("dk_bytes"):\n    st.session_state["dfs_lab_dk_bytes"] = _cached["dk_bytes"]\n    st.session_state["dfs_lab_dk_name"] = _cached.get("dk_name", "DKSalaries.csv")\nif not st.session_state.get("dfs_lab_ss_bytes") and _cached.get("ss_bytes"):\n    st.session_state["dfs_lab_ss_bytes"] = _cached["ss_bytes"]\n    st.session_state["dfs_lab_ss_name"] = _cached.get("ss_name", "SaberSim.csv")\n\n_has_cached_slate=bool(st.session_state.get("dfs_lab_dk_bytes"))\nwith st.expander("✓ SLATE LOADED · Change files" if _has_cached_slate else "＋ LOAD SLATE FILES", expanded=not _has_cached_slate):'''
if anchor not in s:
    raise SystemExit('slate uploader anchor not found')
s = s.replace(anchor, insert, 1)

old = '''if dk_file is not None:\n    st.session_state["dfs_lab_dk_bytes"]=dk_file.getvalue(); st.session_state["dfs_lab_dk_name"]=getattr(dk_file,"name","DKSalaries.csv")\nelif st.session_state.get("dfs_lab_dk_bytes"):\n    dk_file=io.BytesIO(st.session_state["dfs_lab_dk_bytes"]); dk_file.name=st.session_state.get("dfs_lab_dk_name","DKSalaries.csv")\nif ss_file is not None:\n    st.session_state["dfs_lab_ss_bytes"]=ss_file.getvalue(); st.session_state["dfs_lab_ss_name"]=getattr(ss_file,"name","SaberSim.csv")\nelif st.session_state.get("dfs_lab_ss_bytes"):\n    ss_file=io.BytesIO(st.session_state["dfs_lab_ss_bytes"]); ss_file.name=st.session_state.get("dfs_lab_ss_name","SaberSim.csv")'''
new = '''if dk_file is not None:\n    st.session_state["dfs_lab_dk_bytes"]=dk_file.getvalue(); st.session_state["dfs_lab_dk_name"]=getattr(dk_file,"name","DKSalaries.csv")\n    _slate_cache[_slate_session] = {**_slate_cache.get(_slate_session, {}), "dk_bytes": st.session_state["dfs_lab_dk_bytes"], "dk_name": st.session_state["dfs_lab_dk_name"]}\nelif st.session_state.get("dfs_lab_dk_bytes"):\n    dk_file=io.BytesIO(st.session_state["dfs_lab_dk_bytes"]); dk_file.name=st.session_state.get("dfs_lab_dk_name","DKSalaries.csv")\nif ss_file is not None:\n    st.session_state["dfs_lab_ss_bytes"]=ss_file.getvalue(); st.session_state["dfs_lab_ss_name"]=getattr(ss_file,"name","SaberSim.csv")\n    _slate_cache[_slate_session] = {**_slate_cache.get(_slate_session, {}), "ss_bytes": st.session_state["dfs_lab_ss_bytes"], "ss_name": st.session_state["dfs_lab_ss_name"]}\nelif st.session_state.get("dfs_lab_ss_bytes"):\n    ss_file=io.BytesIO(st.session_state["dfs_lab_ss_bytes"]); ss_file.name=st.session_state.get("dfs_lab_ss_name","SaberSim.csv")'''
if old not in s:
    raise SystemExit('slate byte-cache block not found')
s = s.replace(old, new, 1)

p.write_text(s)
