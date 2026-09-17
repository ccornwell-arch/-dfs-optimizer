from pathlib import Path

p = Path('streamlit_app.py')
s = p.read_text()

old = '''        if attempt%10==0: prog.progress(min(1.0,len(rows)/max(1,count)),text=f"Built {len(rows)} / {count}")'''
new = '''        # Update after every accepted lineup so the UI never appears frozen.
        prog.progress(min(1.0,len(rows)/max(1,count)), text=f"Built {len(rows)} / {count}")'''
if old not in s:
    raise SystemExit('progress update target not found')
s = s.replace(old, new, 1)

# Append a final high-contrast primary-button rule so the build button text remains readable.
anchor = "st.markdown('''<div class=\"apple-hero\">"
css = '''st.markdown("""<style>
/* Final primary-button contrast */
.stButton > button[kind="primary"],
.stButton > button[kind="primary"] * {
  background: #1559c7 !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  border-color: #1559c7 !important;
  font-weight: 800 !important;
}
.stButton > button[kind="primary"]:hover {
  background: #0f4cae !important;
}
</style>""", unsafe_allow_html=True)

'''
if 'Final primary-button contrast' not in s:
    if anchor not in s:
        raise SystemExit('hero anchor not found')
    s = s.replace(anchor, css + anchor, 1)

p.write_text(s)
