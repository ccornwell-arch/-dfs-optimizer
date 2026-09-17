from pathlib import Path

p = Path('streamlit_app.py')
s = p.read_text()

marker = '/* DFS LAB iPad surface/background pass */'
if marker not in s:
    anchor = "st.markdown(\"\"\"<style>\n/* Final iPad contrast pass */"
    if anchor not in s:
        raise SystemExit('final contrast style anchor not found')

    surface_fix = '''st.markdown("""<style>
/* DFS LAB iPad surface/background pass */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #dbe4ef 0%, #e7edf5 42%, #dfe8f2 100%) !important;
}
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] .main {
  background: transparent !important;
}
[data-testid="stAppViewContainer"] .block-container {
  background: transparent !important;
}
[data-testid="stExpander"],
[data-testid="stVerticalBlockBorderWrapper"] > div,
[data-testid="stForm"] {
  background: rgba(248,250,252,0.94) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stNumberInput"] > div > div,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  background: #eef3f9 !important;
}
</style>""", unsafe_allow_html=True)

'''
    s = s.replace(anchor, surface_fix + anchor, 1)

p.write_text(s)
