from pathlib import Path

p = Path('streamlit_app.py')
s = p.read_text()
marker = '/* DFS LAB UNIFORM PAGE THEME */'

if marker in s:
    raise SystemExit('uniform theme already present')

css = r'''

# Final global theme lock: keep the same visual treatment before and after lineups are generated.
st.markdown("""<style>
/* DFS LAB UNIFORM PAGE THEME */
html, body,
.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background: #e6edf6 !important;
    color: #1f2937 !important;
}
[data-testid="stHeader"] {
    background: #e6edf6 !important;
}
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.block-container {
    background: transparent !important;
}

/* Keep primary content surfaces consistent at every app state. */
[data-testid="stExpander"],
[data-testid="stMetric"],
.section-card {
    background: #f4f7fb !important;
    border-color: #c8d3e2 !important;
}

/* Inputs stay softly tinted instead of flipping between white and dark. */
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stNumberInput"] > div > div,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #eef3f9 !important;
    color: #1f2937 !important;
    -webkit-text-fill-color: #1f2937 !important;
}

/* Normal body copy stays dark and readable on the unified background. */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p {
    color: #253247;
}

/* Preserve intentionally dark hero/brand areas. */
.hero, .apple-hero {
    color: #f8fafc !important;
}
.hero *, .apple-hero * {
    color: inherit;
}
</style>""", unsafe_allow_html=True)
'''

p.write_text(s + css)
