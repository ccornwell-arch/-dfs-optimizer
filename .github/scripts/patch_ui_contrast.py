from pathlib import Path

p = Path('streamlit_app.py')
s = p.read_text()

old = '[data-testid=\\"stExpander\\"] label,[data-testid=\\"stExpander\\"] p{color:#dbe7f5!important}.stRadio label{color:#dbe7f5!important}'
new = '''[data-testid=\\"stExpander\\"] label,
[data-testid=\\"stExpander\\"] label p,
[data-testid=\\"stExpander\\"] [data-testid=\\"stWidgetLabel\\"] p,
[data-testid=\\"stExpander\\"] p{
  color:#253247!important;
  -webkit-text-fill-color:#253247!important;
  opacity:1!important;
}
.stRadio label,.stRadio label p{
  color:#253247!important;
  -webkit-text-fill-color:#253247!important;
  opacity:1!important;
}'''
if old not in s:
    raise SystemExit('target contrast rule not found')
s = s.replace(old, new, 1)

anchor = 'st.markdown(\'\'\'<div class="apple-hero">'
fix = '''st.markdown("""<style>
/* Final iPad contrast pass */
html, body, [data-testid="stAppViewContainer"] {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif !important;
}
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,
[data-testid="stAppViewContainer"] label p,
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p {
  color:#253247 !important;
  -webkit-text-fill-color:#253247 !important;
  opacity:1 !important;
  font-weight:600 !important;
}
[data-testid="stSegmentedControl"] label,
[data-testid="stSegmentedControl"] button,
[data-testid="stSegmentedControl"] [role="radio"] {
  background:#243041 !important;
  border-color:#506078 !important;
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
  opacity:1 !important;
}
[data-testid="stSegmentedControl"] label *,
[data-testid="stSegmentedControl"] button *,
[data-testid="stSegmentedControl"] [role="radio"] * {
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
  opacity:1 !important;
  font-weight:700 !important;
}
[data-testid="stSegmentedControl"] label:has(input:checked),
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[aria-pressed="true"],
[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] {
  background:#2563eb !important;
  border-color:#2563eb !important;
}
[data-baseweb="button-group"] button,
[data-baseweb="button-group"] button * {
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
  opacity:1 !important;
}
</style>""", unsafe_allow_html=True)

'''
if anchor not in s:
    raise SystemExit('hero anchor not found')
s = s.replace(anchor, fix + anchor, 1)

p.write_text(s)
