"""Runtime UI contrast shim for DFS LAB.

Python imports ``sitecustomize`` automatically during startup when it is on
sys.path. Streamlit Cloud runs the app from the repository root, so this lets
us apply a final, high-specificity accessibility layer after the app's older
CSS blocks without rewriting the optimizer logic.
"""

try:
    import streamlit as _st

    _orig_markdown = _st.markdown

    _CONTRAST_FIX = r"""
<style>
/* DFS LAB accessibility contrast override */

/* Main-workspace field labels/captions live on light surfaces. */
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] label p,
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p {
    color:#263244 !important;
    -webkit-text-fill-color:#263244 !important;
    opacity:1 !important;
    font-weight:650 !important;
}
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p {
    color:#536274 !important;
    -webkit-text-fill-color:#536274 !important;
    opacity:1 !important;
}

/* Segmented controls: every choice must remain readable on iPad Safari. */
[data-testid="stSegmentedControl"] label,
[data-testid="stSegmentedControl"] button,
[data-testid="stSegmentedControl"] [role="radio"] {
    background:#273142 !important;
    border-color:#55657a !important;
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
    box-shadow:inset 0 0 0 1px #93c5fd !important;
}

/* Some Streamlit versions render the two segments through BaseWeb. */
[data-baseweb="button-group"] button {
    background:#273142 !important;
    color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
    border-color:#55657a !important;
    opacity:1 !important;
}
[data-baseweb="button-group"] button * {
    color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
    opacity:1 !important;
}
[data-baseweb="button-group"] button[aria-pressed="true"],
[data-baseweb="button-group"] button[aria-checked="true"] {
    background:#2563eb !important;
    border-color:#2563eb !important;
}

/* Keep dark input/select contents high-contrast. */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stNumberInput"] > div > div {
    color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] svg,
[data-baseweb="input"] input,
[data-testid="stNumberInput"] input {
    color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
    opacity:1 !important;
}

/* Sidebar remains dark, so keep its labels light. */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color:#f8fafc !important;
    -webkit-text-fill-color:#f8fafc !important;
}
</style>
"""

    def _dfs_lab_markdown(body, *args, **kwargs):
        result = _orig_markdown(body, *args, **kwargs)
        # Re-assert the accessibility layer after every HTML/CSS markdown block,
        # so legacy style blocks cannot wash labels or segmented choices out.
        if kwargs.get("unsafe_allow_html", False):
            _orig_markdown(_CONTRAST_FIX, unsafe_allow_html=True)
        return result

    _st.markdown = _dfs_lab_markdown
except Exception:
    # Never prevent the optimizer from starting because of a presentation shim.
    pass
