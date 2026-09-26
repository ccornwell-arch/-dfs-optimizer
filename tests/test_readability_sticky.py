"""Regression tests for sticky tabs + readability hardening.

Covers the fixes for "tabs take so long to get back to" and unreadable
bars/texts:
  1. RCC_DARK_CSS carries an authoritative sticky rule for the primary tab
     bar (position:sticky, top:0, high z-index, solid #101418 background,
     no backdrop-filter which breaks stickiness on iPad Safari), while the
     Showdown results-hub tabs stay explicitly non-sticky.
  2. Expander / select / download-button / lineup-count-pill rules beat the
     legacy light-theme layers on specificity (via #root) so the dark
     command-center look holds regardless of injection order.
  3. The Explorer player grid and swap-suggestion grid render as accessible
     HTML tables with bright, larger headers (st.dataframe headers are
     canvas-drawn and cannot be fixed with CSS).
  4. A headless AppTest render proves the CSS is emitted, the main tabs
     render, and the download button exists.

Run from repo root:  python3 tests/test_readability_sticky.py
"""
import io
import sys

sys.path.insert(0, ".")

import pandas as pd

from dfs_lab import styles
from dfs_lab.ui.results import _player_grid_html, _swap_grid_html, _esc

CSS = styles.RCC_DARK_CSS


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def test_sticky_tab_css():
    _check("pinned: tablist position fixed", "position:fixed!important" in CSS)
    _check("pinned: top 0", "top:0!important" in CSS)
    _check("pinned: full viewport width", "left:0!important" in CSS and "right:0!important" in CSS)
    _check("pinned: high z-index", "z-index:999999!important" in CSS)
    _check("pinned: solid dark background", "background:#101418!important" in CSS)
    _check("pinned: no backdrop-filter (iPad Safari)", "backdrop-filter:none!important" in CSS)
    _check("pinned: flow space reserved for tab content", "padding-top:3.4rem!important" in CSS)
    _check("pinned: scoped off the showdown results hub",
           ":not(.st-key-sd_results_hub)" in CSS)
    _check("pinned: showdown hub pinned to relative",
           ".st-key-sd_results_hub [role=\"tablist\"]" in CSS
           and "position:relative!important" in CSS)


def test_definitive_dark_css():
    _check("expander: #root-boosted summary rule",
           '#root [data-testid="stAppViewContainer"] [data-testid="stExpander"] summary' in CSS)
    _check("expander: dark summary background",
           "background:#1a2430!important" in CSS)
    _check("select: #root-boosted dark rule",
           '#root [data-testid="stSelectbox"] [data-baseweb="select"]>div' in CSS)
    _check("download: #root-boosted rule",
           '#root [data-testid="stDownloadButton"]>button' in CSS)
    _check("download: pure-white text",
           '#root [data-testid="stDownloadButton"]>button,\n'
           'html body #root [data-testid="stDownloadButton"]>button *' in CSS)
    _check("pill: span kept bright on dark pill",
           "html body #root .lineup-count-readout span" in CSS)
    _check("grid: bright header style", "table.rcc-grid th" in CSS
           and "color:#e6eef8" in CSS)


def _sample_pt():
    return pd.DataFrame([
        {"Slot": "QB", "Player": "KC Quarterback", "Pos": "QB", "Team": "KC",
         "Salary": 7200, "Proj": 24.0, "Own %": 14.0, "Value": 3.33, "Leverage": "High"},
        {"Slot": "RB1", "Player": "MIA Runner1 <b>", "Pos": "RB", "Team": "MIA",
         "Salary": 5900, "Proj": 12.0, "Own %": 9.0, "Value": 2.03, "Leverage": "Low"},
    ])


def _sample_alts():
    return [
        {"Player": "KC Wide1", "Team": "KC", "Salary": 6400, "Proj": 16.0,
         "Own %": 12.0, "dProj": -1.15, "dOwn": 1.2, "dSalary": -300,
         "Tradeoff": "-1.2 proj · more chalk · saves $300"},
        {"Player": "MIA Wide1", "Team": "MIA", "Salary": 5600, "Proj": 11.0,
         "Own %": 8.0, "dProj": -3.19, "dOwn": -1.3, "dSalary": -200,
         "Tradeoff": "-3.2 proj · less chalk · saves $200"},
    ]


def test_player_grid_html():
    html = _player_grid_html(_sample_pt())
    _check("player grid: table markup", "<table class='rcc-grid'>" in html)
    _check("player grid: bright headers",
           "<th>Player</th>" in html and "<th class='num'>Salary</th>" in html)
    _check("player grid: salary formatted", "$7,200" in html)
    _check("player grid: names escaped", "&lt;b&gt;" in html and "<b>>" not in html.replace("&lt;b&gt;", ""))
    _check("player grid: no canvas dataframe", "data-testid" not in html)


def test_player_grid_html_empty():
    html = _player_grid_html(_sample_pt().iloc[0:0])
    _check("player grid: empty state", "No players match." in html)


def test_swap_grid_html():
    html = _swap_grid_html(_sample_alts())
    _check("swap grid: table markup", "<table class='rcc-grid'>" in html)
    _check("swap grid: delta headers",
           "<th class='num'>Δ Proj</th>" in html and "<th>Tradeoff</th>" in html)
    _check("swap grid: signed deltas", "-1.15" in html and "-300" in html)
    _check("swap grid: tradeoff text", "more chalk" in html)


def test_headless_render():
    """Headless AppTest: CSS emitted, main tabs render, download exists,
    and the real Lineup Explorer (HTML grids + compact callout) renders
    without exceptions."""
    from streamlit.testing.v1 import AppTest

    script = (
        "import sys\n"
        "sys.path.insert(0, '/home/hatch/workspace/dfs-lab')\n"
        "sys.path.insert(0, '/home/hatch/workspace/dfs-lab/tests')\n"
        "import streamlit as st\n"
        "from dfs_lab import styles\n"
        "from dfs_lab.ui.results import _equity, render_explorer\n"
        "from test_results_command_center import build_pipeline\n"
        "st.markdown(styles.MAIN_TABS_CSS, unsafe_allow_html=True)\n"
        "st.markdown(styles.RCC_DARK_CSS, unsafe_allow_html=True)\n"
        "tabs = st.tabs(['Slate Intel', 'Build', 'Players', 'Rules', 'Lineups', 'Exposure'])\n"
        "with tabs[4]:\n"
        "    st.markdown(\"<div class='lineup-count-readout'><b>20</b><span> lineups selected</span></div>\",\n"
        "                unsafe_allow_html=True)\n"
        "    df, bb_worthy, sim, res, eq = build_pipeline()\n"
        "    disp = _equity(res, sim['worlds'], df)\n"
        "    render_explorer(disp, df)\n"
        "    st.download_button('Download lineup analysis CSV', disp.to_csv(index=False),\n"
        "                       'classic_lineups_v5.csv', 'text/csv', key='rcc_download')\n"
    )
    with open("/tmp/_sticky_apptest.py", "w") as f:
        f.write(script)
    at = AppTest.from_file("/tmp/_sticky_apptest.py")
    at.run(timeout=300)
    _check("apptest: no exception", len(at.exception) == 0)
    _check("apptest: six main tabs rendered", len(at.tabs) == 6)
    css_blobs = " ".join(m.value for m in at.markdown)
    _check("apptest: pinned CSS emitted", "position:fixed!important" in css_blobs)
    _check("apptest: pinned top:0 emitted", "top:0!important" in css_blobs)
    _check("apptest: hub exclusion emitted", ":not(.st-key-sd_results_hub)" in css_blobs)
    _check("apptest: dark expander rule emitted", "background:#1a2430!important" in css_blobs)
    _check("apptest: explorer player grid rendered", "rcc-grid" in css_blobs)
    _check("apptest: swap grid rendered", css_blobs.count("rcc-grid") >= 2)
    _check("apptest: compact current-player callout rendered", "rcc-cur-player" in css_blobs)
    _check("apptest: no giant metric widget in explorer", len(at.metric) == 0)
    _check("apptest: explorer selectboxes rendered", len(at.selectbox) >= 2)
    _check("apptest: download button present", len(at.download_button) == 1)


if __name__ == "__main__":
    test_sticky_tab_css()
    test_definitive_dark_css()
    test_player_grid_html()
    test_player_grid_html_empty()
    test_swap_grid_html()
    test_headless_render()
    print("\nALL READABILITY/STICKY TESTS PASSED")
