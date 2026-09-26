"""Lineups-first restructure tests for the Classic post-build Results view.

Verifies the new stacked order (slim banner -> lineup cards with visible
rosters + avatars -> game worlds -> Lab+Agent -> explorer -> data view last),
the Showdown-style avatar rendering, the single authoritative sticky tab rule,
and that the pre-build empty state is unchanged.

Run from repo root:  python3 tests/test_lineups_first.py
"""
import re
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

import pandas as pd

from dfs_lab.ui.results import (
    _headshot_map, _avatar_html, slim_banner_html, _lineup_card_html,
    _equity, render_results_command_center,
)
from test_results_command_center import build_pipeline


def _check(name, cond, extra=""):
    assert cond, f"FAIL {name} {extra}"
    print(f"PASS {name}" + (f" {extra}" if extra else ""))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_headshot_map():
    df = pd.DataFrame({
        "Name": ["A One", "B Two", "C Three", "D Four"],
        "Headshot URL": ["https://x/y.png", "", "nan", "None"],
    })
    m = _headshot_map(df)
    _check("headshot_map: only real URLs", m == {"A One": "https://x/y.png"}, str(m))
    _check("headshot_map: missing column", _headshot_map(pd.DataFrame({"Name": ["A"]})) == {})
    _check("headshot_map: None df", _headshot_map(None) == {})


def test_avatar_html():
    a = _avatar_html("Patrick Mahomes", "https://x/mahomes.png")
    _check("avatar: img rendered", "<img" in a and "https://x/mahomes.png" in a)
    _check("avatar: photo class", "player-avatar" in a)
    b = _avatar_html("Patrick Mahomes", "")
    _check("avatar: initials fallback", "<img" not in b and "PM" in b, b)
    c = _avatar_html("X", None)
    _check("avatar: none input -> initials", "<img" not in c and "X" in c)
    d = _avatar_html("A B", "nan")
    _check("avatar: 'nan' url -> initials", "<img" not in d and "AB" in d)
    e = _avatar_html("<script>alert(1)</script>", "")
    _check("avatar: name escaped", "<script>" not in e)


def test_slim_banner():
    _check("banner: empty on no data", slim_banner_html(pd.DataFrame()) == "")
    df, _, sim, res, eq = build_pipeline()
    disp = _equity(res, sim["worlds"], df)
    out = slim_banner_html(disp)
    _check("banner: mentions build", "lineups built" in out, out[:80])
    _check("banner: best rank", f"#{int(disp.iloc[0]['Rank'])}" in out)
    _check("banner: grade", str(disp.iloc[0]["Rating"]) in out)
    _check("banner: top proj", f"{float(disp['Projection'].max()):.1f}" in out)
    print("banner sample:", re.sub(r"<[^>]+>", "", out)[:160])


def test_lineup_card(df=None, res=None, eq=None, sim=None):
    if df is None:
        df, _, sim, res, eq = build_pipeline()
    disp = _equity(res, sim["worlds"], df)
    row = disp.iloc[0]
    from dfs_lab.ui.results import player_table_for_lineup
    pt = player_table_for_lineup(row, df)
    hero_player = str(pt.iloc[0]["Player"])
    shots = {hero_player: "https://x/hero.png"}
    html_out = _lineup_card_html(row, df, shots)
    _check("card: rank badge", f"#{int(row['Rank'])}" in html_out)
    _check("card: grade", str(row["Rating"]) in html_out)
    _check("card: story", "rcc-story" in html_out)
    _check("card: roster table", "rcc-roster" in html_out)
    _check("card: 9 roster rows", html_out.count("<tr><td class='av'>") == 9,
           f"found {html_out.count(chr(39)+'av'+chr(39))}")
    _check("card: photo avatar", "https://x/hero.png" in html_out)
    _check("card: initials fallback avatar", "avatar-fallback" in html_out)
    _check("card: chips", "rcc-chips" in html_out)
    _check("card: never raises on junk", _lineup_card_html({}, df, {}) == "" or True)


# ---------------------------------------------------------------------------
# CSS: exactly one pinned tab rule
# ---------------------------------------------------------------------------

def test_single_sticky_tab_rule():
    src = open("dfs_lab/styles.py").read()
    # Split into rule blocks; find ones that pin tab bars.
    blocks = re.findall(r"[^{}]*\{[^{}]*\}", src)
    pinned_tab = [b for b in blocks
                  if ("position:fixed" in b or "position:sticky" in b)
                  and ("tablist" in b or "tab-list" in b)]
    _check("css: one pinned tab rule", len(pinned_tab) == 1, f"found {len(pinned_tab)}")
    rule = pinned_tab[0]
    _check("css: pinned top:0", "top:0!important" in rule)
    _check("css: solid dark background", "#101418!important" in rule)
    _check("css: no backdrop blur", "backdrop-filter:blur" not in rule)
    _check("css: hub exclusion kept", ":not(.st-key-sd_results_hub)" in rule)


def test_pool_cards_hidden_postbuild():
    src = open("dfs_lab/ui/main.py").read()
    _check("main: pool cards gated on built state",
           'st.session_state.get("classic_result_v4")' in src and "_built_now" in src)
    _check("main: pre-build empty state unchanged",
           'st.info("Generate lineups from Build.")' in src)


# ---------------------------------------------------------------------------
# Headless AppTest: full stacked render order
# ---------------------------------------------------------------------------

def test_headless_stacked_order():
    from streamlit.testing.v1 import AppTest

    script = (
        "import sys\n"
        "sys.path.insert(0, '/home/hatch/workspace/dfs-lab')\n"
        "sys.path.insert(0, '/home/hatch/workspace/dfs-lab/tests')\n"
        "import streamlit as st\n"
        "from dfs_lab import styles\n"
        "from dfs_lab.ui.results import _equity, render_results_command_center\n"
        "from test_results_command_center import build_pipeline\n"
        "st.markdown(styles.MAIN_TABS_CSS, unsafe_allow_html=True)\n"
        "st.markdown(styles.RCC_DARK_CSS, unsafe_allow_html=True)\n"
        "tabs = st.tabs(['Slate Intel', 'Build', 'Players', 'Rules', 'Lineups', 'Exposure'])\n"
        "with tabs[4]:\n"
        "    df, bb_worthy, sim, res, eq = build_pipeline()\n"
        "    df = df.copy()\n"
        "    df.loc[df['Name'] == 'KC Quarterback', 'Headshot URL'] = 'https://x/kc_qb.png'\n"
        "    df.loc[df['Name'] == 'BUF Quarterback', 'Headshot URL'] = 'https://x/buf_qb.png'\n"
        "    disp = _equity(res, sim['worlds'], df)\n"
        "    render_results_command_center(res, df, sim['game_table'], sim['worlds'], {})\n"
    )
    with open("/tmp/_lineups_first_apptest.py", "w") as f:
        f.write(script)
    at = AppTest.from_file("/tmp/_lineups_first_apptest.py")
    at.run(timeout=300)
    _check("apptest: no exception", len(at.exception) == 0,
           str(at.exception[0])[:300] if at.exception else "")

    blobs = [m.value for m in at.markdown]
    full = "\n".join(blobs)
    _check("apptest: slim banner rendered", "rcc-banner" in full)
    _check("apptest: banner has no hero", "<div class='rcc-hero'>" not in full)
    card_count = full.count("rcc-card-head")
    _check("apptest: one card per lineup", card_count >= 8, f"cards={card_count}")
    _check("apptest: roster rows visible in cards", full.count("<tr><td class='av'>") >= 8 * 9)
    _check("apptest: photo avatar rendered", "https://x/buf_qb.png" in full)
    _check("apptest: initials fallback rendered", "avatar-fallback" in full)
    _check("apptest: no sub-tab segmented control", len(at.segmented_control) == 0)
    _check("apptest: sort control present", len(at.selectbox) >= 1)

    i_banner = full.find("rcc-banner")
    i_card = full.find("rcc-card")
    i_worlds = full.find("rcc-world-board")
    i_agent = full.find("Lab + Agent")
    i_explorer = full.find("Lineup Explorer")
    _check("apptest: order banner < cards", 0 <= i_banner < i_card)
    _check("apptest: order cards < worlds", i_card < i_worlds)
    _check("apptest: order worlds < agent", i_worlds < i_agent, f"{i_worlds} {i_agent}")
    _check("apptest: order agent < explorer", i_agent < i_explorer)

    exp_labels = [e.label for e in at.expander]
    _check("apptest: no lineup expanders", not any("Lineup #" in (l or "") for l in exp_labels),
           str(exp_labels))
    _check("apptest: data view expander exists",
           any("Data view" in (l or "") for l in exp_labels))
    _check("apptest: data view is the last expander",
           "Data view" in (exp_labels[-1] or "") if exp_labels else False,
           str(exp_labels))
    _check("apptest: download button present", len(at.download_button) == 1)


if __name__ == "__main__":
    test_headshot_map()
    test_avatar_html()
    test_slim_banner()
    test_lineup_card()
    test_single_sticky_tab_rule()
    test_pool_cards_hidden_postbuild()
    test_headless_stacked_order()
    print("\nALL LINEUPS-FIRST TESTS PASSED")
