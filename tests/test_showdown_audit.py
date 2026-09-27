"""Regression tests for the Showdown post-build hard-rule audit and the
app-owned sticky nav (segmented control + programmatic jumps).

Background: a user unchecked Puka Nacua as Captain-eligible and still got him
as Captain in a lineup; separately, the primary tabs disappeared behind the
results on iPad so there was no way back to Players / Exposure.

Covers:
  1. audit_showdown_portfolio() flags a Captain with CPT? unchecked, an
     excluded player in any slot, a CPT-locked player who is not the Captain,
     and a locked player missing from the lineup. A clean portfolio passes.
  2. Headless AppTest: a segmented-control nav plus a jump button reproduces
     the main.py mechanism — tapping the button sets the nav session state
     and the rerun renders the target section.

Run from repo root:  python3 tests/test_showdown_audit.py
"""
import sys

sys.path.insert(0, ".")

import pandas as pd

from dfs_lab.showdown import audit_showdown_portfolio


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def _lineup(rank, cpt_id, cpt_name, flex_ids, flex_names):
    row = {"Rank": rank, "CPT_ID": cpt_id, "CPT": cpt_name}
    for i, (pid, nm) in enumerate(zip(flex_ids, flex_names), start=1):
        row[f"FLEX{i}_ID"] = pid
        row[f"FLEX{i}"] = nm
    return row


_ID2NAME = {"1": "Puka Nacua", "2": "QB One", "3": "Flex Three", "4": "Flex Four",
            "5": "Flex Five", "6": "Flex Six", "7": "Flex Seven"}


def _clean_strategy():
    return {pid: {"CPT Eligible": True, "Exclude": False, "Lock": False,
                  "CPT Lock": False, "Priority": "Neutral"} for pid in _ID2NAME}


def _clean_portfolio():
    return pd.DataFrame([_lineup(1, "1", "Puka Nacua",
                                 ["2", "3", "4", "5", "6"],
                                 ["QB One", "Flex Three", "Flex Four", "Flex Five", "Flex Six"])])


def test_audit_catches_cpt_ineligible_captain():
    strat = _clean_strategy()
    strat["1"]["CPT Eligible"] = False  # the Puka case: CPT? unchecked
    v = audit_showdown_portfolio(_clean_portfolio(), strat, _ID2NAME)
    _check("audit: CPT-ineligible captain flagged", len(v) == 1)
    _check("audit: names the player", "Puka Nacua" in v[0] and "CPT?" in v[0])


def test_audit_catches_excluded_player():
    strat = _clean_strategy()
    strat["3"]["Exclude"] = True
    v = audit_showdown_portfolio(_clean_portfolio(), strat, _ID2NAME)
    _check("audit: excluded player in lineup flagged",
           any("Flex Three" in x and "Out" in x for x in v))


def test_audit_catches_priority_exclude():
    strat = _clean_strategy()
    strat["4"]["Priority"] = "Exclude"
    v = audit_showdown_portfolio(_clean_portfolio(), strat, _ID2NAME)
    _check("audit: Priority=Exclude in lineup flagged",
           any("Flex Four" in x for x in v))


def test_audit_catches_cpt_lock_violation():
    strat = _clean_strategy()
    strat["2"]["CPT Lock"] = True  # QB One locked as Captain but is FLEX here
    v = audit_showdown_portfolio(_clean_portfolio(), strat, _ID2NAME)
    _check("audit: CPT-locked non-captain flagged",
           any("QB One" in x and "captain-locked" in x for x in v))


def test_audit_catches_missing_lock():
    strat = _clean_strategy()
    strat["7"]["Lock"] = True  # Flex Seven locked but not in this lineup
    v = audit_showdown_portfolio(_clean_portfolio(), strat, _ID2NAME)
    _check("audit: locked-but-missing player flagged",
           any("Flex Seven" in x and "missing" in x for x in v))


def test_audit_clean_portfolio_passes():
    v = audit_showdown_portfolio(_clean_portfolio(), _clean_strategy(), _ID2NAME)
    _check("audit: clean portfolio has no violations", v == [])


def test_audit_empty_portfolio():
    _check("audit: empty portfolio has no violations",
           audit_showdown_portfolio(pd.DataFrame(), _clean_strategy(), _ID2NAME) == [])


def test_nav_jump_mechanism():
    """Headless AppTest of the exact nav pattern used in main.py: a keyed
    segmented control whose value lives in session state, plus a jump button
    whose on_click callback sets the nav state (setting a widget key after
    the widget was instantiated raises StreamlitWidgetAlreadyInstantiatedError,
    so the jump must happen in the pre-run callback phase)."""
    from streamlit.testing.v1 import AppTest

    script = (
        "import streamlit as st\n"
        "_TABS = ['Build', 'Players', 'Lineups']\n"
        "def _go(tab):\n"
        "    st.session_state['nav'] = tab\n"
        "st.session_state.setdefault('nav', _TABS[0])\n"
        "nav = st.segmented_control('W', _TABS, key='nav', label_visibility='collapsed')\n"
        "if nav == _TABS[2]:\n"
        "    st.button('Adjust Players', key='jump_players', on_click=_go, args=(_TABS[1],))\n"
        "    st.write('results-here')\n"
        "if nav == _TABS[1]:\n"
        "    st.write('players-editor-here')\n"
    )
    with open("/tmp/_nav_jump_apptest.py", "w") as f:
        f.write(script)
    at = AppTest.from_file("/tmp/_nav_jump_apptest.py")
    at.run(timeout=120)
    _check("nav: no exception", len(at.exception) == 0)
    _check("nav: segmented control rendered", len(at.segmented_control) == 1)
    _check("nav: starts on Build", at.segmented_control(key="nav").value == "Build")

    # go to Lineups, tap the jump button, rerun -> Players section renders
    at.segmented_control(key="nav").set_value("Lineups").run(timeout=120)
    _check("nav: results section rendered",
           any("results-here" in str(m.value) for m in at.markdown))
    _check("nav: jump button present", len(at.button) == 1)
    at.button(key="jump_players").click().run(timeout=120)
    _check("nav: jump moved state to Players",
           at.segmented_control(key="nav").value == "Players")
    _check("nav: players section rendered after jump",
           any("players-editor-here" in str(m.value) for m in at.markdown))


if __name__ == "__main__":
    test_audit_catches_cpt_ineligible_captain()
    test_audit_catches_excluded_player()
    test_audit_catches_priority_exclude()
    test_audit_catches_cpt_lock_violation()
    test_audit_catches_missing_lock()
    test_audit_clean_portfolio_passes()
    test_audit_empty_portfolio()
    test_nav_jump_mechanism()
    print("\nALL SHOWDOWN AUDIT/NAV TESTS PASSED")
