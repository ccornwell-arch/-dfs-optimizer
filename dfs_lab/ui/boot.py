"""Shared boot for Aytia multipage workspace pages.

Every page under pages/ calls boot_page(mode, tab):
  1. Injects the Aytia Gold theme CSS.
  2. Verifies settings exist in session state (else back to the entry/setup).
  3. Sets the nav state key so render_main shows the right section.
Returns the settings dict (never None; redirects via switch_page + stop).
"""

import streamlit as st

from dfs_lab import styles
from dfs_lab.theme import theme_css
from dfs_lab.ui import nav as _nav


def boot_page(mode, tab):
    st.markdown(styles.MAIN_V634_SHELL_CSS, unsafe_allow_html=True)
    st.markdown(styles.MAIN_DYNAMIC_SHELL_CSS, unsafe_allow_html=True)
    st.markdown(styles.MAIN_UNIFORM_THEME_CSS, unsafe_allow_html=True)
    st.markdown(styles.MAIN_EXPANDERS_CSS, unsafe_allow_html=True)
    st.markdown(styles.MAIN_EXPANDER_HEADERS_CSS, unsafe_allow_html=True)
    st.markdown(styles.MAIN_WHY_STRIP_CSS, unsafe_allow_html=True)
    st.markdown(styles.RCC_DARK_CSS, unsafe_allow_html=True)
    st.markdown(theme_css(), unsafe_allow_html=True)
    st.markdown(styles.MAIN_TABS_CSS, unsafe_allow_html=True)
    # Hide Streamlit's default sidebar page nav; the segmented control owns it.
    st.markdown("<style>[data-testid='stSidebarNav']{display:none}</style>",
                unsafe_allow_html=True)

    settings = st.session_state.get("aytia_settings")
    if not settings or settings.get("mode") != mode:
        st.switch_page("streamlit_app.py")
        st.stop()
    # Pre-seed the nav widget state so render_main's segmented control opens
    # on this page. on_change only fires on user taps, not this assignment.
    st.session_state[_nav.nav_state_key(mode)] = tab
    return settings
