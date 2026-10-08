"""Multipage navigation registry for Aytia.

Each workspace section is a real Streamlit page (pages/*.py). The segmented
nav control switches pages via st.switch_page instead of conditionally
rendering sections on one long scrolling page.
"""

import streamlit as st

# (tab label, page file) in nav order.
CLASSIC_PAGES = [
    ("🧠 Slate Intel", "pages/classic_intel.py"),
    ("⚡ Build", "pages/classic_build.py"),
    ("👤 Players", "pages/classic_players.py"),
    ("⚙ Rules", "pages/classic_rules.py"),
    ("📋 Lineups", "pages/classic_lineups.py"),
    ("📊 Exposure", "pages/classic_exposure.py"),
    ("📖 Guide", "pages/classic_guide.py"),
]

SHOWDOWN_PAGES = [
    ("⚡ Build", "pages/showdown_build.py"),
    ("👤 Players", "pages/showdown_players.py"),
    ("🔗 Relationships", "pages/showdown_relationships.py"),
    ("🧠 Game Intel", "pages/showdown_intel.py"),
    ("⚙ Rules", "pages/showdown_rules.py"),
    ("📋 Lineups", "pages/showdown_lineups.py"),
    ("📊 Exposure", "pages/showdown_exposure.py"),
    ("📖 Guide", "pages/showdown_guide.py"),
]

# Story detour pages (not in the main nav; reached via "Tell the story" from Build).
STORY_PAGES = {
    "Classic": "pages/classic_story.py",
    "Showdown": "pages/showdown_story.py",
}

# Map tab label -> page file, for programmatic jumps.
_CLASSIC_MAP = dict(CLASSIC_PAGES)
_SHOWDOWN_MAP = dict(SHOWDOWN_PAGES)


def pages_for(mode):
    return CLASSIC_PAGES if mode == "Classic" else SHOWDOWN_PAGES


def page_file_for(mode, tab):
    m = _CLASSIC_MAP if mode == "Classic" else _SHOWDOWN_MAP
    return m.get(tab)


def goto_page(mode, tab):
    """Programmatic nav jump (safe to call from button on_click)."""
    target = page_file_for(mode, tab)
    if target:
        st.switch_page(target)


def nav_state_key(mode):
    return "classic_nav" if mode == "Classic" else "sd_nav"
