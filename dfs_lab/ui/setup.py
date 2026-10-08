"""Moved verbatim from streamlit_app.py (refactor/modularize). No logic changes."""

import base64
import csv
import math
import io
import os
import json
import re
import difflib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


from dfs_lab import styles


def _brand_logo_img(cls="lab-flask-img"):
    """Aytia brand mark as an inline <img>; falls back to a serif alpha glyph."""
    try:
        # setup.py is dfs_lab/ui/setup.py; the logo lives in repo-root assets/.
        p = Path(__file__).resolve().parent.parent.parent / "assets" / "aytia-logo.webp"
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return (f'<img class="{cls}" alt="Aytia logo" '
                f'src="data:image/webp;base64,{b64}"/>')
    except Exception:
        return '<span class="lab-flask">α</span>'


def _disc_a_img(px=46):
    """The logo's gold disc + alpha, cropped from the logo file, used as the
    wordmark's 'a': [disc]ytia. Falls back to a serif alpha glyph."""
    try:
        # setup.py is dfs_lab/ui/setup.py; the medallion lives in repo-root assets/.
        p = Path(__file__).resolve().parent.parent.parent / "assets" / "aytia-disc-a.png"
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return (f'<img class="lab-disc-a" alt="a" style="height:{px}px;width:{px}px;" '
                f'src="data:image/png;base64,{b64}"/>')
    except Exception:
        return '<span class="lab-flask">α</span>'


def render_setup():
    """Pre-upload setup page (moved verbatim from streamlit_app.py)."""
    st.markdown(styles.SETUP_BASE_CSS, unsafe_allow_html=True)

    st.markdown(styles.SETUP_V43_CSS, unsafe_allow_html=True)

    st.markdown(styles.SETUP_V5_CSS, unsafe_allow_html=True)


    st.markdown(styles.SETUP_LABEL_CSS, unsafe_allow_html=True)
    st.markdown(styles.SETUP_V633_CSS, unsafe_allow_html=True)

    st.markdown(styles.SETUP_V64_CSS, unsafe_allow_html=True)


    st.markdown(styles.SETUP_NATIVE_CC_CSS, unsafe_allow_html=True)

    st.markdown(styles.SETUP_COUNT_READOUT_CSS, unsafe_allow_html=True)
    st.markdown(styles.SETUP_IPAD_SURFACE_CSS, unsafe_allow_html=True)

    st.markdown(styles.SETUP_IPAD_CONTRAST_CSS, unsafe_allow_html=True)

    st.markdown(styles.SETUP_PRIMARY_BUTTON_CSS, unsafe_allow_html=True)

    st.markdown(styles.SETUP_DARK_CC_CSS, unsafe_allow_html=True)

    st.markdown(styles.SETUP_APPBAR_DARK_CSS, unsafe_allow_html=True)

    st.markdown(f'''<div class="lab-appbar">
  <div class="lab-brand">{_disc_a_img(46)}<b>ytia</b></div>
  <div class="lab-appbar-copy"><strong>NFL DFS COMMAND CENTER</strong><span>Build · Explore · Challenge</span></div>
  <div class="lab-live"><i></i> LIVE SLATE</div>
</div>''',unsafe_allow_html=True)

    st.markdown(f"""
<div style="text-align:center; padding: 1.2rem 0 0.6rem;">
<div style="display:flex;align-items:center;justify-content:center;gap:5px;font-family:'Source Serif 4',Georgia,'Times New Roman',serif; font-size:3rem; font-weight:700; letter-spacing:0.01em;">{_disc_a_img(76)}<span>ytia</span></div>
<div style="opacity:0.75; margin-top:0.3rem;">Start with a belief. Update it with evidence.</div>
</div>
""", unsafe_allow_html=True)

    with st.expander("📖 Guide — how to use Aytia", expanded=False):
        from dfs_lab.ui.guide import render_guide
        render_guide(mode="showdown")

    st.markdown("""<div class="command-strip"><div><span class="command-live">⚙</span><b> BUILD CONTROL CENTER</b><span class="command-copy"> Game type · Contest · Entries · Strategy</span></div><div class="command-arrow">SETUP ↓</div></div>""", unsafe_allow_html=True)
    with st.expander("⚙  BUILD CONTROLS  ·  GAME TYPE & CONTEST", expanded=True):
        st.caption("These controls live inside Aytia and stay available after the slate loads.")
        cc1,cc2=st.columns(2)
        with cc1:
            mode=st.segmented_control("Game type",["Classic","Showdown"],default="Showdown")
            preset=st.selectbox("Contest preset",["Large GPP","Small-field GPP","Single Entry","Winner Take All","Cash-ish"])
        defaults={"Large GPP":(50000,"GPP / top-heavy","150-Max"),"Small-field GPP":(500,"GPP / top-heavy","3-Max"),"Single Entry":(300,"Flatter payouts","Single Entry"),"Winner Take All":(500,"Winner take all","Single Entry"),"Cash-ish":(100,"Flatter payouts","Single Entry")}
        dfield,dpayout,dentry=defaults[preset]
        with cc2:
            field_size=st.number_input("Field size",min_value=2,value=int(dfield),step=1)
            entry_format=st.selectbox("Contest max entries",["Single Entry","3-Max","20-Max","150-Max"],index=["Single Entry","3-Max","20-Max","150-Max"].index(dentry),
                help="The contest's entry limit — NOT how many you're building. Sets how the app models the field: higher max = flatter estimated ownership (the field owns contrarian plays more) and higher ceiling requirements.")
        cc3,cc4=st.columns(2)
        with cc3: payout_style=st.selectbox("Payout",["GPP / top-heavy","Winner take all","Flatter payouts"],index=["GPP / top-heavy","Winner take all","Flatter payouts"].index(dpayout))
        with cc4:
            lineup_choice=st.radio("Lineups to build",[5,20,50,100,150,"Custom"],index=1,horizontal=True,key="dfs_lineup_choice",
                help="How many lineups the optimizer generates for you to review. Build more than you'll enter to have options — the game-world cap keeps this candidate pool diverse.")
            if lineup_choice=="Custom":
                lineup_count=int(st.number_input("Custom lineup count",min_value=1,max_value=500,value=int(st.session_state.get("dfs_custom_lineups",100)),step=1,key="dfs_custom_lineups"))
            else:
                lineup_count=int(lineup_choice)
            my_entries=int(st.number_input("My entries — how many I'm actually playing",min_value=1,max_value=500,value=min(int(st.session_state.get("dfs_my_entries",lineup_count)),lineup_count),step=1,key="dfs_my_entries",
                help="How many you'll actually enter in the contest. Used for the final export — download just your entries, not the full candidate pool."))
            st.markdown(f"<div class='lineup-count-readout'><b>{lineup_count}</b><span> lineups built</span> · <b>{my_entries}</b><span> entered</span></div>",unsafe_allow_html=True)
            if entry_format!="Single Entry":
                _emax=int(entry_format.split("-")[0])
                if my_entries>_emax:
                    st.warning(f"You're entering {my_entries} lineups but the contest allows max {_emax}.")
                elif lineup_count>_emax:
                    st.caption(f"Building {lineup_count} candidates for review ({_emax}-max contest) — export your top {my_entries} when you're done.")
        with st.expander("Advanced build settings"):
            seed=st.number_input("Random seed",min_value=1,value=42,step=1)
            st.caption("Change this only when you want a different randomized batch.")
    return {
        "mode": mode,
        "field_size": field_size,
        "entry_format": entry_format,
        "payout_style": payout_style,
        "lineup_count": lineup_count,
        "my_entries": my_entries,
        "seed": seed,
    }
