"""Moved verbatim from streamlit_app.py (refactor/modularize). No logic changes."""

import csv
import math
import io
import os
import json
import re
import difflib
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def _first_existing(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None

def contest_aggression(field_size, payout_style):
    fs = max(2, int(field_size))
    raw = (math.log10(fs) - 2.0) / 4.0
    raw = min(1.0, max(0.08, raw))
    if payout_style == "Winner take all":
        raw = min(1.0, raw + 0.14)
    elif payout_style == "Flatter payouts":
        raw = max(0.04, raw - 0.12)
    return raw

def percentile_label(x, series):
    if len(series) < 2:
        return "Average"
    pct = float((series <= x).mean())
    if pct >= 0.90: return "Excellent"
    if pct >= 0.70: return "Very good"
    if pct >= 0.45: return "Good"
    if pct >= 0.25: return "Average"
    return "Weak"

def player_editor_widget_key(prefix, slate_fp, view_ids):
    """Widget key for a player data_editor whose edits are row-positional.

    st.data_editor reports edits as {row_position: changes}. When the editor
    sits inside a form, edits stay pending until the user presses apply — and
    if the visible row order changes in between (sort/filter), a fixed widget
    key would silently apply the pending edit to the WRONG player (e.g. an
    "Out" check landing on someone else while the intended player stays in
    the build pool). Scoping the key to the exact visible row order forces
    Streamlit to reset pending edits whenever the order changes, so a stale
    positional edit can never exclude or lock the wrong player.
    """
    return f"{prefix}|{slate_fp}|" + ",".join(str(x) for x in view_ids)
