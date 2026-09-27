"""Moved verbatim from streamlit_app.py (refactor/modularize). No logic changes."""

import subprocess
from functools import lru_cache

APP_BUILD = "2026.09.24 · Lineup Why Drawer"


@lru_cache(maxsize=1)
def git_build_stamp():
    """Short git commit hash of the deployed checkout, for the app footer.

    Lets the user confirm which build is live. Never raises; returns
    "unknown" when git is unavailable (e.g. non-git installs).
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        h = (out.stdout or "").strip()
        return h if h else "unknown"
    except Exception:
        return "unknown"
ROSTER_SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
PRIORITY_OPTIONS = ["Core", "Like", "Neutral", "Fade", "Exclude"]
PRIORITY_BONUS = {"Core": 2.8, "Like": 1.35, "Neutral": 0.0, "Fade": -1.5, "Exclude": -100.0}
TEAM_PRIORITY_BONUS = {"Core": 1.8, "Like": 0.9, "Neutral": 0.0, "Fade": -0.8, "Exclude": -100.0}
# Soft bring-back incentive for bringback_mode="Optional": added to the
# objective of active opposing skill players (RB/WR/TE) in the chosen QB's
# game when the opposing offense is worthy of a shootout. Calibrated (Sep
# 2026) on a realistic-scale synthetic slate: 0.6 flipped only ~14% of
# eligible lineups (bonus drowned by projection gaps); 1.2 lands ~55%
# bring-back rate, the ~50% game-stack zone from GPP research (FantasyLabs,
# Fantasy Footballers). Sized at "Like"-scale (1.35): weaker than an explicit
# Core (2.8), so it leans toward correlation without overriding your opinions.
# Never enough to force a bad play; Required/None are untouched.
BRINGBACK_NUDGE = 1.2
SHOWDOWN_SLOTS = ["CPT", "FLEX1", "FLEX2", "FLEX3", "FLEX4", "FLEX5"]
GAME_WORLDS = {}
CONTEXT_FACTOR_WEIGHTS = {
    "Defense": 0.045,
    "Usage": 0.050,
    "Home/Rest": 0.020,
    "Travel": 0.015,
    "Time/Split": 0.010,
}
CONTEXT_RATING_LABELS = {
    -3: "Strong negative", -2: "Negative", -1: "Slight negative", 0: "Neutral",
    1: "Slight positive", 2: "Positive", 3: "Strong positive",
}
