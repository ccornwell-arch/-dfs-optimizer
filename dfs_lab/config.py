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
# game when the opposing offense is worthy of a shootout. Sized below the
# hard-stack bonuses (+1.1 QB / +0.45 WR-TE) and on the order of a strong
# leverage edge: enough to flip close FLEX/WR3 calls toward correlation,
# never enough to force a bad play. GPP best practice is ~50% game-stack
# rate (FantasyLabs, Fantasy Footballers) — a lever, not a mandate.
BRINGBACK_NUDGE = 0.6
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
