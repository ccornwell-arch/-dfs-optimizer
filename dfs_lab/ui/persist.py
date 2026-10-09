"""'Aytia remembers' — cross-session persistence of user inputs.

Problem (reported Oct 9, 2026): on iPhone, leaving the app for ~a minute
suspends the Safari tab; the websocket drops and Streamlit discards the
server-side session. Coming back yields a fresh session with every control
back at its default — rules, players, game type, all of it.

Fix: every user input that matters is mirrored into a single URL query
param (`a_cfg`, a versioned JSON blob). The URL survives tab suspends,
refreshes, bookmarks — even app redeploys (unlike server-side caches).
On a fresh session the values are seeded back into session state before any
widget renders, so the app reopens exactly as the user left it.

Scope is INPUTS only:
  - build controls (game type, contest, field, entries, payout, counts, seed)
  - player state (locks / outs / CPT / priorities / exposures / proj overrides)
  - rules (classic + showdown rule widgets, team priorities, relationships,
    game-intel ratings, scenario engine, story inputs)
  - the page the user was on (so they land back where they were)
NOT persisted: uploaded file bytes (already covered by the `slate_session`
server cache while the app process lives), built lineups (one tap rebuilds),
AI chat, transient view prefs (sort/filter).

Player deltas are guarded by a slate fingerprint (hash of sorted player IDs):
they only re-apply when the same slate file is loaded, so last week's locks
can never leak into this week's pool. When the slate changes mid-session the
slate-bound deltas are dropped from the blob.
"""

import hashlib
import json
import math

PARAM = "a_cfg"
VERSION = 1
_SLATE_KEY = "_persist_slate_fp"
_RESTORED_KEY = "_persist_restored"

# ---------------------------------------------------------------------------
# Pure logic (no streamlit import): unit-testable.
# ---------------------------------------------------------------------------

CLASSIC_DEFAULTS = {
    "Lock": False, "Exclude": False, "Priority": "Neutral",
    "Min Exposure": 0, "Max Exposure": 100,
}
SHOWDOWN_DEFAULTS = {
    "Lock": False, "CPT Lock": False, "Exclude": False,
    "CPT Eligible": False, "Priority": "Neutral",
    "Min Exposure": 0, "Max Exposure": 100, "CPT Min": 0, "CPT Max": 100,
}
TEAM_DEFAULT = "Neutral"
CONTEXT_DEFAULTS = {
    "Defense": 0, "Usage": 0, "Home/Rest": 0, "Travel": 0,
    "Time/Split": 0, "Confidence": 50, "Note": "",
}


def _finite(v):
    return v if not isinstance(v, float) or math.isfinite(v) else None


def player_deltas(strategy, defaults):
    """Non-default fields per player: {pid: {field: value}}.

    The Apply handlers write EVERY table row into the strategy dict, so the
    full dict is far too big for a URL. Only deviations from the table
    defaults are persisted.
    """
    out = {}
    for pid, entry in (strategy or {}).items():
        if not isinstance(entry, dict):
            continue
        delta = {}
        for field, default in defaults.items():
            if field not in entry:
                continue
            v = _finite(entry[field])
            if v is None:
                continue
            if isinstance(default, float):
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
            if v != default:
                delta[field] = v
        if delta:
            out[str(pid)] = delta
    return out


def apply_player_deltas(strategy, deltas, defaults):
    """Merge persisted deltas into an (empty) strategy dict, filling defaults."""
    for pid, delta in (deltas or {}).items():
        if not isinstance(delta, dict):
            continue
        entry = dict(defaults)
        for field, v in delta.items():
            if field in defaults:
                entry[field] = v
        strategy[str(pid)] = entry
    return strategy


def team_deltas(team_strategy):
    return {t: p for t, p in (team_strategy or {}).items()
            if p != TEAM_DEFAULT}


def context_deltas(context):
    out = {}
    for pid, cfg in (context or {}).items():
        if not isinstance(cfg, dict):
            continue
        delta = {k: v for k, v in cfg.items()
                 if k in CONTEXT_DEFAULTS and v != CONTEXT_DEFAULTS[k]}
        if delta:
            out[str(pid)] = delta
    return out


def slate_fingerprint(id_series):
    """Stable guard so one slate's player edits can't apply to another."""
    ids = sorted(str(i) for i in id_series)
    return hashlib.md5(",".join(ids).encode()).hexdigest()[:12]


def encode_blob(blob):
    return json.dumps(blob, separators=(",", ":"), sort_keys=True)


def decode_blob(raw):
    """Parse a blob from the URL. Returns None on any problem (corrupt,
    hand-edited, or future version) — the app then just uses defaults."""
    try:
        if not raw:
            return None
        blob = json.loads(raw)
        if not isinstance(blob, dict) or blob.get("v") != VERSION:
            return None
        return blob
    except Exception:
        return None


# Option-constrained widgets: if a persisted value is no longer among the
# options (e.g. an option was renamed in a later deploy), drop it so the
# widget falls back to its default instead of raising.
ALLOWED_VALUES = {
    "cfg_mode": {"Classic", "Showdown"},
    "cfg_preset": {"Large GPP", "Small-field GPP", "Single Entry",
                   "Winner Take All", "Cash-ish"},
    "cfg_entry": {"Single Entry", "3-Max", "20-Max", "150-Max"},
    "cfg_payout": {"GPP / top-heavy", "Winner take all", "Flatter payouts"},
    "classic_bringback": {"Optional", "Required", "None"},
    "classic_qb_stack": {1, 2, 3},
    "classic_max_team": {4, 5, 6, 7, 8, 9},
    "classic_max_game": {4, 5, 6, 7, 8, 9},
    "classic_max_te": {1, 2, 3},
    "sd_script": {"Neutral", "Auto from score", "Shootout",
                  "Pass-heavy shootout", "Low-scoring game",
                  "Defensive / field-goal battle", "Ground-and-pound",
                  "Team wins close", "Team dominates", "Team plays from ahead",
                  "Team passing comeback"},
    "context_strength": {"Conservative", "Standard", "Aggressive"},
}


def _scalar_ok(key, value):
    allowed = ALLOWED_VALUES.get(key)
    if allowed is None:
        return True
    return value in allowed


# ---------------------------------------------------------------------------
# Registry: which session-state keys persist.
# ---------------------------------------------------------------------------

# Scalar widget values persisted verbatim (JSON keeps their types).
# Setup keys are added to the widgets in dfs_lab/ui/setup.py.
SCALAR_KEYS = [
    # Build controls (setup)
    "cfg_mode", "cfg_preset", "cfg_field", "cfg_entry", "cfg_payout",
    "dfs_lineup_choice", "dfs_custom_lineups", "dfs_my_entries", "cfg_seed",
    # Classic rules
    "classic_min_salary", "classic_qb_stack", "classic_bringback",
    "classic_max_team", "classic_max_game", "classic_max_te",
    "classic_flex_control", "classic_flex_rb", "classic_flex_wr",
    "classic_flex_te", "classic_no_dst", "classic_no_off",
    "classic_allow_qb_rb", "classic_pref_stack",
    # Showdown rules (persistent_widget persistent keys)
    "sd_script", "sd_use_score", "sd_script_team", "sd_score_0", "sd_score_1",
    "sd_intensity", "sd_max_dst", "sd_max_k", "sd_rb_pair", "sd_wrte_pair",
    "sd_allow_33", "sd_allow_42", "sd_allow_51", "sd_auto_shape",
    "sd_world_max_share", "qb_cpt_pc_rule",
    # Shared
    "max_one_rb_team", "context_strength",
    # Story inputs (persistent_widget persistent keys in story.py)
    "story_sd_winner", "story_sd_gtype", "story_sd_as", "story_sd_hs",
    "story_sd_how_won", "story_sd_how_lost", "story_sd_drivers",
    "story_sd_fades",
    "story_cl_love", "story_cl_bust", "story_cl_leverage", "story_cl_fade",
]


def _snapshot_scalars(session_state):
    out = {}
    for key in SCALAR_KEYS:
        if key not in session_state:
            continue
        v = session_state[key]
        if v is None or isinstance(v, (str, int, float, bool, list)):
            # JSON-round-trip to normalize (e.g. tuples -> lists) and to
            # reject anything exotic.
            try:
                out[key] = json.loads(json.dumps(v))
            except Exception:
                continue
    return out


def build_blob(session_state):
    """Snapshot everything worth persisting into a versioned blob dict."""
    slate_fp = session_state.get(_SLATE_KEY)
    blob = {
        "v": VERSION,
        "slate": slate_fp,
        "s": _snapshot_scalars(session_state),
        "pd": {
            "classic": player_deltas(session_state.get("strategy_master"),
                                     CLASSIC_DEFAULTS),
            "showdown": player_deltas(session_state.get("showdown_strategy"),
                                      SHOWDOWN_DEFAULTS),
        },
        "po": {
            "classic": dict(session_state.get("classic_projection_overrides") or {}),
            "showdown": dict(session_state.get("projection_overrides") or {}),
        },
        "td": team_deltas(session_state.get("team_strategy_master")),
        "rel": list(session_state.get("showdown_relationships") or []),
        "ctx": context_deltas(session_state.get("showdown_context")),
        "page": session_state.get("_persist_page"),
    }
    # Drop empty sections to keep the URL short.
    return {k: v for k, v in blob.items() if v not in (None, {}, [], "")}


# ---------------------------------------------------------------------------
# Streamlit-bound glue.
# ---------------------------------------------------------------------------

def _st():
    import streamlit as st
    return st


def restore(session_state=None, query_params=None):
    """Seed session state from the URL blob. First run of a session only.

    Must run before any widget is created. Never overwrites a value the
    session already has (so active editing is never fought).
    Returns True when values were restored from the URL.
    """
    st = _st()
    ss = session_state if session_state is not None else st.session_state
    qp = query_params if query_params is not None else st.query_params
    if ss.get(_RESTORED_KEY):
        return False
    ss[_RESTORED_KEY] = True
    blob = decode_blob(qp.get(PARAM))
    if not blob:
        return False
    restored = False
    for key, value in (blob.get("s") or {}).items():
        if key not in ss and key in SCALAR_KEYS and _scalar_ok(key, value):
            ss[key] = value
            restored = True
    # Stash the blob for the player-state phase (needs the pool first).
    ss["_persist_blob"] = blob
    return restored


def restore_player_state(df, mode, session_state=None):
    """Apply persisted player/context deltas after the pool is built.

    Only fills EMPTY strategy dicts and only when the slate fingerprint
    matches — a returning session starts empty, an active one is never
    clobbered, and a different slate never inherits edits.
    """
    st = _st()
    ss = session_state if session_state is not None else st.session_state
    blob = ss.get("_persist_blob")
    if not blob:
        return False
    fp = slate_fingerprint(df["ID"]) if df is not None else None
    ss[_SLATE_KEY] = fp
    if not fp or blob.get("slate") != fp:
        return False
    strat_key = "strategy_master" if mode == "Classic" else "showdown_strategy"
    defaults = CLASSIC_DEFAULTS if mode == "Classic" else SHOWDOWN_DEFAULTS
    # The strategy dict is the primary player-state store: if it already has
    # data this session is active and nothing is touched. (Projection
    # overrides are only ever written alongside strategy entries, so gating
    # on the strategy dict covers both.)
    if ss.get(strat_key):
        return False
    applied = False
    deltas = (blob.get("pd") or {}).get(mode.lower(), {})
    if deltas:
        ss[strat_key] = {}
        apply_player_deltas(ss[strat_key], deltas, defaults)
        applied = True
    po_key = "classic_projection_overrides" if mode == "Classic" else "projection_overrides"
    if not ss.get(po_key):
        po = (blob.get("po") or {}).get(mode.lower(), {})
        if po:
            ss[po_key] = {str(k): v for k, v in po.items()}
            applied = True
    if mode == "Classic":
        if not ss.get("team_strategy_master"):
            td = blob.get("td") or {}
            if td:
                ss["team_strategy_master"] = dict(td)
                applied = True
    else:
        if not ss.get("showdown_relationships"):
            rel = blob.get("rel") or []
            if rel:
                ss["showdown_relationships"] = list(rel)
                applied = True
        if not ss.get("showdown_context"):
            ctx = blob.get("ctx") or {}
            if ctx:
                merged = {}
                for pid, delta in ctx.items():
                    entry = dict(CONTEXT_DEFAULTS)
                    entry.update({k: v for k, v in delta.items()
                                  if k in CONTEXT_DEFAULTS})
                    merged[str(pid)] = entry
                ss["showdown_context"] = merged
                applied = True
    return applied


def note_page(page_file, session_state=None):
    """Record which page the user is on (called from boot_page)."""
    st = _st()
    ss = session_state if session_state is not None else st.session_state
    ss["_persist_page"] = page_file


def save(session_state=None, query_params=None):
    """Snapshot inputs into the URL blob. Idempotent: only writes when the
    blob changed, and query-param writes don't trigger reruns."""
    st = _st()
    ss = session_state if session_state is not None else st.session_state
    qp = query_params if query_params is not None else st.query_params
    # If the slate changed mid-session, last slate's player edits must not
    # ride along under the new fingerprint.
    blob = build_blob(ss)
    current_fp = ss.get(_SLATE_KEY)
    last_fp = ss.get("_persist_last_blob_slate")
    if current_fp and last_fp and current_fp != last_fp:
        for section in ("pd", "po", "ctx", "rel"):
            blob.pop(section, None)
    if current_fp:
        ss["_persist_last_blob_slate"] = current_fp
    raw = encode_blob(blob)
    try:
        if qp.get(PARAM) != raw:
            qp[PARAM] = raw
    except Exception:
        pass


def clear_saved(session_state=None, query_params=None):
    """Forget everything (the 'start over' escape hatch)."""
    st = _st()
    ss = session_state if session_state is not None else st.session_state
    qp = query_params if query_params is not None else st.query_params
    try:
        if PARAM in qp:
            del qp[PARAM]
    except Exception:
        pass
    for key in list(ss.keys()):
        if (key in SCALAR_KEYS or key in ("strategy_master", "showdown_strategy",
              "team_strategy_master", "showdown_relationships",
              "classic_projection_overrides", "projection_overrides",
              "showdown_context", "_persist_blob", "_persist_page",
              "_persist_last_blob_slate", _SLATE_KEY, _RESTORED_KEY)
                or key.startswith("_w_")):
            ss.pop(key, None)
