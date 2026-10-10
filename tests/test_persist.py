"""Tests for dfs_lab/ui/persist.py — 'Aytia remembers' cross-session persistence.

Background (Oct 9, 2026): iOS Safari suspends background tabs within ~a
minute; the websocket drops and Streamlit discards the server-side session.
The user returned to find every control back at defaults. The fix mirrors
user inputs into a versioned URL query-param blob (`a_cfg`) and re-seeds
session state from it on a fresh session.

These tests cover the pure logic (no Streamlit runtime needed — restore /
save accept plain dicts for session_state and query_params).
"""

import sys

sys.path.insert(0, ".")

import pandas as pd

from dfs_lab.ui import persist


def _ss():
    return {}


def _qp():
    return {}


# --- encode / decode -------------------------------------------------------

def test_blob_round_trip():
    blob = {"v": 1, "slate": "abc123",
            "s": {"cfg_mode": "Showdown", "cfg_field": 5000,
                  "classic_pref_stack": ["DAL", "TB"]},
            "pd": {"showdown": {"8472": {"Lock": True}}},
            "page": "pages/showdown_build.py"}
    raw = persist.encode_blob(blob)
    assert persist.decode_blob(raw) == blob


def test_decode_rejects_garbage():
    assert persist.decode_blob(None) is None
    assert persist.decode_blob("") is None
    assert persist.decode_blob("not-json{{{") is None
    assert persist.decode_blob('{"v": 999, "s": {}}') is None
    assert persist.decode_blob('{"s": {}}') is None
    assert persist.decode_blob('[1,2,3]') is None


# --- player deltas ---------------------------------------------------------

def test_showdown_deltas_only_nondefault():
    strat = {
        "1": dict(persist.SHOWDOWN_DEFAULTS),  # all default -> dropped
        "2": {**persist.SHOWDOWN_DEFAULTS, "Lock": True},
        "3": {**persist.SHOWDOWN_DEFAULTS, "Exclude": True,
              "Priority": "Exclude", "Min Exposure": 20, "Max Exposure": 60},
    }
    d = persist.player_deltas(strat, persist.SHOWDOWN_DEFAULTS)
    assert "1" not in d
    assert d["2"] == {"Lock": True}
    assert d["3"] == {"Exclude": True, "Priority": "Exclude",
                      "Min Exposure": 20, "Max Exposure": 60}


def test_classic_deltas():
    strat = {"9": {**persist.CLASSIC_DEFAULTS, "Priority": "Fade"}}
    d = persist.player_deltas(strat, persist.CLASSIC_DEFAULTS)
    assert d == {"9": {"Priority": "Fade"}}


def test_apply_player_deltas_fills_defaults():
    strat = {}
    persist.apply_player_deltas(strat, {"7": {"Lock": True}},
                                persist.SHOWDOWN_DEFAULTS)
    assert strat["7"]["Lock"] is True
    assert strat["7"]["Priority"] == "Neutral"
    assert strat["7"]["Max Exposure"] == 100
    assert strat["7"]["CPT Eligible"] is False


def test_player_deltas_skips_nonfinite():
    strat = {"5": {**persist.SHOWDOWN_DEFAULTS, "Min Exposure": float("nan")}}
    d = persist.player_deltas(strat, persist.SHOWDOWN_DEFAULTS)
    assert d == {}


# --- slate fingerprint -----------------------------------------------------

def test_slate_fingerprint_stable_and_sensitive():
    a = pd.Series(["8472", "9103", "1234"])
    b = pd.Series(["1234", "8472", "9103"])  # same set, different order
    c = pd.Series(["8472", "9103", "9999"])  # one different
    assert persist.slate_fingerprint(a) == persist.slate_fingerprint(b)
    assert persist.slate_fingerprint(a) != persist.slate_fingerprint(c)
    assert len(persist.slate_fingerprint(a)) == 12


# --- restore ---------------------------------------------------------------

def test_restore_seeds_missing_keys_only():
    ss, qp = _ss(), _qp()
    blob = {"v": 1, "s": {"cfg_mode": "Classic", "cfg_field": 5000,
                          "classic_bringback": "Required"}}
    qp[persist.PARAM] = persist.encode_blob(blob)
    ss["cfg_field"] = 999  # user already has a value: must win
    assert persist.restore(ss, qp) is True
    assert ss["cfg_mode"] == "Classic"
    assert ss["cfg_field"] == 999
    assert ss["classic_bringback"] == "Required"
    # Second call is a no-op (one-shot per session).
    assert persist.restore(ss, qp) is False


def test_restore_ignores_unknown_keys():
    ss, qp = _ss(), _qp()
    blob = {"v": 1, "s": {"cfg_mode": "Showdown", "evil_key": "x",
                          "classic_result_v4": [1, 2, 3]}}
    qp[persist.PARAM] = persist.encode_blob(blob)
    persist.restore(ss, qp)
    assert ss["cfg_mode"] == "Showdown"
    assert "evil_key" not in ss
    assert "classic_result_v4" not in ss


def test_restore_no_blob_is_quiet():
    ss, qp = _ss(), _qp()
    assert persist.restore(ss, qp) is False
    assert ss.get("_persist_restored") is True


# --- restore_player_state --------------------------------------------------

def _df():
    return pd.DataFrame({"ID": ["8472", "9103"]})


def _blob_with_player_state(fp):
    return {
        "v": 1, "slate": fp,
        "pd": {"showdown": {"8472": {"Lock": True}},
               "classic": {"9103": {"Exclude": True}}},
        "po": {"showdown": {"8472": 15.5}, "classic": {}},
        "td": {"DAL": "Love"},
        "rel": [{"a": "x", "rule": "y", "enabled": True}],
        "ctx": {"8472": {"Usage": 2}},
    }


def test_restore_player_state_applies_on_empty_matching_slate():
    ss, qp = _ss(), _qp()
    df = _df()
    fp = persist.slate_fingerprint(df["ID"])
    ss["_persist_blob"] = _blob_with_player_state(fp)
    assert persist.restore_player_state(df, "Showdown", ss) is True
    assert ss["showdown_strategy"]["8472"]["Lock"] is True
    assert ss["showdown_strategy"]["8472"]["Priority"] == "Neutral"
    assert ss["projection_overrides"] == {"8472": 15.5}
    assert ss["showdown_relationships"] == [{"a": "x", "rule": "y",
                                             "enabled": True}]
    assert ss["showdown_context"]["8472"]["Usage"] == 2
    assert ss["showdown_context"]["8472"]["Defense"] == 0


def test_restore_player_state_skips_wrong_slate():
    ss, qp = _ss(), _qp()
    df = _df()
    ss["_persist_blob"] = _blob_with_player_state("deadbeef1234")
    assert persist.restore_player_state(df, "Showdown", ss) is False
    assert ss.get("showdown_strategy") in (None, {})
    assert "projection_overrides" not in ss


def test_restore_player_state_never_clobbers_active_edits():
    ss, qp = _ss(), _qp()
    df = _df()
    fp = persist.slate_fingerprint(df["ID"])
    ss["_persist_blob"] = _blob_with_player_state(fp)
    ss["showdown_strategy"] = {"1111": dict(persist.SHOWDOWN_DEFAULTS)}
    assert persist.restore_player_state(df, "Showdown", ss) is False
    assert "8472" not in ss["showdown_strategy"]


def test_restore_player_state_classic_team_deltas():
    ss, qp = _ss(), _qp()
    df = _df()
    fp = persist.slate_fingerprint(df["ID"])
    ss["_persist_blob"] = _blob_with_player_state(fp)
    assert persist.restore_player_state(df, "Classic", ss) is True
    assert ss["strategy_master"]["9103"]["Exclude"] is True
    assert ss["team_strategy_master"] == {"DAL": "Love"}


# --- save ------------------------------------------------------------------

def test_save_writes_blob_and_is_idempotent():
    ss, qp = _ss(), _qp()
    ss["cfg_mode"] = "Showdown"
    ss["cfg_field"] = 5000
    ss["showdown_strategy"] = {"8472": {**persist.SHOWDOWN_DEFAULTS,
                                        "Lock": True}}
    persist.save(ss, qp)
    first = qp[persist.PARAM]
    blob = persist.decode_blob(first)
    assert blob["s"]["cfg_mode"] == "Showdown"
    assert blob["pd"]["showdown"] == {"8472": {"Lock": True}}
    # Idempotent: second save writes nothing new.
    persist.save(ss, qp)
    assert qp[persist.PARAM] == first


def test_save_drops_slate_bound_sections_on_slate_change():
    ss, qp = _ss(), _qp()
    ss["_persist_slate_fp"] = "fp_one"
    ss["_persist_last_blob_slate"] = "fp_one"
    ss["showdown_strategy"] = {"8472": {**persist.SHOWDOWN_DEFAULTS,
                                        "Lock": True}}
    persist.save(ss, qp)
    assert "8472" in persist.decode_blob(qp[persist.PARAM])["pd"]["showdown"]
    # New slate file loaded mid-session: old slate's edits must not persist.
    ss["_persist_slate_fp"] = "fp_two"
    persist.save(ss, qp)
    blob = persist.decode_blob(qp[persist.PARAM])
    assert blob["slate"] == "fp_two"
    assert "pd" not in blob


def test_clear_saved():
    ss, qp = _ss(), _qp()
    qp[persist.PARAM] = persist.encode_blob({"v": 1})
    ss["cfg_mode"] = "Showdown"
    ss["showdown_strategy"] = {"1": {}}
    ss["_w_sd_script"] = "Shootout"
    ss["unrelated"] = 1
    persist.clear_saved(ss, qp)
    assert persist.PARAM not in qp
    assert "cfg_mode" not in ss
    assert "showdown_strategy" not in ss
    assert "_w_sd_script" not in ss
    assert ss["unrelated"] == 1


# --- registry sanity -------------------------------------------------------

def test_registry_keys_are_stable_identifiers():
    # Every registry key must be a plain string (used as URL + session keys).
    for key in persist.SCALAR_KEYS:
        assert isinstance(key, str) and key and " " not in key
    assert len(set(persist.SCALAR_KEYS)) == len(persist.SCALAR_KEYS)


if __name__ == "__main__":
    names = [n for n in list(globals()) if n.startswith("test_")]
    failed = 0
    for n in names:
        try:
            globals()[n]()
            print(f"PASS {n}")
        except Exception as e:
            failed += 1
            print(f"FAIL {n}: {e!r}")
    print(f"\n{len(names)-failed}/{len(names)} passed")
    sys.exit(1 if failed else 0)


# --- reseed_missing_scalars (Oct 10, 2026: rules reset on page nav) --------

def _blob_with_rules(**overrides):
    s = {"classic_qb_stack": 1, "classic_bringback": "Required",
         "classic_min_salary": 48500, "classic_max_game": 5}
    s.update(overrides)
    return {"v": 1, "s": s}


def test_reseed_fills_missing_widget_keys_from_blob():
    # Simulates the iPhone bug: user set rules on the Rules page, navigated
    # to Build, and the widget-backed keys vanished mid-session.
    ss = {"_persist_blob": _blob_with_rules(),
          "_persist_page": "pages/classic_rules.py"}
    n = persist.reseed_missing_scalars(ss)
    assert n == 4
    assert ss["classic_qb_stack"] == 1
    assert ss["classic_bringback"] == "Required"
    assert ss["classic_min_salary"] == 48500
    assert ss["classic_max_game"] == 5


def test_reseed_never_overwrites_live_values():
    ss = {"_persist_blob": _blob_with_rules(),
          "classic_bringback": "None"}  # user changed it after the save
    n = persist.reseed_missing_scalars(ss)
    assert ss["classic_bringback"] == "None"
    assert ss["classic_qb_stack"] == 1  # missing -> filled
    assert n == 3


def test_reseed_rejects_values_outside_widget_options():
    ss = {"_persist_blob": _blob_with_rules(classic_bringback="Sometimes")}
    n = persist.reseed_missing_scalars(ss)
    assert "classic_bringback" not in ss
    assert ss["classic_qb_stack"] == 1
    assert n == 3


def test_reseed_noop_without_blob():
    assert persist.reseed_missing_scalars({}) == 0
    assert persist.reseed_missing_scalars({"_persist_blob": None}) == 0


def test_save_then_reseed_round_trip():
    # Full loop: Rules page edits -> save -> keys lost -> reseed restores.
    ss = {"classic_qb_stack": 1, "classic_bringback": "Required",
          "classic_min_salary": 48500}
    qp = {}
    persist.save(ss, qp)
    assert persist.PARAM in qp
    # Simulate the page navigation losing widget-backed keys.
    for k in ("classic_qb_stack", "classic_bringback", "classic_min_salary"):
        del ss[k]
    # restore() is a no-op here (same session, already restored) — this is
    # exactly why reseed exists as a separate step in boot_page.
    ss[persist._RESTORED_KEY] = True
    assert persist.restore(ss, qp) is False
    n = persist.reseed_missing_scalars(ss)
    assert n == 3
    assert ss["classic_qb_stack"] == 1
    assert ss["classic_bringback"] == "Required"
