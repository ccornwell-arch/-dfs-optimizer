"""Punt-eligibility floor + locked-captain supply (Oct 27, 2026).

Fix 1: players with effectively no role (the Kallerup/Winchester
mannequin pattern) must not be spent as FLEX/CPT filler — blocked in
generation, last-resort in portfolio selection, and NAMED when a
build genuinely cannot fill without one. Explicit user endorsement
(Lock / CPT Lock / Like / Core / a minimum) overrides the floor.
Legit cheap punts with real roles stay usable.

Fix 2: a CPT Lock build that stalls gets escalating solve
perturbation (hotter noise, then small bans on over-used flex
players, salary-band guarded) so the candidate supply does not
starve while legal lineups remain.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd

from dfs_lab.showdown import (
    _showdown_no_role_players,
    generate_showdown_lineups,
    select_showdown_portfolio,
)


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def _mk(rows):
    """rows: (id, name, pos, team, flex_sal, proj, own) tuples."""
    return pd.DataFrame({
        "ID": [r[0] for r in rows],
        "Name": [r[1] for r in rows],
        "CPT_NameID": [f"{r[1]} ({r[0]})" for r in rows],
        "FLEX_NameID": [f"{r[1]} ({r[0]})" for r in rows],
        "Position": [r[2] for r in rows],
        "Team": [r[3] for r in rows],
        "FlexSalary": [r[4] for r in rows],
        "CaptainSalary": [int(r[4] * 1.5) for r in rows],
        "My Proj": [r[5] for r in rows],
        "Aytia Proj": [r[5] for r in rows],
        "My Own": [r[6] for r in rows],
        "CPT Own": [r[6] * 0.5 for r in rows],
        "Game Info": ["TB@DAL"] * len(rows),
        "Away": ["TB"] * len(rows),
        "Home": ["DAL"] * len(rows),
        "Opponent": ["TB" if r[3] == "DAL" else "DAL" for r in rows],
        "is_QB": [r[2] == "QB" for r in rows],
        "is_RB": [r[2] == "RB" for r in rows],
        "is_WR": [r[2] == "WR" for r in rows],
        "is_TE": [r[2] == "TE" for r in rows],
        "is_DST": [r[2] == "DST" for r in rows],
        "is_K": [r[2] == "K" for r in rows],
        "is_passcatcher": [r[2] in ("WR", "TE") for r in rows],
        "ActiveForBuild": [True] * len(rows),
    })


def _pool(with_punt=True):
    rows = [
        ("100", "Dak Prescott", "QB", "DAL", 10800, 18.5, 25.0),
        ("101", "Javonte Williams", "RB", "DAL", 6200, 11.0, 18.0),
        ("102", "Hunter Luepke", "RB", "DAL", 3200, 4.0, 3.0),
        ("103", "CeeDee Lamb", "WR", "DAL", 10200, 16.2, 30.0),
        ("104", "George Pickens", "WR", "DAL", 8400, 13.5, 22.0),
        ("105", "Ryan Flournoy", "WR", "DAL", 4200, 6.0, 5.0),
        ("106", "Jake Ferguson", "TE", "DAL", 5600, 8.5, 12.0),
        ("107", "Brandon Aubrey", "K", "DAL", 4800, 8.0, 10.0),
        ("108", "Jalon Daniels", "QB", "TB", 7500, 12.0, 8.0),
        ("109", "Bucky Irving", "RB", "TB", 8200, 14.5, 20.0),
        ("110", "Sean Tucker", "RB", "TB", 3800, 5.5, 4.0),
        ("111", "Chris Godwin", "WR", "TB", 7800, 12.8, 16.0),
        ("112", "Emeka Egbuka", "WR", "TB", 6800, 10.2, 12.0),
        ("113", "Tez Johnson", "WR", "TB", 3600, 5.0, 4.0),
        ("114", "Cade Otton", "TE", "TB", 5200, 7.8, 9.0),
        ("115", "Chase McLaughlin", "K", "TB", 4600, 7.5, 9.0),
        # The mannequin: $200, no role, near-zero projection — and
        # near-zero OWNERSHIP, so the GPP leverage term makes him
        # attractive to the solver (the real-slate failure shape:
        # pre-fix, this build puts him in 3 of 5 lineups).
        ("116", "Mannequin", "TE", "TB", 200, 0.4, 0.1),
    ]
    if with_punt:
        # A legitimate $200 punt: same price, real (small) role —
        # but chalky, so leverage alone would dodge him.
        rows.append(("117", "Real Punt", "WR", "TB", 200, 2.4, 60.0))
    return _mk(rows)


def _ids(result):
    out = []
    for _, row in result.iterrows():
        out.append({str(row["CPT_ID"])} |
                   {str(row[f"FLEX{i}_ID"]) for i in range(1, 6)})
    return out


def _gen(df, sm, count=5, min_salary=30000, max_salary=44500,
         weights=None, min_unique=1, seed=42):
    return generate_showdown_lineups(
        df, 5000, "GPP / top-heavy", count, 800, min_salary, max_salary,
        weights or {"3-3": 0, "4-2": 0, "5-1": 1}, "Neutral", "", sm,
        "20-Max", 0, 80, 35, 2, 1, min_unique, seed, show_progress=False)


def _lock_map(df, locks, extra=None):
    sm = {}
    for pid in locks:
        sm[pid] = {"Lock": True}
    for pid, changes in (extra or {}).items():
        sm.setdefault(pid, {}).update(changes)
    return sm


# ------------------------------------------------------------------
# Classifier
# ------------------------------------------------------------------

def test_classifier_fallback_projection_only():
    df = _pool()
    nr = _showdown_no_role_players(df, {})
    _check("fallback: mannequin (proj 0.4) blocked", "116" in nr)
    _check("fallback: real punt (proj 2.4) safe", "117" not in nr)
    _check("fallback: Luepke (proj 4.0) safe", "102" not in nr)
    _check("fallback: kickers exempt even at low proj",
           all(df.loc[df["ID"] == pid, "Position"].iloc[0] != "K"
               for pid in nr))


def test_classifier_exp_columns():
    df = _pool()
    df["Exp Tgt"] = 5.0
    df["Exp Car"] = 1.0
    df["Exp Att"] = 0.0
    # Franks-like: tiny role AND tiny projection -> blocked.
    df.loc[df["ID"] == "116", ["Exp Tgt", "Exp Car"]] = [0.5, 0.1]
    # Sanders-like: real role, small projection -> safe.
    df.loc[df["ID"] == "117", ["Exp Tgt", "Exp Car"]] = [1.8, 0.1]
    df.loc[df["ID"] == "117", "My Proj"] = 2.38
    # Backup-QB-like: no targets/carries but real pass attempts -> safe.
    df.loc[df["ID"] == "108", ["Exp Tgt", "Exp Car", "Exp Att"]] = \
        [0.0, 0.2, 30.0]
    df.loc[df["ID"] == "108", "My Proj"] = 1.5
    nr = _showdown_no_role_players(df, {})
    _check("exp: Franks-like blocked", "116" in nr)
    _check("exp: Sanders-like safe", "117" not in nr)
    _check("exp: QB attempts count as role", "108" not in nr)


def test_classifier_endorsements_override():
    df = _pool()
    for label, strat in (
            ("Like", {"Priority": "Like"}),
            ("Core", {"Priority": "Core"}),
            ("Lock", {"Lock": True}),
            ("CPT Lock", {"CPT Lock": True}),
            ("Min Exposure", {"Min Exposure": 10}),
            ("CPT Min", {"CPT Min": 10})):
        nr = _showdown_no_role_players(df, {"116": strat})
        _check(f"endorsement overrides floor: {label}", "116" not in nr)


# ------------------------------------------------------------------
# Generator: the floor blocks when a clean fill exists; breaches are
# named when it is the only way to fill.
# ------------------------------------------------------------------

def _punt_band_strategy(df, punt_excluded):
    # Four DAL locks + a $44,500 cap under a 5-1-only construction:
    # the last two flex slots can only be Luepke + a $200 TB body.
    sm = _lock_map(df, ["100", "103", "104", "106"])
    if punt_excluded:
        sm["117"] = {"Exclude": True, "CPT Eligible": False}
        # Also out: every other TB body cheap enough to fill the
        # band (Tez, Tucker, McLaughlin) — the mannequin is the
        # ONLY legal completion left.
        for pid in ("113", "110", "115"):
            sm[pid] = {"Exclude": True, "CPT Eligible": False}
    return sm


def test_generator_floor_prefers_real_punt():
    df = _pool()
    sm = _punt_band_strategy(df, punt_excluded=False)
    res = _gen(df, sm, count=5)
    _check("floor: full count built", res is not None and len(res) == 5)
    used = _ids(res)
    _check("floor: mannequin in zero lineups",
           all("116" not in s for s in used))
    _check("floor: real punt used instead",
           any("117" in s for s in used))
    notes = res.attrs.get("showdown_notes") or []
    _check("floor: no breach note when clean fill exists",
           not any(n.startswith("Punt floor breached") for n in notes))


def test_generator_breach_is_named():
    df = _pool()
    sm = _punt_band_strategy(df, punt_excluded=True)
    res = _gen(df, sm, count=5)
    _check("breach: lineups still built", res is not None and len(res) >= 3)
    used = _ids(res)
    _check("breach: mannequin is the only fill and is used",
           all("116" in s for s in used))
    notes = res.attrs.get("showdown_notes") or []
    _check("breach: named in the build notes",
           any(n.startswith("Punt floor breached: Mannequin")
               for n in notes))


# ------------------------------------------------------------------
# Selection: dirty candidates are last-resort and named.
# ------------------------------------------------------------------

def _cand_frame(df, specs):
    """specs: list of 6-ID tuples (captain first), best-rated first."""
    name = {str(r["ID"]): str(r["Name"]) for _, r in df.iterrows()}
    rows = []
    for j, spec in enumerate(specs):
        row = {"CPT": name[spec[0]], "CPT_ID": spec[0],
               "Rating Score": 100.0 - j}
        for i in range(1, 6):
            row[f"FLEX{i}"] = name[spec[i]]
            row[f"FLEX{i}_ID"] = spec[i]
        rows.append(row)
    return pd.DataFrame(rows)


def test_selection_floor_and_breach_note():
    df = _pool()
    clean = [("103", "100", "109", "111", "101", "114"),
             ("103", "100", "109", "112", "102", "107"),
             ("100", "103", "104", "109", "111", "106"),
             ("109", "108", "111", "112", "103", "100"),
             ("111", "108", "109", "112", "110", "114"),
             ("104", "100", "103", "106", "109", "111")]
    dirty = [("103", "100", "109", "111", "101", "116"),
             ("100", "103", "104", "109", "111", "116"),
             ("109", "108", "111", "112", "103", "116"),
             ("111", "108", "109", "112", "110", "116")]
    cand = _cand_frame(df, dirty + clean)  # dirty rated highest
    out, notes = select_showdown_portfolio(cand, df, {}, 5, min_unique=1)
    used = _ids(out)
    _check("selection: clean five delivered over dirty top-rated",
           len(out) == 5 and all("116" not in s for s in used))
    _check("selection: no breach note when clean supply suffices",
           not any(n.startswith("Punt floor breached") for n in notes))
    out2, notes2 = select_showdown_portfolio(cand, df, {}, 8,
                                             min_unique=1)
    used2 = _ids(out2)
    n_dirty = sum(1 for s in used2 if "116" in s)
    _check("selection: dirty admitted only for the unfillable tail",
           len(out2) == 8 and n_dirty == 2)
    _check("selection: breach named with delivered counts",
           any(n.startswith("Punt floor breached: Mannequin")
               and "2 of 8" in n for n in notes2))


# ------------------------------------------------------------------
# Locked-captain supply: a CPT Lock build still fills its count.
# ------------------------------------------------------------------

def test_cpt_lock_build_still_fills():
    df = _pool()
    sm = {"110": {"CPT Lock": True}}
    res = _gen(df, sm, count=10, min_salary=30000, max_salary=50000,
               weights={"3-3": 1, "4-2": 1, "5-1": 0}, min_unique=2)
    _check("cpt-lock supply: full count returned",
           res is not None and len(res) == 10)
    _check("cpt-lock supply: locked captain captains every lineup",
           all(str(r["CPT_ID"]) == "110" for _, r in res.iterrows()))


if __name__ == "__main__":
    test_classifier_fallback_projection_only()
    test_classifier_exp_columns()
    test_classifier_endorsements_override()
    test_generator_floor_prefers_real_punt()
    test_generator_breach_is_named()
    test_selection_floor_and_breach_note()
    test_cpt_lock_build_still_fills()
    print("All punt-floor / locked-captain supply tests passed.")
