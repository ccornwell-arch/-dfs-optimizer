"""Flex QB-linkage coherence guard (Oct 28, 2026).

Chris's rule: three pass catchers for a team can't "get there"
without the quarterback. Before this guard, the QB<->catcher
linkage was only checked at the CAPTAIN slot, so a delivered
lineup carrying three Carolina catchers with no Bryce Young froze
as "Coherence Score: 100.0 — No major football contradictions."

The guard:
  - 3+ pass catchers from one team with no QB from that team
    (captain counts toward his team) = HARD incoherence — blocked
    in-solve, rejected by coherence, last-resort in selection, and
    named whenever one is delivered anyway;
  - exactly 2 catchers, no QB = a named coherence penalty only
    (a legitimate construction);
  - the user's own settings outrank the guard: catchers he locked,
    or a QB he ruled out while backing the catchers, exempt the
    team — delivered shapes are named as overrides, never silent,
    and never scored as clean.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd

from dfs_lab.showdown import (
    _showdown_linkage_breaches,
    _showdown_qb_linkage_exemptions,
    generate_showdown_lineups,
    select_showdown_portfolio,
    showdown_lineup_coherence,
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


def _pool():
    return _mk([
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
    ])


def _chosen(df, cpt_id, flex_ids):
    idx = {str(r["ID"]): i for i, r in df.iterrows()}
    out = [("CPT", idx[str(cpt_id)])]
    out += [(f"FLEX{j}", idx[str(pid)])
            for j, pid in enumerate(flex_ids, start=1)]
    return out


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


# ------------------------------------------------------------------
# Coherence: the three shapes.
# ------------------------------------------------------------------

def test_coherence_three_catchers_no_qb_hard_reject():
    df = _pool()
    # CPT Javonte (RB) + Lamb/Pickens/Ferguson: three DAL catchers,
    # no Dak anywhere — Chris's Lineup 2 shape.
    coh = showdown_lineup_coherence(
        df, _chosen(df, "101", ["103", "104", "106", "109", "110"]))
    _check("3+ catchers no QB: rejected", coh["Accept"] is False)
    _check("3+ catchers no QB: flag names the QB",
           "3 DAL pass catchers without Dak Prescott"
           in coh["Coherence Flags"])
    _check("3+ catchers no QB: flag states the rule",
           "can't all get there without their quarterback"
           in coh["Coherence Flags"])
    _check("3+ catchers no QB: score can no longer read 100",
           coh["Coherence Score"] < 100.0)


def test_coherence_two_catchers_penalty_named_not_blocked():
    df = _pool()
    coh = showdown_lineup_coherence(
        df, _chosen(df, "101", ["103", "106", "102", "109", "111"]))
    _check("2 catchers no QB: still accepted", coh["Accept"] is True)
    _check("2 catchers no QB: penalty is exactly the named -8",
           coh["Coherence Score"] == 92.0)
    _check("2 catchers no QB: flag names QB and count",
           coh["Coherence Flags"]
           == "2 DAL pass catchers without Dak Prescott")


def test_coherence_qb_present_clean():
    df = _pool()
    coh = showdown_lineup_coherence(
        df, _chosen(df, "101", ["100", "103", "104", "106", "109"]))
    _check("3 catchers WITH their QB: accepted", coh["Accept"] is True)
    _check("3 catchers WITH their QB: clean label",
           coh["Coherence Flags"] == "No major football contradictions"
           and coh["Coherence Score"] == 100.0)


def test_coherence_captain_catcher_counts_toward_team():
    df = _pool()
    # CPT Lamb + three more DAL catchers in flex, still no Dak.
    coh = showdown_lineup_coherence(
        df, _chosen(df, "103", ["104", "106", "105", "109", "110"]))
    _check("WR captain counts: 4 DAL catchers no QB rejected",
           coh["Accept"] is False
           and "4 DAL pass catchers without Dak Prescott"
           in coh["Coherence Flags"])


def test_coherence_exempt_shape_delivered_named_not_clean():
    df = _pool()
    reason = ("you locked 3 DAL pass catchers "
              "(CeeDee Lamb, George Pickens, Jake Ferguson)")
    coh = showdown_lineup_coherence(
        df, _chosen(df, "101", ["103", "104", "106", "109", "110"]),
        exempt_teams={"DAL": reason})
    _check("exempt 3+ shape: delivered", coh["Accept"] is True)
    _check("exempt 3+ shape: flag carries the override reason",
           "delivered because you locked 3 DAL pass catchers"
           in coh["Coherence Flags"])
    _check("exempt 3+ shape: score is not the clean 100",
           coh["Coherence Score"] == 60.0)


# ------------------------------------------------------------------
# Exemptions: his settings construct the shape, or they don't.
# ------------------------------------------------------------------

def test_exemptions():
    df = _pool()
    ex = _showdown_qb_linkage_exemptions(df, {
        "103": {"Lock": True}, "104": {"Lock": True},
        "106": {"Lock": True}})
    _check("exempt: 3 locked catchers exempt their team",
           "DAL" in ex and "you locked 3 DAL pass catchers" in ex["DAL"])
    _check("exempt: other team untouched", "TB" not in ex)

    ex = _showdown_qb_linkage_exemptions(df, {
        "100": {"Exclude": True, "Priority": "Exclude"},
        "103": {"Priority": "Core"}})
    _check("exempt: QB ruled out + catcher backed = exempt",
           "DAL" in ex and "Dak Prescott is ruled out" in ex["DAL"])

    ex = _showdown_qb_linkage_exemptions(df, {
        "100": {"Exclude": True, "Priority": "Exclude"}})
    _check("exempt: bare QB exclusion alone is NOT an exemption",
           "DAL" not in ex)

    ex = _showdown_qb_linkage_exemptions(df, {})
    _check("exempt: no settings, no exemptions", ex == {})


# ------------------------------------------------------------------
# Generator: the shape is blocked; his override ships named.
# ------------------------------------------------------------------

def test_generator_blocks_shape():
    df = _pool()
    res = _gen(df, {})
    _check("generator: full count still built",
           res is not None and len(res) == 5)
    used = _ids(res)
    _check("generator: zero delivered 3+ no-QB shapes",
           all(not _showdown_linkage_breaches(df, sorted(s))
               for s in used))
    # Any delivered exactly-2 no-QB lineup must carry its penalty.
    team_of = {str(r["ID"]): str(r["Team"]) for _, r in df.iterrows()}
    pc_of = {str(r["ID"]): bool(r["is_passcatcher"])
             for _, r in df.iterrows()}
    qb_of = {str(r["ID"]): bool(r["is_QB"]) for _, r in df.iterrows()}
    ok = True
    for _, row in res.iterrows():
        ids = [str(row["CPT_ID"])] + \
              [str(row[f"FLEX{i}_ID"]) for i in range(1, 6)]
        for team in ("DAL", "TB"):
            n = sum(1 for p in ids
                    if team_of[p] == team and pc_of[p])
            has_qb = any(team_of[p] == team and qb_of[p] for p in ids)
            if n == 2 and not has_qb:
                if (f"2 {team} pass catchers without"
                        not in str(row["Coherence Flags"])):
                    ok = False
    _check("generator: delivered 2-catcher lineups carry the penalty",
           ok)


def test_generator_user_override_named():
    df = _pool()
    sm = {"103": {"Lock": True}, "104": {"Lock": True},
          "106": {"Lock": True},
          "100": {"Exclude": True, "Priority": "Exclude"}}
    res = _gen(df, sm)
    _check("override: full count built", res is not None and len(res) == 5)
    used = _ids(res)
    _check("override: his locks hold in every lineup",
           all({"103", "104", "106"} <= s for s in used))
    _check("override: excluded QB stays out",
           all("100" not in s for s in used))
    notes = res.attrs.get("showdown_notes") or []
    _check("override: named in the build warnings",
           any(n.startswith("QB-linkage guard overridden by your "
                            "settings")
               and "3+ DAL pass catchers without Dak Prescott" in n
               and "you locked" in n for n in notes))
    _check("override: lineup flags carry the reason, scores not clean",
           all("delivered because" in str(r["Coherence Flags"])
               and float(r["Coherence Score"]) < 100.0
               for _, r in res.iterrows()))


# ------------------------------------------------------------------
# Selection: breach candidates are last-resort and named.
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


def test_selection_guard_last_resort_and_named():
    df = _pool()
    breach = ("101", "103", "104", "106", "109", "110")  # 3 DAL, no Dak
    clean = [("100", "103", "104", "106", "109", "111"),
             ("109", "100", "103", "111", "112", "107"),
             ("111", "108", "112", "114", "101", "105")]
    cand = _cand_frame(df, [breach] + clean)  # breach rated highest
    out, notes = select_showdown_portfolio(cand, df, {}, 3,
                                           min_unique=1)
    used = _ids(out)
    _check("selection: clean three delivered over top-rated breach",
           len(out) == 3
           and all(not _showdown_linkage_breaches(df, sorted(s))
                   for s in used))
    _check("selection: no breach note when clean supply suffices",
           not any(n.startswith("QB-linkage guard breached")
                   for n in notes))
    out2, notes2 = select_showdown_portfolio(cand, df, {}, 4,
                                             min_unique=1)
    _check("selection: breach admitted only for the unfillable tail",
           len(out2) == 4
           and sum(1 for s in _ids(out2)
                   if _showdown_linkage_breaches(df, sorted(s))) == 1)
    _check("selection: delivered breach named with counts",
           any(n.startswith("QB-linkage guard breached")
               and "1 of 4" in n and "Dak Prescott" in n
               for n in notes2))


if __name__ == "__main__":
    test_coherence_three_catchers_no_qb_hard_reject()
    test_coherence_two_catchers_penalty_named_not_blocked()
    test_coherence_qb_present_clean()
    test_coherence_captain_catcher_counts_toward_team()
    test_coherence_exempt_shape_delivered_named_not_clean()
    test_exemptions()
    test_generator_blocks_shape()
    test_generator_user_override_named()
    test_selection_guard_last_resort_and_named()
    print("All QB-linkage guard tests passed.")
