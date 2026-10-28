"""WS26 Piece 2 — the showdown self-audit (dfs_lab/audit.py).

The audit argues against every delivered lineup in the worlds where
that lineup is trying to be right: a challenger is the best-legal
WHOLE construction re-solved against those home worlds, and a
lineup is beaten only decisively (>=60% of its home worlds, >= +2.0
average margin, challenger sim mean within 1.0). Disposition is
replace-by-default; flag-only when replacement would break the
user's settings, duplicate another delivered lineup, or the shape
is the user's own construction. The audit fails LOUD: no worlds ->
a named "Audit: not run" state; a zeroed budget -> partial
language with Tier 1 verdicts standing.

Worlds here are engineered deterministically (base = projection,
plus a TB passing spike in 40% of worlds where Baker out-spikes his
own receivers) so the domination arithmetic is exact, not sampled.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from dfs_lab.audit import audit_showdown_set
from dfs_lab.showdown import (add_showdown_ratings,
                              showdown_lineup_coherence,
                              showdown_lineup_details)
from test_showdown_world_selection import _pool


def _raw_row(pool, cpt_id, flex_ids, strategy=None):
    """Assemble a DELIVERED row without the coherence gate — the
    audit's inputs are sets as delivered (including pre-guard
    shapes the candidate path would now reject)."""
    d = pool.reset_index(drop=True)
    pos = {str(pid): i for i, pid in enumerate(d["ID"].astype(str))}
    chosen = [("CPT", pos[cpt_id])] + [
        (f"FLEX{j}", pos[p]) for j, p in enumerate(flex_ids, start=1)]
    coherence = showdown_lineup_coherence(d, chosen)
    detail = showdown_lineup_details(d, chosen, strategy or {},
                                      "Neutral", "")
    detail.update({
        "Coherence Score": coherence["Coherence Score"],
        "Lineup Story": coherence["Lineup Story"],
        "Coherence Flags": coherence["Coherence Flags"]})
    row = dict(detail)
    for slot, i in chosen:
        row[slot] = d.loc[i, "Name"]
        row[slot + "_ID"] = str(d.loc[i, "ID"])
        row[slot + "_NameID"] = (d.loc[i, "CPT_NameID"] if slot == "CPT"
                                 else d.loc[i, "FLEX_NameID"])
    return row


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def _worlds(pool, n=1000):
    """Base = My Proj everywhere; TB pass spike in 40% of worlds:
    Baker +35, his three receivers +20 each (Baker out-spikes the
    catcher he replaces — the QB construction wins those worlds)."""
    W = np.tile(pool["My Proj"].to_numpy(float), (n, 1))
    pos = {str(pid): i for i, pid in enumerate(pool["ID"].astype(str))}
    tb = slice(0, int(n * 0.4))
    W[tb, pos["109"]] += 35.0
    for pid in ("111", "112", "113"):
        W[tb, pos[pid]] += 20.0
    return pd.DataFrame(W, columns=[str(x) for x in pool["ID"]])


def _settings(**kw):
    s = {"min_salary": 44000, "max_salary": 50000, "max_k": 2,
         "max_dst": 1, "cpt_qb_passcatchers": None,
         "wrte_cpt_qb_pair_pct": None, "min_unique": 1,
         "script": "Neutral", "script_team": "", "team_scores": None,
         "story": None, "cores": [], "entry_format": "Single Entry",
         "field_size": 2000, "payout_style": "Top-heavy payouts"}
    s.update(kw)
    return s


def _delivered(pool, specs, strategy=None):
    rows = [_raw_row(pool, cpt, flex, strategy) for cpt, flex in specs]
    out = add_showdown_ratings(pd.DataFrame(rows), 0.3)
    out = out.sort_values("Rating Score", ascending=False,
                          kind="mergesort").reset_index(drop=True)
    out["Rank"] = range(1, len(out) + 1)
    return out


# CPT Javonte + all three TB catchers, no Baker (the Oct 28 shape,
# generalized): its home worlds are the TB-pass worlds, where the
# same construction WITH Baker beats it.
BAD = ("101", ("111", "112", "113", "103", "106"))
GOOD = ("100", ("103", "106", "111", "110", "107"))
MID = ("103", ("100", "104", "110", "114", "115"))


def test_domination_replaces():
    pool = _pool()
    result = _delivered(pool, [BAD, GOOD, MID])
    final, audit = audit_showdown_set(
        pool, {}, result, worlds_df=_worlds(pool), sim_frame=pool,
        settings=_settings(), challengers=[])
    _check("audit ran", audit["ran"] is True)
    verdicts = {e["lineup"]: e["verdict"] for e in audit["per_lineup"]}
    bad_idx = next(e["lineup"] for e in audit["per_lineup"]
                   if e["captain"] == "Javonte Williams")
    _check("the no-QB 3-catcher lineup is replaced",
           verdicts[bad_idx] == "replaced")
    _check("audit line reports the replacement",
           "1 replaced" in audit["line"])
    rows = [r for _, r in final.iterrows()
            if str(r["CPT"]) == "Javonte Williams"]
    flex_names = ({str(rows[0][c]) for c in
                   ("FLEX1", "FLEX2", "FLEX3", "FLEX4", "FLEX5")}
                  if rows else set())
    _check("the replacement is a QB construction (Baker in, "
           ">=2 TB catchers kept)",
           len(rows) == 1 and "Baker Mayfield" in flex_names
           and len({"Mike Evans", "Chris Godwin", "Emeka Egbuka"}
                   & flex_names) >= 2)
    _check("delivered count preserved", len(final) == 3)
    _check("replacement salary inside the band",
           bool(rows) and 44000 <= int(rows[0]["Salary"]) <= 50000)
    others = [v for k, v in verdicts.items() if k != bad_idx]
    _check("sound lineups stay clean", all(v == "clean" for v in others))


def test_flag_when_replacement_breaks_settings():
    pool = _pool()
    result = _delivered(pool, [BAD, GOOD, MID])
    strategy = {"109": {"Max Exposure": 0}}
    final, audit = audit_showdown_set(
        pool, strategy, result, worlds_df=_worlds(pool),
        sim_frame=pool, settings=_settings(), challengers=[])
    bad = next(e for e in audit["per_lineup"]
               if e["captain"] == "Javonte Williams")
    _check("beaten lineup whose only winning challenger breaks a "
           "Max cap is flagged, not replaced",
           bad["verdict"] == "flagged")
    _check("the flag names the binding setting",
           any("Max Exposure" in d for d in audit["details"]))
    sets = [{str(r[c]) for c in ("CPT", "FLEX1", "FLEX2", "FLEX3",
                                 "FLEX4", "FLEX5")}
            for _, r in final.iterrows()]
    _check("the original lineup is still delivered",
           any("Mike Evans" in s and "Baker Mayfield" not in s
               for s in sets))


def test_user_constructed_shape_is_an_override():
    pool = _pool()
    strategy = {pid: {"Lock": True} for pid in ("111", "112", "113")}
    # Every delivered lineup carries the three locked catchers (a
    # set where only one does would itself break the locks).
    good2 = ("100", ("111", "112", "113", "110", "102"))
    mid2 = ("103", ("111", "112", "113", "110", "102"))
    result = _delivered(pool, [BAD, good2, mid2], strategy)
    final, audit = audit_showdown_set(
        pool, strategy, result, worlds_df=_worlds(pool),
        sim_frame=pool, settings=_settings(), challengers=[])
    bad = next(e for e in audit["per_lineup"]
               if e["captain"] == "Javonte Williams")
    _check("a shape the user locked is delivered as an override, "
           "never a catch", bad["verdict"] == "override")
    _check("overrides are counted separately",
           audit["counts"]["overrides"] >= 1
           and audit["counts"]["replaced"] == 0)
    _check("the audit line names the settings carve-out",
           "by your settings" in audit["line"])


def test_coherence_penalties_and_partial_budget():
    pool = _pool()
    two_catchers = ("101", ("111", "112", "103", "106", "110"))
    bare_qb = ("103", ("109", "101", "104", "110", "107"))
    result = _delivered(pool, [two_catchers, bare_qb, GOOD])
    final, audit = audit_showdown_set(
        pool, {}, result, worlds_df=_worlds(pool), sim_frame=pool,
        settings=_settings(), challengers=[], budget_s=0.0)
    by_cpt = {e["captain"]: e for e in audit["per_lineup"]}
    _check("exactly-2 catchers without the QB is a C2 penalty",
           "C2" in by_cpt["Javonte Williams"]["checks"]
           and by_cpt["Javonte Williams"]["verdict"] == "penalized")
    _check("a bare QB flex (no own-team catcher) is a C3 penalty",
           "C3" in by_cpt["CeeDee Lamb"]["checks"]
           and by_cpt["CeeDee Lamb"]["verdict"] == "penalized")
    _check("a zeroed budget yields partial language, verdicts stand",
           audit["line"].startswith("Audit: partial")
           and audit["counts"]["unaudited"] > 0)


def test_c9_contradictory_requirements():
    pool = _pool()
    n = 1000
    W = np.zeros((n, len(pool))) + 2.0
    pos = {str(pid): i for i, pid in enumerate(pool["ID"].astype(str))}
    W[:, pos["103"]] = 10.0
    W[: n // 2, pos["103"]] += 30.0      # Lamb spikes in half 1
    W[:, pos["111"]] = 15.0
    W[n // 2:, pos["111"]] += 25.0      # Evans spikes in half 2 only
    worlds = pd.DataFrame(W, columns=[str(x) for x in pool["ID"]])
    contra = ("103", ("111", "101", "110", "106", "114"))
    weird_ok = ("107", ("108", "116", "102", "105", "113"))
    result = _delivered(pool, [contra, weird_ok])
    final, audit = audit_showdown_set(
        pool, {}, result, worlds_df=worlds, sim_frame=pool,
        settings=_settings(), challengers=[])
    by_cpt = {e["captain"]: e for e in audit["per_lineup"]}
    _check("a lineup needing two mutually exclusive spikes is "
           "caught by C9 (general story coherence)",
           "C9" in by_cpt["CeeDee Lamb"]["checks"])
    _check("a coherent-but-weird lineup (K captain, both DSTs, "
           "steady pieces) is NOT caught",
           by_cpt["Brandon Aubrey"]["verdict"] == "clean")


def test_not_run_is_named():
    pool = _pool()
    result = _delivered(pool, [BAD, GOOD, MID])
    final, audit = audit_showdown_set(
        pool, {}, result, worlds_df=None, sim_frame=None,
        settings=_settings(), challengers=[])
    _check("no worlds -> named not-run state, set untouched",
           audit["ran"] is False
           and audit["line"].startswith("Audit: not run")
           and len(final) == 3)


if __name__ == "__main__":
    test_domination_replaces()
    test_flag_when_replacement_breaks_settings()
    test_user_constructed_shape_is_an_override()
    test_coherence_penalties_and_partial_budget()
    test_c9_contradictory_requirements()
    test_not_run_is_named()
    print("ALL PASS")
