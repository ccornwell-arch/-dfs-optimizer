"""Regression test for the Oct 8, 2026 Dak Prescott incident.

The status strip listed Dak Prescott as Out while his table checkbox was
unchecked. Root cause: unchecking the Out box and tapping Apply saved
{Exclude: False, Priority: "Exclude"} — the stale "Exclude" lean was never
cleared. Both the strip and the solver treat Priority="Exclude" as an
exclusion, so the player stayed out of builds.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dfs_lab.ui.main import resolve_saved_priority


def _strip_lists_as_out(entry):
    """Mirror of the status-strip predicate in main.py."""
    return bool(entry.get("Exclude") or entry.get("Priority") == "Exclude")


def _solver_excludes(entry):
    """Mirror of the showdown solver's exclusion predicate."""
    return bool(entry.get("Exclude") or entry.get("Priority") == "Exclude"
                or entry.get("ActiveForBuild") is False)


def test_uncheck_out_clears_stale_exclude_priority():
    # Quick Lock Out writes both flags...
    saved = {"Lock": False, "CPT Lock": False, "Exclude": True,
             "CPT Eligible": False, "Priority": "Exclude"}
    assert _strip_lists_as_out(saved)
    # ...user unchecks the Out box in the table and taps Apply.
    # The table's Priority cell still shows the stale "Exclude" lean.
    ex = False
    table_priority = "Exclude"
    saved["Exclude"] = ex
    saved["Priority"] = resolve_saved_priority(ex, table_priority)
    assert saved == {"Lock": False, "CPT Lock": False, "Exclude": False,
                     "CPT Eligible": False, "Priority": "Neutral"}
    assert not _strip_lists_as_out(saved)
    assert not _solver_excludes(saved)


def test_check_out_still_excludes():
    assert resolve_saved_priority(True, "Neutral") == "Exclude"
    assert resolve_saved_priority(True, "Like") == "Exclude"


def test_deliberate_lean_survives_apply():
    # A Core/Like/Fade lean is untouched when the Out box stays unchecked.
    assert resolve_saved_priority(False, "Core") == "Core"
    assert resolve_saved_priority(False, "Like") == "Like"
    assert resolve_saved_priority(False, "Fade") == "Fade"
    assert resolve_saved_priority(False, "Neutral") == "Neutral"


def test_quick_lock_lock_clears_stale_exclude():
    # Quick Lock "Lock" after a Quick Lock "Out": Exclude goes False and the
    # stale Priority must not linger.
    entry = {"Lock": False, "CPT Lock": False, "Exclude": True,
             "CPT Eligible": False, "Priority": "Exclude"}
    entry.update({"Lock": True, "CPT Lock": False, "Exclude": False})
    entry["Priority"] = resolve_saved_priority(False, entry.get("Priority"))
    assert not _strip_lists_as_out(entry)
    assert not _solver_excludes(entry)
    assert entry["Lock"] is True
