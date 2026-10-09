"""Tests for apply_override_rescue — the silent-override trust bug.

Incident: on TNF 10/08/2026 (TB @ DAL) the user typed a projection for Jalon
Daniels to force him into the showdown pool, but Daniels had been
auto-excluded (backup-QB guard picked the wrong primary) and the override
updated the number while leaving ActiveForBuild=False — so the build silently
ignored it. A typed projection is the strongest explicit user take; it must
either rescue the player from model-driven exclusions or say loudly why not.
"""
import pandas as pd

from dfs_lab.data import apply_override_rescue, apply_football_reality_guard


def _row(name, pid, reason, role="Backup QB", active=False):
    return {
        "ID": pid, "Name": name, "Position": "QB", "Team": "TB",
        "FlexSalary": 8000, "My Proj": 0.0,
        "ActiveForBuild": active, "Role Confidence": role,
        "Auto Excluded Reason": reason, "Live Status": "Not verified",
    }


def _df(rows):
    return pd.DataFrame(rows)


def test_rescues_backup_qb_exclusion():
    df = _df([_row("Jalon Daniels", "d1", "Backup QB — Aytia keeps only the primary QB active by default")])
    out, rescued, blocked = apply_override_rescue(df, {"d1": 12.5})
    assert bool(out.loc[0, "ActiveForBuild"]) is True
    assert out.loc[0, "Auto Excluded Reason"] == ""
    assert out.loc[0, "Role Confidence"] not in ("Backup QB", "Inactive / no usable projection")
    assert [n for n, _ in rescued] == ["Jalon Daniels"]
    assert blocked == []


def test_rescues_no_usable_projection():
    df = _df([_row("Deep Punt", "d2", "No usable projection — auto-excluded before the build",
                   role="Inactive / no usable projection")])
    out, rescued, blocked = apply_override_rescue(df, {"d2": 8.0})
    assert bool(out.loc[0, "ActiveForBuild"]) is True
    assert [n for n, _ in rescued] == ["Deep Punt"]
    assert blocked == []


def test_user_out_always_wins():
    df = _df([_row("Baker Mayfield", "d3", "Backup QB — Aytia keeps only the primary QB active by default")])
    out, rescued, blocked = apply_override_rescue(df, {"d3": 20.0}, {"d3": {"Exclude": True}})
    assert bool(out.loc[0, "ActiveForBuild"]) is False
    assert rescued == []
    assert len(blocked) == 1 and "Out" in blocked[0][1]


def test_user_out_via_priority_exclude_wins():
    df = _df([_row("Baker Mayfield", "d3", "Backup QB — Aytia keeps only the primary QB active by default")])
    out, rescued, blocked = apply_override_rescue(df, {"d3": 20.0}, {"d3": {"Priority": "Exclude"}})
    assert bool(out.loc[0, "ActiveForBuild"]) is False
    assert rescued == [] and len(blocked) == 1


def test_official_out_is_not_rescued():
    df = _df([_row("Hurt Guy", "d4", "Official injury report: OUT", role="Unavailable — OUT")])
    out, rescued, blocked = apply_override_rescue(df, {"d4": 15.0})
    assert bool(out.loc[0, "ActiveForBuild"]) is False
    assert rescued == []
    assert len(blocked) == 1 and "OUT" in blocked[0][1]


def test_live_roster_block_is_not_rescued():
    df = _df([_row("Suspended Guy", "d5", "Live roster status: SUS", role="Unavailable — live roster")])
    out, rescued, blocked = apply_override_rescue(df, {"d5": 15.0})
    assert bool(out.loc[0, "ActiveForBuild"]) is False
    assert rescued == [] and len(blocked) == 1


def test_no_roster_record_is_not_rescued():
    df = _df([_row("Free Agent", "d6", "No 2026 NFL roster record — player is not on a team",
                   role="Unavailable — not on a roster")])
    out, rescued, blocked = apply_override_rescue(df, {"d6": 15.0})
    assert bool(out.loc[0, "ActiveForBuild"]) is False
    assert rescued == [] and len(blocked) == 1


def test_no_override_leaves_exclusion_untouched():
    df = _df([_row("Jalon Daniels", "d1", "Backup QB — Aytia keeps only the primary QB active by default")])
    out, rescued, blocked = apply_override_rescue(df, {})
    assert bool(out.loc[0, "ActiveForBuild"]) is False
    assert rescued == [] and blocked == []


def test_zero_override_does_not_rescue():
    df = _df([_row("Jalon Daniels", "d1", "Backup QB — Aytia keeps only the primary QB active by default")])
    out, rescued, blocked = apply_override_rescue(df, {"d1": 0.0})
    assert bool(out.loc[0, "ActiveForBuild"]) is False
    assert rescued == [] and blocked == []


def test_override_on_active_player_is_noop():
    df = _df([_row("Dak Prescott", "d7", "", role="Primary QB", active=True)])
    out, rescued, blocked = apply_override_rescue(df, {"d7": 22.0})
    assert bool(out.loc[0, "ActiveForBuild"]) is True
    assert rescued == [] and blocked == []


def test_daniels_incident_end_to_end():
    """The real incident: guard excludes the backup QB, the user's typed
    projection must put him back in the pool."""
    df = pd.DataFrame([
        {"ID": "mayfield", "Name": "Baker Mayfield", "Position": "QB", "Team": "TB",
         "FlexSalary": 10500, "My Proj": 0.0, "AvgPointsPerGame": 0.0},
        {"ID": "trask", "Name": "Kyle Trask", "Position": "QB", "Team": "TB",
         "FlexSalary": 8000, "My Proj": 9.0, "AvgPointsPerGame": 2.0},
        {"ID": "daniels", "Name": "Jalon Daniels", "Position": "QB", "Team": "TB",
         "FlexSalary": 7500, "My Proj": 3.0, "AvgPointsPerGame": 1.0},
    ])
    guarded = apply_football_reality_guard(df, "FlexSalary", "My Proj")
    # Trask wins the primary slot on salary; Daniels is auto-excluded.
    daniels = guarded[guarded["ID"] == "daniels"].iloc[0]
    assert bool(daniels["ActiveForBuild"]) is False
    assert "Backup QB" in str(daniels["Auto Excluded Reason"])

    out, rescued, blocked = apply_override_rescue(guarded, {"daniels": 14.0})
    back = out[out["ID"] == "daniels"].iloc[0]
    assert bool(back["ActiveForBuild"]) is True
    assert [n for n, _ in rescued] == ["Jalon Daniels"]
    assert blocked == []
    # Mayfield had no usable projection and no override: still out.
    may = out[out["ID"] == "mayfield"].iloc[0]
    assert bool(may["ActiveForBuild"]) is False
