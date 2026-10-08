"""Tests for availability_freshness_note and the guard's stale-snapshot fail-open.

Incident: Baker Mayfield was ruled OUT for TNF (TB @ DAL, 10/08/2026) on Wed
10/07, after nflverse's week-5 injury snapshot. The guard saw report_status=None
and failed open, so Mayfield stayed in the build pool (even as CPT). The guard
behavior is by design (it can't know what the data doesn't say); the freshness
note is the mitigation — it must fire whenever a slate game is today/tomorrow.
"""
import datetime
import pandas as pd

from dfs_lab.data import availability_freshness_note, apply_live_availability_guard
import dfs_lab.data as data_mod


def _loader_ok(monkeypatch):
    monkeypatch.setattr(data_mod, "_load_live_nfl_availability",
                         lambda season=2026: (pd.DataFrame(), pd.DataFrame()))


def _loader_down(monkeypatch):
    def _boom(season=2026):
        raise RuntimeError("no network")
    monkeypatch.setattr(data_mod, "_load_live_nfl_availability", _boom)


def _slate_df(game_info="TB@DAL 10/08/2026 08:15PM ET"):
    return pd.DataFrame([{
        "Name": "Baker Mayfield", "Position": "QB", "Team": "TB", "Salary": 10000,
        "Roster Position": "CPT", "Game Info": game_info,
    }])


def test_warns_when_game_is_tomorrow(monkeypatch):
    _loader_ok(monkeypatch)
    note = availability_freshness_note(_slate_df(), today_et=datetime.date(2026, 10, 7))
    assert note is not None
    assert "TB @ DAL" in note
    assert "tomorrow" in note
    assert "will NOT be auto-excluded" in note


def test_warns_when_game_is_today(monkeypatch):
    _loader_ok(monkeypatch)
    note = availability_freshness_note(_slate_df(), today_et=datetime.date(2026, 10, 8))
    assert note is not None
    assert "today" in note


def test_silent_when_games_are_next_week(monkeypatch):
    _loader_ok(monkeypatch)
    note = availability_freshness_note(_slate_df(), today_et=datetime.date(2026, 10, 1))
    assert note is None


def test_silent_without_game_info(monkeypatch):
    _loader_ok(monkeypatch)
    df = _slate_df().drop(columns=["Game Info"])
    assert availability_freshness_note(df, today_et=datetime.date(2026, 10, 7)) is None


def test_warns_when_live_data_unreachable(monkeypatch):
    _loader_down(monkeypatch)
    note = availability_freshness_note(_slate_df(), today_et=datetime.date(2026, 10, 1))
    assert note is not None
    assert "couldn't reach live roster/injury data" in note


def _mock_snapshot(monkeypatch, report_status):
    roster = pd.DataFrame([{
        "full_name": "Baker Mayfield", "team": "TB", "week": 5, "status": "ACT",
    }])
    inj = pd.DataFrame([{
        "full_name": "Baker Mayfield", "team": "TB", "week": 5,
        "report_status": report_status, "practice_status": "Did Not Participate In Practice",
    }])
    monkeypatch.setattr(data_mod, "_load_live_nfl_availability",
                         lambda season=2026: (roster, inj))


def test_guard_fails_open_on_stale_snapshot(monkeypatch):
    """The incident: week-5 snapshot says None/DNP, real status is OUT.
    The guard cannot know — it must fail open (this is why the note exists)."""
    _mock_snapshot(monkeypatch, None)
    out = apply_live_availability_guard(_slate_df())
    assert bool(out.loc[0, "ActiveForBuild"]) is True
    reason = out["Auto Excluded Reason"].iloc[0] if "Auto Excluded Reason" in out.columns else ""
    assert str(reason).strip() == ""


def test_guard_excludes_official_out(monkeypatch):
    """Positive control: once the snapshot says OUT, the guard excludes."""
    _mock_snapshot(monkeypatch, "Out")
    out = apply_live_availability_guard(_slate_df())
    assert bool(out.loc[0, "ActiveForBuild"]) is False
    assert "OUT" in str(out.loc[0, "Auto Excluded Reason"])
