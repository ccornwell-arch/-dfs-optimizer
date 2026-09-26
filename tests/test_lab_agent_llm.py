"""Tests for the LLM-backed post-build Lab agent.

Covers:
  (a) postbuild_lineups_context serializer shape + token budget on the synthetic pipeline.
  (b) No API key -> falls back to the legacy template answer (unchanged behavior).
  (c) Fake API key + mocked OpenAI client -> prompt contains portfolio data,
      the question, and history; output_text is returned; model recorded.
  (d) The full existing suites still pass (run separately).

No real API key is needed; the OpenAI client is fully mocked.
Run from repo root:  python3 tests/test_lab_agent_llm.py
"""
import os
import sys
from unittest import mock

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

import pandas as pd

from test_results_command_center import build_pipeline
from dfs_lab.classic import (
    classic_postbuild_answer,
    classic_postbuild_llm_answer,
    _DFS_PRO_PLAYBOOK,
)
from dfs_lab.ui.results import postbuild_lineups_context


def make_packet():
    return {
        "portfolio": {
            "built": True,
            "lineups": 8,
            "unique_qbs": 3,
            "qb_usage": [
                {"qb": "KC Quarterback", "exposure_pct": 50.0},
                {"qb": "BUF Quarterback", "exposure_pct": 37.5},
            ],
            "stack_mix": {"QB+2": 5, "QB+1": 3},
            "bringback_mix": {"yes": 6, "no": 2},
            "flex_mix_pct": {"RB": 40.0, "WR": 40.0, "TE": 20.0},
            "salary_left": {"mean": 200.0, "median": 150.0},
            "avg_player_ownership": {"mean": 10.0},
            "top_exposures": [
                {"player": "KC Quarterback", "exposure_pct": 50.0,
                 "field_own_pct": 14.0, "leverage_pct": 36.0},
            ],
            "most_overweight": [
                {"player": "KC Quarterback", "exposure_pct": 50.0, "field_own_pct": 14.0},
            ],
            "most_underweight": [
                {"player": "MIA Quarterback", "exposure_pct": 0.0, "field_own_pct": 8.0},
            ],
            "repeated_pairs": [
                {"players": ["KC Quarterback", "KC Wide1"], "portfolio_pct": 37.5},
            ],
        },
        "contest": {"entry_format": "20-Max", "field_size": 50000, "payout": "GPP / top-heavy"},
        "context_players": [],
    }


def test_playbook_constant_intact():
    assert "DFS PRO PLAYBOOK:" in _DFS_PRO_PLAYBOOK
    assert "Tournament profitability is dominated by rare top-end finishes" in _DFS_PRO_PLAYBOOK
    assert "lineups behind can pivot toward lower-owned ceiling outcomes." in _DFS_PRO_PLAYBOOK
    print("PASS playbook constant")


def test_serializer_shape(df, res, eq, sim):
    disp = res.copy()
    for c in ["Sim Mean", "Ceiling P90", "Break Slate %"]:
        if c in eq.columns:
            disp[c] = eq[c].to_numpy()
    ctx = postbuild_lineups_context(disp, df, sim["game_table"], n=20)
    assert "#1" in ctx, "rank missing"
    assert "Ceil" in ctx and "Break" in ctx, "ceiling/break columns missing"
    assert "GAME WORLDS" in ctx, "game worlds missing"
    assert "QB:" in ctx and "DST:" in ctx, "roster lines missing"
    assert "leads" in ctx or "Balanced" in ctx, "game-script story missing"
    # A ceiling question's answer must be derivable: best-ceiling rank is named.
    best_rank = int(disp.sort_values("Ceiling P90", ascending=False).iloc[0]["Rank"])
    assert f"#{best_rank}" in ctx
    print(f"PASS serializer shape ({len(ctx)} chars for {len(disp)} lineups)")


def test_serializer_budget(df, res, eq, sim):
    disp = res.copy()
    for c in ["Sim Mean", "Ceiling P90", "Break Slate %"]:
        if c in eq.columns:
            disp[c] = eq[c].to_numpy()
    res20 = pd.concat([disp, disp, disp], ignore_index=True).head(20)
    res20["Rank"] = range(1, len(res20) + 1)
    ctx = postbuild_lineups_context(res20, df, sim["game_table"], n=20)
    assert len(ctx) < 12000, f"context too large: {len(ctx)} chars"
    assert ctx.count("| Grade") == 20, "expected 20 lineup sections"
    print(f"PASS serializer budget ({len(ctx)} chars for 20 lineups)")


def test_serializer_empty():
    assert postbuild_lineups_context(None, None) == ""
    assert postbuild_lineups_context(pd.DataFrame(), None) == ""
    print("PASS serializer empty input")


def test_no_key_falls_back_to_templates():
    packet = make_packet()
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_MODEL", None)
        with mock.patch("streamlit.secrets", {}):
            assert classic_postbuild_llm_answer("Which lineup has the best ceiling?", packet, "CTX") is None
            ans = classic_postbuild_answer(
                "Which lineup has the best ceiling, and why?", packet, [], lineups_ctx="CTX")
    # Legacy template behavior: generic portfolio dump, no lineup named.
    assert "Here is what I can defend from the current build" in ans, ans[:300]
    assert "CTX" not in ans
    print("PASS no-key fallback to templates")


def test_llm_path_prompt_and_response():
    packet = make_packet()
    captured = {}

    class FakeResp:
        output_text = "Lineup #2 has the best ceiling because of the BUF double stack."

    class FakeResponses:
        def create(self, **kw):
            captured.update(kw)
            return FakeResp()

    class FakeOpenAI:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key
            self.responses = FakeResponses()

    fake_openai_module = mock.Mock()
    fake_openai_module.OpenAI = FakeOpenAI

    ctx = "SENTINEL_LINEUP_CTX #1 KC Quarterback (KC +2) | Grade A | Proj 150.0"
    history = [("Earlier question?", "Earlier answer.")]

    fake_session = {}
    with mock.patch.dict(sys.modules, {"openai": fake_openai_module}):
        with mock.patch("streamlit.secrets",
                         {"OPENAI_API_KEY": "sk-test-key", "OPENAI_MODEL": "gpt-test-model"}):
            with mock.patch("streamlit.session_state", fake_session):
                ans = classic_postbuild_answer(
                    "Which lineup has the best ceiling, and why?",
                    packet, history, lineups_ctx=ctx)

    assert ans == FakeResp.output_text, ans
    assert captured["api_key"] == "sk-test-key", "API key not passed to client"
    assert captured["model"] == "gpt-test-model", captured.get("model")
    prompt = captured["input"]
    assert "SENTINEL_LINEUP_CTX" in prompt, "portfolio data missing from prompt"
    assert "Which lineup has the best ceiling, and why?" in prompt, "question missing"
    assert "USER: Earlier question?" in prompt, "history missing"
    assert "DFS PRO PLAYBOOK" in prompt, "playbook missing"
    assert "NEVER INVENT" in prompt.upper() or "Never invent" in prompt, "grounding instruction missing"
    assert fake_session.get("classic_ai_model") == "gpt-test-model"
    assert "classic_ai_error" not in fake_session
    print("PASS llm path prompt + response")


def test_headless_agent_view_render(df, res, eq, sim):
    """Headless AppTest: the Lab+Agent view renders with the new wiring,
    starter buttons exist, and the no-key hint caption shows (no key in tests)."""
    from streamlit.testing.v1 import AppTest

    script = (
        "import sys\n"
        "sys.path.insert(0, '/home/hatch/workspace/dfs-lab')\n"
        "sys.path.insert(0, '/home/hatch/workspace/dfs-lab/tests')\n"
        "import streamlit as st\n"
        "from dfs_lab.ui.results import _equity, render_agent_view\n"
        "from test_results_command_center import build_pipeline\n"
        "df, bb_worthy, sim, res, eq = build_pipeline()\n"
        "disp = _equity(res, sim['worlds'], df)\n"
        "packet = {'portfolio': {'built': True}, 'contest': {}}\n"
        "render_agent_view(packet, disp, df, sim['game_table'])\n"
    )
    with open("/tmp/_lab_agent_apptest.py", "w") as f:
        f.write(script)
    at = AppTest.from_file("/tmp/_lab_agent_apptest.py")
    at.run(timeout=300)
    assert len(at.exception) == 0, [str(e) for e in at.exception]
    labels = " ".join(str(b.label) for b in at.button)
    assert "Which lineup has the best ceiling, and why?" in labels, labels
    captions = " ".join(str(c.value) for c in at.caption)
    assert "OPENAI_API_KEY" in captions, "no-key hint caption missing"
    print("PASS headless agent view render")


if __name__ == "__main__":
    test_playbook_constant_intact()
    df, bb_worthy, sim, res, eq = build_pipeline()
    print(f"  built {len(res)} lineups on synthetic KC@MIA + BUF@DET slate")
    test_serializer_shape(df, res, eq, sim)
    test_serializer_budget(df, res, eq, sim)
    test_serializer_empty()
    test_no_key_falls_back_to_templates()
    test_llm_path_prompt_and_response()
    test_headless_agent_view_render(df, res, eq, sim)
    print("\nALL LAB-AGENT-LLM TESTS PASSED")
