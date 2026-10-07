"""Regression tests for the Players "Apply player changes" crash and the
projection-override path.

Background: tapping "Apply player changes" (Classic and Showdown) raised
"Showdown build error: Widget state is read-only because modifying nested
values has no effect on the app." The Apply handler cleared pending edits via
`st.session_state[widget_key]["edited_rows"] = {}`, which Streamlit forbids —
nested widget-state mutation raises. The fix pops the widget key instead
(verified headlessly: whole-dict assignment after instantiation also raises,
pop does not).

The same crash was silently blocking projection overrides: the "Your Proj"
column is editable and Apply already saved overrides, but the crash killed the
handler before anything was written.

Covers:
  1. Headless AppTest: form + data_editor + Apply that pops the widget key
     completes with no exception (the old nested-mutation line raises).
  2. main.py no longer contains the forbidden nested-mutation pattern.
  3. apply_projection_overrides honors user values and ignores the rest.

Run from repo root:  python3 tests/test_apply_player_changes.py
"""
import re
import sys

sys.path.insert(0, ".")

import pandas as pd

from dfs_lab.data import apply_projection_overrides


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def test_no_nested_widget_state_mutation():
    src = open("dfs_lab/ui/main.py").read()
    hits = re.findall(r"_ws\[\"edited_rows\"\]\s*=", src)
    _check("no nested widget-state mutation in main.py", len(hits) == 0)


def test_pop_clears_pending_edits_without_error():
    """The fixed Apply pattern: pop the data_editor widget key after submit."""
    from streamlit.testing.v1 import AppTest

    script = (
        "import streamlit as st\n"
        "import pandas as pd\n"
        "df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})\n"
        "with st.form('f'):\n"
        "    st.data_editor(df, key='ed')\n"
        "    go = st.form_submit_button('Apply')\n"
        "if go:\n"
        "    st.session_state.pop('ed', None)\n"
        "    st.session_state['applied'] = True\n"
    )
    with open("/tmp/_apply_pop_test.py", "w") as f:
        f.write(script)
    at = AppTest.from_file("/tmp/_apply_pop_test.py")
    at.run(timeout=120)
    at.button[0].click().run(timeout=120)
    _check("apptest: no exception on apply+pop", len(at.exception) == 0)
    _check("apptest: apply handler ran", at.session_state.get("applied") is True)


def test_nested_mutation_raises():
    """Documents WHY pop is used: the old pattern raises in Streamlit."""
    from streamlit.testing.v1 import AppTest

    script = (
        "import streamlit as st\n"
        "import pandas as pd\n"
        "df = pd.DataFrame({'a': [1, 2]})\n"
        "with st.form('f'):\n"
        "    st.data_editor(df, key='ed')\n"
        "    go = st.form_submit_button('Apply')\n"
        "if go:\n"
        "    try:\n"
        "        ws = st.session_state.get('ed')\n"
        "        if isinstance(ws, dict):\n"
        "            ws['edited_rows'] = {}\n"
        "        st.session_state['outcome'] = 'no-raise'\n"
        "    except Exception as e:\n"
        "        st.session_state['outcome'] = type(e).__name__\n"
    )
    with open("/tmp/_apply_nested_test.py", "w") as f:
        f.write(script)
    at = AppTest.from_file("/tmp/_apply_nested_test.py")
    at.run(timeout=120)
    at.button[0].click().run(timeout=120)
    _check("apptest: nested widget mutation raises",
           at.session_state.get("outcome") not in (None, "no-raise"))


def test_projection_overrides_applied():
    df = pd.DataFrame({
        "ID": ["1", "2", "3"],
        "Aytia Proj": [10.0, 20.0, 30.0],
        "My Proj": [10.0, 20.0, 30.0],
    })
    out = apply_projection_overrides(df, {"2": 25.5})
    got = dict(zip(out["ID"].astype(str), out["Aytia Proj"]))
    _check("override: user value used", abs(got["2"] - 25.5) < 1e-9)
    _check("override: others keep model", abs(got["1"] - 10.0) < 1e-9 and abs(got["3"] - 30.0) < 1e-9)
    _check("override: flag set", bool(out.loc[out["ID"] == "2", "Projection Override"].iloc[0]) is True)
    _check("override: flag unset elsewhere", bool(out.loc[out["ID"] == "1", "Projection Override"].iloc[0]) is False)


if __name__ == "__main__":
    test_no_nested_widget_state_mutation()
    test_pop_clears_pending_edits_without_error()
    test_nested_mutation_raises()
    test_projection_overrides_applied()
    print("OK")
