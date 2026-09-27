"""QB pool control + coach-note fixes.

- classic_apply_qb_exclusions: unchecking QBs in Slate Intel removes them from
  the build pool, and the pool can never be emptied (top-ranked QB is kept).
- classic_postbuild_report: the bring-back note is skipped when the user set
  bring-back to "None"; the stack-structure note never says "Only 0 of N".
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dfs_lab.classic import classic_apply_qb_exclusions, classic_postbuild_report

def check(name, cond):
    print(("ok: " if cond else "FAIL: ") + name)
    assert cond, name

# --- exclusion helper ---
check("no exclusions -> unchanged",
      classic_apply_qb_exclusions(["a", "b", "c"], set()) == ["a", "b", "c"])
check("excluded QB removed",
      classic_apply_qb_exclusions(["a", "b", "c"], {"b"}) == ["a", "c"])
check("unknown ids ignored",
      classic_apply_qb_exclusions(["a", "b"], {"zzz"}) == ["a", "b"])
check("all excluded -> keeps top-ranked QB",
      classic_apply_qb_exclusions(["a", "b", "c"], {"a", "b", "c"}) == ["a"])
check("empty base -> empty",
      classic_apply_qb_exclusions([], {"a"}) == [])
check("rank order preserved",
      classic_apply_qb_exclusions(["c", "b", "a"], {"b"}) == ["c", "a"])
check("None excluded set -> unchanged",
      classic_apply_qb_exclusions(["a", "b"], None) == ["a", "b"])

def _packet(bringback, stack_mix):
    return {
        "portfolio": {"built": True, "lineups": 20, "unique_qbs": 3,
                      "stack_mix": stack_mix, "bringback_mix": {"0": 19},
                      "flex_mix_pct": {}},
        "contest": {"entry_format": "3-Max", "field_size": 5000, "payout": "Top-heavy"},
        "active_rules": {"bringback": bringback, "qb_pass_catchers": 1},
    }

def _finding_text(findings, title):
    return next((txt for t, txt in findings if t == title), "")

# --- bring-back note respects the user's None setting ---
f_none = classic_postbuild_report(_packet("None", {"1": 20}))
check("bringback=None -> no bring-back note",
      not any(t == "Bring-backs" for t, _ in f_none))
f_opt = classic_postbuild_report(_packet("Optional", {"1": 20}))
check("bringback=Optional, 19/20 without -> note present",
      any(t == "Bring-backs" for t, _ in f_opt))
f_req = classic_postbuild_report(_packet("Required", {"1": 20}))
check("bringback=Required, 19/20 without -> note present",
      any(t == "Bring-backs" for t, _ in f_req))

# --- stack note phrasing ---
stack_zero = _finding_text(classic_postbuild_report(_packet("Optional", {"1": 20})), "Stack structure")
check("zero doubles says 'None of 20'", stack_zero.startswith("None of 20"))
check("never says 'Only 0 of'", "Only 0 of" not in stack_zero)
stack_some = _finding_text(classic_postbuild_report(_packet("Optional", {"1": 17, "2": 3})), "Stack structure")
check("3 doubles says 'Only 3 of 20'", stack_some.startswith("Only 3 of 20"))
stack_many = classic_postbuild_report(_packet("Optional", {"2": 15, "1": 5}))
stack_many_txt = _finding_text(stack_many, "Stack structure")
check("15/20 doubles -> concentrated-bet variant, not skinny variant",
      "concentrated construction bet" in stack_many_txt)

print("ALL QB-POOL-CONTROL TESTS PASSED")
