"""Regression test: nav-independent build inputs.

Background: the primary nav was changed from st.tabs to a keyed segmented
control. Unlike st.tabs (which executes every tab body on each run), only the
active section's body executes now. The Showdown/Classic generate handlers and
other code run UNCONDITIONALLY on every run, so any name they reference must be
assigned unconditionally too — never only inside one nav section.

The real-world failure: tapping GENERATE on the Showdown Build tab raised
"cannot access local variable 'effective_script'" because effective_script was
assigned only in the Rules section while the build block runs every run.

This test statically walks dfs_lab/ui/main.py and asserts that no name
referenced by unconditional code (or by a different section) is assigned
exclusively inside nav sections.

Run from repo root:  python3 tests/test_nav_independent_inputs.py
"""
import re
import sys

sys.path.insert(0, ".")

MAIN = "dfs_lab/ui/main.py"

# (start_line, end_line) 0-indexed, bounding each branch so the classic and
# showdown analyses cannot pollute each other.
_BRANCHES = [
    # branch, nav var, tabs var, first line, last line (exclusive)
    ("classic", "classic_nav", "_CLASSIC_TABS", 242, 790),
    ("sd", "sd_nav", "_SD_TABS", 790, 1798),
]

_SKIP = {
    "st", "pd", "np", "True", "False", "None", "for", "in", "if", "else",
    "elif", "not", "and", "or", "with", "as", "int", "float", "str", "list",
    "dict", "len", "max", "min", "range", "enumerate", "isinstance", "bool",
    "print", "set", "tuple", "sorted", "zip", "round", "abs", "any", "all",
    "Exception", "def", "return", "try", "except", "import", "from", "class",
    "lambda", "is", "raise",
}


def _strip(line):
    code = re.sub(r"#.*$", "", line)
    code = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', code)
    code = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", code)
    return code


def _section_contexts(lines, nav_var, tabs_var):
    """Yield (lineno, section) where section is an int, or None for nav-independent code."""
    header = re.compile(r"^            if " + nav_var + r"==" + tabs_var + r"\[(\d+)\]:$")
    section = "pre"  # code before the first section header: unconditional
    for i, ln in enumerate(lines):
        indent = len(ln) - len(ln.lstrip())
        if indent == 12:
            m = header.match(ln)
            if m:
                section = int(m.group(1))
            elif ln.strip() and not ln.strip().startswith("#"):
                # A new 12-space statement ends the previous section body.
                section = None
        yield i, section, indent


def _analyze(lines, nav_var, tabs_var):
    assigned = {}   # name -> set of sections ("pre"/None/int)
    referenced = {}  # name -> set of sections where referenced
    for i, section, indent in _section_contexts(lines, nav_var, tabs_var):
        if indent < 12:
            continue
        code = _strip(lines[i])
        if not code.strip():
            continue
        targets = set()
        for m in re.finditer(r"(?<![.\w])([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\+|-|\*|/|%|//)?=(?![=>])", code):
            name = m.group(1)
            if name in _SKIP:
                continue
            # '==' slipped through the negative lookahead on some forms; recheck
            targets.add(name)
            assigned.setdefault(name, set()).add(section)
        # for-loop and with-as targets are assignments too (incl. comprehensions).
        for fm in re.finditer(r"\bfor\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)\s+in\b", code):
            for name in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", fm.group(1)):
                if name not in _SKIP:
                    targets.add(name)
                    assigned.setdefault(name, set()).add(section)
        wm = re.search(r"\bwith\b.*\bas\s+([a-zA-Z_][a-zA-Z0-9_]*)", code)
        if wm and wm.group(1) not in _SKIP:
            targets.add(wm.group(1))
            assigned.setdefault(wm.group(1), set()).add(section)
        for m in re.finditer(r"(?<![.\w])([a-zA-Z_][a-zA-Z0-9_]*)\b", code):
            name = m.group(1)
            if name in _SKIP or name in targets:
                continue
            referenced.setdefault(name, set()).add(section)
    return assigned, referenced


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def test_no_section_gated_names_in_unconditional_code():
    lines = open(MAIN).read().split("\n")
    problems = []
    for branch, nav_var, tabs_var, start, end in _BRANCHES:
        assigned, referenced = _analyze(lines[start:end], nav_var, tabs_var)
        for name, rsecs in referenced.items():
            asecs = assigned.get(name, set())
            if not asecs:
                continue  # builtin / import / defined elsewhere
            if "pre" in asecs or None in asecs:
                continue  # assigned unconditionally somewhere
            # Assigned only inside sections: every referencing section must be
            # one of the assigning sections.
            bad = sorted(s for s in rsecs if s not in asecs and s in ("pre", None))
            if bad:
                problems.append(f"{branch}: {name!r} assigned only in sections {sorted(asecs)} but used by nav-independent code")
    _check("no section-gated names referenced by nav-independent code", not problems)
    for p in problems:
        print("   " + p)


if __name__ == "__main__":
    test_no_section_gated_names_in_unconditional_code()
    print("OK")
