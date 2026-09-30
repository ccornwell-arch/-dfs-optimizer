"""Team-color theming: 32-team map, auto-lift for dark accents, single CSS block.

The favorite-team theme must never produce an invisible accent on the
midnight base (dark brand colors like Ravens purple get lifted), text on the
accent must stay readable, and unknown/empty picks must fall back to the
Midnight Volt house theme.
"""
import re

from dfs_lab.theme import (
    LEGACY_SURFACES, TEAM_COLORS, on_accent, resolve_theme, team_options, theme_css, _luminance,
)


def test_all_32_teams_present():
    assert len(TEAM_COLORS) == 32
    assert set(TEAM_COLORS) >= {"PHI", "KC", "DAL", "GB", "NE", "SF"}


def test_all_colors_are_valid_hex():
    for abbr, (name, p, s) in TEAM_COLORS.items():
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", p), abbr
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", s), abbr
        assert name and len(name) > 3, abbr


def test_dark_accents_are_lifted_to_visible():
    # Brand colors that are near-invisible on midnight navy must be lifted.
    for abbr in ("BAL", "IND", "PHI", "NYJ", "HOU"):
        t = resolve_theme(abbr)
        assert _luminance(t["accent"]) >= 0.22, f"{abbr} accent too dark: {t['accent']}"


def test_all_team_accents_visible():
    for abbr in TEAM_COLORS:
        t = resolve_theme(abbr)
        assert _luminance(t["accent"]) >= 0.22, abbr
        # secondary must also survive the midnight base (black -> trim gray)
        assert _luminance(t["accent2"]) >= 0.10, f"{abbr} secondary too dark"


def test_duotone_keys_present():
    t = resolve_theme("SF")
    for k in ("accent", "accent2", "accent_deep", "accent2_deep",
              "on_accent", "on_gradient", "wash_a", "wash_b"):
        assert k in t and t[k], k
    # 49ers: red primary, gold secondary — genuinely two colors
    assert t["accent"] != t["accent2"]


def test_on_accent_contrast():
    assert on_accent("#C6F135") == "#11151D"   # volt -> dark text
    assert on_accent("#E31837") == "#FFFFFF"   # chiefs red -> white text
    t = resolve_theme("GB")
    assert t["on_accent"] in ("#11151D", "#FFFFFF")


def test_default_is_midnight_volt():
    for abbr in (None, "", "XXX"):
        t = resolve_theme(abbr)
        assert t["abbr"] is None
        assert t["name"] == "Midnight Volt"
        assert t["accent"] == "#C6F135"


def test_css_is_single_style_block_with_team_accent():
    css = theme_css("KC")
    assert css.count("<style>") == 1 and css.count("</style>") == 1
    t = resolve_theme("KC")
    assert t["accent"] in css
    assert t["accent2"] in css
    assert t["on_gradient"] in css
    # key brand touchpoints are themed
    for sel in ('button[kind="primary"]', 'button[data-selected="true"]',
                ".lineup-cpt span", ".lab-brand b", 'button[aria-selected="true"]',
                "st-key-fav_team", "stAppViewContainer", "border-image"):
        assert sel in css, sel


def test_css_uses_vars_not_hardcoded_wars():
    css = theme_css("PHI")
    assert "var(--lab-accent)" in css
    # must not reintroduce the old hard-coded blue wars
    assert "#0071e3" not in css and "#5B9DFF" not in css


def test_team_takeover_redefines_legacy_surfaces():
    # Full team takeover: picking a team must re-tint the legacy surfaces,
    # not just the accents — otherwise the app stays midnight with red bits.
    t = resolve_theme("SF")
    for k in ("rcc_bg", "rcc_panel", "rcc_panel2", "rcc_line", "rcc_accent", "rcc_accent2"):
        assert t[k] != LEGACY_SURFACES[k], k
    css = theme_css("SF")
    assert "--rcc-panel:" in css and "--rcc-accent:" in css
    # the takeover must be dark enough to keep text readable
    from dfs_lab.theme import _luminance
    assert _luminance(t["rcc_bg"]) < 0.03
    assert _luminance(t["rcc_panel"]) < 0.06


def test_default_keeps_legacy_surfaces():
    # No team picked -> house theme pixel-identical to before.
    t = resolve_theme(None)
    for k, v in LEGACY_SURFACES.items():
        assert t[k] == v, k


def test_css_selectors_beat_legacy_specificity():
    # Several legacy theme blocks use the `html body` prefix, which outranks a
    # bare selector (0,2,2 beats 0,2,1) no matter the source order. Every real
    # selector in the dynamic theme must carry the prefix, or "injected last"
    # does not win — this is exactly how the GENERATE button stayed blue.
    css = theme_css("SF")
    in_block = False
    for raw in css.splitlines():
        s = raw.strip()
        if not s or s.startswith("/*") or s in ("<style>", "</style>"):
            continue
        if s.startswith(":root{"):
            in_block = True
            continue
        if "{" in s and not in_block:
            in_block = True
            sels = s.split("{")[0]
            for sel in sels.split(","):
                sel = sel.strip()
                assert sel.startswith("html body"), f"unprefixed selector loses the theme war: {sel!r}"
            continue
        if "}" in s:
            in_block = False


def test_team_options_default_first_and_sorted():
    opts = team_options()
    assert opts[0] == (None, "Midnight Volt")
    names = [n for _, n in opts[1:]]
    assert names == sorted(names)
    assert len(opts) == 33
