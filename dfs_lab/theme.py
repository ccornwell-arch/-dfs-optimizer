"""House theme for Aytia: Midnight Ice.

A single dynamic <style> block, injected last in the cascade, that paints the
app's brand touchpoints (primary buttons, selected nav pill, selected tabs,
CPT badge, kickers, links, wordmark accent) in ice cyan over the midnight
base. Team theming was scrapped Sep 2026; theme.py stays as the single brand
source. Pure: no Streamlit calls.
"""

# abbr: (full name, primary accent, secondary accent). The primary is the
# color that must pop on the midnight base; dark primaries are lifted
# toward white automatically (see _lift) so e.g. Ravens purple still reads.
TEAM_COLORS = {
    "ARI": ("Arizona Cardinals", "#97233F", "#FFFFFF"),
    "ATL": ("Atlanta Falcons", "#A7194B", "#FFFFFF"),
    "BAL": ("Baltimore Ravens", "#241773", "#9E7C0C"),
    "BUF": ("Buffalo Bills", "#00338D", "#C60C30"),
    "CAR": ("Carolina Panthers", "#0085CA", "#BFC0BF"),
    "CHI": ("Chicago Bears", "#C83803", "#0B162A"),
    "CIN": ("Cincinnati Bengals", "#FB4F14", "#FFFFFF"),
    "CLE": ("Cleveland Browns", "#FF3C00", "#311D00"),
    "DAL": ("Dallas Cowboys", "#869397", "#003594"),
    "DEN": ("Denver Broncos", "#FB4F14", "#002244"),
    "DET": ("Detroit Lions", "#0076B6", "#B0B7BC"),
    "GB": ("Green Bay Packers", "#FFB612", "#203731"),
    "HOU": ("Houston Texans", "#A7194B", "#03202F"),
    "IND": ("Indianapolis Colts", "#A2AAAD", "#002C5F"),
    "JAX": ("Jacksonville Jaguars", "#D7A22B", "#006778"),
    "KC": ("Kansas City Chiefs", "#E31837", "#FFB81C"),
    "LV": ("Las Vegas Raiders", "#A5ACAF", "#000000"),
    "LAC": ("Los Angeles Chargers", "#FFC20E", "#0080C6"),
    "LAR": ("Los Angeles Rams", "#FFA300", "#003594"),
    "MIA": ("Miami Dolphins", "#FC4C02", "#008E97"),
    "MIN": ("Minnesota Vikings", "#FFC62F", "#4F2683"),
    "NE": ("New England Patriots", "#C60C30", "#002244"),
    "NO": ("New Orleans Saints", "#D3BC8D", "#FFFFFF"),
    "NYG": ("New York Giants", "#A7194B", "#0B2265"),
    "NYJ": ("New York Jets", "#125740", "#FFFFFF"),
    "PHI": ("Philadelphia Eagles", "#004C54", "#A5ACAF"),
    "PIT": ("Pittsburgh Steelers", "#FFB612", "#FFFFFF"),
    "SF": ("San Francisco 49ers", "#AA0000", "#B3995D"),
    "SEA": ("Seattle Seahawks", "#69BE28", "#002244"),
    "TB": ("Tampa Bay Buccaneers", "#FF7900", "#D50A0A"),
    "TEN": ("Tennessee Titans", "#4B92DB", "#0C2340"),
    "WAS": ("Washington Commanders", "#FFB612", "#5A1414"),
}

# The house theme: Midnight Ice (deep navy + ice cyan). Chosen Sep 2026 after
# the neon volt yellow was rejected for readability.
DEFAULT_NAME = "Midnight Ice"
DEFAULT_ACCENT = "#6FD3F2"
DEFAULT_ACCENT2 = "#4D8DFF"


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#%02X%02X%02X" % tuple(max(0, min(255, int(round(c)))) for c in rgb)


def _luminance(h):
    r, g, b = [c / 255.0 for c in _hex_to_rgb(h)]

    def _lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _mix(h1, h2, t):
    a, b = _hex_to_rgb(h1), _hex_to_rgb(h2)
    return _rgb_to_hex(tuple(x + (y - x) * t for x, y in zip(a, b)))


def _shade(h, amt):
    """Mix a hex color toward black by amt (0..1)."""
    return _mix(h, "#000000", amt)


# Legacy dark surfaces (the pre-theme values). Used when no team is picked so
# the house Midnight Ice theme is pixel-identical to before.
LEGACY_SURFACES = {
    "rcc_bg": "#101418", "rcc_bg2": "#0b0f13",
    "rcc_panel": "#151b23", "rcc_panel2": "#1b232e", "rcc_panel3": "#222c39",
    "rcc_line": "#2b3644", "rcc_line2": "#354252",
    "rcc_text": "#eef3f8", "rcc_muted": "#94a5b8",
    "rcc_accent": "#5b9dff", "rcc_accent2": "#7fb3ff", "rcc_accent_dim": "#274b73",
}


def _team_surfaces(primary, accent, accent2):
    """Full team takeover: every legacy dark surface re-tinted in team colors."""
    return {
        "rcc_bg": _shade(primary, 0.93),
        "rcc_bg2": _shade(primary, 0.90),
        "rcc_panel": _mix(primary, "#151b23", 0.80),
        "rcc_panel2": _mix(primary, "#1b232e", 0.74),
        "rcc_panel3": _mix(primary, "#222c39", 0.66),
        "rcc_line": _mix(primary, "#2b3644", 0.52),
        "rcc_line2": _mix(primary, "#354252", 0.48),
        "rcc_text": _mix("#ffffff", primary, 0.05),
        "rcc_muted": _mix("#94a5b8", primary, 0.14),
        "rcc_accent": accent,
        "rcc_accent2": accent2,
        "rcc_accent_dim": _darken(accent, 0.5),
    }


def _lift(h, floor=0.22):
    """Lift a dark brand color just enough to pop on the midnight base.

    Binary-searches lightness in HLS space for the smallest change that
    reaches the luminance floor, so the hue stays faithful: Eagles midnight
    green becomes a deep readable teal (not electric cyan), Ravens purple
    stays purple. Colors already bright enough are returned untouched.
    """
    import colorsys
    h = h.upper()
    if _luminance(h) >= floor:
        return h
    r, g, b = [c / 255.0 for c in _hex_to_rgb(h)]
    hue, light, sat = colorsys.rgb_to_hls(r, g, b)
    lo, hi = light, 1.0
    for _ in range(16):
        mid = (lo + hi) / 2
        r2, g2, b2 = colorsys.hls_to_rgb(hue, mid, sat)
        if _luminance(_rgb_to_hex((r2 * 255, g2 * 255, b2 * 255))) >= floor:
            hi = mid
        else:
            lo = mid
    r2, g2, b2 = colorsys.hls_to_rgb(hue, hi, sat)
    return _rgb_to_hex((r2 * 255, g2 * 255, b2 * 255))


def _darken(h, t=0.35):
    return _mix(h.upper(), "#0B0E14", t)


def _rgba(h, a):
    r, g, b = _hex_to_rgb(h)
    return f"rgba({r},{g},{b},{a})"


def on_accent(h):
    """Readable text color on top of the accent: near-black on bright accents."""
    return "#11151D" if _luminance(h) > 0.42 else "#FFFFFF"


def resolve_theme(team_abbr=None):
    """Resolve a team abbreviation to concrete theme values. Pure."""
    abbr = str(team_abbr or "").upper()
    if abbr in TEAM_COLORS:
        name, primary, secondary = TEAM_COLORS[abbr]
        accent = _lift(primary)
        # The secondary must also survive the midnight base: black becomes a
        # dark trim gray, navy becomes readable blue, white stays white.
        accent2 = _lift(secondary, floor=0.10)
        surfaces = _team_surfaces(primary, accent, accent2)
    else:
        abbr, name = None, DEFAULT_NAME
        accent = DEFAULT_ACCENT
        accent2 = _lift(DEFAULT_ACCENT2, floor=0.10)
        surfaces = dict(LEGACY_SURFACES)
    # Text that sits on a button gradient must stay readable across the whole
    # gradient. Buttons use a deliberately NARROW single-hue gradient
    # (accent -> 14% shaded accent) so one text color keeps >= 4.5:1 contrast
    # at both ends; contrast is judged at the gradient midpoint.
    btn_deep = _shade(accent, 0.14)
    grad_mid = _mix(accent, btn_deep, 0.5)
    return {
        "abbr": abbr,
        "name": name,
        "accent": accent,
        "accent_deep": _darken(accent),
        "accent2": accent2,
        "accent2_deep": _darken(accent2, 0.45),
        "on_accent": on_accent(accent),
        "on_gradient": on_accent(grad_mid),
        "btn_deep": btn_deep,
        "glow": _rgba(accent, 0.28),
        "soft": _rgba(accent, 0.14),
        "wash_a": _rgba(accent, 0.10),
        "wash_b": _rgba(accent2, 0.10),
        **surfaces,
    }


def team_options():
    """Sorted [(abbr, name)] for the picker, with the house theme first."""
    opts = sorted(TEAM_COLORS.items(), key=lambda kv: kv[1][0])
    return [(None, DEFAULT_NAME)] + [(abbr, name) for abbr, (name, _, _) in opts]


def theme_css(team_abbr=None):
    """One dynamic <style> block with the team theme. Injected last. Pure."""
    t = resolve_theme(team_abbr)
    a, ad, oa, glow, soft = t["accent"], t["accent_deep"], t["on_accent"], t["glow"], t["soft"]
    a2, a2d, og, bd = t["accent2"], t["accent2_deep"], t["on_gradient"], t["btn_deep"]
    wa, wb = t["wash_a"], t["wash_b"]
    rcc_vars = "".join(f"--{k.replace('_','-')}:{t[k]};" for k in
                       ("rcc_bg", "rcc_bg2", "rcc_panel", "rcc_panel2", "rcc_panel3",
                        "rcc_line", "rcc_line2", "rcc_text", "rcc_muted",
                        "rcc_accent", "rcc_accent2", "rcc_accent_dim"))
    return f"""<style>
/* Aytia dynamic team theme · {t["name"]} — single brand source, injected last.
   Every selector carries the `html body` prefix: several legacy theme blocks use it,
   and without it they outrank this layer on specificity despite loading earlier.
   Full takeover: the legacy --rcc-* surface vars are redefined here, so every
   legacy rule built on them (cards, expanders, metrics, hovers, focus rings)
   picks up the team tint with no per-component overrides. */
:root{{--lab-accent:{a};--lab-accent-deep:{ad};--lab-btn-deep:{bd};--lab-accent2:{a2};--lab-accent2-deep:{a2d};
--lab-on-accent:{oa};--lab-on-gradient:{og};--lab-glow:{glow};--lab-soft:{soft};
--lab-wash-a:{wa};--lab-wash-b:{wb};{rcc_vars}}}
/* Team wash: the whole page sits in the team colors — tinted base + duotone glow. */
html body [data-testid="stAppViewContainer"]{{
background:radial-gradient(1100px 520px at 8% -6%,var(--lab-wash-a),transparent 60%),
radial-gradient(1000px 520px at 96% 4%,var(--lab-wash-b),transparent 60%),
linear-gradient(145deg,var(--rcc-bg) 0%,var(--rcc-bg2) 60%,var(--rcc-bg) 100%)!important;}}
/* Nav bar + cards/metrics/frames follow the team surfaces */
html body .st-key-sd_nav,html body .st-key-classic_nav{{background:var(--rcc-bg)!important;}}
html body [data-testid="stMetric"],html body [data-testid="stDataFrame"],html body .lineup-card{{
background:var(--rcc-panel)!important;border-color:var(--rcc-line)!important;}}
/* Selectboxes: dark team-tinted, never white */
html body [data-baseweb="select"]>div{{background:var(--rcc-panel2)!important;
border-color:var(--rcc-line2)!important;}}
html body [data-baseweb="select"] [data-testid="stMarkdownContainer"] p{{
color:var(--rcc-text)!important;-webkit-text-fill-color:var(--rcc-text)!important;}}
/* Primary actions: single-hue gradient (accent -> deeper accent) so the
   button text color stays high-contrast across the whole surface.
   Descendants are forced transparent — no legacy blue inner leaking through. */
html body .stButton>button[kind="primary"],html body [data-testid="stFormSubmitButton"] button,
html body [data-testid="stDownloadButton"] button{{
background:linear-gradient(180deg,var(--lab-accent) 0%,var(--lab-btn-deep) 100%)!important;
color:var(--lab-on-gradient)!important;-webkit-text-fill-color:var(--lab-on-gradient)!important;
border:0!important;box-shadow:0 8px 24px var(--lab-glow)!important;}}
html body .stButton>button[kind="primary"] *,html body [data-testid="stFormSubmitButton"] button *,
html body [data-testid="stDownloadButton"] button *{{
background:transparent!important;background-image:none!important;
color:inherit!important;-webkit-text-fill-color:inherit!important;}}
/* Nav: selected pill in primary, gradient rule under the bar in both colors. */
html body .st-key-sd_nav [data-testid="stButtonGroup"] button[data-selected="true"],
html body .st-key-classic_nav [data-testid="stButtonGroup"] button[data-selected="true"]{{
color:var(--lab-on-accent)!important;background:var(--lab-accent)!important;border-color:var(--lab-accent)!important;}}
html body .st-key-sd_nav,html body .st-key-classic_nav{{
border-bottom:2px solid transparent!important;
border-image:linear-gradient(90deg,var(--lab-accent),var(--lab-accent2)) 1!important;}}
/* Tabs + results hub tabs */
html body [data-testid="stTabs"] button[aria-selected="true"]{{background:var(--lab-accent)!important;}}
html body [data-testid="stTabs"] button[aria-selected="true"] p{{
color:var(--lab-on-accent)!important;-webkit-text-fill-color:var(--lab-on-accent)!important;}}
html body .st-key-sd_results_hub [role="tab"][aria-selected="true"]{{
background:linear-gradient(180deg,var(--lab-accent) 0%,var(--lab-btn-deep) 100%)!important;}}
html body .st-key-sd_results_hub [role="tab"][aria-selected="true"] p,
html body .st-key-sd_results_hub [role="tab"][aria-selected="true"] span{{
color:var(--lab-on-gradient)!important;-webkit-text-fill-color:var(--lab-on-gradient)!important;}}
/* Wordmark + picker */
html body .lab-brand b{{color:var(--lab-accent)!important;}}
html body .st-key-fav_team [data-baseweb="select"]>div{{border-color:var(--lab-accent)!important;
box-shadow:0 0 0 1px var(--lab-soft)!important;}}
/* CPT badge in primary, rank highlight in secondary */
html body .lineup-cpt span{{background:var(--lab-accent)!important;color:var(--lab-on-accent)!important;
-webkit-text-fill-color:var(--lab-on-accent)!important;border-radius:6px;padding:1px 8px;}}
html body .lineup-rank span{{color:var(--lab-accent2)!important;}}
/* Kickers + links */
html body .answer-kicker{{color:var(--lab-accent)!important;}}
html body a{{color:var(--lab-accent)!important;}}
/* Results hero: duotone gradient, darkened so text stays readable */
html body .results-hub-hero{{background:linear-gradient(115deg,var(--lab-accent-deep),var(--lab-accent2-deep))!important;
border:1px solid var(--lab-glow)!important;}}
html body [data-testid="stExpander"] summary:hover{{color:var(--lab-accent)!important;}}
</style>"""
