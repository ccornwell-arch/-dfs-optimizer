"""Team-color theming for DFS LAB.

A single dynamic <style> block, injected last in the cascade, that paints the
app's brand touchpoints (primary buttons, selected nav pill, selected tabs,
CPT badge, kickers, links, wordmark accent) in the user's favorite team's
colors over the midnight base. Pure: no Streamlit calls.
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

# The house theme when no team is picked: Midnight Volt.
DEFAULT_NAME = "Midnight Volt"
DEFAULT_ACCENT = "#C6F135"
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
    else:
        abbr, name, secondary = None, DEFAULT_NAME, DEFAULT_ACCENT2
        accent = DEFAULT_ACCENT
    return {
        "abbr": abbr,
        "name": name,
        "accent": accent,
        "accent_deep": _darken(accent),
        "accent2": secondary,
        "on_accent": on_accent(accent),
        "glow": _rgba(accent, 0.28),
        "soft": _rgba(accent, 0.14),
    }


def team_options():
    """Sorted [(abbr, name)] for the picker, with the house theme first."""
    opts = sorted(TEAM_COLORS.items(), key=lambda kv: kv[1][0])
    return [(None, DEFAULT_NAME)] + [(abbr, name) for abbr, (name, _, _) in opts]


def theme_css(team_abbr=None):
    """One dynamic <style> block with the team theme. Injected last. Pure."""
    t = resolve_theme(team_abbr)
    a, ad, oa, glow, soft = t["accent"], t["accent_deep"], t["on_accent"], t["glow"], t["soft"]
    return f"""<style>
/* DFS LAB dynamic team theme · {t["name"]} — single brand source, injected last */
:root{{--lab-accent:{a};--lab-accent-deep:{ad};--lab-on-accent:{oa};--lab-glow:{glow};--lab-soft:{soft};}}
.stButton>button[kind="primary"],[data-testid="stFormSubmitButton"] button,[data-testid="stDownloadButton"] button{{
background:linear-gradient(100deg,var(--lab-accent),var(--lab-accent-deep))!important;
color:var(--lab-on-accent)!important;-webkit-text-fill-color:var(--lab-on-accent)!important;
border:0!important;box-shadow:0 8px 24px var(--lab-glow)!important;}}
.stButton>button[kind="primary"] *,[data-testid="stFormSubmitButton"] button *,[data-testid="stDownloadButton"] button *{{
color:inherit!important;-webkit-text-fill-color:inherit!important;}}
html body .st-key-sd_nav [data-testid="stButtonGroup"] button[data-selected="true"],
html body .st-key-classic_nav [data-testid="stButtonGroup"] button[data-selected="true"]{{
color:var(--lab-on-accent)!important;background:var(--lab-accent)!important;border-color:var(--lab-accent)!important;}}
[data-testid="stTabs"] button[aria-selected="true"]{{background:var(--lab-accent)!important;}}
[data-testid="stTabs"] button[aria-selected="true"] p{{color:var(--lab-on-accent)!important;-webkit-text-fill-color:var(--lab-on-accent)!important;}}
html body .st-key-sd_results_hub [role="tab"][aria-selected="true"]{{
background:linear-gradient(100deg,var(--lab-accent),var(--lab-accent-deep))!important;}}
html body .st-key-sd_results_hub [role="tab"][aria-selected="true"] p,
html body .st-key-sd_results_hub [role="tab"][aria-selected="true"] span{{
color:var(--lab-on-accent)!important;-webkit-text-fill-color:var(--lab-on-accent)!important;}}
.lab-brand b{{color:var(--lab-accent)!important;}}
html body .st-key-fav_team [data-baseweb="select"]>div{{border-color:var(--lab-accent)!important;
box-shadow:0 0 0 1px var(--lab-soft)!important;}}
.lineup-cpt span{{background:var(--lab-accent)!important;color:var(--lab-on-accent)!important;
-webkit-text-fill-color:var(--lab-on-accent)!important;border-radius:6px;padding:1px 8px;}}
.answer-kicker{{color:var(--lab-accent)!important;}}
a{{color:var(--lab-accent)!important;}}
.results-hub-hero{{border:1px solid var(--lab-glow)!important;}}
[data-testid="stExpander"] summary:hover{{color:var(--lab-accent)!important;}}
</style>"""
