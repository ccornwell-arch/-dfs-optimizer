"""Moved verbatim from streamlit_app.py (refactor/modularize). No logic changes."""

# moved verbatim from streamlit_app.py line 32
GLOBAL_CSS = """
<style>
/* ---------- Global ---------- */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}
html, body, [class*="css"]  {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 15% 0%, rgba(46, 204, 113, 0.08), transparent 28%),
        radial-gradient(circle at 100% 20%, rgba(0, 184, 255, 0.06), transparent 30%);
}
h1, h2, h3 {
    letter-spacing: -0.02em;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(17,24,39,0.98), rgba(15,23,42,0.98));
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * {
    color: #f8fafc;
}
[data-testid="stSidebar"] label {
    font-weight: 600;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.12) !important;
}

/* ---------- Hero ---------- */
.hero {
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 22px;
    padding: 22px 24px;
    margin-bottom: 18px;
    background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,41,59,0.92));
    box-shadow: 0 10px 30px rgba(0,0,0,0.10);
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0;
}
.hero-sub {
    color: #cbd5e1;
    margin-top: 4px;
}
.hero-chip {
    display: inline-block;
    margin-top: 12px;
    padding: 5px 10px;
    border-radius: 999px;
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.35);
    color: #86efac;
    font-size: 0.82rem;
    font-weight: 700;
}

/* ---------- Section cards ---------- */
.section-card {
    border: 1px solid rgba(148,163,184,0.20);
    border-radius: 18px;
    padding: 18px;
    background: rgba(255,255,255,0.82);
    box-shadow: 0 8px 22px rgba(15,23,42,0.05);
    margin-bottom: 14px;
}
@media (prefers-color-scheme: dark) {
    .section-card {
        background: rgba(15,23,42,0.55);
    }
}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    border: 1px solid rgba(148,163,184,0.20);
    border-radius: 16px;
    padding: 12px 14px;
    background: rgba(255,255,255,0.70);
}
[data-testid="stMetricValue"] {
    font-weight: 800;
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    padding: 0.65rem 1rem;
    border: 1px solid rgba(148,163,184,0.25);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #16a34a, #22c55e);
    border: 0;
    color: white;
}

/* High-contrast download buttons, including iPad Safari */
[data-testid="stDownloadButton"] > button {
    background: #2563eb !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
}
[data-testid="stDownloadButton"] > button * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* ---------- Dataframes ---------- */
[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(148,163,184,0.20);
}

/* ---------- Tabs ---------- */
button[data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    font-weight: 700;
}

/* ---------- Badges ---------- */
.badge {
    display:inline-block;
    padding:4px 8px;
    border-radius:999px;
    font-size:0.78rem;
    font-weight:800;
}
.badge-a {background:#dcfce7;color:#166534;}
.badge-b {background:#dbeafe;color:#1d4ed8;}
.badge-c {background:#fef3c7;color:#92400e;}
.badge-x {background:#fee2e2;color:#991b1b;}

/* ---------- Small helper ---------- */
.muted { color:#64748b; font-size:0.9rem; }

.dfs-chat-user{margin:10px 0 6px auto;padding:12px 14px;max-width:82%;background:#e8f1ff;border:1px solid #bfd5f7;border-radius:16px 16px 4px 16px;color:#172033;}
.dfs-chat-ai{margin:6px auto 14px 0;padding:14px 16px;max-width:92%;background:#f8fafc;border:1px solid #d7dee8;border-radius:16px 16px 16px 4px;color:#172033;line-height:1.5;}
</style>
"""

# moved verbatim from streamlit_app.py line 188
SIDEBAR_V26_CSS = """
<style>
/* V2.6 iPad/sidebar readability fix */
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="input"] *,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] input {
    background: #ffffff !important;
    color: #111827 !important;
    border-color: rgba(148,163,184,0.35) !important;
}

/* Keep labels and section headings light on dark sidebar */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p {
    color: #f8fafc !important;
}

/* Slider/number value text */
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    color: #111827 !important;
    background: #ffffff !important;
}

/* Select dropdown selected value */
[data-testid="stSidebar"] [role="combobox"] {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}
</style>
"""

# moved verbatim from streamlit_app.py line 3775
SETUP_BASE_CSS = """
<style>
:root{--ink:#111827;--muted:#6b7280;--line:rgba(17,24,39,.09);--surface:rgba(255,255,255,.82);--accent:#0071e3;--good:#15803d;--warn:#a16207;}
html,body,[class*="css"]{font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Segoe UI",sans-serif;}
[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 5% -10%,rgba(0,113,227,.08),transparent 28%),radial-gradient(circle at 95% 10%,rgba(99,102,241,.05),transparent 26%),#f5f5f7;}
.block-container{max-width:1480px;padding-top:1.05rem;padding-bottom:3rem;}
[data-testid="stSidebar"]{background:rgba(255,255,255,.95);border-right:1px solid var(--line);}
[data-testid="stSidebar"] *{color:var(--ink)!important;}
[data-testid="stSidebar"] [data-baseweb="select"]>div,[data-testid="stSidebar"] input{background:#fff!important;color:var(--ink)!important;border-color:rgba(17,24,39,.12)!important;border-radius:12px!important;}
.apple-hero{background:linear-gradient(135deg,rgba(255,255,255,.94),rgba(255,255,255,.72));border:1px solid rgba(255,255,255,.85);border-radius:28px;padding:28px 30px;box-shadow:0 16px 50px rgba(17,24,39,.08);backdrop-filter:blur(20px);margin-bottom:16px;}
.apple-eyebrow{font-size:.79rem;font-weight:800;color:var(--accent);text-transform:uppercase;letter-spacing:.09em;}.apple-title{font-size:2.3rem;line-height:1.03;font-weight:800;color:var(--ink);letter-spacing:-.05em;margin-top:5px;}.apple-sub{font-size:1rem;color:var(--muted);margin-top:8px;max-width:820px;}
.pill{display:inline-block;padding:5px 10px;border-radius:999px;background:#eef6ff;color:#0066cc;font-size:.78rem;font-weight:750;margin-top:13px;}
.card-title{color:var(--ink);font-size:1.08rem;font-weight:780;letter-spacing:-.02em;margin-top:5px;}.card-sub{color:var(--muted);font-size:.92rem;margin-bottom:10px;}
[data-testid="stMetric"]{background:rgba(255,255,255,.75);border:1px solid rgba(17,24,39,.07);border-radius:18px;padding:12px 14px;box-shadow:0 7px 24px rgba(17,24,39,.04);}
[data-testid="stDataFrame"]{border-radius:18px;overflow:hidden;border:1px solid rgba(17,24,39,.08);}
.stButton>button{border-radius:13px;font-weight:750;padding:.65rem 1rem}.stButton>button[kind="primary"]{background:#0071e3;color:#fff;border:0;}
button[data-baseweb="tab"]{font-weight:750;}
hr{border-color:rgba(17,24,39,.07)!important;}

/* V6.1 iPad sidebar fix.
   IMPORTANT: let Streamlit own the sidebar transform/width so its native << button
   can actually collapse it. The main canvas then expands into the released space. */
[data-testid="stMain"], [data-testid="stMainBlockContainer"], .block-container{max-width:100%!important;width:100%!important;}
section[data-testid="stSidebar"]{box-shadow:14px 0 38px rgba(0,0,0,.18);}
section[data-testid="stSidebar"][aria-expanded="false"]{box-shadow:none!important;}
</style>
"""

# moved verbatim from streamlit_app.py line 3803
SETUP_V43_CSS = """
<style>
/* V4.3 iPad readability + scenario engine */
[data-testid="stSidebar"] {
    background: #f5f5f7 !important;
    border-right: 1px solid rgba(17,24,39,.08) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label *,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] * {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [role="combobox"] {
    background: #ffffff !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    border-color: rgba(17,24,39,.14) !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] * {
    -webkit-text-fill-color: initial !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color:#111827 !important;
    -webkit-text-fill-color:#111827 !important;
}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label * {
    opacity: 1 !important;
    color:#374151 !important;
    -webkit-text-fill-color:#374151 !important;
    font-weight:600 !important;
}

/* Make segmented control choices readable on the light sidebar. */
[data-testid="stSidebar"] [data-baseweb="button-group"] button,
[data-testid="stSidebar"] [data-baseweb="button-group"] button * {
    color:#111827 !important;
    -webkit-text-fill-color:#111827 !important;
}
/* Do not force sidebar width on iPad. A fixed width prevents Streamlit's
   collapse transform from taking effect in Safari. */
</style>
"""

# moved verbatim from streamlit_app.py line 3861
SETUP_V5_CSS = """
<style>
/* DFS Lab V5 — dark iPad control deck */
:root{--v5-bg:#f5f5f7;--v5-panel:#ffffff;--v5-line:rgba(0,0,0,.10);--v5-text:#1d1d1f;--v5-muted:#6e6e73;}
[data-testid="stAppViewContainer"]{background:linear-gradient(180deg,#fbfbfd 0%,#f5f5f7 52%,#f2f2f4 100%)!important;color:var(--v5-text)!important;}
[data-testid="stHeader"]{background:transparent!important}.block-container{max-width:1480px;padding-top:1rem}
.apple-hero{background:rgba(255,255,255,.88)!important;border:1px solid rgba(0,0,0,.08)!important;box-shadow:0 18px 50px rgba(0,0,0,.07)!important}
.apple-title,.card-title,h1,h2,h3,h4{color:var(--v5-text)!important}.apple-sub,.card-sub,.muted{color:var(--v5-muted)!important}.pill{background:rgba(47,129,247,.15)!important;color:#8ec5ff!important}
[data-testid="stMetric"]{background:#fff!important;border:1px solid var(--v5-line)!important;box-shadow:0 8px 24px rgba(0,0,0,.04)!important}[data-testid="stMetricLabel"],[data-testid="stMetricValue"]{color:var(--v5-text)!important}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,#1769d2,#2f81f7)!important;box-shadow:0 8px 24px rgba(47,129,247,.22)}
.lineup-card{background:#fff;border:1px solid var(--v5-line);border-radius:18px;padding:16px 18px;margin:10px 0;box-shadow:0 10px 30px rgba(0,0,0,.05)}.lineup-rank{font-size:.78rem;font-weight:800;color:#0071e3}.lineup-rank span{background:rgba(0,113,227,.10);padding:3px 7px;border-radius:999px}.lineup-cpt{font-size:1.15rem;font-weight:800;color:#1d1d1f;margin-top:7px}.lineup-flex{font-size:.92rem;color:#424245;margin-top:5px}.lineup-meta{font-size:.80rem;color:#6e6e73;margin-top:9px}
@media(max-width:800px){.apple-title{font-size:1.85rem!important}.apple-hero{padding:21px!important}.block-container{padding-left:.8rem!important;padding-right:.8rem!important}.lineup-card{padding:14px}}
</style>
"""

# moved verbatim from streamlit_app.py line 3877
SETUP_LABEL_CSS = """<style>
[data-testid="stAppViewContainer"] label {color:#1d1d1f !important;}
.stCaption, [data-testid="stCaptionContainer"], .card-sub {color:#6e6e73 !important;}
[data-testid="stTabs"] button p {color:#6e6e73 !important;}
[data-testid="stTabs"] button[aria-selected="true"] p {color:#1d1d1f !important;}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {color:#303033;}
[data-testid="stSidebarCollapseButton"] button{background:#1d1d1f!important;color:white!important;border-radius:999px!important;min-width:42px!important;min-height:42px!important;box-shadow:0 4px 16px rgba(0,0,0,.18)!important;}
[data-testid="stSidebarCollapseButton"] svg{fill:white!important;color:white!important;}
</style>"""

# moved verbatim from streamlit_app.py line 3886
SETUP_V633_CSS = r"""
<style>
/* DFS LAB V6.3.3 — product design system */
:root{--lab-bg:#e9edf3;--lab-canvas:#f2f4f7;--lab-panel:#ffffff;--lab-sidebar:#e3e8ef;--lab-ink:#101828;--lab-muted:#667085;--lab-line:#d6dce5;--lab-blue:#1267d6;--lab-blue2:#2f7eea;}
[data-testid="stAppViewContainer"]{background:linear-gradient(135deg,#e6ebf2 0%,#f4f6f9 46%,#e9eef5 100%)!important;color:var(--lab-ink)!important;}
[data-testid="stHeader"]{background:rgba(242,244,247,.82)!important;backdrop-filter:blur(18px)!important;border-bottom:1px solid rgba(16,24,40,.06)!important;}
.block-container{max-width:1500px!important;padding-top:1.25rem!important;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#e0e6ee,#edf0f5)!important;border-right:1px solid #cbd3df!important;box-shadow:10px 0 32px rgba(25,39,62,.08)!important;}
section[data-testid="stSidebar"] *{color:var(--lab-ink)!important;-webkit-text-fill-color:initial!important;}
section[data-testid="stSidebar"] [data-baseweb="select"]>div,section[data-testid="stSidebar"] input{background:rgba(255,255,255,.82)!important;border:1px solid #cbd3df!important;box-shadow:none!important;}
.apple-hero{background:linear-gradient(120deg,#111b2d 0%,#182944 62%,#173760 100%)!important;border:1px solid rgba(255,255,255,.08)!important;box-shadow:0 20px 50px rgba(26,45,75,.16)!important;border-radius:24px!important;padding:28px 32px!important;}
.apple-hero .apple-eyebrow{color:#79b8ff!important}.apple-hero .apple-title{color:#fff!important;font-size:2.55rem!important}.apple-hero .apple-sub{color:#c9d4e3!important}.apple-hero .pill{background:rgba(58,139,253,.16)!important;color:#9ac8ff!important;border:1px solid rgba(121,184,255,.22)!important;}
[data-testid="stMetric"]{background:rgba(255,255,255,.88)!important;border:1px solid var(--lab-line)!important;border-radius:16px!important;box-shadow:0 8px 24px rgba(31,50,81,.06)!important;}
[data-testid="stDataFrame"]{background:#fff!important;border:1px solid var(--lab-line)!important;border-radius:16px!important;box-shadow:0 8px 26px rgba(31,50,81,.05)!important;}
[data-testid="stTabs"]{background:transparent!important;}
button[data-baseweb="tab"]{padding-top:.7rem!important;padding-bottom:.7rem!important;}
.stButton>button{border-radius:12px!important;min-height:44px!important;font-weight:750!important;}
.stButton>button[kind="primary"],[data-testid="stDownloadButton"]>button{background:linear-gradient(90deg,var(--lab-blue),var(--lab-blue2))!important;color:#fff!important;-webkit-text-fill-color:#fff!important;border:0!important;box-shadow:0 8px 20px rgba(18,103,214,.20)!important;}
.stButton>button[kind="primary"] *,[data-testid="stDownloadButton"]>button *{color:#fff!important;-webkit-text-fill-color:#fff!important;opacity:1!important;}
/* Make Streamlit's native collapse control look like a product control, not a black orb. */
[data-testid="stSidebarCollapseButton"] button{background:var(--lab-blue)!important;color:#fff!important;border:1px solid rgba(255,255,255,.5)!important;border-radius:10px!important;min-width:54px!important;min-height:40px!important;box-shadow:0 6px 18px rgba(18,103,214,.22)!important;}
[data-testid="stSidebarCollapseButton"] svg{fill:#fff!important;color:#fff!important;stroke:#fff!important;}
[data-testid="stSidebarCollapseButton"] button:hover{background:#0b5fc8!important;}
.card-title,h1,h2,h3,h4{color:var(--lab-ink)!important}.card-sub,.muted,[data-testid="stCaptionContainer"]{color:var(--lab-muted)!important;}
.lineup-card{background:#fff!important;border:1px solid var(--lab-line)!important;box-shadow:0 10px 28px rgba(31,50,81,.06)!important;}
@media(max-width:900px){.apple-hero{padding:22px 24px!important}.apple-hero .apple-title{font-size:2.1rem!important}.block-container{padding-left:1rem!important;padding-right:1rem!important;}}
</style>
"""

# moved verbatim from streamlit_app.py line 3915
SETUP_V64_CSS = r"""
<style>
/* V6.4 final visual layer — medium slate, depth, motion, no spreadsheet wall */
:root{--fun-bg:#cfd7e2;--fun-canvas:#e3e8ef;--fun-panel:#f4f6f9;--fun-card:#ffffff;--fun-ink:#162033;--fun-muted:#667085;--fun-line:#c4cedb;--fun-blue:#1677ff;--fun-purple:#7457ff;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#d9e0e8!important;}
[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 82% 2%,rgba(22,119,255,.12),transparent 25%),linear-gradient(135deg,#cfd7e2 0%,#e5eaf0 45%,#d5dde7 100%)!important;}
[data-testid="stHeader"]{background:rgba(217,224,232,.92)!important;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#c8d1dd,#d8e0e9)!important;border-right:1px solid #b8c3d1!important;}
.apple-hero{background:linear-gradient(115deg,#26364d 0%,#304967 55%,#285d91 100%)!important;box-shadow:0 18px 45px rgba(37,56,82,.20)!important;}
[data-testid="stMetric"]{background:rgba(247,249,252,.92)!important;border-color:#c7d0dc!important;box-shadow:0 8px 22px rgba(39,55,78,.08)!important;transition:transform .18s ease,box-shadow .18s ease!important;}
[data-testid="stMetric"]:hover{transform:translateY(-2px);box-shadow:0 12px 28px rgba(39,55,78,.13)!important;}
[data-testid="stDataFrame"]{background:#f8fafc!important;border:1px solid #c2ccd9!important;box-shadow:0 8px 24px rgba(39,55,78,.08)!important;}
[data-testid="stVerticalBlockBorderWrapper"]>div{background:rgba(246,248,251,.72);border-color:#c5cfdb!important;border-radius:18px!important;box-shadow:0 8px 24px rgba(39,55,78,.07);}
.lineup-card{background:linear-gradient(135deg,#fbfcfe,#edf3f9)!important;border:1px solid #b9c7d7!important;border-left:5px solid var(--fun-blue)!important;border-radius:20px!important;padding:18px 20px!important;margin:14px 0!important;box-shadow:0 10px 28px rgba(39,55,78,.11)!important;transition:transform .2s ease,box-shadow .2s ease!important;animation:cardIn .35s ease both;}
.lineup-card:hover{transform:translateY(-3px) scale(1.004);box-shadow:0 16px 36px rgba(39,55,78,.16)!important;}
.lineup-rank{font-size:.78rem!important;letter-spacing:.06em;text-transform:uppercase}.lineup-cpt{font-size:1.3rem!important;color:#172033!important}.lineup-flex{color:#46556a!important;line-height:1.6}.lineup-meta{color:#65758a!important}
[data-testid="stTabs"] [data-baseweb="tab-list"]{background:rgba(244,247,250,.62);padding:6px;border:1px solid #c2ccd8;border-radius:15px;gap:3px;}
[data-testid="stTabs"] button[data-baseweb="tab"]{border-radius:10px!important;padding:.65rem .85rem!important;transition:all .18s ease!important;}
[data-testid="stTabs"] button[aria-selected="true"]{background:#fff!important;box-shadow:0 5px 14px rgba(38,55,78,.12)!important;}
[data-baseweb="select"]>div,[data-baseweb="input"]>div,textarea{background:#f8fafc!important;border-color:#b9c5d3!important;border-radius:13px!important;box-shadow:inset 0 1px 2px rgba(20,32,50,.03)!important;}
.stButton>button{transition:transform .14s ease,box-shadow .14s ease,filter .14s ease!important;}.stButton>button:active{transform:scale(.975)!important}.stButton>button:hover{transform:translateY(-1px);}
.agent-status{display:flex;align-items:center;gap:9px;flex-wrap:wrap;background:rgba(246,249,252,.78);border:1px solid #bdc9d7;border-radius:14px;padding:10px 13px;margin:9px 0 12px;color:#526176;font-size:.82rem}.agent-status b{color:#1e2a3b}.agent-dot{width:9px;height:9px;border-radius:50%;display:inline-block}.agent-dot.live{background:#22c55e;box-shadow:0 0 0 5px rgba(34,197,94,.12);animation:pulse 1.8s infinite}.agent-dot.local{background:#f59e0b}.answer-kicker{font-size:.72rem;font-weight:850;letter-spacing:.11em;color:#5d6c80;margin:18px 0 7px}.chat-user{background:#dce9fb;border:1px solid #b8cdeb;border-radius:16px 16px 5px 16px;padding:13px 15px;color:#1d2c40;margin-bottom:12px;animation:slideUp .25s ease}.chat-label,.chat-agent-label{font-size:.68rem;font-weight:850;letter-spacing:.1em;color:#4670a7;margin-bottom:5px}.chat-agent-label{color:#6c55c7;margin-top:6px}
[data-testid="stProgress"]>div>div{background:linear-gradient(90deg,var(--fun-blue),var(--fun-purple))!important;}
@keyframes cardIn{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:none}}@keyframes slideUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}@keyframes pulse{0%,100%{opacity:1}50%{opacity:.45}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
@media(max-width:900px){.lineup-card{padding:16px!important}.block-container{padding-left:.85rem!important;padding-right:.85rem!important}}
</style>
"""

# moved verbatim from streamlit_app.py line 3945
SETUP_NATIVE_CC_CSS = """<style>
/* DFS LAB native command center — no dependency on Streamlit sidebar */
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"]{display:none!important;}
.command-strip{display:flex;justify-content:space-between;align-items:center;gap:16px;margin:18px 0 8px;padding:14px 18px;border:1px solid #b8c7d9;border-radius:16px;background:linear-gradient(110deg,#f8fbff,#e7f0fb);box-shadow:0 8px 22px rgba(37,56,82,.08);color:#1a2a40;animation:slideUp .28s ease both}.command-live{font-size:.72rem;letter-spacing:.08em;color:#0b72e7;margin-right:8px}.command-copy{color:#63748a;margin-left:12px;font-size:.86rem}.command-arrow{font-size:.72rem;font-weight:850;letter-spacing:.06em;color:#0b67cf;white-space:nowrap}
[data-testid="stExpander"]{border:1px solid #b9c7d8!important;border-radius:17px!important;background:rgba(248,250,253,.88)!important;box-shadow:0 8px 22px rgba(39,55,78,.07)!important;margin-bottom:12px!important;overflow:hidden!important}
[data-testid="stExpander"] summary{min-height:58px!important;padding:0 16px!important;font-weight:850!important;color:#17283e!important;-webkit-text-fill-color:#17283e!important;background:linear-gradient(100deg,#f9fbfe,#edf3f9)!important}
[data-testid="stExpander"] summary:hover{background:linear-gradient(100deg,#f4f8fd,#e4eef9)!important}
[data-testid="stExpander"] summary svg{color:#1267d6!important;fill:#1267d6!important;width:22px!important;height:22px!important}
@media(max-width:900px){.command-copy{display:none}.command-strip{padding:12px 14px}.command-arrow{font-size:.66rem}}
</style>"""

# moved verbatim from streamlit_app.py line 3956
SETUP_COUNT_READOUT_CSS = """<style>
.lineup-count-readout{margin-top:8px;padding:10px 12px;border-radius:12px;background:#eaf3ff;border:1px solid #b8d4f6;color:#24364b}.lineup-count-readout b{font-size:1.25rem;color:#1268c4}
.agent-scenario{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:13px 15px;margin:10px 0;border-radius:14px;background:#e8f3ff;border:1px solid #9fc8f5;color:#183653}.agent-scenario b{color:#0a62b7}.scenario-proposal{padding:15px;margin:12px 0;border-radius:15px;background:#f3efff;border:1px solid #c7b9f4;color:#252a3a}.scenario-proposal span{color:#68758a;font-size:.82rem}
[data-testid=\"stExpander\"] label,
[data-testid=\"stExpander\"] label p,
[data-testid=\"stExpander\"] [data-testid=\"stWidgetLabel\"] p,
[data-testid=\"stExpander\"] p{
  color:#253247!important;
  -webkit-text-fill-color:#253247!important;
  opacity:1!important;
}
.stRadio label,.stRadio label p{
  color:#253247!important;
  -webkit-text-fill-color:#253247!important;
  opacity:1!important;
}
</style>"""

# moved verbatim from streamlit_app.py line 3973
SETUP_IPAD_SURFACE_CSS = """<style>
/* DFS LAB iPad surface/background pass */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #dbe4ef 0%, #e7edf5 42%, #dfe8f2 100%) !important;
}
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] .main {
  background: transparent !important;
}
[data-testid="stAppViewContainer"] .block-container {
  background: transparent !important;
}
[data-testid="stExpander"],
[data-testid="stVerticalBlockBorderWrapper"] > div,
[data-testid="stForm"] {
  background: rgba(248,250,252,0.94) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stNumberInput"] > div > div,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  background: #eef3f9 !important;
}
</style>"""

# moved verbatim from streamlit_app.py line 3998
SETUP_IPAD_CONTRAST_CSS = """<style>
/* Final iPad contrast pass */
html, body, [data-testid="stAppViewContainer"] {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif !important;
}
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,
[data-testid="stAppViewContainer"] label p,
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p {
  color:#253247 !important;
  -webkit-text-fill-color:#253247 !important;
  opacity:1 !important;
  font-weight:600 !important;
}
[data-testid="stSegmentedControl"] label,
[data-testid="stSegmentedControl"] button,
[data-testid="stSegmentedControl"] [role="radio"] {
  background:#243041 !important;
  border-color:#506078 !important;
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
  opacity:1 !important;
}
[data-testid="stSegmentedControl"] label *,
[data-testid="stSegmentedControl"] button *,
[data-testid="stSegmentedControl"] [role="radio"] * {
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
  opacity:1 !important;
  font-weight:700 !important;
}
[data-testid="stSegmentedControl"] label:has(input:checked),
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[aria-pressed="true"],
[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] {
  background:#2563eb !important;
  border-color:#2563eb !important;
}
[data-baseweb="button-group"] button,
[data-baseweb="button-group"] button * {
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
  opacity:1 !important;
}
/* Selected segment must be unmistakable at a glance: bright pill, dark bold text.
   (Unselected stays dark navy with white text, so the two states can't be confused.) */
[data-testid="stSegmentedControl"] button[aria-pressed="true"],
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] label:has(input:checked),
[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] {
  background:#9adcff !important;
  border-color:#9adcff !important;
  box-shadow:0 0 0 2px rgba(154,220,255,.45) !important;
}
[data-testid="stSegmentedControl"] button[aria-pressed="true"] *,
[data-testid="stSegmentedControl"] button[aria-checked="true"] *,
[data-testid="stSegmentedControl"] label:has(input:checked) *,
[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] * {
  color:#0b1a2e !important;
  -webkit-text-fill-color:#0b1a2e !important;
  font-weight:800 !important;
  opacity:1 !important;
}
</style>"""

# moved verbatim from streamlit_app.py line 4043
SETUP_PRIMARY_BUTTON_CSS = """<style>
/* Final primary-button contrast */
.stButton > button[kind="primary"],
.stButton > button[kind="primary"] * {
  background: #1559c7 !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  border-color: #1559c7 !important;
  font-weight: 800 !important;
}
.stButton > button[kind="primary"]:hover {
  background: #0f4cae !important;
}
</style>"""

# moved verbatim from streamlit_app.py line 4058
SETUP_DARK_CC_CSS = """<style>
/* DFS LAB DARK COMMAND CENTER · final visual cascade */
:root{--lab-bg:#07111d;--lab-panel:#0c1a29;--lab-panel2:#102235;--lab-border:#243b53;--lab-blue:#087cff;--lab-text:#f4f8fc;--lab-muted:#9db0c5;--lab-green:#18c96e}
html,body,[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 72% -10%,#102a46 0%,#081522 34%,#050d17 100%)!important;color:var(--lab-text)!important}
[data-testid="stHeader"]{background:rgba(5,13,23,.88)!important;border-bottom:1px solid rgba(74,112,151,.22)!important;backdrop-filter:blur(16px)!important}
[data-testid="stAppViewContainer"] .block-container{max-width:1560px!important;padding-top:.8rem!important;padding-left:1.15rem!important;padding-right:1.15rem!important}
[data-testid="stAppViewContainer"] h1,[data-testid="stAppViewContainer"] h2,[data-testid="stAppViewContainer"] h3,[data-testid="stAppViewContainer"] h4{color:#f4f8fc!important;-webkit-text-fill-color:#f4f8fc!important}
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,[data-testid="stAppViewContainer"] label p,[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p,.stCaption{color:#9db0c5!important;-webkit-text-fill-color:#9db0c5!important}
.lab-appbar{display:flex;align-items:center;gap:22px;padding:10px 4px 15px;border-bottom:1px solid #1b3147;margin-bottom:12px}.lab-brand{display:flex;align-items:center;gap:6px;font-size:1.8rem;font-weight:900;letter-spacing:-.04em;color:#fff}.lab-brand b{color:#1493ff}.lab-flask{font-size:1.75rem;color:#49b7ff}.lab-appbar-copy{display:flex;flex-direction:column;gap:1px;border-left:1px solid #294057;padding-left:20px}.lab-appbar-copy strong{font-size:.76rem;letter-spacing:.12em;color:#d8e6f5}.lab-appbar-copy span{font-size:.72rem;color:#7f96ad}.lab-live{margin-left:auto;padding:7px 11px;border:1px solid #274762;border-radius:999px;font-size:.72rem;font-weight:850;letter-spacing:.06em;color:#b7c9da}.lab-live i{display:inline-block;width:8px;height:8px;border-radius:50%;background:#18c96e;box-shadow:0 0 0 4px rgba(24,201,110,.12);margin-right:6px}
.lab-flask-img{width:34px;height:34px;border-radius:9px;object-fit:cover;flex:0 0 auto;box-shadow:0 0 0 1px #1b3147}
.command-strip{background:linear-gradient(100deg,#0c1c2c,#10263a)!important;border:1px solid #25425e!important;color:#eaf4ff!important;box-shadow:0 10px 28px rgba(0,0,0,.20)!important}.command-live,.command-arrow{color:#55adff!important}.command-copy{color:#89a0b7!important}
[data-testid="stExpander"],[data-testid="stVerticalBlockBorderWrapper"]>div,[data-testid="stForm"]{background:rgba(11,27,43,.94)!important;border:1px solid #203a53!important;box-shadow:0 10px 26px rgba(0,0,0,.18)!important}
[data-testid="stExpander"] summary{background:linear-gradient(100deg,#0d1d2d,#10263a)!important;color:#eaf4ff!important;-webkit-text-fill-color:#eaf4ff!important}
[data-testid="stExpander"] summary:hover{background:#122a40!important}
[data-testid="stExpander"] p,[data-testid="stExpander"] label,[data-testid="stExpander"] label p{color:#b8c9da!important;-webkit-text-fill-color:#b8c9da!important}
[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,[data-testid="stNumberInput"] input{background:#0b1927!important;border-color:#29445f!important;color:#eef6ff!important;-webkit-text-fill-color:#eef6ff!important}
[data-baseweb="popover"],[role="listbox"]{background:#0c1a29!important;color:#eef6ff!important}[role="option"]{color:#eef6ff!important}
[data-testid="stMetric"]{background:linear-gradient(145deg,#0d1d2d,#102338)!important;border:1px solid #29445e!important;box-shadow:0 10px 26px rgba(0,0,0,.20)!important;min-height:112px}
[data-testid="stMetricLabel"] p{color:#a8bbce!important;-webkit-text-fill-color:#a8bbce!important;font-weight:750!important}[data-testid="stMetricValue"]{color:#fff!important;-webkit-text-fill-color:#fff!important;font-weight:900!important}
[data-testid="stDataFrame"]{background:#091725!important;border:1px solid #29445e!important;box-shadow:0 10px 24px rgba(0,0,0,.18)!important}
.stButton>button,[data-testid="stDownloadButton"]>button,[data-testid="stFormSubmitButton"] button{background:#102338!important;border:1px solid #31516f!important;color:#eaf4ff!important;-webkit-text-fill-color:#eaf4ff!important;box-shadow:none!important}.stButton>button *,[data-testid="stDownloadButton"]>button *,[data-testid="stFormSubmitButton"] button *{color:#eaf4ff!important;-webkit-text-fill-color:#eaf4ff!important}
.stButton>button[kind="primary"],[data-testid="stFormSubmitButton"] button[kind="primary"],[data-testid="stDownloadButton"]>button{background:linear-gradient(100deg,#0869e8,#0b86ff)!important;border-color:#1692ff!important}
html body [data-testid="stTabs"] [role="tablist"],html body [data-testid="stTabs"] [data-baseweb="tab-list"]{background:#0a1724!important;border:1px solid #213b53!important;box-shadow:0 10px 28px rgba(0,0,0,.22)!important}
html body [data-testid="stTabs"] [role="tab"],html body [data-testid="stTabs"] button[data-baseweb="tab"]{background:transparent!important;border:1px solid transparent!important}
html body [data-testid="stTabs"] [role="tab"] p,html body [data-testid="stTabs"] [role="tab"] span,html body [data-testid="stTabs"] button[data-baseweb="tab"] p{color:#9fb2c6!important;-webkit-text-fill-color:#9fb2c6!important}
html body [data-testid="stTabs"] [role="tab"][aria-selected="true"],html body [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"]{background:linear-gradient(100deg,#075fd8,#0b82ff)!important;border-color:#198eff!important;box-shadow:0 7px 20px rgba(0,112,255,.24)!important}
html body [data-testid="stTabs"] [role="tab"][aria-selected="true"] p,html body [data-testid="stTabs"] [role="tab"][aria-selected="true"] span,html body [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p{color:#fff!important;-webkit-text-fill-color:#fff!important}
.st-key-sd_results_hub [role="tablist"]{background:#07131f!important;border:1px solid #28506f!important;padding:8px!important}.st-key-sd_results_hub [role="tab"]{min-height:66px!important;background:#0d2031!important;border:1px solid #29455e!important}.st-key-sd_results_hub [role="tab"][aria-selected="true"]{background:linear-gradient(110deg,#0664e9,#078cff)!important;border-color:#31a0ff!important;box-shadow:0 0 0 1px rgba(49,160,255,.25),0 10px 30px rgba(0,105,240,.28)!important}
.results-hub-hero{background:linear-gradient(115deg,#0b1d2f 0%,#0d2844 58%,#0a3d70 100%)!important;border:1px solid #147fe6!important;box-shadow:0 0 0 1px rgba(25,143,255,.18),0 18px 45px rgba(0,0,0,.25)!important}.results-hub-kicker{color:#65b8ff!important}.results-hub-title{color:#fff!important}.results-hub-sub{color:#b9cce0!important}.results-hub-sub b{color:#fff!important}.results-view-title{color:#f4f8fc!important}.results-view-sub{color:#8fa5ba!important}
.lineup-card{background:linear-gradient(145deg,#0c1c2b,#102338)!important;border:1px solid #29465f!important;border-left:1px solid #29465f!important;box-shadow:0 12px 28px rgba(0,0,0,.22)!important;color:#eef6ff!important}.lineup-card-grid{min-height:252px!important;position:relative!important;overflow:hidden!important}.lineup-card-grid:before{content:"";position:absolute;left:0;right:0;top:0;height:3px;background:linear-gradient(90deg,#0788ff,#16c784)}
.lineup-card-head{display:flex;justify-content:space-between;align-items:center}.lineup-rank{display:inline-flex;padding:5px 10px;border-radius:999px;background:#ffbd3e;color:#1a2028!important;font-size:.78rem!important;font-weight:950!important}.lineup-grade{display:inline-flex;padding:5px 9px;border-radius:7px;background:#17a95d;color:white;font-weight:900}.lineup-points{font-size:1.55rem;font-weight:950;color:#fff;margin:12px 0 8px}.lineup-points small{font-size:.76rem;color:#8fa7be;font-weight:700}.lineup-cpt{color:#f4f8fc!important;font-size:1.08rem!important}.lineup-cpt span{display:inline-block;padding:3px 6px;margin-right:6px;border-radius:5px;border:1px solid #e5b928;color:#ffe35a!important;font-size:.66rem;font-weight:900}.lineup-flex{color:#b9c9d9!important;line-height:1.55;margin-top:9px}.lineup-meta{color:#7f96ac!important;border-top:1px solid #223b52;padding-top:9px;margin-top:11px}.lineup-world{margin-top:10px;color:#68b7ff;font-size:.78rem;font-weight:750}
.world-board{background:#0a1826;border:1px solid #28435b;border-radius:17px;padding:14px 16px;box-shadow:0 10px 26px rgba(0,0,0,.18)}.world-row{display:grid;grid-template-columns:minmax(155px,1.4fr) minmax(180px,3fr) 90px;gap:12px;align-items:center;padding:9px 0;border-bottom:1px solid rgba(55,82,105,.35)}.world-row:last-child{border-bottom:0}.world-label{color:#dbe8f4;font-size:.88rem;font-weight:700}.world-track{height:11px;border-radius:999px;background:#162a3d;overflow:hidden}.world-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#086fff,#14a1ff)}.world-count{text-align:right;color:#fff;font-weight:850}.world-count span{color:#7f97ae;font-weight:650;margin-left:5px}
html body .intel-card{background:linear-gradient(135deg,#0c1d2d,#10263a)!important;border-color:#294760!important}html body .intel-kicker{color:#54adff!important}html body .intel-big{color:#f4f8fc!important}html body .intel-copy{color:#9fb2c6!important}.agent-status,.dfs-chat-ai,.chat-agent{background:#0b1b2b!important;border-color:#29445e!important;color:#c8d8e8!important}.dfs-chat-user,.chat-user{background:#0d3158!important;border-color:#19528b!important;color:#eef6ff!important}
.lineup-count-readout{background:#0b2238!important;border-color:#23517b!important;color:#a9bed2!important}.lineup-count-readout b{color:#3ca4ff!important}hr{border-color:#20384e!important}
@media(max-width:900px){.lab-appbar-copy{display:none}.lab-brand{font-size:1.5rem}.lab-live{font-size:.65rem}.world-row{grid-template-columns:1.3fr 1.7fr 64px;gap:8px}.lineup-card-grid{min-height:0!important}}
</style>"""

# moved verbatim from streamlit_app.py line 4207
MAIN_TABS_CSS = """
        <style>
        /* NOTE: sticky positioning for the primary tab bar lives ONLY in the
           authoritative rule in RCC_DARK_CSS (top:0, solid background, no
           backdrop-filter — the translucent attempt broke sticky on iPad
           Safari). This legacy block keeps visual styling only. */
        [data-testid="stTabs"] [data-baseweb="tab-list"]{
            background:rgba(229,234,240,.96)!important;
            border:1px solid #b9c5d3!important;border-radius:14px!important;padding:6px!important;
            box-shadow:0 8px 22px rgba(37,52,76,.12)!important;margin-bottom:12px!important;
        }
        [data-testid="stTabs"] button[data-baseweb="tab"]{
            min-height:44px!important;border-radius:10px!important;padding:.65rem 1rem!important;
        }
        [data-testid="stTabs"] button[data-baseweb="tab"] p{
            font-weight:850!important;font-size:.96rem!important;color:#334155!important;
        }
        [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"]{
            background:#1559c7!important;box-shadow:0 5px 14px rgba(21,89,199,.25)!important;
        }
        [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p{
            color:#fff!important;-webkit-text-fill-color:#fff!important;
        }
        .intel-card{background:linear-gradient(135deg,#eef5ff,#f8fbff);border:1px solid #b8cae3;
            border-radius:18px;padding:16px 18px;margin:10px 0;box-shadow:0 8px 22px rgba(37,52,76,.07)}
        .intel-kicker{font-size:.72rem;font-weight:850;letter-spacing:.09em;color:#1559c7;text-transform:uppercase}
        .intel-big{font-size:1.18rem;font-weight:850;color:#172033;margin-top:4px}
        .intel-copy{font-size:.9rem;color:#59677a;line-height:1.45;margin-top:4px}
        [data-testid="stTabs"] [role="tablist"]{
            background:#e5ebf3!important;border:1px solid #aebccc!important;
            border-radius:14px!important;padding:6px 8px!important;
            box-shadow:0 8px 22px rgba(37,52,76,.16)!important;
        }
        [data-testid="stTabs"] [role="tab"]{
            opacity:1!important;visibility:visible!important;min-height:46px!important;
            border-radius:10px!important;
        }
        [data-testid="stTabs"] [role="tab"] p,
        [data-testid="stTabs"] [role="tab"] span{
            color:#26364c!important;-webkit-text-fill-color:#26364c!important;
            opacity:1!important;font-weight:800!important;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"]{
            background:#1559c7!important;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] p,
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] span{
            color:#fff!important;-webkit-text-fill-color:#fff!important;
        }
        .stButton > button[kind="primary"],
        .stButton > button[kind="primary"] *,
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stFormSubmitButton"] button *{
            color:#fff!important;-webkit-text-fill-color:#fff!important;font-weight:850!important;
        }
        </style>
        """

# moved verbatim from streamlit_app.py line 4750
SHOWDOWN_SHELL_CSS = """
        <style>
        /* Keep Showdown on the same command-tab shell as Classic.
           Sticky positioning lives ONLY in the authoritative RCC_DARK_CSS rule. */
        [data-testid="stTabs"] [role="tablist"]{
            background:#e5ebf3!important;border:1px solid #aebccc!important;
            border-radius:14px!important;padding:6px 8px!important;
            box-shadow:0 8px 22px rgba(37,52,76,.16)!important;
            overflow-x:auto!important;flex-wrap:nowrap!important;
        }
        [data-testid="stTabs"] [role="tab"]{
            opacity:1!important;visibility:visible!important;min-height:46px!important;
            border-radius:10px!important;padding:.65rem 1rem!important;
        }
        [data-testid="stTabs"] [role="tab"] p,
        [data-testid="stTabs"] [role="tab"] span{
            color:#26364c!important;-webkit-text-fill-color:#26364c!important;
            opacity:1!important;font-weight:800!important;white-space:nowrap!important;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"]{background:#1559c7!important;}
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] p,
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] span{
            color:#fff!important;-webkit-text-fill-color:#fff!important;
        }
        </style>
        """

# moved verbatim from streamlit_app.py line 5708
MAIN_V634_SHELL_CSS = r"""
<style>
/* DFS LAB V6.3.4 — iPad-first product shell */
:root{--shell:#171c24;--shell2:#202733;--surface:#272f3c;--surface2:#303947;--surface3:#394454;--ink:#f6f8fb;--muted:#b5bfcd;--line:rgba(255,255,255,.10);--blue:#2f80ff;--blue2:#5a9cff;}
[data-testid="stAppViewContainer"]{background:linear-gradient(145deg,#171c24 0%,#202733 55%,#252d39 100%)!important;color:var(--ink)!important;}
[data-testid="stHeader"]{background:rgba(23,28,36,.88)!important;border-bottom:1px solid var(--line)!important;backdrop-filter:blur(18px)!important;}
.block-container{max-width:1380px!important;padding-top:1rem!important;}
section[data-testid="stSidebar"]{background:#1c222c!important;border-right:1px solid var(--line)!important;box-shadow:12px 0 35px rgba(0,0,0,.18)!important;}
section[data-testid="stSidebar"] *, [data-testid="stAppViewContainer"] label{color:var(--ink)!important;-webkit-text-fill-color:initial!important;}
.apple-hero{background:linear-gradient(120deg,#202b3a,#18345b 70%,#164579)!important;border:1px solid rgba(255,255,255,.10)!important;box-shadow:0 20px 55px rgba(0,0,0,.22)!important;}
.apple-hero .apple-title{color:#fff!important}.apple-hero .apple-sub{color:#c9d5e5!important}.apple-hero .apple-eyebrow{color:#86bcff!important}
h1,h2,h3,h4,h5,.card-title,[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{color:var(--ink)!important;}
.card-sub,.muted,[data-testid="stCaptionContainer"],small{color:var(--muted)!important;}
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:6px;background:#1b222c;border:1px solid var(--line);padding:6px;border-radius:14px;}
[data-testid="stTabs"] button[data-baseweb="tab"]{border-radius:10px!important;padding:.65rem 1rem!important;}
[data-testid="stTabs"] button[data-baseweb="tab"] p{color:#aeb9c8!important;font-weight:700!important;}
[data-testid="stTabs"] button[aria-selected="true"]{background:#303b4a!important;}
[data-testid="stTabs"] button[aria-selected="true"] p{color:#fff!important;}
[data-baseweb="select"]>div,[data-baseweb="input"]>div,input,textarea{background:#303947!important;border-color:rgba(255,255,255,.15)!important;color:#fff!important;border-radius:12px!important;}
[data-baseweb="select"] span,[data-baseweb="select"] svg,input::placeholder{color:#d5dce6!important;}
[data-testid="stMetric"],[data-testid="stDataFrame"],.lineup-card,[data-testid="stVerticalBlockBorderWrapper"]>div{background:#272f3c!important;border-color:var(--line)!important;box-shadow:0 10px 28px rgba(0,0,0,.14)!important;}
[data-testid="stMetricLabel"],[data-testid="stMetricValue"]{color:var(--ink)!important;}
.stButton>button{background:#303947!important;color:#f7f9fc!important;border:1px solid rgba(255,255,255,.13)!important;min-height:46px!important;box-shadow:none!important;}
.stButton>button:hover{background:#394657!important;border-color:rgba(90,156,255,.55)!important;}
.stButton>button[kind="primary"],[data-testid="stFormSubmitButton"] button,[data-testid="stDownloadButton"] button{background:linear-gradient(90deg,#2474ed,#3e8cff)!important;color:#fff!important;-webkit-text-fill-color:#fff!important;border:0!important;box-shadow:0 8px 24px rgba(47,128,255,.24)!important;}
.stButton>button *,[data-testid="stFormSubmitButton"] button *,[data-testid="stDownloadButton"] button *{color:inherit!important;-webkit-text-fill-color:inherit!important;}
/* Selectboxes must read as controls, especially the Lineup chooser. */
[data-testid="stSelectbox"]>div>div{min-height:50px!important;background:#303947!important;border:1px solid rgba(90,156,255,.42)!important;box-shadow:inset 0 0 0 1px rgba(255,255,255,.02)!important;}
[data-testid="stSelectbox"] svg{color:#80b5ff!important;}
.answer-kicker{font-size:.72rem;letter-spacing:.13em;font-weight:850;color:#78b1ff;margin:16px 0 7px;}
.lineup-card{padding:16px 18px!important;margin:10px 0!important;border-radius:16px!important;}
.lineup-rank{font-size:.82rem;font-weight:850;letter-spacing:.04em;color:#9fb0c6!important;}
.lineup-rank span{color:#8bbcff!important;}
.lineup-cpt{font-size:1.12rem;font-weight:850;margin-top:5px;color:#fff!important;}
.lineup-flex{font-size:.92rem;margin-top:5px;color:#dce4ef!important;}
.lineup-meta{font-size:.78rem;margin-top:8px;color:#aeb9c8!important;}
.results-hub-hero{margin:18px 0 12px;padding:18px 20px;border-radius:18px;background:linear-gradient(115deg,#102d54 0%,#174f8f 58%,#216bc0 100%);border:1px solid rgba(255,255,255,.16);box-shadow:0 14px 34px rgba(24,61,107,.20);}
.results-hub-kicker{font-size:.70rem;letter-spacing:.15em;font-weight:900;color:#a9d0ff!important;}
.results-hub-title{font-size:1.35rem;line-height:1.2;font-weight:900;color:#fff!important;margin-top:4px;}
.results-hub-sub{font-size:.90rem;line-height:1.45;color:#d9e9fb!important;margin-top:6px;max-width:920px;}
.results-view-title{font-size:1.18rem;font-weight:900;color:#172235!important;margin:14px 0 2px;}
.results-view-sub{font-size:.88rem;color:#52647a!important;margin:0 0 12px;}
.st-key-sd_results_hub [role="tablist"]{position:relative!important;top:auto!important;z-index:5!important;background:#f7f9fc!important;border:1px solid #aebccc!important;border-radius:16px!important;padding:7px!important;box-shadow:0 10px 28px rgba(37,52,76,.12)!important;gap:7px!important;}
.st-key-sd_results_hub [role="tab"]{flex:1 1 0!important;justify-content:center!important;min-height:54px!important;background:#e9eef5!important;border:1px solid transparent!important;border-radius:12px!important;}
.st-key-sd_results_hub [role="tab"] p,.st-key-sd_results_hub [role="tab"] span{font-size:.86rem!important;font-weight:900!important;color:#34465d!important;-webkit-text-fill-color:#34465d!important;}
.st-key-sd_results_hub [role="tab"][aria-selected="true"]{background:linear-gradient(100deg,#185fc8,#2d78df)!important;box-shadow:0 7px 18px rgba(30,96,195,.24)!important;}
.st-key-sd_results_hub [role="tab"][aria-selected="true"] p,.st-key-sd_results_hub [role="tab"][aria-selected="true"] span{color:#fff!important;-webkit-text-fill-color:#fff!important;}
@media(max-width:760px){.results-hub-hero{padding:15px 16px}.results-hub-title{font-size:1.18rem}.st-key-sd_results_hub [role="tab"]{min-width:150px!important}.st-key-sd_results_hub [role="tablist"]{overflow-x:auto!important;flex-wrap:nowrap!important;}}
[data-testid="stExpander"]{background:#242c37!important;border:1px solid var(--line)!important;border-radius:12px!important;}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *{color:#dce4ef!important;font-weight:700!important;}
hr{border-color:rgba(255,255,255,.10)!important;}
@media(max-width:900px){.block-container{padding-left:.9rem!important;padding-right:.9rem!important}.apple-hero{padding:20px!important}.apple-hero .apple-title{font-size:2rem!important}[data-testid="stTabs"] [data-baseweb="tab-list"]{overflow-x:auto;flex-wrap:nowrap}.stButton>button{min-height:48px!important}}
</style>
"""

# moved verbatim from streamlit_app.py line 5766
MAIN_DYNAMIC_SHELL_CSS = r"""
<style>
/* DFS LAB Dynamic Shell — final cascade */
:root{--page:#d8e0ea;--page2:#cdd7e4;--navy:#15243a;--navy2:#173f6d;--card:#f7f9fc;--card2:#edf2f7;--ink:#172235;--muted:#52647a;--line:#b8c5d5;--blue:#1473e6;--cyan:#4aa3ff;--good:#167c65;--warn:#8a5b00;}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:linear-gradient(145deg,var(--page) 0%,#e8edf3 46%,var(--page2) 100%)!important;color:var(--ink)!important;}
[data-testid="stHeader"]{background:#263243!important;border-bottom:1px solid rgba(255,255,255,.10)!important;}
.block-container{max-width:1420px!important;padding-top:1.4rem!important;}
/* Typography must work on the LIGHT workspace. */
[data-testid="stAppViewContainer"] h1,[data-testid="stAppViewContainer"] h2,[data-testid="stAppViewContainer"] h3,[data-testid="stAppViewContainer"] h4,[data-testid="stAppViewContainer"] h5,[data-testid="stAppViewContainer"] .card-title,[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li,[data-testid="stAppViewContainer"] label{color:var(--ink)!important;-webkit-text-fill-color:var(--ink)!important;}
[data-testid="stAppViewContainer"] .card-sub,[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],[data-testid="stAppViewContainer"] small{color:var(--muted)!important;-webkit-text-fill-color:var(--muted)!important;}
.apple-hero{position:relative;overflow:hidden;background:linear-gradient(118deg,#14243b 0%,#183b64 58%,#1262a7 100%)!important;border:1px solid rgba(255,255,255,.20)!important;border-radius:28px!important;padding:30px 38px!important;box-shadow:0 22px 55px rgba(27,48,78,.22)!important;}
.apple-hero:after{content:"";position:absolute;width:360px;height:360px;right:-100px;top:-180px;border-radius:50%;background:radial-gradient(circle,rgba(94,183,255,.32),rgba(94,183,255,0) 68%);pointer-events:none;}
.apple-hero .apple-title{color:#fff!important;-webkit-text-fill-color:#fff!important;font-size:2.65rem!important;letter-spacing:-.045em!important}.apple-hero .apple-sub{color:#d9e8f8!important;-webkit-text-fill-color:#d9e8f8!important;font-size:1.08rem!important}.apple-hero .apple-eyebrow{color:#8fc7ff!important;-webkit-text-fill-color:#8fc7ff!important;letter-spacing:.14em!important}.hero-actions{display:flex;align-items:center;gap:12px;margin-top:17px}.hero-hint{color:#c7d9ec;font-size:.83rem;font-weight:700}.pill{background:rgba(45,151,255,.18)!important;color:#d8ecff!important;-webkit-text-fill-color:#d8ecff!important;border:1px solid rgba(137,200,255,.35)!important;}
/* Light cards with real separation instead of a flat sheet. */
[data-testid="stMetric"],.lineup-card,[data-testid="stVerticalBlockBorderWrapper"]>div{background:linear-gradient(145deg,var(--card),#eef3f8)!important;border:1px solid #b8c5d5!important;box-shadow:0 12px 28px rgba(34,53,78,.10)!important;color:var(--ink)!important;}
.lineup-card{border-radius:20px!important;transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease!important}.lineup-card:hover{transform:translateY(-3px);border-color:#7eafe7!important;box-shadow:0 18px 36px rgba(34,72,120,.16)!important}.lineup-card .lineup-cpt,.lineup-card .lineup-flex,.lineup-card .lineup-meta{color:var(--ink)!important}.lineup-card .lineup-meta{color:var(--muted)!important}
/* Inputs are clearly interactive, not white-on-white. */
[data-baseweb="select"]>div,[data-baseweb="input"]>div,input,textarea,[data-testid="stFileUploaderDropzone"]{background:#f8fafc!important;border:1px solid #aebdce!important;color:var(--ink)!important;border-radius:14px!important;}
[data-baseweb="select"] span,[data-baseweb="select"] svg,input::placeholder,textarea::placeholder{color:#53657b!important;-webkit-text-fill-color:#53657b!important;}
[data-testid="stFileUploader"] label,[data-testid="stFileUploader"] small,[data-testid="stFileUploader"] span,[data-testid="stFileUploader"] p{color:#33465e!important;-webkit-text-fill-color:#33465e!important;}
[data-testid="stFileUploaderDropzone"] button{background:#e4ebf4!important;color:#20344e!important;-webkit-text-fill-color:#20344e!important;border:1px solid #b3c1d1!important;}
/* Alerts: dark readable type on tinted surfaces. */
[data-testid="stAlert"]{border-radius:16px!important;border:1px solid rgba(71,91,116,.20)!important;box-shadow:0 7px 18px rgba(38,56,80,.06)!important;}
[data-testid="stAlert"] *{color:#23354a!important;-webkit-text-fill-color:#23354a!important;}
/* Tabs become a floating command strip. */
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:7px;background:rgba(37,52,72,.92)!important;border:1px solid rgba(255,255,255,.12)!important;padding:7px!important;border-radius:16px!important;box-shadow:0 10px 24px rgba(34,49,69,.14)!important;overflow-x:auto!important;}
[data-testid="stTabs"] button[data-baseweb="tab"]{border-radius:11px!important;padding:.72rem 1.05rem!important;transition:all .16s ease!important}[data-testid="stTabs"] button[data-baseweb="tab"] p{color:#c8d4e2!important;-webkit-text-fill-color:#c8d4e2!important;font-weight:800!important}[data-testid="stTabs"] button[aria-selected="true"]{background:linear-gradient(135deg,#1876df,#3a8ef0)!important;box-shadow:0 5px 14px rgba(20,115,230,.30)!important}[data-testid="stTabs"] button[aria-selected="true"] p{color:white!important;-webkit-text-fill-color:white!important;}
/* Sidebar is a real control deck. */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#1d2a3b,#25354a)!important;border-right:1px solid rgba(255,255,255,.12)!important;box-shadow:14px 0 35px rgba(25,38,56,.18)!important;}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3,section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:#edf4fb!important;-webkit-text-fill-color:#edf4fb!important;}
section[data-testid="stSidebar"] [data-baseweb="select"]>div,section[data-testid="stSidebar"] input{background:#f4f7fb!important;color:#172235!important;-webkit-text-fill-color:#172235!important;border-color:#91a4ba!important;}
/* Make BOTH Streamlit sidebar controls impossible to miss on iPad. */
[data-testid="stSidebarCollapseButton"],[data-testid="stSidebarCollapsedControl"]{display:flex!important;visibility:visible!important;opacity:1!important;z-index:100000!important;}
[data-testid="stSidebarCollapseButton"] button,[data-testid="stSidebarCollapsedControl"] button{display:flex!important;visibility:visible!important;opacity:1!important;background:linear-gradient(135deg,#1473e6,#4a9cf5)!important;color:white!important;-webkit-text-fill-color:white!important;border:2px solid rgba(255,255,255,.75)!important;border-radius:999px!important;width:48px!important;height:48px!important;min-width:48px!important;min-height:48px!important;box-shadow:0 7px 20px rgba(20,70,130,.30)!important;}
[data-testid="stSidebarCollapseButton"] svg,[data-testid="stSidebarCollapsedControl"] svg{color:white!important;fill:white!important;width:22px!important;height:22px!important;}
/* Buttons */
.stButton>button{background:#e8eef6!important;color:#1c314b!important;-webkit-text-fill-color:#1c314b!important;border:1px solid #aebdce!important;box-shadow:0 4px 10px rgba(35,53,76,.06)!important;transition:transform .14s ease,box-shadow .14s ease!important}.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 18px rgba(35,70,115,.12)!important}.stButton>button:active{transform:scale(.98)!important}.stButton>button[kind="primary"],[data-testid="stFormSubmitButton"] button,[data-testid="stDownloadButton"] button{background:linear-gradient(100deg,#126fd8,#3b91ef)!important;color:white!important;-webkit-text-fill-color:white!important;border:0!important;}
[data-testid="stFormSubmitButton"] button *,[data-testid="stDownloadButton"] button *{color:white!important;-webkit-text-fill-color:white!important;}
/* Generate motion without turning the app into a toy. */
@keyframes labRise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}@keyframes labPulse{0%,100%{filter:brightness(1)}50%{filter:brightness(1.12)}}
.apple-hero{animation:labRise .38s ease both}.lineup-card{animation:labRise .28s ease both}[data-testid="stProgress"]>div>div{background:linear-gradient(90deg,#1473e6,#5b62ef,#16a6c9)!important;animation:labPulse 1.2s ease-in-out infinite!important;}
@media(max-width:900px){.block-container{padding-left:.8rem!important;padding-right:.8rem!important}.apple-hero{padding:23px 24px!important}.apple-hero .apple-title{font-size:2.15rem!important}.hero-hint{display:none}[data-testid="stSidebarCollapseButton"] button,[data-testid="stSidebarCollapsedControl"] button{width:52px!important;height:52px!important;min-width:52px!important;min-height:52px!important;}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
/* number-input steppers: visible in dark theme */
[data-testid="stNumberInputStepDown"],[data-testid="stNumberInputStepUp"]{
  background:var(--rcc-panel3)!important;border:1px solid var(--rcc-line2)!important;
  color:var(--rcc-text)!important;opacity:1!important;
}
[data-testid="stNumberInputStepDown"] svg,[data-testid="stNumberInputStepUp"] svg{
  fill:var(--rcc-text)!important;stroke:var(--rcc-text)!important;
}
[data-testid="stNumberInputStepDown"]:hover,[data-testid="stNumberInputStepUp"]:hover{
  background:var(--rcc-accent-dim)!important;
}
</style>
"""

# moved verbatim from streamlit_app.py line 5814
MAIN_UNIFORM_THEME_CSS = """<style>
/* DFS LAB UNIFORM PAGE THEME */
html, body,
.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background: #e6edf6 !important;
    color: #1f2937 !important;
}
[data-testid="stHeader"] {
    background: #e6edf6 !important;
}
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.block-container {
    background: transparent !important;
}

/* Keep primary content surfaces consistent at every app state. */
[data-testid="stExpander"],
[data-testid="stMetric"],
.section-card {
    background: #f4f7fb !important;
    border-color: #c8d3e2 !important;
}

/* Inputs stay softly tinted instead of flipping between white and dark. */
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stNumberInput"] > div > div,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #eef3f9 !important;
    color: #1f2937 !important;
    -webkit-text-fill-color: #1f2937 !important;
}

/* Normal body copy stays dark and readable on the unified background. */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p {
    color: #253247;
}

/* Preserve intentionally dark hero/brand areas. */
.hero, .apple-hero {
    color: #f8fafc !important;
}
.hero *, .apple-hero * {
    color: inherit;
}
</style>"""

# moved verbatim from streamlit_app.py line 5868
MAIN_EXPANDERS_CSS = """<style>
/* ---------- EXPANDERS / DARK BARS: never dark-on-dark ---------- */
html body [data-testid="stExpander"] summary,
html body [data-testid="stExpander"] summary:hover {
    background: linear-gradient(100deg,#12263a,#17324d) !important;
    border-color:#294a68 !important;
}
html body [data-testid="stExpander"] summary *,
html body [data-testid="stExpander"] summary p,
html body [data-testid="stExpander"] summary span,
html body [data-testid="stExpander"] summary svg {
    color:#f5f9ff !important;
    -webkit-text-fill-color:#f5f9ff !important;
    fill:#f5f9ff !important;
    opacity:1 !important;
    font-weight:800 !important;
}
html body [data-testid="stExpander"] > details > div {
    background:#f4f7fb !important;
    color:#243247 !important;
}
html body [data-testid="stExpander"] > details > div p,
html body [data-testid="stExpander"] > details > div label,
html body [data-testid="stExpander"] > details > div span:not([data-baseweb="tag"] span) {
    color:#243247 !important;
    -webkit-text-fill-color:#243247 !important;
}

/* ---------- TOP NAV: same navy + electric blue family ---------- */
html body [data-testid="stTabs"] [role="tablist"],
html body [data-testid="stTabs"] [data-baseweb="tab-list"] {
    background:#0d1c2d !important;
    border-color:#213d59 !important;
}
html body [data-testid="stTabs"] [role="tab"] p,
html body [data-testid="stTabs"] [role="tab"] span {
    color:#b9c9da !important;
    -webkit-text-fill-color:#b9c9da !important;
}
html body [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background:linear-gradient(100deg,#176fe8,#2d8cff) !important;
    border-color:#4aa0ff !important;
}
html body [data-testid="stTabs"] [role="tab"][aria-selected="true"] p,
html body [data-testid="stTabs"] [role="tab"][aria-selected="true"] span {
    color:#fff !important;
    -webkit-text-fill-color:#fff !important;
}

/* ---------- LINEUP CARDS: blue-only visual language ---------- */
html body .lineup-card.lineup-card-grid {
    background:linear-gradient(150deg,#f8fbff,#edf4fb) !important;
    border:1px solid #a9bfd6 !important;
    border-top:3px solid #2788f5 !important;
    color:#17283b !important;
    box-shadow:0 12px 28px rgba(33,69,108,.12) !important;
    min-height:365px !important;
    padding:18px 20px !important;
}
html body .lineup-card-grid:before{display:none !important}
html body .lineup-card-grid .lineup-rank {
    background:#e4f0ff !important;
    border:1px solid #87b9ef !important;
    color:#1767bd !important;
    -webkit-text-fill-color:#1767bd !important;
}
html body .lineup-card-grid .lineup-grade {
    background:#d9eaff !important;
    border:1px solid #7db4ef !important;
    color:#125ea9 !important;
    -webkit-text-fill-color:#125ea9 !important;
}
html body .lineup-card-grid .lineup-points {
    color:#10243a !important;
    -webkit-text-fill-color:#10243a !important;
    margin:0 0 7px !important;
}
html body .lineup-card-grid .lineup-points small {
    color:#627a92 !important;
    -webkit-text-fill-color:#627a92 !important;
}
html body .lineup-card-grid .lineup-cpt {
    color:#10243a !important;
    -webkit-text-fill-color:#10243a !important;
    font-weight:900 !important;
}
html body .lineup-card-grid .lineup-cpt span {
    background:#e6f1ff !important;
    border:1px solid #6ba9eb !important;
    color:#1768bd !important;
    -webkit-text-fill-color:#1768bd !important;
}
html body .lineup-card-grid .lineup-meta {
    color:#50677f !important;
    -webkit-text-fill-color:#50677f !important;
    border-color:#bdcddd !important;
}
html body .lineup-card-grid .lineup-world {
    color:#176fc9 !important;
    -webkit-text-fill-color:#176fc9 !important;
}
.captain-feature{display:flex;align-items:center;gap:14px;margin:12px 0 16px}
.player-avatar{overflow:hidden;flex:0 0 auto;border-radius:50%;background:linear-gradient(145deg,#d9eaff,#bcd9f7);border:2px solid #7fb7ee;display:flex;align-items:center;justify-content:center;color:#155f9f!important;-webkit-text-fill-color:#155f9f!important;font-weight:900}
.player-avatar img{width:100%;height:100%;object-fit:cover;object-position:center top;display:block}
.captain-avatar{width:76px;height:76px;box-shadow:0 7px 18px rgba(30,104,181,.16)}
.flex-avatar{width:36px;height:36px;border-width:1px}
.avatar-fallback{font-size:.72rem}
.captain-avatar.avatar-fallback{font-size:1rem}
.lineup-flex-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px 9px;margin:2px 0 10px}
.flex-person{display:flex;align-items:center;gap:7px;min-width:0;color:#24384d!important;-webkit-text-fill-color:#24384d!important;font-size:.76rem;font-weight:750}
.flex-person span{color:#24384d!important;-webkit-text-fill-color:#24384d!important;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* ---------- RESULT COMMAND CENTER ---------- */
html body .results-hub-hero {
    background:linear-gradient(115deg,#10273e,#154a7d) !important;
    border-color:#2c7fc8 !important;
}
html body .results-hub-hero * {
    color:#f4f9ff !important;
    -webkit-text-fill-color:#f4f9ff !important;
}
html body .world-board{background:#f5f8fc !important;border-color:#b8c9da !important}
html body .world-label,html body .world-count{color:#20364d !important;-webkit-text-fill-color:#20364d !important}
html body .world-count span{color:#647b92 !important;-webkit-text-fill-color:#647b92 !important}
html body .world-track{background:#dce7f2 !important}
html body .world-fill{background:linear-gradient(90deg,#176fe8,#46a0ff) !important}

/* ---------- iPad ---------- */
@media(max-width:900px){
    html body .lineup-card.lineup-card-grid{min-height:0!important}
    .captain-avatar{width:64px;height:64px}
    .lineup-flex-grid{grid-template-columns:1fr}
}
</style>"""

# moved verbatim from streamlit_app.py line 6006
MAIN_EXPANDER_HEADERS_CSS = """<style>
/* EXPANDER HEADERS — light surface + dark text, no more black-on-navy failure */
html body [data-testid="stAppViewContainer"] [data-testid="stExpander"] summary,
html body [data-testid="stAppViewContainer"] [data-testid="stExpander"] summary:hover,
html body [data-testid="stAppViewContainer"] details summary,
html body [data-testid="stAppViewContainer"] details summary:hover {
    background:#e8f1fb !important;
    background-image:none !important;
    border-bottom:1px solid #b6c9dc !important;
    color:#152a42 !important;
    -webkit-text-fill-color:#152a42 !important;
    opacity:1 !important;
}
html body [data-testid="stAppViewContainer"] [data-testid="stExpander"] summary *,
html body [data-testid="stAppViewContainer"] [data-testid="stExpander"] summary p,
html body [data-testid="stAppViewContainer"] [data-testid="stExpander"] summary span,
html body [data-testid="stAppViewContainer"] [data-testid="stExpander"] summary div,
html body [data-testid="stAppViewContainer"] details summary *,
html body [data-testid="stAppViewContainer"] details summary p,
html body [data-testid="stAppViewContainer"] details summary span,
html body [data-testid="stAppViewContainer"] details summary div {
    color:#152a42 !important;
    -webkit-text-fill-color:#152a42 !important;
    opacity:1 !important;
    font-weight:850 !important;
    text-shadow:none !important;
}
html body [data-testid="stAppViewContainer"] [data-testid="stExpander"] summary svg,
html body [data-testid="stAppViewContainer"] details summary svg {
    color:#176fc9 !important;
    fill:#176fc9 !important;
    opacity:1 !important;
}

/* The selected-lineup-count strip had the same dark-on-dark problem. */
html body .lineup-count-readout {
    background:#e8f1fb !important;
    border:1px solid #b6c9dc !important;
    color:#263b52 !important;
    box-shadow:none !important;
}
html body .lineup-count-readout b {
    color:#176fc9 !important;
    -webkit-text-fill-color:#176fc9 !important;
}
html body .lineup-count-readout span {
    color:#263b52 !important;
    -webkit-text-fill-color:#263b52 !important;
    opacity:1 !important;
}

/* Expander body stays light as one continuous card. */
html body [data-testid="stAppViewContainer"] [data-testid="stExpander"],
html body [data-testid="stAppViewContainer"] [data-testid="stExpander"] > details > div {
    background:#f5f8fc !important;
    border-color:#b6c9dc !important;
}
</style>"""

# moved verbatim from streamlit_app.py line 6067
MAIN_WHY_STRIP_CSS = """<style>
.lineup-why-strip{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.lineup-why-strip span{padding:4px 7px;border-radius:999px;background:#e8f1fb;border:1px solid #b9d2ec;color:#285f96!important;-webkit-text-fill-color:#285f96!important;font-size:.64rem;font-weight:900;letter-spacing:.035em}
.why-drawer{margin:16px 0 12px;padding:18px 20px;border:1px solid #a9bfd6;border-radius:18px;background:linear-gradient(145deg,#f8fbff,#eef5fc);box-shadow:0 12px 28px rgba(33,69,108,.10);color:#17283b}
.why-head{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;padding-bottom:13px;border-bottom:1px solid #c7d6e5}
.why-kicker{font-size:.68rem;letter-spacing:.13em;font-weight:950;color:#176fc9!important;-webkit-text-fill-color:#176fc9!important}
.why-title{margin-top:3px;font-size:1.18rem;font-weight:950;color:#152a42!important;-webkit-text-fill-color:#152a42!important}.why-title span{color:#60768c!important;-webkit-text-fill-color:#60768c!important}
.why-proj{font-size:1.35rem;font-weight:950;color:#152a42!important;-webkit-text-fill-color:#152a42!important;white-space:nowrap}.why-proj small{font-size:.68rem;color:#60768c!important;-webkit-text-fill-color:#60768c!important}
.why-chips{display:flex;flex-wrap:wrap;gap:7px;margin:14px 0}.why-chips span{padding:7px 9px;border-radius:9px;background:#e5f0fb;border:1px solid #b8cee4;color:#49657f!important;-webkit-text-fill-color:#49657f!important;font-size:.67rem;font-weight:800}.why-chips b{color:#176fc9!important;-webkit-text-fill-color:#176fc9!important;margin-right:3px}
.why-section{padding:3px 0 12px}.why-section b{color:#1a3550!important;-webkit-text-fill-color:#1a3550!important}.why-section p{margin:5px 0 0!important;color:#344e67!important;-webkit-text-fill-color:#344e67!important;line-height:1.5}
.why-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:3px 0 12px}.why-grid>div{padding:10px 11px;border-radius:10px;background:#edf4fb;border:1px solid #c5d6e6;min-width:0}.why-grid small{display:block;font-size:.59rem;letter-spacing:.07em;color:#6d8297!important;-webkit-text-fill-color:#6d8297!important;font-weight:850}.why-grid strong{display:block;margin-top:3px;color:#203b56!important;-webkit-text-fill-color:#203b56!important;font-size:.78rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.why-watch{display:flex;gap:9px;align-items:flex-start;padding:10px 12px;border-radius:10px;background:#e8f1fb;border-left:3px solid #4d8fcd}.why-watch b{color:#245a8b!important;-webkit-text-fill-color:#245a8b!important;white-space:nowrap}.why-watch span{color:#496176!important;-webkit-text-fill-color:#496176!important;font-size:.79rem;line-height:1.4}
@media(max-width:900px){.why-grid{grid-template-columns:1fr 1fr}.why-drawer{padding:15px 16px}.why-watch{display:block}.why-watch b{display:block;margin-bottom:4px}}
</style>"""

# DFS LAB Results Command Center — dark theme lock (final cascade).
# Dark gray surfaces (~#121212, elevated panels slightly lighter), off-white
# primary text, muted secondary text, desaturated accents, tabular numerals.
# Injected last in render_main so it wins over the older light layers, and it
# fixes the unreadable dark-bar text contrast (dark bars always get light text).
RCC_DARK_CSS = r"""
<style>
:root{
  --rcc-bg:#101418; --rcc-bg2:#0b0f13;
  --rcc-panel:#151b23; --rcc-panel2:#1b232e; --rcc-panel3:#222c39;
  --rcc-line:#2b3644; --rcc-line2:#354252;
  --rcc-text:#eef3f8; --rcc-muted:#94a5b8; --rcc-faint:#6b7c8f;
  --rcc-accent:#5b9dff; --rcc-accent2:#7fb3ff; --rcc-accent-dim:#274b73;
  --rcc-green:#43b581; --rcc-amber:#c99a4b; --rcc-red:#c97070;
}
/* ---------- page ---------- */
html,body,.stApp,[data-testid="stApp"],[data-testid="stAppViewContainer"],[data-testid="stMain"]{
  background:radial-gradient(circle at 70% -10%,#16202b 0%,#0e1319 42%,#0a0e12 100%)!important;
  color:var(--rcc-text)!important;
}
[data-testid="stHeader"]{background:rgba(11,15,19,.9)!important;border-bottom:1px solid var(--rcc-line)!important;}
[data-testid="stMain"]>div,[data-testid="stMainBlockContainer"],.block-container{background:transparent!important;}
.block-container{max-width:1500px!important;padding-top:1rem!important;}
html body [data-testid="stAppViewContainer"] h1,html body [data-testid="stAppViewContainer"] h2,
html body [data-testid="stAppViewContainer"] h3,html body [data-testid="stAppViewContainer"] h4,
html body [data-testid="stAppViewContainer"] h5,
html body [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
html body [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li,
html body [data-testid="stAppViewContainer"] label,
html body [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p{
  color:var(--rcc-text)!important;-webkit-text-fill-color:var(--rcc-text)!important;
}
html body [data-testid="stAppViewContainer"] .card-sub,
html body [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
html body [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p,
html body [data-testid="stAppViewContainer"] small,.stCaption{color:var(--rcc-muted)!important;-webkit-text-fill-color:var(--rcc-muted)!important;}
html body [data-testid="stAppViewContainer"] a{color:var(--rcc-accent2)!important;}
html body hr{border-color:var(--rcc-line)!important;}
/* tabular numerals for money / projections / percentages */
html body [data-testid="stMetricValue"],html body [data-testid="stDataFrame"],
html body .rcc-hero,html body .rcc-world-count,html body .rcc-roster td.num{
  font-variant-numeric:tabular-nums!important;
}
/* ---------- metrics / tiles ---------- */
html body [data-testid="stMetric"]{
  background:linear-gradient(150deg,var(--rcc-panel),var(--rcc-panel2))!important;
  border:1px solid var(--rcc-line)!important;border-radius:16px!important;
  box-shadow:0 10px 26px rgba(0,0,0,.28)!important;
}
html body [data-testid="stMetricLabel"] p{color:var(--rcc-muted)!important;-webkit-text-fill-color:var(--rcc-muted)!important;font-weight:700!important;}
html body [data-testid="stMetricValue"]{color:#fff!important;-webkit-text-fill-color:#fff!important;font-weight:850!important;}
/* ---------- dataframes ---------- */
html body [data-testid="stDataFrame"]{background:var(--rcc-panel)!important;border:1px solid var(--rcc-line)!important;border-radius:14px!important;box-shadow:0 10px 26px rgba(0,0,0,.25)!important;}
/* ---------- expanders: dark bars, always light text (contrast fix) ---------- */
html body [data-testid="stExpander"]{background:var(--rcc-panel)!important;border:1px solid var(--rcc-line)!important;border-radius:14px!important;box-shadow:0 10px 26px rgba(0,0,0,.25)!important;}
html body [data-testid="stExpander"] summary,
html body [data-testid="stExpander"] summary:hover,
html body [data-testid="stAppViewContainer"] details summary,
html body [data-testid="stAppViewContainer"] details summary:hover{
  background:linear-gradient(100deg,#182230,#1e2a38)!important;background-image:none!important;
  border-bottom:1px solid var(--rcc-line)!important;
  color:#eef4fa!important;-webkit-text-fill-color:#eef4fa!important;opacity:1!important;
}
html body [data-testid="stExpander"] summary *,
html body [data-testid="stExpander"] summary p,
html body [data-testid="stExpander"] summary span,
html body [data-testid="stExpander"] summary div,
html body [data-testid="stAppViewContainer"] details summary *{
  color:#eef4fa!important;-webkit-text-fill-color:#eef4fa!important;opacity:1!important;text-shadow:none!important;
}
html body [data-testid="stExpander"] summary svg,
html body [data-testid="stAppViewContainer"] details summary svg{color:var(--rcc-accent2)!important;fill:var(--rcc-accent2)!important;opacity:1!important;}
html body [data-testid="stExpander"]>details>div,
html body [data-testid="stAppViewContainer"] [data-testid="stExpander"],
html body [data-testid="stAppViewContainer"] [data-testid="stExpander"]>details>div{
  background:var(--rcc-panel)!important;border-color:var(--rcc-line)!important;color:var(--rcc-text)!important;
}
html body [data-testid="stExpander"]>details>div p,
html body [data-testid="stExpander"]>details>div label,
html body [data-testid="stExpander"]>details>div span{color:var(--rcc-text)!important;-webkit-text-fill-color:var(--rcc-text)!important;}
/* ---------- tabs: dark command strip ---------- */
html body [data-testid="stTabs"] [role="tablist"],html body [data-testid="stTabs"] [data-baseweb="tab-list"]{
  background:#131a22!important;border:1px solid var(--rcc-line)!important;border-radius:14px!important;
  padding:6px!important;box-shadow:0 10px 26px rgba(0,0,0,.30)!important;gap:6px!important;
}
html body [data-testid="stTabs"] [role="tab"],html body [data-testid="stTabs"] button[data-baseweb="tab"]{
  border-radius:10px!important;min-height:46px!important;opacity:1!important;visibility:visible!important;
}
html body [data-testid="stTabs"] [role="tab"] p,html body [data-testid="stTabs"] [role="tab"] span,
html body [data-testid="stTabs"] button[data-baseweb="tab"] p{
  color:var(--rcc-muted)!important;-webkit-text-fill-color:var(--rcc-muted)!important;font-weight:800!important;opacity:1!important;
}
html body [data-testid="stTabs"] [role="tab"][aria-selected="true"],
html body [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"]{
  background:linear-gradient(100deg,#2f6fd0,#4d94f2)!important;border-color:#6aa8ff!important;
  box-shadow:0 6px 18px rgba(70,140,240,.28)!important;
}
html body [data-testid="stTabs"] [role="tab"][aria-selected="true"] p,
html body [data-testid="stTabs"] [role="tab"][aria-selected="true"] span,
html body [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p{
  color:#fff!important;-webkit-text-fill-color:#fff!important;
}
/* ---------- segmented control: dark bar, bright selected pill ---------- */
html body [data-testid="stSegmentedControl"] [role="radiogroup"]{background:#131a22!important;border:1px solid var(--rcc-line)!important;border-radius:14px!important;padding:5px!important;}
html body [data-testid="stSegmentedControl"] button,
html body [data-testid="stSegmentedControl"] button *{
  color:var(--rcc-muted)!important;-webkit-text-fill-color:var(--rcc-muted)!important;opacity:1!important;font-weight:750!important;
}
html body [data-testid="stSegmentedControl"] button[aria-checked="true"],
html body [data-testid="stSegmentedControl"] button[aria-pressed="true"]{background:#8fc0ff!important;border-color:#8fc0ff!important;}
html body [data-testid="stSegmentedControl"] button[aria-checked="true"] *,
html body [data-testid="stSegmentedControl"] button[aria-pressed="true"] *{
  color:#0d1a2b!important;-webkit-text-fill-color:#0d1a2b!important;font-weight:850!important;opacity:1!important;
}
/* ---------- inputs ---------- */
html body [data-baseweb="select"]>div,html body [data-baseweb="input"]>div,
html body [data-testid="stTextInput"] input,html body [data-testid="stTextArea"] textarea,
html body [data-testid="stNumberInput"] input,html body [data-testid="stFileUploaderDropzone"]{
  background:#131a23!important;border:1px solid var(--rcc-line2)!important;color:var(--rcc-text)!important;
  -webkit-text-fill-color:var(--rcc-text)!important;border-radius:12px!important;
}
html body [data-baseweb="select"] span,html body [data-baseweb="select"] svg,
html body input::placeholder,html body textarea::placeholder{color:var(--rcc-faint)!important;-webkit-text-fill-color:var(--rcc-faint)!important;}
html body [data-baseweb="popover"],html body [role="listbox"]{background:#141c26!important;color:var(--rcc-text)!important;}
html body [role="option"]{color:var(--rcc-text)!important;}
html body [data-testid="stSelectbox"]>div>div{min-height:50px!important;}
/* ---------- buttons ---------- */
html body .stButton>button{background:var(--rcc-panel2)!important;color:var(--rcc-text)!important;
  -webkit-text-fill-color:var(--rcc-text)!important;border:1px solid var(--rcc-line2)!important;border-radius:12px!important;box-shadow:none!important;}
html body .stButton>button *,html body .stButton>button p{color:var(--rcc-text)!important;-webkit-text-fill-color:var(--rcc-text)!important;}
html body .stButton>button:hover{background:var(--rcc-panel3)!important;border-color:var(--rcc-accent-dim)!important;}
html body .stButton>button[kind="primary"],html body [data-testid="stFormSubmitButton"] button,
html body [data-testid="stDownloadButton"]>button{
  background:linear-gradient(100deg,#2f6fd0,#4d94f2)!important;color:#fff!important;
  -webkit-text-fill-color:#fff!important;border:1px solid #6aa8ff!important;
  box-shadow:0 8px 22px rgba(70,140,240,.25)!important;font-weight:800!important;
}
html body .stButton>button[kind="primary"] *,html body [data-testid="stFormSubmitButton"] button *,
html body [data-testid="stDownloadButton"]>button *{color:#fff!important;-webkit-text-fill-color:#fff!important;}
/* ---------- alerts ---------- */
html body [data-testid="stAlert"]{background:var(--rcc-panel2)!important;border:1px solid var(--rcc-line2)!important;border-radius:14px!important;}
html body [data-testid="stAlert"] *{color:var(--rcc-text)!important;-webkit-text-fill-color:var(--rcc-text)!important;}
/* ---------- bordered containers ---------- */
html body [data-testid="stVerticalBlockBorderWrapper"]>div{
  background:rgba(21,27,35,.85)!important;border:1px solid var(--rcc-line)!important;
  border-radius:16px!important;box-shadow:0 10px 26px rgba(0,0,0,.25)!important;
}
/* ---------- command strip / readouts ---------- */
html body .command-strip{background:linear-gradient(100deg,#141d27,#182635)!important;border:1px solid var(--rcc-line2)!important;color:var(--rcc-text)!important;box-shadow:0 10px 26px rgba(0,0,0,.25)!important;}
html body .command-live,html body .command-arrow{color:var(--rcc-accent2)!important;}
html body .command-copy{color:var(--rcc-muted)!important;}
html body .lineup-count-readout{background:#14202c!important;border:1px solid var(--rcc-line2)!important;color:var(--rcc-muted)!important;box-shadow:none!important;}
html body .lineup-count-readout b{color:var(--rcc-accent2)!important;-webkit-text-fill-color:var(--rcc-accent2)!important;}
/* legacy light cards forced dark */
html body .intel-card{background:linear-gradient(150deg,var(--rcc-panel),var(--rcc-panel2))!important;border:1px solid var(--rcc-line)!important;box-shadow:0 10px 26px rgba(0,0,0,.25)!important;}
html body .intel-kicker{color:var(--rcc-accent2)!important;}
html body .intel-big{color:var(--rcc-text)!important;}
html body .intel-copy{color:var(--rcc-muted)!important;}
html body .section-card{background:var(--rcc-panel)!important;border:1px solid var(--rcc-line)!important;}
html body .muted{color:var(--rcc-muted)!important;}
html body .badge-a{background:rgba(67,181,129,.16);color:#7fd6a8;}
html body .badge-b{background:rgba(91,157,255,.16);color:#9cc2ff;}
html body .badge-c{background:rgba(201,154,75,.16);color:#e3bd7d;}
html body .badge-x{background:rgba(201,112,112,.16);color:#e59a9a;}
html body .dfs-chat-user,html body .chat-user{background:#173a5e!important;border-color:#2b5a86!important;color:#eef4fa!important;}
html body .dfs-chat-ai,html body .chat-agent,html body .agent-status{background:var(--rcc-panel2)!important;border-color:var(--rcc-line2)!important;color:#c9d6e2!important;}
/* ============================================================
   Results Command Center components
   ============================================================ */
.rcc-kicker{font-size:.7rem;font-weight:850;letter-spacing:.14em;color:var(--rcc-accent2);}
.rcc-hero{background:linear-gradient(115deg,#12233a 0%,#14324f 55%,#123f6e 100%);border:1px solid #2c5b8f;
  border-radius:20px;padding:20px 24px;margin:6px 0 14px;box-shadow:0 16px 40px rgba(0,0,0,.35);}
.rcc-hero-row{display:flex;align-items:center;gap:18px;margin-top:10px;}
.rcc-grade{display:inline-flex;align-items:center;justify-content:center;min-width:74px;height:74px;border-radius:18px;
  font-size:2rem;font-weight:950;color:#fff;}
.rcc-grade-a{background:linear-gradient(140deg,#1f7a4d,#2fa36b);box-shadow:0 8px 22px rgba(47,163,107,.35);}
.rcc-grade-b{background:linear-gradient(140deg,#2b5f9e,#4d94f2);box-shadow:0 8px 22px rgba(77,148,242,.30);}
.rcc-grade-c{background:linear-gradient(140deg,#8a6420,#c99a4b);box-shadow:0 8px 22px rgba(201,154,75,.30);}
.rcc-hero-title{font-size:1.35rem;font-weight:900;color:#fff;letter-spacing:-.01em;}
.rcc-hero-sub{font-size:.9rem;color:#a9c0d6;margin-top:3px;}
.rcc-section-title{font-size:1.15rem;font-weight:900;color:var(--rcc-text);margin:20px 0 2px;}
.rcc-section-sub{font-size:.88rem;color:var(--rcc-muted);margin-bottom:12px;}
.rcc-story{font-size:1.02rem;font-weight:800;color:#cfe3f7;background:#16283c;border:1px solid #2c5378;
  border-radius:12px;padding:10px 14px;margin:2px 0 14px;}
.rcc-chips{display:flex;flex-wrap:wrap;gap:7px;margin:12px 0 4px;}
.rcc-chips span{padding:6px 10px;border-radius:999px;background:#182637;border:1px solid #2c4a68;
  color:#a9c4dc;font-size:.72rem;font-weight:800;letter-spacing:.02em;}
/* ---------- lineups-first: slim banner + cards + avatars ---------- */
.rcc-banner{display:flex;align-items:center;gap:10px;flex-wrap:wrap;background:linear-gradient(115deg,#0e2a1e,#14402a);
  border:1px solid #2f7355;border-radius:14px;padding:12px 16px;margin:2px 0 16px;
  color:#d9ecdf;font-size:.92rem;font-weight:600;box-shadow:0 10px 26px rgba(0,0,0,.25);}
.rcc-banner b{color:#fff;font-weight:850;}
.rcc-card{background:var(--rcc-panel);border:1px solid var(--rcc-line);border-radius:18px;
  padding:16px 18px;margin:0 0 14px;box-shadow:0 12px 30px rgba(0,0,0,.28);}
.rcc-card-head{display:flex;align-items:center;gap:12px;margin-bottom:10px;}
.rcc-rank{display:inline-flex;align-items:center;justify-content:center;min-width:46px;height:32px;
  padding:0 10px;border-radius:999px;background:#ffbd3e;color:#1a2028!important;
  font-size:.82rem;font-weight:950;}
.rcc-grade-sm{min-width:46px!important;height:32px!important;font-size:.88rem!important;border-radius:10px!important;}
.rcc-card-titlewrap{min-width:0;}
.rcc-card-title{font-size:1.06rem;font-weight:900;color:#fff;letter-spacing:-.01em;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.rcc-card-meta{font-size:.8rem;color:var(--rcc-muted);margin-top:2px;}
.rcc-card-foot{font-size:.82rem;color:var(--rcc-muted);line-height:1.5;margin-top:10px;
  border-top:1px solid var(--rcc-line);padding-top:10px;}
.rcc-avatar{width:34px;height:34px;border-width:1px;flex:0 0 auto;}
.rcc-avatar.avatar-fallback{font-size:.6rem;}
table.rcc-roster td.av,table.rcc-grid td.av{width:46px;padding:5px 2px 5px 9px;}
@media(max-width:900px){.rcc-card{padding:12px 13px;}.rcc-card-title{font-size:.95rem;white-space:normal;}}
table.rcc-roster{width:100%;border-collapse:collapse;margin:6px 0 4px;font-size:.88rem;}
table.rcc-roster th{text-align:left;font-size:.68rem;letter-spacing:.08em;color:var(--rcc-faint);
  padding:7px 9px;border-bottom:1px solid var(--rcc-line2);text-transform:uppercase;}
table.rcc-roster th.num,table.rcc-roster td.num{text-align:right;}
table.rcc-roster td{padding:8px 9px;border-bottom:1px solid var(--rcc-line);color:var(--rcc-text);}
table.rcc-roster td b{color:#fff;}
table.rcc-roster tr:last-child td{border-bottom:0;}
.rcc-world-board{background:var(--rcc-panel);border:1px solid var(--rcc-line);border-radius:16px;
  padding:16px 18px;box-shadow:0 10px 26px rgba(0,0,0,.25);}
.rcc-world-row{display:grid;grid-template-columns:minmax(170px,1.2fr) minmax(160px,3fr) 76px;gap:14px;
  align-items:center;padding:10px 0;border-bottom:1px solid var(--rcc-line);}
.rcc-world-row:last-child{border-bottom:0;}
.rcc-world-label{color:#dbe6f2;font-size:.9rem;font-weight:750;}
.rcc-world-label span{display:block;font-size:.74rem;font-weight:600;color:var(--rcc-muted);}
.rcc-world-track{height:12px;border-radius:999px;background:#0d141c;overflow:hidden;border:1px solid var(--rcc-line);}
.rcc-world-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#2f6fd0,#6aa8ff);}
.rcc-world-count{text-align:right;color:#fff;font-weight:850;font-size:.95rem;}
.rcc-chat-user{margin:10px 0 6px auto;padding:12px 14px;max-width:82%;background:#173a5e;border:1px solid #2b5a86;
  border-radius:16px 16px 4px 16px;color:#eef4fa;}
.rcc-chat-ai{margin:6px auto 14px 0;padding:14px 16px;max-width:94%;background:var(--rcc-panel2);
  border:1px solid var(--rcc-line2);border-radius:16px 16px 16px 4px;color:#cfdae6;line-height:1.55;}
@media(max-width:900px){
  .rcc-hero-row{gap:12px;}
  .rcc-grade{min-width:60px;height:60px;font-size:1.6rem;}
  .rcc-world-row{grid-template-columns:1.2fr 1.6fr 62px;gap:8px;}
}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
/* number-input steppers: visible in dark theme */
[data-testid="stNumberInputStepDown"],[data-testid="stNumberInputStepUp"]{
  background:var(--rcc-panel3)!important;border:1px solid var(--rcc-line2)!important;
  color:var(--rcc-text)!important;opacity:1!important;
}
[data-testid="stNumberInputStepDown"] svg,[data-testid="stNumberInputStepUp"] svg{
  fill:var(--rcc-text)!important;stroke:var(--rcc-text)!important;
}
[data-testid="stNumberInputStepDown"]:hover,[data-testid="stNumberInputStepUp"]:hover{
  background:var(--rcc-accent-dim)!important;
}
/* ============================================================
   Readability + sticky-tabs hardening.
   These selectors beat the legacy light-theme layers on specificity
   (via #root), so the dark command-center look holds regardless of
   injection order. Injected last in render_main.
   ============================================================ */
/* ---------- pinned primary tab bar: always visible ----------
   THE single pinned rule for the primary tab bar. All legacy sticky tab
   declarations (MAIN_TABS_CSS, SHOWDOWN_SHELL_CSS) were neutralized so they
   cannot fight this one. position:fixed (not sticky): the bar is permanently
   App-owned sticky nav (Sep 2026): the primary workspace nav is now a
   segmented control whose keyed element containers (.st-key-sd_nav /
   .st-key-classic_nav) stick to the top of the viewport. Unlike the old
   fixed-tablist CSS, this does not depend on Streamlit's internal tab DOM,
   and the selected section is session state so post-build buttons can jump
   the user to Players / Exposure. Solid background + no backdrop-filter
   (backdrop-filter is what broke sticky on iPad Safari). */
html body .st-key-sd_nav,
html body .st-key-classic_nav{
  position:sticky!important;top:0!important;z-index:900!important;
  background:#101418!important;
  backdrop-filter:none!important;-webkit-backdrop-filter:none!important;
  padding:.45rem .75rem!important;margin:0 -.75rem!important;
  border-bottom:1px solid rgba(122,162,255,.18)!important;
}
/* segmented-control pills: match the dark command-center tab look */
html body .st-key-sd_nav [data-testid="stButtonGroup"] button,
html body .st-key-classic_nav [data-testid="stButtonGroup"] button{
  color:#cdd8ea!important;background:transparent!important;
  border:1px solid transparent!important;border-radius:.5rem!important;
  font-weight:600!important;white-space:nowrap!important;
}
html body .st-key-sd_nav [data-testid="stButtonGroup"] button[data-selected="true"],
html body .st-key-classic_nav [data-testid="stButtonGroup"] button[data-selected="true"]{
  color:#fff!important;background:rgba(90,140,255,.22)!important;
  border-color:rgba(122,162,255,.45)!important;
}
/* the showdown results-hub tabs stay in normal flow (explicitly non-sticky) */
html body .st-key-sd_results_hub [role="tablist"],
html body .st-key-sd_results_hub [data-baseweb="tab-list"]{
  position:relative!important;top:auto!important;
}
/* ---------- expanders: definitive dark ---------- */
html body #root [data-testid="stAppViewContainer"] [data-testid="stExpander"],
html body #root [data-testid="stAppViewContainer"] [data-testid="stExpander"]>details,
html body #root [data-testid="stAppViewContainer"] [data-testid="stExpander"]>details>div{
  background:var(--rcc-panel)!important;border-color:var(--rcc-line)!important;
}
html body #root [data-testid="stAppViewContainer"] [data-testid="stExpander"] summary,
html body #root [data-testid="stAppViewContainer"] [data-testid="stExpander"] summary:hover,
html body #root [data-testid="stAppViewContainer"] details summary,
html body #root [data-testid="stAppViewContainer"] details summary:hover{
  background:#1a2430!important;background-image:none!important;
  border-bottom:1px solid var(--rcc-line)!important;border-radius:12px!important;
  color:#eef4fa!important;-webkit-text-fill-color:#eef4fa!important;
}
html body #root [data-testid="stAppViewContainer"] [data-testid="stExpander"] summary *,
html body #root [data-testid="stAppViewContainer"] details summary *{
  color:#eef4fa!important;-webkit-text-fill-color:#eef4fa!important;opacity:1!important;
}
html body #root [data-testid="stAppViewContainer"] [data-testid="stExpander"]>details>div{
  padding-top:14px!important;
}
/* ---------- selects: definitive dark ---------- */
html body #root [data-testid="stSelectbox"] [data-baseweb="select"]>div,
html body #root [data-testid="stAppViewContainer"] [data-baseweb="select"]>div{
  background:#131a23!important;border:1px solid var(--rcc-line2)!important;
  color:var(--rcc-text)!important;-webkit-text-fill-color:var(--rcc-text)!important;
}
html body #root [data-testid="stSelectbox"] [data-baseweb="select"] span,
html body #root [data-testid="stSelectbox"] [data-baseweb="select"] svg{
  color:var(--rcc-text)!important;-webkit-text-fill-color:var(--rcc-text)!important;
  fill:var(--rcc-text)!important;
}
/* ---------- download buttons: unmissable ---------- */
html body #root [data-testid="stDownloadButton"]>button{
  background:linear-gradient(100deg,#1d4ed8,#2563eb)!important;
  border:1px solid #60a5fa!important;min-height:52px!important;font-size:1rem!important;
}
html body #root [data-testid="stDownloadButton"]>button,
html body #root [data-testid="stDownloadButton"]>button *{
  color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;opacity:1!important;
}
/* ---------- results data grids: bright, larger headers ---------- */
.rcc-scroll-x{overflow-x:auto!important;-webkit-overflow-scrolling:touch!important;}
table.rcc-grid{width:100%;border-collapse:collapse;font-size:.88rem;min-width:640px;}
table.rcc-grid th{text-align:left;font-size:.78rem;font-weight:800;letter-spacing:.05em;
  color:#e6eef8;padding:10px;border-bottom:2px solid var(--rcc-line2);
  background:#1a2430;white-space:nowrap;}
table.rcc-grid th.num,table.rcc-grid td.num{text-align:right;font-variant-numeric:tabular-nums;}
table.rcc-grid td{padding:9px 10px;border-bottom:1px solid var(--rcc-line);
  color:#eef3f8;white-space:nowrap;}
table.rcc-grid td.wrap{white-space:normal;min-width:220px;}
table.rcc-grid td b{color:#fff;}
table.rcc-grid tr:last-child td{border-bottom:0;}
.rcc-cur-player{background:var(--rcc-panel2);border:1px solid var(--rcc-line2);border-radius:12px;
  padding:14px 16px;font-size:.95rem;color:var(--rcc-text);line-height:1.4;}
.rcc-cur-player b{color:#fff;font-size:1.05rem;}
.rcc-swap-head{margin:28px 0 12px;font-size:1rem;}
/* lineup-count pill: keep the trailing span bright on the dark pill */
html body #root .lineup-count-readout{color:var(--rcc-text)!important;-webkit-text-fill-color:var(--rcc-text)!important;}
html body #root .lineup-count-readout span{color:var(--rcc-text)!important;-webkit-text-fill-color:var(--rcc-text)!important;}
html body #root .lineup-count-readout b{color:var(--rcc-accent2)!important;-webkit-text-fill-color:var(--rcc-accent2)!important;}
</style>
"""
