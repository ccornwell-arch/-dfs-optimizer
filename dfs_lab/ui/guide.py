"""Shared Guide content — used by the pre-upload setup screen and the in-app Guide tab."""

import streamlit as st


def render_guide(mode="showdown"):
    """Render the full user guide. mode is 'showdown' or 'classic'."""
    is_sd = (mode == "showdown")
    st.markdown('<div class="card-title">Guide</div><div class="card-sub">The short version of how everything works. Walkthrough videos coming soon.</div>', unsafe_allow_html=True)
    with st.expander("🎬 Walkthrough videos", expanded=False):
        st.info("Video walkthroughs are coming — first build, player rules, reading your results.")
    with st.expander("Getting your DraftKings file", expanded=False):
        if is_sd:
            st.markdown("DK app → Lobby → tap your Showdown contest → download the salaries CSV (single-game Showdown file, not the main slate). That's the only required file. SaberSim is optional — without it, Aytia builds its own projections and estimated ownership.")
        else:
            st.markdown("DK app → Lobby → tap your contest → download the salaries CSV. That's the only required file. SaberSim is optional — without it, Aytia builds its own projections and estimated ownership.")
    with st.expander("The tabs", expanded=False):
        if is_sd:
            st.markdown("**⚡ Build** — lineup count, constructions (3-3, 4-2, 5-1), captain rules. **👤 Players** — Out / Lock / CPT / exposures (tap Apply player changes). **🔗 Relationships** — stack rules. **🧠 Game Intel** — slate notes. **⚙ Rules** — your game story: script, score, scenario influence. **📋 Lineups** — results, grades, exports. **📊 Exposure** — where the portfolio landed.")
        else:
            st.markdown("**🧠 Slate Intel** — matchup notes and leverage plays. **⚡ Build** — lineup count, stack and roster rules. **👤 Players** — Out / Lock / exposures (tap Apply player changes). **⚙ Rules** — your game story. **📋 Lineups** — results, grades, exports. **📊 Exposure** — where the portfolio landed.")
    with st.expander("Player rules", expanded=False):
        if is_sd:
            st.markdown("**Out** = never in a lineup. **Lock** = in every lineup. **CPT** = captain every lineup. **CPT?** = may captain (unchecked = never). **Min/Max %** = exposure targets. Always tap **Apply player changes** — unsaved edits don't count.")
        else:
            st.markdown("**Out** = never in a lineup. **Lock** = in every lineup. **Min/Max %** = exposure targets. Always tap **Apply player changes** — unsaved edits don't count.")
    with st.expander("Reading your results", expanded=False):
        if is_sd:
            st.markdown("**Grade** = overall quality. **Story** = the game script. **Game World** = which outcome bucket (used for diversification). **Dup Risk** = how chalky the lineup is.")
        else:
            st.markdown("**Grade** = overall quality. **Story** = the game script. **Dup Risk** = how chalky the lineup is. **Ceiling P90 / Break Slate %** come from 10,000 simulated worlds.")
    with st.expander("Exporting", expanded=False):
        if is_sd:
            st.markdown("**DK-format CSV** uploads straight to DraftKings (contest → Upload lineups). **My entries CSV** exports just your top N by grade. On iPhone the download opens Files — use the app switcher to come back.")
        else:
            st.markdown("**DK-format CSV** uploads straight to DraftKings (contest → Upload lineups). On iPhone the download opens Files — use the app switcher to come back.")
    with st.expander("Troubleshooting", expanded=False):
        if is_sd:
            st.markdown("**\"No captain-eligible players\"?** Check a CPT? box, Apply, rebuild. **No lineups?** A rule combo is impossible — loosen a lock or exclusion. **Out player in a lineup?** You didn't tap Apply. **Blank app?** Close the tab and reopen the URL fresh.")
        else:
            st.markdown("**No lineups?** A rule combo is impossible — loosen a lock or exclusion. **Out player in a lineup?** You didn't tap Apply. **Blank app?** Close the tab and reopen the URL fresh.")
