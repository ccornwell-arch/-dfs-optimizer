import streamlit as st

st.set_page_config(page_title="Aytia — Story", page_icon="🏈", layout="wide", initial_sidebar_state="collapsed")

from dfs_lab.ui.boot import boot_page

# Story is a detour, not a nav tab. boot_page validates settings; the detour
# flag tells render_main to render only the story (no nav).
settings = boot_page("Showdown", "⚡ Build")
st.session_state["_story_detour"] = "showdown"

from dfs_lab.ui.main import render_main
render_main(settings)
