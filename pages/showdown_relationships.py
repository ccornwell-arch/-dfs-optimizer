import streamlit as st

st.set_page_config(page_title="Aytia \u2014 Relationships", page_icon="\U0001f3c8", layout="wide", initial_sidebar_state="collapsed")

from dfs_lab.ui.boot import boot_page

settings = boot_page("Showdown", "🔗 Relationships")

from dfs_lab.ui.main import render_main
render_main(settings)
