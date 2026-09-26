import streamlit as st

st.set_page_config(page_title="DFS LAB", page_icon="🏈", layout="wide", initial_sidebar_state="collapsed")

from dfs_lab.ui import render_app

render_app()
