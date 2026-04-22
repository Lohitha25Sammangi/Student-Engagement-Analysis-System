# app/app.py
import streamlit as st

# Add app/views to path via package import; views are under app/views
from views import login_view, user_app_view, admin_dashboard_view

st.set_page_config(
    page_title="Student Engagement Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state basics
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = 'Login'

# Preserve a persistent DB manager instance if desired (some views also create one)
if 'db_manager' not in st.session_state:
    from utils.database_manager import DatabaseManager
    st.session_state.db_manager = DatabaseManager()

# Simple app router
if st.session_state.app_mode == 'Login':
    login_view.render()
elif st.session_state.app_mode == 'Main App':
    user_app_view.render()
elif st.session_state.app_mode == 'Admin Dashboard':
    admin_dashboard_view.render()
