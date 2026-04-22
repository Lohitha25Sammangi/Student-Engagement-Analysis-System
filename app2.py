# app/app.py
import streamlit as st
import cv2
import imutils
from datetime import datetime
import time
import pandas as pd
import numpy as np
import os
import json
import sys
import tempfile 

# --- Path setup and imports ---
# Add utils folder to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from emotion_detector import EmotionDetector, EMOTIONS
from database_manager import DatabaseManager

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Student Engagement Analyzer", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .stApp { background-color: #0000; }
    .css-1d391kg { padding-top: 35px; } 
    .metric-box {
        background-color: #333333; /* CHANGED: Dark gray background */
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        margin-bottom: 10px;
        color: #FFFFFF; /* ADDED: Ensure default text color is white */
    }
    /* Hide the default Streamlit footer for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --- Initialize Session States and Database ---
@st.cache_resource
def get_db_manager():
    """Initialize DB Manager once and cache it."""
    return DatabaseManager()

# Initialize all necessary session states
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = get_db_manager()
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = 'Login'
    
# Persistent key for the video running state (True = running, False = stopped)
if 'video_run_toggle' not in st.session_state:
    st.session_state.video_run_toggle = False

# **CRITICAL FIX:** Persistent storage for live session metrics
if 'live_session_metrics' not in st.session_state:
    st.session_state.live_session_metrics = {
        'start_time': None, 
        'total_time': 0.0, 
        'attentive_time': 0.0, 
        'distracted_time': 0.0, 
        'emotion_counts': {e: 0 for e in EMOTIONS + ['No Face']}
    }
    
# Flags for displaying status messages after a rerun
if 'log_status_message' not in st.session_state:
    st.session_state.log_status_message = None
if 'log_status_type' not in st.session_state:
    st.session_state.log_status_type = 'info'


# --- Utility Functions for State Management ---

def start_analysis():
    """Sets the toggle to True and resets metrics."""
    # Reset metrics state immediately before starting a new session
    st.session_state.live_session_metrics = {
        'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_time': 0.0, 
        'attentive_time': 0.0, 
        'distracted_time': 0.0, 
        'emotion_counts': {e: 0 for e in EMOTIONS + ['No Face']}
    }
    st.session_state.log_status_message = None # Clear previous message
    st.session_state.video_run_toggle = True

# **CRITICAL FIX**
def stop_analysis():
    """Performs final logging using persisted metrics and then sets the toggle to False."""
    
    # 1. Retrieve the metrics recorded in the previous session context
    final_metrics = st.session_state.live_session_metrics
    total_time = final_metrics['total_time']
    
    # 2. Perform Logging Check and Action
    if total_time > 5.0 and st.session_state.user_id is not None: 
        try:
            st.session_state.db_manager.log_engagement_data(
                user_id=st.session_state.user_id,
                start_time=final_metrics['start_time'],
                end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                total_duration=total_time,
                attentive_time=final_metrics['attentive_time'],
                distracted_time=final_metrics['distracted_time'],
                emotion_counts=final_metrics['emotion_counts']
            )
            # Set success message to be displayed on the *next* run
            st.session_state.log_status_message = f"Analysis stopped. Session data of {total_time:.1f} seconds logged successfully to 'My History'!"
            st.session_state.log_status_type = 'success'
            print("DEBUG [App]: Data logged successfully during stop_analysis callback.")
        except Exception as e:
            st.session_state.log_status_message = f"Error logging data to database. Error: {e}"
            st.session_state.log_status_type = 'error'
            print(f"ERROR [App]: {e}")
            
    else:
        reason = []
        if total_time <= 5.0:
            reason.append(f"Session too short ({total_time:.1f}s, requires > 5.0s)")
        if st.session_state.user_id is None:
            reason.append("User not logged in (User ID is None)")
        
        st.session_state.log_status_message = f"Analysis stopped. Data NOT logged. Reason(s): {', '.join(reason)}."
        st.session_state.log_status_type = 'info'

    # 3. Reset toggle and session data
    st.session_state.video_run_toggle = False
    st.session_state.live_session_metrics['start_time'] = None 
    # The button click automatically triggers a rerun, so st.rerun() is not explicitly needed here.

def handle_login(username, password, admin_mode=False):
    user = st.session_state.db_manager.get_user(username, password, is_admin_check=admin_mode)
    if user:
        # Reset video toggle state and metrics on login
        st.session_state.video_run_toggle = False 
        st.session_state.live_session_metrics['start_time'] = None # Forces a reset on the first run
        st.session_state.logged_in = True
        st.session_state.user_id = user[0]
        st.session_state.username = user[1]
        st.session_state.is_admin = (user[2] == 1)
        st.session_state.app_mode = 'Main App' if not st.session_state.is_admin else 'Admin Dashboard'
        st.success(f"Welcome, {username}!")
        st.rerun()
    else:
        st.error("Invalid credentials or access level.")

def handle_signup(username, password):
    if st.session_state.db_manager.create_user(username, password):
        st.success("Account created successfully! Please log in.")
    else:
        st.error("Username already exists. Please choose another one.")
        
def logout():
    # Ensure video stream is stopped before logging out
    st.session_state.video_run_toggle = False
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.is_admin = False
    st.session_state.app_mode = 'Login'
    st.rerun() 

# --- Detector Initialization ---
@st.cache_resource
def get_emotion_detector():
    """Initializes the EmotionDetector once."""
    try:
        return EmotionDetector()
    except Exception as e:
        st.error(f"Failed to load the model or cascades. Error: {e}")
        return None

# --- Metrics Display Function ---
def update_metric_placeholders(total, attentive, distracted, total_time_ph, attentive_ph, distracted_ph):
    """Dynamically updates the Streamlit metric boxes."""
    
    total_time_ph.markdown(f"<div class='metric-box'>Total Time<br><span style='font-size: 24px; color: #FFFFFF;'>{total:.1f} s</span></div>", unsafe_allow_html=True)
    
    attentive_perc = (attentive / total * 100) if total > 0 else 0
    distracted_perc = (distracted / total * 100) if total > 0 else 0
    
    attentive_ph.markdown(f"<div class='metric-box'>Attentive<br><span style='font-size: 24px; color: #00FF00;'>{attentive_perc:.1f} %</span><br><span style='color: #BBBBBB;'>{attentive:.1f} s</span></div>", unsafe_allow_html=True)
    distracted_ph.markdown(f"<div class='metric-box'>Distracted<br><span style='font-size: 24px; color: #FF0000;'>{distracted_perc:.1f} %</span><br><span style='color: #BBBBBB;'>{distracted:.1f} s</span></div>", unsafe_allow_html=True)

# --- View Definitions ---

def login_view():
    """The main login/signup page."""
    st.title("🎓 Student Engagement System")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("👤 User Login")
        with st.form("user_login_form"):
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            if st.form_submit_button("Student Login"):
                handle_login(login_user, login_pass, admin_mode=False)

    with col2:
        st.header("🔑 Admin Login")
        st.warning("Default Admin: admin/admin")
        with st.form("admin_login_form"):
            admin_user = st.text_input("Admin Username", key="admin_user")
            admin_pass = st.text_input("Admin Password", type="password", key="admin_pass")
            if st.form_submit_button("Admin Login"):
                handle_login(admin_user, admin_pass, admin_mode=True)
                
    with col3:
        st.header("✍️ Register")
        with st.form("signup_form"):
            signup_user = st.text_input("New Username", key="signup_user")
            signup_pass = st.text_input("New Password", type="password", key="signup_pass")
            if st.form_submit_button("Create Account"):
                handle_signup(signup_user, signup_pass)


def user_app_view():
    st.sidebar.title(f"Hello, {st.session_state.username}")
    st.sidebar.button("Logout", on_click=logout, type="secondary")
    
    selected_page = st.sidebar.radio("Navigation", ['Live Call Analysis', 'Video File Analysis', 'My History']) 
    
    if selected_page == 'Live Call Analysis':
        live_analysis_view()
    elif selected_page == 'Video File Analysis':
        file_analysis_view() 
    elif selected_page == 'My History':
        user_history_view()


def live_analysis_view():
    st.header("📞 Live Call Analysis (Webcam)")
    st.warning("This mode captures real-time engagement data from your webcam and logs it to 'My History'.")

    detector = get_emotion_detector()
    if not detector: return 

    col_config, col_metrics = st.columns([1, 2])
    
    with col_config:
        st.subheader("Control")
        st.info("Source: Webcam (0) is fixed for live analysis.")
        
        # --- START/STOP BUTTONS ---
        if not st.session_state.video_run_toggle:
            st.button("Start Live Analysis", key='start_button', on_click=start_analysis, help="Click to start the live webcam feed analysis.", 
                      use_container_width=True, type="primary") 
        else:
            st.button("Stop Analysis", key='stop_button', on_click=stop_analysis, help="Click to stop the live webcam feed analysis and log data.",
                      use_container_width=True, type="secondary")
        # --- END FIXED BUTTONS ---
        
    with col_metrics:
        st.subheader("Live Metrics")
        col_att, col_dist, col_total = st.columns(3)
        attentive_placeholder = col_att.empty()
        distracted_placeholder = col_dist.empty()
        total_time_placeholder = col_total.empty()

    # Placeholders for video output
    st.markdown("---")
    st.subheader("Visual Feedback")
    
    status_msg_placeholder = st.empty() 
    
    # Check for status message from previous session (logged in stop_analysis)
    if st.session_state.log_status_message:
        if st.session_state.log_status_type == 'success':
            status_msg_placeholder.success(st.session_state.log_status_message)
        elif st.session_state.log_status_type == 'error':
            status_msg_placeholder.error(st.session_state.log_status_message)
        else:
            status_msg_placeholder.info(st.session_state.log_status_message)
        st.session_state.log_status_message = None # Clear message after display

    col_video, col_emotion_bars = st.columns([2, 1])
    frame_placeholder = col_video.empty()
    emotion_placeholder = col_emotion_bars.empty()
    
    # Initial/Current Metric Display
    current_metrics = st.session_state.live_session_metrics
    update_metric_placeholders(
        current_metrics['total_time'], 
        current_metrics['attentive_time'], 
        current_metrics['distracted_time'],
        total_time_placeholder, attentive_placeholder, distracted_placeholder
    ) 

    if st.session_state.video_run_toggle:
        
        status_msg_placeholder.info("Attempting to initialize webcam...")
        
        video_source = 0 # Fixed to webcam
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            status_msg_placeholder.error("Could not open Webcam (Source 0). Please check camera permissions and availability. Stopping analysis.")
            st.session_state.video_run_toggle = False 
            st.rerun()
            return
            
        status_msg_placeholder.success("Webcam initialized successfully. Starting stream. Click 'Stop Analysis' to finish.")
        
        # Use a local reference to the session metrics for faster access within the loop
        metrics = st.session_state.live_session_metrics
        last_frame_time = time.time()
        
        # --- Main Video Loop ---
        while st.session_state.video_run_toggle:
            
            ret, frame = cap.read()
            
            if not ret:
                # If stream fails mid-run, stop analysis and rely on stop_analysis to log metrics
                st.session_state.video_run_toggle = False 
                break
                
            current_frame_time = time.time()
            time_elapsed = current_frame_time - last_frame_time
            last_frame_time = current_frame_time
            
            # 1. Process Frame
            frame = imutils.resize(frame, width=600)
            annotated_frame, emotion_canvas, attentive_status, emotion_label = detector.process_frame(frame)
            
            # 2. Update Persistent Session Metrics (CRITICAL FIX)
            if emotion_label != 'No Face':
                if attentive_status:
                    metrics['attentive_time'] += time_elapsed
                else: 
                    metrics['distracted_time'] += time_elapsed 
            
            # Update emotion counts
            metrics['emotion_counts'][emotion_label if emotion_label in metrics['emotion_counts'] else 'No Face'] += 1
            metrics['total_time'] = metrics['attentive_time'] + metrics['distracted_time']
            
            # 3. Display Frames
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            emotion_rgb = cv2.cvtColor(emotion_canvas, cv2.COLOR_BGR2RGB)
            
            frame_placeholder.image(frame_rgb, channels="RGB")
            emotion_placeholder.image(emotion_rgb, channels="RGB")
            
            # 4. Update Metrics Display
            update_metric_placeholders(
                metrics['total_time'], 
                metrics['attentive_time'], 
                metrics['distracted_time'],
                total_time_placeholder, attentive_placeholder, distracted_placeholder
            )
            
            time.sleep(0.01) # Small delay for stability

        # --- Cleanup (Executed only if loop breaks naturally/unexpectedly) ---
        cap.release()
        
        # If the loop exited, we need to manually trigger the logging 
        # that stop_analysis would have done, then force a rerun.
        if not st.session_state.video_run_toggle:
            # Call stop_analysis directly to perform the logging and state cleanup
            stop_analysis() 
            st.rerun()
            
    else:
        # Initial status display when app starts or after stopping
        if not st.session_state.log_status_message:
            status_msg_placeholder.info("Click 'Start Live Analysis' to begin capturing data from your webcam.")


def file_analysis_view():
    st.header("🎞️ Video File Analysis (Test Mode)")
    st.info("Upload a video file to analyze its engagement. Data from this mode IS NOT logged to the database.")

    detector = get_emotion_detector()
    if not detector: return 
    
    uploaded_file = st.file_uploader("Upload a video file (MP4, MOV, AVI)", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close() 
        video_path = tfile.name
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Error opening video file. Please ensure the file is a valid video format.")
            os.unlink(video_path) 
            return

        # Setup Placeholders and Metrics
        st.markdown("---")
        st.subheader("Visual Feedback")
        col_video, col_emotion_bars = st.columns([2, 1])
        frame_placeholder = col_video.empty()
        emotion_placeholder = col_emotion_bars.empty()
        
        status_placeholder = st.empty()
        
        # Metrics setup
        attentive_time = 0.0
        distracted_time = 0.0
        last_frame_time = time.time()
        
        status_placeholder.text("Processing... Please wait for the video to finish.")
        
        # Loop through the video file
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_time = time.time()
            time_elapsed = current_frame_time - last_frame_time
            last_frame_time = current_frame_time

            # Process Frame
            frame = imutils.resize(frame, width=600)
            annotated_frame, emotion_canvas, attentive_status, emotion_label = detector.process_frame(frame)
            
            # Update Time Metrics
            if emotion_label != 'No Face':
                if attentive_status:
                    attentive_time += time_elapsed
                else: 
                    distracted_time += time_elapsed 
            
            # Display Frames
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            emotion_rgb = cv2.cvtColor(emotion_canvas, cv2.COLOR_BGR2RGB)
            
            frame_placeholder.image(frame_rgb, channels="RGB")
            emotion_placeholder.image(emotion_rgb, channels="RGB")
            
            time.sleep(0.01) 

        # Cleanup
        cap.release()
        os.unlink(video_path) 
        
        total_time = attentive_time + distracted_time
        
        status_placeholder.success(f"File Analysis Complete! Total Analyzed Duration: {total_time:.1f} seconds.")
        
        # Display Final Summary
        col_file_att, col_file_dist = st.columns(2)
        
        attentive_perc = (attentive_time / total_time * 100) if total_time > 0 else 0
        distracted_perc = (distracted_time / total_time * 100) if total_time > 0 else 0

        col_file_att.markdown(f"<div class='metric-box'>**Total Attentive Time**<br><span style='font-size: 24px; color: green;'>{attentive_perc:.1f}%</span><br>{attentive_time:.1f} s</div>", unsafe_allow_html=True)
        col_file_dist.markdown(f"<div class='metric-box'>**Total Distracted Time**<br><span style='font-size: 24px; color: red;'>{distracted_perc:.1f}%</span><br>{distracted_time:.1f} s</div>", unsafe_allow_html=True)

    else:
        st.warning("Please upload a video file to start analysis.")


def user_history_view():
    st.header("📖 My Engagement History")
    st.markdown(f"Below is a record of your past analysis sessions, {st.session_state.username}.")
    
    logs_df = st.session_state.db_manager.get_user_logs(st.session_state.user_id)
    
    if logs_df.empty:
        st.info("No engagement sessions logged yet. Start a Live Call Analysis to record data.")
        return

    st.subheader("Session Log Table")
    st.dataframe(logs_df)
    
    st.subheader("Overall Attentive vs. Distracted Time")
    
    total_attentive = logs_df['Attentive Time (s)'].sum()
    total_distracted = logs_df['Distracted Time (s)'].sum()
    
    summary_data = {
        'Status': ['Attentive', 'Distracted'],
        'Total Time (s)': [total_attentive, total_distracted]
    }
    summary_df = pd.DataFrame(summary_data)
    
    st.bar_chart(summary_df.set_index('Status'))

def admin_dashboard_view():
    st.header("👑 Admin Dashboard: All Engagement Data")
    st.sidebar.title(f"Admin: {st.session_state.username}")
    st.sidebar.button("Logout", on_click=logout, type="secondary")

    st.subheader("Global Student Engagement Logs")
    
    all_logs_df = st.session_state.db_manager.get_all_logs()
    
    if all_logs_df.empty:
        st.info("No data has been logged by any user yet.")
        return

    st.dataframe(all_logs_df, use_container_width=True)
    
    st.subheader("Aggregated Student Performance Overview")
    
    chart_data = all_logs_df.groupby('User')[['Attentive Time (s)', 'Distracted Time (s)']].sum()
    
    st.bar_chart(chart_data)
    
    st.subheader("Global Emotion Distribution")
    
    all_emotions = {e: 0 for e in EMOTIONS + ['No Face']}
    
    for emotion_dict in all_logs_df['Emotion Data']:
        for emotion, count in emotion_dict.items():
            if emotion in all_emotions:
                all_emotions[emotion] += count
    
    if 'No Face' in all_emotions:
        del all_emotions['No Face'] 
            
    if sum(all_emotions.values()) > 0:
        emotion_df = pd.DataFrame(all_emotions.items(), columns=['Emotion', 'Count'])
        st.bar_chart(emotion_df.set_index('Emotion'))
    else:
        st.info("No predicted emotion data to display.")


# --- Main Application Flow (Calls defined views) ---

if st.session_state.app_mode == 'Login':
    login_view()
elif st.session_state.app_mode == 'Main App':
    user_app_view()
elif st.session_state.app_mode == 'Admin Dashboard':
    admin_dashboard_view()