# app/utils/database_manager.py
import sqlite3
import pandas as pd
import json
import os
import sys

# Set encoding for better compatibility in some environments
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

class DatabaseManager:
    """Manages the SQLite database for user accounts and engagement logs."""

    def __init__(self, db_name='engagement_db.sqlite'):
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        self._initialize_db()

    def _initialize_db(self):
        """Connects to the database and ensures tables/columns exist."""
        try:
            # IMPORTANT: check_same_thread=False for Streamlit
            self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
            self.cursor = self.conn.cursor()

            # Create Users Table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    is_admin INTEGER DEFAULT 0
                )
            """)

            # Create Logs Table (original columns); we'll ensure new columns exist below
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    total_duration REAL,
                    attentive_time REAL,
                    distracted_time REAL,
                    emotion_data TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)

            # Migration: add new columns if they don't exist (engagement_percent, distraction_percent, emotion_percentages)
            self._ensure_column('logs', 'engagement_percent', 'REAL', default='NULL')
            self._ensure_column('logs', 'distraction_percent', 'REAL', default='NULL')
            self._ensure_column('logs', 'emotion_percentages', 'TEXT', default='NULL')  # JSON string of percentages

            # Ensure default admin exists
            self._create_default_admin()
            self.conn.commit()
            print("DEBUG [DB]: Database initialized and admin user checked.")
        except sqlite3.Error as e:
            print(f"ERROR [DB]: SQLite initialization error: {e}")

    def _ensure_column(self, table, column, col_type, default='NULL'):
        """Add a column to a table if it does not exist (SQLite)."""
        try:
            self.cursor.execute(f"PRAGMA table_info({table})")
            cols = [r[1] for r in self.cursor.fetchall()]
            if column not in cols:
                print(f"DEBUG [DB]: Adding column {column} to {table}")
                self.cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type} DEFAULT {default}")
                self.conn.commit()
        except sqlite3.Error as e:
            print(f"ERROR [DB]: Failed to ensure column {column} on {table}: {e}")

    def _create_default_admin(self):
        """Creates a default admin user if one doesn't exist."""
        try:
            self.cursor.execute("SELECT id FROM users WHERE username='admin' AND is_admin=1")
            if not self.cursor.fetchone():
                try:
                    self.cursor.execute(
                        "INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                        ('admin', 'admin', 1)
                    )
                    self.conn.commit()
                    print("DEBUG [DB]: Default admin user created.")
                except sqlite3.Error as e:
                    print(f"WARNING [DB]: Failed to create admin, likely already exists. Error: {e}")
        except sqlite3.Error as e:
            print(f"ERROR [DB]: Error checking/creating default admin: {e}")

    # --- Authentication Methods ---

    def create_user(self, username, password):
        """Creates a new student user."""
        try:
            self.cursor.execute(
                "INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                (username, password, 0)
            )
            self.conn.commit()
            print(f"DEBUG [DB]: User '{username}' created successfully.")
            return True
        except sqlite3.IntegrityError:
            print(f"ERROR [DB]: Username '{username}' already exists.")
            return False
        except sqlite3.Error as e:
            print(f"ERROR [DB]: Error creating user: {e}")
            return False

    def get_user(self, username, password, is_admin_check=False):
        """Authenticates a user and returns their data (id, username, is_admin)."""
        try:
            query = "SELECT id, username, is_admin FROM users WHERE username=? AND password=? AND is_admin=?"
            admin_flag = 1 if is_admin_check else 0
            self.cursor.execute(query, (username, password, admin_flag))
            user = self.cursor.fetchone()
            return user
        except sqlite3.Error as e:
            print(f"ERROR [DB]: Error fetching user: {e}")
            return None

    # --- Logging Methods ---

    def log_engagement_data(self, user_id, start_time, end_time, total_duration,
                            attentive_time, distracted_time, emotion_counts=None, emotion_percentages=None):
        """
        Logs a single session's engagement metrics.

        Args:
            emotion_counts (dict): Optional - legacy counts dictionary.
            emotion_percentages (dict): Preferred - dict mapping emotion->percentage (sums to ~100).
        """
        # Prepare the JSON fields
        try:
            emotion_data_json = json.dumps(emotion_counts) if emotion_counts is not None else json.dumps({})
        except TypeError as e:
            print(f"FATAL ERROR [DB]: Failed to serialize emotion counts to JSON. Error: {e}")
            emotion_data_json = json.dumps({})

        try:
            emotion_percentages_json = json.dumps(emotion_percentages) if emotion_percentages is not None else None
        except TypeError as e:
            print(f"ERROR [DB]: Failed to serialize emotion_percentages to JSON. Error: {e}")
            emotion_percentages_json = None

        # Calc engagement & distraction percents (time based)
        try:
            engagement_percent = (attentive_time / total_duration * 100) if (total_duration and total_duration > 0) else None
            distraction_percent = (distracted_time / total_duration * 100) if (total_duration and total_duration > 0) else None
        except Exception:
            engagement_percent = None
            distraction_percent = None

        try:
            print(f"DEBUG [DB]: Attempting to log data for User ID {user_id}. Data: Attentive={attentive_time:.2f}s, Distracted={distracted_time:.2f}s")
            # Use explicit column list; columns may exist from initialization and migration
            self.cursor.execute(
                """INSERT INTO logs (user_id, start_time, end_time, total_duration, attentive_time, distracted_time, emotion_data, engagement_percent, distraction_percent, emotion_percentages)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, start_time, end_time, total_duration, attentive_time, distracted_time, emotion_data_json, engagement_percent, distraction_percent, emotion_percentages_json)
            )
            self.conn.commit()
            print(f"DEBUG [DB]: Successfully logged session of {total_duration:.2f}s for user {user_id}.")
            return True
        except sqlite3.Error as e:
            print(f"ERROR [DB]: SQLite insertion or commit failed. Error: {e}")
            self.conn.rollback()
            return False
        except Exception as e:
            print(f"CRITICAL ERROR [DB]: An unexpected error occurred during logging: {e}")
            return False

    # --- Reporting Methods ---

    def get_user_logs(self, user_id):
        """Fetches all engagement logs for a specific user ID."""
        try:
            self.cursor.execute("""
                SELECT id, start_time, end_time, total_duration, attentive_time, distracted_time, emotion_data, engagement_percent, distraction_percent, emotion_percentages
                FROM logs WHERE user_id=? ORDER BY start_time DESC
            """, (user_id,))

            rows = self.cursor.fetchall()

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=[
                'ID', 'Start Time', 'End Time', 'Total Duration (s)', 'Attentive Time (s)', 'Distracted Time (s)',
                'Emotion Data JSON', 'Engagement %', 'Distraction %', 'Emotion Percentages JSON'
            ])

            # Deserialize JSON emotion data for display/charting
            df['Emotion Data'] = df['Emotion Data JSON'].apply(lambda x: json.loads(x) if x else {})
            df['Emotion Percentages'] = df['Emotion Percentages JSON'].apply(lambda x: json.loads(x) if x else {})

            df = df.drop(columns=['Emotion Data JSON', 'Emotion Percentages JSON'])

            return df
        except sqlite3.Error as e:
            print(f"ERROR [DB]: Error fetching user logs: {e}")
            return pd.DataFrame()

    def get_all_logs(self):
        """Fetches all engagement logs for the admin dashboard (joined with username)."""
        try:
            self.cursor.execute("""
                SELECT l.id, u.username, l.start_time, l.end_time, l.total_duration, l.attentive_time, l.distracted_time, l.emotion_data, l.engagement_percent, l.distraction_percent, l.emotion_percentages
                FROM logs l JOIN users u ON l.user_id = u.id
                ORDER BY l.start_time DESC
            """)
            rows = self.cursor.fetchall()

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=[
                'ID', 'User', 'Start Time', 'End Time', 'Total Duration (s)', 'Attentive Time (s)', 'Distracted Time (s)',
                'Emotion Data JSON', 'Engagement %', 'Distraction %', 'Emotion Percentages JSON'
            ])

            df['Emotion Data'] = df['Emotion Data JSON'].apply(lambda x: json.loads(x) if x else {})
            df['Emotion Percentages'] = df['Emotion Percentages JSON'].apply(lambda x: json.loads(x) if x else {})

            df = df.drop(columns=['Emotion Data JSON', 'Emotion Percentages JSON'])

            return df
        except sqlite3.Error as e:
            print(f"ERROR [DB]: Error fetching all logs: {e}")
            return pd.DataFrame()
