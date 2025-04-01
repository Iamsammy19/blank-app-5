import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import sqlite3
import os
import logging
from fuzzywuzzy import process
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from datetime import datetime
import joblib
import bcrypt
from pathlib import Path
import tempfile
import urllib.request

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# --- Path Configuration ---
BASE_DIR = Path(__file__).parent

# Use /tmp for writable files in Streamlit Cloud
DB_PATH = Path(tempfile.gettempdir()) / "football_predictor.db"
MODEL_PATH = Path(tempfile.gettempdir()) / "xgboost_model.pkl"

# Soccer DB path (will download if missing)
SOCCER_DB_PATH = BASE_DIR / "soccer.sqlite"

# --- Database Download (if missing) ---
def ensure_database_exists():
    if not SOCCER_DB_PATH.exists():
        try:
            with st.spinner("Downloading soccer database (first-time setup)..."):
                url = "https://github.com/jokecamp/FootballData/raw/master/openFootballData/database.sqlite"
                urllib.request.urlretrieve(url, str(SOCCER_DB_PATH))
                st.success("Database downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download database: {str(e)}")
            st.stop()

ensure_database_exists()

# --- Secrets Configuration ---
try:
    FOOTBALL_DATA_API_KEY = st.secrets["7563e489e2c84b77a0e4f8d7116dc19c"]
    ODDS_API_KEY = st.secrets["c9b67d8274042fb5755ad88c3a63eab7"]
except (KeyError, FileNotFoundError):
    # Fallback for local development
    from dotenv import load_dotenv
    load_dotenv()
    FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
    ODDS_API_KEY = os.getenv("ODDS_API_KEY")

class FootballPredictor:
    def __init__(self):
        self.matches_df = pd.DataFrame()
        self.team_mapping = {}
        self.model = None
        self.feature_names = []
        self.init_db()
        self.load_model()

    def init_db(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(str(DB_PATH)) as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    # ... (keep all other FootballPredictor methods unchanged) ...

# --- Optimized Data Loading ---
@st.cache_data(ttl=3600, show_spinner="Loading match data...")
def load_data():
    """Load only essential match data with error handling."""
    try:
        if not SOCCER_DB_PATH.exists():
            st.error("Database file missing! Please ensure soccer.sqlite exists.")
            st.stop()

        with sqlite3.connect(str(SOCCER_DB_PATH)) as conn:
            # Only load essential columns to save memory
            matches_df = pd.read_sql_query("""
                SELECT 
                    id, date, 
                    home_team_api_id, away_team_api_id,
                    home_team_goal, away_team_goal
                FROM Match 
                WHERE date > '2010-01-01'
                LIMIT 50000
            """, conn)

            team_mapping = pd.read_sql_query(
                "SELECT team_api_id, team_long_name FROM Team", conn
            ).set_index('team_long_name')['team_api_id'].to_dict()

        matches_df = add_features(matches_df)
        return matches_df, tuple(team_mapping.items())
    except Exception as e:
        logging.error(f"Data loading failed: {str(e)}")
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame(), ()

# ... (keep all other functions unchanged) ...

def main():
    st.title("⚽ Football Predictor")

    # Display debug info in sidebar
    with st.sidebar:
        st.subheader("Debug Info")
        st.write(f"Database path: {SOCCER_DB_PATH}")
        st.write(f"File exists: {SOCCER_DB_PATH.exists()}")
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")

    predictor = FootballPredictor()
    matches_df, team_mapping_tuple = load_data()
    
    if matches_df.empty:
        st.error("No data loaded - check database file")
        st.stop()

    predictor.matches_df = matches_df
    predictor.team_mapping = dict(team_mapping_tuple)

    # ... (rest of your existing UI code) ...

if __name__ == "__main__":
    main()
