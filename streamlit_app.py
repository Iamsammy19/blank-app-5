import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import sqlite3
import os
import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import bcrypt
from pathlib import Path
import tempfile
import urllib.request
from datetime import datetime

# Handle optional psutil import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    st.sidebar.warning("psutil not installed - memory monitoring disabled")

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                  format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DB_SOURCES = [
    "https://github.com/jokecamp/FootballData/raw/master/openFootballData/database.sqlite",
    "https://drive.google.com/uc?export=download&id=1G1x3UWdQl4zZ3N1-Bk1Q2Q4LdR9D7Qy_",
    "https://www.dropbox.com/s/8u3wq5q7v9y5s9x/database.sqlite?dl=1"
]

# --- Path Configuration ---
BASE_DIR = Path(__file__).parent
DB_PATH = Path(tempfile.gettempdir()) / "football_predictor.db"
MODEL_PATH = Path(tempfile.gettempdir()) / "xgboost_model.pkl"
SOCCER_DB_PATH = BASE_DIR / "soccer.sqlite"

class FootballPredictor:
    def __init__(self):
        self.is_cloud = os.getenv('IS_STREAMLIT_CLOUD', False)
        self.data_limit = 20_000 if self.is_cloud else 50_000
        self.matches_df = pd.DataFrame()
        self.team_mapping = {}
        self.model = None
        self.feature_names = []
        
        self.init_db()
        self.load_data()
        self.load_model()
        self._verify_environment()

    def init_db(self):
        """Initialize the SQLite database connection and setup."""
        try:
            # Ensure the database file exists at DB_PATH
            if not DB_PATH.exists():
                logging.info(f"Database not found at {DB_PATH}. Attempting download...")
                downloaded = False
                for url in DB_SOURCES:
                    try:
                        logging.info(f"Downloading from {url}")
                        urllib.request.urlretrieve(url, DB_PATH)
                        # Verify it's a SQLite database
                        with open(DB_PATH, 'rb') as f:
                            header = f.read(16)  # SQLite header is 16 bytes
                            if header.startswith(b"SQLite format 3"):
                                logging.info(f"Valid SQLite database downloaded from {url}")
                                downloaded = True
                                break
                            else:
                                logging.warning(f"Downloaded file from {url} is not a SQLite database")
                                DB_PATH.unlink()  # Remove invalid file
                    except Exception as e:
                        logging.warning(f"Failed to download from {url}: {e}")
                        if DB_PATH.exists():
                            DB_PATH.unlink()  # Clean up partial download
                        continue

                if not downloaded:
                    logging.info("All downloads failed. Creating a minimal fallback database.")
                    # Create a new empty database as fallback
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS matches (
                            id INTEGER PRIMARY KEY,
                            date TEXT,
                            home_team TEXT,
                            away_team TEXT,
                            home_goals INTEGER,
                            away_goals INTEGER
                        )
                    """)
                    conn.commit()
                    conn.close()
                    logging.info(f"Fallback database created at {DB_PATH}")

            # Connect to the database
            self.conn = sqlite3.connect(DB_PATH)
            self.cursor = self.conn.cursor()
            logging.info(f"Database connection established at {DB_PATH}")

            # Ensure the matches table exists
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY,
                    date TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    home_goals INTEGER,
                    away_goals INTEGER
                )
            """)
            self.conn.commit()

        except sqlite3.DatabaseError as e:
            logging.error(f"SQLite database error: {e}")
            st.error(f"Database error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in init_db: {e}")
            st.error(f"Application error: {str(e)}")
            raise

    def _verify_environment(self):
        """Environment checks with psutil fallback"""
        if self.is_cloud and not Path('/tmp').exists():
            st.error("Invalid cloud filesystem configuration")
            st.stop()
            
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            st.session_state.available_mb = mem.available / (1024 ** 2)
            if st.session_state.available_mb < 500:
                st.warning(f"Low memory: {st.session_state.available_mb:.0f}MB available")
        else:
            # Lightweight memory check
            try:
                test_array = np.zeros((1000, 1000))  # ~8MB
                del test_array
            except MemoryError:
                st.warning("Memory constraints detected - reducing functionality")
                self.data_limit = 10_000

    # [Keep all other methods unchanged from previous implementation]
    # load_data(), _add_features(), load_model(), 
    # _create_fallback_model(), authenticate_user(), etc.

def main():
    st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")
    
    # Initialize session
    if 'logged_in' not in st.session_state:
        st.session_state.update({
            'logged_in': False,
            'username': None,
            'login_attempts': 0,
            'available_mb': 'Unknown'
        })

    try:
        predictor = FootballPredictor()
        
        # Debug panel
        if st.sidebar.checkbox("Show system info"):
            st.sidebar.metric("Available Memory", f"{st.session_state.available_mb:.0f}MB")
            st.sidebar.write(f"Data: {len(predictor.matches_df)} matches")
            st.sidebar.write(f"Model: {type(predictor.model).__name__}")
            
            if st.sidebar.button("Refresh App"):
                st.cache_data.clear()
                st.rerun()
        
        # Authentication flow
        if not st.session_state.logged_in:
            show_login_page(predictor)
        else:
            show_prediction_page(predictor)
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.stop()

# [Keep your existing UI functions show_login_page() and show_prediction_page()]

if __name__ == "__main__":
    main()
