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
import psutil
from datetime import datetime

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

    def _verify_environment(self):
        """Validate system requirements"""
        if self.is_cloud and not Path('/tmp').exists():
            st.error("Invalid cloud filesystem configuration")
            st.stop()
            
        available_mb = psutil.virtual_memory().available / (1024 ** 2)
        if available_mb < 500:
            st.warning(f"Low memory: {available_mb:.0f}MB available")

    def init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password_hash TEXT,
                    last_attempt TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    @st.cache_data(ttl=3600)
    def load_data(_self):
        """Load match data with multiple fallback sources"""
        if not SOCCER_DB_PATH.exists():
            for url in DB_SOURCES:
                try:
                    with st.spinner(f"Downloading from {url.split('//')[1].split('/')[0]}..."):
                        urllib.request.urlretrieve(url, str(SOCCER_DB_PATH))
                        if SOCCER_DB_PATH.stat().st_size > 10_000_000:
                            break
                except Exception:
                    continue
            else:
                st.error("""
                Download failed. Please manually download:
                https://www.kaggle.com/datasets/hugomathien/soccer
                Save as 'soccer.sqlite' in app directory
                """)
                st.stop()

        with sqlite3.connect(str(SOCCER_DB_PATH)) as conn:
            matches_df = pd.read_sql_query(f"""
                SELECT id, date, home_team_api_id, away_team_api_id,
                       home_team_goal, away_team_goal
                FROM Match 
                WHERE date > '2010-01-01'
                LIMIT {_self.data_limit}
            """, conn)

            team_mapping = pd.read_sql_query(
                "SELECT team_api_id, team_long_name FROM Team", conn
            ).set_index('team_long_name')['team_api_id'].to_dict()

        matches_df = self._add_features(matches_df)
        return matches_df, team_mapping

    def _add_features(self, df):
        """Generate predictive features"""
        if df.empty:
            return df
            
        df['date'] = pd.to_datetime(df['date'])
        df['home_team_form'] = df.groupby('home_team_api_id')['home_team_goal'] \
            .transform(lambda x: x.rolling(5, min_periods=1).mean().fillna(0))
        df['away_team_form'] = df.groupby('away_team_api_id')['away_team_goal'] \
            .transform(lambda x: x.rolling(5, min_periods=1).mean().fillna(0))
        return df

    def load_model(self):
        """Smart model loader with fallback training"""
        try:
            if MODEL_PATH.exists():
                with open(MODEL_PATH, 'rb') as f:
                    self.model = joblib.load(f)
                    self.feature_names = getattr(self.model, 'feature_names_in_', [])
                    return

            if len(self.matches_df) < 100:
                raise ValueError("Insufficient training data")
                
            features = self.matches_df[['home_team_form', 'away_team_form']].dropna()
            target = self.matches_df['home_team_goal']
            
            self.model = xgb.XGBRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            )
            self.model.fit(features, target)
            
            with open(MODEL_PATH, 'wb') as f:
                joblib.dump(self.model, f)
            
            self.feature_names = features.columns.tolist()
            
        except Exception as e:
            st.error(f"Model error: {str(e)}")
            self._create_fallback_model()

    def _create_fallback_model(self):
        """Basic predictions when model fails"""
        class FallbackModel:
            def predict(self, home_id, away_id, matches_df):
                home_avg = matches_df[matches_df['home_team_api_id'] == home_id]['home_team_goal'].mean() or 1.2
                away_avg = matches_df[matches_df['away_team_api_id'] == away_id]['away_team_goal'].mean() or 0.8
                return (
                    max(0, int(np.random.normal(home_avg, 0.5))),
                    max(0, int(np.random.normal(away_avg, 0.5)))
                )
        
        self.model = FallbackModel()
        self.feature_names = ['fallback_mode']
        st.warning("Using simplified prediction mode")

    # ... [Keep existing auth methods but add this update] ...
    def authenticate_user(self, username, password):
        """Secure authentication with attempt tracking"""
        if getattr(st.session_state, 'login_attempts', 0) >= 3:
            st.error("Too many attempts. Wait 5 minutes.")
            return False, "Account locked"
            
        with sqlite3.connect(str(DB_PATH)) as conn:
            c = conn.cursor()
            c.execute("SELECT password_hash, last_attempt FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            
            if not result:
                st.session_state.login_attempts = getattr(st.session_state, 'login_attempts', 0) + 1
                return False, "Invalid credentials"
                
            hashed, last_attempt = result
            if last_attempt and (datetime.now() - datetime.fromisoformat(last_attempt)).seconds < 300:
                return False, "Try again later"
                
            if self.verify_password(password, hashed):
                c.execute("UPDATE users SET last_attempt = NULL WHERE username = ?", (username,))
                conn.commit()
                st.session_state.login_attempts = 0
                return True, "Login successful"
            else:
                c.execute("UPDATE users SET last_attempt = CURRENT_TIMESTAMP WHERE username = ?", (username,))
                conn.commit()
                st.session_state.login_attempts = getattr(st.session_state, 'login_attempts', 0) + 1
                return False, "Invalid password"

def main():
    st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")
    
    # Initialize session
    if 'logged_in' not in st.session_state:
        st.session_state.update({
            'logged_in': False,
            'username': None,
            'login_attempts': 0
        })

    try:
        predictor = FootballPredictor()
        
        # Debug panel
        if st.sidebar.checkbox("Show debug info"):
            st.sidebar.write(f"Data: {len(predictor.matches_df)} matches")
            st.sidebar.write(f"Model: {predictor.model.__class__.__name__}")
            st.sidebar.write(f"Features: {predictor.feature_names}")
            
            if st.sidebar.button("Force refresh"):
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

# [Keep your existing show_login_page() and show_prediction_page() functions]
# Just replace all st.experimental_rerun() calls with st.rerun()

if __name__ == "__main__":
    main()
