import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import sqlite3
import os
import logging
from fuzzywuzzy import process
import xgboost as xgb
from sklearn.model_selection import train_test_split
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
DB_PATH = Path(tempfile.gettempdir()) / "football_predictor.db"
MODEL_PATH = Path(tempfile.gettempdir()) / "xgboost_model.pkl"
SOCCER_DB_PATH = BASE_DIR / "soccer.sqlite"

class FootballPredictor:
    def __init__(self):
        self.matches_df = pd.DataFrame()
        self.team_mapping = {}
        self.model = None
        self.feature_names = []
        self.init_db()
        self._load_data_initial()  # Changed to use the new loading approach
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

    def _load_data_initial(self):
        """Wrapper to load data and update instance variables"""
        self.matches_df, self.team_mapping = self._load_data_cached()

    @st.cache_data(ttl=3600)
    def _load_data_cached(_dummy_arg=None):
        """Cached data loading (static method pattern)"""
        try:
            if not SOCCER_DB_PATH.exists():
                raise FileNotFoundError("Database file missing")
                
            with sqlite3.connect(str(SOCCER_DB_PATH)) as conn:
                matches_df = pd.read_sql_query("""
                    SELECT id, date, home_team_api_id, away_team_api_id,
                           home_team_goal, away_team_goal
                    FROM Match 
                    WHERE date > '2010-01-01'
                    LIMIT 50000
                """, conn)

                team_mapping = pd.read_sql_query(
                    "SELECT team_api_id, team_long_name FROM Team", conn
                ).set_index('team_long_name')['team_api_id'].to_dict()

            matches_df = FootballPredictor._add_features_static(matches_df)
            return matches_df, team_mapping
            
        except Exception as e:
            logging.error(f"Data loading failed: {str(e)}")
            return pd.DataFrame(), {}

    @staticmethod
    def _add_features_static(df):
        """Static version of add_features for cached method"""
        if df.empty:
            return df
            
        df['date'] = pd.to_datetime(df['date'])
        df['home_team_form'] = df.groupby('home_team_api_id')['home_team_goal'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().fillna(0))
        df['away_team_form'] = df.groupby('away_team_api_id')['away_team_goal'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().fillna(0))
        return df

    def add_features(self, df):
        """Instance method wrapper for feature adding"""
        return self._add_features_static(df)

    def load_model(self):
        """Load or train model with error handling"""
        try:
            if MODEL_PATH.exists():
                self.model = joblib.load(str(MODEL_PATH))
                self.feature_names = getattr(self.model, 'feature_names_in_', [])
                return

            # Fallback: Train new model
            if not self.matches_df.empty:
                features = self.matches_df[['home_team_form', 'away_team_form']]
                target = self.matches_df['home_team_goal']
                
                X_train, _, y_train, _ = train_test_split(
                    features, target, test_size=0.2, random_state=42)
                
                self.model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42
                )
                self.model.fit(X_train, y_train)
                self.model.feature_names_in_ = features.columns.tolist()
                self.feature_names = features.columns.tolist()
                
                joblib.dump(self.model, str(MODEL_PATH))
                
        except Exception as e:
            logging.error(f"Model error: {str(e)}")
            self.model = None

    # ... (Keep all other existing methods like hash_password, authenticate_user, etc.)

def main():
    st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")
    
    # Initialize predictor with error handling
    try:
        predictor = FootballPredictor()
        
        # Debug info
        if st.sidebar.checkbox("Show debug info"):
            st.sidebar.write("Model status:", "Loaded" if predictor.model else "Not loaded")
            st.sidebar.write("Data shape:", predictor.matches_df.shape)
        
        # Authentication flow
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
            
        if not st.session_state.logged_in:
            show_login_page(predictor)
        else:
            show_prediction_page(predictor)
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.stop()

# ... (Keep all your existing UI functions like show_login_page, show_prediction_page)

if __name__ == "__main__":
    main()
