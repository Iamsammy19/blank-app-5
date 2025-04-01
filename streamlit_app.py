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
        self._load_data_initial()
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

    def hash_password(self, password):
        """Hash a password for storing."""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def verify_password(self, stored_hash, provided_password):
        """Verify a stored password against one provided by user"""
        return bcrypt.checkpw(provided_password.encode('utf-8'), stored_hash.encode('utf-8'))

    def create_user(self, username, password):
        """Create a new user in the database"""
        with sqlite3.connect(str(DB_PATH)) as conn:
            c = conn.cursor()
            password_hash = self.hash_password(password)
            try:
                c.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, password_hash)
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                raise ValueError("Username already exists")
            except Exception as e:
                raise Exception(f"Failed to create user: {str(e)}")

    def authenticate_user(self, username, password):
        """Authenticate a user"""
        with sqlite3.connect(str(DB_PATH)) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT password_hash FROM users WHERE username = ?",
                (username,)
            )
            result = c.fetchone()
            if result:
                stored_hash = result[0]
                return self.verify_password(stored_hash, password)
            return False

def show_login_page(predictor):
    """Display login/registration interface"""
    st.title("Football Predictor Login")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if predictor.authenticate_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials")

    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Register")
            
            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords don't match")
                else:
                    try:
                        predictor.create_user(new_username, new_password)
                        st.success("Registration successful! Please login.")
                    except Exception as e:
                        st.error(f"Registration failed: {str(e)}")

def show_prediction_page(predictor):
    """Display the main prediction interface"""
    st.title(f"⚽ Football Predictor")
    st.write(f"Welcome, {st.session_state.username}!")
    
    # Team selection
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", options=list(predictor.team_mapping.keys()))
    with col2:
        away_team = st.selectbox("Away Team", options=list(predictor.team_mapping.keys()))
    
    # Make prediction
    if st.button("Predict Match Outcome"):
        try:
            if not predictor.model:
                st.error("Prediction model not loaded")
                return
                
            home_id = predictor.team_mapping[home_team]
            away_id = predictor.team_mapping[away_team]
            
            # Get team form data
            home_form = predictor.matches_df[
                predictor.matches_df['home_team_api_id'] == home_id]['home_team_form'].mean()
            away_form = predictor.matches_df[
                predictor.matches_df['away_team_api_id'] == away_id]['away_team_form'].mean()
            
            # Prepare features
            features = pd.DataFrame([[home_form, away_form]], 
                                  columns=predictor.feature_names)
            
            # Predict
            pred_home_goals = predictor.model.predict(features)[0]
            pred_away_goals = pred_home_goals * 0.8  # Simple adjustment for away performance
            
            # Display results
            st.subheader("Prediction Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Home Goals", f"{pred_home_goals:.1f}")
            col2.metric("Away Goals", f"{pred_away_goals:.1f}")
            col3.metric("Winner", 
                       home_team if pred_home_goals > pred_away_goals else away_team if pred_away_goals > pred_home_goals else "Draw")
            
            # Poisson probabilities
            st.subheader("Scoreline Probabilities")
            max_goals = 5
            prob_matrix = np.zeros((max_goals+1, max_goals+1))
            
            for i in range(max_goals+1):
                for j in range(max_goals+1):
                    prob_matrix[i,j] = poisson.pmf(i, pred_home_goals) * poisson.pmf(j, pred_away_goals)
            
            # Display as a dataframe
            prob_df = pd.DataFrame(prob_matrix, 
                                 index=[f"Home {i}" for i in
