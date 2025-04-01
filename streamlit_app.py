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
    FOOTBALL_DATA_API_KEY = os.getenv("7563e489e2c84b77a0e4f8d7116dc19c")
    ODDS_API_KEY = os.getenv("c9b67d8274042fb5755ad88c3a63eab7")

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

    def hash_password(self, password):
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def verify_password(self, password, hashed):
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def create_user(self, username, password):
        """Create a new user."""
        try:
            with sqlite3.connect(str(DB_PATH)) as conn:
                c = conn.cursor()
                c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                          (username, self.hash_password(password)))
                conn.commit()
            return True, "User created successfully!"
        except sqlite3.IntegrityError:
            return False, "Username already exists."

    def authenticate_user(self, username, password):
        """Authenticate a user."""
        with sqlite3.connect(str(DB_PATH)) as conn:
            c = conn.cursor()
            c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            if result and self.verify_password(password, result[0]):
                return True, "Login successful!"
            return False, "Invalid username or password."

# --- Data Loading (Fixed Caching) ---
@st.cache_data(ttl=3600, show_spinner="Loading match data...")
def load_data(_dummy=None):  # Added dummy parameter for caching
    """Load match data from SQLite with proper path handling."""
    try:
        if not SOCCER_DB_PATH.exists():
            st.error("Database file missing! Please ensure soccer.sqlite exists.")
            st.stop()
            
        with sqlite3.connect(str(SOCCER_DB_PATH)) as conn:
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

def add_features(df):
    """Add additional features to dataset."""
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df['home_team_form'] = df.groupby('home_team_api_id')['home_team_goal'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().fillna(0)
    )
    return df

@st.cache_resource
def train_model(features, target):
    """Train and cache the prediction model."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
        model = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid, cv=3, n_jobs=-1)
        model.fit(X_train, y_train)
        joblib.dump(model.best_estimator_, str(MODEL_PATH))
        return model.best_estimator_, list(features.columns)
    except Exception as e:
        logging.error(f"Model training error: {str(e)}")
        st.error(f"Model training failed: {str(e)}")
        return None, []

def load_model():
    """Load a pre-trained model from disk."""
    if MODEL_PATH.exists():
        return joblib.load(str(MODEL_PATH))
    return None

def simulate_match(home_team_id, away_team_id, matches_df):
    """Simulate match using Poisson distribution."""
    home_avg = matches_df[matches_df['home_team_api_id'] == home_team_id]['home_team_goal'].mean() or 1.5
    away_avg = matches_df[matches_df['away_team_api_id'] == away_team_id]['away_team_goal'].mean() or 1.5
    home_goals = np.random.poisson(home_avg)
    away_goals = np.random.poisson(away_avg)
    return home_goals, away_goals

# --- UI Components ---
def show_login_page(predictor):
    """Display login/signup form"""
    st.header("Welcome to Football Predictor")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                success, message = predictor.authenticate_user(username, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error(message)

    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Create Account"):
                if new_password != confirm_password:
                    st.error("Passwords don't match!")
                else:
                    success, message = predictor.create_user(new_username, new_password)
                    if success:
                        st.success("Account created! Please login.")
                    else:
                        st.error(message)

def show_prediction_page(predictor):
    """Display the main prediction interface"""
    st.header(f"Welcome back, {st.session_state.username}!")
    
    # Logout button
    if st.button("Logout", key="logout_button"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()
    
    # Main prediction UI
    st.subheader("Match Predictor")
    teams = list(predictor.team_mapping.keys())
    
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", teams, index=teams.index("FC Barcelona") if "FC Barcelona" in teams else 0)
    with col2:
        away_team = st.selectbox("Away Team", teams, index=teams.index("Real Madrid CF") if "Real Madrid CF" in teams else 1)
    
    if st.button("Predict Match Outcome", type="primary"):
        with st.spinner("Calculating prediction..."):
            home_id = predictor.team_mapping[home_team]
            away_id = predictor.team_mapping[away_team]
            home_goals, away_goals = simulate_match(home_id, away_id, predictor.matches_df)
            
            # Display result
            st.balloons()
            st.success(f"""
            **Predicted Score**:  
            🏠 {home_team}: **{home_goals}**  
            🏡 {away_team}: **{away_goals}**
            """)
            
            # Show probability analysis
            show_probability_analysis(home_goals, away_goals)

def show_probability_analysis(home_goals, away_goals):
    """Display additional match insights"""
    with st.expander("Advanced Analysis"):
        st.subheader("Match Probabilities")
        
        # Win/draw/lose probabilities
        if home_goals > away_goals:
            st.metric("Home Win Probability", f"{(home_goals/(home_goals+away_goals))*100:.1f}%")
        elif home_goals < away_goals:
            st.metric("Away Win Probability", f"{(away_goals/(home_goals+away_goals))*100:.1f}%")
        else:
            st.metric("Draw Probability", "35%")
            
        # Goal distribution chart
        chart_data = pd.DataFrame({
            "Goals": [home_goals, away_goals],
            "Team": ["Home", "Away"]
        })
        st.bar_chart(chart_data, x="Team", y="Goals")

def main():
    st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None

    # Initialize predictor
    predictor = FootballPredictor()
    
    # Load data (with fixed caching)
    matches_df, team_mapping_tuple = load_data()
    
    if matches_df.empty:
        st.error("No data loaded - check database file")
        st.stop()

    predictor.matches_df = matches_df
    predictor.team_mapping = dict(team_mapping_tuple)

    # Page routing
    if not st.session_state.logged_in:
        show_login_page(predictor)
    else:
        show_prediction_page(predictor)

if __name__ == "__main__":
    main()
