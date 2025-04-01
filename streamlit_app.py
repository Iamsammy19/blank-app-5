# football_predictor.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
from bs4 import BeautifulSoup
import sqlite3
import os
import hashlib
import logging
from fuzzywuzzy import process
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from datetime import datetime, timedelta
import plotly.express as px
import joblib
from dotenv import load_dotenv
import json
import bcrypt

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# API Keys with validation
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
for key, name in [(FOOTBALL_DATA_API_KEY, "Football Data"), 
                  (ODDS_API_KEY, "Odds"), 
                  (WEATHER_API_KEY, "Weather")]:
    if not key:
        logging.warning(f"{name} API key not found. Some features may be limited.")

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer/odds/"
FOOTBALL_DATA_URL = "https://api.football-data.org/v4/matches"
DB_PATH = os.getenv("DB_PATH", "football_predictor.db")
SOCCER_DB_PATH = os.getenv("SOCCER_DB_PATH", "soccer.sqlite")
MODEL_PATH = "xgboost_model.pkl"

# Football Predictor Class
class FootballPredictor:
    def __init__(self):
        self.matches_df = pd.DataFrame()
        self.team_mapping = {}
        self.model = None
        self.feature_names = []
        self.init_db()

    # --------------------------
    # Database and User Management
    # --------------------------
    def init_db(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id INTEGER,
                    favorite_teams TEXT,
                    notification_preferences TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    user_id INTEGER,
                    match_id TEXT,
                    prediction_accuracy INTEGER,
                    comments TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_feedback_match ON feedback(match_id)")
            conn.commit()

    def hash_password(self, password):
        """Hash a password securely using bcrypt."""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def verify_password(self, password, hashed):
        """Verify a password against its hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def create_user(self, username, password):
        """Create a new user account with validation."""
        if len(username) < 4 or len(password) < 8:
            return False, "Username must be 4+ characters and password 8+ characters."
        try:
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                         (username, self.hash_password(password)))
                conn.commit()
            logging.info(f"User {username} created successfully.")
            return True, "User created successfully!"
        except sqlite3.IntegrityError:
            return False, "Username already exists."

    def authenticate_user(self, username, password):
        """Authenticate a user."""
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            if result and self.verify_password(password, result[0]):
                logging.info(f"User {username} authenticated successfully.")
                return True, "Login successful!"
            return False, "Invalid username or password."

    # --------------------------
    # Data Loading and Processing
    # --------------------------
    @st.cache_data(ttl=3600)
    def load_data(self):
        """Load and preprocess match data from SQLite or API."""
        try:
            with sqlite3.connect(SOCCER_DB_PATH) as conn:
                self.matches_df = pd.read_sql_query("SELECT * FROM Match", conn)
                self.team_mapping = pd.read_sql_query(
                    "SELECT team_api_id, team_long_name FROM Team", conn
                ).set_index('team_long_name')['team_api_id'].to_dict()
            self.matches_df = self.add_features(self.matches_df)
            logging.info("Match data loaded successfully from SQLite.")
        except Exception as e:
            logging.error(f"Error loading data from SQLite: {str(e)}")
            st.error("Failed to load match data from SQLite. Attempting API fallback.")
            self.fetch_fresh_data()
        return self.matches_df, self.team_mapping

    def fetch_fresh_data(self):
        """Fetch fresh data from Football Data API if available."""
        if not FOOTBALL_DATA_API_KEY:
            logging.warning("No API key for fresh data fetch.")
            return
        try:
            headers = {'X-Auth-Token': FOOTBALL_DATA_API_KEY}
            response = requests.get(FOOTBALL_DATA_URL, headers=headers)
            response.raise_for_status()
            data = response.json()
            # Simplified: assumes API returns match data in a compatible format
            self.matches_df = pd.DataFrame(data['matches'])
            self.matches_df = self.add_features(self.matches_df)
            logging.info("Fresh match data fetched from API.")
        except Exception as e:
            logging.error(f"Error fetching fresh data: {str(e)}")
            self.matches_df = pd.DataFrame()

    def add_features(self, df):
        """Add advanced features to the dataset."""
        if df.empty:
            return df
        df['date'] = pd.to_datetime(df['date'])
        df['home_team_form'] = df.groupby('home_team_api_id')['home_team_goal'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().fillna(0)
        )
        df['away_team_form'] = df.groupby('away_team_api_id')['away_team_goal'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().fillna(0)
        )
        df['h2h_home_wins'] = df.groupby(['home_team_api_id', 'away_team_api_id'])['home_team_goal'].transform(
            lambda x: (x > x.shift()).cumsum().fillna(0)
        )
        df['h2h_away_wins'] = df.groupby(['home_team_api_id', 'away_team_api_id'])['away_team_goal'].transform(
            lambda x: (x > x.shift()).cumsum().fillna(0)
        )
        return df

    # --------------------------
    # Model Training and Prediction
    # --------------------------
    @st.cache_resource
    def train_model(self, features, target):
        """Train and cache the prediction model with hyperparameter tuning."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1]
            }
            model = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid, cv=3, n_jobs=-1)
            model.fit(X_train, y_train)
            self.model = model.best_estimator_
            self.feature_names = list(features.columns)
            joblib.dump(self.model, MODEL_PATH)
            logging.info(f"Model trained with best params: {model.best_params_}")
            return self.model, self.feature_names
        except Exception as e:
            logging.error(f"Model training error: {str(e)}")
            st.error("Failed to train model. Using fallback predictions.")
            return None, []

    def load_model(self):
        """Load a pre-trained model from disk."""
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            logging.info("Pre-trained model loaded from disk.")
        else:
            logging.info("No pre-trained model found. Training required.")

    def prepare_match_features(self, home_team_id, away_team_id):
        """Prepare features for a match prediction."""
        # Simplified: assumes features match training data
        home_form = self.matches_df[self.matches_df['home_team_api_id'] == home_team_id]['home_team_form'].mean()
        away_form = self.matches_df[self.matches_df['away_team_api_id'] == away_team_id]['away_team_form'].mean()
        h2h_home = self.matches_df[(self.matches_df['home_team_api_id'] == home_team_id) & 
                                  (self.matches_df['away_team_api_id'] == away_team_id)]['h2h_home_wins'].mean()
        h2h_away = self.matches_df[(self.matches_df['home_team_api_id'] == home_team_id) & 
                                  (self.matches_df['away_team_api_id'] == away_team_id)]['h2h_away_wins'].mean()
        return [home_form or 0, away_form or 0, h2h_home or 0, h2h_away or 0]

    def simulate_match(self, home_team_id, away_team_id, home_team_name, away_team_name):
        """Simulate a match and return predictions."""
        try:
            features = self.prepare_match_features(home_team_id, away_team_id)
            if self.model and self.feature_names:
                prediction = self.model.predict(np.array([features]))[0]
                home_goals, away_goals = int(round(prediction)), int(round(prediction * 0.9))  # Simplified adjustment
            else:
                home_avg = self.matches_df[self.matches_df['home_team_api_id'] == home_team_id]['home_team_goal'].mean() or 1.5
                away_avg = self.matches_df[self.matches_df['away_team_api_id'] == away_team_id]['away_team_goal'].mean() or 1.5
                home_goals = np.random.poisson(home_avg)
                away_goals = np.random.poisson(away_avg)

            probabilities = {
                'Home Win': self.calculate_win_probability(home_goals, away_goals),
                'Draw': self.calculate_draw_probability(home_goals, away_goals),
                'Away Win': self.calculate_win_probability(away_goals, home_goals)
            }
            odds = self.fetch_odds(home_team_name, away_team_name)
            value_bets = self.calculate_value_bets(probabilities, odds)
            best_bets = {
                outcome: (f"Bet {min(self.kelly_criterion(probabilities[outcome], odds[outcome]) * 100, 10):.1f}% of bankroll", 
                         f"{value_bets[outcome]:.1f}%")
                for outcome in value_bets
            }
            return {
                'predicted_score': (home_goals, away_goals),
                'probabilities': probabilities,
                'odds': odds,
                'value_bets': value_bets,
                'best_bets': best_bets
            }
        except Exception as e:
            logging.error(f"Match simulation error: {str(e)}")
            return {
                'predicted_score': (0, 0),
                'probabilities': {'Home Win': 0, 'Draw': 0, 'Away Win': 0},
                'odds': {'Home Win': 1.0, 'Draw': 1.0, 'Away Win': 1.0},
                'value_bets': {},
                'best_bets': {}
            }

    def calculate_win_probability(self, goals1, goals2):
        """Calculate win probability using Poisson (placeholder)."""
        return poisson.pmf(goals1, goals1) * poisson.sf(goals2, goals2)

    def calculate_draw_probability(self, goals1, goals2):
        """Calculate draw probability (placeholder)."""
        return poisson.pmf(goals1, goals1) * poisson.pmf(goals2, goals2)

    def calculate_value_bets(self, probabilities, odds):
        """Calculate value bets."""
        return {k: (probabilities[k] * 100 - (1 / odds[k] * 100)) for k in probabilities if k in odds}

    def kelly_criterion(self, prob, odds):
        """Calculate Kelly Criterion bet size."""
        b = odds - 1
        p = prob
        q = 1 - p
        return (b * p - q) / b if b > 0 else 0

    def fetch_odds(self, home_team_name, away_team_name):
        """Fetch betting odds with caching."""
        cache_file = 'odds_cache.json'
        cache = {}
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        key = f"{home_team_name}_vs_{away_team_name}"
        if key in cache and (datetime.now() - datetime.fromisoformat(cache[key]['timestamp'])).seconds < 3600:
            return cache[key]['odds']

        if not ODDS_API_KEY:
            return {'Home Win': 1.0, 'Draw': 1.0, 'Away Win': 1.0}
        try:
            params = {'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h', 'sports': 'soccer'}
            response = requests.get(ODDS_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            match = process.extractOne(f"{home_team_name} vs {away_team_name}", 
                                     [event['sport_title'] for event in data])
            if match[1] > 80:
                odds = next((event['bookmakers'][0]['markets'][0]['outcomes'] 
                            for event in data if match[0] in event['sport_title']), None)
                if odds:
                    odds_dict = {o['name']: o['price'] for o in odds}
                    cache[key] = {'odds': odds_dict, 'timestamp': datetime.now().isoformat()}
                    with open(cache_file, 'w') as f:
                        json.dump(cache, f)
                    return odds_dict
            return {'Home Win': 1.0, 'Draw': 1.0, 'Away Win': 1.0}
        except Exception as e:
            logging.error(f"Error fetching odds: {str(e)}")
            return {'Home Win': 1.0, 'Draw': 1.0, 'Away Win': 1.0}

# --------------------------
# Streamlit UI
# --------------------------
def main():
    predictor = FootballPredictor()
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None

    if not st.session_state.logged_in:
        st.title("Football Predictor - Login/Signup")
        option = st.selectbox("Choose an option", ["Login", "Signup"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Submit"):
            if option == "Signup":
                success, message = predictor.create_user(username, password)
                st.write(message)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
            else:
                success, message = predictor.authenticate_user(username, password)
                st.write(message)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
    else:
        st.title(f"Welcome, {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()

        predictor.load_data()
        predictor.load_model()
        teams = list(predictor.team_mapping.keys())
        home_team = st.selectbox("Home Team", teams, format_func=lambda x: x)
        away_team = st.selectbox("Away Team", teams, format_func=lambda x: x)
        if st.button("Simulate Match"):
            home_id = predictor.team_mapping[home_team]
            away_id = predictor.team_mapping[away_team]
            result = predictor.simulate_match(home_id, away_id, home_team, away_team)
            st.write(f"Predicted Score: {home_team} {result['predicted_score'][0]} - {result['predicted_score'][1]} {away_team}")
            st.write("Probabilities:", result['probabilities'])
            st.write("Odds:", result['odds'])
            st.write("Best Bets:", result['best_bets'])
            fig = px.bar(x=list(result['probabilities'].keys()), y=[v*100 for v in result['probabilities'].values()],
                        labels={'x': 'Outcome', 'y': 'Probability (%)'}, title="Match Outcome Probabilities")
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
