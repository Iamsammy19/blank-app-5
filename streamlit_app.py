import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson, skellam
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
import joblib
import sqlite3
from pathlib import Path
import tempfile
import datetime
import time
import random
import logging
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    filename='football_predictor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# API Configuration
FOOTBALL_API_KEY = "7563e489e2c84b77a0e4f8d7116dc19c"
LIVE_ODDS_API_KEY = "c9b67d8274042fb5755ad88c3a63eab7"
FOOTBALL_API_URL = "https://api.football-data.org/v4"
LIVE_ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"

# Database Configuration
DB_PATH = Path(tempfile.gettempdir()) / "football_predictor.db"
MODEL_PATH = Path(tempfile.gettempdir()) / "football_model.pkl"
HISTORICAL_DATA_PATH = Path(tempfile.gettempdir()) / "historical_data.csv"

class UltimateFootballPredictor:
    def __init__(self):
        """Initialize with comprehensive error handling"""
        self.match_data = []
        self.odds_data = {}
        self.historical_data = pd.DataFrame()
        self.last_update = 0
        self.db_conn = None
        
        try:
            self._init_database()
            self._load_historical_data()
            self.model = self._load_or_train_model()
            self._fetch_initial_data()
        except Exception as e:
            logging.critical(f"Initialization failed: {str(e)}", exc_info=True)
            raise RuntimeError("System initialization failed") from e

    def _init_database(self) -> None:
        """Initialize SQLite database with error recovery"""
        try:
            self.db_conn = sqlite3.connect(DB_PATH, timeout=10)
            cursor = self.db_conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY,
                    date TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    home_goals INTEGER,
                    away_goals INTEGER,
                    home_odds REAL,
                    draw_odds REAL,
                    away_odds REAL
                )
            """)
            
            # Create prediction cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    match_id INTEGER PRIMARY KEY,
                    home_win REAL,
                    draw REAL,
                    away_win REAL,
                    btts_yes REAL,
                    last_updated TEXT
                )
            """)
            
            self.db_conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Database error: {str(e)}")
            raise RuntimeError("Database initialization failed") from e

    def _load_historical_data(self) -> None:
        """Load historical match data with fallback"""
        try:
            if HISTORICAL_DATA_PATH.exists():
                self.historical_data = pd.read_csv(HISTORICAL_DATA_PATH)
            else:
                # Create minimal historical data
                self.historical_data = pd.DataFrame({
                    'home_team': ['Team A', 'Team B'],
                    'away_team': ['Team B', 'Team A'],
                    'home_goals': [2, 1],
                    'away_goals': [1, 0]
                })
                self.historical_data.to_csv(HISTORICAL_DATA_PATH, index=False)
        except Exception as e:
            logging.warning(f"Historical data load failed: {str(e)}")
            self.historical_data = pd.DataFrame()

    def _load_or_train_model(self):
        """Load or train ML model with comprehensive error handling"""
        try:
            if MODEL_PATH.exists():
                try:
                    model = joblib.load(MODEL_PATH)
                    # Verify the model is properly trained
                    if hasattr(model, 'predict_proba'):
                        return model
                    raise NotFittedError("Model exists but isn't properly trained")
                except (EOFError, NotFittedError) as e:
                    logging.warning(f"Model load failed, retraining: {str(e)}")
            
            # Fallback model training
            logging.info("Training fallback model...")
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=1000, n_features=10)
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X, y)
            
            try:
                joblib.dump(model, MODEL_PATH)
            except Exception as e:
                logging.warning(f"Model save failed: {str(e)}")
            
            return model
        except Exception as e:
            logging.error(f"Model initialization failed: {str(e)}")
            raise RuntimeError("Model initialization failed") from e

    def _fetch_initial_data(self) -> None:
        """Fetch initial data with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self._fetch_live_data():
                    return
            except Exception as e:
                logging.warning(f"Data fetch attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise RuntimeError("Initial data fetch failed after retries") from e
                time.sleep(2 ** attempt)  # Exponential backoff

    def _fetch_live_data(self) -> bool:
        """Fetch live data from APIs with comprehensive error handling"""
        try:
            # Football API
            football_response = requests.get(
                f"{FOOTBALL_API_URL}/matches",
                headers={"X-Auth-Token": FOOTBALL_API_KEY},
                params={"status": "LIVE", "limit": 20},
                timeout=10
            )
            football_response.raise_for_status()
            self.match_data = football_response.json().get("matches", [])
            
            # Live Odds API
            odds_response = requests.get(
                LIVE_ODDS_API_URL,
                params={"apiKey": LIVE_ODDS_API_KEY, "regions": "eu"},
                timeout=10
            )
            odds_response.raise_for_status()
            self.odds_data = {item['id']: item for item in odds_response.json()}
            
            self.last_update = time.time()
            self._cache_data()
            return True
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {str(e)}")
            raise RuntimeError("Live data fetch failed") from e
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Data parsing failed: {str(e)}")
            raise RuntimeError("Data parsing failed") from e

    def _cache_data(self) -> None:
        """Cache data to database with error handling"""
        try:
            cursor = self.db_conn.cursor()
            for match in self.match_data:
                odds = self.odds_data.get(str(match['id']), {})
                cursor.execute("""
                    INSERT OR REPLACE INTO matches 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    match['id'],
                    match.get('utcDate', ''),
                    match['homeTeam']['name'],
                    match['awayTeam']['name'],
                    match['score']['fullTime']['home'] if 'score' in match else None,
                    match['score']['fullTime']['away'] if 'score' in match else None,
                    odds.get('home_odds'),
                    odds.get('draw_odds'),
                    odds.get('away_odds')
                ))
            self.db_conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Database cache failed: {str(e)}")
            self.db_conn.rollback()

    def simulate_match(self, match: Dict, n_simulations: int = 1000) -> Dict:
        """Run match simulations with comprehensive error handling"""
        try:
            # Get base stats with fallbacks
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']
            
            home_attack = self._get_team_stat(home_team, 'goals_scored', 1.5)
            away_attack = self._get_team_stat(away_team, 'goals_scored', 1.2)
            home_defense = self._get_team_stat(home_team, 'goals_conceded', 1.1)
            away_defense = self._get_team_stat(away_team, 'goals_conceded', 1.3)
            
            # Adjust based on odds if available
            match_odds = self._get_match_odds(home_team, away_team)
            if match_odds:
                home_attack *= match_odds.get('home_adj', 1)
                away_attack *= match_odds.get('away_adj', 1)
            
            # Run simulations
            results = {
                'home_wins': 0,
                'draws': 0,
                'away_wins': 0,
                'goals': [],
                'btts': 0,
                'upset': False,
                'error': None
            }
            
            for _ in range(n_simulations):
                try:
                    home_goals = np.random.poisson(home_attack * (1/away_defense))
