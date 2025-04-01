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
                    # Add sample data
                    sample_matches = [
                        ('2023-01-01', 'Team A', 'Team B', 2, 1),
                        ('2023-01-02', 'Team C', 'Team D', 0, 0),
                        ('2023-01-03', 'Team E', 'Team F', 1, 3)
                    ]
                    cursor.executemany(
                        "INSERT INTO matches (date, home_team, away_team, home_goals, away_goals) VALUES (?, ?, ?, ?, ?)",
                        sample_matches
                    )
                    conn.commit()
                    conn.close()
                    logging.info(f"Fallback database created at {DB_PATH} with sample data")

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

    def load_data(self):
        """Load football match data from the SQLite database into a DataFrame."""
        try:
            # Query the matches table with a limit based on environment
            query = "SELECT * FROM matches LIMIT ?"
            self.matches_df = pd.read_sql_query(query, self.conn, params=(self.data_limit,))
            logging.info(f"Loaded {len(self.matches_df)} matches from database")
            
            if self.matches_df.empty:
                logging.warning("No data found in matches table")
                st.warning("No match data available in the database")
                return False
            
            # Optional: Basic data validation or preprocessing
            required_columns = ['date', 'home_team', 'away_team', 'home_goals', 'away_goals']
            if not all(col in self.matches_df.columns for col in required_columns):
                raise ValueError("Database missing required columns")
            
            # Create team mapping for consistent team IDs
            all_teams = pd.concat([self.matches_df['home_team'], self.matches_df['away_team']]).unique()
            self.team_mapping = {team: idx for idx, team in enumerate(all_teams)}
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            st.error(f"Data loading error: {str(e)}")
            raise

    def load_model(self):
        """Load the trained XGBoost model from file or create a fallback model."""
        try:
            if MODEL_PATH.exists():
                logging.info(f"Loading model from {MODEL_PATH}")
                # Verify the model file is valid
                if MODEL_PATH.stat().st_size < 100:  # Minimum expected size
                    raise ValueError("Model file appears corrupted")
                    
                with open(MODEL_PATH, 'rb') as f:
                    self.model = joblib.load(f)
                    
                # Verify the loaded model has the predict method
                if not hasattr(self.model, 'predict'):
                    raise AttributeError("Loaded model missing predict method")
                    
                logging.info("Model loaded and validated")
            else:
                logging.warning(f"No model found at {MODEL_PATH}")
                self._create_fallback_model()
                
        except (EOFError, ValueError) as e:
            logging.error(f"Model file corrupted: {e}")
            st.warning("Model file corrupted - using fallback")
            self._create_fallback_model()
        except Exception as e:
            logging.error(f"Unexpected error loading model: {e}")
            st.warning("Failed to load prediction model - using fallback")
            self._create_fallback_model()

    def _create_fallback_model(self):
        """Create a simple fallback model when no trained model is available."""
        logging.info("Creating fallback model")
        
        class FallbackModel:
            def __init__(self, matches_df):
                self.matches_df = matches_df
                
            def predict(self, home_team, away_team):
                """Predict match outcome using historical averages."""
                try:
                    if self.matches_df.empty:
                        return (1.5, 1.0)  # Default averages if no data
                    
                    # Calculate team-specific averages if available
                    home_avg = self.matches_df[self.matches_df['home_team'] == home_team]['home_goals'].mean()
                    away_avg = self.matches_df[self.matches_df['away_team'] == away_team]['away_goals'].mean()
                    
                    # Fall back to general averages if team-specific data not available
                    if np.isnan(home_avg):
                        home_avg = self.matches_df['home_goals'].mean()
                    if np.isnan(away_avg):
                        away_avg = self.matches_df['away_goals'].mean()
                        
                    return (home_avg, away_avg)
                    
                except Exception as e:
                    logging.error(f"Fallback model prediction error: {e}")
                    return (1.5, 1.0)  # Default averages if error occurs
                    
        self.model = FallbackModel(self.matches_df)
        logging.warning("Using fallback prediction model")

    def predict_match(self, home_team, away_team):
        """Predict the outcome of a match between two teams."""
        try:
            if not hasattr(self, 'model') or self.model is None:
                raise AttributeError("Prediction model not loaded")
                
            home_goals, away_goals = self.model.predict(home_team, away_team)
            
            # Convert to Poisson probabilities
