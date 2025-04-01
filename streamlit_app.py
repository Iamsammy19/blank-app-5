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
import time
import threading
from typing import Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    filename='football_predictor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FootballPredictor:
    def __init__(self):
        self.is_cloud = os.getenv('IS_STREAMLIT_CLOUD', False)
        self.data_limit = 10_000 if self.is_cloud else 50_000
        self.matches_df = pd.DataFrame()
        self.team_mapping: Dict[str, int] = {}
        self.model = None
        
        self._init_db()
        self._load_data()
        self._load_model()
        self._verify_environment()
        
        if self.is_cloud:
            self._start_memory_watchdog()

    def _init_db(self) -> None:
        """Initialize and validate the database connection."""
        self.DB_PATH = Path(tempfile.gettempdir()) / "football_predictor.db"
        
        try:
            # Retry logic for database connection
            for attempt in range(3):
                try:
                    self.conn = sqlite3.connect(
                        f"file:{self.DB_PATH}?mode=rwc", 
                        uri=True,
                        timeout=10
                    )
                    self.cursor = self.conn.cursor()
                    break
                except sqlite3.OperationalError as e:
                    if attempt == 2:
                        raise
                    time.sleep(1)
            
            # Schema initialization with error recovery
            with self.conn:
                self.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS matches (
                        id INTEGER PRIMARY KEY,
                        date TEXT NOT NULL,
                        home_team TEXT NOT NULL,
                        away_team TEXT NOT NULL,
                        home_goals INTEGER CHECK(home_goals >= 0),
                        away_goals INTEGER CHECK(away_goals >= 0),
                        CHECK(home_team != away_team)
                """)
                
                # Add sample data if empty
                if self.cursor.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 0:
                    self._load_sample_matches()
                    
        except Exception as e:
            logging.critical(f"Database initialization failed: {str(e)}")
            st.error("Critical system error - please try again later")
            raise SystemExit(1)

    def _load_sample_matches(self) -> None:
        """Load sample matches into empty database."""
        sample_data = [
            ('2023-01-01', 'Arsenal', 'Chelsea', 2, 1),
            ('2023-01-02', 'Man City', 'Liverpool', 1, 1),
            ('2023-01-03', 'Tottenham', 'Man United', 0, 2)
        ]
        try:
            self.cursor.executemany(
                """INSERT INTO matches 
                (date, home_team, away_team, home_goals, away_goals)
                VALUES (?, ?, ?, ?, ?)""",
                sample_data
            )
            logging.info("Loaded sample match data")
        except Exception as e:
            logging.error(f"Failed to load sample data: {str(e)}")

    def _load_data(self) -> bool:
        """Load and validate match data from database."""
        try:
            query = """
                SELECT date, home_team, away_team, home_goals, away_goals 
                FROM matches 
                ORDER BY date DESC 
                LIMIT ?
            """
            self.matches_df = pd.read_sql_query(
                query, 
                self.conn, 
                params=(self.data_limit,),
                parse_dates=['date']
            )
            
            if self.matches_df.empty:
                logging.warning("No match data found in database")
                return False
                
            # Data validation
            self.matches_df.dropna(inplace=True)
            self.matches_df = self.matches_df[
                (self.matches_df['home_goals'] >= 0) & 
                (self.matches_df['away_goals'] >= 0)
            ]
            
            # Create team mapping
            all_teams = pd.concat([
                self.matches_df['home_team'], 
                self.matches_df['away_team']
            ]).unique()
            self.team_mapping = {team: idx for idx, team in enumerate(all_teams)}
            
            return True
            
        except Exception as e:
            logging.error(f"Data loading failed: {str(e)}")
            st.error("Failed to load match data")
            return False

    def _load_model(self) -> None:
        """Load or create prediction model with validation."""
        self.MODEL_PATH = Path(tempfile.gettempdir()) / "xgboost_model.pkl"
        
        try:
            if self.MODEL_PATH.exists():
                # Model validation
                if self.MODEL_PATH.stat().st_size < 100:
                    raise ValueError("Model file too small")
                    
                with open(self.MODEL_PATH, 'rb') as f:
                    self.model = joblib.load(f)
                    
                if not hasattr(self.model, 'predict'):
                    raise AttributeError("Invalid model format")
                    
                logging.info("Model loaded successfully")
            else:
                self._create_fallback_model()
                
        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            self._create_fallback_model()

    def _create_fallback_model(self) -> None:
        """Create a fallback prediction model."""
        logging.warning("Creating fallback prediction model")
        
        class FallbackModel:
            def __init__(self, matches_df: pd.DataFrame):
                self.data = matches_df
                self.home_avg = matches_df['home_goals'].mean()
                self.away_avg = matches_df['away_goals'].mean()
                
            def predict(self, home_team: str, away_team: str) -> Tuple[float, float]:
                """Predict using team-specific averages or fallback to league averages."""
                try:
                    home_avg = self.data[
                        self.data['home_team'] == home_team
                    ]['home_goals'].mean()
                    
                    away_avg = self.data[
                        self.data['away_team'] == away_team
                    ]['away_goals'].mean()
                    
                    return (
                        home_avg if not np.isnan(home_avg) else self.home_avg,
                        away_avg if not np.isnan(away_avg) else self.away_avg
                    )
                except:
                    return (self.home_avg, self.away_avg)
                    
        self.model = FallbackModel(self.matches_df)

    def _verify_environment(self) -> None:
        """Check system resources and constraints."""
        try:
            # Memory check
            test_array = np.zeros((1000, 1000))  # ~8MB
            del test_array
            
            # Cloud-specific checks
            if self.is_cloud and not Path('/tmp').exists():
                raise RuntimeError("Invalid cloud environment")
                
        except MemoryError:
            logging.warning("Memory constraints detected")
            self.data_limit = 5_000
        except Exception as e:
            logging.warning(f"Environment verification failed: {str(e)}")

    def _start_memory_watchdog(self) -> None:
        """Monitor memory usage in cloud environment."""
        def watchdog():
            while True:
                try:
                    mem = psutil.virtual_memory()
                    if mem.percent > 90:
                        logging.critical("Memory limit exceeded - exiting")
                        os._exit(1)
                    time.sleep(30)
                except:
                    pass
                    
        threading.Thread(target=watchdog, daemon=True).start()

    def _validate_teams(self, home_team: str, away_team: str) -> bool:
        """Validate team inputs before prediction."""
        if not all(isinstance(t, str) for t in [home_team, away_team]):
            raise ValueError("Team names must be strings")
        if home_team not in self.team_mapping:
            raise ValueError(f"Unknown home team: {home_
