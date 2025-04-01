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
from typing import Dict, Tuple, Optional, List
import psutil

# Configure logging
logging.basicConfig(
    filename='football_predictor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FootballPredictor:
    def __init__(self):
        """Initialize the football prediction system."""
        self.is_cloud = os.getenv('IS_STREAMLIT_CLOUD', False)
        self.data_limit = 10_000 if self.is_cloud else 50_000
        self.matches_df = pd.DataFrame()
        self.team_mapping: Dict[str, int] = {}
        self.model = None
        
        # Initialize components with error handling
        try:
            self._init_db()
            self._load_data()
            self._load_model()
            self._verify_environment()
            
            if self.is_cloud:
                self._start_memory_watchdog()
                
            logging.info("FootballPredictor initialized successfully")
        except Exception as e:
            logging.critical(f"Initialization failed: {str(e)}", exc_info=True)
            raise RuntimeError("System initialization failed") from e

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
            st.error("Critical database error - please try again later")
            raise

    def _load_sample_matches(self) -> None:
        """Load sample matches into empty database."""
        sample_data = [
            ('2023-01-01', 'Arsenal', 'Chelsea', 2, 1),
            ('2023-01-02', 'Man City', 'Liverpool', 1, 1),
            ('2023-01-03', 'Tottenham', 'Man United', 0, 2),
            ('2023-01-04', 'Newcastle', 'Aston Villa', 3, 0),
            ('2023-01-05', 'Everton', 'Southampton', 1, 1)
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
                st.warning("Using limited sample data - predictions may be less accurate")
                return False
                
            # Data validation and cleaning
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
            st.error("Failed to load match data - using fallback mode")
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
                self.home_avg = matches_df['home_goals'].mean() if not matches_df.empty else 1.5
                self.away_avg = matches_df['away_goals'].mean() if not matches_df.empty else 1.0
                
            def predict(self, home_team: str, away_team: str) -> Tuple[float, float]:
                """Predict using team-specific averages or fallback to league averages."""
                try:
                    home_data = self.data[self.data['home_team'] == home_team]
                    home_avg = home_data['home_goals'].mean() if not home_data.empty else np.nan
                    
                    away_data = self.data[self.data['away_team'] == away_team]
                    away_avg = away_data['away_goals'].mean() if not away_data.empty else np.nan
                    
                    return (
                        home_avg if not np.isnan(home_avg) else self.home_avg,
                        away_avg if not np.isnan(away_avg) else self.away_avg
                    )
                except Exception:
                    return (self.home_avg, self.away_avg)
                    
        self.model = FallbackModel(self.matches_df)
        logging.info("Fallback model initialized")

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
            if not self.matches_df.empty:
                self.matches_df = self.matches_df.sample(min(5000, len(self.matches_df)))
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
                except Exception:
                    pass
                    
        threading.Thread(target=watchdog, daemon=True).start()

    def get_team_list(self) -> List[str]:
        """Return sorted list of available teams."""
        return sorted(self.team_mapping.keys())

    def _validate_teams(self, home_team: str, away_team: str) -> bool:
        """Validate team inputs before prediction."""
        if not all(isinstance(t, str) for t in [home_team, away_team]):
            raise ValueError("Team names must be strings")
        if home_team not in self.team_mapping:
            raise ValueError(f"Unknown home team: {home_team}")
        if away_team not in self.team_mapping:
            raise ValueError(f"Unknown away team: {away_team}")
        if home_team == away_team:
            raise ValueError("Teams cannot play themselves")
        return True

    def predict_match(self, home_team: str, away_team: str) -> Optional[Dict]:
        """Generate match prediction with probabilities."""
        try:
            self._validate_teams(home_team, away_team)
            
            # Get prediction with bounds checking
            home_goals, away_goals = self.model.predict(home_team, away_team)
            home_goals = max(0, min(float(home_goals), 10))
            away_goals = max(0, min(float(away_goals), 10))
            
            # Calculate Poisson probabilities
            home_probs = poisson.pmf(np.arange(6), home_goals)
            away_probs = poisson.pmf(np.arange(6), away_goals)
            score_matrix = np.outer(home_probs, away_probs)
            
            # Normalize and calculate outcomes
            score_matrix /= score_matrix.sum()
            home_win = np.sum(np.tril(score_matrix, -1))
            draw = np.sum(np.diag(score_matrix))
            away_win = np.sum(np.triu(score_matrix, 1))
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'expected_home_goals': round(home_goals, 2),
                'expected_away_goals': round(away_goals, 2),
                'home_win_prob': round(home_win * 100, 1),
                'draw_prob': round(draw * 100, 1),
                'away_win_prob': round(away_win * 100, 1),
                'score_probs': score_matrix.round(4)
            }
            
        except ValueError as e:
            logging.warning(f"Prediction input error: {str(e)}")
            st.error(f"Invalid prediction request: {str(e)}")
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}", exc_info=True)
            st.error("Prediction service unavailable - try again later")
        return None

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Football Predictor Pro",
        page_icon="⚽",
        layout="wide"
    )
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        try:
            st.session_state.predictor = FootballPredictor()
        except RuntimeError as e:
            st.error("System initialization failed. Please refresh the page.")
            st.stop()
    
    display_ui(st.session_state.predictor)

def display_ui(predictor: FootballPredictor) -> None:
    """Render the application user interface."""
    st.title("⚽ Football Predictor Pro")
    
    with st.sidebar:
        st.header("System Status")
        st.write(f"Teams loaded: {len(predictor.team_mapping)}")
        st.write(f"Matches loaded: {len(predictor.matches_df)}")
        st.write(f"Model type: {type(predictor.model).__name__}")
        
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    st.header("Match Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox(
            "Select Home Team",
            predictor.get_team_list(),
            index=0
        )
    
    with col2:
        away_team = st.selectbox(
            "Select Away Team",
            predictor.get_team_list(),
            index=1 if len(predictor.get_team_list()) > 1 else 0
        )
    
    if st.button("Predict Match Outcome", type="primary"):
        with st.spinner("Calculating prediction..."):
            result = predictor.predict_match(home_team, away_team)
            display_results(result)

def display_results(result: Optional[Dict]) -> None:
    """Display prediction results in the UI."""
    if result is None:
        return
        
    st.subheader(f"Prediction: {result['home_team']} vs {result['away_team']}")
    
    # Outcome probabilities
    cols = st.columns(3)
    cols[0].metric(
        f"{result['home_team']} Win", 
        f"{result['home_win_prob']}%"
    )
    cols[1].metric("Draw", f"{result['draw_prob']}%")
    cols[2].metric(
        f"{result['away_team']} Win", 
        f"{result['away_win_prob']}%"
    )
    
    # Expected score
    st.markdown("### Expected Score")
    st.write(
        f"**{result['expected_home_goals']:.1f}** - **{result['expected_away_goals']:.1f}**"
    )
    
    # Score probability matrix
    st.markdown("### Score Probabilities")
    prob_df = pd.DataFrame(
        result['score_probs'][:5, :5] * 100,
        columns=[f"Away {i}" for i in range(5)],
        index=[f"Home {i}" for i in range(5)]
    )
    st.dataframe(
        prob_df.style.format("{:.1f}%")
        .background_gradient(cmap='Blues', axis=None)
    )
    
    # Match history placeholder
    st.markdown("### Recent Similar Matches")
    st.info("Match history analysis coming soon!")

if __name__ == "__main__":
    main()
