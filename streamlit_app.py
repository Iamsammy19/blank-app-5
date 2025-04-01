import streamlit as st
import sqlite3
import os
from pathlib import Path
import tempfile
import logging
import pandas as pd
import numpy as np
from scipy.stats import poisson
from typing import Dict, Optional, List, Tuple
import time

# Configure logging
logging.basicConfig(
    filename='football_predictor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FootballPredictor:
    def __init__(self):
        """Initialize with robust error handling"""
        try:
            self.is_cloud = os.getenv('IS_STREAMLIT_CLOUD', False)
            self.data_limit = 10_000 if self.is_cloud else 50_000
            self.matches_df = pd.DataFrame()
            self.team_mapping: Dict[str, int] = {}
            self.model = None
            
            # Initialize with retry logic
            self._initialize_with_retry(max_retries=3)
            
        except Exception as e:
            logging.critical(f"Initialization failed: {str(e)}", exc_info=True)
            raise RuntimeError("System initialization failed") from e

    def _initialize_with_retry(self, max_retries: int = 3) -> None:
        """Initialize components with retry logic"""
        for attempt in range(max_retries):
            try:
                self._init_db()
                self._load_data()
                self._load_model()
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)
                continue

    def _init_db(self) -> None:
        """Initialize database with multiple fallback options"""
        try:
            # Try primary database location
            self.DB_PATH = Path(tempfile.gettempdir()) / "football_predictor.db"
            self.conn = sqlite3.connect(str(self.DB_PATH), timeout=10)
            self.cursor = self.conn.cursor()
            
            # Create tables if they don't exist
            self._create_tables()
            
            # Verify we have data
            if self._is_database_empty():
                self._load_sample_data()
                
        except sqlite3.Error as e:
            logging.error(f"Database error: {str(e)}")
            raise RuntimeError("Database initialization failed") from e

    def _create_tables(self) -> None:
        """Create required tables with proper schema"""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY,
                date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_goals INTEGER CHECK(home_goals >= 0),
                away_goals INTEGER CHECK(away_goals >= 0),
                CHECK(home_team != away_team)
            )
        """)
        self.conn.commit()

    def _is_database_empty(self) -> bool:
        """Check if database has no matches"""
        self.cursor.execute("SELECT COUNT(*) FROM matches")
        return self.cursor.fetchone()[0] == 0

    def _load_sample_data(self) -> None:
        """Load sample match data"""
        sample_matches = [
            ('2023-01-01', 'Arsenal', 'Chelsea', 2, 1),
            ('2023-01-02', 'Man City', 'Liverpool', 1, 1),
            ('2023-01-03', 'Tottenham', 'Man United', 0, 2)
        ]
        try:
            self.cursor.executemany(
                "INSERT INTO matches (date, home_team, away_team, home_goals, away_goals) VALUES (?, ?, ?, ?, ?)",
                sample_matches
            )
            self.conn.commit()
            logging.info("Loaded sample match data")
        except sqlite3.Error as e:
            logging.error(f"Failed to load sample data: {str(e)}")
            raise

    def _load_data(self) -> None:
        """Load match data with validation"""
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
                logging.warning("No match data found")
                raise ValueError("No match data available")
                
            # Create team mapping
            all_teams = pd.concat([
                self.matches_df['home_team'], 
                self.matches_df['away_team']
            ]).unique()
            self.team_mapping = {team: idx for idx, team in enumerate(all_teams)}
            
        except Exception as e:
            logging.error(f"Data loading failed: {str(e)}")
            raise

    def _load_model(self) -> None:
        """Initialize prediction model with fallback"""
        try:
            # In a real app, you would load your trained model here
            # For this example, we'll use a simple fallback model
            self.model = FallbackModel(self.matches_df)
            logging.info("Initialized fallback prediction model")
        except Exception as e:
            logging.error(f"Model initialization failed: {str(e)}")
            raise

class FallbackModel:
    """Simple fallback prediction model"""
    def __init__(self, matches_df: pd.DataFrame):
        self.matches_df = matches_df
        self.home_avg = matches_df['home_goals'].mean() if not matches_df.empty else 1.5
        self.away_avg = matches_df['away_goals'].mean() if not matches_df.empty else 1.0

    def predict(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """Predict using team averages"""
        try:
            home_avg = self.matches_df[
                self.matches_df['home_team'] == home_team
            ]['home_goals'].mean()
            
            away_avg = self.matches_df[
                self.matches_df['away_team'] == away_team
            ]['away_goals'].mean()
            
            return (
                home_avg if not np.isnan(home_avg) else self.home_avg,
                away_avg if not np.isnan(away_avg) else self.away_avg
            )
        except Exception:
            return (self.home_avg, self.away_avg)

def main():
    """Main application entry point with error handling"""
    st.set_page_config(
        page_title="Football Predictor",
        page_icon="⚽",
        layout="wide"
    )
    
    try:
        # Initialize with loading indicator
        with st.spinner("Initializing system..."):
            predictor = FootballPredictor()
        
        show_ui(predictor)
        
    except RuntimeError as e:
        st.error("""
            ⚠️ System initialization failed.  
            Please refresh the page or try again later.
            """)
        logging.critical(f"Application failed to start: {str(e)}")
    except Exception as e:
        st.error("An unexpected error occurred")
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)

def show_ui(predictor: FootballPredictor) -> None:
    """Display the main user interface"""
    st.title("⚽ Football Predictor")
    
    if predictor.matches_df.empty:
        st.warning("Using limited sample data - predictions may be less accurate")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox(
            "Home Team",
            sorted(predictor.team_mapping.keys())
        )
    
    with col2:
        away_team = st.selectbox(
            "Away Team",
            sorted(predictor.team_mapping.keys()),
            index=1 if len(predictor.team_mapping) > 1 else 0
        )
    
    if st.button("Predict Match"):
        with st.spinner("Calculating prediction..."):
            result = predict_match(predictor, home_team, away_team)
            if result:
                display_result(result)

def predict_match(predictor: FootballPredictor, home_team: str, away_team: str) -> Optional[Dict]:
    """Handle match prediction with error handling"""
    try:
        # Validate teams
        if home_team not in predictor.team_mapping:
            raise ValueError(f"Unknown home team: {home_team}")
        if away_team not in predictor.team_mapping:
            raise ValueError(f"Unknown away team: {away_team}")
        if home_team == away_team:
            raise ValueError("A team cannot play itself")
        
        # Get prediction
        home_goals, away_goals = predictor.model.predict(home_team, away_team)
        
        # Calculate probabilities
        home_probs = poisson.pmf(np.arange(6), home_goals)
        away_probs = poisson.pmf(np.arange(6), away_goals)
        score_matrix = np.outer(home_probs, away_probs)
        
        # Normalize probabilities
        score_matrix /= score_matrix.sum()
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_win_prob': round(np.sum(np.tril(score_matrix, -1)) * 100, 1),
            'draw_prob': round(np.sum(np.diag(score_matrix)) * 100, 1),
            'away_win_prob': round(np.sum(np.triu(score_matrix, 1)) * 100, 1),
            'expected_home_goals': round(home_goals, 1),
            'expected_away_goals': round(away_goals, 1)
        }
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        logging.error(f"Prediction error: {str(e)}", exc_info=True)
        return None

def display_result(result: Dict) -> None:
    """Display prediction results"""
    st.subheader(f"Prediction: {result['home_team']} vs {result['away_team']}")
    
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
    
    st.write(f"**Expected Score:** {result['expected_home_goals']} - {result['expected_away_goals']}")

if __name__ == "__main__":
    main()
    
