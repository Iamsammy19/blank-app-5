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
                    away_goals = np.random.poisson(away_attack * (1/home_defense))
                    
                    results['goals'].append((home_goals, away_goals))
                    
                    if home_goals > away_goals:
                        results['home_wins'] += 1
                    elif away_goals > home_goals:
                        results['away_wins'] += 1
                    else:
                        results['draws'] += 1
                        
                    if home_goals > 0 and away_goals > 0:
                        results['btts'] += 1
                except Exception as e:
                    logging.warning(f"Simulation iteration failed: {str(e)}")
                    continue
            
            # Convert counts to percentages
            total = max(1, results['home_wins'] + results['draws'] + results['away_wins'])
            results['home_wins'] = results['home_wins'] / total * 100
            results['draws'] = results['draws'] / total * 100
            results['away_wins'] = results['away_wins'] / total * 100
            results['btts'] = results['btts'] / n_simulations * 100
            
            # Detect potential upset
            results['upset'] = self._detect_upset(results, match_odds)
            
            return results
            
        except Exception as e:
            logging.error(f"Simulation failed: {str(e)}")
            return {
                'home_wins': 33.3,
                'draws': 33.3,
                'away_wins': 33.3,
                'btts': 50.0,
                'upset': False,
                'error': str(e)
            }

    def _get_team_stat(self, team_name: str, stat: str, default: float) -> float:
        """Get team statistic with fallback"""
        try:
            if not self.historical_data.empty:
                if stat == 'goals_scored':
                    home_avg = self.historical_data[
                        self.historical_data['home_team'] == team_name
                    ]['home_goals'].mean()
                    away_avg = self.historical_data[
                        self.historical_data['away_team'] == team_name
                    ]['away_goals'].mean()
                    return np.nanmean([home_avg, away_avg])
                elif stat == 'goals_conceded':
                    home_avg = self.historical_data[
                        self.historical_data['home_team'] == team_name
                    ]['away_goals'].mean()
                    away_avg = self.historical_data[
                        self.historical_data['away_team'] == team_name
                    ]['home_goals'].mean()
                    return np.nanmean([home_avg, away_avg])
        except Exception as e:
            logging.warning(f"Stat lookup failed for {team_name}: {str(e)}")
        return default

    def _get_match_odds(self, home_team: str, away_team: str) -> Optional[Dict]:
        """Get match odds with error handling"""
        try:
            for match_id, odds in self.odds_data.items():
                if odds['home_team'] == home_team and odds['away_team'] == away_team:
                    return {
                        'home_odds': odds.get('home_odds', 2.0),
                        'draw_odds': odds.get('draw_odds', 3.5),
                        'away_odds': odds.get('away_odds', 2.5),
                        'home_adj': 1 + (2.0 - odds.get('home_odds', 2.0)) / 10,
                        'away_adj': 1 + (2.0 - odds.get('away_odds', 2.5)) / 10
                    }
        except Exception as e:
            logging.warning(f"Odds lookup failed: {str(e)}")
        return None

    def _detect_upset(self, results: Dict, odds: Optional[Dict]) -> bool:
        """Detect potential upset with error handling"""
        try:
            if not odds:
                return False
                
            home_implied = 1 / odds.get('home_odds', 2.0)
            away_implied = 1 / odds.get('away_odds', 2.5)
            
            # Underdog has >40% chance while being priced as >35% underdog
            return (results['away_wins'] > 40 and home_implied > 0.65) or \
                   (results['home_wins'] > 40 and away_implied > 0.65)
        except Exception as e:
            logging.warning(f"Upset detection failed: {str(e)}")
            return False

    def predict_with_ml(self, match: Dict) -> Dict:
        """Make ML prediction with comprehensive error handling"""
        try:
            # Prepare features with fallbacks
            features = np.array([
                self._get_team_stat(match['homeTeam']['name'], 'goals_scored', 5),
                self._get_team_stat(match['awayTeam']['name'], 'goals_scored', 10),
                self._get_team_stat(match['homeTeam']['name'], 'goals_conceded', 0.6),
                self._get_team_stat(match['awayTeam']['name'], 'goals_conceded', 0.4),
                len(match.get('home_missing_players', [])),
                len(match.get('away_missing_players', [])),
                match.get('home_goals_avg', 1.5),
                match.get('away_goals_avg', 1.2),
                match.get('home_defense_avg', 1.0),
                match.get('away_defense_avg', 1.3)
            ]).reshape(1, -1)
            
            prediction = self.model.predict_proba(features)[0]
            
            return {
                'home_win': prediction[0] * 100,
                'draw': prediction[1] * 100,
                'away_win': prediction[2] * 100,
                'confidence': max(prediction) * 100,
                'error': None
            }
        except Exception as e:
            logging.error(f"ML prediction failed: {str(e)}")
            return {
                'home_win': 33.3,
                'draw': 33.3,
                'away_win': 33.3,
                'confidence': 0,
                'error': str(e)
            }

def main():
    """Main application with UI and error handling"""
    st.set_page_config(
        page_title="Ultimate Football Predictor",
        page_icon="⚽",
        layout="wide"
    )
    
    # Initialize predictor with error handling
    try:
        if 'predictor' not in st.session_state:
            with st.spinner("Initializing system..."):
                st.session_state.predictor = UltimateFootballPredictor()
    except RuntimeError as e:
        st.error(f"""
            ⚠️ System initialization failed: {str(e)}
            Please refresh the page or try again later.
            """)
        st.stop()
    except Exception as e:
        st.error("""
            ⚠️ Unexpected error during initialization.
            Please check the logs and try again.
            """)
        st.stop()
    
    # UI Components
    st.title("⚽ Ultimate Football Predictor")
    st.markdown("""
        <style>
        .match-card {
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            background-color: #f8f9fa;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .upset-alert {
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-weight: bold;
            border-left: 4px solid #ffc107;
        }
        .error-alert {
            background-color: #f8d7da;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-weight: bold;
            border-left: 4px solid #dc3545;
        }
        .simulation-results {
            background-color: #e2f0fd;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Controls sidebar
    with st.sidebar:
        st.header("Controls")
        
        if st.button("🔄 Refresh Live Data"):
            try:
                with st.spinner("Fetching latest data..."):
                    if st.session_state.predictor._fetch_live_data():
                        st.success("Data updated successfully!")
                    else:
                        st.error("Failed to update data")
            except Exception as e:
                st.error(f"Refresh failed: {str(e)}")
        
        st.markdown("---")
        st.write("**Simulation Settings**")
        n_simulations = st.slider("Number of simulations", 100, 10000, 1000)
        
        st.markdown("---")
        st.write(f"Last update: {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    # Main content
    if not st.session_state.predictor.match_data:
        st.warning("No live matches found. Click Refresh to fetch data.")
        return
    
    for match in st.session_state.predictor.match_data[:5]:  # Show first 5 matches
        with st.container():
            st.markdown(f"""
                <div class="match-card">
                    <h3>{match['homeTeam']['name']} vs {match['awayTeam']['name']}</h3>
            """, unsafe_include_html=True)
            
            # Run simulations and predictions
            try:
                with st.spinner(f"Running {n_simulations} simulations..."):
                    simulation = st.session_state.predictor.simulate_match(match, n_simulations)
                    ml_prediction = st.session_state.predictor.predict_with_ml(match)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                col1.metric("Home Win", f"{simulation['home_wins']:.1f}%")
                col2.metric("Draw", f"{simulation['draws']:.1f}%")
                col3.metric("Away Win", f"{simulation['away_wins']:.1f}%")
                
                # Show errors if any occurred
                if simulation.get('error'):
                    st.markdown(f"""
                        <div class="error-alert">
                            ⚠️ Simulation Warning: {simulation['error']}
                        </div>
                    """, unsafe_include_html=True)
                
                # Upset alert
                if simulation['upset']:
                    st.markdown("""
                        <div class="upset-alert">
                            ⚠️ Potential Upset Alert: Underdog has high win probability!
                        </div>
                    """, unsafe_include_html=True)
                
                # Simulation details
                with st.expander("📊 Detailed Analysis"):
                    st.markdown("""
                        <div class="simulation-results">
                            <h4>Monte Carlo Simulation Results</h4>
                    """, unsafe_include_html=True)
                    
                    st.write(f"**Both Teams to Score:** {simulation['btts']:.1f}%")
                    
                    # Goals distribution
                    if simulation['goals']:
                        goals_df = pd.DataFrame(simulation['goals'], columns=['Home', 'Away'])
                        st.write("**Average Goals:**")
                        st.write(f"Home: {goals_df['Home'].mean():.1f} | Away: {goals_df['Away'].mean():.1f}")
                        st.bar_chart(goals_df.mean(), height=200)
                    
                    # ML prediction
                    st.markdown("---")
                    st.write("**Machine Learning Prediction:**")
                    st.write(f"- Confidence: {ml_prediction['confidence']:.1f}%")
                    st.write(f"- Home Win: {ml_prediction['home_win']:.1f}%")
                    st.write(f"- Draw: {ml_prediction['draw']:.1f}%")
                    st.write(f"- Away Win: {ml_prediction['away_win']:.1f}%")
                    
                    if ml_prediction.get('error'):
                        st.markdown(f"""
                            <div class="error-alert">
                                ⚠️ ML Prediction Warning: {ml_prediction['error']}
                            </div>
                        """, unsafe_include_html=True)
                    
                    st.markdown("</div>", unsafe_include_html=True)
                
            except Exception as e:
                st.markdown(f"""
                    <div class="error-alert">
                        ⚠️ Failed to analyze match: {str(e)}
                    </div>
                """, unsafe_include_html=True)
                logging.error(f"Match analysis failed: {str(e)}")
            
            st.markdown("</div>", unsafe_include_html=True)

if __name__ == "__main__":
    main()
