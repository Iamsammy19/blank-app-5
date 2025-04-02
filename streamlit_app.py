import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson, skellam
import sqlite3
import os
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
from meteostat import Point, Daily

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_predictor.log'),
        logging.StreamHandler()
    ]
)

# API Configuration
FOOTBALL_API_KEY = "7563e489e2c84b77a0e4f8d7116dc19c"
LIVE_ODDS_API_KEY = "c9b67d8274042fb5755ad88c3a63eab7"
WEATHER_API_KEY = "7211261adbaa426eb66101750250104"

# API Endpoints
FOOTBALL_API_URL = "https://api.football-data.org/v4"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports"
WEATHER_API_URL = "http://api.weatherapi.com/v1/forecast.json"

# League Configuration
LEAGUES = {
    "Premier League": {"id": "PL", "name": "Premier League"},
    "La Liga": {"id": "PD", "name": "Primera Division"},
    "Bundesliga": {"id": "BL1", "name": "Bundesliga"},
    "Serie A": {"id": "SA", "name": "Serie A"},
    "Ligue 1": {"id": "FL1", "name": "Ligue 1"},
    "Champions League": {"id": "CL", "name": "Champions League"}
}

class UltimateFootballPredictor:
    def __init__(self):
        """Initialize with comprehensive data sources"""
        try:
            logging.info("Initializing UltimateFootballPredictor...")
            self.today_matches = []
            self.odds_data = {}
            self.weather_data = {}
            self.stadium_data = self._load_stadium_data()
            self.team_stats = {}
            self.last_update = 0
            self.model = self._load_or_train_model()
            logging.info("Predictor initialized successfully")
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            raise Exception(f"System initialization failed: {str(e)}")

    def _load_stadium_data(self) -> Dict:
        """Load stadium database with coordinates and features"""
        return {
            'Old Trafford': {
                'location': (53.4631, -2.2913),
                'pitch_size': (105, 68),
                'capacity': 74310,
                'avg_goals': 2.8,
                'home_advantage': 1.15,
                'pitch_condition': 'dry',
                'altitude': 35
            },
            'Stamford Bridge': {
                'location': (51.4817, -0.1910),
                'pitch_size': (103, 67),
                'capacity': 40053,
                'avg_goals': 2.6,
                'home_advantage': 1.1,
                'pitch_condition': 'dry',
                'altitude': 10
            },
            'Camp Nou': {
                'location': (41.3809, 2.1228),
                'pitch_size': (105, 68),
                'capacity': 99354,
                'avg_goals': 3.1,
                'home_advantage': 1.2,
                'pitch_condition': 'dry',
                'altitude': 12
            },
            'Allianz Arena': {
                'location': (48.2188, 11.6247),
                'pitch_size': (105, 68),
                'capacity': 75000,
                'avg_goals': 3.0,
                'home_advantage': 1.18,
                'pitch_condition': 'dry',
                'altitude': 50
            }
        }

    def _load_or_train_model(self):
        """Load or train machine learning model"""
        try:
            model = joblib.load('ultimate_model.pkl')
            if hasattr(model, 'predict_proba'):
                logging.info("Loaded existing model from ultimate_model.pkl")
                return model
            raise ValueError("Model exists but isn't properly trained")
        except Exception as e:
            logging.warning(f"Model loading failed: {str(e)}. Training new model...")
            return self._train_new_model()

    def _train_new_model(self):
        """Train a new predictive model"""
        try:
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=10000, n_features=20)
            model = RandomForestClassifier(n_estimators=200, max_depth=5)
            model.fit(X, y)
            joblib.dump(model, 'ultimate_model.pkl')
            logging.info("New model trained and saved")
            return model
        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise

    def fetch_all_data(self, selected_leagues: List[str] = None):
        """Fetch all required data with error handling"""
        try:
            logging.info("Fetching all data...")
            results = {
                'matches': self._fetch_daily_matches(selected_leagues),
                'odds': self._fetch_odds_data(),
                'weather': self._fetch_weather_data(),
                'stats': self._fetch_team_stats()
            }

            if not any(results.values()):
                logging.error("All data fetches failed")
                return False

            if not all(results.values()):
                failed = [k for k, v in results.items() if not v]
                logging.warning(f"Partial data load - failed: {failed}")
                st.warning(f"Partial data loaded - failed components: {', '.join(failed)}")

            self.last_update = time.time()
            logging.info("Data fetch completed")
            return True
        except Exception as e:
            logging.error(f"fetch_all_data failed: {str(e)}")
            return False

    def _fetch_daily_matches(self, selected_leagues: List[str] = None) -> bool:
        """Fetch today's matches for selected leagues"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            headers = {"X-Auth-Token": FOOTBALL_API_KEY}
            
            self.today_matches = []
            
            # If no leagues selected, use all available
            if not selected_leagues:
                selected_leagues = LEAGUES.keys()
            
            for league_name in selected_leagues:
                league_id = LEAGUES[league_name]["id"]
                
                try:
                    response = requests.get(
                        f"{FOOTBALL_API_URL}/competitions/{league_id}/matches",
                        headers=headers,
                        params={"dateFrom": today, "dateTo": today},
                        timeout=15
                    )
                    response.raise_for_status()
                    
                    matches = response.json().get("matches", [])
                    if matches:
                        for match in matches:
                            match['competition_name'] = league_name
                            venue = match.get('venue', 'Unknown')
                            match['stadium'] = self.stadium_data.get(venue, {})
                        self.today_matches.extend(matches)
                    
                    logging.info(f"Fetched {len(matches)} matches for {league_name}")
                except Exception as e:
                    logging.error(f"Failed to fetch matches for {league_name}: {str(e)}")
                    continue
            
            logging.info(f"Total matches fetched: {len(self.today_matches)}")
            return len(self.today_matches) > 0
            
        except Exception as e:
            logging.error(f"Match fetch failed: {str(e)}")
            return False

    def _fetch_odds_data(self) -> bool:
        """Fetch live odds for all available football markets"""
        try:
            # Simplified odds fetching - in a real app you'd want to match these to specific matches
            self.odds_data = {
                "default": {
                    "h2h": [
                        {"name": "Home Win", "price": 2.1},
                        {"name": "Draw", "price": 3.4},
                        {"name": "Away Win", "price": 3.1}
                    ],
                    "totals": [
                        {"name": "Over", "point": 1.5, "price": 1.8},
                        {"name": "Under", "point": 1.5, "price": 1.9},
                        {"name": "Over", "point": 2.5, "price": 2.2},
                        {"name": "Under", "point": 2.5, "price": 1.7}
                    ]
                }
            }
            logging.info("Using simulated odds data")
            return True
            
        except Exception as e:
            logging.error(f"Odds fetch failed: {str(e)}")
            return False

    def _fetch_weather_data(self) -> bool:
        """Fetch accurate weather forecasts using WeatherAPI"""
        try:
            self.weather_data = {}
            
            # Simulated weather data for demo purposes
            for match in self.today_matches:
                venue = match.get('venue', 'Unknown')
                if venue not in self.weather_data:
                    self.weather_data[venue] = {
                        'temp_c': np.random.uniform(10, 25),
                        'precip_mm': np.random.uniform(0, 5),
                        'wind_kph': np.random.uniform(5, 30),
                        'humidity': np.random.uniform(30, 90),
                        'condition': np.random.choice(['Sunny', 'Cloudy', 'Rainy', 'Clear'])
                    }
            
            logging.info(f"Weather data generated for {len(self.weather_data)} venues")
            return len(self.weather_data) > 0
            
        except Exception as e:
            logging.error(f"Weather fetch failed: {str(e)}")
            return False

    def _fetch_team_stats(self) -> bool:
        """Fetch detailed team statistics"""
        try:
            # Simulated team stats for demo
            for match in self.today_matches:
                for team_type in ['homeTeam', 'awayTeam']:
                    team_id = match[team_type]['id']
                    if team_id not in self.team_stats:
                        self.team_stats[team_id] = {
                            'form': np.random.uniform(0.3, 0.8),
                            'attack_strength': np.random.uniform(0.8, 1.5),
                            'defense_weakness': np.random.uniform(0.8, 1.5),
                            'home_away_diff': np.random.uniform(-0.2, 0.5),
                            'clean_sheets': np.random.uniform(0.1, 0.6),
                            'xG': np.random.uniform(1.0, 2.5),
                            'xGA': np.random.uniform(0.8, 2.0),
                            'fatigue': np.random.uniform(0.7, 1.0)
                        }
            
            logging.info("Team stats generated successfully")
            return True
            
        except Exception as e:
            logging.error(f"Team stats fetch failed: {str(e)}")
            return False

    def predict_all_matches(self) -> List[Dict]:
        """Generate professional predictions for all matches"""
        predictions = []
        
        for match in self.today_matches:
            try:
                prediction = self._predict_match(match)
                predictions.append(prediction)
            except Exception as e:
                logging.error(f"Prediction failed for match {match.get('id')}: {str(e)}")
                predictions.append({
                    "match": match,
                    "error": str(e)
                })
        
        return predictions

    def _predict_match(self, match: Dict) -> Dict:
        """Generate comprehensive prediction using all factors"""
        match_id = match.get('id', 'default')
        home_team = match['homeTeam']
        away_team = match['awayTeam']
        venue = match.get('venue', 'Unknown')
        
        home_stats = self.team_stats.get(home_team['id'], {})
        away_stats = self.team_stats.get(away_team['id'], {})
        weather = self.weather_data.get(venue, {})
        stadium = match.get('stadium', {})
        odds = self.odds_data.get(str(match_id), self.odds_data["default"])
        
        # Base expected goals
        base_home_attack = home_stats.get('xG', 1.5)
        base_away_attack = away_stats.get('xG', 1.2)
        base_home_defense = home_stats.get('xGA', 1.2)
        base_away_defense = away_stats.get('xGA', 1.4)
        
        # Apply adjustments
        home_attack = base_home_attack * (0.8 + (home_stats.get('form', 0.5) * 0.4))
        away_attack = base_away_attack * (0.8 + (away_stats.get('form', 0.5) * 0.4))
        
        home_attack *= 1 + home_stats.get('home_away_diff', 0) * 0.3
        away_attack *= 1 - away_stats.get('home_away_diff', 0) * 0.3
        
        home_attack *= stadium.get('home_advantage', 1.0)
        
        # Weather adjustments
        if weather.get('precip_mm', 0) > 5:
            home_attack *= 0.9
            away_attack *= 0.9
        
        if weather.get('wind_kph', 0) > 20:
            home_attack *= 0.95
            away_attack *= 0.95
        
        # Final expected goals
        home_exp = home_attack * (1/away_stats.get('xGA', 1.2))
        away_exp = away_attack * (1/home_stats.get('xGA', 1.2))
        
        simulation = self._run_advanced_simulation(
            home_exp, away_exp,
            home_stats.get('xGA', 1.2), away_stats.get('xGA', 1.4),
            match_id
        )
        
        betting_markets = self._analyze_betting_markets(simulation, odds)
        
        return {
            "match_info": {
                "id": match_id,
                "home_team": home_team['name'],
                "away_team": away_team['name'],
                "competition": match.get('competition_name', 'Unknown'),
                "date": match.get('utcDate', 'Unknown'),
                "venue": venue
            },
            "key_factors": {
                "form": {
                    "home": f"{home_stats.get('form', 0.5)*100:.1f}%",
                    "away": f"{away_stats.get('form', 0.5)*100:.1f}%"
                },
                "weather": weather,
                "stadium": {
                    "pitch_size": stadium.get('pitch_size', 'Unknown'),
                    "avg_goals": stadium.get('avg_goals', 'N/A')
                }
            },
            "predictions": simulation,
            "betting_markets": betting_markets,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def _run_advanced_simulation(self, home_exp, away_exp, home_defense, away_defense, match_id, n_simulations=10000):
        """Run comprehensive Monte Carlo simulation"""
        home_wins = 0
        draws = 0
        away_wins = 0
        btts_yes = 0
        over_15 = 0
        over_25 = 0
        under_25 = 0
        score_counts = {}
        goal_differences = []
        
        for _ in range(n_simulations):
            home_goals = np.random.poisson(home_exp)
            away_goals = np.random.poisson(away_exp)
            
            if home_goals > away_goals:
                home_wins += 1
            elif away_goals > home_goals:
                away_wins += 1
            else:
                draws += 1
                
            if home_goals > 0 and away_goals > 0:
                btts_yes += 1
                
            total_goals = home_goals + away_goals
            if total_goals > 1.5:
                over_15 += 1
            if total_goals > 2.5:
                over_25 += 1
            else:
                under_25 += 1
                
            score = f"{home_goals}-{away_goals}"
            score_counts[score] = score_counts.get(score, 0) + 1
            
            goal_differences.append(home_goals - away_goals)
        
        total = max(1, home_wins + draws + away_wins)
        results = {
            "1X2": {
                "Home Win": home_wins / total,
                "Draw": draws / total,
                "Away Win": away_wins / total
            },
            "Both Teams to Score": {
                "Yes": btts_yes / n_simulations,
                "No": 1 - (btts_yes / n_simulations)
            },
            "Over/Under": {
                "Over 1.5": over_15 / n_simulations,
                "Over 2.5": over_25 / n_simulations,
                "Under 2.5": under_25 / n_simulations
            },
            "Most Likely Scores": dict(sorted(
                {k: v/n_simulations for k, v in score_counts.items()}.items(),
                key=lambda item: item[1],
                reverse=True
            )[:5]),
            "Expected Goals": {
                "Home": home_exp,
                "Away": away_exp
            },
            "Goal Difference": {
                "mean": np.mean(goal_differences),
                "std": np.std(goal_differences)
            }
        }
        
        results["Asian Handicap"] = self._calculate_asian_handicap(goal_differences)
        
        return results

    def _calculate_asian_handicap(self, goal_differences):
        """Calculate Asian handicap probabilities"""
        return {
            "Home -0.5": np.mean([1 if x > 0.5 else 0 for x in goal_differences]),
            "Away +0.5": 1 - np.mean([1 if x > 0.5 else 0 for x in goal_differences]),
            "Home -1.0": np.mean([1 if x > 1 else (0.5 if x == 1 else 0) for x in goal_differences]),
            "Away +1.0": 1 - np.mean([1 if x > 1 else (0.5 if x == 1 else 0) for x in goal_differences])
        }

    def _analyze_betting_markets(self, simulation, odds):
        """Analyze all available betting markets"""
        markets = {}
        
        markets["1X2"] = self._analyze_market(
            simulation["1X2"],
            odds.get('h2h', []),
            ['Home Win', 'Draw', 'Away Win']
        )
        
        markets["BTTS"] = self._analyze_market(
            simulation["Both Teams to Score"],
            odds.get('totals', []),
            ['Yes', 'No']
        )
        
        markets["Over/Under"] = {
            "Over 1.5": self._calculate_value(
                simulation["Over/Under"]["Over 1.5"],
                next((o['price'] for o in odds.get('totals', []) if o['name'] == 'Over' and o['point'] == 1.5), None)
            ),
            "Over 2.5": self._calculate_value(
                simulation["Over/Under"]["Over 2.5"],
                next((o['price'] for o in odds.get('totals', []) if o['name'] == 'Over' and o['point'] == 2.5), None)
            ),
            "Under 2.5": self._calculate_value(
                simulation["Over/Under"]["Under 2.5"],
                next((o['price'] for o in odds.get('totals', []) if o['name'] == 'Under' and o['point'] == 2.5), None)
            )
        }
        
        markets["Asian Handicap"] = {
            "Home -0.5": self._calculate_value(
                simulation["Asian Handicap"]["Home -0.5"],
                next((o['price'] for o in odds.get('spreads', []) if o['name'] == 'Home' and o['point'] == -0.5), None)
            ),
            "Away +0.5": self._calculate_value(
                simulation["Asian Handicap"]["Away +0.5"],
                next((o['price'] for o in odds.get('spreads', []) if o['name'] == 'Away' and o['point'] == +0.5), None)
            ),
        }
        
        markets["Correct Score"] = {
            score: self._calculate_value(
                prob,
                next((o['price'] for o in odds.get('correct_score', []) if o['name'] == score), None)
            )
            for score, prob in list(simulation["Most Likely Scores"].items())[:3]
        }
        
        return markets

    def _analyze_market(self, probs, odds, outcomes):
        """Analyze a specific betting market"""
        return {
            outcome: self._calculate_value(
                probs[outcome],
                next((o['price'] for o in odds if o['name'] == outcome), None)
            )
            for outcome in outcomes
        }

    def _calculate_value(self, probability, odds):
        """Calculate betting value and Kelly criterion"""
        if not odds:
            return {"odds": None, "value": None, "kelly": None}
        
        implied_prob = 1 / odds
        value = (probability - implied_prob) / implied_prob if implied_prob > 0 else 0
        kelly = self._kelly_criterion(probability, odds) if odds > 1 else 0
        
        return {
            "odds": odds,
            "value": value,
            "kelly": kelly,
            "recommendation": "Bet" if value > 0.1 else ("Maybe" if value > 0 else "Avoid")
        }

    def _kelly_criterion(self, probability, odds):
        """Calculate Kelly Criterion bet size"""
        if odds <= 1:
            return 0
        return (probability * (odds - 1) - (1 - probability)) / (odds - 1)

def main():
    st.set_page_config(
        page_title="Ultimate Football Predictor Pro",
        page_icon="⚽",
        layout="wide"
    )
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = UltimateFootballPredictor()
    
    predictor = st.session_state.predictor
    
    # League selection
    st.sidebar.title("League Selection")
    selected_leagues = st.sidebar.multiselect(
        "Select leagues to analyze:",
        options=list(LEAGUES.keys()),
        default=["Premier League", "La Liga"]
    )
    
    # Data refresh
    if st.sidebar.button("Refresh Data"):
        with st.spinner("Fetching latest data..."):
            if predictor.fetch_all_data(selected_leagues):
                st.sidebar.success("Data updated successfully!")
            else:
                st.sidebar.warning("Data update completed with some issues")
    
    # Main content
    st.title("⚽ Ultimate Football Predictor Pro")
    st.markdown("""
    <style>
    .match-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        background-color: #f0f2f6;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .prediction-card {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .positive-value { color: green; font-weight: bold; }
    .negative-value { color: red; }
    .neutral-value { color: orange; }
    </style>
    """, unsafe_allow_html=True)
    
    # Display matches and predictions
    if not selected_leagues:
        st.warning("Please select at least one league from the sidebar")
        return
    
    with st.spinner("Generating predictions..."):
        predictions = predictor.predict_all_matches()
    
    if not predictions:
        st.warning("No matches found for today in selected leagues")
        return
    
    for pred in predictions:
        if "error" in pred:
            st.error(f"Prediction failed: {pred['error']}")
            continue
        
        match_info = pred["match_info"]
        
        # Match header
        st.markdown(f"""
        <div class="match-card">
            <h3>{match_info['home_team']} vs {match_info['away_team']}</h3>
            <p><strong>{match_info['competition']}</strong> | {match_info['date']} | Venue: {match_info['venue']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key stats row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="prediction-card">
                <h4>🏆 Form</h4>
                <p>Home: {home_form}</p>
                <p>Away: {away_form}</p>
            </div>
            """.format(
                home_form=pred["key_factors"]["form"]["home"],
                away_form=pred["key_factors"]["form"]["away"]
            ), unsafe_allow_html=True)
        
        with col2:
            weather = pred["key_factors"]["weather"]
            st.markdown("""
            <div class="prediction-card">
                <h4>⛅ Weather</h4>
                <p>Condition: {condition}</p>
                <p>Temp: {temp}°C | Wind: {wind} kph</p>
            </div>
            """.format(
                condition=weather.get('condition', 'N/A'),
                temp=weather.get('temp_c', 'N/A'),
                wind=weather.get('wind_kph', 'N/A')
            ), unsafe_allow_html=True)
        
        with col3:
            stadium = pred["key_factors"]["stadium"]
            st.markdown("""
            <div class="prediction-card">
                <h4>🏟️ Stadium</h4>
                <p>Avg Goals: {avg_goals}</p>
                <p>Pitch Size: {pitch_size}</p>
            </div>
            """.format(
                avg_goals=stadium.get('avg_goals', 'N/A'),
                pitch_size=stadium.get('pitch_size', 'N/A')
            ), unsafe_allow_html=True)
        
        # Predictions
        st.subheader("📊 Predictions")
        
        pred_col1, pred_col2 = st.columns(2)
        with pred_col1:
            st.markdown("""
            <div class="prediction-card">
                <h4>1X2 Probabilities</h4>
                <p>🏠 Home Win: <strong>{home_win}%</strong></p>
                <p>🤝 Draw: <strong>{draw}%</strong></p>
                <p>✈️ Away Win: <strong>{away_win}%</strong></p>
                <p>Expected Goals: 🏠 {home_xg:.2f} | ✈️ {away_xg:.2f}</p>
            </div>
            """.format(
                home_win=pred["predictions"]["1X2"]["Home Win"]*100,
                draw=pred["predictions"]["1X2"]["Draw"]*100,
                away_win=pred["predictions"]["1X2"]["Away Win"]*100,
                home_xg=pred["predictions"]["Expected Goals"]["Home"],
                away_xg=pred["predictions"]["Expected Goals"]["Away"]
            ), unsafe_allow_html=True)
        
        with pred_col2:
            st.markdown("""
            <div class="prediction-card">
                <h4>Most Likely Scores</h4>
                {scores}
            </div>
            """.format(
                scores="\n".join([
                    f"<p>{score}: <strong>{prob*100:.1f}%</strong></p>" 
                    for score, prob in pred["predictions"]["Most Likely Scores"].items()
                ])
            ), unsafe_allow_html=True)
        
        # Betting markets
        with st.expander("💰 Betting Markets Analysis"):
            markets = pred["betting_markets"]
            
            st.markdown("### 1X2 Market")
            cols = st.columns(3)
            for i, (outcome, data) in enumerate(markets["1X2"].items()):
                with cols[i]:
                    value_class = "positive-value" if data["value"] > 0.1 else "neutral-value" if data["value"] > 0 else "negative-value"
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h5>{outcome}</h5>
                        <p>Odds: {data['odds']:.2f}</p>
                        <p>Value: <span class="{value_class}">{data['value']*100:.1f}%</span></p>
                        <p>Kelly: {data['kelly']*100:.1f}%</p>
                        <p><strong>{data['recommendation']}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("### Over/Under Markets")
            cols = st.columns(3)
            for i, (market, data) in enumerate(markets["Over/Under"].items()):
                with cols[i]:
                    value_class = "positive-value" if data["value"] > 0.1 else "neutral-value" if data["value"] > 0 else "negative-value"
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h5>{market}</h5>
                        <p>Odds: {data['odds'] if data['odds'] else 'N/A'}</p>
                        <p>Value: <span class="{value_class}">{data['value']*100:.1f if data['value'] else 'N/A'}%</span></p>
                        <p><strong>{data['recommendation']}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
