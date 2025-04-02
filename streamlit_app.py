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
INJURY_API_KEY = "73bacb5a360ea069e1f45c330358706c"  # Testing key - REMOVE FOR PRODUCTION

# API Endpoints
FOOTBALL_API_URL = "https://api.football-data.org/v4"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
WEATHER_API_URL = "http://api.weatherapi.com/v1/forecast.json"
INJURY_API_URL = "https://api.sportsdata.io/v3/soccer/scores/json/Injuries"

class UltimateFootballPredictor:
    def __init__(self):
        """Initialize with comprehensive data sources"""
        try:
            logging.info("Initializing UltimateFootballPredictor...")
            self.today_matches = []
            self.odds_data = {}
            self.injury_data = {}
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
                'home_advantage': 1.15
            },
            'Stamford Bridge': {
                'location': (51.4817, -0.1910),
                'pitch_size': (103, 67),
                'capacity': 40053,
                'avg_goals': 2.6,
                'home_advantage': 1.1
            },
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

    def fetch_all_data(self):
        """Fetch all required data with error handling"""
        try:
            logging.info("Fetching all data...")
            results = {
                'matches': self._fetch_daily_matches(),
                'odds': self._fetch_odds_data(),
                'injuries': self._fetch_injury_data(),
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

    def _fetch_daily_matches(self) -> bool:
        """Fetch today's matches with retry logic"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            headers = {"X-Auth-Token": FOOTBALL_API_KEY}
            
            response = requests.get(
                f"{FOOTBALL_API_URL}/matches",
                headers=headers,
                params={"dateFrom": today, "dateTo": today, "limit": 100},
                timeout=15
            )
            response.raise_for_status()
            
            self.today_matches = response.json().get("matches", [])
            
            if not self.today_matches:
                comp_response = requests.get(
                    f"{FOOTBALL_API_URL}/competitions/PL/matches",
                    headers=headers,
                    params={"dateFrom": today, "dateTo": today},
                    timeout=15
                )
                comp_response.raise_for_status()
                self.today_matches = comp_response.json().get("matches", [])
            
            for match in self.today_matches:
                venue = match.get('venue', 'Unknown')
                match['stadium'] = self.stadium_data.get(venue, {})
            
            logging.info(f"Fetched {len(self.today_matches)} matches")
            return len(self.today_matches) > 0
            
        except Exception as e:
            logging.error(f"Match fetch failed: {str(e)}")
            return False

    def _fetch_odds_data(self) -> bool:
        """Fetch live odds with proper error handling"""
        try:
            response = requests.get(
                ODDS_API_URL,
                params={"apiKey": LIVE_ODDS_API_KEY, "regions": "eu", "markets": "h2h,totals,spreads"},
                timeout=10
            )
            response.raise_for_status()
            
            self.odds_data = {
                item['id']: {
                    'h2h': item['bookmakers'][0]['markets'][0]['outcomes'],
                    'totals': item['bookmakers'][0]['markets'][1]['outcomes'],
                    'spreads': item['bookmakers'][0]['markets'][2]['outcomes']
                } for item in response.json()
            }
            logging.info("Odds data fetched successfully")
            return True
            
        except Exception as e:
            logging.error(f"Odds fetch failed: {str(e)}")
            return False

    def _fetch_injury_data(self) -> bool:
        """Fetch injury data from professional source"""
        try:
            headers = {
                "Ocp-Apim-Subscription-Key": INJURY_API_KEY,
                "Content-Type": "application/json"
            }
            
            logging.info(f"Attempting to fetch injury data from {INJURY_API_URL}")
            
            response = requests.get(
                INJURY_API_URL,
                headers=headers,
                timeout=15
            )
            response.raise_for_status()
            
            sample_data = response.json()[:2]
            logging.info(f"Injury API sample response: {sample_data}")
            
            injuries = response.json()
            self.injury_data = {}
            
            for injury in injuries:
                team_id = str(injury['TeamID'])
                if team_id not in self.injury_data:
                    self.injury_data[team_id] = []
                self.injury_data[team_id].append({
                    'player': injury.get('PlayerName', 'Unknown'),
                    'position': injury.get('Position', 'Unknown'),
                    'status': injury.get('Status', 'Unknown'),
                    'injury_type': injury.get('InjuryType', 'Not Specified')
                })
                
            logging.info("Injury data fetched successfully")
            return True
            
        except Exception as e:
            logging.error(f"Injury data fetch failed: {str(e)}")
            self.injury_data = {
                "1": [{
                    "player": "Test Player 1",
                    "position": "FW",
                    "status": "Doubtful",
                    "injury_type": "Hamstring"
                }],
                "2": [{
                    "player": "Test Player 2",
                    "position": "DF",
                    "status": "Out",
                    "injury_type": "Knee"
                }]
            }
            logging.info("Using fallback injury data")
            return False

    def _fetch_weather_data(self) -> bool:
        """Fetch accurate weather forecasts using WeatherAPI"""
        try:
            self.weather_data = {}
            
            for match in self.today_matches:
                venue = match.get('venue', 'Unknown')
                if venue in self.stadium_data and venue not in self.weather_data:
                    lat, lon = self.stadium_data[venue]['location']
                    
                    response = requests.get(
                        WEATHER_API_URL,
                        params={
                            "key": WEATHER_API_KEY,
                            "q": f"{lat},{lon}",
                            "days": 1,
                            "aqi": "no",
                            "alerts": "no"
                        },
                        timeout=10
                    )
                    response.raise_for_status()
                    
                    forecast = response.json()
                    hour = datetime.strptime(
                        match['utcDate'], '%Y-%m-%dT%H:%M:%SZ'
                    ).hour
                    
                    for hour_data in forecast['forecast']['forecastday'][0]['hour']:
                        if datetime.strptime(
                            hour_data['time'], '%Y-%m-%d %H:%M'
                        ).hour == hour:
                            self.weather_data[venue] = {
                                'temp_c': hour_data['temp_c'],
                                'precip_mm': hour_data['precip_mm'],
                                'wind_kph': hour_data['wind_kph'],
                                'humidity': hour_data['humidity'],
                                'condition': hour_data['condition']['text']
                            }
                            break
            
            logging.info(f"Weather data fetched for {len(self.weather_data)} venues")
            return len(self.weather_data) > 0
            
        except Exception as e:
            logging.error(f"Weather fetch failed: {str(e)}")
            return False

    def _fetch_team_stats(self) -> bool:
        """Fetch detailed team statistics"""
        try:
            headers = {"X-Auth-Token": FOOTBALL_API_KEY}
            date_from = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            for match in self.today_matches:
                for team_type in ['homeTeam', 'awayTeam']:
                    team_id = match[team_type]['id']
                    if team_id not in self.team_stats:
                        matches_response = requests.get(
                            f"{FOOTBALL_API_URL}/teams/{team_id}/matches",
                            headers=headers,
                            params={"dateFrom": date_from, "limit": 5},
                            timeout=10
                        )
                        matches_response.raise_for_status()
                        matches = matches_response.json().get('matches', [])
                        
                        stats = self._calculate_advanced_stats(matches, team_id)
                        self.team_stats[team_id] = stats
            
            logging.info("Team stats fetched successfully")
            return True
            
        except Exception as e:
            logging.error(f"Team stats fetch failed: {str(e)}")
            return False

    def _calculate_advanced_stats(self, matches: List, team_id: int) -> Dict:
        """Calculate comprehensive team statistics"""
        if not matches:
            return {
                'form': 0.5,
                'attack_strength': 1.0,
                'defense_weakness': 1.0,
                'home_away_diff': 0.0,
                'clean_sheets': 0.0,
                'xG': 1.5,
                'xGA': 1.2
            }
        
        points = 0
        goals_for = 0
        goals_against = 0
        home_goals = 0
        away_goals = 0
        clean_sheets = 0
        xG = 0
        xGA = 0
        
        for match in matches:
            is_home = match['homeTeam']['id'] == team_id
            
            if is_home:
                team_goals = match['score']['fullTime']['home']
                opponent_goals = match['score']['fullTime']['away']
                home_goals += team_goals
            else:
                team_goals = match['score']['fullTime']['away']
                opponent_goals = match['score']['fullTime']['home']
                away_goals += team_goals
                
            if team_goals > opponent_goals:
                points += 3
            elif team_goals == opponent_goals:
                points += 1
                
            goals_for += team_goals
            goals_against += opponent_goals
            
            if opponent_goals == 0:
                clean_sheets += 1
            
            xG += team_goals * 0.95
            xGA += opponent_goals * 1.05
        
        form = points / (len(matches) * 3)
        attack_strength = goals_for / len(matches) / 1.5
        defense_weakness = goals_against / len(matches) / 1.2
        
        home_avg = home_goals / max(1, len([m for m in matches if m['homeTeam']['id'] == team_id]))
        away_avg = away_goals / max(1, len([m for m in matches if m['awayTeam']['id'] == team_id]))
        home_away_diff = home_avg - away_avg
        
        return {
            'form': form,
            'attack_strength': attack_strength,
            'defense_weakness': defense_weakness,
            'home_away_diff': home_away_diff,
            'clean_sheets': clean_sheets / len(matches),
            'xG': xG / len(matches),
            'xGA': xGA / len(matches)
        }

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
        match_id = match['id']
        home_team = match['homeTeam']
        away_team = match['awayTeam']
        venue = match.get('venue', 'Unknown')
        
        home_stats = self.team_stats.get(home_team['id'], {})
        away_stats = self.team_stats.get(away_team['id'], {})
        weather = self.weather_data.get(venue, {})
        injuries = {
            'home': len(self.injury_data.get(str(home_team['id']), [])),
            'away': len(self.injury_data.get(str(away_team['id']), []))
        }
        stadium = match.get('stadium', {})
        odds = self.odds_data.get(str(match_id), {})
        
        base_home_attack = home_stats.get('xG', 1.5)
        base_away_attack = away_stats.get('xG', 1.2)
        base_home_defense = home_stats.get('xGA', 1.2)
        base_away_defense = home_stats.get('xGA', 1.4)
        
        (final_home_attack, 
         final_away_attack, 
         final_home_defense, 
         final_away_defense) = self._apply_all_factors(
            base_home_attack, base_away_attack,
            base_home_defense, base_away_defense,
            home_stats, away_stats,
            weather, injuries, stadium, odds
        )
        
        simulation = self._run_advanced_simulation(
            final_home_attack, final_away_attack,
            final_home_defense, final_away_defense,
            match_id
        )
        
        betting_markets = self._analyze_betting_markets(simulation, odds)
        
        return {
            "match_info": {
                "id": match_id,
                "home_team": home_team['name'],
                "away_team": away_team['name'],
                "competition": match['competition']['name'],
                "date": match['utcDate'],
                "venue": venue
            },
            "key_factors": {
                "form": {
                    "home": f"{home_stats.get('form', 0.5)*100:.1f}%",
                    "away": f"{away_stats.get('form', 0.5)*100:.1f}%"
                },
                "injuries": injuries,
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

    def _apply_all_factors(self, home_attack, away_attack, home_defense, away_defense,
                         home_stats, away_stats, weather, injuries, stadium, odds):
        """Apply all adjustment factors to base expectations"""
        home_attack *= 0.8 + (home_stats.get('form', 0.5) * 0.4)
        away_attack *= 0.8 + (away_stats.get('form', 0.5) * 0.4)
        
        home_attack *= 1 + home_stats.get('home_away_diff', 0) * 0.3
        away_attack *= 1 - away_stats.get('home_away_diff', 0) * 0.3
        
        home_injuries = len(injuries.get('home', []))
        away_injuries = len(injuries.get('away', []))
        
        home_injury_factor = 1 - (min(home_injuries, 5) * 0.04)
        away_injury_factor = 1 - (min(away_injuries, 5) * 0.04)

        home_attack *= home_injury_factor
        away_attack *= away_injury_factor
        
        if weather:
            if weather.get('precip_mm', 0) > 5:
                home_attack *= 0.9
                away_attack *= 0.9
            if weather.get('wind_kph', 0) > 20:
                home_attack *= 0.95
                away_attack *= 0.95
            if weather.get('temp_c', 20) > 28:
                home_attack *= 0.97
                away_attack *= 0.97
        
        home_attack *= stadium.get('home_advantage', 1.0)
        if stadium.get('pitch_size') == 'large':
            home_attack *= 1.05
            away_defense *= 0.98
        
        if odds:
            home_implied = 1 / odds.get('home_odds', 2.0)
            away_implied = 1 / odds.get('away_odds', 2.5)
            
            home_attack = (home_attack * 0.7) + (home_implied * 1.5 * 0.3)
            away_attack = (away_attack * 0.7) + (away_implied * 1.5 * 0.3)
        
        return home_attack, away_attack, home_defense, away_defense

    def _run_advanced_simulation(self, home_attack, away_attack, home_defense, away_defense, match_id, n_simulations=10000):
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
            home_goals = np.random.poisson(home_attack * (1/away_defense))
            away_goals = np.random.poisson(away_attack * (1/home_defense))
            
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
                "Home": home_attack * (1/away_defense),
                "Away": away_attack * (1/home_defense)
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
            "Away +1.0": 1 - np.mean([1 if x > 1 else (0.5 if x == 1 else 0) for x in goal_differences]),
            "Home +0.5": np.mean([1 if x > -0.5 else 0 for x in goal_differences]),
            "Away -0.5": 1 - np.mean([1 if x > -0.5 else 0 for x in goal_differences])
        }

    def _analyze_betting_markets(self, simulation, odds):
        """Analyze all available betting markets"""
        markets = {}
        
        if not odds:
            return {"error": "No odds data available"}
        
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

    def _display_market(self, market_data, highlight_value):
        """Display a betting market with optional value highlighting"""
        if not market_data or 'error' in market_data:
            st.warning("No data available for this market")
            return
        
        for outcome, data in market_data.items():
            if not data or data['odds'] is None:
                continue
                
            value_class = ""
            if highlight_value:
                if data.get('value', 0) > 0.1:
                    value_class = "value-bet"
                elif data.get('value', 0) > 0:
                    value_class = "neutral-bet"
                else:
                    value_class = "bad-bet"
            
            cols = st.columns([2, 1, 1, 1, 2])
            with cols[0]:
                st.markdown(f"**{outcome}**")
            with cols[1]:
                st.write(f"{data['odds']:.2f}")
            with cols[2]:
                st.write(f"{data['value']*100:.1f}%" if data['value'] is not None else "N/A")
            with cols[3]:
                st.write(f"{data['kelly']*100:.1f}%" if data['kelly'] is not None else "N/A")
            with cols[4]:
                st.write(data['recommendation'])
            
            st.markdown("---")

def main():
    st.set_page_config(
        page_title="Ultimate Football Predictor Pro",
        page_icon="⚽",
        layout="wide"
    )
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.update({
            'logged_in': False,
            'username': None,
            'login_attempts': 0,
            'available_mb': 'Unknown'
        })

    try:
        # Initialize predictor
        if 'predictor' not in st.session_state:
            with st.spinner("Initializing system..."):
                st.session_state.predictor = UltimateFootballPredictor()
                if not st.session_state.predictor.fetch_all_data():
                    st.warning("Initial data load partially failed - some features may be limited")
                else:
                    st.success("System initialized successfully")
        
        predictor = st.session_state.predictor
        
        # Debug panel
        with st.sidebar:
            st.header("Debug Info")
            if st.checkbox("Show system info"):
                st.sidebar.write(f"Matches: {len(predictor.today_matches)}")
                st.sidebar.write(f"Model: {type(predictor.model).__name__}")
                st.sidebar.write(f"Last Update: {datetime.fromtimestamp(predictor.last_update).strftime('%Y-%m-%d %H:%M:%S')}")
                
                if st.button("Refresh App"):
                    st.cache_data.clear()
                    st.rerun()
        
        # Authentication flow
        if not st.session_state.logged_in:
            show_login_page(predictor)
        else:
            show_prediction_page(predictor)
            
    except Exception as e:
        logging.error(f"Main loop error: {str(e)}")
        st.error(f"System initialization failed: {str(e)}. Please refresh the page or try again later.")
        st.stop()

# Placeholder UI functions
def show_login_page(predictor):
    st.write("Login page placeholder - please implement authentication logic.")
    if st.button("Login (Demo)"):
        st.session_state.logged_in = True
        st.rerun()

def show_prediction_page(predictor):
    st.write("Prediction page placeholder - displaying predictor output.")
    predictions = predictor.predict_all_matches()
    for pred in predictions:
        st.json(pred)

if __name__ == "__main__":
    main()
