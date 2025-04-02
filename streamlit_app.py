import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
import logging
from datetime import datetime, timedelta
import time
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('ultimate_predictor.log'), logging.StreamHandler()]
)

# API Configuration
FOOTBALL_API_KEY = "7563e489e2c84b77a0e4f8d7116dc19c"
LIVE_ODDS_API_KEY = "c9b67d8274042fb5755ad88c3a63eab7"
WEATHER_API_KEY = "7211261adbaa426eb66101750250104"
INJURY_API_KEY = "73bacb5a360ea069e1f45c330358706c"  # Testing key - REMOVE FOR PRODUCTION

# API Endpoints
FOOTBALL_API_URL = "https://api.football-data.org/v4"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports"
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
            logging.info("Predictor initialized successfully")
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            raise

    def _load_stadium_data(self) -> Dict:
        """Load stadium database with coordinates and features"""
        return {
            'Old Trafford': {'location': (53.4631, -2.2913), 'home_advantage': 1.15, 'pitch_size': (105, 68)},
            'Stamford Bridge': {'location': (51.4817, -0.1910), 'home_advantage': 1.1, 'pitch_size': (103, 67)},
            'Allianz Arena': {'location': (48.2188, 11.6247), 'home_advantage': 1.2, 'pitch_size': (105, 68)},  # Bundesliga
            'Camp Nou': {'location': (41.3809, 2.1228), 'home_advantage': 1.25, 'pitch_size': (105, 68)},  # La Liga
        }

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
            self.last_update = time.time()
            if not any(results.values()):
                logging.error("All data fetches failed")
                return False
            failed = [k for k, v in results.items() if not v]
            if failed:
                logging.warning(f"Partial data load - failed: {failed}")
                st.warning(f"Partial data loaded - failed components: {', '.join(failed)}")
            else:
                logging.info("Data fetch completed successfully")
            return True
        except Exception as e:
            logging.error(f"fetch_all_data failed: {str(e)}")
            return False

    def _fetch_daily_matches(self) -> bool:
        """Fetch today's matches across all competitions"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            headers = {"X-Auth-Token": FOOTBALL_API_KEY}
            response = requests.get(
                f"{FOOTBALL_API_URL}/matches",
                headers=headers,
                params={"dateFrom": today, "dateTo": today, "limit": 500},
                timeout=15
            )
            response.raise_for_status()
            self.today_matches = response.json().get("matches", [])
            for match in self.today_matches:
                venue = match.get('venue', 'Unknown')
                match['stadium'] = self.stadium_data.get(venue, {})
            logging.info(f"Fetched {len(self.today_matches)} matches: {[m['competition']['name'] for m in self.today_matches[:5]]}")
            return len(self.today_matches) > 0
        except Exception as e:
            logging.error(f"Match fetch failed: {str(e)}")
            self.today_matches = [
                {"id": 1, "homeTeam": {"id": 1, "name": "Manchester United"}, "awayTeam": {"id": 2, "name": "Chelsea"},
                 "competition": {"name": "Premier League"}, "utcDate": today + "T15:00:00Z", "venue": "Old Trafford"}
            ]  # Fallback
            return True

    def _fetch_odds_data(self) -> bool:
        """Fetch live odds for all soccer competitions"""
        try:
            response = requests.get(
                ODDS_API_URL,
                params={"apiKey": LIVE_ODDS_API_KEY, "all": True},
                timeout=10
            )
            response.raise_for_status()
            sports = [sport['key'] for sport in response.json() if 'soccer' in sport['key'].lower()]
            self.odds_data = {}
            for sport in sports[:5]:  # Limit to 5 for testing
                odds_response = requests.get(
                    f"{ODDS_API_URL}/{sport}/odds",
                    params={"apiKey": LIVE_ODDS_API_KEY, "regions": "eu", "markets": "h2h"},
                    timeout=10
                )
                odds_response.raise_for_status()
                for item in odds_response.json():
                    self.odds_data[item['id']] = {'h2h': item['bookmakers'][0]['markets'][0]['outcomes'] if item['bookmakers'] else []}
            logging.info(f"Odds fetched for {len(self.odds_data)} matches across {len(sports)} leagues")
            return True
        except Exception as e:
            logging.error(f"Odds fetch failed: {str(e)}")
            self.odds_data = {"1": {"h2h": [{"name": "Home Win", "price": 2.0}, {"name": "Away Win", "price": 2.5}]}}  # Fallback
            return False

    def _fetch_injury_data(self) -> bool:
        """Fetch injury data"""
        try:
            headers = {"Ocp-Apim-Subscription-Key": INJURY_API_KEY}
            response = requests.get(INJURY_API_URL, headers=headers, timeout=15)
            response.raise_for_status()
            injuries = response.json()
            self.injury_data = {}
            for injury in injuries:
                team_id = str(injury['TeamID'])
                self.injury_data[team_id] = self.injury_data.get(team_id, []) + [{
                    'player': injury.get('PlayerName', 'Unknown'),
                    'position': injury.get('Position', 'Unknown'),
                    'status': injury.get('Status', 'Unknown')
                }]
            logging.info(f"Injury data fetched for {len(self.injury_data)} teams")
            return True
        except Exception as e:
            logging.error(f"Injury fetch failed: {str(e)}")
            self.injury_data = {"1": [{"player": "Test Player", "position": "FW", "status": "Out"}]}
            return False

    def _fetch_weather_data(self) -> bool:
        """Fetch weather forecasts"""
        try:
            self.weather_data = {}
            for match in self.today_matches:
                venue = match.get('venue', 'Unknown')
                if venue in self.stadium_data and venue not in self.weather_data:
                    lat, lon = self.stadium_data[venue]['location']
                    response = requests.get(
                        WEATHER_API_URL,
                        params={"key": WEATHER_API_KEY, "q": f"{lat},{lon}", "days": 1},
                        timeout=10
                    )
                    response.raise_for_status()
                    forecast = response.json()['forecast']['forecastday'][0]['hour'][0]
                    self.weather_data[venue] = {
                        'temp_c': forecast['temp_c'],
                        'precip_mm': forecast['precip_mm'],
                        'wind_kph': forecast['wind_kph']
                    }
            logging.info(f"Weather fetched for {len(self.weather_data)} venues")
            return True
        except Exception as e:
            logging.error(f"Weather fetch failed: {str(e)}")
            self.weather_data = {"Old Trafford": {"temp_c": 15, "precip_mm": 0, "wind_kph": 10}}
            return False

    def _fetch_team_stats(self) -> bool:
        """Fetch team statistics"""
        try:
            headers = {"X-Auth-Token": FOOTBALL_API_KEY}
            date_from = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            for match in self.today_matches:
                for team_type in ['homeTeam', 'awayTeam']:
                    team_id = match[team_type]['id']
                    if team_id not in self.team_stats:
                        response = requests.get(
                            f"{FOOTBALL_API_URL}/teams/{team_id}/matches",
                            headers=headers,
                            params={"dateFrom": date_from, "limit": 5},
                            timeout=10
                        )
                        response.raise_for_status()
                        self.team_stats[team_id] = self._calculate_advanced_stats(response.json().get('matches', []), team_id)
            logging.info(f"Stats fetched for {len(self.team_stats)} teams")
            return True
        except Exception as e:
            logging.error(f"Stats fetch failed: {str(e)}")
            self.team_stats = {"1": {"xG": 1.5, "xGA": 1.2, "form": 0.6, "fatigue": 0.9}}
            return False

    def _calculate_advanced_stats(self, matches: List, team_id: int) -> Dict:
        """Calculate team stats with weighted form and fatigue"""
        if not matches:
            return {'xG': 1.5, 'xGA': 1.2, 'form': 0.5, 'fatigue': 1.0}
        points, goals_for, goals_against, match_dates = 0, 0, 0, []
        for i, match in enumerate(sorted(matches, key=lambda x: x['utcDate'], reverse=True)):
            weight = 0.9 ** i
            is_home = match['homeTeam']['id'] == team_id
            team_goals = match['score']['fullTime']['home'] if is_home else match['score']['fullTime']['away']
            opp_goals = match['score']['fullTime']['away'] if is_home else match['score']['fullTime']['home']
            points += (3 if team_goals > opp_goals else 1 if team_goals == opp_goals else 0) * weight
            goals_for += team_goals * weight
            goals_against += opp_goals * weight
            match_dates.append(datetime.strptime(match['utcDate'], '%Y-%m-%dT%H:%M:%SZ'))
        total_weight = sum(0.9 ** i for i in range(len(matches)))
        form = points / (len(matches) * 3 * total_weight) if total_weight > 0 else 0.5
        fatigue = 1.0 - min(0.05 * len([d for d in match_dates if (datetime.now() - d).days <= 14]), 0.3)
        return {
            'xG': goals_for / total_weight,
            'xGA': goals_against / total_weight,
            'form': form,
            'fatigue': fatigue
        }

    def predict_all_matches(self) -> List[Dict]:
        """Generate predictions for all matches"""
        predictions = []
        for match in self.today_matches:
            try:
                pred = self._predict_match(match)
                predictions.append(pred)
            except Exception as e:
                logging.error(f"Prediction failed for match {match.get('id')}: {str(e)}")
                predictions.append({"match": match, "error": str(e)})
        logging.info(f"Generated {len(predictions)} predictions")
        return predictions

    def _predict_match(self, match: Dict) -> Dict:
        """Generate prediction for a single match"""
        match_id = match['id']
        home_team = match['homeTeam']
        away_team = match['awayTeam']
        venue = match.get('venue', 'Unknown')
        
        home_stats = self.team_stats.get(home_team['id'], {'xG': 1.5, 'xGA': 1.2, 'form': 0.5, 'fatigue': 1.0})
        away_stats = self.team_stats.get(away_team['id'], {'xG': 1.2, 'xGA': 1.4, 'form': 0.5, 'fatigue': 1.0})
        weather = self.weather_data.get(venue, {})
        injuries = {
            'home': self.injury_data.get(str(home_team['id']), []),
            'away': self.injury_data.get(str(away_team['id']), [])
        }
        stadium = match.get('stadium', {})
        odds = self.odds_data.get(str(match_id), {})
        
        home_attack, away_attack, home_defense, away_defense = self._apply_all_factors(
            home_stats['xG'], away_stats['xG'], home_stats['xGA'], away_stats['xGA'],
            home_stats, away_stats, weather, injuries, stadium, odds, match['competition']
        )
        
        simulation = self._run_advanced_simulation(home_attack, away_attack, home_defense, away_defense)
        
        return {
            "match_info": {
                "id": match_id,
                "home_team": home_team['name'],
                "away_team": away_team['name'],
                "competition": match['competition']['name'],
                "date": match['utcDate'],
                "venue": venue
            },
            "predictions": simulation,
            "key_factors": {
                "form": {"home": f"{home_stats['form']*100:.1f}%", "away": f"{away_stats['form']*100:.1f}%"},
                "injuries": {"home": len(injuries['home']), "away": len(injuries['away'])},
                "weather": weather
            }
        }

    def _apply_all_factors(self, home_attack, away_attack, home_defense, away_defense,
                           home_stats, away_stats, weather, injuries, stadium, odds, competition):
        """Apply all adjustment factors"""
        home_attack *= 0.8 + (home_stats['form'] * 0.4)
        away_attack *= 0.8 + (away_stats['form'] * 0.4)
        home_attack *= home_stats['fatigue']
        away_attack *= away_stats['fatigue']
        
        home_injury_factor = 1.0
        away_injury_factor = 1.0
        for injury in injuries['home']:
            if injury['position'] in ['FW', 'MF']:
                home_injury_factor -= 0.06 if injury['status'] == 'Out' else 0.03
            elif injury['position'] == 'DF':
                home_defense *= 1.05 if injury['status'] == 'Out' else 1.03
        for injury in injuries['away']:
            if injury['position'] in ['FW', 'MF']:
                away_injury_factor -= 0.06 if injury['status'] == 'Out' else 0.03
            elif injury['position'] == 'DF':
                away_defense *= 1.05 if injury['status'] == 'Out' else 1.03
        home_attack *= max(0.7, home_injury_factor)
        away_attack *= max(0.7, away_injury_factor)
        
        if weather.get('precip_mm', 0) > 5:
            home_attack *= 0.9
            away_attack *= 0.9
        home_attack *= stadium.get('home_advantage', 1.0)
        
        comp_factor = 1.1 if competition.get('type', 'LEAGUE') == 'CUP' else 1.0
        home_attack *= comp_factor
        away_attack *= comp_factor
        
        return home_attack, away_attack, home_defense, away_defense

    def _run_advanced_simulation(self, home_attack, away_attack, home_defense, away_defense, n_simulations=1000):
        """Run Monte Carlo simulation"""
        home_wins, draws, away_wins = 0, 0, 0
        for _ in range(n_simulations):
            home_goals = np.random.poisson(home_attack * (1/away_defense))
            away_goals = np.random.poisson(away_attack * (1/home_defense))
            if home_goals > away_goals:
                home_wins += 1
            elif away_goals > home_goals:
                away_wins += 1
            else:
                draws += 1
        total = n_simulations
        return {
            "1X2": {
                "Home Win": home_wins / total,
                "Draw": draws / total,
                "Away Win": away_wins / total
            },
            "Expected Goals": {"Home": home_attack * (1/away_defense), "Away": away_attack * (1/home_defense)}
        }

def main():
    st.set_page_config(page_title="Ultimate Football Predictor Pro", page_icon="⚽", layout="wide")
    
    if 'predictor' not in st.session_state:
        with st.spinner("Initializing system..."):
            st.session_state.predictor = UltimateFootballPredictor()
            if not st.session_state.predictor.fetch_all_data():
                st.warning("Initial data load failed - using fallback data")
            else:
                st.success("System initialized")
    
    predictor = st.session_state.predictor
    
    with st.sidebar:
        st.header("Debug Info")
        if st.checkbox("Show system info"):
            st.write(f"Matches: {len(predictor.today_matches)}")
            st.write(f"Leagues: {set(m['competition']['name'] for m in predictor.today_matches)}")
            st.write(f"Last Update: {datetime.fromtimestamp(predictor.last_update).strftime('%Y-%m-%d %H:%M:%S')}")
        if st.button("Refresh"):
            st.session_state.predictor = UltimateFootballPredictor()
            st.session_state.predictor.fetch_all_data()
            st.rerun()
    
    show_prediction_page(predictor)

def show_prediction_page(predictor):
    st.title("Ultimate Football Predictor Pro")
    st.markdown("### Today's Match Predictions")
    
    predictions = predictor.predict_all_matches()
    if not predictions:
        st.warning("No predictions available")
        return
    
    for pred in predictions:
        if "error" in pred:
            st.error(f"Prediction failed: {pred['error']}")
            continue
        
        match_info = pred["match_info"]
        st.markdown(f"#### {match_info['home_team']} vs {match_info['away_team']}")
        st.write(f"Competition: {match_info['competition']} | Date: {match_info['date']} | Venue: {match_info['venue']}")
        
        with st.expander("Key Factors"):
            factors = pred["key_factors"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Form**")
                st.write(f"Home: {factors['form']['home']}")
                st.write(f"Away: {factors['form']['away']}")
            with col2:
                st.write("**Injuries**")
                st.write(f"Home: {factors['injuries']['home']}")
                st.write(f"Away: {factors['injuries']['away']}")
            with col3:
                st.write("**Weather**")
                st.write(f"Temp: {factors['weather'].get('temp_c', 'N/A')}°C")
        
        st.subheader("Predictions")
        preds = pred["predictions"]
        col1, col2 = st.columns(2)
        with col1:
            st.write("**1X2 Probabilities**")
            st.table({
                "Home Win": f"{preds['1X2']['Home Win']*100:.1f}%",
                "Draw": f"{preds['1X2']['Draw']*100:.1f}%",
                "Away Win": f"{preds['1X2']['Away Win']*100:.1f}%"
            })
        with col2:
            st.write("**Expected Goals**")
            st.write(f"Home: {preds['Expected Goals']['Home']:.2f}")
            st.write(f"Away: {preds['Expected Goals']['Away']:.2f}")
        st.markdown("---")

if __name__ == "__main__":
    main()
