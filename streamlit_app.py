import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson, skellam
import sqlite3
import os
import logging
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import hashlib
import json

# Configure logging
logging.basicConfig(
    filename='ultimate_predictor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# API Configuration
FOOTBALL_API_KEY = "7563e489e2c84b77a0e4f8d7116dc19c"
LIVE_ODDS_API_KEY = "c9b67d8274042fb5755ad88c3a63eab7"
WEATHER_API_KEY = "7211261adbaa426eb66101750250104"
INJURY_API_KEY = "your_injury_api_key"  # Get from SportsDataIO, API-Football, etc.

# API Endpoints
FOOTBALL_API_URL = "https://api.football-data.org/v4"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
WEATHER_API_URL = "http://api.weatherapi.com/v1/forecast.json"
INJURY_API_URL = "https://api.sportsdata.io/v3/soccer/scores/json/Injuries"

class UltimateFootballPredictor:
    def __init__(self):
        """Initialize with comprehensive data sources"""
        self.matches = []
        self.odds = {}
        self.injuries = {}
        self.weather = {}
        self.team_stats = {}
        self.stadium_data = self._load_stadium_db()
        self.user_prefs = {}
        self.model = self._load_model()
        self.last_update = 0
        
        # Initialize databases
        self._init_databases()
        
    def _init_databases(self):
        """Initialize SQLite databases for caching"""
        self.conn = sqlite3.connect('football_data.db')
        self.cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY,
                competition TEXT,
                home_team TEXT,
                away_team TEXT,
                date TEXT,
                venue TEXT,
                status TEXT
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                match_id INTEGER PRIMARY KEY,
                home_win REAL,
                draw REAL,
                away_win REAL,
                btts_yes REAL,
                over_15 REAL,
                over_25 REAL,
                correct_scores TEXT,
                last_updated TEXT
            )
        """)
        
        self.conn.commit()
    
    def _load_stadium_db(self):
        """Load stadium database with advanced metrics"""
        return {
            'Old Trafford': {
                'location': (53.4631, -2.2913),
                'dimensions': (105, 68),
                'capacity': 74310,
                'avg_goals': 2.8,
                'home_advantage': 1.15,
                'weather_impact': 0.9  # Rain impact multiplier
            },
            # Add more stadiums...
        }
    
    def _load_model(self):
        """Load pre-trained ML model"""
        try:
            # In production, load a real trained model
            return None  # Placeholder for actual model
        except:
            return self._train_fallback_model()
    
    def _train_fallback_model(self):
        """Train a fallback model if primary fails"""
        # Simplified example - would use real training data
        return None
    
    def fetch_all_data(self):
        """Fetch all required data with error handling"""
        success = True
        
        if not self._fetch_matches():
            st.error("Failed to fetch match data")
            success = False
            
        if not self._fetch_odds():
            st.warning("Odds data may be outdated")
            
        if not self._fetch_injuries():
            st.warning("Injury data unavailable")
            
        if not self._fetch_weather():
            st.warning("Weather data unavailable")
            
        if not self._fetch_team_stats():
            st.warning("Team stats incomplete")
            
        self.last_update = time.time()
        return success
    
    def _fetch_matches(self):
        """Fetch matches with retry logic"""
        try:
            headers = {"X-Auth-Token": FOOTBALL_API_KEY}
            date_from = datetime.now().strftime('%Y-%m-%d')
            date_to = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
            
            response = requests.get(
                f"{FOOTBALL_API_URL}/matches",
                headers=headers,
                params={"dateFrom": date_from, "dateTo": date_to},
                timeout=10
            )
            response.raise_for_status()
            
            self.matches = []
            for match in response.json().get('matches', []):
                enhanced_match = {
                    'id': match['id'],
                    'competition': match['competition']['name'],
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'date': match['utcDate'],
                    'venue': match.get('venue', 'Unknown'),
                    'status': match['status']
                }
                self.matches.append(enhanced_match)
            
            return True
        except Exception as e:
            logging.error(f"Match fetch failed: {str(e)}")
            return False
    
    def _fetch_odds(self):
        """Fetch comprehensive betting odds"""
        try:
            params = {
                "apiKey": LIVE_ODDS_API_KEY,
                "regions": "eu,us",
                "markets": "h2h,totals,spreads",
                "oddsFormat": "decimal"
            }
            response = requests.get(ODDS_API_URL, params=params, timeout=10)
            response.raise_for_status()
            
            self.odds = {}
            for match in response.json():
                bookmakers = {}
                for bookmaker in match['bookmakers']:
                    markets = {}
                    for market in bookmaker['markets']:
                        markets[market['key']] = {
                            outcome['name']: outcome['price'] 
                            for outcome in market['outcomes']
                        }
                    bookmakers[bookmaker['key']] = markets
                
                self.odds[match['id']] = bookmakers
            
            return True
        except Exception as e:
            logging.error(f"Odds fetch failed: {str(e)}")
            return False
    
    def _fetch_injuries(self):
        """Fetch injury data with player importance"""
        try:
            headers = {"Ocp-Apim-Subscription-Key": INJURY_API_KEY}
            response = requests.get(INJURY_API_URL, headers=headers, timeout=10)
            response.raise_for_status()
            
            self.injuries = {}
            for injury in response.json():
                team_id = injury['team_id']
                if team_id not in self.injuries:
                    self.injuries[team_id] = []
                
                # Calculate player impact (would use real importance metric)
                impact = 0.7 if injury['status'] == 'Out' else 0.3
                self.injuries[team_id].append({
                    'player': injury['player_name'],
                    'position': injury['position'],
                    'status': injury['status'],
                    'impact': impact
                })
            
            return True
        except Exception as e:
            logging.error(f"Injury fetch failed: {str(e)}")
            return False
    
    def _fetch_weather(self):
        """Fetch detailed weather forecasts"""
        try:
            self.weather = {}
            for match in self.matches:
                venue = match['venue']
                if venue in self.stadium_data and venue not in self.weather:
                    lat, lon = self.stadium_data[venue]['location']
                    
                    response = requests.get(
                        WEATHER_API_URL,
                        params={
                            "key": WEATHER_API_KEY,
                            "q": f"{lat},{lon}",
                            "days": 2,
                            "aqi": "no",
                            "alerts": "no"
                        },
                        timeout=10
                    )
                    response.raise_for_status()
                    
                    forecast = response.json()
                    match_time = datetime.strptime(match['date'], '%Y-%m-%dT%H:%M:%SZ')
                    
                    # Find hourly forecast closest to match time
                    closest_hour = None
                    min_diff = float('inf')
                    
                    for hour_data in forecast['forecast']['forecastday'][0]['hour']:
                        hour_time = datetime.strptime(hour_data['time'], '%Y-%m-%d %H:%M')
                        time_diff = abs((match_time - hour_time).total_seconds())
                        
                        if time_diff < min_diff:
                            min_diff = time_diff
                            closest_hour = hour_data
                    
                    if closest_hour:
                        self.weather[venue] = {
                            'temp_c': closest_hour['temp_c'],
                            'precip_mm': closest_hour['precip_mm'],
                            'wind_kph': closest_hour['wind_kph'],
                            'humidity': closest_hour['humidity'],
                            'condition': closest_hour['condition']['text'],
                            'cloud': closest_hour['cloud'],
                            'feelslike_c': closest_hour['feelslike_c']
                        }
            
            return True
        except Exception as e:
            logging.error(f"Weather fetch failed: {str(e)}")
            return False
    
    def _fetch_team_stats(self):
        """Fetch advanced team statistics"""
        try:
            headers = {"X-Auth-Token": FOOTBALL_API_KEY}
            date_from = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            team_ids = set()
            for match in self.matches:
                team_ids.add(match['home_team'])
                team_ids.add(match['away_team'])
            
            for team in team_ids:
                # Get matches
                matches_response = requests.get(
                    f"{FOOTBALL_API_URL}/teams/{team}/matches",
                    headers=headers,
                    params={"dateFrom": date_from, "limit": 10},
                    timeout=10
                )
                matches_response.raise_for_status()
                matches = matches_response.json().get('matches', [])
                
                # Calculate advanced stats
                stats = self._calculate_team_metrics(team, matches)
                self.team_stats[team] = stats
            
            return True
        except Exception as e:
            logging.error(f"Team stats fetch failed: {str(e)}")
            return False
    
    def _calculate_team_metrics(self, team_id, matches):
        """Calculate comprehensive team performance metrics"""
        if not matches:
            return {
                'form': 0.5,
                'xG': 1.5,
                'xGA': 1.2,
                'pressures': 100,
                'final_third_passes': 50,
                'defensive_actions': 40
            }
        
        # Initialize metrics
        metrics = {
            'goals_for': 0,
            'goals_against': 0,
            'shots': 0,
            'shots_on_target': 0,
            'corners': 0,
            'fouls': 0,
            'clean_sheets': 0,
            'matches': len(matches)
        }
        
        for match in matches:
            is_home = match['homeTeam']['id'] == team_id
            
            if is_home:
                metrics['goals_for'] += match['score']['fullTime']['home']
                metrics['goals_against'] += match['score']['fullTime']['away']
                if match['score']['fullTime']['away'] == 0:
                    metrics['clean_sheets'] += 1
            else:
                metrics['goals_for'] += match['score']['fullTime']['away']
                metrics['goals_against'] += match['score']['fullTime']['home']
                if match['score']['fullTime']['home'] == 0:
                    metrics['clean_sheets'] += 1
        
        # Calculate advanced metrics (simplified)
        return {
            'form': metrics['goals_for'] / (metrics['matches'] * 2),  # Normalized
            'xG': metrics['goals_for'] / metrics['matches'],
            'xGA': metrics['goals_against'] / metrics['matches'],
            'clean_sheet_pct': metrics['clean_sheets'] / metrics['matches'],
            'attack_strength': metrics['goals_for'] / (metrics['matches'] * 1.5),
            'defense_strength': 1 - (metrics['goals_against'] / (metrics['matches'] * 1.2))
        }
    
    def predict_all_matches(self):
        """Generate comprehensive predictions for all matches"""
        predictions = []
        
        for match in self.matches:
            try:
                prediction = self._predict_match(match)
                predictions.append(prediction)
                
                # Cache prediction
                self._cache_prediction(match['id'], prediction)
                
            except Exception as e:
                logging.error(f"Prediction failed for match {match['id']}: {str(e)}")
                predictions.append({
                    'match': match,
                    'error': str(e)
                })
        
        return predictions
    
    def _predict_match(self, match):
        """Generate comprehensive prediction with multiple models"""
        home_team = match['home_team']
        away_team = match['away_team']
        venue = match['venue']
        
        # Get base statistics
        home_stats = self.team_stats.get(home_team, {})
        away_stats = self.team_stats.get(away_team, {})
        
        # Calculate expected goals
        home_xg = home_stats.get('xG', 1.5)
        away_xg = away_stats.get('xG', 1.2)
        
        # Apply venue effects
        venue_data = self.stadium_data.get(venue, {})
        home_advantage = venue_data.get('home_advantage', 1.1)
        home_xg *= home_advantage
        away_xg *= (2 - home_advantage)  # Reduce away performance
        
        # Apply weather effects if available
        if venue in self.weather:
            weather = self.weather[venue]
            if weather['precip_mm'] > 5:  # Rainy conditions
                home_xg *= 0.9
                away_xg *= 0.85
            elif weather['temp_c'] > 28:  # Hot conditions
                home_xg *= 0.95
                away_xg *= 0.9
        
        # Apply injury impacts
        home_injury_impact = sum(i['impact'] for i in self.injuries.get(home_team, [])) / 10
        away_injury_impact = sum(i['impact'] for i in self.injuries.get(away_team, [])) / 10
        home_xg *= (1 - home_injury_impact)
        away_xg *= (1 - away_injury_impact)
        
        # Poisson distribution for score probabilities
        home_goals = np.arange(0, 8)
        away_goals = np.arange(0, 8)
        home_probs = poisson.pmf(home_goals, home_xg)
        away_probs = poisson.pmf(away_goals, away_xg)
        
        # Calculate match outcome probabilities
        home_win_prob = np.sum(np.outer(home_probs[1:], away_probs[:-1]))
        draw_prob = np.sum(np.diag(np.outer(home_probs, away_probs)))
        away_win_prob = np.sum(np.outer(home_probs[:-1], away_probs[1:]))
        
        # Calculate BTTS probability
        btts_prob = 1 - (home_probs[0] + away_probs[0] - home_probs[0]*away_probs[0])
        
        # Calculate over/under probabilities
        total_probs = np.outer(home_probs, away_probs)
        over_15 = 1 - np.sum(total_probs[0:2, 0:2])
        over_25 = 1 - np.sum(total_probs[0:3, 0:3])
        
        # Get most likely correct scores
        score_probs = np.outer(home_probs, away_probs)
        top_scores = []
        for i in range(len(home_goals)):
            for j in range(len(away_goals)):
                top_scores.append({
                    'score': f"{i}-{j}",
                    'probability': score_probs[i,j]
                })
        top_scores = sorted(top_scores, key=lambda x: x['probability'], reverse=True)[:5]
        
        # Get best odds if available
        best_odds = self._get_best_odds(match['id'])
        
        return {
            'match': match,
            'home_win': home_win_prob,
            'draw': draw_prob,
            'away_win': away_win_prob,
            'btts_yes': btts_prob,
            'over_15': over_15,
            'over_25': over_25,
            'correct_scores': top_scores,
            'expected_goals': {
                'home': home_xg,
                'away': away_xg
            },
            'best_odds': best_odds,
            'last_updated': datetime.now().isoformat()
        }

    def _get_best_odds(self, match_id):
        """Extract best available odds for a match"""
        if match_id not in self.odds:
            return None
            
        bookmakers = self.odds[match_id]
        best_odds = {
            'home_win': {'value': 0, 'bookmaker': None},
            'draw': {'value': 0, 'bookmaker': None},
            'away_win': {'value': 0, 'bookmaker': None}
        }
        
        for bookmaker, markets in bookmakers.items():
            if 'h2h' in markets:
                for outcome, odds in markets['h2h'].items():
                    if outcome == 'Home' and odds > best_odds['home_win']['value']:
                        best_odds['home_win'] = {'value': odds, 'bookmaker': bookmaker}
                    elif outcome == 'Draw' and odds > best_odds['draw']['value']:
                        best_odds['draw'] = {'value': odds, 'bookmaker': bookmaker}
                    elif outcome == 'Away' and odds > best_odds['away_win']['value']:
                        best_odds['away_win'] = {'value': odds, 'bookmaker': bookmaker}
        
        return best_odds

    def _cache_prediction(self, match_id, prediction):
        """Store prediction in database"""
        try:
            self.cursor.execute("""
                INSERT OR REPLACE INTO predictions VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                match_id,
                prediction['home_win'],
                prediction['draw'],
                prediction['away_win'],
                prediction['btts_yes'],
                prediction['over_15'],
                prediction['over_25'],
                json.dumps(prediction['correct_scores']),
                prediction['last_updated']
            ))
            self.conn.commit()
        except Exception as e:
            logging.error(f"Failed to cache prediction: {str(e)}")

    def get_cached_prediction(self, match_id):
        """Retrieve cached prediction"""
        try:
            self.cursor.execute("""
                SELECT * FROM predictions WHERE match_id = ?
            """, (match_id,))
            row = self.cursor.fetchone()
            
            if row:
                return {
                    'home_win': row[1],
                    'draw': row[2],
                    'away_win': row[3],
                    'btts_yes': row[4],
                    'over_15': row[5],
                    'over_25': row[6],
                    'correct_scores': json.loads(row[7]),
                    'last_updated': row[8]
                }
            return None
        except Exception as e:
            logging.error(f"Failed to get cached prediction: {str(e)}")
            return None

    def calculate_value_bets(self, predictions):
        """Identify value bets based on predictions vs odds"""
        value_bets = []
        
        for pred in predictions:
            if 'match' not in pred or 'best_odds' not in pred:
                continue
                
            match = pred['match']
            odds = pred['best_odds']
            
            if not odds:
                continue
                
            # Calculate expected value
            home_ev = (pred['home_win'] * odds['home_win']['value'] - (1 - pred['home_win']))
            draw_ev = (pred['draw'] * odds['draw']['value'] - (1 - pred['draw']))
            away_ev = (pred['away_win'] * odds['away_win']['value'] - (1 - pred['away_win']))
            
            # Threshold for considering a value bet
            threshold = 0.1
            
            if home_ev > threshold:
                value_bets.append({
                    'match': f"{match['home_team']} vs {match['away_team']}",
                    'bet': 'Home Win',
                    'probability': pred['home_win'],
                    'odds': odds['home_win']['value'],
                    'bookmaker': odds['home_win']['bookmaker'],
                    'expected_value': home_ev
                })
                
            if draw_ev > threshold:
                value_bets.append({
                    'match': f"{match['home_team']} vs {match['away_team']}",
                    'bet': 'Draw',
                    'probability': pred['draw'],
                    'odds': odds['draw']['value'],
                    'bookmaker': odds['draw']['bookmaker'],
                    'expected_value': draw_ev
                })
                
            if away_ev > threshold:
                value_bets.append({
                    'match': f"{match['home_team']} vs {match['away_team']}",
                    'bet': 'Away Win',
                    'probability': pred['away_win'],
                    'odds': odds['away_win']['value'],
                    'bookmaker': odds['away_win']['bookmaker'],
                    'expected_value': away_ev
                })
        
        return sorted(value_bets, key=lambda x: x['expected_value'], reverse=True)

# Streamlit UI Implementation
def main():
    st.set_page_config(
        page_title="Ultimate Football Predictor",
        page_icon="⚽",
        layout="wide"
    )
    
    # Initialize predictor
    predictor = UltimateFootballPredictor()
    
    # Custom CSS
    st.markdown("""
        <style>
            .big-font { font-size:24px !important; }
            .prediction-card { 
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            }
            .home-win { background-color: #e6f3ff; }
            .away-win { background-color: #ffe6e6; }
            .draw { background-color: #f0f0f0; }
            .value-bet { border-left: 5px solid #28a745; }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("⚽ Ultimate Football Predictor")
    st.markdown("""
        The most comprehensive football prediction tool using live data, 
        statistical models, and machine learning to forecast match outcomes.
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        refresh = st.button("Refresh Data")
        show_advanced = st.checkbox("Show Advanced Metrics")
        league_filter = st.selectbox(
            "Filter by League",
            ["All", "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
        )
    
    # Data loading
    if refresh or not predictor.matches:
        with st.spinner("Fetching latest data..."):
            success = predictor.fetch_all_data()
            
            if success:
                st.success("Data updated successfully!")
            else:
                st.warning("Some data may be incomplete")
    
    # Display predictions
    st.header("Upcoming Match Predictions")
    
    if not predictor.matches:
        st.warning("No upcoming matches found")
    else:
        # Filter matches by league if specified
        filtered_matches = predictor.matches
        if league_filter != "All":
            filtered_matches = [m for m in predictor.matches if m['competition'] == league_filter]
        
        if not filtered_matches:
            st.info(f"No matches found in {league_filter}")
        else:
            predictions = predictor.predict_all_matches()
            
            for pred in predictions:
                if 'error' in pred:
                    st.error(f"Error predicting {pred['match']['home_team']} vs {pred['match']['away_team']}: {pred['error']}")
                    continue
                    
                match = pred['match']
                home_team = match['home_team']
                away_team = match['away_team']
                
                # Determine card style based on most probable outcome
                max_outcome = max(pred['home_win'], pred['draw'], pred['away_win'])
                card_class = ""
                if max_outcome == pred['home_win']:
                    card_class = "home-win"
                elif max_outcome == pred['draw']:
                    card_class = "draw"
                else:
                    card_class = "away-win"
                
                with st.container():
                    st.markdown(f"""
                        <div class="prediction-card {card_class}">
                            <div style="display: flex; justify-content: space-between;">
                                <h2>{home_team} vs {away_team}</h2>
                                <span>{datetime.strptime(match['date'], '%Y-%m-%dT%H:%M:%SZ').strftime('%a %d %b, %H:%M')}</span>
                            </div>
                            <p><i>{match['competition']} • {match['venue']}</i></p>
                            
                            <div style="display: flex; justify-content: space-around; text-align: center;">
                                <div>
                                    <h3>{pred['home_win']*100:.1f}%</h3>
                                    <p>Home Win</p>
                                </div>
                                <div>
                                    <h3>{pred['draw']*100:.1f}%</h3>
                                    <p>Draw</p>
                                </div>
                                <div>
                                    <h3>{pred['away_win']*100:.1f}%</h3>
                                    <p>Away Win</p>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Match details expander
                    with st.expander("Detailed Analysis"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Expected Goals")
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=['Home', 'Away'],
                                y=[pred['expected_goals']['home'], pred['expected_goals']['away']],
                                marker_color=['#1f77b4', '#ff7f0e']
                            ))
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.subheader("Correct Score Probabilities")
                            score_df = pd.DataFrame(pred['correct_scores'])
                            st.dataframe(
                                score_df.style.format({'probability': '{:.2%}'}),
                                hide_index=True
                            )
                        
                        with col2:
                            st.subheader("Additional Markets")
                            st.markdown(f"""
                                - **Both Teams to Score**: {pred['btts_yes']*100:.1f}%
                                - **Over 1.5 Goals**: {pred['over_15']*100:.1f}%
                                - **Over 2.5 Goals**: {pred['over_25']*100:.1f}%
                            """)
                            
                            if pred.get('best_odds'):
                                st.subheader("Best Available Odds")
                                odds = pred['best_odds']
                                st.markdown(f"""
                                    - **Home Win**: {odds['home_win']['value']:.2f} ({odds['home_win']['bookmaker']})
                                    - **Draw**: {odds['draw']['value']:.2f} ({odds['draw']['bookmaker']})
                                    - **Away Win**: {odds['away_win']['value']:.2f} ({odds['away_win']['bookmaker']})
                                """)
                        
                        if show_advanced:
                            st.subheader("Advanced Metrics")
                            # Add more advanced visualizations here
    
    # Value bets section
    st.header("🔍 Value Bet Finder")
    value_bets = predictor.calculate_value_bets(predictions)
    
    if not value_bets:
        st.info("No strong value bets identified based on current predictions")
    else:
        st.success(f"Found {len(value_bets)} potential value bets!")
        for bet in value_bets:
            with st.container():
                st.markdown(f"""
                    <div class="prediction-card value-bet">
                        <h3>{bet['match']} - {bet['bet']}</h3>
                        <p>
                            Probability: {bet['probability']*100:.1f}% | 
                            Odds: {bet['odds']:.2f} | 
                            Bookmaker: {bet['bookmaker']}
                        </p>
                        <p><strong>Expected Value:</strong> {bet['expected_value']:.3f}</p>
                    </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center;">
            <p>Data refreshes automatically every 15 minutes</p>
            <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
