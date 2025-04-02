import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
import sqlite3
import logging
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FootballPredictor:
    def __init__(self):
        self.matches = []
        self.team_stats = {}
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite database with error handling"""
        try:
            self.conn = sqlite3.connect('football_data.db', timeout=10)
            self.cursor = self.conn.cursor()
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    match_id INTEGER PRIMARY KEY,
                    home_win REAL,
                    draw REAL,
                    away_win REAL,
                    last_updated TEXT
                )
            """)
            self.conn.commit()
        except Exception as e:
            logging.error(f"Database initialization failed: {str(e)}")
            st.error("Failed to initialize database. Please check permissions.")
            sys.exit(1)

    def fetch_sample_data(self):
        """Use sample data if API calls fail"""
        self.matches = [
            {
                'id': 1,
                'home_team': 'Team A',
                'away_team': 'Team B',
                'date': (datetime.now() + timedelta(days=1)).isoformat(),
                'competition': 'Premier League',
                'venue': 'Sample Stadium'
            }
        ]
        
        self.team_stats = {
            'Team A': {'xG': 1.8, 'xGA': 1.2},
            'Team B': {'xG': 1.5, 'xGA': 1.4}
        }
        return True

    def predict_match(self, home_team, away_team):
        """Simplified prediction with fallbacks"""
        try:
            home_xg = self.team_stats.get(home_team, {}).get('xG', 1.5)
            away_xg = self.team_stats.get(away_team, {}).get('xG', 1.2)
            
            # Poisson distribution
            home_probs = poisson.pmf(np.arange(0, 8), home_xg)
            away_probs = poisson.pmf(np.arange(0, 8), away_xg)
            
            home_win = np.sum(np.outer(home_probs[1:], away_probs[:-1]))
            draw = np.sum(np.diag(np.outer(home_probs, away_probs)))
            away_win = np.sum(np.outer(home_probs[:-1], away_probs[1:]))
            
            return {
                'home_win': home_win,
                'draw': draw,
                'away_win': away_win,
                'expected_goals': {
                    'home': home_xg,
                    'away': away_xg
                }
            }
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return {
                'home_win': 0.45,
                'draw': 0.3,
                'away_win': 0.25,
                'expected_goals': {
                    'home': 1.5,
                    'away': 1.2
                }
            }

def main():
    st.set_page_config(
        page_title="Football Predictor",
        page_icon="⚽",
        layout="wide"
    )
    
    # Initialize with error handling
    try:
        predictor = FootballPredictor()
        if not predictor.fetch_sample_data():
            st.error("Failed to initialize data")
            return
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        return
    
    # UI Elements
    st.title("⚽ Football Match Predictor")
    st.write("This app predicts match outcomes using statistical models")
    
    # Match selection
    if not predictor.matches:
        st.warning("No matches available")
        return
        
    match = predictor.matches[0]  # Using first match for demo
    
    # Display match info
    st.header(f"{match['home_team']} vs {match['away_team']}")
    st.write(f"Competition: {match['competition']}")
    st.write(f"Date: {match['date']}")
    
    # Make prediction
    with st.spinner("Calculating predictions..."):
        prediction = predictor.predict_match(
            match['home_team'],
            match['away_team']
        )
    
    # Display results
    st.subheader("Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Home Win", f"{prediction['home_win']*100:.1f}%")
    with col2:
        st.metric("Draw", f"{prediction['draw']*100:.1f}%")
    with col3:
        st.metric("Away Win", f"{prediction['away_win']*100:.1f}%")
    
    # Expected goals visualization
    st.subheader("Expected Goals")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Home', 'Away'],
        y=[prediction['expected_goals']['home'], prediction['expected_goals']['away']],
        marker_color=['blue', 'red']
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
