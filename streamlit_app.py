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
WEATHER_API_KEY = "7211261adbaa426eb66101750250104"

# API Endpoints
FOOTBALL_API_URL = "https://api.football-data.org/v4"
WEATHER_API_URL = "http://api.weatherapi.com/v1/forecast.json"

class UltimateFootballPredictor:
    def __init__(self):
        """Initialize with data sources"""
        logging.info("Initializing UltimateFootballPredictor...")
        self.today_matches = []
        self.weather_data = {}
        self.stadium_data = self._load_stadium_data()
        self.team_stats = {}
        self.last_update = 0
        logging.info("Predictor initialized successfully")

    def _load_stadium_data(self) -> Dict:
        """Load stadium database"""
        return {
            'Old Trafford': {'location': (53.4631, -2.2913), 'home_advantage': 1.15},
            'Stamford Bridge': {'location': (51.4817, -0.1910), 'home_advantage': 1.1},
            'Allianz Arena': {'location': (48.2188, 11.6247), 'home_advantage': 1.2},
            'Camp Nou': {'location': (41.3809, 2.1228), 'home_advantage': 1.25},
        }

    def fetch_all_data(self):
        """Fetch all required data"""
        logging.info("Fetching all data...")
        results = {
            'matches': self._fetch_daily_matches(),
            'weather': self._fetch_weather_data(),
            'stats': self._fetch_team_stats()
        }
        self.last_update = time.time()
        if not results['matches']:
            logging.error("No matches fetched - check API or date")
            return False
        if not all(results.values()):
            failed = [k for k, v in results.items() if not v]
            logging.warning(f"Partial data load - failed: {failed}")
            st.warning(f"Partial data loaded - failed: {', '.join(failed)}")
        else:
            logging.info("Data fetch completed successfully")
        return True

    def _fetch_daily_matches(self) -> bool:
        """Fetch today's matches across all competitions"""
        try:
            # Use a known match day for testing; revert to datetime.now().strftime('%Y-%m-%d') for live use
            test_date = "2025-03-15"  # Saturday, likely to have matches
            headers = {"X-Auth-Token": FOOTBALL_API_KEY}
            response = requests.get(
                f"{FOOTBALL_API_URL}/matches",
                headers=headers,
                params={"dateFrom": test_date, "dateTo": test_date, "limit": 500},
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
            return False

    def _fetch_team_stats(self) -> bool:
        """Fetch team statistics"""
        try:
            headers = {"X-Auth-Token": FOOTBALL_API_KEY}
            date_from = "2025-02-15"  # 30 days prior to test_date
            for match in self.today_matches:
                for team_type in ['homeTeam', 'awayTeam']:
                    team_id = match[team_type]['id']
                    if team_id not in self.team_stats:
                        response = requests.get(
                            f"{FOOTBALL_API_URL}/teams/{team...

Something went wrong. Please try again.
