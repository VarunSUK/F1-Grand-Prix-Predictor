import fastf1
import fastf1.ergast.interface
fastf1.ergast.interface.ERGAST_API_URL = "https://api.jolpi.ca/ergast/f1/"
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import requests
import time
from collections import deque
from datetime import datetime, timedelta

# Configure FastF1
import os
cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
fastf1.Cache.enable_cache(cache_dir)

# In-memory fallback event log for monitoring
FALLBACK_EVENTS = deque(maxlen=100)
FALLBACK_THRESHOLD = 3  # Number of fallbacks before alert
FALLBACK_WINDOW = timedelta(minutes=10)

class F1DataService:
    """
    Service for fetching and processing F1 data using FastF1
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("F1DataService initialized")
    
    async def get_race_schedule(self, season: int) -> List[Dict]:
        """
        Get the race schedule for a specific season
        
        Args:
            season: F1 season year
            
        Returns:
            List of race information
        """
        try:
            schedule = fastf1.get_event_schedule(season)
            races = []
            
            for _, race in schedule.iterrows():
                if race['EventFormat'] == 'conventional':
                    races.append({
                        'round': race['RoundNumber'],
                        'name': race['EventName'],
                        'date': race['EventDate'].strftime('%Y-%m-%d'),
                        'circuit': race['CircuitShortName']
                    })
            
            return races
            
        except Exception as e:
            self.logger.error(f"Error fetching race schedule: {e}")
            return []
    
    async def get_qualifying_results(self, season: int, round_num: int) -> pd.DataFrame:
        """
        Get qualifying results for a specific race
        
        Args:
            season: F1 season year
            round_num: Race round number
            
        Returns:
            DataFrame with qualifying results
        """
        try:
            session = fastf1.get_session(season, round_num, 'Q')
            session.load()
            
            results = session.results[['Driver', 'Q3', 'Position']].copy()
            results['Q3'] = pd.to_timedelta(results['Q3']).dt.total_seconds()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error fetching qualifying results: {e}")
            return pd.DataFrame()
    
    async def get_driver_performance(self, driver: str, season: int, last_races: int = 5) -> Dict:
        """
        Get recent performance data for a specific driver
        
        Args:
            driver: Driver name
            season: F1 season year
            last_races: Number of recent races to analyze
            
        Returns:
            Dictionary with driver performance metrics
        """
        try:
            # Get driver's recent race results
            results = []
            schedule = fastf1.get_event_schedule(season)
            
            for _, race in schedule.head(last_races).iterrows():
                if race['EventFormat'] == 'conventional':
                    try:
                        session = fastf1.get_session(season, race['RoundNumber'], 'R')
                        session.load()
                        
                        driver_result = session.results[session.results['Driver'] == driver]
                        if not driver_result.empty:
                            results.append({
                                'round': race['RoundNumber'],
                                'position': driver_result.iloc[0]['Position'],
                                'points': driver_result.iloc[0]['Points']
                            })
                    except:
                        continue
            
            if results:
                avg_position = sum(r['position'] for r in results) / len(results)
                total_points = sum(r['points'] for r in results)
                
                return {
                    'driver': driver,
                    'avg_position': avg_position,
                    'total_points': total_points,
                    'races_analyzed': len(results),
                    'recent_results': results
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error fetching driver performance: {e}")
            return {}
    
    async def get_team_performance(self, team: str, season: int, last_races: int = 3) -> Dict:
        """
        Get recent performance data for a specific team
        
        Args:
            team: Team name
            season: F1 season year
            last_races: Number of recent races to analyze
            
        Returns:
            Dictionary with team performance metrics
        """
        try:
            # TODO: Implement team performance analysis
            # This would aggregate performance of both team drivers
            
            return {
                'team': team,
                'avg_team_position': 0.0,
                'total_team_points': 0,
                'reliability_score': 0.95
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching team performance: {e}")
            return {}
    
    async def get_track_statistics(self, circuit: str, season: int) -> Dict:
        """
        Get track-specific statistics and history
        
        Args:
            circuit: Circuit name
            season: F1 season year
            
        Returns:
            Dictionary with track statistics
        """
        try:
            # TODO: Implement track statistics analysis
            # This would analyze historical performance at this track
            
            return {
                'circuit': circuit,
                'track_type': 'high_speed',
                'avg_pit_stops': 2.1,
                'overtaking_difficulty': 'medium'
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching track statistics: {e}")
            return {}
    
    def _jolpica_api_request(self, url, max_retries=3, delay=2):
        """Centralized Jolpica API request with retry and rate limit handling."""
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 429:
                    self.logger.warning(f"Jolpica API rate limit hit (429) on {url}, attempt {attempt+1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        continue
                    else:
                        return None, 'rate_limit'
                if resp.status_code >= 500:
                    self.logger.warning(f"Jolpica API server error {resp.status_code} on {url}, attempt {attempt+1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                        continue
                    else:
                        return None, 'server_error'
                resp.raise_for_status()
                return resp.json(), None
            except Exception as e:
                self.logger.warning(f"Jolpica API request error on {url}: {e}, attempt {attempt+1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    return None, 'exception'
        return None, 'unknown_error'

    async def get_race_features(self, season: int, round_num: int):
        """
        Extract comprehensive features for all drivers in a race
        For 2024/2025, only Jolpica is used. If Jolpica fails, do NOT fallback to FastF1 or mock data; instead, return None and a clear error note.
        For other years, fallback to mock data is still allowed.
        Returns a tuple: (features_df, used_round, partial_fields, note)
        """
        try:
            self.logger.info(f"Extracting race features for {season} Round {round_num}")
            for r in range(round_num, 0, -1):
                self.logger.info(f"Attempting to load data for round {r}")
            driver_features = []
            partial_fields = set()
            note = None
            # --- Try Jolpica API first ---
            try:
                base_url = "https://api.jolpi.ca/ergast/f1"
                results_url = f"{base_url}/{season}/{r}/results.json"
                quali_url = f"{base_url}/{season}/{r}/qualifying.json"
                results_data, results_err = self._jolpica_api_request(results_url)
                quali_data, quali_err = self._jolpica_api_request(quali_url)
                if results_err == 'rate_limit' or quali_err == 'rate_limit':
                    note = f"Jolpica API rate limit hit for round {r}. No fallback allowed for {season}."
                    self.logger.warning(note)
                    if season in (2024, 2025):
                        self._log_fallback_event(season, round_num, note)
                        return None, None, set(), note
                    else:
                        raise Exception(note)
                if results_err or quali_err:
                    note = f"Jolpica API error ({results_err or ''} {quali_err or ''}) for round {r}. No fallback allowed for {season}."
                    self.logger.warning(note)
                    if season in (2024, 2025):
                        self._log_fallback_event(season, round_num, note)
                        return None, None, set(), note
                    else:
                        raise Exception(note)
                results_list = results_data.get('MRData', {}).get('RaceTable', {}).get('Races', [{}])[0].get('Results', [])
                quali_list = quali_data.get('MRData', {}).get('RaceTable', {}).get('Races', [{}])[0].get('QualifyingResults', [])
                # --- Detect and log new fields ---
                known_result_fields = {"number","position","positionText","points","Driver","Constructor","grid","laps","status","Time","FastestLap"}
                for res in results_list:
                    for key in res.keys():
                        if key not in known_result_fields:
                            self.logger.info(f"[Jolpica] New or unexpected field in results: {key}")
                known_quali_fields = {"number","position","Driver","Constructor","Q1","Q2","Q3"}
                for q in quali_list:
                    for key in q.keys():
                        if key not in known_quali_fields:
                            self.logger.info(f"[Jolpica] New or unexpected field in qualifying: {key}")
                if results_list:
                    self.logger.info(f"Loaded {len(results_list)} results from Jolpica for round {r}")
                    quali_map = {q['Driver']['driverId']: q for q in quali_list}
                    for res in results_list:
                        features = {}
                        driver_id = res['Driver']['driverId']
                        features['driver'] = driver_id
                        features['team'] = res['Constructor']['name'] if 'Constructor' in res else 'Unknown'
                        features['final_position'] = int(res.get('position', 20))
                        features['points_scored'] = float(res.get('points', 0))
                        # Jolpica status enumeration defensive handling for 2025+
                        status = res.get('status', 'Unknown')
                        if season >= 2025:
                            allowed_statuses = {"Finished", "Disqualified", "Accident", "Retired", "Lapped"}
                            if status not in allowed_statuses:
                                status = "Retired"
                        features['race_status'] = status
                        features['team_race'] = res['Constructor']['name'] if 'Constructor' in res else 'Unknown'
                        # Qualifying
                        quali = quali_map.get(driver_id)
                        if quali:
                            q3 = quali.get('Q3') or quali.get('Q2') or quali.get('Q1')
                            if q3:
                                features['qualifying_time'] = self._convert_time_to_seconds(q3)
                            else:
                                features['qualifying_time'] = 999.0
                                partial_fields.add('qualifying_time')
                            features['grid_position'] = int(quali.get('position', 20))
                        else:
                            features['qualifying_time'] = 999.0
                            features['grid_position'] = 20
                            partial_fields.update(['qualifying_time', 'grid_position'])
                        # Derived features
                        features.update(self._calculate_derived_features(features))
                        driver_features.append(features)
                    features_df = pd.DataFrame(driver_features)
                    self.logger.info(f"Feature extraction complete from Jolpica. DataFrame shape: {features_df.shape}")
                    return features_df, r, partial_fields, note
            except Exception as e:
                self.logger.warning(f"Jolpica API failed for round {r}: {e}")
            # --- Fallbacks for pre-2024 only ---
            if season in (2024, 2025):
                return None, None, set(), note  # Do not fallback for 2024/2025
            self.logger.info("Loading qualifying session...")
            try:
                quali_session = fastf1.get_session(season, r, 'Q')
                quali_session.load()
                quali_results = quali_session.results[['Driver', 'Q3', 'Position', 'TeamName']].copy()
                self.logger.info(f"Qualifying data loaded for {len(quali_results)} drivers")
            except Exception as e:
                self.logger.warning(f"Could not load qualifying data for round {r}: {e}")
                quali_results = pd.DataFrame()
            self.logger.info("Loading race session...")
            try:
                race_session = fastf1.get_session(season, r, 'R')
                race_session.load()
                race_results = race_session.results[['Driver', 'Position', 'Points', 'Status', 'TeamName']].copy()
                self.logger.info(f"Race data loaded for {len(race_results)} drivers")
            except Exception as e:
                self.logger.warning(f"Could not load race data for round {r}: {e}")
                race_results = pd.DataFrame()
            all_drivers = set()
            if not quali_results.empty:
                all_drivers.update(quali_results['Driver'].unique())
            if not race_results.empty:
                all_drivers.update(race_results['Driver'].unique())
                if all_drivers:
                    self.logger.info(f"Using data from round {r} for prediction (FastF1 fallback)")
                    driver_features = []
            for driver in all_drivers:
                self.logger.info(f"Processing driver: {driver}")
                features = {'driver': driver}
                driver_missing = set()
                # Qualifying features
                if not quali_results.empty:
                    driver_quali = quali_results[quali_results['Driver'] == driver]
                    if not driver_quali.empty:
                        quali_row = driver_quali.iloc[0]
                        if pd.isna(quali_row['Position']):
                            features['grid_position'] = 20
                            driver_missing.add('grid_position')
                        else:
                            features['grid_position'] = quali_row['Position']
                        if pd.isna(quali_row['Q3']):
                            features['qualifying_time'] = 999.0
                            driver_missing.add('qualifying_time')
                        else:
                            features['qualifying_time'] = self._convert_time_to_seconds(quali_row['Q3'])
                        if pd.isna(quali_row['TeamName']):
                            features['team'] = 'Unknown'
                            driver_missing.add('team')
                        else:
                            features['team'] = quali_row['TeamName']
                    else:
                        features.update({'grid_position': 20, 'qualifying_time': 999.0, 'team': 'Unknown'})
                        driver_missing.update(['grid_position', 'qualifying_time', 'team'])
                else:
                    features.update({'grid_position': 20, 'qualifying_time': 999.0, 'team': 'Unknown'})
                    driver_missing.update(['grid_position', 'qualifying_time', 'team'])
                # Race features
                if not race_results.empty:
                    driver_race = race_results[race_results['Driver'] == driver]
                    if not driver_race.empty:
                        race_row = driver_race.iloc[0]
                        if pd.isna(race_row['Position']):
                            features['final_position'] = 20
                            driver_missing.add('final_position')
                        else:
                            features['final_position'] = race_row['Position']
                        if pd.isna(race_row['Points']):
                            features['points_scored'] = 0
                            driver_missing.add('points_scored')
                        else:
                            features['points_scored'] = race_row['Points']
                        if pd.isna(race_row['Status']):
                            features['race_status'] = 'Unknown'
                            driver_missing.add('race_status')
                        else:
                            features['race_status'] = race_row['Status']
                        if pd.isna(race_row['TeamName']):
                            features['team_race'] = 'Unknown'
                            driver_missing.add('team_race')
                        else:
                            features['team_race'] = race_row['TeamName']
                    else:
                        features.update({'final_position': 20, 'points_scored': 0, 'race_status': 'DNF', 'team_race': 'Unknown'})
                        driver_missing.update(['final_position', 'points_scored', 'race_status', 'team_race'])
                else:
                    features.update({'final_position': 20, 'points_scored': 0, 'race_status': 'Unknown', 'team_race': 'Unknown'})
                    driver_missing.update(['final_position', 'points_scored', 'race_status', 'team_race'])
                # Tire strategy and pit stop data
                try:
                    tire_data = self._extract_tire_strategy(race_session, driver)
                    features.update(tire_data)
                except Exception as e:
                    self.logger.warning(f"Could not extract tire data for {driver}: {e}")
                    features.update({'avg_stint_length': 20.0, 'total_pit_stops': 2, 'compound_usage': 'Medium-Hard', 'total_laps': 50})
                    driver_missing.update(['avg_stint_length', 'total_pit_stops', 'compound_usage', 'total_laps'])
                features.update(self._calculate_derived_features(features))
                driver_features.append(features)
                partial_fields.update(driver_missing)
            features_df = pd.DataFrame(driver_features)
            self.logger.info(f"Feature extraction complete. DataFrame shape: {features_df.shape}")
            # If no real data found, fallback to mock data for pre-2024 seasons
            if features_df.empty and season not in (2024, 2025):
                self.logger.warning(f"No real data found for {season} Rounds 1-{round_num}, using mock data")
                from utils.mock_data import create_mock_features
                note = "Prediction is based on mock data, as no real data is available for this season."
                return create_mock_features(), None, set(), note
            return features_df, r, partial_fields, note
        except Exception as e:
            self.logger.error(f"Error extracting race features: {e}")
            if season in (2024, 2025):
                note = f"Error extracting race features for {season}: {e}. No prediction possible."
                self._log_fallback_event(season, round_num, note)
                return None, None, set(), note
            else:
                self.logger.info("Falling back to mock data")
                from utils.mock_data import create_mock_features
                note = "Prediction is based on mock data, as no real data is available for this season."
                return create_mock_features(), None, set(), note
    
    def _convert_time_to_seconds(self, time_value) -> float:
        """Convert qualifying time to seconds"""
        try:
            if pd.isna(time_value):
                return 999.0
            
            if isinstance(time_value, str):
                # Handle string time format (e.g., "1:23.456")
                parts = time_value.split(':')
                if len(parts) == 2:
                    minutes = int(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
                else:
                    return float(time_value)
            
            # Handle timedelta objects
            if hasattr(time_value, 'total_seconds'):
                return time_value.total_seconds()
            
            return float(time_value)
            
        except Exception as e:
            self.logger.warning(f"Error converting time: {e}")
            return 999.0
    
    def _extract_tire_strategy(self, race_session, driver: str) -> Dict:
        """Extract tire strategy and pit stop information for a driver"""
        try:
            # Get driver's lap data
            driver_laps = race_session.laps.pick_driver(driver)
            
            if driver_laps.empty:
                return {
                    'avg_stint_length': 20.0,
                    'total_pit_stops': 2,
                    'compound_usage': 'Medium-Hard',
                    'total_laps': 50
                }
            
            # Extract pit stops
            pit_stops = driver_laps[driver_laps['PitOutTime'].notna()]
            total_pit_stops = len(pit_stops)
            
            # Calculate stint lengths
            stint_lengths = []
            current_stint = 0
            
            for _, lap in driver_laps.iterrows():
                if pd.isna(lap['PitOutTime']):
                    current_stint += 1
                else:
                    if current_stint > 0:
                        stint_lengths.append(current_stint)
                    current_stint = 0
            
            # Add final stint
            if current_stint > 0:
                stint_lengths.append(current_stint)
            
            avg_stint_length = np.mean(stint_lengths) if stint_lengths else 20.0
            
            # Extract compound usage
            compounds = driver_laps['Compound'].dropna().unique()
            compound_usage = '-'.join(compounds) if len(compounds) > 0 else 'Unknown'
            
            return {
                'avg_stint_length': avg_stint_length,
                'total_pit_stops': total_pit_stops,
                'compound_usage': compound_usage,
                'total_laps': len(driver_laps)
            }
            
        except Exception as e:
            self.logger.warning(f"Error extracting tire strategy for {driver}: {e}")
            return {
                'avg_stint_length': 20.0,
                'total_pit_stops': 2,
                'compound_usage': 'Medium-Hard',
                'total_laps': 50
            }
    
    def _calculate_derived_features(self, features: Dict) -> Dict:
        """Calculate additional derived features"""
        derived = {}
        
        # Qualifying performance
        if 'qualifying_time' in features and features['qualifying_time'] != 999.0:
            derived['qualifying_performance'] = 1.0  # Normalized score
        else:
            derived['qualifying_performance'] = 0.0
        
        # Grid position score (lower is better)
        grid_pos = features.get('grid_position', 20)
        derived['grid_position_score'] = max(0, (20 - grid_pos) / 20)
        
        # Team consistency
        team_quali = features.get('team', 'Unknown')
        team_race = features.get('team_race', 'Unknown')
        derived['team_consistency'] = 1.0 if team_quali == team_race else 0.5
        
        return derived 

    def _log_fallback_event(self, season, round_num, note):
        now = datetime.utcnow()
        FALLBACK_EVENTS.append(now)
        # Remove events outside the window
        while FALLBACK_EVENTS and FALLBACK_EVENTS[0] < now - FALLBACK_WINDOW:
            FALLBACK_EVENTS.popleft()
        if len(FALLBACK_EVENTS) >= FALLBACK_THRESHOLD:
            self.logger.critical(f"CRITICAL: Fallback to mock/no prediction for {season} round {round_num} occurred {len(FALLBACK_EVENTS)} times in the last {FALLBACK_WINDOW}.") 