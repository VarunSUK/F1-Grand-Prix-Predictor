#!/usr/bin/env python3
"""
Feature Matrix Builder for F1 Race Prediction

This module extracts features from FastF1 qualifying and race sessions
to create a feature matrix suitable for machine learning models.
"""

import pandas as pd
import numpy as np
import fastf1
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sessions(season: int, round_num: int) -> Tuple[fastf1.core.Session, fastf1.core.Session]:
    """
    Load qualifying and race sessions for the given season and round
    
    Args:
        season: F1 season year
        round_num: Race round number
        
    Returns:
        Tuple of (qualifying_session, race_session)
    """
    logger.info(f"Loading sessions for {season} Round {round_num}")
    
    try:
        # Load qualifying session
        quali_session = fastf1.get_session(season, round_num, 'Q')
        quali_session.load()
        logger.info(f"✓ Qualifying session loaded: {quali_session.event['EventName']}")
        
        # Load race session
        race_session = fastf1.get_session(season, round_num, 'R')
        race_session.load()
        logger.info(f"✓ Race session loaded: {race_session.event['EventName']}")
        
        return quali_session, race_session
        
    except Exception as e:
        logger.error(f"Error loading sessions: {e}")
        raise

def extract_qualifying_features(quali_session: fastf1.core.Session) -> pd.DataFrame:
    """
    Extract qualifying features for all drivers
    
    Args:
        quali_session: Loaded qualifying session
        
    Returns:
        DataFrame with qualifying features per driver
    """
    logger.info("Extracting qualifying features")
    
    try:
        # Get qualifying results - check available columns
        logger.info(f"Available columns in qualifying results: {quali_session.results.columns.tolist()}")
        # --- Log new/unexpected columns ---
        expected_cols = {'Driver', 'Abbreviation', 'DriverNumber', 'TeamName', 'Team', 'Q3', 'Q3Time', 'Q2', 'Q1', 'Position'}
        for col in quali_session.results.columns:
            if col not in expected_cols:
                logger.info(f"[Jolpica] New or unexpected qualifying column: {col}")
        
        # Robust driver and team column selection
        driver_col = None
        for col in ['Driver', 'Abbreviation', 'DriverNumber']:
            if col in quali_session.results.columns:
                driver_col = col
                break
        if driver_col is None:
            logger.error(f"No driver column found in qualifying results. Available columns: {quali_session.results.columns.tolist()}")
            return pd.DataFrame()
        team_col = 'TeamName' if 'TeamName' in quali_session.results.columns else ('Team' if 'Team' in quali_session.results.columns else None)
        q3_col = 'Q3' if 'Q3' in quali_session.results.columns else ('Q3Time' if 'Q3Time' in quali_session.results.columns else None)
        if team_col is None or q3_col is None:
            logger.error(f"Missing team or Q3 column in qualifying results. Available columns: {quali_session.results.columns.tolist()}")
            return pd.DataFrame()
        
        # Get qualifying results
        quali_results = quali_session.results[[driver_col, team_col, q3_col]].copy()
        quali_results = quali_results.dropna(subset=[q3_col])  # Remove drivers without Q3 time
        
        logger.info(f"Qualifying results shape: {quali_results.shape}")
        logger.info(f"Sample qualifying results:\n{quali_results.head()}")
        
        # Calculate qualifying features
        features = []
        
        for _, driver_result in quali_results.iterrows():
            driver_code = driver_result[driver_col]
            team = driver_result[team_col]
            q3_time = driver_result[q3_col]
            
            # Get qualifying rank (position based on Q3 time)
            quali_rank = quali_results[quali_results[q3_col] <= q3_time].shape[0]
            
            # Convert timedelta to seconds
            if hasattr(q3_time, 'total_seconds'):
                q3_time_seconds = q3_time.total_seconds()
            else:
                q3_time_seconds = float(q3_time) / 1e9  # Convert nanoseconds to seconds
            
            # Calculate gap to pole
            pole_time = quali_results[q3_col].min()
            if hasattr(pole_time, 'total_seconds'):
                pole_time_seconds = pole_time.total_seconds()
            else:
                pole_time_seconds = float(pole_time) / 1e9  # Convert nanoseconds to seconds
            
            gap_to_pole = q3_time_seconds - pole_time_seconds if q3_time_seconds > pole_time_seconds else 0.0
            
            features.append({
                'driver': driver_code,
                'team': team,
                'qualifying_time': q3_time_seconds,
                'qualifying_rank': quali_rank,
                'qualifying_gap_to_pole': gap_to_pole
            })
        
        quali_features = pd.DataFrame(features)
        logger.info(f"✓ Extracted qualifying features for {len(quali_features)} drivers")
        return quali_features
        
    except Exception as e:
        logger.error(f"Error extracting qualifying features: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def extract_weather_at_start(race_session):
    weather_df = race_session.weather_data
    race_start = getattr(race_session, 'session_start_time', None)
    if race_start is None:
        logger.warning("Race session has no session_start_time; using first weather row.")

    # Find first valid weather snapshot after race start
    weather_row = None
    if not weather_df.empty:
        if race_start is not None:
            after_start = weather_df[weather_df['Time'] >= race_start]
            if not after_start.empty:
                weather_row = after_start.iloc[0]
            else:
                weather_row = weather_df.iloc[0]  # fallback: first available
        else:
            weather_row = weather_df.iloc[0]  # fallback: first available

    # Defaults
    weather = {
        'air_temp': 25.0,
        'track_temp': 30.0,
        'rainfall': 0.0,
        'wind_speed': 5.0,
        'humidity': 50.0,
        'weather_code': 'dry',
        'air_temp_missing': False,
        'track_temp_missing': False,
        'rainfall_missing': False,
        'wind_speed_missing': False,
        'humidity_missing': False,
        'weather_code_missing': False,
    }

    if weather_row is not None:
        for key, col, default in [
            ('AirTemp', 'air_temp', 25.0),
            ('TrackTemp', 'track_temp', 30.0),
            ('Rainfall', 'rainfall', 0.0),
            ('WindSpeed', 'wind_speed', 5.0),
            ('Humidity', 'humidity', 50.0),
            ('Weather', 'weather_code', 'dry')
        ]:
            value = weather_row.get(key, None)
            if value is not None and not pd.isna(value):
                weather[col] = value
            else:
                weather[f'{col}_missing'] = True

    logger.info(f"Weather at race start: {weather}")
    return weather

def extract_race_features(race_session: fastf1.core.Session) -> pd.DataFrame:
    """
    Extract race features for all drivers
    
    Args:
        race_session: Loaded race session
        
    Returns:
        DataFrame with race features per driver
    """
    logger.info("Extracting race features")
    
    try:
        # Get race results for grid positions - check available columns
        logger.info(f"Available columns in race results: {race_session.results.columns.tolist()}")
        # --- Log new/unexpected columns ---
        expected_cols = {'Driver', 'Abbreviation', 'DriverNumber', 'GridPosition', 'Grid', 'Position', 'TeamName', 'Team', 'Status', 'Points'}
        for col in race_session.results.columns:
            if col not in expected_cols:
                logger.info(f"[Jolpica] New or unexpected race column: {col}")
        
        # Robust driver column selection
        driver_col = None
        for col in ['Driver', 'Abbreviation', 'DriverNumber']:
            if col in race_session.results.columns:
                driver_col = col
                break
        grid_col = 'GridPosition' if 'GridPosition' in race_session.results.columns else ('Grid' if 'Grid' in race_session.results.columns else None)
        if driver_col is None or grid_col is None:
            logger.error(f"No driver or grid column found in race results. Available columns: {race_session.results.columns.tolist()}")
            return pd.DataFrame()
        
        # Get race results for grid positions
        race_results = race_session.results[[driver_col, grid_col]].copy()
        
        # Get weather data (enhanced)
        weather = extract_weather_at_start(race_session)
        features = []
        
        for _, driver_result in race_results.iterrows():
            driver_code = driver_result[driver_col]
            grid_position = driver_result[grid_col]
            
            # Get driver's lap data
            try:
                driver_laps = race_session.laps.pick_driver(driver_code)
                
                if not driver_laps.empty:
                    # Pit stop analysis
                    pit_stops = driver_laps[driver_laps['PitOutTime'].notna()]
                    num_pit_stops = len(pit_stops)
                    
                    # Stint analysis
                    stints = driver_laps.groupby('Stint')
                    avg_stint_length = stints.size().mean() if len(stints) > 0 else 0
                    
                    # Tire compound analysis
                    compounds_used = driver_laps['Compound'].unique()
                    compound_mix = list(compounds_used) if len(compounds_used) > 0 else []
                    
                    # Starting compound (first lap)
                    starting_compound = driver_laps.iloc[0]['Compound'] if not driver_laps.empty else None
                    
                    # Pit stop duration analysis
                    pit_durations = []
                    for _, pit_stop in pit_stops.iterrows():
                        pit_in_time = pit_stop['PitInTime']
                        pit_out_time = pit_stop['PitOutTime']
                        if pd.notna(pit_in_time) and pd.notna(pit_out_time):
                            duration = (pit_out_time - pit_in_time).total_seconds()
                            # Only include positive durations (valid pit stops)
                            if duration > 0:
                                pit_durations.append(duration)
                    
                    total_pit_duration = sum(pit_durations) if pit_durations else 0.0
                    
                else:
                    # Driver has no lap data (DNF, DNS, etc.)
                    num_pit_stops = 0
                    avg_stint_length = 0
                    compound_mix = []
                    starting_compound = None
                    total_pit_duration = 0.0
                    
            except Exception as e:
                logger.warning(f"Error processing laps for {driver_code}: {e}")
                num_pit_stops = 0
                avg_stint_length = 0
                compound_mix = []
                starting_compound = None
                total_pit_duration = 0.0
            
            features.append({
                'driver': driver_code,
                'grid_position': grid_position,
                'num_pit_stops': num_pit_stops,
                'avg_stint_length': avg_stint_length,
                'starting_compound': starting_compound,
                'compound_mix': compound_mix,
                'total_pit_duration': total_pit_duration,
                # Weather features
                'air_temp': weather['air_temp'],
                'track_temp': weather['track_temp'],
                'rainfall': weather['rainfall'],
                'wind_speed': weather['wind_speed'],
                'humidity': weather['humidity'],
                'weather_code': weather['weather_code'],
                # Optionally, add *_missing flags
                'air_temp_missing': weather['air_temp_missing'],
                'track_temp_missing': weather['track_temp_missing'],
                'rainfall_missing': weather['rainfall_missing'],
                'wind_speed_missing': weather['wind_speed_missing'],
                'humidity_missing': weather['humidity_missing'],
                'weather_code_missing': weather['weather_code_missing'],
            })
        
        race_features = pd.DataFrame(features)
        logger.info(f"✓ Extracted race features for {len(race_features)} drivers")
        return race_features
        
    except Exception as e:
        logger.error(f"Error extracting race features: {e}")
        return pd.DataFrame()

def merge_features(quali_features: pd.DataFrame, race_features: pd.DataFrame) -> pd.DataFrame:
    """
    Merge qualifying and race features into a single feature matrix
    
    Args:
        quali_features: DataFrame with qualifying features
        race_features: DataFrame with race features
        
    Returns:
        Merged feature matrix
    """
    logger.info("Merging qualifying and race features")
    
    try:
        # Merge on driver code
        feature_matrix = pd.merge(
            quali_features, 
            race_features, 
            on='driver', 
            how='outer'
        )
        
        # Handle missing values
        feature_matrix = feature_matrix.fillna({
            'qualifying_time': np.nan,
            'qualifying_rank': np.nan,
            'qualifying_gap_to_pole': np.nan,
            'grid_position': np.nan,
            'num_pit_stops': 0,
            'avg_stint_length': 0,
            'starting_compound': 'Unknown',
            'compound_mix': '[]',
            'total_pit_duration': 0.0,
            'air_temp': 25.0,
            'track_temp': 30.0,
            'rainfall': 0.0,
            'wind_speed': 5.0,
            'humidity': 50.0,
            'weather_code': 'dry',
            'air_temp_missing': False,
            'track_temp_missing': False,
            'rainfall_missing': False,
            'wind_speed_missing': False,
            'humidity_missing': False,
            'weather_code_missing': False,
        })
        
        # Convert compound_mix to string for ML compatibility
        feature_matrix['compound_mix'] = feature_matrix['compound_mix'].apply(
            lambda x: str(x) if isinstance(x, list) else '[]'
        )
        
        # Ensure numeric columns are numeric
        numeric_columns = [
            'qualifying_time', 'qualifying_rank', 'qualifying_gap_to_pole',
            'grid_position', 'num_pit_stops', 'avg_stint_length',
            'total_pit_duration', 'air_temp', 'track_temp', 'rainfall',
            'wind_speed', 'humidity'
        ]
        
        for col in numeric_columns:
            if col in feature_matrix.columns:
                feature_matrix[col] = pd.to_numeric(feature_matrix[col], errors='coerce')
        
        logger.info(f"✓ Merged feature matrix shape: {feature_matrix.shape}")
        return feature_matrix
        
    except Exception as e:
        logger.error(f"Error merging features: {e}")
        return pd.DataFrame()

def build_feature_matrix(season: int, round_num: int) -> pd.DataFrame:
    """
    Build a complete feature matrix for the given season and round
    
    Args:
        season: F1 season year
        round_num: Race round number
        
    Returns:
        DataFrame with features for all drivers
    """
    logger.info(f"Building feature matrix for {season} Round {round_num}")
    
    try:
        # Step 1: Load sessions
        quali_session, race_session = load_sessions(season, round_num)
        
        # Step 2: Extract qualifying features
        quali_features = extract_qualifying_features(quali_session)
        if quali_features.empty:
            logger.error("Failed to extract qualifying features")
            return pd.DataFrame()
        
        # Step 3: Extract race features
        race_features = extract_race_features(race_session)
        if race_features.empty:
            logger.error("Failed to extract race features")
            return pd.DataFrame()
        
        # Step 4: Merge features
        feature_matrix = merge_features(quali_features, race_features)
        
        if feature_matrix.empty:
            logger.error("Failed to create feature matrix")
            return pd.DataFrame()
        
        logger.info(f"✓ Feature matrix built successfully")
        logger.info(f"  - Drivers: {len(feature_matrix)}")
        logger.info(f"  - Features: {len(feature_matrix.columns)}")
        logger.info(f"  - Columns: {list(feature_matrix.columns)}")
        
        return feature_matrix
        
    except Exception as e:
        logger.error(f"Error building feature matrix: {e}")
        return pd.DataFrame()

def main():
    """Test the feature matrix builder"""
    logger.info("=" * 60)
    logger.info("FEATURE MATRIX BUILDER TEST")
    logger.info("=" * 60)
    
    # Test with 2023 Round 3 (Australian Grand Prix)
    season = 2023
    round_num = 3
    
    try:
        # Build feature matrix
        feature_matrix = build_feature_matrix(season, round_num)
        
        if not feature_matrix.empty:
            logger.info("\nFeature Matrix Preview:")
            logger.info("=" * 40)
            print(feature_matrix.head())
            
            logger.info("\nFeature Matrix Info:")
            logger.info("=" * 40)
            print(feature_matrix.info())
            
            logger.info("\nFeature Matrix Description:")
            logger.info("=" * 40)
            print(feature_matrix.describe())
            
            logger.info("\n" + "=" * 60)
            logger.info("TEST COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
        else:
            logger.error("Failed to build feature matrix")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 