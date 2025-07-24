import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

class F1DataPreprocessor:
    """
    Utility class for preprocessing F1 data and feature engineering
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def clean_qualifying_data(self, qualifying_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess qualifying data
        
        Args:
            qualifying_df: Raw qualifying results DataFrame
            
        Returns:
            Cleaned qualifying DataFrame
        """
        try:
            if qualifying_df.empty:
                return qualifying_df
            
            # Create a copy to avoid modifying original
            df = qualifying_df.copy()
            
            # Handle missing qualifying times
            df['Q3'] = pd.to_numeric(df['Q3'], errors='coerce')
            
            # Fill missing positions with a high value
            df['Position'] = df['Position'].fillna(20)
            
            # Convert qualifying time to seconds if it's a timedelta
            if df['Q3'].dtype == 'object':
                df['Q3'] = pd.to_timedelta(df['Q3']).dt.total_seconds()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error cleaning qualifying data: {e}")
            return qualifying_df
    
    def engineer_features(self, driver_data: Dict, qualifying_data: pd.DataFrame) -> Dict:
        """
        Engineer additional features from raw data
        
        Args:
            driver_data: Driver performance data
            qualifying_data: Qualifying results
            
        Returns:
            Dictionary with engineered features
        """
        try:
            features = {}
            
            # Qualifying performance features
            if not qualifying_data.empty:
                features['qualifying_gap_to_pole'] = self._calculate_qualifying_gap(qualifying_data)
                features['qualifying_consistency'] = self._calculate_qualifying_consistency(qualifying_data)
            
            # Driver form features
            if driver_data:
                features['recent_form_trend'] = self._calculate_form_trend(driver_data)
                features['points_scoring_consistency'] = self._calculate_points_consistency(driver_data)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error engineering features: {e}")
            return {}
    
    def _calculate_qualifying_gap(self, qualifying_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate qualifying gap to pole position for each driver"""
        try:
            if qualifying_data.empty:
                return {}
            
            # Get pole position time
            pole_time = qualifying_data['Q3'].min()
            
            gaps = {}
            for _, row in qualifying_data.iterrows():
                driver = row['Driver']
                q3_time = row['Q3']
                
                if pd.notna(q3_time) and pd.notna(pole_time):
                    gap = q3_time - pole_time
                    gaps[driver] = gap
                else:
                    gaps[driver] = 999.0  # Large gap for missing data
            
            return gaps
            
        except Exception as e:
            self.logger.error(f"Error calculating qualifying gap: {e}")
            return {}
    
    def _calculate_qualifying_consistency(self, qualifying_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate qualifying consistency (placeholder for future implementation)"""
        # TODO: Implement qualifying consistency calculation
        # This would analyze qualifying performance over multiple sessions
        return {}
    
    def _calculate_form_trend(self, driver_data: Dict) -> Dict[str, float]:
        """Calculate recent form trend for drivers"""
        try:
            trends = {}
            
            for driver, data in driver_data.items():
                if 'recent_results' in data and data['recent_results']:
                    positions = [r['position'] for r in data['recent_results']]
                    
                    if len(positions) >= 2:
                        # Calculate trend (negative = improving, positive = declining)
                        trend = np.polyfit(range(len(positions)), positions, 1)[0]
                        trends[driver] = trend
                    else:
                        trends[driver] = 0.0
                else:
                    trends[driver] = 0.0
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error calculating form trend: {e}")
            return {}
    
    def _calculate_points_consistency(self, driver_data: Dict) -> Dict[str, float]:
        """Calculate points scoring consistency for drivers"""
        try:
            consistency = {}
            
            for driver, data in driver_data.items():
                if 'recent_results' in data and data['recent_results']:
                    points_races = sum(1 for r in data['recent_results'] if r['points'] > 0)
                    total_races = len(data['recent_results'])
                    
                    if total_races > 0:
                        consistency[driver] = points_races / total_races
                    else:
                        consistency[driver] = 0.0
                else:
                    consistency[driver] = 0.0
            
            return consistency
            
        except Exception as e:
            self.logger.error(f"Error calculating points consistency: {e}")
            return {}
    
    def normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical features for better model performance
        
        Args:
            features: DataFrame with features to normalize
            
        Returns:
            DataFrame with normalized features
        """
        try:
            if features.empty:
                return features
            
            df = features.copy()
            
            # List of numerical columns to normalize
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_cols = [col for col in numerical_cols if col != 'driver']
            
            # Min-max normalization
            for col in numerical_cols:
                if df[col].std() > 0:
                    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error normalizing features: {e}")
            return features 