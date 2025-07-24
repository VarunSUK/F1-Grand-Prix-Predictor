import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import joblib
import logging
import os
from pathlib import Path
from utils.mock_data import create_mock_predictions

# Move all non-optional imports to the top of the file.
# Add comments for clarity where fallback to mock predictions occurs.

class F1Predictor:
    """
    LightGBM-based F1 race winner predictor
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_names = []
        self.logger = logging.getLogger(__name__)
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def prepare_features(self, driver_data: Dict, qualifying_data: pd.DataFrame, 
                        team_data: Dict, track_data: Dict) -> pd.DataFrame:
        """
        Prepare feature vector for prediction
        
        Args:
            driver_data: Driver performance metrics
            qualifying_data: Qualifying results DataFrame
            team_data: Team performance metrics
            track_data: Track-specific statistics
            
        Returns:
            DataFrame with features for all drivers
        """
        try:
            features = []
            
            # Get all drivers from qualifying data
            drivers = qualifying_data['Driver'].unique()
            
            for driver in drivers:
                driver_features = {
                    'driver': driver,
                    'qualifying_position': qualifying_data[qualifying_data['Driver'] == driver]['Position'].iloc[0],
                    'qualifying_time': qualifying_data[qualifying_data['Driver'] == driver]['Q3'].iloc[0],
                }
                
                # Add driver performance features
                if driver in driver_data:
                    driver_features.update({
                        'avg_position': driver_data[driver].get('avg_position', 10.0),
                        'total_points': driver_data[driver].get('total_points', 0),
                        'races_analyzed': driver_data[driver].get('races_analyzed', 0)
                    })
                
                # Add team performance features
                # TODO: Map driver to team and add team features
                driver_features.update({
                    'team_performance': team_data.get('avg_team_position', 10.0),
                    'team_reliability': team_data.get('reliability_score', 0.9)
                })
                
                # Add track features
                driver_features.update({
                    'track_type': track_data.get('track_type', 'unknown'),
                    'avg_pit_stops': track_data.get('avg_pit_stops', 2.0),
                    'overtaking_difficulty': track_data.get('overtaking_difficulty', 'medium')
                })
                
                features.append(driver_features)
            
            return pd.DataFrame(features)
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def train_model(self, training_data: pd.DataFrame, target_column: str = 'winner'):
        """
        Train the LightGBM model
        
        Args:
            training_data: DataFrame with features and target
            target_column: Name of the target column
        """
        try:
            # Prepare features and target
            X = training_data.drop([target_column, 'driver'], axis=1, errors='ignore')
            y = training_data[target_column]
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Create and train LightGBM model
            self.model = lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
            
            self.model.fit(X, y)
            self.logger.info("Model trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
    
    def predict_winner(self, features: pd.DataFrame) -> Dict:
        """
        Predict race winner and probabilities
        
        Args:
            features: DataFrame with features for all drivers
            
        Returns:
            Dictionary with predictions
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")
            
            # Prepare features for prediction
            X = features.drop(['driver'], axis=1, errors='ignore')
            
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(X.columns)
            for feature in missing_features:
                X[feature] = 0  # Default value for missing features
            
            # Reorder columns to match training data
            X = X[self.feature_names]
            
            # Get predictions
            probabilities = self.model.predict_proba(X)
            predictions = self.model.predict(X)
            
            # Create results
            results = []
            for i, driver in enumerate(features['driver']):
                results.append({
                    'driver': driver,
                    'win_probability': probabilities[i][1] if len(probabilities[i]) > 1 else probabilities[i][0],
                    'predicted_winner': bool(predictions[i])
                })
            
            # Sort by win probability
            results.sort(key=lambda x: x['win_probability'], reverse=True)
            
            # Get top predictions
            predicted_winner = results[0]['driver']
            confidence = results[0]['win_probability']
            
            # Create podium predictions
            podium = results[:3]
            
            return {
                'predicted_winner': predicted_winner,
                'confidence': confidence,
                'podium_predictions': [
                    {
                        'position': i + 1,
                        'driver': result['driver'],
                        'probability': result['win_probability']
                    }
                    for i, result in enumerate(podium)
                ],
                'all_predictions': results
            }
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            return {}
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model"""
        try:
            if self.model is None:
                return {}
            
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance))
            
            # Sort by importance
            return dict(sorted(feature_importance.items(), 
                             key=lambda x: x[1], reverse=True))
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}


class RacePredictor:
    """
    LightGBM-based race winner predictor with model loading and prediction capabilities
    If LightGBM is not available, always uses mock predictions.
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.feature_names = []
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.lightgbm_available = True # Changed to True as per original file
        self.model_loaded = False
        if self.lightgbm_available:
            self._load_model()
        else:
            self.logger.warning("LightGBM not available, will always use mock predictions.")
    
    def _load_model(self):
        """Load the trained LightGBM model from disk"""
        if not self.lightgbm_available:
            self.model_loaded = False
            return
        try:
            model_file = Path(self.model_path)
            if model_file.exists():
                self.logger.info(f"Loading model from {self.model_path}")
                model_data = joblib.load(self.model_path)
                if isinstance(model_data, dict):
                    self.model = model_data.get('model')
                    self.feature_names = model_data.get('feature_names', [])
                else:
                    self.model = model_data
                self.model_loaded = self.model is not None
                if self.model_loaded:
                    self.logger.info("Model loaded successfully")
                else:
                    self.logger.error("Model file found but model is None!")
            else:
                self.logger.error(f"Model file not found at {self.model_path}")
                self.model = None
                self.model_loaded = False
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.model = None
            self.model_loaded = False
    
    def reload_model(self):
        """Reload the model from disk (for hot-reloading)"""
        self._load_model()
    
    def predict(self, features: pd.DataFrame) -> List[Dict]:
        """
        Predict race winner probabilities for all drivers
        
        Args:
            features: DataFrame with features for all drivers
            
        Returns:
            List of dictionaries with driver predictions, sorted by win probability
        """
        if not self.lightgbm_available or self.model is None or not self.model_loaded:
            self.logger.warning("No model loaded or LightGBM unavailable, using mock predictions")
            return self._mock_predictions(features)
        try:
            # Prepare features for prediction
            X = self._prepare_features(features)
            
            # Get predictions
            probabilities = self.model.predict_proba(X)
            
            # Create predictions list
            predictions = []
            for i, driver in enumerate(features['driver']):
                # Get win probability (second column for binary classification)
                win_prob = probabilities[i][1] if len(probabilities[i]) > 1 else probabilities[i][0]
                driver_info = features[features['driver'] == driver].iloc[0]
                team = driver_info.get('team', None)
                if not team or team == 'Unknown':
                    self.logger.warning(f"Missing team info for driver {driver}")
                    team = 'Unknown'
                predictions.append({
                    'driver': driver,  # Use 'driver' key for code/name
                    'team': team,
                    'win_probability': float(win_prob),
                    'grid_position': driver_info.get('grid_position', 20),
                    'qualifying_time': driver_info.get('qualifying_time', 999.0),
                    'qualifying_performance': driver_info.get('qualifying_performance', 0.0)
                })
            
            # Sort by win probability (descending)
            predictions.sort(key=lambda x: x['win_probability'], reverse=True)
            
            # Normalize probabilities to sum to 1
            total_prob = sum(p['win_probability'] for p in predictions)
            if total_prob > 0:
                for pred in predictions:
                    pred['win_probability'] = pred['win_probability'] / total_prob
            
            self.logger.info(f"Predictions generated for {len(predictions)} drivers")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return self._mock_predictions(features)
    
    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model prediction"""
        try:
            # Select numerical features (exclude driver name and other non-numeric columns)
            numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'driver']
            
            # If we have trained model features, use only those
            if self.feature_names:
                self.logger.info(f"Model expects features: {self.feature_names}")
                self.logger.info(f"Available features: {numeric_cols}")
                
                # Ensure all required features are present
                missing_features = set(self.feature_names) - set(numeric_cols)
                for feature in missing_features:
                    features[feature] = 0  # Default value for missing features
                    self.logger.warning(f"Missing feature '{feature}', using default value 0")
                
                # Select only the features the model was trained on
                X = features[self.feature_names]
            else:
                # Fallback to all numeric features
                X = features[numeric_cols]
            
            self.logger.info(f"Prepared feature matrix shape: {X.shape}")
            return X
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return features.select_dtypes(include=[np.number]).drop('driver', axis=1, errors='ignore')
    
    def _mock_predictions(self, features: pd.DataFrame) -> List[Dict]:
        """Generate mock predictions when model is not available"""
        self.logger.info("Generating mock predictions (via utils.mock_data)")
        return create_mock_predictions(features)
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_loaded': self.lightgbm_available and self.model is not None and self.model_loaded,
            'model_path': self.model_path,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names[:10] if self.feature_names else []  # Show first 10
        } 