#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.append('backend')

from backend.services.f1_data_service import F1DataService
from backend.utils.preprocessing import F1DataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class F1ModelTrainer:
    
    def __init__(self):
        self.f1_service = F1DataService()
        self.preprocessor = F1DataPreprocessor()
        self.model = None
        self.feature_names = []
        self.training_data = pd.DataFrame()
        
    def get_race_schedule(self, season: int) -> List[Dict]:
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            schedule = loop.run_until_complete(self.f1_service.get_race_schedule(season))
            loop.close()
            return schedule
        except Exception as e:
            logger.warning(f"Could not get schedule for {season}: {e}")
            return []
    
    def extract_race_data(self, season: int, round_num: int) -> pd.DataFrame:
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            features_df = loop.run_until_complete(self.f1_service.get_race_features(season, round_num))
            loop.close()
            
            if features_df.empty:
                logger.warning(f"No data for {season} Round {round_num}")
                return pd.DataFrame()
            
            try:
                import fastf1
                race_session = fastf1.get_session(season, round_num, 'R')
                race_session.load()
                race_results = race_session.results[['Driver', 'Position']].copy()
                
                features_df['winner'] = 0
                if not race_results.empty:
                    winner = race_results[race_results['Position'] == 1]['Driver'].iloc[0]
                    features_df.loc[features_df['driver'] == winner, 'winner'] = 1
                    logger.info(f"Winner for {season} Round {round_num}: {winner}")
                
            except Exception as e:
                logger.warning(f"Could not get race results for {season} Round {round_num}: {e}")
                features_df['winner'] = 0
                features_df.loc[0, 'winner'] = 1
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error extracting data for {season} Round {round_num}: {e}")
            return pd.DataFrame()
    
    def collect_training_data(self, start_season: int = 2021, end_season: int = 2023) -> pd.DataFrame:
        logger.info(f"Collecting training data from {start_season} to {end_season}")
        
        all_races_data = []
        total_races = 0
        successful_races = 0
        
        for season in range(start_season, end_season + 1):
            logger.info(f"Processing season {season}")
            
            schedule = self.get_race_schedule(season)
            if not schedule:
                logger.warning(f"No schedule found for {season}, using default races")
                schedule = [{'round': i, 'name': f'Race {i}'} for i in range(1, 23)]
            
            for race in schedule:
                round_num = race['round']
                total_races += 1
                
                logger.info(f"Processing {season} Round {round_num}: {race.get('name', 'Unknown')}")
                
                race_data = self.extract_race_data(season, round_num)
                
                if not race_data.empty:
                    race_data['season'] = season
                    race_data['round'] = round_num
                    all_races_data.append(race_data)
                    successful_races += 1
                    logger.info(f"✓ Successfully extracted data for {season} Round {round_num}")
                else:
                    logger.warning(f"✗ No data for {season} Round {round_num}")
        
        if not all_races_data:
            logger.error("No training data collected!")
            return pd.DataFrame()
        
        combined_data = pd.concat(all_races_data, ignore_index=True)
        logger.info(f"Collected data from {successful_races}/{total_races} races")
        logger.info(f"Total samples: {len(combined_data)}")
        logger.info(f"Winners: {combined_data['winner'].sum()}")
        
        return combined_data
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        logger.info("Preparing features for training")
        
        feature_columns = [
            'grid_position', 'qualifying_time', 'qualifying_performance',
            'grid_position_score', 'team_consistency', 'avg_stint_length',
            'total_pit_stops', 'total_laps',
            'air_temp', 'track_temp', 'rainfall', 'wind_speed', 'humidity'
        ]
        
        available_features = [col for col in feature_columns if col in data.columns]
        logger.info(f"Using features: {available_features}")
        
        X = data[available_features].copy()
        
        X = X.fillna(X.median())
        
        X = self.preprocessor.normalize_features(X)
        
        return X, available_features
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> lgb.LGBMClassifier:
        logger.info("Training LightGBM model")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Training winners: {y_train.sum()}")
        logger.info(f"Test winners: {y_test.sum()}")
        
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            random_state=42,
            verbose=-1,
            class_weight='balanced'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"Model Accuracy: {accuracy:.4f}")
        logger.info(f"Model AUC: {auc:.4f}")
        logger.info(f"Feature Importance: {dict(zip(feature_names, model.feature_importances_))}")
        
        self.test_results = {
            'accuracy': accuracy,
            'auc': auc,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return model
    
    def save_model(self, model: lgb.LGBMClassifier, feature_names: List[str], filepath: str = "backend/model/lgbm_model.pkl"):
        logger.info(f"Saving model to {filepath}")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': model,
            'feature_names': feature_names,
            'training_info': {
                'total_samples': len(self.training_data),
                'winners': self.training_data['winner'].sum(),
                'test_results': self.test_results
            }
        }
        
        joblib.dump(model_data, filepath)
        logger.info("Model saved successfully!")
    
    def run_training(self, start_season: int = 2021, end_season: int = 2023):
        logger.info("Starting F1 Model Training Pipeline")
        
        try:
            self.training_data = self.collect_training_data(start_season, end_season)
            
            if self.training_data.empty:
                logger.error("No training data available!")
                return
            
            X, feature_names = self.prepare_features(self.training_data)
            y = self.training_data['winner']
            
            logger.info(f"Feature matrix shape: {X.shape}")
            logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            model = self.train_model(X, y, feature_names)
            
            self.save_model(model, feature_names)
            
            logger.info("Training completed successfully!")
            logger.info(f"Model saved with {len(feature_names)} features")
            logger.info(f"Test accuracy: {self.test_results['accuracy']:.4f}")
            logger.info(f"Test AUC: {self.test_results['auc']:.4f}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

def main():
    logger.info("=" * 60)
    logger.info("F1 GRAND PRIX WINNER PREDICTION MODEL TRAINING")
    logger.info("=" * 60)
    
    trainer = F1ModelTrainer()
    
    trainer.run_training(start_season=2021, end_season=2023)
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 