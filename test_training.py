#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append('backend')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_mock_training_data():
    logger.info("Creating mock training data")
    
    np.random.seed(42)
    
    all_data = []
    
    for season in range(2021, 2024):
        for round_num in range(1, 21):
            for driver_idx in range(20):
                grid_position = np.random.randint(1, 21)
                qualifying_time = 80.0 + np.random.normal(0, 2)
                qualifying_performance = max(0, 1.0 - (grid_position - 1) * 0.05)
                grid_position_score = max(0, (20 - grid_position) / 20)
                team_consistency = np.random.uniform(0.8, 1.0)
                avg_stint_length = 20.0 + np.random.normal(0, 3)
                total_pit_stops = np.random.randint(1, 4)
                total_laps = 50 + np.random.randint(-5, 5)
                
                winner = 1 if driver_idx == 0 else 0
                
                all_data.append({
                    'driver': f'Driver_{driver_idx}',
                    'season': season,
                    'round': round_num,
                    'grid_position': grid_position,
                    'qualifying_time': qualifying_time,
                    'qualifying_performance': qualifying_performance,
                    'grid_position_score': grid_position_score,
                    'team_consistency': team_consistency,
                    'avg_stint_length': avg_stint_length,
                    'total_pit_stops': total_pit_stops,
                    'total_laps': total_laps,
                    'winner': winner
                })
    
    return pd.DataFrame(all_data)

def train_mock_model():
    logger.info("Training model on mock data")
    
    data = create_mock_training_data()
    logger.info(f"Created {len(data)} training samples")
    logger.info(f"Winners: {data['winner'].sum()}")
    
    feature_columns = [
        'grid_position', 'qualifying_time', 'qualifying_performance',
        'grid_position_score', 'team_consistency', 'avg_stint_length',
        'total_pit_stops', 'total_laps'
    ]
    
    X = data[feature_columns]
    y = data['winner']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    logger.info(f"Model AUC: {auc:.4f}")
    logger.info(f"Feature Importance: {dict(zip(feature_columns, model.feature_importances_))}")
    
    return model, feature_columns, {'accuracy': accuracy, 'auc': auc}

def save_model(model, feature_names, test_results, filepath="backend/model/lgbm_model.pkl"):
    logger.info(f"Saving model to {filepath}")
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'training_info': {
            'total_samples': 1200,
            'winners': 60,
            'test_results': test_results
        }
    }
    
    joblib.dump(model_data, filepath)
    logger.info("Model saved successfully!")

def main():
    logger.info("=" * 60)
    logger.info("F1 MODEL TRAINING TEST (MOCK DATA)")
    logger.info("=" * 60)
    
    try:
        model, feature_names, test_results = train_mock_model()
        
        save_model(model, feature_names, test_results)
        
        logger.info("=" * 60)
        logger.info("TEST TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test training failed: {e}")
        raise

if __name__ == "__main__":
    main() 