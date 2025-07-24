#!/usr/bin/env python3
"""
Test version of F1 Model Training Script

This script tests the training pipeline using mock data to verify everything works.
"""

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

# Add backend to path for imports
sys.path.append('backend')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_mock_training_data():
    """Create mock training data for testing"""
    logger.info("Creating mock training data")
    
    # Create realistic mock data
    np.random.seed(42)
    
    # Simulate 3 seasons with 20 races each
    all_data = []
    
    for season in range(2021, 2024):
        for round_num in range(1, 21):
            # Create 20 drivers per race
            for driver_idx in range(20):
                # Realistic features
                grid_position = np.random.randint(1, 21)
                qualifying_time = 80.0 + np.random.normal(0, 2)
                qualifying_performance = max(0, 1.0 - (grid_position - 1) * 0.05)
                grid_position_score = max(0, (20 - grid_position) / 20)
                team_consistency = np.random.uniform(0.8, 1.0)
                avg_stint_length = 20.0 + np.random.normal(0, 3)
                total_pit_stops = np.random.randint(1, 4)
                total_laps = 50 + np.random.randint(-5, 5)
                
                # Winner (1 for first driver, 0 for others)
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
    """Train model on mock data"""
    logger.info("Training model on mock data")
    
    # Create mock data
    data = create_mock_training_data()
    logger.info(f"Created {len(data)} training samples")
    logger.info(f"Winners: {data['winner'].sum()}")
    
    # Select features
    feature_columns = [
        'grid_position', 'qualifying_time', 'qualifying_performance',
        'grid_position_score', 'team_consistency', 'avg_stint_length',
        'total_pit_stops', 'total_laps'
    ]
    
    X = data[feature_columns]
    y = data['winner']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train model
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    logger.info(f"Model AUC: {auc:.4f}")
    logger.info(f"Feature Importance: {dict(zip(feature_columns, model.feature_importances_))}")
    
    return model, feature_columns, {'accuracy': accuracy, 'auc': auc}

def save_model(model, feature_names, test_results, filepath="backend/model/lgbm_model.pkl"):
    """Save the trained model"""
    logger.info(f"Saving model to {filepath}")
    
    # Create model directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare model data
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'training_info': {
            'total_samples': 1200,  # 3 seasons * 20 races * 20 drivers
            'winners': 60,  # 3 seasons * 20 races
            'test_results': test_results
        }
    }
    
    # Save model
    joblib.dump(model_data, filepath)
    logger.info("Model saved successfully!")

def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("F1 MODEL TRAINING TEST (MOCK DATA)")
    logger.info("=" * 60)
    
    try:
        # Train model on mock data
        model, feature_names, test_results = train_mock_model()
        
        # Save model
        save_model(model, feature_names, test_results)
        
        logger.info("=" * 60)
        logger.info("TEST TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test training failed: {e}")
        raise

if __name__ == "__main__":
    main() 