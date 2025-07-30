import pandas as pd
import numpy as np

# --- Mock Features ---
def create_mock_features() -> pd.DataFrame:
    """Create mock features for testing when real data is unavailable"""
    # Realistic F1 2024 teams and drivers
    mock_drivers = [
        ('Max Verstappen', 'Red Bull Racing'),
        ('Sergio Perez', 'Red Bull Racing'),
        ('Lando Norris', 'McLaren'),
        ('Oscar Piastri', 'McLaren'),
        ('Charles Leclerc', 'Ferrari'),
        ('Carlos Sainz', 'Ferrari'),
        ('Lewis Hamilton', 'Mercedes'),
        ('George Russell', 'Mercedes'),
        ('Fernando Alonso', 'Aston Martin'),
        ('Lance Stroll', 'Aston Martin'),
        ('Valtteri Bottas', 'Kick Sauber'),
        ('Zhou Guanyu', 'Kick Sauber'),
        ('Daniel Ricciardo', 'RB'),
        ('Yuki Tsunoda', 'RB'),
        ('Alexander Albon', 'Williams'),
        ('Logan Sargeant', 'Williams'),
        ('Nico Hulkenberg', 'Haas F1 Team'),
        ('Kevin Magnussen', 'Haas F1 Team'),
        ('Esteban Ocon', 'Alpine'),
        ('Pierre Gasly', 'Alpine')
    ]
    mock_data = []
    for i, (driver, team) in enumerate(mock_drivers):
        quali_time = 80.0 + (i * 0.3) + np.random.normal(0, 0.1)
        grid_pos = i + 1 if i < 5 else min(20, i + 1 + np.random.randint(-2, 3))
        final_pos = max(1, grid_pos + np.random.randint(-3, 4))
        points = 0
        if final_pos == 1: points = 25
        elif final_pos == 2: points = 18
        elif final_pos == 3: points = 15
        elif final_pos == 4: points = 12
        elif final_pos == 5: points = 10
        elif final_pos == 6: points = 8
        elif final_pos == 7: points = 6
        elif final_pos == 8: points = 4
        elif final_pos == 9: points = 2
        elif final_pos == 10: points = 1
        compounds = ['Soft-Medium-Hard', 'Medium-Hard', 'Soft-Hard', 'Medium-Medium-Hard']
        compound_usage = compounds[np.random.randint(0, len(compounds))]
        pit_stops = 2 if 'Medium-Hard' in compound_usage else 3
        mock_data.append({
            'driver': driver,
            'grid_position': grid_pos,
            'qualifying_time': round(quali_time, 3),
            'team': team,
            'final_position': final_pos,
            'points_scored': points,
            'race_status': 'Finished',
            'team_race': team,
            'avg_stint_length': round(20.0 + np.random.normal(0, 2), 1),
            'total_pit_stops': pit_stops,
            'compound_usage': compound_usage,
            'total_laps': 50,
            'qualifying_performance': max(0, 1.0 - (i * 0.05)),
            'grid_position_score': max(0, (20 - grid_pos) / 20),
            'team_consistency': 1.0
        })
    return pd.DataFrame(mock_data)

# --- Mock Predictions ---
def create_mock_predictions(features: pd.DataFrame) -> list:
    """Generate mock predictions when model is not available"""
    predictions = []
    for i, (_, driver) in enumerate(features.iterrows()):
        grid_pos = driver.get('grid_position', 20)
        quali_perf = driver.get('qualifying_performance', 0.0)
        base_prob = max(0.01, (20 - grid_pos) / 20) * 0.8
        win_prob = base_prob + np.random.normal(0, 0.05)
        win_prob = max(0.01, min(0.95, win_prob))
        predictions.append({
            'name': driver['driver'],
            'team': driver.get('team', 'Unknown'),
            'grid_position': grid_pos,
            'win_probability': float(win_prob),
            'qualifying_time': driver.get('qualifying_time', 999.0),
            'qualifying_performance': quali_perf
        })
    predictions.sort(key=lambda x: x['win_probability'], reverse=True)
    total_prob = sum(p['win_probability'] for p in predictions)
    if total_prob > 0:
        for pred in predictions:
            pred['win_probability'] = pred['win_probability'] / total_prob
    return predictions 