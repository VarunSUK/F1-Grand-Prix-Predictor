import pytest
import pandas as pd
import numpy as np
import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.f1_data_service import F1DataService

# Helper to run async functions in sync tests
def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

def test_get_race_features_mock_fallback():
    service = F1DataService()
    # Use an obviously invalid season/round to force fallback
    features_df, used_round, partial_fields, note = run_async(service.get_race_features(1900, 99))
    assert isinstance(features_df, pd.DataFrame)
    assert not features_df.empty
    assert 'driver' in features_df.columns
    # Should contain mock drivers
    assert any('Verstappen' in d for d in features_df['driver'])
    # Optionally check the note
    assert note is not None and 'mock data' in note

def test_get_race_schedule_returns_list():
    service = F1DataService()
    # Use a recent season, should return a list (may be empty if FastF1 fails)
    races = run_async(service.get_race_schedule(2024))
    assert isinstance(races, list)
    # If FastF1 fails, should return an empty list, not raise

def test_get_qualifying_results_handles_error():
    service = F1DataService()
    # Use an invalid round to force error
    df = run_async(service.get_qualifying_results(1900, 99))
    assert isinstance(df, pd.DataFrame)
    # Should be empty DataFrame on error
    assert df.empty

def test_get_driver_performance_handles_error():
    service = F1DataService()
    # Use invalid driver/season
    perf = run_async(service.get_driver_performance('NotADriver', 1900))
    assert isinstance(perf, dict)
    # Should be empty dict on error
    assert perf == {} or 'driver' in perf

def test_get_team_performance_returns_dict():
    service = F1DataService()
    perf = run_async(service.get_team_performance('NotATeam', 1900))
    assert isinstance(perf, dict)
    assert 'team' in perf
    assert 'avg_team_position' in perf
    assert 'total_team_points' in perf
    assert 'reliability_score' in perf

def test_get_track_statistics_returns_dict():
    service = F1DataService()
    stats = run_async(service.get_track_statistics('NotACircuit', 1900))
    assert isinstance(stats, dict)
    assert 'circuit' in stats
    assert 'track_type' in stats
    assert 'avg_pit_stops' in stats
    assert 'overtaking_difficulty' in stats 

def test_jolpica_status_enumeration_2025():
    import requests
    import pytest
    url = "https://api.jolpi.ca/ergast/f1/2025/1/results.json"
    resp = requests.get(url, timeout=10)
    if resp.status_code == 429:
        pytest.skip("Jolpica API rate limit exceeded (429). Skipping test.")
    data = resp.json()
    results = data.get('MRData', {}).get('RaceTable', {}).get('Races', [{}])[0].get('Results', [])
    allowed_statuses = {"Finished", "Disqualified", "Accident", "Retired", "Lapped"}
    for res in results:
        status = res.get('status')
        assert status in allowed_statuses, f"Unexpected status: {status}"

def test_jolpica_time_decimal_places():
    import requests
    import pytest
    url = "https://api.jolpi.ca/ergast/f1/2024/1/results.json"
    resp = requests.get(url, timeout=10)
    if resp.status_code == 429:
        pytest.skip("Jolpica API rate limit exceeded (429). Skipping test.")
    data = resp.json()
    results = data.get('MRData', {}).get('RaceTable', {}).get('Races', [{}])[0].get('Results', [])
    for res in results:
        time_obj = res.get('Time', {})
        if isinstance(time_obj, dict):
            time_str = time_obj.get('time')
            if time_str and "." in time_str:
                decimal_part = time_str.split(".")[-1]
                assert len(decimal_part) == 3, f"Time does not have 3 decimal places: {time_str}"

def test_jolpica_session_name_mapping():
    import requests
    import pytest
    # 2023: SecondPractice should be SprintShootout
    url_2023 = "https://api.jolpi.ca/ergast/f1/2023/4/races.json"  # Azerbaijan GP, has SprintShootout
    resp_2023 = requests.get(url_2023, timeout=10)
    if resp_2023.status_code == 429:
        pytest.skip("Jolpica API rate limit exceeded (429). Skipping test.")
    data_2023 = resp_2023.json()
    races_2023 = data_2023.get('MRData', {}).get('RaceTable', {}).get('Races', [])
    if not races_2023:
        pytest.skip("No races found for 2023/4.")
    race_2023 = races_2023[0]
    # Should have 'SprintShootout' in the response
    assert 'SprintShootout' in race_2023, "Expected 'SprintShootout' session name in 2023 response."
    # 2024: SecondPractice should be SprintQualifying
    url_2024 = "https://api.jolpi.ca/ergast/f1/2024/5/races.json"  # China GP, has SprintQualifying
    resp_2024 = requests.get(url_2024, timeout=10)
    if resp_2024.status_code == 429:
        pytest.skip("Jolpica API rate limit exceeded (429). Skipping test.")
    data_2024 = resp_2024.json()
    races_2024 = data_2024.get('MRData', {}).get('RaceTable', {}).get('Races', [])
    if not races_2024:
        pytest.skip("No races found for 2024/5.")
    race_2024 = races_2024[0]
    # Should have 'SprintQualifying' in the response
    assert 'SprintQualifying' in race_2024, "Expected 'SprintQualifying' session name in 2024 response."

def test_jolpica_duplicate_filter_handling():
    import requests
    import pytest
    # Jolpica should ignore all but the last filter, not return 400
    url = "https://api.jolpi.ca/ergast/f1/2024/drivers/alonso/drivers/hamilton/results.json"
    resp = requests.get(url, timeout=10)
    if resp.status_code == 429:
        pytest.skip("Jolpica API rate limit exceeded (429). Skipping test.")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    # Should only return results for the last driver (hamilton)
    results = data.get('MRData', {}).get('RaceTable', {}).get('Races', [{}])[0].get('Results', [])
    for res in results:
        driver_id = res.get('Driver', {}).get('driverId')
        assert driver_id == 'hamilton', f"Expected only hamilton results, got {driver_id}"

def test_jolpica_required_year_parameter():
    import requests
    import pytest
    # Jolpica requires a year for standings endpoints
    url = "https://api.jolpi.ca/ergast/f1/driverstandings.json"
    resp = requests.get(url, timeout=10)
    if resp.status_code == 429:
        pytest.skip("Jolpica API rate limit exceeded (429). Skipping test.")
    # Should not be 200, should be 404 or error
    assert resp.status_code != 200, f"Expected non-200 for missing year, got {resp.status_code}" 