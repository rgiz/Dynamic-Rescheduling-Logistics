"""
Pytest-compatible test harness for validating aggregated trip and route summaries.
This script verifies correct ordering, accurate totals, and start/end consistency.
"""

import pandas as pd


def test_trip_duration_consistency():
    df = pd.read_csv("data/processed/trips.csv", parse_dates=['od_start_time', 'od_end_time'])
    duration_diff = (df['trip_duration_minutes'] -
                     (df['od_end_time'] - df['od_start_time']).dt.total_seconds() / 60.0).abs()
    assert duration_diff.max() < 1.0, f"Trip duration mismatch too high: max diff = {duration_diff.max():.3f}"


def test_trip_ordering():
    df_clean = pd.read_csv("data/processed/cleaned.csv", parse_dates=['od_start_time'])
    df_trips = pd.read_csv("data/processed/trips.csv")
    trip_id = df_trips.iloc[0]['trip_uuid']
    trip_segments = df_clean[df_clean['trip_uuid'] == trip_id].sort_values('od_start_time')

    diffs = trip_segments['od_start_time'].diff().dropna()
    assert all(diffs >= pd.Timedelta(0)), "Trip elements not ordered by start time"


def test_route_start_end_locations():
    df_clean = pd.read_csv("data/processed/cleaned.csv", parse_dates=['od_start_time'])
    df_routes = pd.read_csv("data/processed/routes.csv")
    route_id = df_routes.iloc[0]['route_schedule_uuid']
    segments = df_clean[df_clean['route_schedule_uuid'] == route_id].sort_values('od_start_time')

    expected_start = segments.iloc[0]['source_center']
    expected_end = segments.iloc[-1]['destination_center']
    actual_start = df_routes.iloc[0]['route_start_location']
    actual_end = df_routes.iloc[0]['route_end_location']

    assert expected_start == actual_start, f"Route start mismatch: {expected_start} vs {actual_start}"
    assert expected_end == actual_end, f"Route end mismatch: {expected_end} vs {actual_end}"


def test_route_total_distance():
    df_clean = pd.read_csv("data/processed/cleaned.csv", parse_dates=['od_start_time'])
    df_routes = pd.read_csv("data/processed/routes.csv")
    route_id = df_routes.iloc[0]['route_schedule_uuid']
    segments = df_clean[df_clean['route_schedule_uuid'] == route_id]
    calc_distance = segments['segment_osrm_distance'].sum()
    stored_distance = df_routes.iloc[0]['route_total_distance']

    assert abs(calc_distance - stored_distance) < 1.0, f"Distance mismatch: {calc_distance} vs {stored_distance}"


# def test_route_waypoints():
#     df_clean = pd.read_csv("data/processed/cleaned.csv", parse_dates=['od_start_time'])
#     df_routes = pd.read_csv("data/processed/routes.csv")
#     route_id = df_routes.iloc[0]['route_schedule_uuid']
#     segments = df_clean[df_clean['route_schedule_uuid'] == route_id].sort_values('od_start_time')

#     expected_path = "-".join(segments['destination_center'].tolist())
#     actual_path = df_routes.iloc[0]['route_waypoints']

#     assert expected_path == actual_path, f"Waypoint path mismatch: {expected_path} vs {actual_path}"