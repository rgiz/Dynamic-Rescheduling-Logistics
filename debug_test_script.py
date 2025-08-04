"""Checks that df_trips and df_routes faithfully aggregate df_cleaned.
Run with `pytest -q`. Uses column names from `schema.py` so it remains
robust to future renaming.
"""

import sys, os
from pathlib import Path
import pandas as pd

# ensure src/ is importable when the package isn't installed editable
sys.path.append(os.path.abspath("src"))

import schema as S  # pylint: disable=import-error,wrong-import-position

BASE = Path("data/processed")
CLEAN   = BASE / "cleaned.csv"
TRIPS   = BASE / "trips.csv"
ROUTES  = BASE / "routes.csv"

# ────────────────────────────────────────────────────────────────────────────
# Trip‑level aggregation: df_cleaned ➜ df_trips
# ────────────────────────────────────────────────────────────────────────────

def test_trip_aggregation():
    df_clean = pd.read_csv(CLEAN,  parse_dates=[S.START_TS, S.END_TS])
    df_trips = pd.read_csv(TRIPS,  parse_dates=[S.START_TS, S.END_TS])

    agg = (
        df_clean.groupby(S.TRIP_ID).agg(
            **{S.START_LOC: (S.START_LOC, "first")},
            **{S.END_LOC:   (S.END_LOC,   "last")},
            **{S.START_TS:  (S.START_TS,  "first")},
            **{S.END_TS:    (S.END_TS,    "last")},
            **{S.DURATION:  (S.DURATION,  "sum")},
            **{S.MILEAGE:   (S.MILEAGE,   "sum")},
        ).reset_index()
    )

    merged = pd.merge(df_trips, agg, on=S.TRIP_ID, suffixes=("", "_calc"))

    mismatched = (
        ((merged[S.DURATION] - merged[f"{S.DURATION}_calc"]).abs() > 0.01)
        | ((merged[S.MILEAGE]  - merged[f"{S.MILEAGE}_calc"]).abs()  > 0.01)
        | (merged[S.START_TS] != merged[f"{S.START_TS}_calc"])
        | (merged[S.END_TS]   != merged[f"{S.END_TS}_calc"])
        | (merged[S.START_LOC] != merged[f"{S.START_LOC}_calc"])
        | (merged[S.END_LOC]   != merged[f"{S.END_LOC}_calc"])
    )
    assert not mismatched.any(), "df_trips mismatch with aggregation from df_cleaned"

# ────────────────────────────────────────────────────────────────────────────
# Route‑level aggregation: df_trips ➜ df_routes
# ────────────────────────────────────────────────────────────────────────────

def test_route_aggregation():
    df_trips  = pd.read_csv(TRIPS,  parse_dates=[S.START_TS, S.END_TS])
    df_routes = pd.read_csv(ROUTES, parse_dates=[S.ROUTE_START_TS, S.ROUTE_END_TS])

    agg = (
        df_trips.groupby(S.ROUTE_ID).agg(
            **{S.ROUTE_START_TS: (S.START_TS, "min")},
            **{S.ROUTE_END_TS:   (S.END_TS,   "max")},
            # Test both metrics
            route_driving_time   =(S.DURATION, "sum"),      # Sum of actual driving time
            route_total_distance =(S.MILEAGE,  "sum"),      # Sum of distances
            n_trips              =(S.TRIP_ID,  "nunique"),
        ).reset_index()
    )
    
    # Calculate shift duration from the aggregated start/end times
    agg["route_shift_duration"] = (
        agg[S.ROUTE_END_TS] - agg[S.ROUTE_START_TS]
    ).dt.total_seconds() / 60.0

    merged = pd.merge(df_routes, agg, on=S.ROUTE_ID, suffixes=("", "_calc"))

    # Test both driving time and shift duration separately
    driving_time_mismatch = ((merged["route_driving_time"] - merged["route_driving_time_calc"]).abs() > 0.01)
    distance_mismatch = ((merged["route_total_distance"] - merged["route_total_distance_calc"]).abs() > 0.01)
    shift_duration_mismatch = ((merged["route_shift_duration"] - merged["route_shift_duration_calc"]).abs() > 0.01)
    start_time_mismatch = (merged[S.ROUTE_START_TS] != merged[f"{S.ROUTE_START_TS}_calc"])
    end_time_mismatch = (merged[S.ROUTE_END_TS] != merged[f"{S.ROUTE_END_TS}_calc"])
    
    mismatched = (
        driving_time_mismatch | distance_mismatch | shift_duration_mismatch | 
        start_time_mismatch | end_time_mismatch
    )

    assert not mismatched.any(), "df_routes mismatch with aggregation from df_trips"