from pathlib import Path 
import pandas as pd
import numpy as np
from sklearn.manifold import MDS

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw CSV into a DataFrame.
    """
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded data with shape: {df.shape}")
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning: convert timestamps, fill nulls if needed.
    """
    time_cols = ['trip_creation_time', 'od_start_time', 'od_end_time', 'cutoff_timestamp']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


def group_by_trip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group all rows belonging to the same trip. One row = one segment.
    This function returns a DataFrame grouped by trip, with aggregate stats.
    Ensures correct ordering to get accurate start/end metadata.
    """
    group_cols = ['route_schedule_uuid', 'trip_uuid']

    # Sort by trip and start time within each trip
    df_sorted = df.sort_values(by=['trip_uuid', 'od_start_time'])

    # Aggregate over each trip
    grouped = df_sorted.groupby(group_cols).agg({
        'segment_actual_time': 'sum',
        'segment_osrm_time': 'sum',
        'segment_osrm_distance': 'sum',
        'actual_time': 'sum',
        'osrm_time': 'sum',
        'osrm_distance': 'sum',
        'cutoff_factor': 'mean',
        'start_scan_to_end_scan': 'sum',
        'od_start_time': 'min',
        'od_end_time': 'max',
        'source_center': 'first',            # from sorted segments
        'destination_center': 'last',        # from sorted segments
    }).reset_index()

    # Compute clock-time trip duration (elapsed time from start to end)
    grouped['trip_duration_minutes'] = (grouped['od_end_time'] - grouped['od_start_time']).dt.total_seconds() / 60.0

    return grouped


def group_by_route(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate at the route level. One row per route_schedule_uuid.
    Computes both total driving time and total shift duration for regulatory compliance.
    """
    group_col = "route_schedule_uuid"

    grouped = df.groupby(group_col).agg({
        "trip_uuid": "nunique",
        "segment_osrm_distance": "sum",
        "segment_actual_time": "sum",    # Total driving time (active)
        "od_start_time": "min",          # Shift start time
        "od_end_time": "max",            # Shift end time
    }).reset_index()

    # Calculate total shift duration (elapsed time including breaks)
    shift_duration_minutes = (grouped["od_end_time"] - grouped["od_start_time"]).dt.total_seconds() / 60.0

    grouped = grouped.rename(columns={
        "trip_uuid": "num_trips",
        "segment_osrm_distance": "route_total_distance",
        "segment_actual_time": "route_driving_time",     # Active driving time
        "od_start_time": "route_start_time",
        "od_end_time": "route_end_time"
    })

    # Add the shift duration as a separate column
    grouped["route_shift_duration"] = shift_duration_minutes

    # For regulatory compliance and optimization, we might want total time to refer to shift duration
    # But keep both metrics available
    grouped["route_total_time"] = shift_duration_minutes  # For backward compatibility with tests

    # For all values derived from sorted sequences, use df.groupby().apply()
    sorted_groups = df.groupby(group_col)

    grouped["route_start_location"] = sorted_groups.apply(
        lambda g: g.sort_values("od_start_time").iloc[0]["source_center"], 
        include_groups=False  # Suppress the pandas warning
    ).values
    
    grouped["route_end_location"] = sorted_groups.apply(
        lambda g: g.sort_values("od_start_time").iloc[-1]["destination_center"],
        include_groups=False  # Suppress the pandas warning
    ).values

    return grouped


def load_or_generate_coordinates(project_root: Path = None) -> pd.DataFrame:
    if project_root is None:
        project_root = Path.cwd().parent  # assumes we're inside /notebooks/

    coord_path = project_root / "data" / "processed" / "center_coordinates.csv"

    if coord_path.exists():
        return pd.read_csv(coord_path)

    # Load trips to build from scratch
    df = pd.read_csv(project_root / "data" / "processed" / "trips.csv")
    df_pairs = df[['source_center', 'destination_center', 'segment_osrm_distance']]
    df_pairs = df_pairs[df_pairs['source_center'] != df_pairs['destination_center']]

    pair_medians = df_pairs.groupby(['source_center', 'destination_center'])['segment_osrm_distance'].median().reset_index()
    locations = sorted(set(pair_medians['source_center']).union(set(pair_medians['destination_center'])))
    loc_idx = {loc: i for i, loc in enumerate(locations)}
    n = len(locations)

    matrix = np.full((n, n), np.nan)
    for _, row in pair_medians.iterrows():
        i, j = loc_idx[row['source_center']], loc_idx[row['destination_center']]
        matrix[i, j] = matrix[j, i] = row['segment_osrm_distance']

    np.fill_diagonal(matrix, 0)
    matrix = np.nan_to_num(matrix, nan=np.nanmax(matrix))

    coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(matrix)
    df_coords = pd.DataFrame(coords, columns=["x", "y"])
    df_coords["center_id"] = locations
    coord_path.parent.mkdir(parents=True, exist_ok=True)
    df_coords.to_csv(coord_path, index=False)

    return df_coords