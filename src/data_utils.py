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
    """
    Load existing coordinates or generate them with PROPER SCALING.
    Fixed version that calibrates MDS coordinates to real distances.
    """
    if project_root is None:
        project_root = Path.cwd().parent  # assumes we're inside /notebooks/

    coord_path = project_root / "data" / "processed" / "center_coordinates.csv"

    if coord_path.exists():
        print(f"📍 Loading existing coordinates from {coord_path}")
        return pd.read_csv(coord_path)

    print(f"🗺️ Generating coordinates with PROPER SCALING...")
    
    # Load trips to build from scratch
    trips_path = project_root / "data" / "processed" / "trips.csv"
    if not trips_path.exists():
        raise FileNotFoundError(f"Trips file not found: {trips_path}")
        
    df = pd.read_csv(trips_path)
    
    # Extract source-destination-distance triples
    df_pairs = df[['source_center', 'destination_center', 'segment_osrm_distance']].copy()
    df_pairs = df_pairs[df_pairs['source_center'] != df_pairs['destination_center']]

    print(f"   Loaded {len(df_pairs)} location pairs")
    print(f"   Distance range: {df_pairs['segment_osrm_distance'].min():.1f} - {df_pairs['segment_osrm_distance'].max():.1f} km")

    # Aggregate: median distance between source-destination pairs
    pair_medians = df_pairs.groupby(['source_center', 'destination_center'])['segment_osrm_distance'].median().reset_index()
    
    locations = sorted(set(pair_medians['source_center']).union(set(pair_medians['destination_center'])))
    loc_idx = {loc: i for i, loc in enumerate(locations)}
    n = len(locations)

    print(f"   Found {n} unique locations")

    # Build symmetric distance matrix (in REAL kilometers)
    real_distance_matrix = np.full((n, n), np.nan)
    for _, row in pair_medians.iterrows():
        i, j = loc_idx[row['source_center']], loc_idx[row['destination_center']]
        distance_km = row['segment_osrm_distance']
        real_distance_matrix[i, j] = distance_km
        real_distance_matrix[j, i] = distance_km

    # Fill diagonal and replace NaNs
    np.fill_diagonal(real_distance_matrix, 0)
    max_distance = np.nanmax(real_distance_matrix)
    real_distance_matrix = np.nan_to_num(real_distance_matrix, nan=max_distance)

    print(f"   Built real distance matrix with max distance: {max_distance:.1f} km")

    # Apply MDS to get coordinates in arbitrary units
    from sklearn.manifold import MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_coords = mds.fit_transform(real_distance_matrix)

    print(f"   MDS triangulation complete")

    # CRITICAL STEP: Calibrate MDS coordinates to real distances
    print(f"   🔧 Calibrating MDS coordinates to real kilometers...")
    
    # Calculate distances in MDS space
    mds_distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = mds_coords[i, 0] - mds_coords[j, 0]
                dy = mds_coords[i, 1] - mds_coords[j, 1]
                mds_distance_matrix[i, j] = np.sqrt(dx*dx + dy*dy)

    # Find scaling factor by comparing MDS distances to real distances
    valid_pairs = (mds_distance_matrix > 0) & (real_distance_matrix > 0)
    mds_vals = mds_distance_matrix[valid_pairs]
    real_vals = real_distance_matrix[valid_pairs]

    scaling_factor = np.median(real_vals / mds_vals)
    print(f"   Scaling factor: {scaling_factor:.4f} km per MDS unit")

    # Scale coordinates to real kilometers
    scaled_coords = mds_coords * scaling_factor

    print(f"   Scaled coordinate ranges:")
    print(f"   X: {scaled_coords[:, 0].min():.2f} to {scaled_coords[:, 0].max():.2f} km")
    print(f"   Y: {scaled_coords[:, 1].min():.2f} to {scaled_coords[:, 1].max():.2f} km")

    # Verification - check a few distances
    print(f"   🔍 Verification (sample distances):")
    verification_count = 0
    total_error = 0
    
    for i in range(min(3, n)):
        for j in range(i+1, min(i+3, n)):
            if verification_count >= 5:  # Limit output
                break
                
            # Real distance
            real_dist = real_distance_matrix[i, j]
            
            # Calculated distance from scaled coordinates
            dx = scaled_coords[i, 0] - scaled_coords[j, 0]
            dy = scaled_coords[i, 1] - scaled_coords[j, 1]
            calc_dist = np.sqrt(dx*dx + dy*dy)
            
            error = abs(real_dist - calc_dist)
            error_pct = (error / real_dist * 100) if real_dist > 0 else 0
            total_error += error_pct
            verification_count += 1
            
            if verification_count <= 3:  # Show first 3
                print(f"     {locations[i][:12]:12} -> {locations[j][:12]:12}: Real {real_dist:6.1f}km, Calc {calc_dist:6.1f}km, Err {error_pct:4.1f}%")
        
        if verification_count >= 5:
            break

    avg_error = total_error / verification_count if verification_count > 0 else 0
    print(f"   Average error: {avg_error:.1f}%")

    # Create DataFrame with properly scaled coordinates
    df_coords = pd.DataFrame(scaled_coords, columns=["x", "y"])
    df_coords["center_id"] = locations

    # Save coordinates
    coord_path.parent.mkdir(parents=True, exist_ok=True)
    df_coords.to_csv(coord_path, index=False)

    print(f"   ✅ PROPERLY SCALED coordinates saved to {coord_path}")
    print(f"   Ready for distance matrix generation!")

    return df_coords