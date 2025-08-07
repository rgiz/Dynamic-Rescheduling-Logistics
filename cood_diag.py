#!/usr/bin/env python3
"""
Coordinate Scaling Verification Script
=====================================

This script verifies each step of the coordinate scaling process to identify issues.
Run from project root: python3 coord_scaling_diagnostic.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def setup_path():
    """Add src to Python path."""
    src_path = Path(__file__).parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
        return True
    return False

def step1_analyze_original_trip_data():
    """Step 1: Analyze the original trip distance data"""
    print("STEP 1: ANALYZING ORIGINAL TRIP DATA")
    print("=" * 50)
    
    trips_path = Path("data/processed/trips.csv")
    if not trips_path.exists():
        print(f"❌ Trips file not found: {trips_path}")
        return None
    
    df = pd.read_csv(trips_path)
    print(f"✅ Loaded trips data: {df.shape}")
    
    # Check distance column
    if 'segment_osrm_distance' not in df.columns:
        print(f"❌ segment_osrm_distance column not found")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    distances = df['segment_osrm_distance'].dropna()
    print(f"\n📊 Original Trip Distances Analysis:")
    print(f"   Count: {len(distances):,}")
    print(f"   Min: {distances.min():.1f} km")
    print(f"   Max: {distances.max():.1f} km")
    print(f"   Mean: {distances.mean():.1f} km")
    print(f"   Median: {distances.median():.1f} km")
    print(f"   Std: {distances.std():.1f} km")
    
    # Show percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n   Percentiles:")
    for p in percentiles:
        value = distances.quantile(p/100)
        print(f"     {p:2d}%: {value:6.1f} km")
    
    # Check for unrealistic values
    print(f"\n🔍 Data Quality Checks:")
    print(f"   Zero distances: {(distances == 0).sum():,}")
    print(f"   Negative distances: {(distances < 0).sum():,}")
    print(f"   Very long distances (>2000km): {(distances > 2000).sum():,}")
    print(f"   Extremely long distances (>3000km): {(distances > 3000).sum():,}")
    
    if distances.max() > 4000:
        print(f"   ⚠️ WARNING: Maximum distance {distances.max():.1f}km seems very high for India")
    
    return df

def step2_analyze_distance_pairs():
    """Step 2: Analyze location pairs and their distances"""
    print(f"\nSTEP 2: ANALYZING LOCATION PAIRS")
    print("=" * 50)
    
    df = pd.read_csv("data/processed/trips.csv")
    
    # Extract pairs as done in data_utils.py
    df_pairs = df[['source_center', 'destination_center', 'segment_osrm_distance']].copy()
    df_pairs = df_pairs[df_pairs['source_center'] != df_pairs['destination_center']]
    
    print(f"✅ Extracted {len(df_pairs):,} location pairs")
    
    # Get unique locations
    locations = sorted(set(df_pairs['source_center']).union(set(df_pairs['destination_center'])))
    print(f"✅ Found {len(locations)} unique locations")
    
    # Show some example locations
    print(f"   Sample locations: {locations[:10]}")
    
    # Aggregate to medians as in data_utils.py
    pair_medians = df_pairs.groupby(['source_center', 'destination_center'])['segment_osrm_distance'].median().reset_index()
    
    print(f"✅ Aggregated to {len(pair_medians):,} unique pairs")
    
    # Analyze pair distances
    pair_distances = pair_medians['segment_osrm_distance']
    print(f"\n📊 Pair Distance Analysis:")
    print(f"   Min: {pair_distances.min():.1f} km")
    print(f"   Max: {pair_distances.max():.1f} km")
    print(f"   Mean: {pair_distances.mean():.1f} km")
    print(f"   Median: {pair_distances.median():.1f} km")
    
    # Find the longest distances to check if they're reasonable
    print(f"\n🔍 Longest Distance Pairs:")
    longest_pairs = pair_medians.nlargest(10, 'segment_osrm_distance')
    for _, row in longest_pairs.iterrows():
        print(f"   {row['source_center']} -> {row['destination_center']}: {row['segment_osrm_distance']:.1f} km")
    
    return pair_medians, locations

def step3_build_real_distance_matrix(pair_medians, locations):
    """Step 3: Build the real distance matrix"""
    print(f"\nSTEP 3: BUILDING REAL DISTANCE MATRIX")
    print("=" * 50)
    
    loc_idx = {loc: i for i, loc in enumerate(locations)}
    n = len(locations)
    
    # Build symmetric distance matrix as in data_utils.py
    real_distance_matrix = np.full((n, n), np.nan)
    
    filled_entries = 0
    for _, row in pair_medians.iterrows():
        i, j = loc_idx[row['source_center']], loc_idx[row['destination_center']]
        distance_km = row['segment_osrm_distance']
        real_distance_matrix[i, j] = distance_km
        real_distance_matrix[j, i] = distance_km
        filled_entries += 2
    
    print(f"✅ Filled {filled_entries:,} matrix entries from {len(pair_medians):,} pairs")
    
    # Fill diagonal and replace NaNs
    np.fill_diagonal(real_distance_matrix, 0)
    max_distance = np.nanmax(real_distance_matrix)
    nan_count = np.isnan(real_distance_matrix).sum()
    print(f"   Matrix size: {n} x {n} = {n*n:,} entries")
    print(f"   NaN entries: {nan_count:,}")
    print(f"   Max distance: {max_distance:.1f} km")
    
    real_distance_matrix = np.nan_to_num(real_distance_matrix, nan=max_distance)
    print(f"✅ Replaced NaNs with max distance: {max_distance:.1f} km")
    
    return real_distance_matrix

def step4_apply_mds(real_distance_matrix):
    """Step 4: Apply MDS to get coordinates"""
    print(f"\nSTEP 4: APPLYING MDS")
    print("=" * 50)
    
    try:
        from sklearn.manifold import MDS
    except ImportError:
        print("❌ sklearn not available, cannot test MDS")
        return None
    
    print("🔧 Applying MDS...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_coords = mds.fit_transform(real_distance_matrix)
    
    print(f"✅ MDS complete, stress: {mds.stress_:.2f}")
    print(f"   MDS coordinate ranges:")
    print(f"     X: {mds_coords[:, 0].min():.4f} to {mds_coords[:, 0].max():.4f}")
    print(f"     Y: {mds_coords[:, 1].min():.4f} to {mds_coords[:, 1].max():.4f}")
    
    # Calculate distances in MDS space
    n = len(mds_coords)
    mds_distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = mds_coords[i, 0] - mds_coords[j, 0]
                dy = mds_coords[i, 1] - mds_coords[j, 1]
                mds_distance_matrix[i, j] = np.sqrt(dx*dx + dy*dy)
    
    print(f"   MDS distance ranges:")
    non_zero_mds = mds_distance_matrix[mds_distance_matrix > 0]
    print(f"     Min: {non_zero_mds.min():.4f} MDS units")
    print(f"     Max: {non_zero_mds.max():.4f} MDS units")
    
    return mds_coords, mds_distance_matrix

def step5_calculate_scaling_factor(mds_distance_matrix, real_distance_matrix):
    """Step 5: Calculate scaling factor"""
    print(f"\nSTEP 5: CALCULATING SCALING FACTOR")
    print("=" * 50)
    
    # Find valid pairs for comparison
    valid_pairs = (mds_distance_matrix > 0) & (real_distance_matrix > 0)
    mds_vals = mds_distance_matrix[valid_pairs]
    real_vals = real_distance_matrix[valid_pairs]
    
    print(f"✅ Found {len(mds_vals):,} valid pairs for scaling calculation")
    
    if len(mds_vals) == 0:
        print("❌ No valid pairs found for scaling!")
        return None
    
    # Calculate scaling factor as in data_utils.py
    scaling_factor = np.median(real_vals / mds_vals)
    
    print(f"📊 Scaling Analysis:")
    print(f"   MDS range: {mds_vals.min():.4f} to {mds_vals.max():.4f}")
    print(f"   Real range: {real_vals.min():.1f} to {real_vals.max():.1f} km")
    print(f"   Scaling factor: {scaling_factor:.4f} km per MDS unit")
    
    # Show some example scaling comparisons
    print(f"\n🔍 Scaling Examples:")
    ratios = real_vals / mds_vals
    print(f"   Min ratio: {ratios.min():.4f}")
    print(f"   Max ratio: {ratios.max():.4f}")
    print(f"   Mean ratio: {ratios.mean():.4f}")
    print(f"   Median ratio: {ratios.median():.4f}")
    print(f"   Std ratio: {ratios.std():.4f}")
    
    # Check if scaling factors are consistent
    if ratios.std() / ratios.mean() > 0.5:  # High coefficient of variation
        print(f"   ⚠️ WARNING: High variation in scaling ratios - MDS may not be good fit")
    
    return scaling_factor

def step6_verify_final_coordinates(mds_coords, scaling_factor, real_distance_matrix, locations):
    """Step 6: Verify final scaled coordinates"""
    print(f"\nSTEP 6: VERIFYING FINAL COORDINATES")
    print("=" * 50)
    
    if scaling_factor is None:
        print("❌ Cannot verify - no scaling factor")
        return None
    
    # Scale coordinates
    scaled_coords = mds_coords * scaling_factor
    
    print(f"✅ Final coordinate ranges:")
    print(f"   X: {scaled_coords[:, 0].min():.2f} to {scaled_coords[:, 0].max():.2f} km")
    print(f"   Y: {scaled_coords[:, 1].min():.2f} to {scaled_coords[:, 1].max():.2f} km")
    
    x_span = scaled_coords[:, 0].max() - scaled_coords[:, 0].min()
    y_span = scaled_coords[:, 1].max() - scaled_coords[:, 1].min()
    print(f"   X span: {x_span:.2f} km")
    print(f"   Y span: {y_span:.2f} km")
    
    # Verify against known geography
    print(f"\n🗺️ Geographic Reality Check:")
    print(f"   India's actual dimensions: ~3,000 km E-W, ~3,200 km N-S")
    
    if x_span > 4000 or y_span > 4000:
        print(f"   ⚠️ WARNING: Coordinate span seems too large for India")
    elif x_span < 1000 or y_span < 1000:
        print(f"   ⚠️ WARNING: Coordinate span seems too small for India")
    else:
        print(f"   ✅ Coordinate spans look reasonable for India")
    
    # Verification - check some distances
    print(f"\n🔍 Distance Verification:")
    verification_count = 0
    total_error = 0
    max_error = 0
    
    n = len(scaled_coords)
    for i in range(min(10, n)):
        for j in range(i+1, min(i+5, n)):
            if verification_count >= 20:  # Limit verification
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
            max_error = max(max_error, error_pct)
            verification_count += 1
            
            if verification_count <= 5:  # Show first 5
                loc1 = locations[i][:12] if i < len(locations) else f"Loc{i}"
                loc2 = locations[j][:12] if j < len(locations) else f"Loc{j}"
                print(f"   {loc1:12} -> {loc2:12}: Real {real_dist:6.1f}km, Calc {calc_dist:6.1f}km, Err {error_pct:4.1f}%")
        
        if verification_count >= 20:
            break
    
    avg_error = total_error / verification_count if verification_count > 0 else 0
    print(f"\n📊 Verification Results:")
    print(f"   Average error: {avg_error:.1f}%")
    print(f"   Maximum error: {max_error:.1f}%")
    
    if avg_error < 10:
        print(f"   ✅ Coordinate scaling appears accurate")
    elif avg_error < 25:
        print(f"   ⚠️ Moderate scaling errors - check MDS fit")
    else:
        print(f"   ❌ High scaling errors - MDS may not be suitable")
    
    return scaled_coords

def main():
    """Run complete coordinate scaling verification"""
    print("🔍 COORDINATE SCALING VERIFICATION")
    print("=" * 60)
    
    if not setup_path():
        print("❌ Could not setup Python path")
        return 1
    
    # Step 1: Analyze original trip data
    df = step1_analyze_original_trip_data()
    if df is None:
        return 1
    
    # Step 2: Analyze location pairs
    try:
        pair_medians, locations = step2_analyze_distance_pairs()
    except Exception as e:
        print(f"❌ Error in step 2: {e}")
        return 1
    
    # Step 3: Build real distance matrix
    try:
        real_distance_matrix = step3_build_real_distance_matrix(pair_medians, locations)
    except Exception as e:
        print(f"❌ Error in step 3: {e}")
        return 1
    
    # Step 4: Apply MDS
    try:
        mds_result = step4_apply_mds(real_distance_matrix)
        if mds_result is None:
            return 1
        mds_coords, mds_distance_matrix = mds_result
    except Exception as e:
        print(f"❌ Error in step 4: {e}")
        return 1
    
    # Step 5: Calculate scaling factor
    try:
        scaling_factor = step5_calculate_scaling_factor(mds_distance_matrix, real_distance_matrix)
    except Exception as e:
        print(f"❌ Error in step 5: {e}")
        return 1
    
    # Step 6: Verify final coordinates
    try:
        scaled_coords = step6_verify_final_coordinates(mds_coords, scaling_factor, real_distance_matrix, locations)
    except Exception as e:
        print(f"❌ Error in step 6: {e}")
        return 1
    
    print(f"\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    # Compare with existing coordinates if they exist
    coord_path = Path("data/processed/center_coordinates.csv")
    if coord_path.exists():
        print(f"\n🔄 Comparing with existing coordinates...")
        existing_coords = pd.read_csv(coord_path)
        
        if len(existing_coords) == len(locations):
            existing_x_span = existing_coords['x'].max() - existing_coords['x'].min()
            existing_y_span = existing_coords['y'].max() - existing_coords['y'].min()
            
            print(f"   Existing coordinate spans: X={existing_x_span:.1f} km, Y={existing_y_span:.1f} km")
            
            if scaled_coords is not None:
                new_x_span = scaled_coords[:, 0].max() - scaled_coords[:, 0].min()
                new_y_span = scaled_coords[:, 1].max() - scaled_coords[:, 1].min()
                print(f"   Recalculated spans: X={new_x_span:.1f} km, Y={new_y_span:.1f} km")
                
                if abs(existing_x_span - new_x_span) < 100 and abs(existing_y_span - new_y_span) < 100:
                    print(f"   ✅ Coordinate spans match - scaling is consistent")
                else:
                    print(f"   ⚠️ Coordinate spans differ - may need regeneration")
        else:
            print(f"   ⚠️ Location count mismatch: existing={len(existing_coords)}, calculated={len(locations)}")
    
    print(f"\n✅ Verification complete!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
    