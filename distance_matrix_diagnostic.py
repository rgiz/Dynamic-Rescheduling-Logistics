#!/usr/bin/env python3
"""
Distance Matrix Diagnostic Script
=================================

Focused debugging to understand distance matrix issues and candidate generation.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

def setup_path():
    """Add src to Python path."""
    src_path = Path(__file__).parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
        return True
    return False

def diagnose_distance_matrix():
    """Comprehensive distance matrix diagnosis."""
    print("🔍 DISTANCE MATRIX DIAGNOSIS")
    print("=" * 50)
    
    # 1. Check if matrix file exists and load
    dist_file = Path("data/dist_matrix.npz")
    if not dist_file.exists():
        print(f"❌ Distance matrix not found at {dist_file}")
        return None, None, None
    
    try:
        dist_data = np.load(str(dist_file), allow_pickle=True)
        print(f"✅ Loaded distance matrix file")
        print(f"   File size: {dist_file.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Check what's in the file
        print(f"   Keys in file: {list(dist_data.keys())}")
        
        distance_matrix = dist_data['time']
        location_ids = dist_data['ids']
        
        print(f"   Matrix shape: {distance_matrix.shape}")
        print(f"   Location count: {len(location_ids)}")
        print(f"   Data type: {distance_matrix.dtype}")
        
    except Exception as e:
        print(f"❌ Error loading distance matrix: {e}")
        return None, None, None
    
    # 2. Analyze the distance values in detail
    print(f"\n📊 DISTANCE VALUE ANALYSIS:")
    
    # Remove diagonal (self-distances should be 0)
    non_diagonal_mask = ~np.eye(distance_matrix.shape[0], dtype=bool)
    non_diagonal_distances = distance_matrix[non_diagonal_mask]
    non_zero_distances = non_diagonal_distances[non_diagonal_distances > 0]
    
    if len(non_zero_distances) > 0:
        print(f"   Non-zero distances: {len(non_zero_distances):,} out of {len(non_diagonal_distances):,}")
        print(f"   Min distance: {np.min(non_zero_distances):.1f}")
        print(f"   Max distance: {np.max(non_zero_distances):.1f}")
        print(f"   Mean distance: {np.mean(non_zero_distances):.1f}")
        print(f"   Median distance: {np.median(non_zero_distances):.1f}")
        print(f"   Std deviation: {np.std(non_zero_distances):.1f}")
        
        # Show percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"   Percentiles:")
        for p in percentiles:
            value = np.percentile(non_zero_distances, p)
            print(f"     {p:2d}%: {value:8.1f} minutes ({value/60:6.1f} hours)")
    
    # 3. Check specific location pairs
    print(f"\n🗺️ SAMPLE LOCATION DISTANCES:")
    sample_locations = location_ids[:10] if len(location_ids) > 10 else location_ids
    location_to_index = {str(loc): i for i, loc in enumerate(location_ids)}
    
    for i in range(min(5, len(sample_locations))):
        for j in range(i+1, min(i+3, len(sample_locations))):
            loc1, loc2 = sample_locations[i], sample_locations[j]
            distance = distance_matrix[i, j]
            print(f"   {str(loc1)[:15]:15} -> {str(loc2)[:15]:15}: {distance:8.1f} min ({distance/60:6.1f} hr)")
    
    # 4. Unit analysis
    print(f"\n🔢 UNIT ANALYSIS:")
    if len(non_zero_distances) > 0:
        median_distance = np.median(non_zero_distances)
        mean_distance = np.mean(non_zero_distances)
        
        print(f"   If these are MINUTES:")
        print(f"     Median travel time: {median_distance/60:.1f} hours")
        print(f"     Mean travel time: {mean_distance/60:.1f} hours")
        
        print(f"   If these are SECONDS:")
        print(f"     Median travel time: {median_distance/60:.1f} minutes")
        print(f"     Mean travel time: {mean_distance/60:.1f} minutes")
        
        print(f"   If these are in WRONG UNITS (e.g., 60x too big):")
        print(f"     Corrected median: {median_distance/60:.1f} minutes")
        print(f"     Corrected mean: {mean_distance/60:.1f} minutes")
    
    return distance_matrix, location_ids, location_to_index

def trace_candidate_generation():
    """Trace exactly how candidate generation calculates deadhead."""
    print(f"\n🔄 CANDIDATE GENERATION TRACING")
    print("=" * 50)
    
    # Import required modules
    try:
        from models.driver_state import DriverState, DailyAssignment
        from opt.candidate_gen_v2 import CandidateGeneratorV2
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Load distance matrix
    distance_matrix, location_ids, location_to_index = diagnose_distance_matrix()
    if distance_matrix is None:
        return
    
    # Create simple test case
    print(f"\n🧪 Creating test case...")
    
    # Pick two real locations from the matrix
    if len(location_ids) >= 2:
        loc1, loc2 = str(location_ids[0]), str(location_ids[1])
        expected_distance = distance_matrix[0, 1]
        
        print(f"   Test locations: {loc1} -> {loc2}")
        print(f"   Matrix distance: {expected_distance:.1f} minutes")
        print(f"   Matrix distance: {expected_distance/60:.1f} hours")
    else:
        loc1, loc2 = "TEST_LOC_A", "TEST_LOC_B"
        expected_distance = None
        print(f"   Using dummy locations (no matrix lookup possible)")
    
    # Create test driver state
    driver_state = DriverState(driver_id="test_driver", route_id="test_route")
    
    # Add assignment ending at loc1 (so driver is currently at loc1)
    past_assignment = DailyAssignment(
        trip_id="past_trip",
        start_time=datetime(2025, 1, 15, 8, 0),
        end_time=datetime(2025, 1, 15, 12, 0),
        duration_minutes=240,
        start_location='DEPOT',
        end_location=loc1
    )
    driver_state.add_assignment('2025-01-15', past_assignment)
    
    # Add future assignment starting from different location
    future_assignment = DailyAssignment(
        trip_id="future_trip",
        start_time=datetime(2025, 1, 15, 16, 0),
        end_time=datetime(2025, 1, 15, 20, 0),
        duration_minutes=240,
        start_location=loc2,
        end_location='DEPOT'
    )
    driver_state.add_assignment('2025-01-15', future_assignment)
    
    print(f"   Driver currently at: {loc1}")
    print(f"   Driver next trip from: {loc2}")
    
    # Create disrupted trip
    disrupted_trip = {
        'id': 'test_disrupted',
        'start_time': datetime(2025, 1, 15, 13, 0),
        'end_time': datetime(2025, 1, 15, 15, 0),
        'duration_minutes': 120,
        'start_location': loc1,  # Same as where driver is
        'end_location': loc2     # Same as where driver needs to be next
    }
    
    print(f"   Disrupted trip: {loc1} -> {loc2}")
    print(f"   Trip duration: 120 minutes")
    
    # Initialize candidate generator
    driver_states = {"test_driver": driver_state}
    candidate_generator = CandidateGeneratorV2(
        driver_states=driver_states,
        distance_matrix=distance_matrix,
        location_to_index=location_to_index
    )
    
    # Generate candidates with tracing
    print(f"\n🔍 Generating candidates with detailed tracing...")
    
    # Monkey patch the distance calculation to add tracing
    original_calculate_travel_time = candidate_generator._calculate_travel_time
    
    def traced_calculate_travel_time(from_location, to_location):
        print(f"     🧭 Travel time calculation:")
        print(f"        From: {from_location}")
        print(f"        To: {to_location}")
        
        result = original_calculate_travel_time(from_location, to_location)
        print(f"        Result: {result:.1f} minutes")
        
        # Also try manual lookup for comparison
        if location_to_index and from_location in location_to_index and to_location in location_to_index:
            from_idx = location_to_index[from_location]
            to_idx = location_to_index[to_location]
            matrix_result = distance_matrix[from_idx, to_idx]
            print(f"        Matrix lookup: {matrix_result:.1f} minutes")
            if abs(result - matrix_result) > 0.1:
                print(f"        ⚠️ MISMATCH: Function returned {result:.1f}, matrix has {matrix_result:.1f}")
        else:
            print(f"        Matrix lookup: Not possible (location not in index)")
        
        return result
    
    candidate_generator._calculate_travel_time = traced_calculate_travel_time
    
    # Generate candidates
    try:
        candidates = candidate_generator.generate_candidates(
            disrupted_trip, include_cascades=False, include_outsource=False
        )
        
        print(f"\n📊 Generated {len(candidates)} candidates:")
        for i, candidate in enumerate(candidates):
            candidate_type = getattr(candidate, 'candidate_type', 'unknown')
            driver_id = getattr(candidate, 'assigned_driver_id', None)
            deadhead_minutes = getattr(candidate, 'deadhead_minutes', 0)
            
            print(f"   [{i}] {candidate_type} | Driver: {driver_id} | Deadhead: {deadhead_minutes:.1f} min")
            
            if deadhead_minutes == 0 and expected_distance and expected_distance > 0:
                print(f"       ⚠️ Zero deadhead but matrix shows {expected_distance:.1f} min distance!")
            elif deadhead_minutes == 60 and expected_distance and abs(expected_distance - 60) > 10:
                print(f"       ⚠️ 60-min deadhead but matrix shows {expected_distance:.1f} min distance!")
        
    except Exception as e:
        print(f"❌ Error generating candidates: {e}")
        import traceback
        traceback.print_exc()

def check_source_files():
    """Check the source files that create the distance matrix."""
    print(f"\n📁 SOURCE FILE ANALYSIS")
    print("=" * 50)
    
    files_to_check = [
        "src/triangulation.py",
        "scripts/generate_distance_matrix.py",
        "data/processed/trips.csv",
        "data/processed/center_coordinates.csv"
    ]
    
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
            if file_path.endswith('.csv'):
                try:
                    df = pd.read_csv(path)
                    print(f"   Shape: {df.shape}")
                    print(f"   Columns: {list(df.columns)}")
                    if 'segment_osrm_distance' in df.columns:
                        distances = df['segment_osrm_distance'].dropna()
                        if len(distances) > 0:
                            print(f"   Distance range: {distances.min():.1f} - {distances.max():.1f}")
                            print(f"   Distance mean: {distances.mean():.1f}")
                except Exception as e:
                    print(f"   Error reading: {e}")
        else:
            print(f"❌ {file_path} - NOT FOUND")

def main():
    """Run all diagnostics."""
    print("🕵️ DISTANCE MATRIX & CANDIDATE GENERATION DIAGNOSIS")
    print("=" * 60)
    
    if not setup_path():
        return 1
    
    # 1. Distance matrix analysis
    diagnose_distance_matrix()
    
    # 2. Source files check
    check_source_files()
    
    # 3. Candidate generation tracing
    trace_candidate_generation()
    
    print(f"\n" + "=" * 60)
    print("🏁 DIAGNOSIS COMPLETE")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)