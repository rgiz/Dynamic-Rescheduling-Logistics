#!/usr/bin/env python3
"""
Validation Script - Check if Distance Matrix Fix Worked
======================================================

This script validates that:
1. Coordinates were regenerated with proper scaling
2. Distance matrix has realistic values  
3. Candidate generation now works correctly
"""

import sys
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

def validate_coordinates():
    """Check if coordinates were regenerated properly."""
    print("📍 VALIDATING COORDINATES")
    print("-" * 40)
    
    coords_file = Path("data/processed/center_coordinates.csv")
    if not coords_file.exists():
        print("❌ Coordinates file doesn't exist!")
        return False
    
    # Check file modification time
    mod_time = datetime.fromtimestamp(coords_file.stat().st_mtime)
    print(f"   File last modified: {mod_time}")
    
    # Load and analyze coordinates
    df_coords = pd.read_csv(coords_file)
    print(f"   Shape: {df_coords.shape}")
    print(f"   Columns: {list(df_coords.columns)}")
    
    x_range = df_coords['x'].max() - df_coords['x'].min()
    y_range = df_coords['y'].max() - df_coords['y'].min()
    
    print(f"   X range: {df_coords['x'].min():.1f} to {df_coords['x'].max():.1f} (span: {x_range:.1f})")
    print(f"   Y range: {df_coords['y'].min():.1f} to {df_coords['y'].max():.1f} (span: {y_range:.1f})")
    
    # Check if coordinates look like they're in kilometers (not arbitrary MDS units)
    if x_range > 100 and x_range < 10000 and y_range > 100 and y_range < 10000:
        print("   ✅ Coordinate ranges look reasonable for India (in km)")
        return True
    elif x_range < 10 and y_range < 10:
        print("   ❌ Coordinates look like unscaled MDS units (too small)")
        return False
    else:
        print(f"   ⚠️ Coordinate ranges unclear - need manual inspection")
        return None

def validate_distance_matrix():
    """Check if distance matrix has realistic values."""
    print("\n🗺️ VALIDATING DISTANCE MATRIX")
    print("-" * 40)
    
    dist_file = Path("data/dist_matrix.npz")
    if not dist_file.exists():
        print("❌ Distance matrix doesn't exist!")
        return False
    
    # Check file modification time
    mod_time = datetime.fromtimestamp(dist_file.stat().st_mtime)
    print(f"   File last modified: {mod_time}")
    
    # Load and analyze matrix
    try:
        dist_data = np.load(str(dist_file), allow_pickle=True)
        time_matrix = dist_data['time']
        location_ids = dist_data['ids']
        
        print(f"   Matrix shape: {time_matrix.shape}")
        print(f"   Locations: {len(location_ids)}")
        
        # Analyze travel times
        non_diagonal = time_matrix[~np.eye(time_matrix.shape[0], dtype=bool)]
        non_zero_times = non_diagonal[non_diagonal > 0]
        
        if len(non_zero_times) > 0:
            min_time = np.min(non_zero_times)
            max_time = np.max(non_zero_times)
            mean_time = np.mean(non_zero_times)
            median_time = np.median(non_zero_times)
            
            print(f"   Travel time range: {min_time:.1f} - {max_time:.1f} minutes")
            print(f"   Mean: {mean_time:.1f} min ({mean_time/60:.1f} hr)")
            print(f"   Median: {median_time:.1f} min ({median_time/60:.1f} hr)")
            
            # Check if times are realistic for India
            if median_time > 30 and median_time < 600 and max_time < 1200:  # 30min to 10hr, max 20hr
                print("   ✅ Travel times look realistic for India")
                return True
            elif median_time > 1000:  # > 16 hours median
                print("   ❌ Travel times still too high - fix didn't work")
                return False
            else:
                print(f"   ⚠️ Travel times unclear - manual inspection needed")
                return None
        else:
            print("   ❌ No non-zero travel times found!")
            return False
            
    except Exception as e:
        print(f"   ❌ Error loading matrix: {e}")
        return False

def validate_candidate_generation():
    """Test candidate generation with real scenario."""
    print("\n🔄 VALIDATING CANDIDATE GENERATION")
    print("-" * 40)
    
    try:
        # Import required modules
        from models.driver_state import DriverState, DailyAssignment
        from opt.candidate_gen_v2 import CandidateGeneratorV2
        
        # Load distance matrix
        dist_data = np.load("data/dist_matrix.npz", allow_pickle=True)
        distance_matrix = dist_data['time']
        location_ids = dist_data['ids']
        location_to_index = {str(loc): i for i, loc in enumerate(location_ids)}
        
        # Create simple test scenario
        test_locations = [str(loc) for loc in location_ids[:3]]
        print(f"   Using test locations: {test_locations}")
        
        # Create driver at location 0, with future trip from location 1
        driver_state = DriverState(driver_id="test_driver", route_id="test_route")
        
        # Past assignment ending at location 0
        past_assignment = DailyAssignment(
            trip_id="past_trip",
            start_time=datetime(2025, 1, 15, 8, 0),
            end_time=datetime(2025, 1, 15, 12, 0),
            duration_minutes=240,
            start_location='DEPOT',
            end_location=test_locations[0]
        )
        driver_state.add_assignment('2025-01-15', past_assignment)
        
        # Future assignment starting from location 1
        future_assignment = DailyAssignment(
            trip_id="future_trip",
            start_time=datetime(2025, 1, 15, 16, 0),
            end_time=datetime(2025, 1, 15, 20, 0),
            duration_minutes=240,
            start_location=test_locations[1],
            end_location='DEPOT'
        )
        driver_state.add_assignment('2025-01-15', future_assignment)
        
        # Create disrupted trip from location 0 to location 2
        disrupted_trip = {
            'id': 'validation_trip',
            'start_time': datetime(2025, 1, 15, 13, 0),
            'end_time': datetime(2025, 1, 15, 15, 0),
            'duration_minutes': 120,
            'start_location': test_locations[0],
            'end_location': test_locations[2]
        }
        
        # Expected distances from matrix
        expected_distance_to_next = distance_matrix[location_to_index[test_locations[2]], location_to_index[test_locations[1]]]
        
        print(f"   Driver currently at: {test_locations[0]}")
        print(f"   Disrupted trip: {test_locations[0]} -> {test_locations[2]}")
        print(f"   Driver next trip from: {test_locations[1]}")
        print(f"   Expected deadhead to next trip: {expected_distance_to_next:.1f} minutes")
        
        # Initialize candidate generator
        driver_states = {"test_driver": driver_state}
        candidate_generator = CandidateGeneratorV2(
            driver_states=driver_states,
            distance_matrix=distance_matrix,
            location_to_index=location_to_index
        )
        
        # Generate candidates
        candidates = candidate_generator.generate_candidates(
            disrupted_trip, include_cascades=False, include_outsource=False
        )
        
        print(f"   Generated {len(candidates)} candidates:")
        
        realistic_deadheads = 0
        zero_deadheads = 0
        
        for i, candidate in enumerate(candidates):
            candidate_type = getattr(candidate, 'candidate_type', 'unknown')
            driver_id = getattr(candidate, 'assigned_driver_id', None)
            deadhead_minutes = getattr(candidate, 'deadhead_minutes', 0)
            
            print(f"     [{i}] {candidate_type} | Driver: {driver_id} | Deadhead: {deadhead_minutes:.1f} min")
            
            if deadhead_minutes == 0:
                zero_deadheads += 1
            elif deadhead_minutes > 10 and deadhead_minutes < 1000:  # Reasonable range
                realistic_deadheads += 1
        
        # Analyze results
        if len(candidates) == 0:
            print("   ❌ No candidates generated!")
            return False
        elif zero_deadheads == len(candidates):
            print("   ⚠️ All candidates have zero deadhead - may still be an issue")
            return None
        elif realistic_deadheads > 0:
            print("   ✅ Found candidates with realistic deadhead values")
            return True
        else:
            print("   ❌ No realistic deadhead values found")
            return False
            
    except Exception as e:
        print(f"   ❌ Error in candidate generation test: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_against_trip_data():
    """Compare distance matrix values against original trip data."""
    print("\n📊 VALIDATING AGAINST TRIP DATA")
    print("-" * 40)
    
    try:
        # Load trip data
        trips_df = pd.read_csv("data/processed/trips.csv")
        trip_distances = trips_df['segment_osrm_distance'].dropna()
        
        print(f"   Trip distances - Min: {trip_distances.min():.1f}km, Max: {trip_distances.max():.1f}km, Mean: {trip_distances.mean():.1f}km")
        
        # Load distance matrix
        dist_data = np.load("data/dist_matrix.npz", allow_pickle=True)
        time_matrix = dist_data['time']
        
        # Convert times back to distances (assuming 60 km/h)
        distance_matrix_km = time_matrix / 60 * 60  # time_min / (km/h) * (min/h) = km... wait, this is wrong
        # Correct: time_min * (km/h) / (min/h) = time_min * 60km/h / 60min/h = time_min * 1 km/min = distance_km
        distance_matrix_km = time_matrix * 60 / 60  # This simplifies to just time_matrix, which is wrong
        
        # Actually: time = distance / speed, so distance = time * speed
        # time_minutes = distance_km / speed_kmph * 60
        # So: distance_km = time_minutes * speed_kmph / 60
        distance_matrix_km = time_matrix * 60 / 60  # speed=60kmph
        
        non_diagonal = distance_matrix_km[~np.eye(distance_matrix_km.shape[0], dtype=bool)]
        matrix_distances = non_diagonal[non_diagonal > 0]
        
        if len(matrix_distances) > 0:
            print(f"   Matrix distances - Min: {matrix_distances.min():.1f}km, Max: {matrix_distances.max():.1f}km, Mean: {matrix_distances.mean():.1f}km")
            
            # Compare ranges
            trip_range = (trip_distances.min(), trip_distances.max())
            matrix_range = (matrix_distances.min(), matrix_distances.max())
            
            # Check if ranges are similar (within 2x factor)
            range_ratio_min = matrix_range[0] / trip_range[0] if trip_range[0] > 0 else float('inf')
            range_ratio_max = matrix_range[1] / trip_range[1] if trip_range[1] > 0 else float('inf')
            
            print(f"   Range comparison - Min ratio: {range_ratio_min:.2f}x, Max ratio: {range_ratio_max:.2f}x")
            
            if 0.5 <= range_ratio_min <= 2.0 and 0.5 <= range_ratio_max <= 2.0:
                print("   ✅ Distance ranges are reasonably consistent")
                return True
            else:
                print("   ❌ Distance ranges are inconsistent - may indicate scaling issues")
                return False
        else:
            print("   ❌ No distances found in matrix")
            return False
            
    except Exception as e:
        print(f"   ❌ Error comparing with trip data: {e}")
        return False

def main():
    """Run all validations."""
    print("🔍 VALIDATION: Checking if Distance Matrix Fix Worked")
    print("=" * 60)
    
    if not setup_path():
        print("❌ Could not setup Python path")
        return 1
    
    results = []
    
    # Run all validation tests
    results.append(("Coordinates", validate_coordinates()))
    results.append(("Distance Matrix", validate_distance_matrix()))
    results.append(("Trip Data Comparison", validate_against_trip_data()))
    results.append(("Candidate Generation", validate_candidate_generation()))
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    unclear = 0
    
    for test_name, result in results:
        if result is True:
            print(f"✅ {test_name}: PASSED")
            passed += 1
        elif result is False:
            print(f"❌ {test_name}: FAILED")
            failed += 1
        else:
            print(f"⚠️ {test_name}: UNCLEAR")
            unclear += 1
    
    print(f"\nResults: {passed} passed, {failed} failed, {unclear} unclear")
    
    if failed == 0 and passed > 0:
        print("\n🎉 SUCCESS: Distance matrix fix appears to have worked!")
        print("Your optimization should now use realistic travel times.")
    elif failed > 0:
        print("\n❌ ISSUES DETECTED: Some validations failed.")
        print("The fix may not have worked completely - check the failures above.")
    else:
        print("\n⚠️ UNCLEAR: Results are ambiguous - manual inspection needed.")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)