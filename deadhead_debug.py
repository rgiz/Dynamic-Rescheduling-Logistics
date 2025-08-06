#!/usr/bin/env python3
"""
Deadhead Debugging Script
=========================

Standalone script to diagnose zero deadhead mileage issue.
Place in project root and run directly.

Usage:
    python deadhead_debug.py
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
    else:
        print(f"❌ Source directory not found at {src_path}")
        print("   Make sure you're running from project root")
        return False

def test_imports():
    """Test if required modules can be imported."""
    try:
        from models.driver_state import DriverState, DailyAssignment
        from opt.candidate_gen_v2 import CandidateGeneratorV2
        from opt.cpsat_model_v2 import CPSATOptimizer
        return True, (DriverState, DailyAssignment, CandidateGeneratorV2, CPSATOptimizer)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False, None

def test_distance_matrix():
    """Point 2: Test distance matrix validity."""
    print("\n📍 TESTING DISTANCE MATRIX VALIDATION")
    print("-" * 50)
    
    dist_file = Path("data/dist_matrix.npz")
    if not dist_file.exists():
        print(f"❌ Distance matrix not found at {dist_file}")
        print("   Expected file: data/dist_matrix.npz")
        return None, None, None
    
    try:
        dist_data = np.load(str(dist_file), allow_pickle=True)
        distance_matrix = dist_data['time']
        location_ids = dist_data['ids']
        location_to_index = {str(loc): i for i, loc in enumerate(location_ids)}
        
        print(f"✅ Loaded distance matrix: {distance_matrix.shape}")
        print(f"✅ Number of locations: {len(location_ids)}")
        
        # Test 1: Matrix is not all zeros
        non_zero_distances = np.count_nonzero(distance_matrix)
        total_entries = distance_matrix.size
        non_zero_ratio = non_zero_distances / total_entries
        
        print(f"\n📊 Matrix Analysis:")
        print(f"   Non-zero distances: {non_zero_distances:,}/{total_entries:,} ({non_zero_ratio:.1%})")
        
        if non_zero_ratio < 0.5:
            print("   ⚠️ WARNING: Too many zero distances - matrix may be corrupted")
        else:
            print("   ✅ Good ratio of non-zero distances")
        
        # Test 2: Diagonal should be zero
        diagonal_zeros = np.count_nonzero(np.diag(distance_matrix) == 0)
        diagonal_size = len(np.diag(distance_matrix))
        print(f"   Diagonal zeros: {diagonal_zeros}/{diagonal_size}")
        
        if diagonal_zeros != diagonal_size:
            print("   ⚠️ WARNING: Diagonal should be all zeros (self-distances)")
        else:
            print("   ✅ Diagonal correctly all zeros")
        
        # Test 3: Sample specific distances
        print(f"\n📏 Sample Distances:")
        sample_count = 0
        zero_sample_count = 0
        
        for i in range(min(5, len(location_ids))):
            for j in range(i+1, min(i+3, len(location_ids))):
                distance = distance_matrix[i, j]
                loc1_str = str(location_ids[i])[:12]
                loc2_str = str(location_ids[j])[:12]
                print(f"   {loc1_str} -> {loc2_str}: {distance:.1f} minutes")
                sample_count += 1
                if distance == 0:
                    zero_sample_count += 1
        
        if sample_count > 0:
            zero_sample_ratio = zero_sample_count / sample_count
            print(f"   Zero distances in samples: {zero_sample_count}/{sample_count} ({zero_sample_ratio:.1%})")
            
            if zero_sample_ratio > 0.5:
                print("   ⚠️ WARNING: Too many zero distances between different locations")
            else:
                print("   ✅ Good mix of non-zero distances")
        
        # Test 4: Distribution analysis
        non_diagonal_mask = ~np.eye(distance_matrix.shape[0], dtype=bool)
        non_diagonal = distance_matrix[non_diagonal_mask]
        non_zero_non_diagonal = non_diagonal[non_diagonal > 0]
        
        if len(non_zero_non_diagonal) > 0:
            print(f"\n📈 Distance Distribution:")
            print(f"   Mean distance: {np.mean(non_zero_non_diagonal):.1f} minutes")
            print(f"   Median distance: {np.median(non_zero_non_diagonal):.1f} minutes")
            print(f"   Min distance: {np.min(non_zero_non_diagonal):.1f} minutes")
            print(f"   Max distance: {np.max(non_zero_non_diagonal):.1f} minutes")
        
        return distance_matrix, location_ids, location_to_index
        
    except Exception as e:
        print(f"❌ Error loading distance matrix: {e}")
        return None, None, None

def create_test_scenario(distance_matrix, location_ids, location_to_index, DriverState, DailyAssignment):
    """Create test scenario with known locations."""
    print("\n🔧 CREATING TEST SCENARIO")
    print("-" * 50)
    
    # Use real location IDs if available
    if location_ids is not None and len(location_ids) >= 4:
        test_locations = [str(loc) for loc in location_ids[:4]]
    else:
        test_locations = ['LOC_A', 'LOC_B', 'LOC_C', 'LOC_D']
    
    print(f"Using test locations: {test_locations}")
    
    # Create driver states
    driver_states = {}
    
    for i, end_location in enumerate(test_locations[:3]):
        driver_id = f"test_driver_{i}"
        driver_state = DriverState(driver_id=driver_id, route_id=f"test_route_{i}")
        
        # Add assignment that ends at specific location
        past_assignment = DailyAssignment(
            trip_id=f"past_trip_{i}",
            start_time=datetime(2025, 1, 15, 8, 0),
            end_time=datetime(2025, 1, 15, 12, 0),
            duration_minutes=240,
            start_location='DEPOT',
            end_location=end_location
        )
        driver_state.add_assignment('2025-01-15', past_assignment)
        
        # Add future assignment starting from different location
        next_location = test_locations[(i + 2) % len(test_locations)]
        future_assignment = DailyAssignment(
            trip_id=f"future_trip_{i}",
            start_time=datetime(2025, 1, 15, 16, 0),
            end_time=datetime(2025, 1, 15, 20, 0),
            duration_minutes=240,
            start_location=next_location,
            end_location='DEPOT'
        )
        driver_state.add_assignment('2025-01-15', future_assignment)
        
        driver_states[driver_id] = driver_state
        
        print(f"   Driver {driver_id}:")
        print(f"     Currently at: {end_location}")
        print(f"     Next trip from: {next_location}")
        
        # Calculate expected deadhead if distance matrix available
        if distance_matrix is not None and location_to_index is not None:
            try:
                idx1 = location_to_index[end_location]
                idx2 = location_to_index[next_location]
                distance = distance_matrix[idx1, idx2]
                print(f"     Distance to next: {distance:.1f} minutes")
            except (KeyError, IndexError):
                print(f"     Distance to next: Unknown (location not in matrix)")
    
    # Create disrupted trips
    disrupted_trips = []
    for i, start_location in enumerate(test_locations[:2]):
        end_location = test_locations[(i + 1) % len(test_locations)]
        
        disrupted_trips.append({
            'id': f'test_disrupted_{i}',
            'start_time': datetime(2025, 1, 15, 13, 0),
            'end_time': datetime(2025, 1, 15, 15, 0),
            'duration_minutes': 120,
            'start_location': start_location,
            'end_location': end_location
        })
        
        print(f"   Disrupted trip {i}: {start_location} -> {end_location}")
    
    return driver_states, disrupted_trips

def get_driver_current_location(driver_state, date, time):
    """Get driver's current location at given time."""
    if not hasattr(driver_state, 'daily_assignments') or date not in driver_state.daily_assignments:
        return None
    
    assignments = driver_state.daily_assignments[date]
    for assignment in reversed(assignments):
        if assignment.end_time <= time:
            return assignment.end_location
    
    return None

def get_driver_next_location(driver_state, date, time):
    """Get driver's next location after given time."""
    if not hasattr(driver_state, 'daily_assignments') or date not in driver_state.daily_assignments:
        return None
    
    assignments = driver_state.daily_assignments[date]
    for assignment in assignments:
        if assignment.start_time >= time:
            return assignment.start_location
    
    return None

def get_distance(distance_matrix, location_to_index, loc1, loc2):
    """Get distance between two locations."""
    if not distance_matrix or not location_to_index:
        return 30.0  # Default
    
    try:
        idx1 = location_to_index[str(loc1)]
        idx2 = location_to_index[str(loc2)]
        return float(distance_matrix[idx1, idx2])
    except (KeyError, IndexError):
        return 30.0  # Default

def analyze_candidate_costs_and_constraints(candidates, trip, driver_states):
    """Analyze why certain candidates might be rejected - cost and constraint analysis."""
    print(f"         💰 DETAILED CANDIDATE ANALYSIS:")
    
    for i, candidate in enumerate(candidates):
        candidate_type = getattr(candidate, 'candidate_type', 'unknown')
        driver_id = getattr(candidate, 'assigned_driver_id', None)
        deadhead_minutes = getattr(candidate, 'deadhead_minutes', 0)
        delay_minutes = getattr(candidate, 'delay_minutes', 0)
        total_cost = getattr(candidate, 'total_cost', 0)
        emergency_rest = getattr(candidate, 'emergency_rest_used', False)
        feasible = getattr(candidate, 'is_feasible', True)
        violations = getattr(candidate, 'violations', [])
        
        print(f"         [{i}] {candidate_type} | Driver: {driver_id}")
        print(f"             Deadhead: {deadhead_minutes:.1f}m | Delay: {delay_minutes:.1f}m")
        print(f"             Total Cost: £{total_cost:.2f} | Emergency Rest: {emergency_rest}")
        print(f"             Feasible: {feasible} | Violations: {violations}")
        
        # Analyze cost components if possible
        if candidate_type != 'outsource' and driver_id and driver_id in driver_states:
            analyze_cost_breakdown(candidate, deadhead_minutes, delay_minutes, emergency_rest)
            analyze_constraint_feasibility(candidate, driver_id, driver_states[driver_id], trip)
        
        print()

def analyze_cost_breakdown(candidate, deadhead_minutes, delay_minutes, emergency_rest):
    """Analyze the cost breakdown for a candidate."""
    # Try to reverse-engineer cost components based on common patterns
    total_cost = getattr(candidate, 'total_cost', 0)
    
    print(f"             💸 Cost Breakdown Analysis:")
    
    # Common cost components (adjust these based on your actual cost model)
    estimated_deadhead_cost = deadhead_minutes * 0.5  # £0.50 per minute
    estimated_delay_cost = delay_minutes * 1.0        # £1.00 per minute  
    estimated_admin_cost = 10.0                       # £10 admin cost
    estimated_emergency_penalty = 50.0 if emergency_rest else 0.0
    
    estimated_total = estimated_deadhead_cost + estimated_delay_cost + estimated_admin_cost + estimated_emergency_penalty
    
    print(f"               - Deadhead cost (est): £{estimated_deadhead_cost:.2f}")
    print(f"               - Delay cost (est): £{estimated_delay_cost:.2f}")
    print(f"               - Admin cost (est): £{estimated_admin_cost:.2f}")
    print(f"               - Emergency penalty (est): £{estimated_emergency_penalty:.2f}")
    print(f"               - Estimated total: £{estimated_total:.2f}")
    print(f"               - Actual total: £{total_cost:.2f}")
    
    diff = abs(total_cost - estimated_total)
    if diff > 1.0:
        print(f"               ⚠️ Large difference (£{diff:.2f}) - cost model may differ")

def analyze_constraint_feasibility(candidate, driver_id, driver_state, trip):
    """Analyze constraint feasibility for a candidate."""
    print(f"             🚦 Constraint Feasibility Analysis:")
    
    # Check daily duty limit
    trip_date = '2025-01-15'  # Our test date
    current_usage = get_daily_usage(driver_state, trip_date)
    trip_duration = trip.get('duration_minutes', 120)
    deadhead_minutes = getattr(candidate, 'deadhead_minutes', 0)
    
    total_usage_with_trip = current_usage + trip_duration + deadhead_minutes
    duty_limit = 13 * 60  # 13 hours in minutes
    
    print(f"               - Current daily usage: {current_usage:.0f} minutes")
    print(f"               - Trip duration: {trip_duration:.0f} minutes")
    print(f"               - Deadhead time: {deadhead_minutes:.0f} minutes")
    print(f"               - Total with trip: {total_usage_with_trip:.0f} minutes")
    print(f"               - Daily limit: {duty_limit} minutes")
    
    if total_usage_with_trip > duty_limit:
        print(f"               ❌ DUTY LIMIT VIOLATION: Exceeds by {total_usage_with_trip - duty_limit:.0f} minutes")
    else:
        remaining = duty_limit - total_usage_with_trip
        print(f"               ✅ Within duty limit (remaining: {remaining:.0f} minutes)")
    
    # Check rest period constraints
    emergency_rest = getattr(candidate, 'emergency_rest_used', False)
    if emergency_rest:
        print(f"               ⚠️ Requires emergency rest (9h instead of 11h)")
        
        # Check emergency rest quota
        current_emergency_usage = getattr(driver_state, 'emergency_rests_used_this_week', 0)
        max_emergency = 2
        print(f"               - Emergency rests used this week: {current_emergency_usage}/{max_emergency}")
        
        if current_emergency_usage >= max_emergency:
            print(f"               ❌ EMERGENCY QUOTA VIOLATION: Already at maximum")
        else:
            print(f"               ✅ Emergency rest available")

def get_daily_usage(driver_state, date):
    """Get current daily usage for a driver."""
    if not hasattr(driver_state, 'daily_assignments') or date not in driver_state.daily_assignments:
        return 0
    
    assignments = driver_state.daily_assignments[date]
    total_usage = sum(getattr(assignment, 'duration_minutes', 0) for assignment in assignments)
    return total_usage
    """Point 3: Test candidate generation deadhead calculation."""
    print("\n🔄 TESTING CANDIDATE GENERATION")
    print("-" * 50)
    
    candidate_generator = CandidateGeneratorV2(
        driver_states=driver_states,
        distance_matrix=distance_matrix,
        location_to_index=location_to_index
    )
    
    all_candidates = {}
    
    for trip in disrupted_trips:
        print(f"\n🚛 Testing trip {trip['id']}:")
        print(f"   Route: {trip['start_location']} -> {trip['end_location']}")
        print(f"   Duration: {trip['duration_minutes']} minutes")
        
        # Generate candidates
        try:
            candidates = candidate_generator.generate_candidates(
                trip, include_cascades=True, include_outsource=True
            )
        except Exception as e:
            print(f"   ❌ Error generating candidates: {e}")
            continue
        
        all_candidates[trip['id']] = candidates
        print(f"   Generated {len(candidates)} candidates")
        
        if len(candidates) == 0:
            print("   ⚠️ WARNING: No candidates generated!")
            continue
        
def test_candidate_generation(distance_matrix, location_to_index, driver_states, disrupted_trips, CandidateGeneratorV2):
    """Point 3: Test candidate generation deadhead calculation."""
    print("\n🔄 TESTING CANDIDATE GENERATION")
    print("-" * 50)
    
    candidate_generator = CandidateGeneratorV2(
        driver_states=driver_states,
        distance_matrix=distance_matrix,
        location_to_index=location_to_index
    )
    
    all_candidates = {}
    
    for trip in disrupted_trips:
        print(f"\n🚛 Testing trip {trip['id']}:")
        print(f"   Route: {trip['start_location']} -> {trip['end_location']}")
        print(f"   Duration: {trip['duration_minutes']} minutes")
        
        # Generate candidates
        try:
            candidates = candidate_generator.generate_candidates(
                trip, include_cascades=True, include_outsource=True
            )
        except Exception as e:
            print(f"   ❌ Error generating candidates: {e}")
            continue
        
        all_candidates[trip['id']] = candidates
        print(f"   Generated {len(candidates)} candidates")
        
        if len(candidates) == 0:
            print("   ⚠️ WARNING: No candidates generated!")
            continue
        
        zero_deadhead_count = 0
        non_zero_deadhead_count = 0
        
        for i, candidate in enumerate(candidates):
            candidate_type = getattr(candidate, 'candidate_type', 'unknown')
            driver_id = getattr(candidate, 'assigned_driver_id', None)
            deadhead_minutes = getattr(candidate, 'deadhead_minutes', 0)
            delay_minutes = getattr(candidate, 'delay_minutes', 0)
            total_cost = getattr(candidate, 'total_cost', 0)
            
            driver_str = str(driver_id)[:12] if driver_id else 'None'
            print(f"     [{i}] {candidate_type:8} | Driver: {driver_str:12} | Deadhead: {deadhead_minutes:6.1f}m | Cost: £{total_cost:6.2f}")
            
            if candidate_type != 'outsource':
                if deadhead_minutes == 0:
                    zero_deadhead_count += 1
                    print(f"         🔍 ZERO DEADHEAD ANALYSIS:")
                    
                    # Detailed analysis for zero deadhead
                    if driver_id and driver_id in driver_states:
                        driver_state = driver_states[driver_id]
                        
                        # Find driver's current location
                        current_location = get_driver_current_location(driver_state, '2025-01-15', trip['start_time'])
                        next_location = get_driver_next_location(driver_state, '2025-01-15', trip['end_time'])
                        
                        print(f"         - Driver currently at: {current_location}")
                        print(f"         - Trip starts at: {trip['start_location']}")
                        print(f"         - Trip ends at: {trip['end_location']}")
                        print(f"         - Driver's next trip at: {next_location}")
                        
                        # Check if zero deadhead is justified
                        if current_location and current_location != trip['start_location']:
                            distance = get_distance(distance_matrix, location_to_index, current_location, trip['start_location'])
                            if distance > 0:
                                print(f"         ⚠️ Should have deadhead to start: {distance:.1f} minutes")
                            else:
                                print(f"         ✅ Zero deadhead to start justified (locations same or distance=0)")
                        else:
                            print(f"         ✅ Zero deadhead to start justified (driver already there)")
                        
                        if next_location and next_location != trip['end_location']:
                            distance = get_distance(distance_matrix, location_to_index, trip['end_location'], next_location)
                            if distance > 0:
                                print(f"         ⚠️ Should have deadhead to next: {distance:.1f} minutes")
                            else:
                                print(f"         ✅ Zero deadhead to next justified (locations same or distance=0)")
                        else:
                            print(f"         ✅ Zero deadhead to next justified (trip ends where next starts)")
                else:
                    non_zero_deadhead_count += 1
        
        # Detailed candidate analysis
        print(f"\n   🔬 DETAILED CANDIDATE ANALYSIS FOR {trip['id']}:")
        analyze_candidate_costs_and_constraints(candidates, trip, driver_states)
        
        # Summary
        total_non_outsource = zero_deadhead_count + non_zero_deadhead_count
        if total_non_outsource > 0:
            zero_ratio = zero_deadhead_count / total_non_outsource
            print(f"   Summary: {zero_deadhead_count} zero deadhead, {non_zero_deadhead_count} non-zero ({zero_ratio:.1%} zero)")
            
            if zero_ratio == 1.0:
                print("   ⚠️ WARNING: ALL non-outsource candidates have zero deadhead!")
            elif zero_ratio > 0.8:
                print("   ⚠️ WARNING: Very high proportion of zero deadhead candidates")
            else:
                print("   ✅ Mix of zero and non-zero deadhead found")
    
    return all_candidates

def test_assignment_logic(distance_matrix, location_to_index, driver_states, disrupted_trips, candidates_per_trip, CPSATOptimizer):
    """Point 4: Test assignment logic."""
    print("\n🚀 TESTING ASSIGNMENT LOGIC")
    print("-" * 50)
    
    # Initialize optimizer
    try:
        cpsat_optimizer = CPSATOptimizer(
            driver_states=driver_states,
            distance_matrix=distance_matrix,
            location_to_index=location_to_index
        )
    except Exception as e:
        print(f"❌ Error initializing optimizer: {e}")
        return
    
    # Store original candidate deadheads for comparison
    original_deadheads = {}
    for trip_id, candidates in candidates_per_trip.items():
        original_deadheads[trip_id] = [(i, getattr(c, 'deadhead_minutes', 0)) for i, c in enumerate(candidates)]
    
    # Run optimization
    try:
        solution = cpsat_optimizer.cpsat_model.solve(
            disrupted_trips,
            candidates_per_trip,
            {'cost_weight': 0.5, 'service_weight': 0.5}
        )
        
        print(f"📊 Optimization Results:")
        print(f"   Status: {solution.status}")
        print(f"   Assignments made: {len(solution.assignments)}/{len(disrupted_trips)}")
        print(f"   Objective value: {solution.objective_value:.2f}")
        print(f"   Solve time: {solution.solve_time_seconds:.2f}s")
        
        if len(solution.assignments) == 0:
            print("   ⚠️ WARNING: No assignments made - optimization may have failed")
            return
        
        # Analyze each assignment
        for assignment in solution.assignments:
            trip_id = assignment.get('trip_id')
            assigned_deadhead = assignment.get('deadhead_minutes', 0)
            assigned_driver = assignment.get('driver_id')
            assignment_type = assignment.get('type', 'unknown')
            assigned_cost = assignment.get('total_cost', 0)
            
            print(f"\n   📝 Assignment for {trip_id}:")
            print(f"      Type: {assignment_type}")
            print(f"      Driver: {assigned_driver}")
            print(f"      Deadhead: {assigned_deadhead:.1f} minutes")
            print(f"      Cost: £{assigned_cost:.2f}")
            
            # Compare with original candidates
            if trip_id in original_deadheads:
                candidate_deadheads = [dh for _, dh in original_deadheads[trip_id]]
                candidates = candidates_per_trip.get(trip_id, [])
                
                if candidate_deadheads and candidates:
                    min_dh = min(candidate_deadheads)
                    max_dh = max(candidate_deadheads)
                    avg_dh = sum(candidate_deadheads) / len(candidate_deadheads)
                    
                    print(f"      Original candidates: {min_dh:.1f} - {max_dh:.1f} min (avg: {avg_dh:.1f})")
                    
                    # Find zero deadhead alternatives
                    zero_candidates = [(i, c) for i, c in enumerate(candidates) 
                                     if getattr(c, 'deadhead_minutes', 0) == 0 
                                     and getattr(c, 'candidate_type', '') != 'outsource']
                    
                    if zero_candidates and assigned_deadhead > 0:
                        print(f"      🔍 OPTIMIZER CHOICE ANALYSIS:")
                        print(f"         Selected: {assigned_deadhead:.1f}min deadhead, £{assigned_cost:.2f} cost")
                        
                        for idx, zero_candidate in zero_candidates:
                            zero_cost = getattr(zero_candidate, 'total_cost', 0)
                            zero_driver = getattr(zero_candidate, 'assigned_driver_id', None)
                            zero_feasible = getattr(zero_candidate, 'is_feasible', True)
                            zero_violations = getattr(zero_candidate, 'violations', [])
                            
                            print(f"         Alternative: 0.0min deadhead, £{zero_cost:.2f} cost, Driver: {zero_driver}")
                            print(f"                     Feasible: {zero_feasible}, Violations: {zero_violations}")
                            
                            # Analyze why zero option wasn't chosen
                            cost_diff = assigned_cost - zero_cost
                            print(f"                     Cost difference: £{cost_diff:.2f} ({'SELECTED CHEAPER' if cost_diff < 0 else 'SELECTED MORE EXPENSIVE'})")
                            
                            if not zero_feasible:
                                print(f"                     ❌ REASON: Zero deadhead option was INFEASIBLE")
                            elif len(zero_violations) > 0:
                                print(f"                     ❌ REASON: Zero deadhead option had VIOLATIONS: {zero_violations}")
                            elif cost_diff < -1:
                                print(f"                     ❌ REASON: Zero deadhead option was £{abs(cost_diff):.2f} MORE EXPENSIVE")
                            else:
                                print(f"                     ❓ UNCLEAR: Zero deadhead seems feasible and cheaper - potential issue!")
                    
                    # Check if assignment is within candidate range
                    if assigned_deadhead < min_dh - 0.1 or assigned_deadhead > max_dh + 0.1:
                        print(f"      ⚠️ WARNING: Assigned deadhead outside candidate range!")
                    
                    # Count zero vs non-zero options
                    zero_candidates_count = sum(1 for dh in candidate_deadheads if dh == 0)
                    non_zero_candidates_count = len(candidate_deadheads) - zero_candidates_count
                    
                    if assigned_deadhead == 0 and non_zero_candidates_count > 0:
                        print(f"      📊 Selected zero deadhead (had {zero_candidates_count} zero, {non_zero_candidates_count} non-zero options)")
                    elif assigned_deadhead > 0 and zero_candidates_count > 0:
                        print(f"      📊 Selected non-zero deadhead despite {zero_candidates_count} zero options available")
                        print(f"          (This suggests zero options were infeasible or more expensive)")
        
        # Overall analysis
        zero_assigned = sum(1 for a in solution.assignments if a.get('deadhead_minutes', 0) == 0)
        total_assigned = len(solution.assignments)
        zero_assigned_ratio = zero_assigned / total_assigned if total_assigned > 0 else 0
        
        print(f"\n📈 Overall Assignment Analysis:")
        print(f"   Zero deadhead assignments: {zero_assigned}/{total_assigned} ({zero_assigned_ratio:.1%})")
        
        if zero_assigned_ratio == 1.0:
            print("   ⚠️ CRITICAL: ALL assignments have zero deadhead!")
        elif zero_assigned_ratio > 0.8:
            print("   ⚠️ WARNING: Very high proportion of zero deadhead assignments")
        else:
            print("   ✅ Mix of zero and non-zero deadhead assignments")
        
    except Exception as e:
        print(f"❌ Optimization failed: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Run all debugging tests."""
    print("🕵️ DEADHEAD DEBUGGING ANALYSIS")
    print("=" * 60)
    
    # Setup
    if not setup_path():
        return 1
    
    # Test imports
    imports_ok, modules = test_imports()
    if not imports_ok:
        return 1
    
    DriverState, DailyAssignment, CandidateGeneratorV2, CPSATOptimizer = modules
    print("✅ All required modules imported successfully")
    
    # Test 1: Distance Matrix
    distance_matrix, location_ids, location_to_index = test_distance_matrix()
    if distance_matrix is None:
        print("⚠️ Continuing with limited testing (no distance matrix)")
    
    # Test 2: Create test scenario
    driver_states, disrupted_trips = create_test_scenario(
        distance_matrix, location_ids, location_to_index, DriverState, DailyAssignment
    )
    
    # Test 3: Candidate generation
    candidates_per_trip = test_candidate_generation(
        distance_matrix, location_to_index, driver_states, disrupted_trips, CandidateGeneratorV2
    )
    
    # Test 4: Assignment logic
    if candidates_per_trip:
        test_assignment_logic(
            distance_matrix, location_to_index, driver_states, 
            disrupted_trips, candidates_per_trip, CPSATOptimizer
        )
    else:
        print("⚠️ Skipping assignment logic test (no candidates generated)")
    
    print("\n" + "=" * 60)
    print("🏁 DEBUGGING ANALYSIS COMPLETE")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)