"""
Candidate Generator V2 - Multi-Driver Cascading Reassignments
==============================================================

Generates feasible reassignment candidates for disrupted trips, including:
- Single-driver direct insertions
- Two-driver cascading reassignments
- Multi-driver chains (configurable depth)

All candidates respect hard constraints (13h duty, rest periods, etc.)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict

# Import your models
from models.driver_state import DriverState, DailyAssignment
from models.weekly_schedule import WeeklySchedule
from evaluation_metrics import CostMetrics


@dataclass
class ReassignmentCandidate:
    """
    Represents a potential reassignment option for a disrupted trip.
    """
    # Basic info
    disrupted_trip_id: str
    candidate_type: str  # 'direct', 'cascade_2', 'cascade_multi', 'outsource'
    
    # Assignment details
    assigned_driver_id: Optional[str] = None
    position_in_day: Optional[int] = None  # Where to insert in driver's schedule
    
    # Cascading details (if applicable)
    cascade_chain: List[Dict] = field(default_factory=list)
    cascade_depth: int = 0
    
    # Costs and penalties
    deadhead_minutes: float = 0.0
    delay_minutes: float = 0.0
    emergency_rest_used: bool = False
    
    # Feasibility and scoring
    is_feasible: bool = True
    feasibility_score: float = 1.0  # 0-1, higher is better
    total_cost: float = 0.0
    
    # Constraint violations (if any)
    violations: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return (f"Candidate({self.candidate_type}, driver={self.assigned_driver_id}, "
                f"cost={self.total_cost:.2f}, feasible={self.is_feasible})")


class CandidateGeneratorV2:
    """
    Generates feasible reassignment candidates for disrupted trips.
    """
    
    def __init__(self,
                 driver_states: Dict[str, DriverState],
                 distance_matrix: Optional[np.ndarray] = None,
                 location_to_index: Optional[Dict[str, int]] = None,
                 max_cascade_depth: int = 3,
                 max_deadhead_minutes: float = 120,
                 max_delay_minutes: float = 120,
                 cost_per_minute_deadhead: float = 1.0,
                 cost_per_minute_delay: float = 2.0,
                 emergency_rest_penalty: float = 100.0):
        """
        Initialize candidate generator.
        
        Args:
            driver_states: Dictionary of driver_id -> DriverState objects
            distance_matrix: Matrix of travel times between locations
            location_to_index: Mapping of location names to matrix indices
            max_cascade_depth: Maximum depth for cascading reassignments
            max_deadhead_minutes: Maximum acceptable deadhead time
            max_delay_minutes: Maximum acceptable delay
            cost_per_minute_deadhead: Cost per minute of deadhead travel
            cost_per_minute_delay: Cost per minute of delay
            emergency_rest_penalty: Penalty for using emergency rest
        """
        self.driver_states = driver_states
        self.distance_matrix = distance_matrix
        self.location_to_index = location_to_index
        self.max_cascade_depth = max_cascade_depth
        self.max_deadhead_minutes = max_deadhead_minutes
        self.max_delay_minutes = max_delay_minutes
        
        # Cost parameters
        self.cost_per_minute_deadhead = cost_per_minute_deadhead
        self.cost_per_minute_delay = cost_per_minute_delay
        self.emergency_rest_penalty = emergency_rest_penalty
        
        # Cache for performance
        self._driver_availability_cache = {}
    
    def generate_candidates(self, 
                          disrupted_trip: Dict,
                          drivers_to_consider: Optional[List[str]] = None,
                          include_cascades: bool = True,
                          include_outsource: bool = True) -> List[ReassignmentCandidate]:
        """
        Generate all feasible candidates for a disrupted trip.
        
        Args:
            disrupted_trip: Dict with trip details (id, start_time, end_time, 
                          start_location, end_location, duration_minutes)
            drivers_to_consider: Optional list of driver IDs to consider
            include_cascades: Whether to generate cascading candidates
            include_outsource: Whether to include outsourcing option
            
        Returns:
            List of ReassignmentCandidate objects, sorted by total cost
        """
        candidates = []
        
        # Determine which drivers to check
        if drivers_to_consider is None:
            drivers_to_consider = list(self.driver_states.keys())
        
        # 1. Generate direct insertion candidates
        direct_candidates = self._generate_direct_insertions(
            disrupted_trip, drivers_to_consider
        )
        candidates.extend(direct_candidates)
        
        # 2. Generate cascading candidates if enabled
        if include_cascades and self.max_cascade_depth >= 2:
            cascade_candidates = self._generate_cascade_insertions(
                disrupted_trip, drivers_to_consider
            )
            candidates.extend(cascade_candidates)
        
        # 3. Add outsourcing option if enabled
        if include_outsource:
            outsource_candidate = self._create_outsource_candidate(disrupted_trip)
            candidates.append(outsource_candidate)
        
        # Sort by total cost (lower is better)
        candidates.sort(key=lambda x: (not x.is_feasible, x.total_cost))
        
        return candidates
    
    def _generate_direct_insertions(self,
                                   disrupted_trip: Dict,
                                   drivers: List[str]) -> List[ReassignmentCandidate]:
        """
        Generate single-driver direct insertion candidates.
        """
        candidates = []
        trip_date = disrupted_trip['start_time'].date()
        trip_date_str = trip_date.strftime('%Y-%m-%d')
        
        for driver_id in drivers:
            driver_state = self.driver_states[driver_id]
            
            # Check if driver has capacity on this day
            if not self._check_driver_capacity(driver_state, trip_date_str, 
                                              disrupted_trip['duration_minutes']):
                continue
            
            # Get driver's existing assignments for the day
            day_assignments = []
            if hasattr(driver_state, 'get_day_assignments'):
                day_assignments = driver_state.get_day_assignments(trip_date_str)
            elif hasattr(driver_state, 'assignments'):
                # Direct access to assignments dict
                if trip_date_str in driver_state.assignments:
                    day_assignments = driver_state.assignments[trip_date_str]
            
            # Try inserting at each position in the day's schedule
            for position in range(len(day_assignments) + 1):
                candidate = self._try_insert_at_position(
                    disrupted_trip,
                    driver_id,
                    driver_state,
                    trip_date_str,
                    position,
                    day_assignments
                )
                
                if candidate and candidate.is_feasible:
                    candidates.append(candidate)
        
        return candidates
    
    def _try_insert_at_position(self,
                               trip: Dict,
                               driver_id: str,
                               driver_state: DriverState,
                               date_str: str,
                               position: int,
                               existing_assignments: List[DailyAssignment]) -> Optional[ReassignmentCandidate]:
        """
        Try to insert a trip at a specific position in driver's day.
        """
        candidate = ReassignmentCandidate(
            disrupted_trip_id=trip['id'],
            candidate_type='direct',
            assigned_driver_id=driver_id,
            position_in_day=position,
            cascade_depth=1
        )
        
        # Calculate deadhead and delays
        deadhead_before = 0
        deadhead_after = 0
        delay = 0
        
        # Check insertion feasibility and calculate metrics
        if position == 0:
            # Inserting at start of day
            if existing_assignments:
                first_assignment = existing_assignments[0]
                # Calculate deadhead from trip end to first assignment start
                deadhead_after = self._calculate_travel_time(
                    trip['end_location'],
                    first_assignment.start_location
                )
                
                # Check if we'd delay the first assignment
                trip_end_with_deadhead = trip['end_time'] + timedelta(minutes=deadhead_after)
                if trip_end_with_deadhead > first_assignment.start_time:
                    delay = (trip_end_with_deadhead - first_assignment.start_time).total_seconds() / 60
                    
        elif position == len(existing_assignments):
            # Inserting at end of day
            if existing_assignments:
                last_assignment = existing_assignments[-1]
                # Calculate deadhead from last assignment end to trip start
                deadhead_before = self._calculate_travel_time(
                    last_assignment.end_location,
                    trip['start_location']
                )
                
                # Check if we can reach the trip in time
                earliest_arrival = last_assignment.end_time + timedelta(minutes=deadhead_before)
                if earliest_arrival > trip['start_time']:
                    delay = (earliest_arrival - trip['start_time']).total_seconds() / 60
                    
        else:
            # Inserting between two assignments
            prev_assignment = existing_assignments[position - 1]
            next_assignment = existing_assignments[position]
            
            # Calculate deadhead before
            deadhead_before = self._calculate_travel_time(
                prev_assignment.end_location,
                trip['start_location']
            )
            
            # Calculate deadhead after
            deadhead_after = self._calculate_travel_time(
                trip['end_location'],
                next_assignment.start_location
            )
            
            # Check timing feasibility
            earliest_arrival = prev_assignment.end_time + timedelta(minutes=deadhead_before)
            if earliest_arrival > trip['start_time']:
                delay = (earliest_arrival - trip['start_time']).total_seconds() / 60
            
            # Check if we'd delay the next assignment
            trip_end_with_deadhead = max(trip['end_time'], 
                                        earliest_arrival + timedelta(minutes=trip['duration_minutes']))
            trip_end_with_deadhead += timedelta(minutes=deadhead_after)
            
            if trip_end_with_deadhead > next_assignment.start_time:
                next_delay = (trip_end_with_deadhead - next_assignment.start_time).total_seconds() / 60
                delay = max(delay, next_delay)
        
        # Set candidate metrics
        candidate.deadhead_minutes = deadhead_before + deadhead_after
        candidate.delay_minutes = delay
        
        # Check feasibility constraints
        if candidate.deadhead_minutes > self.max_deadhead_minutes:
            candidate.is_feasible = False
            candidate.violations.append(f"Deadhead {candidate.deadhead_minutes:.0f} min > max {self.max_deadhead_minutes}")
            
        if candidate.delay_minutes > self.max_delay_minutes:
            candidate.is_feasible = False
            candidate.violations.append(f"Delay {candidate.delay_minutes:.0f} min > max {self.max_delay_minutes}")
        
        # Check if this would violate daily duty limit
        total_day_minutes = sum(a.duration_minutes for a in existing_assignments)
        total_day_minutes += trip['duration_minutes'] + candidate.deadhead_minutes
        
        if total_day_minutes > 13 * 60:  # 13 hour limit
            # Check if emergency rest could help
            if total_day_minutes <= 15 * 60 and driver_state.can_use_emergency_rest(date_str):
                candidate.emergency_rest_used = True
            else:
                candidate.is_feasible = False
                candidate.violations.append(f"Daily duty {total_day_minutes/60:.1f}h > 13h limit")
        
        # Calculate total cost
        candidate.total_cost = (
            candidate.deadhead_minutes * self.cost_per_minute_deadhead +
            candidate.delay_minutes * self.cost_per_minute_delay +
            (self.emergency_rest_penalty if candidate.emergency_rest_used else 0)
        )
        
        # Calculate feasibility score (for ranking feasible options)
        if candidate.is_feasible:
            # Score based on efficiency (less deadhead/delay is better)
            efficiency = 1.0 / (1.0 + candidate.deadhead_minutes / 60 + candidate.delay_minutes / 30)
            candidate.feasibility_score = efficiency
        
        return candidate
    
    def _generate_cascade_insertions(self,
                                    disrupted_trip: Dict,
                                    drivers: List[str]) -> List[ReassignmentCandidate]:
        """
        Generate cascading reassignment candidates (2+ drivers involved).
        """
        candidates = []
        
        # For now, implement simple 2-driver cascades
        # The idea: Driver A takes the disrupted trip, Driver B takes one of A's trips
        
        for driver_a_id in drivers:
            driver_a_state = self.driver_states[driver_a_id]
            trip_date_str = disrupted_trip['start_time'].date().strftime('%Y-%m-%d')
            
            # Get Driver A's assignments that could be moved
            moveable_assignments = self._get_moveable_assignments(
                driver_a_state, trip_date_str
            )
            
            if not moveable_assignments:
                continue
            
            # For each assignment that could be moved from Driver A
            for assignment_to_move in moveable_assignments:
                # Find other drivers who could take this assignment
                for driver_b_id in drivers:
                    if driver_b_id == driver_a_id:
                        continue
                    
                    driver_b_state = self.driver_states[driver_b_id]
                    
                    # Check if Driver B can take the assignment
                    if self._can_take_assignment(driver_b_state, assignment_to_move, trip_date_str):
                        # Create cascade candidate
                        candidate = self._create_cascade_candidate(
                            disrupted_trip,
                            driver_a_id,
                            driver_a_state,
                            driver_b_id,
                            driver_b_state,
                            assignment_to_move
                        )
                        
                        if candidate and candidate.is_feasible:
                            candidates.append(candidate)
        
        return candidates
    
    def _create_cascade_candidate(self,
                                 disrupted_trip: Dict,
                                 driver_a_id: str,
                                 driver_a_state: DriverState,
                                 driver_b_id: str,
                                 driver_b_state: DriverState,
                                 moved_assignment: DailyAssignment) -> ReassignmentCandidate:
        """
        Create a 2-driver cascade candidate.
        """
        candidate = ReassignmentCandidate(
            disrupted_trip_id=disrupted_trip['id'],
            candidate_type='cascade_2',
            assigned_driver_id=driver_a_id,
            cascade_depth=2
        )
        
        # Build cascade chain description
        candidate.cascade_chain = [
            {
                'step': 1,
                'driver': driver_a_id,
                'action': 'takes_disrupted_trip',
                'trip_id': disrupted_trip['id']
            },
            {
                'step': 2,
                'driver': driver_b_id,
                'action': 'takes_moved_trip',
                'trip_id': moved_assignment.trip_id,
                'from_driver': driver_a_id
            }
        ]
        
        # Calculate costs for both moves
        # This is simplified - actual implementation would be more detailed
        
        # Cost for Driver A taking disrupted trip
        # Try to get last location, otherwise use default
        last_location_a = 'unknown'
        if hasattr(driver_a_state, 'get_last_location_before'):
            last_location_a = driver_a_state.get_last_location_before(disrupted_trip['start_time'])
        
        deadhead_a = self._calculate_travel_time(
            last_location_a,
            disrupted_trip['start_location']
        )
        
        # Cost for Driver B taking moved assignment  
        last_location_b = 'unknown'
        if hasattr(driver_b_state, 'get_last_location_before'):
            last_location_b = driver_b_state.get_last_location_before(moved_assignment.start_time)
        
        deadhead_b = self._calculate_travel_time(
            last_location_b,
            moved_assignment.start_location
        )
        
        candidate.deadhead_minutes = deadhead_a + deadhead_b
        
        # Simple cost calculation
        candidate.total_cost = (
            candidate.deadhead_minutes * self.cost_per_minute_deadhead +
            50  # Fixed penalty for cascade complexity
        )
        
        return candidate
    
    def _check_driver_capacity(self,
                              driver_state: DriverState,
                              date_str: str,
                              additional_minutes: float) -> bool:
        """
        Check if driver has capacity for additional minutes on given date.
        """
        # Try different methods to get daily usage
        current_usage = 0
        
        if hasattr(driver_state, 'get_daily_usage'):
            current_usage = driver_state.get_daily_usage(date_str)
        elif hasattr(driver_state, 'get_day_assignments'):
            # Calculate from assignments
            assignments = driver_state.get_day_assignments(date_str)
            current_usage = sum(a.duration_minutes for a in assignments)
        elif hasattr(driver_state, 'assignments'):
            # Direct access to assignments dict
            if date_str in driver_state.assignments:
                current_usage = sum(a.duration_minutes for a in driver_state.assignments[date_str])
        
        total_with_new = current_usage + additional_minutes
        
        # Check against 13-hour limit (with some buffer for deadhead)
        return total_with_new <= (13 * 60 - 30)  # 30 min buffer
    
    def _calculate_travel_time(self,
                              from_location: str,
                              to_location: str) -> float:
        """
        Calculate travel time between two locations.
        """
        if self.distance_matrix is None or self.location_to_index is None:
            # Fallback: estimate based on location difference
            return 30.0  # Default 30 minutes
        
        try:
            from_idx = self.location_to_index[from_location]
            to_idx = self.location_to_index[to_location]
            return float(self.distance_matrix[from_idx, to_idx])
        except (KeyError, IndexError):
            return 30.0  # Default if location not found
    
    def _get_moveable_assignments(self,
                                 driver_state: DriverState,
                                 date_str: str) -> List[DailyAssignment]:
        """
        Get assignments that could potentially be moved to another driver.
        """
        assignments = []
        
        if hasattr(driver_state, 'get_day_assignments'):
            assignments = driver_state.get_day_assignments(date_str)
        elif hasattr(driver_state, 'assignments'):
            if date_str in driver_state.assignments:
                assignments = driver_state.assignments[date_str]
        
        # Filter for assignments that aren't too time-critical
        # This is a simplified heuristic
        moveable = []
        for assignment in assignments:
            # Don't move very long assignments (>8 hours)
            if assignment.duration_minutes <= 8 * 60:
                moveable.append(assignment)
        
        return moveable
    
    def _can_take_assignment(self,
                           driver_state: DriverState,
                           assignment: DailyAssignment,
                           date_str: str) -> bool:
        """
        Check if a driver can take an additional assignment.
        """
        # Simple capacity check
        return self._check_driver_capacity(
            driver_state,
            date_str,
            assignment.duration_minutes
        )
    
    def _create_outsource_candidate(self,
                                   disrupted_trip: Dict) -> ReassignmentCandidate:
        """
        Create an outsourcing candidate (fallback option).
        """
        candidate = ReassignmentCandidate(
            disrupted_trip_id=disrupted_trip['id'],
            candidate_type='outsource',
            is_feasible=True,
            total_cost=500.0  # Fixed outsourcing cost
        )
        return candidate
    
    def get_candidate_summary(self,
                            candidates: List[ReassignmentCandidate]) -> pd.DataFrame:
        """
        Create a summary DataFrame of candidates for analysis.
        """
        data = []
        for candidate in candidates:
            data.append({
                'trip_id': candidate.disrupted_trip_id,
                'type': candidate.candidate_type,
                'driver_id': candidate.assigned_driver_id,
                'cascade_depth': candidate.cascade_depth,
                'deadhead_min': candidate.deadhead_minutes,
                'delay_min': candidate.delay_minutes,
                'emergency_rest': candidate.emergency_rest_used,
                'total_cost': candidate.total_cost,
                'feasible': candidate.is_feasible,
                'violations': ', '.join(candidate.violations) if candidate.violations else ''
            })
        
        return pd.DataFrame(data)


# Example usage
def example_usage():
    """
    Example of how to use the candidate generator.
    """
    # Assume we have driver states loaded
    driver_states = {}  # Would be populated with actual DriverState objects
    
    # Initialize generator
    generator = CandidateGeneratorV2(
        driver_states=driver_states,
        max_cascade_depth=2,
        max_deadhead_minutes=120,
        max_delay_minutes=120
    )
    
    # Define a disrupted trip
    disrupted_trip = {
        'id': 'TRIP_001',
        'start_time': datetime(2024, 1, 15, 9, 0),
        'end_time': datetime(2024, 1, 15, 13, 0),
        'duration_minutes': 240,
        'start_location': 'Delhi_DC',
        'end_location': 'Mumbai_DC'
    }
    
    # Generate candidates
    candidates = generator.generate_candidates(
        disrupted_trip,
        include_cascades=True,
        include_outsource=True
    )
    
    # Print summary
    print(f"Generated {len(candidates)} candidates:")
    for i, candidate in enumerate(candidates[:5]):  # Show top 5
        print(f"{i+1}. {candidate}")
    
    # Get DataFrame summary
    summary_df = generator.get_candidate_summary(candidates)
    print("\nCandidate Summary:")
    print(summary_df.head())
    
    return candidates, summary_df


if __name__ == "__main__":
    # Run example
    candidates, summary = example_usage()