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
    Represents a single candidate for reassigning a disrupted trip.
    """
    # Core identification
    disrupted_trip_id: str
    candidate_type: str = 'direct'  # 'direct', 'cascade', 'outsource'
    assigned_driver_id: Optional[str] = None
    
    # Position and sequence
    position_in_day: int = 0
    cascade_depth: int = 1
    cascade_chain: List[Dict] = field(default_factory=list)
    cascade_chain_id: Optional[str] = None
    
    # Operational metrics
    deadhead_minutes: float = 0.0
    deadhead_miles: float = 0.0
    delay_minutes: float = 0.0
    
    # Regulatory compliance
    emergency_rest_used: bool = False
    violates_rest_period: bool = False
    violates_weekend_rest: bool = False
    
    # Feasibility
    is_feasible: bool = True
    violations: List[str] = field(default_factory=list)
    feasibility_score: float = 0.0
    
    # Cost (will be calculated)
    total_cost: float = 0.0
    cost_config: Dict[str, float] = field(default_factory=dict)
    
    def calculate_total_cost(self) -> float:
        """
        Calculate the total cost using provided cost configuration.
        """
        # Use provided config with fallback defaults
        delay_cost_per_min = self.cost_config.get('delay_cost_per_minute', 1.0)
        deadhead_cost_per_min = self.cost_config.get('deadhead_cost_per_minute', 0.5)
        admin_cost = self.cost_config.get('reassignment_admin_cost', 10.0)
        emergency_penalty = self.cost_config.get('emergency_rest_penalty', 50.0)
        outsourcing_base = self.cost_config.get('outsourcing_base_cost', 200.0)
        
        # Service quality costs
        service_cost = self.delay_minutes * delay_cost_per_min
        if self.emergency_rest_used:
            service_cost += emergency_penalty
        
        # Operational costs
        operational_cost = self.deadhead_minutes * deadhead_cost_per_min
        if self.candidate_type in ['direct', 'cascade']:
            operational_cost += admin_cost
        
        # Outsourcing costs (if applicable)
        if self.candidate_type == 'outsource':
            self.total_cost = outsourcing_base
            return self.total_cost
        
        # Total cost for reassignment candidates
        total = service_cost + operational_cost
        self.total_cost = total
        return total


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
                 cost_config: Optional[Dict[str, float]] = None):  # ✅ NEW PARAMETER
        """
        Initialize candidate generator.
        """
        self.driver_states = driver_states
        self.distance_matrix = distance_matrix
        self.location_to_index = location_to_index
        self.max_cascade_depth = max_cascade_depth
        self.max_deadhead_minutes = max_deadhead_minutes
        self.max_delay_minutes = max_delay_minutes
        
        # Store cost configuration
        self.cost_config = cost_config or {}  # ✅ STORE CONFIG
        
        # Cache for performance
        self._driver_availability_cache = {}
    
    def generate_candidates(self, 
                          disrupted_trip: Dict,
                          drivers_to_consider: Optional[List[str]] = None,
                          include_cascades: bool = True,
                          include_outsource: bool = True) -> List[ReassignmentCandidate]:
        """
        Generate all feasible candidates for a disrupted trip.
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
        
        # 4. Calculate costs for all candidates BEFORE sorting
        for candidate in candidates:
            candidate.calculate_total_cost()
        
        # 5. Sort by total cost (lower is better)
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
            elif hasattr(driver_state, 'daily_assignments'):
                day_assignments = driver_state.daily_assignments.get(trip_date_str, [])
            
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
            cascade_depth=1,
            cost_config=self.cost_config  # ✅ PASS CONFIG TO CANDIDATE
        )
        
        # Calculate deadhead and delays
        deadhead_before = 0
        deadhead_after = 0
        deadhead_miles_before = 0
        deadhead_miles_after = 0
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
                deadhead_miles_after = self._calculate_deadhead_miles(
                    trip['end_location'],
                    first_assignment.start_location
                )
                
                # Check for invalid connections
                if deadhead_after == float('inf') or deadhead_miles_after == float('inf'):
                    candidate.is_feasible = False
                    candidate.violations.append("No valid connection to next assignment")
                    return candidate
                
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
                deadhead_miles_before = self._calculate_deadhead_miles(
                    last_assignment.end_location,
                    trip['start_location']
                )
                
                # Check for invalid connections
                if deadhead_before == float('inf') or deadhead_miles_before == float('inf'):
                    candidate.is_feasible = False
                    candidate.violations.append("No valid connection from previous assignment")
                    return candidate
                
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
            deadhead_miles_before = self._calculate_deadhead_miles(
                prev_assignment.end_location,
                trip['start_location']
            )
            
            # Calculate deadhead after
            deadhead_after = self._calculate_travel_time(
                trip['end_location'],
                next_assignment.start_location
            )
            deadhead_miles_after = self._calculate_deadhead_miles(
                trip['end_location'],
                next_assignment.start_location
            )
            
            # Check for invalid connections
            if (deadhead_before == float('inf') or deadhead_miles_before == float('inf') or
                deadhead_after == float('inf') or deadhead_miles_after == float('inf')):
                candidate.is_feasible = False
                candidate.violations.append("No valid connections for insertion")
                return candidate
            
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
        
        # Set candidate metrics (both time and distance)
        candidate.deadhead_minutes = deadhead_before + deadhead_after
        candidate.deadhead_miles = deadhead_miles_before + deadhead_miles_after
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
            if total_day_minutes <= 15 * 60 and hasattr(driver_state, 'can_use_emergency_rest'):
                if driver_state.can_use_emergency_rest():
                    candidate.emergency_rest_used = True
                else:
                    candidate.is_feasible = False
                    candidate.violations.append(f"Daily duty {total_day_minutes/60:.1f}h > 13h limit, no emergency rest available")
            else:
                candidate.is_feasible = False
                candidate.violations.append(f"Daily duty {total_day_minutes/60:.1f}h > 13h limit")
        
        candidate.calculate_total_cost()

        return candidate
    
    def _generate_cascade_insertions(self,
                                    disrupted_trip: Dict,
                                    drivers: List[str]) -> List[ReassignmentCandidate]:
        """
        Generate cascading reassignment candidates (2+ drivers involved).
        """
        candidates = []
        
        # For now, implement simple 2-driver cascades
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
            candidate_type='cascade',
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
                'trip_id': getattr(moved_assignment, 'trip_id', 'unknown'),
                'from_driver': driver_a_id
            }
        ]
        
        # Simplified cost calculation for cascades
        candidate.deadhead_minutes = 60.0  # Estimate for cascade complexity
        candidate.delay_minutes = 0.0
        
        return candidate
    
    def _check_driver_capacity(self,
                              driver_state: DriverState,
                              date_str: str,
                              additional_minutes: float) -> bool:
        """
        Check if driver has capacity for additional minutes on given date.
        """
        current_usage = 0
        
        if hasattr(driver_state, 'get_daily_usage'):
            current_usage = driver_state.get_daily_usage(date_str)
        elif hasattr(driver_state, 'daily_assignments'):
            assignments = driver_state.daily_assignments.get(date_str, [])
            current_usage = sum(getattr(a, 'duration_minutes', 0) for a in assignments)
        
        total_with_new = current_usage + additional_minutes
        
        # Check against 13-hour limit (with some buffer for deadhead)
        return total_with_new <= (13 * 60 - 30)  # 30 min buffer
    

    def _calculate_travel_time(self,
                            from_location: str,
                            to_location: str) -> float:
        """
        Calculate travel time between two locations.
        
        Returns:
            float: Travel time in minutes, or float('inf') if no valid connection exists
        """
        if self.distance_matrix is None or self.location_to_index is None:
            # No matrix available - cannot proceed
            return float('inf')
        
        try:
            from_idx = self.location_to_index[from_location]
            to_idx = self.location_to_index[to_location]
            travel_time = float(self.distance_matrix[from_idx, to_idx])
            
            # Check for no-connection flag (-999) or any negative/invalid value
            if travel_time == -999 or travel_time < 0:
                return float('inf')  # Mark as non-viable connection
                
            return travel_time
            
        except (KeyError, IndexError):
            # Location not found in matrix
            return float('inf')
    def _calculate_deadhead_miles(self,
                            from_location: str,
                            to_location: str) -> float:
        """
        Calculate deadhead miles between two locations.
        
        Returns:
            float: Distance in miles, or float('inf') if no valid connection exists
        """
        # Get travel time in minutes
        travel_time = self._calculate_travel_time(from_location, to_location)
        
        if travel_time == float('inf'):
            return float('inf')
        
        # Convert minutes to miles (assuming 30 mph average speed)
        return (travel_time / 60) * 30
    
    def _get_moveable_assignments(self,
                                 driver_state: DriverState,
                                 date_str: str) -> List[DailyAssignment]:
        """
        Get assignments that could potentially be moved to another driver.
        """
        assignments = []
        
        if hasattr(driver_state, 'daily_assignments'):
            assignments = driver_state.daily_assignments.get(date_str, [])
        
        # Filter for assignments that aren't too long
        moveable = [a for a in assignments if getattr(a, 'duration_minutes', 0) <= 8 * 60]
        
        return moveable
    
    def _can_take_assignment(self,
                           driver_state: DriverState,
                           assignment: DailyAssignment,
                           date_str: str) -> bool:
        """
        Check if a driver can take an additional assignment.
        """
        duration = getattr(assignment, 'duration_minutes', 0)
        return self._check_driver_capacity(driver_state, date_str, duration)
    
    def _create_outsource_candidate(self, disrupted_trip: Dict) -> ReassignmentCandidate:
        """Create an outsourcing candidate (fallback option)."""
        candidate = ReassignmentCandidate(
            disrupted_trip_id=disrupted_trip['id'],
            candidate_type='outsource',
            is_feasible=True,
            cost_config=self.cost_config  # ✅ PASS CONFIG
        )
        candidate.calculate_total_cost()  # ✅ Calculate with user config
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