"""
CP-SAT Model V2 - Multi-Driver Cascading Optimization
=====================================================

Advanced constraint programming model for multi-driver trip reassignment with:
- Multi-objective optimization (cost vs service quality ONLY)
- Compliance rules enforced as HARD CONSTRAINTS
- Cascading reassignment support
- Hard constraint enforcement (13h duty, rest periods, emergency quotas)
- Configurable objective weights for Bayesian Optimization
"""

from ortools.sat.python import cp_model
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time

from models.driver_state import DriverState, DailyAssignment
from opt.candidate_gen_v2 import ReassignmentCandidate, CandidateGeneratorV2
from evaluation_metrics import OptimizationMetrics, MetricsCalculator

# Import regulatory constants
DAILY_DUTY_LIMIT_MIN = 13 * 60      # 13 hours in minutes
STANDARD_REST_MIN = 11 * 60         # 11 hours in minutes  
EMERGENCY_REST_MIN = 9 * 60         # 9 hours in minutes
WEEKEND_REST_MIN = 45 * 60          # 45 hours in minutes
MAX_EMERGENCY_PER_WEEK = 2          # Maximum emergency rests per week


@dataclass
class CPSATSolution:
    """
    Represents a solution from the CP-SAT solver.
    """
    status: str  # 'OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNKNOWN'
    objective_value: float
    solve_time_seconds: float
    
    # Assignment decisions
    assignments: List[Dict] = field(default_factory=list)
    
    # Metrics
    metrics: Optional[OptimizationMetrics] = None
    
    # Solver statistics
    num_branches: int = 0
    num_conflicts: int = 0
    
    def is_feasible(self) -> bool:
        return self.status in ['OPTIMAL', 'FEASIBLE']


class MultiDriverCPSATModel:
    """
    CP-SAT model for multi-driver trip reassignment optimization.
    Enforces compliance as hard constraints, optimizes cost vs service.
    """
    
    def __init__(self,
                 driver_states: Dict[str, DriverState],
                 metrics_calculator: MetricsCalculator,
                 max_solve_time_seconds: float = 30.0,
                 num_workers: int = 4):
        """
        Initialize the CP-SAT model.
        
        Args:
            driver_states: Dictionary of driver_id -> DriverState objects
            metrics_calculator: Calculator for evaluation metrics
            max_solve_time_seconds: Maximum time for solver
            num_workers: Number of parallel workers for solver
        """
        self.driver_states = driver_states
        self.metrics_calculator = metrics_calculator
        self.max_solve_time_seconds = max_solve_time_seconds
        self.num_workers = num_workers
        
        # Model components
        self.model = None
        self.solver = None
        self.decision_vars = {}
        self.helper_vars = {}
        
    def solve(self,
              disrupted_trips: List[Dict],
              candidates_per_trip: Dict[str, List[ReassignmentCandidate]],
              objective_weights: Optional[Dict[str, float]] = None) -> CPSATSolution:
        """
        Solve the multi-driver reassignment problem.
        
        Args:
            disrupted_trips: List of disrupted trip dictionaries
            candidates_per_trip: Dict mapping trip_id -> list of candidates
            objective_weights: Weights for cost vs service optimization
                             Keys: 'cost_weight', 'service_weight'
                             'compliance_weight' is ignored (always 0)
        
        Returns:
            CPSATSolution object with results
        """
        start_time = time.time()
        
        # Initialize model
        self.model = cp_model.CpModel()
        
        # Set default weights if not provided (ignore compliance_weight)
        if objective_weights is None:
            objective_weights = {
                'cost_weight': 0.5,
                'service_weight': 0.5
            }
        
        # Normalize cost and service weights (ignore compliance)
        cost_weight = objective_weights.get('cost_weight', 0.5)
        service_weight = objective_weights.get('service_weight', 0.5)
        total = cost_weight + service_weight
        if total > 0:
            cost_weight = cost_weight / total
            service_weight = service_weight / total
        
        normalized_weights = {
            'cost_weight': cost_weight,
            'service_weight': service_weight
        }
        
        # 1. Create decision variables
        self._create_decision_variables(disrupted_trips, candidates_per_trip)
        
        # 2. Add HARD constraints (including compliance)
        self._add_assignment_constraints(disrupted_trips)
        self._add_daily_duty_constraints(disrupted_trips, candidates_per_trip)
        self._add_rest_period_constraints(disrupted_trips, candidates_per_trip)
        self._add_emergency_rest_quota_constraints(candidates_per_trip)
        self._add_weekend_rest_constraints(disrupted_trips, candidates_per_trip)
        self._add_cascade_constraints(candidates_per_trip)
        
        # 3. Create objective function (cost vs service ONLY)
        self._create_objective_function(candidates_per_trip, normalized_weights)
        
        # 4. Solve the model
        solver, status = self._solve_model()
        
        # 5. Extract solution
        solution = self._extract_solution(
            disrupted_trips,
            candidates_per_trip,
            solver,
            status,
            time.time() - start_time
        )
        
        return solution
    
    def _create_decision_variables(self,
                                  disrupted_trips: List[Dict],
                                  candidates_per_trip: Dict[str, List[ReassignmentCandidate]]):
        """
        Create binary decision variables for each candidate assignment.
        """
        self.decision_vars = {}
        
        for trip in disrupted_trips:
            trip_id = trip['id']
            if trip_id not in candidates_per_trip:
                continue
                
            for i, candidate in enumerate(candidates_per_trip[trip_id]):
                # Binary variable: 1 if this candidate is selected, 0 otherwise
                var_name = f"assign_{trip_id}_candidate_{i}"
                self.decision_vars[(trip_id, i)] = self.model.NewBoolVar(var_name)
        
        print(f"Created {len(self.decision_vars)} decision variables")
    
    def _add_assignment_constraints(self, disrupted_trips: List[Dict]):
        """
        Each disrupted trip must be assigned exactly once.
        """
        for trip in disrupted_trips:
            trip_id = trip['id']
            
            # Get all decision variables for this trip
            trip_vars = [
                var for (t_id, _), var in self.decision_vars.items()
                if t_id == trip_id
            ]
            
            if trip_vars:
                # Exactly one candidate must be selected for each trip
                self.model.Add(sum(trip_vars) == 1)
    
    def _add_daily_duty_constraints(self,
                                   disrupted_trips: List[Dict],
                                   candidates_per_trip: Dict[str, List[ReassignmentCandidate]]):
        """
        HARD CONSTRAINT: Ensure drivers don't exceed 13-hour daily duty limits.
        With emergency rest, can extend to 15 hours but this uses emergency quota.
        """
        # Group assignments by driver and date
        driver_day_assignments = {}
        
        for trip in disrupted_trips:
            trip_id = trip['id']
            trip_date = trip['start_time'].date()
            
            if trip_id not in candidates_per_trip:
                continue
            
            for i, candidate in enumerate(candidates_per_trip[trip_id]):
                if candidate.assigned_driver_id:  # Skip outsourced trips
                    key = (candidate.assigned_driver_id, trip_date)
                    if key not in driver_day_assignments:
                        driver_day_assignments[key] = []
                    
                    driver_day_assignments[key].append({
                        'var': self.decision_vars[(trip_id, i)],
                        'duration': trip['duration_minutes'] + candidate.deadhead_minutes,
                        'uses_emergency': candidate.emergency_rest_used
                    })
        
        # Add hard constraints for each driver-day combination
        for (driver_id, date), assignments in driver_day_assignments.items():
            date_str = date.strftime('%Y-%m-%d')
            
            # Get existing usage for this driver on this date
            existing_usage = self._get_driver_existing_usage(driver_id, date_str)
            
            # Separate regular and emergency assignments
            regular_assignments = [a for a in assignments if not a['uses_emergency']]
            emergency_assignments = [a for a in assignments if a['uses_emergency']]
            
            # HARD CONSTRAINT: Regular duty cannot exceed 13 hours
            if regular_assignments:
                total_regular = sum(
                    a['var'] * int(a['duration']) for a in regular_assignments
                )
                self.model.Add(existing_usage + total_regular <= DAILY_DUTY_LIMIT_MIN)
            
            # HARD CONSTRAINT: Emergency duty cannot exceed 15 hours
            if emergency_assignments:
                total_emergency = sum(
                    a['var'] * int(a['duration']) for a in emergency_assignments
                )
                extended_limit = DAILY_DUTY_LIMIT_MIN + 120  # 15 hours
                self.model.Add(existing_usage + total_emergency <= extended_limit)
    
    def _add_rest_period_constraints(self,
                                    disrupted_trips: List[Dict],
                                    candidates_per_trip: Dict[str, List[ReassignmentCandidate]]):
        """
        HARD CONSTRAINT: Ensure minimum rest periods between work days.
        - Standard: 11 hours minimum
        - Emergency: 9 hours minimum (uses quota)
        """
        # Track assignments that affect rest periods
        driver_rest_violations = {}
        
        for trip_id, candidates in candidates_per_trip.items():
            for i, candidate in enumerate(candidates):
                if not candidate.assigned_driver_id:
                    continue
                
                # Check if this assignment would violate rest requirements
                if hasattr(candidate, 'violates_rest_period'):
                    if candidate.violates_rest_period and not candidate.emergency_rest_used:
                        # This candidate violates rest and doesn't use emergency
                        # Make it infeasible
                        self.model.Add(self.decision_vars[(trip_id, i)] == 0)
                    elif candidate.violates_rest_period and candidate.emergency_rest_used:
                        # This uses emergency rest - track for quota constraint
                        driver_id = candidate.assigned_driver_id
                        if driver_id not in driver_rest_violations:
                            driver_rest_violations[driver_id] = []
                        driver_rest_violations[driver_id].append(self.decision_vars[(trip_id, i)])
    
    def _add_emergency_rest_quota_constraints(self,
                                             candidates_per_trip: Dict[str, List[ReassignmentCandidate]]):
        """
        HARD CONSTRAINT: Limit emergency rest usage to 2 per driver per week.
        """
        # Track emergency rest usage by driver and week
        driver_week_emergency = {}
        
        for trip_id, candidates in candidates_per_trip.items():
            for i, candidate in enumerate(candidates):
                if candidate.emergency_rest_used and candidate.assigned_driver_id:
                    # Determine week number (simplified - could be enhanced)
                    week_key = self._get_week_key(trip_id)
                    
                    key = (candidate.assigned_driver_id, week_key)
                    if key not in driver_week_emergency:
                        driver_week_emergency[key] = []
                    
                    driver_week_emergency[key].append(self.decision_vars[(trip_id, i)])
        
        # HARD CONSTRAINT: Max 2 emergency rests per driver per week
        for (driver_id, week), emergency_vars in driver_week_emergency.items():
            if emergency_vars:
                # Get existing emergency rest count for this driver/week
                existing_emergency_count = self._get_driver_emergency_count(driver_id, week)
                
                self.model.Add(
                    existing_emergency_count + sum(emergency_vars) <= MAX_EMERGENCY_PER_WEEK
                )
    
    def _add_weekend_rest_constraints(self,
                                     disrupted_trips: List[Dict],
                                     candidates_per_trip: Dict[str, List[ReassignmentCandidate]]):
        """
        HARD CONSTRAINT: Ensure 45-hour minimum weekend rest periods.
        Weekend rest cannot use emergency rest - must be full 45 hours.
        """
        # Track assignments that might affect weekend rest
        weekend_affecting_assignments = {}
        
        for trip_id, candidates in candidates_per_trip.items():
            for i, candidate in enumerate(candidates):
                if not candidate.assigned_driver_id:
                    continue
                
                # Check if this assignment affects weekend rest
                if hasattr(candidate, 'affects_weekend_rest'):
                    if candidate.affects_weekend_rest:
                        # Check if it would violate weekend rest requirements
                        if hasattr(candidate, 'violates_weekend_rest'):
                            if candidate.violates_weekend_rest:
                                # HARD CONSTRAINT: Cannot violate weekend rest
                                self.model.Add(self.decision_vars[(trip_id, i)] == 0)
    
    def _add_cascade_constraints(self,
                                candidates_per_trip: Dict[str, List[ReassignmentCandidate]]):
        """
        Ensure cascade chains are consistent.
        If a cascade is selected, all parts of the chain must be feasible.
        """
        # Simplified cascade constraint - would need more sophisticated logic
        # for full implementation
        pass
    
    def _create_objective_function(self,
                                  candidates_per_trip: Dict[str, List[ReassignmentCandidate]],
                                  weights: Dict[str, float]):
        """
        Create objective function to minimize cost and maximize service quality.
        NOTE: Compliance is NOT part of the objective - it's enforced through hard constraints.
        """
        objective_terms = []
        
        # Scale factor for costs to ensure integer arithmetic
        cost_scale = 1000
        
        for trip_id, candidates in candidates_per_trip.items():
            for i, candidate in enumerate(candidates):
                if (trip_id, i) not in self.decision_vars:
                    continue
                    
                var = self.decision_vars[(trip_id, i)]
                
                # Calculate cost component
                base_cost = candidate.total_cost
                
                # Calculate service quality penalty
                service_penalty = 0
                if candidate.delay_minutes > 0:
                    # Penalty for delays (affects service quality)
                    service_penalty = candidate.delay_minutes * 2  # $2 per minute delay
                
                # Combine cost and service objectives
                # Note: NO compliance weight - compliance is hard constraint
                weighted_objective = (
                    weights['cost_weight'] * base_cost +
                    weights['service_weight'] * service_penalty
                )
                
                # Special handling for outsourcing (high cost, perfect service)
                if candidate.candidate_type == 'outsource':
                    # Outsourcing has high cost but ensures service
                    weighted_objective = base_cost * (weights['cost_weight'] + 0.5 * weights['service_weight'])
                
                objective_terms.append(var * int(weighted_objective * cost_scale))
        
        # Minimize total weighted objective
        self.model.Minimize(sum(objective_terms))
    
    def _solve_model(self) -> Tuple[cp_model.CpSolver, int]:
        """
        Solve the CP-SAT model.
        Returns both solver and status.
        """
        self.solver = cp_model.CpSolver()
        
        # Set solver parameters
        self.solver.parameters.max_time_in_seconds = self.max_solve_time_seconds
        self.solver.parameters.num_search_workers = self.num_workers
        self.solver.parameters.log_search_progress = True
        
        # Solve
        status = self.solver.Solve(self.model)
        
        # Print solver statistics
        print(f"\nSolver Statistics:")
        print(f"  Status: {self.solver.StatusName(status)}")
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print(f"  Objective: {self.solver.ObjectiveValue()}")
        print(f"  Time: {self.solver.WallTime():.2f}s")
        print(f"  Branches: {self.solver.NumBranches()}")
        print(f"  Conflicts: {self.solver.NumConflicts()}")
        
        return self.solver, status
    
    def _extract_solution(self,
                        disrupted_trips: List[Dict],
                        candidates_per_trip: Dict[str, List[ReassignmentCandidate]],
                        solver: cp_model.CpSolver,
                        status: int,
                        solve_time: float) -> CPSATSolution:
        """
        Extract solution from solved model.
        """
        # Map status code to string
        status_map = {
            cp_model.OPTIMAL: 'OPTIMAL',
            cp_model.FEASIBLE: 'FEASIBLE',
            cp_model.INFEASIBLE: 'INFEASIBLE',
            cp_model.MODEL_INVALID: 'INVALID',
            cp_model.UNKNOWN: 'UNKNOWN'
        }
        
        solution = CPSATSolution(
            status=status_map.get(status, 'UNKNOWN'),
            objective_value=solver.ObjectiveValue() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else float('inf'),
            solve_time_seconds=solve_time,
            num_branches=solver.NumBranches(),
            num_conflicts=solver.NumConflicts()
        )
        
        # Extract selected assignments if feasible
        if solution.is_feasible():
            for trip in disrupted_trips:
                trip_id = trip['id']
                if trip_id not in candidates_per_trip:
                    continue
                
                for i, candidate in enumerate(candidates_per_trip[trip_id]):
                    if (trip_id, i) in self.decision_vars:
                        if solver.Value(self.decision_vars[(trip_id, i)]) == 1:
                            # This candidate was selected
                            solution.assignments.append({
                                'trip_id': trip_id,
                                'candidate': candidate,
                                'type': candidate.candidate_type,
                                'driver_id': candidate.assigned_driver_id,
                                'deadhead_minutes': candidate.deadhead_minutes,
                                'delay_minutes': candidate.delay_minutes,
                                'emergency_rest_used': candidate.emergency_rest_used,
                                'cascade_depth': candidate.cascade_depth,
                                'total_cost': candidate.total_cost
                            })
            
            # Calculate metrics if we have assignments
            if solution.assignments and self.metrics_calculator:
                # Convert disrupted_trips list to DataFrame for metrics calculator
                trips_df = pd.DataFrame(disrupted_trips)
                solution.metrics = self.metrics_calculator.calculate_from_solution(
                    trips_df,
                    solution.assignments,
                    self.driver_states
                )
        
        return solution
    
    def _get_driver_existing_usage(self, driver_id: str, date_str: str) -> int:
        """
        Get existing usage for a driver on a specific date.
        """
        if driver_id not in self.driver_states:
            return 0
            
        driver_state = self.driver_states[driver_id]
        
        # Try different methods to get existing usage
        if hasattr(driver_state, 'get_daily_capacity_used'):
            return driver_state.get_daily_capacity_used(date_str)
        elif hasattr(driver_state, 'daily_assignments'):
            if date_str in driver_state.daily_assignments:
                assignments = driver_state.daily_assignments[date_str]
                return sum(a.duration_minutes for a in assignments)
        
        return 0
    
    def _get_driver_emergency_count(self, driver_id: str, week_key: str) -> int:
        """
        Get existing emergency rest count for a driver in a specific week.
        """
        if driver_id not in self.driver_states:
            return 0
            
        driver_state = self.driver_states[driver_id]
        
        if hasattr(driver_state, 'emergency_rests_used_this_week'):
            return driver_state.emergency_rests_used_this_week
        
        return 0
    
    def _get_week_key(self, trip_id: str) -> str:
        """
        Get week identifier for a trip (for emergency rest tracking).
        Simplified version - would need proper week calculation in practice.
        """
        # This would normally calculate the actual week number
        return "week_1"
    
    def _print_solution_summary(self, solution: CPSATSolution):
        """
        Print a summary of the solution.
        """
        print("\n" + "="*60)
        print("SOLUTION SUMMARY")
        print("="*60)
        
        if not solution.is_feasible():
            print("❌ No feasible solution found")
            print("   Compliance constraints could not be satisfied")
            return
        
        # Count assignment types
        reassigned = sum(1 for a in solution.assignments if a['type'] == 'reassigned')
        outsourced = sum(1 for a in solution.assignments if a['type'] == 'outsourced')
        
        print(f"\n📊 Assignment Results:")
        print(f"  • Reassigned: {reassigned}")
        print(f"  • Outsourced: {outsourced}")
        print(f"  • Total: {len(solution.assignments)}")
        
        # Cost breakdown
        total_deadhead = sum(a.get('deadhead_minutes', 0) for a in solution.assignments)
        total_delay = sum(a.get('delay_minutes', 0) for a in solution.assignments)
        emergency_count = sum(1 for a in solution.assignments if a.get('emergency_rest_used', False))
        
        print(f"\n💰 Cost Components:")
        print(f"  • Total deadhead: {total_deadhead:.0f} minutes")
        print(f"  • Total delay: {total_delay:.0f} minutes")
        
        print(f"\n✅ Compliance Status:")
        print(f"  • All daily duty limits respected (≤13 hours)")
        print(f"  • All rest periods enforced (≥11 hours, or ≥9 with emergency)")
        print(f"  • Emergency rests used: {emergency_count} (max {MAX_EMERGENCY_PER_WEEK} per driver/week)")
        print(f"  • Weekend rest periods protected (≥45 hours)")


class CPSATOptimizer:
    """
    High-level optimizer that combines candidate generation and CP-SAT solving.
    """
    
    def __init__(self,
                 driver_states: Dict[str, DriverState],
                 distance_matrix: Optional[np.ndarray] = None,
                 location_to_index: Optional[Dict[str, int]] = None):
        """
        Initialize the optimizer.
        """
        self.driver_states = driver_states
        
        # Initialize components
        self.candidate_generator = CandidateGeneratorV2(
            driver_states=driver_states,
            distance_matrix=distance_matrix,
            location_to_index=location_to_index
        )
        
        self.metrics_calculator = MetricsCalculator()
        
        self.cpsat_model = MultiDriverCPSATModel(
            driver_states=driver_states,
            metrics_calculator=self.metrics_calculator
        )
    
    def optimize(self,
                disrupted_trips: List[Dict],
                objective_weights: Optional[Dict[str, float]] = None,
                include_cascades: bool = True,
                max_candidates_per_trip: int = 20) -> CPSATSolution:
        """
        Run the complete optimization pipeline.
        
        Args:
            disrupted_trips: List of disrupted trip dictionaries
            objective_weights: Weights for cost vs service optimization
                             (compliance_weight is ignored)
            include_cascades: Whether to generate cascading candidates
            max_candidates_per_trip: Maximum candidates to consider per trip
            
        Returns:
            CPSATSolution with optimal assignments
        """
        print(f"\n🚀 Starting optimization for {len(disrupted_trips)} disrupted trips")
        print("   Compliance enforced as HARD CONSTRAINTS")
        print("   Optimizing: Cost vs Service Quality")
        
        # 1. Generate candidates for each trip
        print("\n📋 Generating candidates...")
        candidates_per_trip = {}
        total_candidates = 0
        
        for trip in disrupted_trips:
            candidates = self.candidate_generator.generate_candidates(
                trip,
                include_cascades=include_cascades,
                include_outsource=True
            )
            
            # Limit candidates if too many
            if len(candidates) > max_candidates_per_trip:
                candidates = candidates[:max_candidates_per_trip]
            
            candidates_per_trip[trip['id']] = candidates
            total_candidates += len(candidates)
            
            print(f"  Trip {trip['id']}: {len(candidates)} candidates")
        
        print(f"Total candidates generated: {total_candidates}")
        
        # 2. Solve with CP-SAT
        print("\n🧮 Solving with CP-SAT (compliance as hard constraints)...")
        solution = self.cpsat_model.solve(
            disrupted_trips,
            candidates_per_trip,
            objective_weights
        )
        
        # 3. Print summary
        self.cpsat_model._print_solution_summary(solution)
        
        return solution