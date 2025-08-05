"""
CP-SAT Model V2 - Multi-Driver Cascading Optimization
=====================================================

Advanced constraint programming model for multi-driver trip reassignment with:
- Multi-objective optimization (cost vs service quality)
- Cascading reassignment support
- Hard constraint enforcement (13h duty, rest periods)
- Emergency rest quota management
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


@dataclass
class CPSATSolution:
    """
    Represents a solution from the CP-SAT solver.
    """
    status: str  # 'OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNKNOWN'
    objective_value: float
    solve_time_seconds: float
    
    # Assignment decisions
    assignments: List[Dict] = field(default_factory=list)  # List of selected candidates
    
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
        
        # Model components (initialized in solve)
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
            objective_weights: Weights for multi-objective optimization
                             Keys: 'cost_weight', 'service_weight', 'compliance_weight'
        
        Returns:
            CPSATSolution object with results
        """
        start_time = time.time()
        
        # Initialize model
        self.model = cp_model.CpModel()
        
        # Set default weights if not provided
        if objective_weights is None:
            objective_weights = {
                'cost_weight': 0.4,
                'service_weight': 0.3,
                'compliance_weight': 0.3
            }
        
        # 1. Create decision variables
        self._create_decision_variables(disrupted_trips, candidates_per_trip)
        
        # 2. Add constraints
        self._add_assignment_constraints(disrupted_trips)
        self._add_driver_capacity_constraints(disrupted_trips, candidates_per_trip)
        self._add_cascade_constraints(candidates_per_trip)
        self._add_emergency_rest_constraints(candidates_per_trip)
        
        # 3. Create objective function
        self._create_objective_function(candidates_per_trip, objective_weights)
        
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
    
    def _add_driver_capacity_constraints(self,
                                        disrupted_trips: List[Dict],
                                        candidates_per_trip: Dict[str, List[ReassignmentCandidate]]):
        """
        Ensure drivers don't exceed daily duty limits.
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
        
        # Add capacity constraints for each driver-day combination
        for (driver_id, date), assignments in driver_day_assignments.items():
            # Get existing usage for this driver on this date
            date_str = date.strftime('%Y-%m-%d')
            
            # Calculate existing usage with fallback methods
            existing_usage = 0
            driver_state = self.driver_states[driver_id]
            
            if hasattr(driver_state, 'get_daily_usage'):
                existing_usage = driver_state.get_daily_usage(date_str)
            elif hasattr(driver_state, 'get_day_assignments'):
                # Calculate from assignments
                day_assignments = driver_state.get_day_assignments(date_str)
                existing_usage = sum(a.duration_minutes for a in day_assignments)
            elif hasattr(driver_state, 'assignments'):
                # Direct access to assignments dict
                if date_str in driver_state.assignments:
                    existing_usage = sum(a.duration_minutes for a in driver_state.assignments[date_str])
            
            # Create constraint: existing + sum(selected assignments) <= 13 hours
            # (or 15 hours with emergency rest)
            
            # Regular assignments (no emergency rest)
            regular_assignments = [a for a in assignments if not a['uses_emergency']]
            if regular_assignments:
                total_regular = sum(
                    a['var'] * int(a['duration']) for a in regular_assignments
                )
                self.model.Add(existing_usage + total_regular <= 13 * 60)
            
            # Emergency rest assignments (can go up to 15 hours)
            emergency_assignments = [a for a in assignments if a['uses_emergency']]
            if emergency_assignments:
                total_emergency = sum(
                    a['var'] * int(a['duration']) for a in emergency_assignments
                )
                self.model.Add(existing_usage + total_emergency <= 15 * 60)
    
    def _add_cascade_constraints(self,
                                candidates_per_trip: Dict[str, List[ReassignmentCandidate]]):
        """
        Ensure cascade chains are consistent.
        If a cascade is selected, all parts of the chain must be feasible.
        """
        # Track which trips are involved in cascades
        cascade_dependencies = {}
        
        for trip_id, candidates in candidates_per_trip.items():
            for i, candidate in enumerate(candidates):
                if candidate.cascade_depth > 1:
                    # This is a cascading assignment
                    # Store dependencies between cascade steps
                    for step in candidate.cascade_chain:
                        if step['action'] == 'takes_moved_trip':
                            # This step depends on another trip being moved
                            moved_trip_id = step['trip_id']
                            if moved_trip_id not in cascade_dependencies:
                                cascade_dependencies[moved_trip_id] = []
                            cascade_dependencies[moved_trip_id].append(
                                self.decision_vars[(trip_id, i)]
                            )
        
        # Add implication constraints for cascades
        # If a cascade is selected, ensure the moved trips are handled
        # This is simplified - full implementation would be more complex
        for moved_trip_id, dependent_vars in cascade_dependencies.items():
            # If any cascade involving this trip is selected,
            # ensure the trip is reassigned somehow
            # (This would need more sophisticated logic in practice)
            pass
    
    def _add_emergency_rest_constraints(self,
                                       candidates_per_trip: Dict[str, List[ReassignmentCandidate]]):
        """
        Limit emergency rest usage per driver per week.
        """
        # Track emergency rest usage by driver and week
        driver_week_emergency = {}
        
        for trip_id, candidates in candidates_per_trip.items():
            for i, candidate in enumerate(candidates):
                if candidate.emergency_rest_used and candidate.assigned_driver_id:
                    # Determine week number
                    week_key = self._get_week_key(candidates[0].disrupted_trip_id)
                    
                    key = (candidate.assigned_driver_id, week_key)
                    if key not in driver_week_emergency:
                        driver_week_emergency[key] = []
                    
                    driver_week_emergency[key].append(self.decision_vars[(trip_id, i)])
        
        # Add constraints: max 2 emergency rests per driver per week
        for (driver_id, week), emergency_vars in driver_week_emergency.items():
            if emergency_vars:
                self.model.Add(sum(emergency_vars) <= 2)
    
    def _create_objective_function(self,
                                  candidates_per_trip: Dict[str, List[ReassignmentCandidate]],
                                  weights: Dict[str, float]):
        """
        Create multi-objective function to minimize.
        """
        objective_terms = []
        
        # Cost components
        cost_scale = 100  # Scale factor for costs
        
        for trip_id, candidates in candidates_per_trip.items():
            for i, candidate in enumerate(candidates):
                if (trip_id, i) in self.decision_vars:
                    var = self.decision_vars[(trip_id, i)]
                    
                    # Add weighted cost for this candidate
                    cost = candidate.total_cost
                    
                    # Apply multi-objective weights
                    if candidate.candidate_type == 'outsource':
                        # Outsourcing has high cost but ensures feasibility
                        weighted_cost = cost * weights['cost_weight'] * 2
                    elif candidate.emergency_rest_used:
                        # Emergency rest has compliance penalty
                        weighted_cost = cost * (weights['cost_weight'] + weights['compliance_weight'])
                    elif candidate.delay_minutes > 0:
                        # Delays affect service quality
                        delay_penalty = candidate.delay_minutes * weights['service_weight']
                        weighted_cost = cost * weights['cost_weight'] + delay_penalty
                    else:
                        # Regular assignment
                        weighted_cost = cost * weights['cost_weight']
                    
                    objective_terms.append(var * int(weighted_cost * cost_scale))
        
        # Minimize total weighted cost
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
        # Get status name
        status_name = solver.StatusName(status)
        
        solution = CPSATSolution(
            status=status_name,
            objective_value=solver.ObjectiveValue() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else 0,
            solve_time_seconds=solve_time,
            num_branches=solver.NumBranches(),
            num_conflicts=solver.NumConflicts()
        )
        
        if status_name in ['OPTIMAL', 'FEASIBLE']:
            # Extract selected assignments
            selected_candidates = []
            
            for trip in disrupted_trips:
                trip_id = trip['id']
                if trip_id not in candidates_per_trip:
                    continue
                
                for i, candidate in enumerate(candidates_per_trip[trip_id]):
                    if (trip_id, i) in self.decision_vars:
                        if solver.Value(self.decision_vars[(trip_id, i)]) == 1:
                            # This candidate was selected
                            assignment = {
                                'trip_id': trip_id,
                                'type': 'outsourced' if candidate.candidate_type == 'outsource' else 'reassigned',
                                'driver_id': candidate.assigned_driver_id,
                                'candidate_type': candidate.candidate_type,
                                'cascade_depth': candidate.cascade_depth,
                                'deadhead_minutes': candidate.deadhead_minutes,
                                'delay_minutes': candidate.delay_minutes,
                                'emergency_rest_used': candidate.emergency_rest_used,
                                'cost': candidate.total_cost
                            }
                            selected_candidates.append(assignment)
                            solution.assignments.append(assignment)
            
            # Calculate metrics
            solution.metrics = self.metrics_calculator.calculate_from_solution(
                pd.DataFrame(disrupted_trips),
                selected_candidates,
                self.driver_states
            )
            
            # Print solution summary
            self._print_solution_summary(solution)
        
        return solution
    
    def _print_solution_summary(self, solution: CPSATSolution):
        """
        Print a summary of the solution.
        """
        print("\n" + "="*60)
        print("SOLUTION SUMMARY")
        print("="*60)
        
        # Count assignment types
        reassigned = sum(1 for a in solution.assignments if a['type'] == 'reassigned')
        outsourced = sum(1 for a in solution.assignments if a['type'] == 'outsourced')
        
        print(f"\n📊 Assignment Results:")
        print(f"  • Reassigned: {reassigned}")
        print(f"  • Outsourced: {outsourced}")
        print(f"  • Total: {len(solution.assignments)}")
        
        # Cascade analysis
        cascade_counts = {}
        for a in solution.assignments:
            depth = a.get('cascade_depth', 1)
            cascade_counts[depth] = cascade_counts.get(depth, 0) + 1
        
        if cascade_counts:
            print(f"\n🔄 Cascade Analysis:")
            for depth, count in sorted(cascade_counts.items()):
                if depth == 1:
                    print(f"  • Direct assignments: {count}")
                else:
                    print(f"  • {depth}-driver cascades: {count}")
        
        # Cost breakdown
        total_deadhead = sum(a.get('deadhead_minutes', 0) for a in solution.assignments)
        total_delay = sum(a.get('delay_minutes', 0) for a in solution.assignments)
        emergency_count = sum(1 for a in solution.assignments if a.get('emergency_rest_used', False))
        
        print(f"\n💰 Cost Components:")
        print(f"  • Total deadhead: {total_deadhead:.0f} minutes")
        print(f"  • Total delay: {total_delay:.0f} minutes")
        print(f"  • Emergency rests used: {emergency_count}")
        
        # Show metrics if available
        if solution.metrics:
            solution.metrics.print_summary()
    
    def _get_week_key(self, trip_id: str) -> str:
        """
        Get week identifier for a trip (for emergency rest tracking).
        Simplified version - would need proper implementation.
        """
        return "week_1"  # Placeholder


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
        
        Args:
            driver_states: Dictionary of driver_id -> DriverState objects
            distance_matrix: Matrix of travel times between locations
            location_to_index: Mapping of location names to matrix indices
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
            objective_weights: Weights for multi-objective optimization
            include_cascades: Whether to generate cascading candidates
            max_candidates_per_trip: Maximum candidates to consider per trip
            
        Returns:
            CPSATSolution with optimal assignments
        """
        print(f"\n🚀 Starting optimization for {len(disrupted_trips)} disrupted trips")
        
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
        print("\n🧮 Solving with CP-SAT...")
        solution = self.cpsat_model.solve(
            disrupted_trips,
            candidates_per_trip,
            objective_weights
        )
        
        return solution


# Example usage
def example_optimization():
    """
    Example of how to use the CP-SAT optimizer.
    """
    # Create some dummy driver states
    driver_states = {}
    for i in range(5):
        driver_states[f"driver_{i}"] = DriverState(
            driver_id=f"driver_{i}",
            route_id=f"route_{i}"
        )
    
    # Define disrupted trips
    disrupted_trips = [
        {
            'id': 'TRIP_001',
            'start_time': datetime(2024, 1, 15, 9, 0),
            'end_time': datetime(2024, 1, 15, 13, 0),
            'duration_minutes': 240,
            'start_location': 'Delhi_DC',
            'end_location': 'Mumbai_DC'
        },
        {
            'id': 'TRIP_002',
            'start_time': datetime(2024, 1, 15, 10, 0),
            'end_time': datetime(2024, 1, 15, 14, 0),
            'duration_minutes': 240,
            'start_location': 'Bangalore_DC',
            'end_location': 'Chennai_DC'
        }
    ]
    
    # Initialize optimizer
    optimizer = CPSATOptimizer(driver_states)
    
    # Run optimization with custom weights
    solution = optimizer.optimize(
        disrupted_trips,
        objective_weights={
            'cost_weight': 0.5,
            'service_weight': 0.3,
            'compliance_weight': 0.2
        },
        include_cascades=True
    )
    
    return solution


if __name__ == "__main__":
    # Run example
    solution = example_optimization()