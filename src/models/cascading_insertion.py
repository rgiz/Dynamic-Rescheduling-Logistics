from dataclasses import dataclass, field
from typing import List, Optional, Set
from enum import Enum


class InsertionType(Enum):
    """Types of trip insertions possible."""
    DIRECT = "direct"                    # Single driver takes disrupted trip
    CASCADE_2 = "cascade_2"             # 2-driver reassignment chain
    CASCADE_MULTI = "cascade_multi"     # 3+ driver reassignment chain
    DELAY_PROPAGATION = "delay_prop"    # Allow delays to subsequent trips
    EMERGENCY_REST = "emergency_rest"   # Use emergency rest periods


@dataclass 
class InsertionStep:
    """Single step in a cascading insertion chain."""
    driver_id: str
    original_trip_id: Optional[str]  # Trip being displaced (None for disrupted trip)
    new_trip_id: str                 # Trip being assigned to this driver
    date: str                        # Date of the assignment
    position: int                    # Position in driver's daily schedule (0 = first)
    deadhead_minutes: int            # Travel time required to reach this trip
    creates_delay_minutes: int       # Delay caused to subsequent trips in driver's schedule
    requires_emergency_rest: bool    # Whether this step needs emergency rest
    cost_impact: float              # Cost impact of this step
    
    def __repr__(self) -> str:
        """Human-readable representation of this step."""
        displaced = f" (displacing {self.original_trip_id})" if self.original_trip_id else ""
        emergency = " [EMERGENCY REST]" if self.requires_emergency_rest else ""
        return f"Driver {self.driver_id}: {self.new_trip_id}{displaced} on {self.date}{emergency}"


@dataclass
class CascadingInsertion:
    """
    Represents a complete multi-driver reassignment solution for a disrupted trip.
    
    Example scenarios:
    - DIRECT: Driver A takes disrupted trip directly
    - CASCADE_2: Driver A takes disrupted trip → Driver B takes A's displaced trip  
    - CASCADE_MULTI: Driver A → disrupted, Driver B → A's trip, Driver C → B's trip
    - DELAY_PROPAGATION: Insert with delays to subsequent scheduled trips
    - EMERGENCY_REST: Use 9-hour rest instead of 11-hour rest
    """
    disrupted_trip_id: str
    insertion_type: InsertionType
    steps: List[InsertionStep] = field(default_factory=list)
    
    # Impact metrics
    total_deadhead_minutes: int = 0
    total_delay_minutes: int = 0
    total_cost_impact: float = 0.0
    emergency_rests_required: int = 0
    drivers_affected: Set[str] = field(default_factory=set)
    
    # Feasibility
    is_feasible: bool = True
    infeasibility_reason: str = ""
    
    def add_step(self, step: InsertionStep) -> None:
        """Add a step to the cascading insertion."""
        self.steps.append(step)
        self.drivers_affected.add(step.driver_id)
        self.total_deadhead_minutes += step.deadhead_minutes
        self.total_delay_minutes += step.creates_delay_minutes
        self.total_cost_impact += step.cost_impact
        if step.requires_emergency_rest:
            self.emergency_rests_required += 1
    
    def get_complexity_score(self) -> float:
        """
        Calculate complexity score for this insertion.
        Lower scores are preferred (simpler solutions).
        """
        base_score = len(self.steps)  # Number of drivers involved
        
        # Penalties for different types of complexity
        delay_penalty = self.total_delay_minutes * 0.1
        emergency_penalty = self.emergency_rests_required * 50
        deadhead_penalty = self.total_deadhead_minutes * 0.05
        
        return base_score + delay_penalty + emergency_penalty + deadhead_penalty
    
    def get_cost_estimate(self, weights: dict = None) -> float:
        """
        Calculate estimated cost of this insertion using configurable weights.
        
        Parameters:
        -----------
        weights : dict
            Cost weights for different factors. Default:
            {'deadhead': 1.0, 'delay': 5.0, 'emergency': 50.0, 'reassignment': 10.0}
        """
        if weights is None:
            weights = {
                'deadhead': 1.0,      # per minute of deadhead travel
                'delay': 5.0,         # per minute of schedule delay
                'emergency': 50.0,    # per emergency rest used
                'reassignment': 10.0  # per driver reassignment
            }
        
        deadhead_cost = self.total_deadhead_minutes * weights.get('deadhead', 1.0)
        delay_cost = self.total_delay_minutes * weights.get('delay', 5.0)
        emergency_cost = self.emergency_rests_required * weights.get('emergency', 50.0)
        reassignment_cost = len(self.drivers_affected) * weights.get('reassignment', 10.0)
        
        return deadhead_cost + delay_cost + emergency_cost + reassignment_cost
    
    def get_service_impact_score(self) -> float:
        """
        Calculate service quality impact score.
        Lower scores are better (less service disruption).
        """
        # Service impact mainly driven by delays and emergency rest usage
        delay_impact = min(self.total_delay_minutes / 60.0, 5.0)  # Cap at 5 hours
        emergency_impact = self.emergency_rests_required * 2.0    # Each emergency rest = 2 points
        driver_impact = (len(self.drivers_affected) - 1) * 0.5   # Multiple drivers = coordination complexity
        
        return delay_impact + emergency_impact + driver_impact
    
    def mark_infeasible(self, reason: str) -> None:
        """Mark this insertion as infeasible with a reason."""
        self.is_feasible = False
        self.infeasibility_reason = reason
    
    def get_chain_description(self) -> str:
        """Get human-readable description of the reassignment chain."""
        if not self.steps:
            return f"No solution for {self.disrupted_trip_id}"
        
        if self.insertion_type == InsertionType.DIRECT:
            step = self.steps[0]
            return f"Direct: {step.driver_id} takes {self.disrupted_trip_id}"
        
        descriptions = []
        for i, step in enumerate(self.steps):
            if i == 0:
                descriptions.append(f"{step.driver_id} takes {self.disrupted_trip_id}")
            else:
                prev_step = self.steps[i-1]
                displaced_trip = prev_step.original_trip_id
                descriptions.append(f"{step.driver_id} takes {displaced_trip}")
        
        return " → ".join(descriptions)
    
    def validate_feasibility(self, driver_states: dict) -> bool:
        """
        Validate that this cascading insertion is actually feasible
        given current driver states.
        
        Parameters:
        -----------
        driver_states : dict
            Dictionary of driver_id -> DriverState objects
        
        Returns:
        --------
        bool : True if feasible, False otherwise
        """
        try:
            for step in self.steps:
                driver_state = driver_states.get(step.driver_id)
                if not driver_state:
                    self.mark_infeasible(f"Driver {step.driver_id} not found")
                    return False
                
                # Check daily capacity
                if not driver_state.can_add_trip(step.date, 
                                                step.new_trip_id, 
                                                step.deadhead_minutes):
                    self.mark_infeasible(f"Driver {step.driver_id} exceeds daily capacity on {step.date}")
                    return False
                
                # Check emergency rest availability
                if step.requires_emergency_rest and not driver_state.can_use_emergency_rest():
                    self.mark_infeasible(f"Driver {step.driver_id} has no emergency rest quota left")
                    return False
            
            return True
            
        except Exception as e:
            self.mark_infeasible(f"Validation error: {str(e)}")
            return False
    
    def get_summary(self) -> dict:
        """Get summary statistics for this cascading insertion."""
        return {
            'disrupted_trip_id': self.disrupted_trip_id,
            'insertion_type': self.insertion_type.value,
            'num_steps': len(self.steps),
            'drivers_affected': len(self.drivers_affected),
            'total_deadhead_hours': self.total_deadhead_minutes / 60.0,
            'total_delay_hours': self.total_delay_minutes / 60.0,
            'emergency_rests_required': self.emergency_rests_required,
            'complexity_score': self.get_complexity_score(),
            'estimated_cost': self.get_cost_estimate(),
            'service_impact': self.get_service_impact_score(),
            'is_feasible': self.is_feasible,
            'chain_description': self.get_chain_description()
        }
    
    def __repr__(self) -> str:
        """String representation of this cascading insertion."""
        status = "✓" if self.is_feasible else "✗"
        return f"CascadingInsertion[{status}] {self.insertion_type.value}: {self.get_chain_description()}"
    
    @classmethod
    def create_direct_insertion(cls, disrupted_trip_id: str, driver_id: str, 
                              date: str, position: int, deadhead_minutes: int = 0,
                              creates_delay_minutes: int = 0, cost_impact: float = 0.0) -> 'CascadingInsertion':
        """
        Factory method to create a simple direct insertion (single driver).
        
        Parameters:
        -----------
        disrupted_trip_id : str
            ID of the trip being inserted
        driver_id : str
            ID of driver taking the trip
        date : str
            Date of insertion (YYYY-MM-DD format)
        position : int
            Position in driver's daily schedule
        deadhead_minutes : int
            Travel time required
        creates_delay_minutes : int
            Delay to subsequent trips
        cost_impact : float
            Cost impact of this insertion
        """
        insertion = cls(
            disrupted_trip_id=disrupted_trip_id,
            insertion_type=InsertionType.DIRECT
        )
        
        step = InsertionStep(
            driver_id=driver_id,
            original_trip_id=None,  # No displacement for direct insertion
            new_trip_id=disrupted_trip_id,
            date=date,
            position=position,
            deadhead_minutes=deadhead_minutes,
            creates_delay_minutes=creates_delay_minutes,
            requires_emergency_rest=False,
            cost_impact=cost_impact
        )
        
        insertion.add_step(step)
        return insertion
    
    @classmethod
    def create_cascade_insertion(cls, disrupted_trip_id: str, steps_data: List[dict]) -> 'CascadingInsertion':
        """
        Factory method to create a cascading insertion from step data.
        
        Parameters:
        -----------
        disrupted_trip_id : str
            ID of the trip being inserted
        steps_data : List[dict]
            List of dictionaries containing step information
        """
        num_steps = len(steps_data)
        if num_steps == 1:
            insertion_type = InsertionType.DIRECT
        elif num_steps == 2:
            insertion_type = InsertionType.CASCADE_2
        else:
            insertion_type = InsertionType.CASCADE_MULTI
        
        insertion = cls(
            disrupted_trip_id=disrupted_trip_id,
            insertion_type=insertion_type
        )
        
        for step_data in steps_data:
            step = InsertionStep(**step_data)
            insertion.add_step(step)
        
        return insertion