from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd


@dataclass
class WeekendBreak:
    """Represents a weekend break period."""
    start_time: datetime
    end_time: datetime
    duration_hours: float
    is_compliant: bool
    week_ending: str  # Date string of week ending (e.g., "2025-08-10")
    
    def __post_init__(self):
        """Calculate duration and compliance after initialization."""
        if self.start_time and self.end_time:
            self.duration_hours = (self.end_time - self.start_time).total_seconds() / 3600
            self.is_compliant = self.duration_hours >= WEEKEND_REST_MIN / 60.0


@dataclass 
class WeeklySchedule:
    """
    Manages a driver's schedule across multiple days with weekend break validation.
    
    Handles:
    - Multi-day route scheduling
    - Weekend break compliance (45-hour minimum)
    - Exact start times for new weeks  
    - Emergency rest quota tracking
    - Cross-day delay propagation
    """
    
    driver_id: str
    route_id: str
    week_start_date: datetime  # Monday of this week
    
    # Daily driver states for each day of the week
    daily_states: Dict[str, DriverState] = field(default_factory=dict)  # date_str -> DriverState
    
    # Weekend break tracking
    weekend_breaks: List[WeekendBreak] = field(default_factory=list)
    
    # Week-level constraints
    total_emergency_rests_used: int = 0
    emergency_rest_dates: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize daily states for the week."""
        if not self.daily_states:
            # Create daily states for Monday through Sunday
            for i in range(7):
                date = self.week_start_date + timedelta(days=i)
                date_str = date.strftime('%Y-%m-%d')
                self.daily_states[date_str] = DriverState(
                    driver_id=self.driver_id,
                    route_id=self.route_id
                )
    
    def get_week_bounds(self) -> Tuple[datetime, datetime]:
        """Get start and end of this week."""
        week_end = self.week_start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
        return self.week_start_date, week_end
    
    def get_work_days(self) -> List[str]:
        """Get list of work days (Monday-Friday) for this week."""
        work_days = []
        for i in range(5):  # Monday through Friday
            date = self.week_start_date + timedelta(days=i)
            work_days.append(date.strftime('%Y-%m-%d'))
        return work_days
    
    def get_weekend_days(self) -> List[str]:
        """Get list of weekend days (Saturday-Sunday) for this week.""" 
        weekend_days = []
        for i in range(5, 7):  # Saturday and Sunday
            date = self.week_start_date + timedelta(days=i)
            weekend_days.append(date.strftime('%Y-%m-%d'))
        return weekend_days
    
    def add_assignment(self, date: str, assignment: DailyAssignment) -> bool:
        """
        Add a trip assignment to a specific date.
        
        Parameters:
        -----------
        date : str
            Date in YYYY-MM-DD format
        assignment : DailyAssignment
            Assignment to add
            
        Returns:
        --------
        bool : True if successful, False if violates constraints
        """
        if date not in self.daily_states:
            return False
        
        return self.daily_states[date].add_assignment(date, assignment)
    
    def remove_assignment(self, date: str, trip_id: str) -> Optional[DailyAssignment]:
        """Remove a trip assignment from a specific date."""
        if date not in self.daily_states:
            return None
        
        return self.daily_states[date].remove_assignment(date, trip_id)
    
    def validate_weekend_breaks(self) -> List[str]:
        """
        Validate all weekend breaks in this schedule.
        
        Returns:
        --------
        List[str] : List of validation issues (empty if all compliant)
        """
        issues = []
        
        # Get Friday and Monday work times
        friday_date = (self.week_start_date + timedelta(days=4)).strftime('%Y-%m-%d')
        monday_date = self.week_start_date.strftime('%Y-%m-%d')
        
        friday_state = self.daily_states.get(friday_date)
        monday_state = self.daily_states.get(monday_date)
        
        if friday_state and monday_state:
            friday_start, friday_end = friday_state.get_work_day_bounds(friday_date)
            monday_start, monday_end = monday_state.get_work_day_bounds(monday_date)
            
            if friday_end and monday_start:
                weekend_duration = (monday_start - friday_end).total_seconds() / 3600
                
                if weekend_duration < WEEKEND_REST_MIN / 60.0:
                    issues.append(f"Weekend break too short: {weekend_duration:.1f}h < {WEEKEND_REST_MIN/60:.1f}h")
                
                # Check if Monday starts exactly on time (allowing some tolerance)
                expected_monday_start = friday_end + timedelta(hours=WEEKEND_REST_MIN/60.0)
                time_diff = abs((monday_start - expected_monday_start).total_seconds())
                
                if time_diff > 3600:  # More than 1 hour difference
                    issues.append(f"Monday start time not exact: {time_diff/3600:.1f}h difference from expected")
        
        return issues
    
    def can_use_emergency_rest(self) -> bool:
        """Check if any more emergency rests can be used this week."""
        return self.total_emergency_rests_used < MAX_EMERGENCY_PER_WEEK
    
    def use_emergency_rest(self, date: str) -> bool:
        """
        Use an emergency rest for a specific date.
        
        Parameters:
        -----------
        date : str
            Date to apply emergency rest
            
        Returns:
        --------
        bool : True if successful, False if quota exceeded
        """
        if not self.can_use_emergency_rest():
            return False
        
        if date in self.daily_states:
            success = self.daily_states[date].use_emergency_rest(date)
            if success:
                self.total_emergency_rests_used += 1
                self.emergency_rest_dates.append(date)
                return True
        
        return False
    
    def get_week_utilization(self) -> Dict[str, any]:
        """Get utilization statistics for the entire week."""
        work_days = self.get_work_days()
        
        total_capacity_used = 0
        total_capacity_available = 0
        daily_utilizations = {}
        
        for date in work_days:
            if date in self.daily_states:
                daily_used = self.daily_states[date].get_daily_capacity_used(date)
                daily_available = DAILY_DUTY_LIMIT_MIN
                
                total_capacity_used += daily_used
                total_capacity_available += daily_available
                
                daily_utilizations[date] = {
                    'used_hours': daily_used / 60.0,
                    'available_hours': daily_available / 60.0,
                    'utilization_pct': (daily_used / daily_available * 100) if daily_available > 0 else 0
                }
        
        overall_utilization = (total_capacity_used / total_capacity_available * 100) if total_capacity_available > 0 else 0
        
        return {
            'week_start': self.week_start_date.strftime('%Y-%m-%d'),
            'total_capacity_used_hours': total_capacity_used / 60.0,
            'total_capacity_available_hours': total_capacity_available / 60.0,
            'overall_utilization_pct': overall_utilization,
            'emergency_rests_used': self.total_emergency_rests_used,
            'emergency_rests_remaining': MAX_EMERGENCY_PER_WEEK - self.total_emergency_rests_used,
            'daily_utilizations': daily_utilizations,
            'weekend_compliance': len(self.validate_weekend_breaks()) == 0
        }
    
    def get_available_capacity(self, date: str) -> int:
        """Get available capacity in minutes for a specific date."""
        if date not in self.daily_states:
            return 0
        
        return self.daily_states[date].get_daily_capacity_remaining(date)
    
    def can_accommodate_trip(self, date: str, trip_duration_min: int, 
                           deadhead_min: int = 0) -> bool:
        """
        Check if a trip can be accommodated on a specific date.
        
        Parameters:
        -----------
        date : str
            Target date
        trip_duration_min : int
            Duration of trip in minutes
        deadhead_min : int
            Required deadhead travel time
            
        Returns:
        --------
        bool : True if trip can be accommodated
        """
        if date not in self.daily_states:
            return False
        
        return self.daily_states[date].can_add_trip(date, trip_duration_min, deadhead_min)
    
    def validate_cross_day_rest(self, date1: str, date2: str) -> Tuple[bool, str]:
        """
        Validate rest period between two consecutive work days.
        
        Parameters:
        -----------
        date1 : str
            Earlier date
        date2 : str  
            Later date (should be next day)
            
        Returns:
        --------
        Tuple[bool, str] : (is_compliant, reason_if_not)
        """
        if date1 not in self.daily_states or date2 not in self.daily_states:
            return True, "No work scheduled"
        
        return self.daily_states[date1].validate_rest_compliance(date1, date2)
    
    def simulate_insertion(self, insertion: CascadingInsertion) -> Tuple[bool, List[str]]:
        """
        Simulate a cascading insertion to check feasibility without modifying state.
        
        Parameters:
        -----------
        insertion : CascadingInsertion
            The insertion to simulate
            
        Returns:
        --------
        Tuple[bool, List[str]] : (is_feasible, list_of_issues)
        """
        issues = []
        
        # Create temporary copy of driver states for simulation
        temp_states = {}
        for date, state in self.daily_states.items():
            # Create shallow copy for simulation
            temp_states[date] = DriverState(
                driver_id=state.driver_id,
                route_id=state.route_id,
                daily_assignments=state.daily_assignments.copy(),
                emergency_rests_used_this_week=state.emergency_rests_used_this_week,
                emergency_rest_dates=state.emergency_rest_dates.copy()
            )
        
        # Simulate each step
        for step in insertion.steps:
            temp_state = temp_states.get(step.date)
            if not temp_state:
                issues.append(f"Date {step.date} not in schedule")
                continue
            
            # Check capacity
            if not temp_state.can_add_trip(step.date, step.new_trip_id, step.deadhead_minutes):
                issues.append(f"Driver {step.driver_id} exceeds capacity on {step.date}")
            
            # Check emergency rest quota
            if step.requires_emergency_rest and not self.can_use_emergency_rest():
                issues.append(f"No emergency rest quota remaining")
            
            # Simulate the assignment (without actually modifying state)
            # This would involve more complex logic to check rest periods, etc.
        
        return len(issues) == 0, issues
    
    def get_schedule_summary(self) -> Dict[str, any]:
        """Get comprehensive summary of this weekly schedule."""
        utilization = self.get_week_utilization()
        weekend_issues = self.validate_weekend_breaks()
        
        # Count total assignments
        total_assignments = 0
        for date, state in self.daily_states.items():
            if date in state.daily_assignments:
                total_assignments += len(state.daily_assignments[date])
        
        return {
            'driver_id': self.driver_id,
            'route_id': self.route_id,
            'week_start': self.week_start_date.strftime('%Y-%m-%d'),
            'total_assignments': total_assignments,
            'utilization': utilization,
            'weekend_compliance': len(weekend_issues) == 0,
            'weekend_issues': weekend_issues,
            'emergency_rests_used': self.total_emergency_rests_used,
            'has_weekend_work': any(len(self.daily_states[date].daily_assignments.get(date, [])) > 0 
                                  for date in self.get_weekend_days())
        }
    
    @classmethod
    def from_driver_state(cls, driver_state: DriverState, week_start: datetime) -> 'WeeklySchedule':
        """
        Create a WeeklySchedule from an existing DriverState.
        
        Parameters:
        -----------
        driver_state : DriverState
            Source driver state
        week_start : datetime
            Start of the week (should be a Monday)
            
        Returns:
        --------
        WeeklySchedule : New weekly schedule
        """
        schedule = cls(
            driver_id=driver_state.driver_id,
            route_id=driver_state.route_id,
            week_start_date=week_start
        )
        
        # Copy assignments from driver state to appropriate dates
        for date_str, assignments in driver_state.daily_assignments.items():
            if date_str in schedule.daily_states:
                for assignment in assignments:
                    schedule.add_assignment(date_str, assignment)
        
        # Copy emergency rest usage
        schedule.total_emergency_rests_used = driver_state.emergency_rests_used_this_week
        schedule.emergency_rest_dates = list(driver_state.emergency_rest_dates)
        
        return schedule
    
    @classmethod 
    def create_empty_week(cls, driver_id: str, route_id: str, week_start: datetime) -> 'WeeklySchedule':
        """
        Create an empty weekly schedule for a driver.
        
        Parameters:
        -----------
        driver_id : str
            Driver identifier
        route_id : str
            Route identifier
        week_start : datetime
            Start of week (Monday)
            
        Returns:
        --------
        WeeklySchedule : Empty weekly schedule
        """
        return cls(
            driver_id=driver_id,
            route_id=route_id,
            week_start_date=week_start
        )