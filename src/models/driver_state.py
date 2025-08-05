"""
Enhanced data models for multi-driver optimization.
File: src/models.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
from datetime import datetime, timedelta
import pandas as pd

# Import constraints - we'll define these constants
DAILY_DUTY_LIMIT_MIN = 13 * 60      # 13 hours in minutes
STANDARD_REST_MIN = 11 * 60         # 11 hours in minutes
EMERGENCY_REST_MIN = 9 * 60         # 9 hours in minutes
WEEKEND_REST_MIN = 45 * 60          # 45 hours in minutes
MAX_EMERGENCY_PER_WEEK = 2          # Maximum emergency rests per week
MAX_DELAY_TOLERANCE_MIN = 2 * 60    # 2 hours delay tolerance


@dataclass
class DailyAssignment:
    """Represents a single trip assignment within a driver's day."""
    trip_id: str
    start_time: datetime
    end_time: datetime
    duration_minutes: int
    start_location: str
    end_location: str
    is_original: bool = True  # False if this is a reassigned trip
    deadhead_before_minutes: int = 0  # Travel time to reach this trip
    deadhead_after_minutes: int = 0   # Travel time from this trip to next


@dataclass
class DriverState:
    """
    Tracks a driver's state across multiple days including capacity,
    rest periods, and regulatory compliance.
    """
    driver_id: str
    route_id: str  # The multi-day route this driver is assigned to
    
    # Daily assignments organized by date
    daily_assignments: Dict[str, List[DailyAssignment]] = field(default_factory=dict)
    
    # Rest period tracking
    emergency_rests_used_this_week: int = 0
    emergency_rest_dates: Set[str] = field(default_factory=set)
    
    # Weekend tracking
    last_weekend_end: Optional[datetime] = None
    next_weekend_start: Optional[datetime] = None
    
    def get_daily_capacity_used(self, date: str) -> int:
        """Return total minutes used on a specific date (including deadhead)."""
        if date not in self.daily_assignments:
            return 0
        
        assignments = self.daily_assignments[date]
        if not assignments:
            return 0
        
        total_work = sum(a.duration_minutes for a in assignments)
        total_deadhead = sum(a.deadhead_before_minutes + a.deadhead_after_minutes 
                           for a in assignments)
        return total_work + total_deadhead
    
    def get_daily_capacity_remaining(self, date: str) -> int:
        """Return remaining capacity in minutes for a specific date."""
        used = self.get_daily_capacity_used(date)
        return max(0, DAILY_DUTY_LIMIT_MIN - used)
    
    def can_add_trip(self, date: str, trip_duration_min: int, deadhead_min: int = 0) -> bool:
        """Check if driver can handle additional trip on specified date."""
        total_additional = trip_duration_min + deadhead_min
        remaining = self.get_daily_capacity_remaining(date)
        return total_additional <= remaining
    
    def get_work_day_bounds(self, date: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """Return start and end times for a driver's work day."""
        if date not in self.daily_assignments or not self.daily_assignments[date]:
            return None, None
        
        assignments = sorted(self.daily_assignments[date], key=lambda x: x.start_time)
        earliest_start = assignments[0].start_time
        latest_end = max(a.end_time for a in assignments)
        
        return earliest_start, latest_end
    
    def can_use_emergency_rest(self) -> bool:
        """Check if driver can use emergency rest (9h instead of 11h)."""
        return self.emergency_rests_used_this_week < MAX_EMERGENCY_PER_WEEK
    
    def get_required_rest_before(self, date: str, use_emergency: bool = False) -> int:
        """Get required rest period before starting work on specified date."""
        if use_emergency and self.can_use_emergency_rest():
            return EMERGENCY_REST_MIN
        return STANDARD_REST_MIN
    
    def validate_rest_compliance(self, date: str, next_date: str) -> tuple[bool, str]:
        """
        Validate that rest period between two work days meets requirements.
        Returns (is_compliant, reason_if_not)
        """
        current_day_start, current_day_end = self.get_work_day_bounds(date)
        next_day_start, next_day_end = self.get_work_day_bounds(next_date)
        
        if not current_day_end or not next_day_start:
            return True, "No work scheduled"
        
        rest_period = (next_day_start - current_day_end).total_seconds() / 60
        
        # Check if this spans a weekend
        if self._spans_weekend(current_day_end, next_day_start):
            if rest_period < WEEKEND_REST_MIN:
                return False, f"Weekend rest too short: {rest_period/60:.1f}h < {WEEKEND_REST_MIN/60}h"
        
        # Check regular rest requirements
        required_rest = self.get_required_rest_before(next_date)
        if rest_period < required_rest:
            # Could we use emergency rest?
            if self.can_use_emergency_rest() and rest_period >= EMERGENCY_REST_MIN:
                return True, f"Emergency rest needed: {rest_period/60:.1f}h"
            else:
                return False, f"Insufficient rest: {rest_period/60:.1f}h < {required_rest/60:.1f}h"
        
        return True, "Compliant"
    
    def add_assignment(self, date: str, assignment: DailyAssignment) -> bool:
        """
        Add a trip assignment to a specific date.
        Returns True if successful, False if it violates capacity.
        """
        if not self.can_add_trip(date, assignment.duration_minutes, 
                                assignment.deadhead_before_minutes + assignment.deadhead_after_minutes):
            return False
        
        if date not in self.daily_assignments:
            self.daily_assignments[date] = []
        
        self.daily_assignments[date].append(assignment)
        # Keep assignments sorted by start time
        self.daily_assignments[date].sort(key=lambda x: x.start_time)
        return True
    
    def remove_assignment(self, date: str, trip_id: str) -> Optional[DailyAssignment]:
        """Remove a trip assignment from a specific date."""
        if date not in self.daily_assignments:
            return None
        
        for i, assignment in enumerate(self.daily_assignments[date]):
            if assignment.trip_id == trip_id:
                return self.daily_assignments[date].pop(i)
        
        return None
    
    def use_emergency_rest(self, date: str) -> bool:
        """
        Mark emergency rest as used for a specific date.
        Returns True if successful, False if quota exceeded.
        """
        if not self.can_use_emergency_rest():
            return False
        
        self.emergency_rests_used_this_week += 1
        self.emergency_rest_dates.add(date)
        return True
    
    def get_utilization_summary(self) -> Dict[str, any]:
        """Return summary statistics for this driver's utilization."""
        total_days = len(self.daily_assignments)
        if total_days == 0:
            return {'total_days': 0, 'avg_utilization': 0.0, 'emergency_rests': 0}
        
        total_capacity_used = sum(self.get_daily_capacity_used(date) 
                                for date in self.daily_assignments.keys())
        total_capacity_available = total_days * DAILY_DUTY_LIMIT_MIN
        avg_utilization = total_capacity_used / total_capacity_available if total_capacity_available > 0 else 0
        
        return {
            'total_days': total_days,
            'avg_utilization': avg_utilization,
            'total_capacity_used_hours': total_capacity_used / 60,
            'emergency_rests_used': self.emergency_rests_used_this_week,
            'capacity_by_date': {date: self.get_daily_capacity_used(date) / 60 
                               for date in self.daily_assignments.keys()}
        }
    
    def _spans_weekend(self, start: datetime, end: datetime) -> bool:
        """Check if a time period spans a weekend."""
        # Simple weekend detection - could be enhanced based on business rules
        start_weekday = start.weekday()  # 0=Monday, 6=Sunday
        end_weekday = end.weekday()
        
        # If we go from Friday (4) to Monday (0) or span Saturday/Sunday
        if start_weekday >= 4 and end_weekday <= 1:
            return True
        if (end - start).days >= 2:  # Multi-day span likely includes weekend
            return True
        
        return False

    @classmethod
    def from_route_data(cls, route_df: pd.DataFrame, trips_df: pd.DataFrame) -> 'DriverState':
        """
        Create DriverState from existing route and trip data.
        
        Parameters:
        -----------
        route_df : pd.DataFrame
            Single row containing route information
        trips_df : pd.DataFrame  
            All trips for this route, with columns: trip_uuid, od_start_time, od_end_time, etc.
        """
        if len(route_df) != 1:
            raise ValueError("route_df must contain exactly one route")
        
        route_row = route_df.iloc[0]
        driver_id = route_row['route_schedule_uuid']  # Using route as driver ID for now
        
        # Create driver state
        driver_state = cls(
            driver_id=driver_id,
            route_id=driver_id
        )
        
        # Process each trip into daily assignments
        for _, trip in trips_df.iterrows():
            date_str = trip['od_start_time'].strftime('%Y-%m-%d')
            
            assignment = DailyAssignment(
                trip_id=trip['trip_uuid'],
                start_time=trip['od_start_time'],
                end_time=trip['od_end_time'], 
                duration_minutes=int(trip['trip_duration_minutes']),
                start_location=trip['source_center'],
                end_location=trip['destination_center'],
                is_original=True
            )
            
            driver_state.add_assignment(date_str, assignment)
        
        return driver_state