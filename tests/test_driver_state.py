"""
Test suite for DriverState class
File: tests/test_driver_state.py
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import DriverState, DailyAssignment, DAILY_DUTY_LIMIT_MIN


class TestDriverState:
    """Test suite for DriverState class functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.driver = DriverState(
            driver_id="DRIVER_001",
            route_id="ROUTE_001"
        )
        
        # Sample assignments
        self.assignment_8h = DailyAssignment(
            trip_id="TRIP_001",
            start_time=datetime(2025, 8, 5, 8, 0),
            end_time=datetime(2025, 8, 5, 16, 0),
            duration_minutes=480,  # 8 hours
            start_location="LOC_A",
            end_location="LOC_B"
        )
        
        self.assignment_4h = DailyAssignment(
            trip_id="TRIP_002", 
            start_time=datetime(2025, 8, 5, 18, 0),
            end_time=datetime(2025, 8, 5, 22, 0),
            duration_minutes=240,  # 4 hours
            start_location="LOC_C",
            end_location="LOC_D"
        )
    
    def test_empty_driver_capacity(self):
        """Test capacity calculations for driver with no assignments."""
        assert self.driver.get_daily_capacity_used("2025-08-05") == 0
        assert self.driver.get_daily_capacity_remaining("2025-08-05") == DAILY_DUTY_LIMIT_MIN
        assert self.driver.can_add_trip("2025-08-05", 480)  # 8 hours should fit
    
    def test_single_assignment_capacity(self):
        """Test capacity calculations with single assignment."""
        date = "2025-08-05"
        self.driver.add_assignment(date, self.assignment_8h)
        
        assert self.driver.get_daily_capacity_used(date) == 480
        assert self.driver.get_daily_capacity_remaining(date) == DAILY_DUTY_LIMIT_MIN - 480
        assert self.driver.can_add_trip(date, 240)  # 4 more hours should fit
        assert not self.driver.can_add_trip(date, 360)  # 6 more hours should NOT fit
    
    def test_multiple_assignments_capacity(self):
        """Test capacity with multiple assignments in one day."""
        date = "2025-08-05"
        self.driver.add_assignment(date, self.assignment_8h)
        self.driver.add_assignment(date, self.assignment_4h)
        
        total_used = 480 + 240  # 12 hours
        assert self.driver.get_daily_capacity_used(date) == total_used
        assert self.driver.get_daily_capacity_remaining(date) == DAILY_DUTY_LIMIT_MIN - total_used
        
        # Should have 1 hour remaining (13h limit - 12h used)
        remaining = DAILY_DUTY_LIMIT_MIN - total_used
        assert remaining == 60  # 1 hour
        assert self.driver.can_add_trip(date, 60)
        assert not self.driver.can_add_trip(date, 61)
    
    def test_capacity_with_deadhead_time(self):
        """Test capacity calculations including deadhead travel time."""
        date = "2025-08-05"
        assignment_with_deadhead = DailyAssignment(
            trip_id="TRIP_003",
            start_time=datetime(2025, 8, 5, 8, 0),
            end_time=datetime(2025, 8, 5, 16, 0),
            duration_minutes=480,  # 8 hours work
            start_location="LOC_A",
            end_location="LOC_B",
            deadhead_before_minutes=60,  # 1 hour to get there
            deadhead_after_minutes=30    # 30 min to next location
        )
        
        self.driver.add_assignment(date, assignment_with_deadhead)
        
        # Should include work + deadhead time
        expected_total = 480 + 60 + 30  # 9.5 hours
        assert self.driver.get_daily_capacity_used(date) == expected_total
    
    def test_work_day_bounds(self):
        """Test calculation of work day start and end times."""
        date = "2025-08-05"
        
        # No assignments
        start, end = self.driver.get_work_day_bounds(date)
        assert start is None and end is None
        
        # Single assignment
        self.driver.add_assignment(date, self.assignment_8h)
        start, end = self.driver.get_work_day_bounds(date)
        assert start == self.assignment_8h.start_time
        assert end == self.assignment_8h.end_time
        
        # Multiple assignments
        self.driver.add_assignment(date, self.assignment_4h)
        start, end = self.driver.get_work_day_bounds(date)
        assert start == self.assignment_8h.start_time  # Earlier start
        assert end == self.assignment_4h.end_time      # Later end
    
    def test_emergency_rest_quota(self):
        """Test emergency rest quota tracking."""
        assert self.driver.can_use_emergency_rest()
        assert self.driver.emergency_rests_used_this_week == 0
        
        # Use first emergency rest
        assert self.driver.use_emergency_rest("2025-08-05")
        assert self.driver.emergency_rests_used_this_week == 1
        assert "2025-08-05" in self.driver.emergency_rest_dates
        
        # Use second emergency rest
        assert self.driver.use_emergency_rest("2025-08-07")
        assert self.driver.emergency_rests_used_this_week == 2
        
        # Should not be able to use third
        assert not self.driver.can_use_emergency_rest()
        assert not self.driver.use_emergency_rest("2025-08-09")
        assert self.driver.emergency_rests_used_this_week == 2  # No change
    
    def test_rest_compliance_standard(self):
        """Test rest period compliance with standard 11-hour rest."""
        # Day 1: Work 8am-4pm
        date1 = "2025-08-05"
        assignment1 = DailyAssignment(
            trip_id="TRIP_001",
            start_time=datetime(2025, 8, 5, 8, 0),
            end_time=datetime(2025, 8, 5, 16, 0),  # Ends at 4 PM
            duration_minutes=480,
            start_location="LOC_A",
            end_location="LOC_B"
        )

        # Day 2: Work 1am-9am (9 hours rest from 4pm to 1am - TOO SHORT EVEN FOR EMERGENCY)
        date2 = "2025-08-06"
        assignment2 = DailyAssignment(
            trip_id="TRIP_002",
            start_time=datetime(2025, 8, 6, 1, 0),  # 1 AM = 9 hours rest (exactly at emergency limit)
            end_time=datetime(2025, 8, 6, 9, 0),    # 9 AM
            duration_minutes=480,
            start_location="LOC_C",
            end_location="LOC_D"
        )

        self.driver.add_assignment(date1, assignment1)
        self.driver.add_assignment(date2, assignment2)

        is_compliant, reason = self.driver.validate_rest_compliance(date1, date2)
        
        # 9 hours is exactly the emergency rest limit, so it should be compliant with emergency rest
        assert is_compliant
        assert "Emergency rest needed" in reason
        
        # Now test with LESS than 9 hours (8.5 hours) - this should truly fail
        assignment2_too_early = DailyAssignment(
            trip_id="TRIP_002", 
            start_time=datetime(2025, 8, 6, 0, 30),  # 12:30 AM = 8.5 hours rest
            end_time=datetime(2025, 8, 6, 8, 30),    
            duration_minutes=480,
            start_location="LOC_C",
            end_location="LOC_D"
        )
        
        self.driver.remove_assignment(date2, "TRIP_002")
        self.driver.add_assignment(date2, assignment2_too_early)
        
        is_compliant, reason = self.driver.validate_rest_compliance(date1, date2)
        assert not is_compliant  # 8.5 hours should NOT be compliant (less than 9h emergency minimum)
        assert "Insufficient rest" in reason
    
    def test_rest_compliance_emergency(self):
        """Test rest period compliance with emergency 9-hour rest."""
        # Day 1: Work 8am-4pm  
        date1 = "2025-08-05"
        assignment1 = DailyAssignment(
            trip_id="TRIP_001",
            start_time=datetime(2025, 8, 5, 8, 0),
            end_time=datetime(2025, 8, 5, 16, 0),
            duration_minutes=480,
            start_location="LOC_A", 
            end_location="LOC_B"
        )
        
        # Day 2: Work 3am-11am (11 hours rest - emergency acceptable)
        date2 = "2025-08-06"
        assignment2 = DailyAssignment(
            trip_id="TRIP_002",
            start_time=datetime(2025, 8, 6, 3, 0),
            end_time=datetime(2025, 8, 6, 11, 0),
            duration_minutes=480,
            start_location="LOC_C",
            end_location="LOC_D"
        )
        
        self.driver.add_assignment(date1, assignment1)
        self.driver.add_assignment(date2, assignment2)
        
        is_compliant, reason = self.driver.validate_rest_compliance(date1, date2)
        assert is_compliant
        # Should not need emergency rest for 11 hours
        assert "Emergency rest" not in reason
        
        # Now test with 9.5 hours rest (emergency needed)
        assignment2_early = DailyAssignment(
            trip_id="TRIP_002",
            start_time=datetime(2025, 8, 6, 1, 30),  # 9.5 hours after 4pm
            end_time=datetime(2025, 8, 6, 9, 30),
            duration_minutes=480,
            start_location="LOC_C",
            end_location="LOC_D"
        )
        
        self.driver.remove_assignment(date2, "TRIP_002")
        self.driver.add_assignment(date2, assignment2_early)
        
        is_compliant, reason = self.driver.validate_rest_compliance(date1, date2)
        assert is_compliant
        assert "Emergency rest needed" in reason
    
    def test_assignment_sorting(self):
        """Test that assignments are kept sorted by start time."""
        date = "2025-08-05"
        
        # Add assignments out of order
        late_assignment = DailyAssignment(
            trip_id="LATE",
            start_time=datetime(2025, 8, 5, 20, 0),
            end_time=datetime(2025, 8, 5, 22, 0),
            duration_minutes=120,
            start_location="LOC_A",
            end_location="LOC_B"
        )
        
        early_assignment = DailyAssignment(
            trip_id="EARLY",
            start_time=datetime(2025, 8, 5, 8, 0),
            end_time=datetime(2025, 8, 5, 10, 0),
            duration_minutes=120,
            start_location="LOC_A",
            end_location="LOC_B"
        )
        
        # Add late one first
        self.driver.add_assignment(date, late_assignment)
        self.driver.add_assignment(date, early_assignment)
        
        # Should be sorted by start time
        assignments = self.driver.daily_assignments[date]
        assert len(assignments) == 2
        assert assignments[0].trip_id == "EARLY"
        assert assignments[1].trip_id == "LATE"
    
    def test_utilization_summary(self):
        """Test utilization summary calculations."""
        # Empty driver
        summary = self.driver.get_utilization_summary()
        assert summary['total_days'] == 0
        assert summary['avg_utilization'] == 0.0
        
        # Add some assignments
        self.driver.add_assignment("2025-08-05", self.assignment_8h)
        self.driver.add_assignment("2025-08-06", self.assignment_4h)
        
        summary = self.driver.get_utilization_summary()
        assert summary['total_days'] == 2
        
        # 8h + 4h = 12h total, across 2 days = 26h available, so 12/26 = ~46% utilization
        expected_utilization = (480 + 240) / (2 * DAILY_DUTY_LIMIT_MIN)
        assert abs(summary['avg_utilization'] - expected_utilization) < 0.01
        
        assert summary['total_capacity_used_hours'] == (480 + 240) / 60  # 12 hours
        assert summary['emergency_rests_used'] == 0


class TestDriverStateFromData:
    """Test DriverState creation from pandas DataFrames."""
    
    def test_from_route_data(self):
        """Test creating DriverState from route and trip data."""
        # Mock route data
        route_df = pd.DataFrame([{
            'route_schedule_uuid': 'ROUTE_001',
            'route_start_time': datetime(2025, 8, 5, 8, 0),
            'route_end_time': datetime(2025, 8, 6, 18, 0),
            'num_trips': 2
        }])
        
        # Mock trip data
        trips_df = pd.DataFrame([
            {
                'trip_uuid': 'TRIP_001',
                'od_start_time': datetime(2025, 8, 5, 8, 0),
                'od_end_time': datetime(2025, 8, 5, 16, 0),
                'trip_duration_minutes': 480,
                'source_center': 'LOC_A',
                'destination_center': 'LOC_B'
            },
            {
                'trip_uuid': 'TRIP_002',
                'od_start_time': datetime(2025, 8, 6, 10, 0),
                'od_end_time': datetime(2025, 8, 6, 18, 0),
                'trip_duration_minutes': 480,
                'source_center': 'LOC_C',
                'destination_center': 'LOC_D'
            }
        ])
        
        driver_state = DriverState.from_route_data(route_df, trips_df)
        
        assert driver_state.driver_id == 'ROUTE_001'
        assert driver_state.route_id == 'ROUTE_001'
        assert len(driver_state.daily_assignments) == 2
        assert '2025-08-05' in driver_state.daily_assignments
        assert '2025-08-06' in driver_state.daily_assignments
        
        # Check assignments
        day1_assignments = driver_state.daily_assignments['2025-08-05']
        assert len(day1_assignments) == 1
        assert day1_assignments[0].trip_id == 'TRIP_001'
        assert day1_assignments[0].duration_minutes == 480


if __name__ == "__main__":
    # Run basic tests
    test_driver = TestDriverState()
    test_driver.setup_method()
    
    print("Testing empty driver capacity...")
    test_driver.test_empty_driver_capacity()
    print("✓ Passed")
    
    print("Testing single assignment capacity...")
    test_driver.test_single_assignment_capacity() 
    print("✓ Passed")
    
    print("Testing emergency rest quota...")
    test_driver.test_emergency_rest_quota()
    print("✓ Passed")
    
    print("Testing rest compliance...")
    test_driver.test_rest_compliance_standard()
    print("✓ Passed")
    
    print("\nAll basic tests passed! DriverState class is working correctly.")
    print(f"Daily duty limit: {DAILY_DUTY_LIMIT_MIN/60:.1f} hours")
    print(f"Driver can track capacity, rest periods, and regulatory compliance.")