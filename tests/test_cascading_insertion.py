"""
Test suite for CascadingInsertion class
File: tests/test_cascading_insertion.py
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import CascadingInsertion, InsertionStep, InsertionType, DriverState, DailyAssignment
from datetime import datetime


class TestInsertionStep:
    """Test InsertionStep functionality."""
    
    def test_insertion_step_creation(self):
        """Test basic InsertionStep creation."""
        step = InsertionStep(
            driver_id="DRIVER_001",
            original_trip_id="TRIP_001",
            new_trip_id="DISRUPTED_TRIP",
            date="2025-08-05",
            position=1,
            deadhead_minutes=30,
            creates_delay_minutes=15,
            requires_emergency_rest=False,
            cost_impact=25.0
        )
        
        assert step.driver_id == "DRIVER_001"
        assert step.original_trip_id == "TRIP_001"
        assert step.new_trip_id == "DISRUPTED_TRIP"
        assert step.deadhead_minutes == 30
        assert not step.requires_emergency_rest
    
    def test_insertion_step_repr(self):
        """Test string representation of InsertionStep."""
        step = InsertionStep(
            driver_id="DRIVER_001",
            original_trip_id="TRIP_001",
            new_trip_id="DISRUPTED_TRIP",
            date="2025-08-05",
            position=1,
            deadhead_minutes=30,
            creates_delay_minutes=0,
            requires_emergency_rest=True,
            cost_impact=25.0
        )
        
        repr_str = repr(step)
        assert "DRIVER_001" in repr_str
        assert "DISRUPTED_TRIP" in repr_str
        assert "displacing TRIP_001" in repr_str
        assert "EMERGENCY REST" in repr_str


class TestCascadingInsertion:
    """Test CascadingInsertion functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.disrupted_trip = "DISRUPTED_001"
    
    def test_direct_insertion_creation(self):
        """Test creating a direct insertion."""
        insertion = CascadingInsertion.create_direct_insertion(
            disrupted_trip_id=self.disrupted_trip,
            driver_id="DRIVER_001",
            date="2025-08-05",
            position=2,
            deadhead_minutes=45,
            creates_delay_minutes=0,
            cost_impact=30.0
        )
        
        assert insertion.disrupted_trip_id == self.disrupted_trip
        assert insertion.insertion_type == InsertionType.DIRECT
        assert len(insertion.steps) == 1
        assert len(insertion.drivers_affected) == 1
        assert insertion.total_deadhead_minutes == 45
        assert insertion.is_feasible
        
        step = insertion.steps[0]
        assert step.driver_id == "DRIVER_001"
        assert step.original_trip_id is None  # No displacement
        assert step.new_trip_id == self.disrupted_trip
    
    def test_cascade_insertion_creation(self):
        """Test creating a 2-driver cascade insertion."""
        steps_data = [
            {
                'driver_id': 'DRIVER_A',
                'original_trip_id': 'TRIP_A1',
                'new_trip_id': 'DISRUPTED_001',
                'date': '2025-08-05',
                'position': 1,
                'deadhead_minutes': 30,
                'creates_delay_minutes': 0,
                'requires_emergency_rest': False,
                'cost_impact': 20.0
            },
            {
                'driver_id': 'DRIVER_B',
                'original_trip_id': None,
                'new_trip_id': 'TRIP_A1',
                'date': '2025-08-05',
                'position': 0,
                'deadhead_minutes': 60,
                'creates_delay_minutes': 30,
                'requires_emergency_rest': False,
                'cost_impact': 35.0
            }
        ]
        
        insertion = CascadingInsertion.create_cascade_insertion(
            disrupted_trip_id=self.disrupted_trip,
            steps_data=steps_data
        )
        
        assert insertion.insertion_type == InsertionType.CASCADE_2
        assert len(insertion.steps) == 2
        assert len(insertion.drivers_affected) == 2
        assert insertion.total_deadhead_minutes == 90  # 30 + 60
        assert insertion.total_delay_minutes == 30
        assert insertion.total_cost_impact == 55.0  # 20 + 35
    
    def test_multi_cascade_insertion(self):
        """Test creating a 3+ driver cascade insertion."""
        steps_data = [
            {'driver_id': 'A', 'original_trip_id': 'TA', 'new_trip_id': 'DISRUPTED', 
             'date': '2025-08-05', 'position': 0, 'deadhead_minutes': 20, 
             'creates_delay_minutes': 0, 'requires_emergency_rest': False, 'cost_impact': 10.0},
            {'driver_id': 'B', 'original_trip_id': 'TB', 'new_trip_id': 'TA',
             'date': '2025-08-05', 'position': 1, 'deadhead_minutes': 30,
             'creates_delay_minutes': 0, 'requires_emergency_rest': False, 'cost_impact': 15.0},
            {'driver_id': 'C', 'original_trip_id': None, 'new_trip_id': 'TB',
             'date': '2025-08-05', 'position': 0, 'deadhead_minutes': 40,
             'creates_delay_minutes': 0, 'requires_emergency_rest': False, 'cost_impact': 20.0}
        ]
        
        insertion = CascadingInsertion.create_cascade_insertion(
            disrupted_trip_id=self.disrupted_trip,
            steps_data=steps_data
        )
        
        assert insertion.insertion_type == InsertionType.CASCADE_MULTI
        assert len(insertion.steps) == 3
        assert len(insertion.drivers_affected) == 3
    
    def test_complexity_scoring(self):
        """Test complexity score calculation."""
        # Simple direct insertion
        direct = CascadingInsertion.create_direct_insertion(
            disrupted_trip_id=self.disrupted_trip,
            driver_id="DRIVER_001",
            date="2025-08-05",
            position=0,
            deadhead_minutes=20,
            creates_delay_minutes=0
        )
        
        # Complex cascade with delays and emergency rest
        complex_insertion = CascadingInsertion(
            disrupted_trip_id=self.disrupted_trip,
            insertion_type=InsertionType.CASCADE_2
        )
        
        step1 = InsertionStep(
            driver_id="DRIVER_A", original_trip_id="TRIP_A", new_trip_id=self.disrupted_trip,
            date="2025-08-05", position=0, deadhead_minutes=60, creates_delay_minutes=120,
            requires_emergency_rest=True, cost_impact=50.0
        )
        step2 = InsertionStep(
            driver_id="DRIVER_B", original_trip_id=None, new_trip_id="TRIP_A",
            date="2025-08-05", position=0, deadhead_minutes=30, creates_delay_minutes=0,
            requires_emergency_rest=False, cost_impact=25.0
        )
        
        complex_insertion.add_step(step1)
        complex_insertion.add_step(step2)
        
        direct_score = direct.get_complexity_score()
        complex_score = complex_insertion.get_complexity_score()
        
        # Complex insertion should have higher score
        assert complex_score > direct_score
        assert complex_insertion.emergency_rests_required == 1
    
    def test_cost_estimation(self):
        """Test cost estimation with different weights."""
        insertion = CascadingInsertion(
            disrupted_trip_id=self.disrupted_trip,
            insertion_type=InsertionType.DIRECT
        )
        
        step = InsertionStep(
            driver_id="DRIVER_001", original_trip_id=None, new_trip_id=self.disrupted_trip,
            date="2025-08-05", position=0, deadhead_minutes=60, creates_delay_minutes=30,
            requires_emergency_rest=True, cost_impact=0.0
        )
        insertion.add_step(step)
        
        # Test with default weights
        default_cost = insertion.get_cost_estimate()
        assert default_cost > 0
        
        # Test with custom weights favoring delay over deadhead
        delay_focused_weights = {'deadhead': 0.5, 'delay': 10.0, 'emergency': 50.0, 'reassignment': 10.0}
        delay_cost = insertion.get_cost_estimate(delay_focused_weights)
        
        # Should be different due to different weights
        assert delay_cost != default_cost
    
    def test_service_impact_scoring(self):
        """Test service impact scoring."""
        # Low impact insertion
        low_impact = CascadingInsertion.create_direct_insertion(
            disrupted_trip_id=self.disrupted_trip,
            driver_id="DRIVER_001",
            date="2025-08-05", 
            position=0,
            creates_delay_minutes=0
        )
        
        # High impact insertion
        high_impact = CascadingInsertion(
            disrupted_trip_id=self.disrupted_trip,
            insertion_type=InsertionType.CASCADE_MULTI
        )
        
        for i in range(3):  # 3 drivers affected
            step = InsertionStep(
                driver_id=f"DRIVER_{i}", original_trip_id=f"TRIP_{i}" if i < 2 else None,
                new_trip_id=self.disrupted_trip if i == 0 else f"TRIP_{i-1}",
                date="2025-08-05", position=0, deadhead_minutes=30, creates_delay_minutes=60,
                requires_emergency_rest=True, cost_impact=25.0
            )
            high_impact.add_step(step)
        
        low_score = low_impact.get_service_impact_score()
        high_score = high_impact.get_service_impact_score()
        
        assert high_score > low_score
    
    def test_chain_description(self):
        """Test human-readable chain descriptions."""
        # Direct insertion
        direct = CascadingInsertion.create_direct_insertion(
            disrupted_trip_id="DISRUPTED_001",
            driver_id="DRIVER_A",
            date="2025-08-05",
            position=0
        )
        
        description = direct.get_chain_description()
        assert "Direct: DRIVER_A takes DISRUPTED_001" == description
        
        # Cascade insertion
        steps_data = [
            {'driver_id': 'DRIVER_A', 'original_trip_id': 'TRIP_A', 'new_trip_id': 'DISRUPTED_001',
             'date': '2025-08-05', 'position': 0, 'deadhead_minutes': 0, 'creates_delay_minutes': 0,
             'requires_emergency_rest': False, 'cost_impact': 0.0},
            {'driver_id': 'DRIVER_B', 'original_trip_id': None, 'new_trip_id': 'TRIP_A',
             'date': '2025-08-05', 'position': 0, 'deadhead_minutes': 0, 'creates_delay_minutes': 0,
             'requires_emergency_rest': False, 'cost_impact': 0.0}
        ]
        
        cascade = CascadingInsertion.create_cascade_insertion("DISRUPTED_001", steps_data)
        description = cascade.get_chain_description()
        
        assert "DRIVER_A takes DISRUPTED_001 → DRIVER_B takes TRIP_A" == description
    
    def test_infeasibility_handling(self):
        """Test marking insertions as infeasible."""
        insertion = CascadingInsertion.create_direct_insertion(
            disrupted_trip_id=self.disrupted_trip,
            driver_id="DRIVER_001",
            date="2025-08-05",
            position=0
        )
        
        assert insertion.is_feasible
        
        insertion.mark_infeasible("Driver exceeds daily capacity")
        assert not insertion.is_feasible
        assert insertion.infeasibility_reason == "Driver exceeds daily capacity"
    
    def test_summary_statistics(self):
        """Test summary statistics generation."""
        insertion = CascadingInsertion.create_direct_insertion(
            disrupted_trip_id=self.disrupted_trip,
            driver_id="DRIVER_001", 
            date="2025-08-05",
            position=0,
            deadhead_minutes=45,
            creates_delay_minutes=30
        )
        
        summary = insertion.get_summary()
        
        assert summary['disrupted_trip_id'] == self.disrupted_trip
        assert summary['insertion_type'] == 'direct'
        assert summary['num_steps'] == 1
        assert summary['drivers_affected'] == 1
        assert summary['total_deadhead_hours'] == 0.75  # 45 minutes
        assert summary['total_delay_hours'] == 0.5      # 30 minutes
        assert summary['is_feasible'] == True
        assert 'complexity_score' in summary
        assert 'estimated_cost' in summary


if __name__ == "__main__":
    # Run basic tests
    test_step = TestInsertionStep()
    test_step.test_insertion_step_creation()
    print("✓ InsertionStep creation test passed")
    
    test_cascade = TestCascadingInsertion()
    test_cascade.setup_method()
    
    test_cascade.test_direct_insertion_creation()
    print("✓ Direct insertion test passed")
    
    test_cascade.test_cascade_insertion_creation()
    print("✓ Cascade insertion test passed")
    
    test_cascade.test_complexity_scoring()
    print("✓ Complexity scoring test passed")
    
    test_cascade.test_chain_description()
    print("✓ Chain description test passed")
    
    print("\nAll CascadingInsertion tests passed!")
    print("The class can handle direct insertions, cascading reassignments, and impact calculations.")