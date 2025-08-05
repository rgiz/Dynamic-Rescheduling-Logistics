"""
Quick verification script to test imports after reorganization.
File: verify_imports.py (run from project root)
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path("src")
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def test_imports():
    """Test all model imports work correctly."""
    print("Testing model imports...")
    
    try:
        # Test package-level imports
        from models import (
            DriverState, DailyAssignment, 
            CascadingInsertion, InsertionStep, InsertionType,
            WeeklySchedule, WeekendBreak,
            DAILY_DUTY_LIMIT_MIN, WEEKEND_REST_MIN
        )
        print("✅ Package-level imports successful")
        
        # Test individual module imports  
        from models.driver_state import DriverState as DS1
        from models.cascading_insertion import CascadingInsertion as CI1
        from models.weekly_schedule import WeeklySchedule as WS1
        print("✅ Individual module imports successful")
        
        # Test basic instantiation
        driver = DriverState(driver_id="TEST", route_id="TEST")
        print(f"✅ DriverState creation successful: {driver.driver_id}")
        
        insertion = CascadingInsertion.create_direct_insertion(
            disrupted_trip_id="TEST_TRIP",
            driver_id="TEST_DRIVER", 
            date="2025-08-05",
            position=0
        )
        print(f"✅ CascadingInsertion creation successful: {insertion.disrupted_trip_id}")
        
        from datetime import datetime
        schedule = WeeklySchedule.create_empty_week(
            driver_id="TEST",
            route_id="TEST", 
            week_start=datetime(2025, 8, 4)
        )
        print(f"✅ WeeklySchedule creation successful: {schedule.driver_id}")
        
        # Test constants
        print(f"✅ Constants accessible: Daily limit = {DAILY_DUTY_LIMIT_MIN/60:.1f}h")
        
        print("\n🎉 All imports working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_individual_files():
    """Test that individual files exist and are properly structured."""
    expected_files = [
        "src/models/__init__.py",
        "src/models/driver_state.py", 
        "src/models/cascading_insertion.py",
        "src/models/weekly_schedule.py"
    ]
    
    print("\nChecking file structure...")
    missing_files = []
    
    for file_path in expected_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        return False
    else:
        print("\n✅ All expected files present")
        return True

if __name__ == "__main__":
    print("=" * 50)
    print("MODEL REORGANIZATION VERIFICATION")
    print("=" * 50)
    
    # Check file structure first
    files_ok = test_individual_files()
    
    if files_ok:
        # Test imports
        imports_ok = test_imports()
        
        if imports_ok:
            print("\n🚀 Ready to run tests!")
            print("Try running: python -m pytest tests/ -v")
        else:
            print("\n⚠️  Fix import issues before running tests")
    else:
        print("\n⚠️  Create missing files before testing imports")