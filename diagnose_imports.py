"""
Detailed import diagnostic to find the exact issue.
File: diagnose_imports.py (run from project root)
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path("src")
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def test_individual_modules():
    """Test each module individually to isolate the problem."""
    print("Testing individual module imports...")
    
    # Test 1: driver_state.py
    try:
        print("1. Testing driver_state.py...")
        from models.driver_state import DriverState, DailyAssignment
        print("   ✅ driver_state imports successful")
        
        # Test basic creation
        driver = DriverState(driver_id="TEST", route_id="TEST") 
        print(f"   ✅ DriverState creation works: {driver.driver_id}")
        
    except Exception as e:
        print(f"   ❌ driver_state failed: {e}")
        return False
    
    # Test 2: cascading_insertion.py  
    try:
        print("2. Testing cascading_insertion.py...")
        from models.cascading_insertion import CascadingInsertion, InsertionStep, InsertionType
        print("   ✅ cascading_insertion imports successful")
        
        # Test basic creation
        insertion = CascadingInsertion.create_direct_insertion(
            disrupted_trip_id="TEST", driver_id="TEST", date="2025-08-05", position=0
        )
        print(f"   ✅ CascadingInsertion creation works: {insertion.disrupted_trip_id}")
        
    except Exception as e:
        print(f"   ❌ cascading_insertion failed: {e}")
        return False
    
    # Test 3: weekly_schedule.py
    try:
        print("3. Testing weekly_schedule.py...")
        from models.weekly_schedule import WeeklySchedule, WeekendBreak
        print("   ✅ weekly_schedule imports successful")
        
        # Test basic creation
        from datetime import datetime
        schedule = WeeklySchedule.create_empty_week(
            driver_id="TEST", route_id="TEST", week_start=datetime(2025, 8, 4)
        )
        print(f"   ✅ WeeklySchedule creation works: {schedule.driver_id}")
        
    except Exception as e:
        print(f"   ❌ weekly_schedule failed: {e}")
        return False
    
    print("✅ All individual modules work!")
    return True

def test_package_init():
    """Test the __init__.py package imports."""
    print("\n4. Testing package __init__.py...")
    
    try:
        # This is what's failing in the main script
        from models import DriverState
        print("   ✅ models package import successful")
        return True
    except ImportError as e:
        print(f"   ❌ Package import failed: {e}")
        
        # Let's check what's actually in the __init__.py
        init_file = Path("src/models/__init__.py")
        if init_file.exists():
            print(f"   📄 __init__.py exists (size: {init_file.stat().st_size} bytes)")
            print("   First few lines of __init__.py:")
            with open(init_file, 'r') as f:
                for i, line in enumerate(f):
                    if i < 10:  # Show first 10 lines
                        print(f"      {i+1}: {line.rstrip()}")
                    else:
                        break
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error in package import: {e}")
        return False

def check_circular_imports():
    """Check for potential circular import issues."""
    print("\n5. Checking for circular imports...")
    
    files_to_check = [
        "src/models/driver_state.py",
        "src/models/cascading_insertion.py", 
        "src/models/weekly_schedule.py"
    ]
    
    for file_path in files_to_check:
        print(f"   📄 {file_path}:")
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Check for problematic imports
            import_lines = [line.strip() for line in content.split('\n') 
                          if line.strip().startswith(('from models', 'import models'))]
            
            if import_lines:
                for line in import_lines:
                    print(f"      🔍 {line}")
            else:
                print("      ✅ No circular imports detected")
                
        except Exception as e:
            print(f"      ❌ Could not read file: {e}")

def main():
    print("=" * 60)
    print("DETAILED IMPORT DIAGNOSTIC")
    print("=" * 60)
    
    # Test individual modules first
    individual_ok = test_individual_modules()
    
    if individual_ok:
        # Test package import
        package_ok = test_package_init()
        
        if not package_ok:
            # Check for circular imports if package fails
            check_circular_imports()
    
    print("\n" + "=" * 60)
    if individual_ok and package_ok:
        print("🎉 All imports working! Ready to run tests.")
    else:
        print("🔧 Fix the issues above, then run tests.")

if __name__ == "__main__":
    main()