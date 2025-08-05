"""
Diagnostic and fix for notebook import issues
"""

import sys
import os
from pathlib import Path

def diagnose_import_issue():
    """Diagnose and fix the models import issue."""
    
    print("🔍 DIAGNOSING IMPORT ISSUE")
    print("=" * 40)
    
    # Check current working directory
    current_dir = Path.cwd()
    print(f"📂 Current directory: {current_dir}")
    
    # Check if we're in the right location
    expected_structure = [
        "src/models/__init__.py",
        "src/models/driver_state.py", 
        "src/models/cascading_insertion.py",
        "src/models/weekly_schedule.py",
        "notebooks/",
        "data/processed/"
    ]
    
    print(f"\n📁 Checking project structure:")
    missing_files = []
    for file_path in expected_structure:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    # Check Python path
    print(f"\n🐍 Python path status:")
    src_path = current_dir / "src"
    if str(src_path) in sys.path:
        print(f"✅ {src_path} is in Python path")
    else:
        print(f"❌ {src_path} NOT in Python path")
        print(f"   Adding it now...")
        sys.path.insert(0, str(src_path))
        print(f"✅ Added {src_path} to Python path")
    
    # Test the import
    print(f"\n🧪 Testing import:")
    try:
        from models import DriverState, WeeklySchedule, CascadingInsertion
        print("✅ Import successful!")
        
        # Test basic functionality
        driver = DriverState(driver_id="TEST", route_id="TEST")
        print(f"✅ DriverState creation works: {driver.driver_id}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print(f"\n🔧 Debugging import error:")
        
        # Check if models directory exists
        models_dir = src_path / "models"
        if not models_dir.exists():
            print(f"❌ Models directory missing: {models_dir}")
            return False
        
        # Check __init__.py
        init_file = models_dir / "__init__.py"
        if not init_file.exists():
            print(f"❌ __init__.py missing: {init_file}")
            return False
        
        # Try to read __init__.py content
        try:
            with open(init_file, 'r') as f:
                init_content = f.read()
            print(f"📄 __init__.py exists ({len(init_content)} characters)")
            
            # Check for syntax errors in __init__.py
            try:
                compile(init_content, str(init_file), 'exec')
                print("✅ __init__.py compiles successfully")
            except SyntaxError as se:
                print(f"❌ Syntax error in __init__.py: {se}")
                
        except Exception as e:
            print(f"❌ Error reading __init__.py: {e}")
        
        return False

if __name__ == "__main__":
    success = diagnose_import_issue()
    
    if success:
        print("\n🎉 IMPORT ISSUE RESOLVED!")
        print("You can now use: from models import DriverState, WeeklySchedule, CascadingInsertion")
    else:
        print("\n🔧 IMPORT ISSUE NEEDS MANUAL FIX")
        print("Check the missing files above and ensure proper project structure")

# Run the diagnostic
diagnose_import_issue()