"""
Check the actual content of driver_state.py to find the issue.
File: check_driver_state.py
"""

from pathlib import Path

def check_driver_state_file():
    """Check what's actually in the driver_state.py file."""
    
    file_path = Path("src/models/driver_state.py")
    
    if not file_path.exists():
        print("❌ driver_state.py not found!")
        return
    
    print(f"📄 Reading {file_path} (size: {file_path.stat().st_size} bytes)")
    print("=" * 60)
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        print(f"Total lines: {len(lines)}")
        print("\nFirst 30 lines:")
        print("-" * 40)
        
        for i, line in enumerate(lines[:30]):
            print(f"{i+1:2d}: {line}")
        
        print("\n" + "-" * 40)
        
        # Look for class definitions
        class_lines = []
        for i, line in enumerate(lines):
            if line.strip().startswith('class '):
                class_lines.append((i+1, line.strip()))
        
        if class_lines:
            print("Found class definitions:")
            for line_num, line in class_lines:
                print(f"  Line {line_num}: {line}")
        else:
            print("❌ No class definitions found!")
        
        # Look for obvious syntax errors
        print("\nChecking for common issues:")
        
        # Check for unmatched quotes or brackets
        open_brackets = content.count('(') - content.count(')')
        open_square = content.count('[') - content.count(']') 
        open_curly = content.count('{') - content.count('}')
        
        if open_brackets != 0:
            print(f"⚠️  Unmatched parentheses: {open_brackets}")
        if open_square != 0:
            print(f"⚠️  Unmatched square brackets: {open_square}")
        if open_curly != 0:
            print(f"⚠️  Unmatched curly brackets: {open_curly}")
        
        # Check for DriverState specifically
        if 'class DriverState' in content:
            print("✅ Found 'class DriverState' definition")
        else:
            print("❌ 'class DriverState' not found!")
            
        # Check for DailyAssignment  
        if 'class DailyAssignment' in content:
            print("✅ Found 'class DailyAssignment' definition")
        else:
            print("❌ 'class DailyAssignment' not found!")
            
        # Look for incomplete lines
        incomplete_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and (stripped.endswith(':') and not stripped.startswith('#')):
                # Check if next line exists and is indented
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if not next_line.strip() or not next_line.startswith('    '):
                        incomplete_lines.append((i+1, line.strip()))
                else:
                    incomplete_lines.append((i+1, line.strip()))
        
        if incomplete_lines:
            print("\n⚠️  Potentially incomplete definitions:")
            for line_num, line in incomplete_lines:
                print(f"  Line {line_num}: {line}")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")

def syntax_check():
    """Try to compile the file to check for syntax errors."""
    print("\n" + "=" * 60)
    print("SYNTAX CHECK")
    print("=" * 60)
    
    file_path = Path("src/models/driver_state.py")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Try to compile
        compile(content, file_path, 'exec')
        print("✅ File compiles successfully - no syntax errors")
        
    except SyntaxError as e:
        print(f"❌ Syntax Error found:")
        print(f"  Line {e.lineno}: {e.text}")
        print(f"  Error: {e.msg}")
        print(f"  Position: {' ' * (e.offset-1)}^")
        
    except Exception as e:
        print(f"❌ Other error: {e}")

if __name__ == "__main__":
    check_driver_state_file()
    syntax_check()