#!/usr/bin/env python3
"""
Test script for the generator to verify it works correctly.
"""

import os
import sys
import subprocess

def test_generator():
    """Test the generator script with both profiles."""
    
    print("🧪 Testing Minion/MinionS Function Generator")
    print("=" * 50)
    
    # Check if generator script exists
    if not os.path.exists("generator_script.py"):
        print("❌ Error: generator_script.py not found")
        return False
    
    # Check if config file exists
    if not os.path.exists("generation_config.json"):
        print("❌ Error: generation_config.json not found")
        return False
    
    # Check if partials directory exists
    if not os.path.exists("partials"):
        print("❌ Error: partials directory not found")
        return False
    
    # Test listing profiles
    print("\n📋 Testing profile listing...")
    try:
        result = subprocess.run([
            sys.executable, "generator_script.py", "--list-profiles"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Profile listing works:")
            print(result.stdout)
        else:
            print(f"❌ Profile listing failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Profile listing error: {e}")
        return False
    
    # Test generating minion function
    print("\n🎯 Testing Minion function generation...")
    try:
        result = subprocess.run([
            sys.executable, "generator_script.py", "minion_default",
            "--output_dir", "test_output"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Minion function generated successfully")
            print(result.stdout)
            
            # Check if file was created
            output_file = "test_output/minion_default_function.py"
            if os.path.exists(output_file):
                print(f"✅ Output file created: {output_file}")
                
                # Check file size
                size = os.path.getsize(output_file)
                print(f"📊 File size: {size:,} bytes")
                
                # Quick syntax check
                try:
                    with open(output_file, 'r') as f:
                        content = f.read()
                    
                    # Compile to check for syntax errors
                    compile(content, output_file, 'exec')
                    print("✅ Generated file has valid Python syntax")
                    
                    # Check for key components
                    checks = [
                        ("class MinionPipe", "Pipe class definition"),
                        ("def pipe(", "Pipe method"),
                        ("LocalAssistantResponse", "Response model"),
                        ("MinionValves", "Valves configuration"),
                        ("execute_minion_protocol", "Protocol logic"),
                        ("call_claude", "API calls"),
                    ]
                    
                    missing = []
                    for check, desc in checks:
                        if check not in content:
                            missing.append(desc)
                    
                    if missing:
                        print(f"⚠️  Missing components: {', '.join(missing)}")
                    else:
                        print("✅ All expected components found")
                        
                except Exception as e:
                    print(f"❌ Syntax error in generated file: {e}")
                    return False
            else:
                print(f"❌ Output file not created: {output_file}")
                return False
        else:
            print(f"❌ Minion generation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Minion generation error: {e}")
        return False
    
    # Test generating minions function
    print("\n🎯 Testing MinionS function generation...")
    try:
        result = subprocess.run([
            sys.executable, "generator_script.py", "minions_default",
            "--output_dir", "test_output"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ MinionS function generated successfully")
            print(result.stdout)
            
            # Check if file was created
            output_file = "test_output/minions_default_function.py"
            if os.path.exists(output_file):
                print(f"✅ Output file created: {output_file}")
                
                # Check file size
                size = os.path.getsize(output_file)
                print(f"📊 File size: {size:,} bytes")
                
                # Quick syntax check
                try:
                    with open(output_file, 'r') as f:
                        content = f.read()
                    
                    # Compile to check for syntax errors
                    compile(content, output_file, 'exec')
                    print("✅ Generated file has valid Python syntax")
                    
                    # Check for key components
                    checks = [
                        ("class MinionsPipe", "Pipe class definition"),
                        ("def pipe(", "Pipe method"),
                        ("TaskResult", "Task result model"),
                        ("MinionsValves", "Valves configuration"),
                        ("execute_minions_protocol", "Protocol logic"),
                        ("parse_minions_tasks", "Task parsing"),
                        ("create_minions_chunks", "Chunk creation"),
                    ]
                    
                    missing = []
                    for check, desc in checks:
                        if check not in content:
                            missing.append(desc)
                    
                    if missing:
                        print(f"⚠️  Missing components: {', '.join(missing)}")
                    else:
                        print("✅ All expected components found")
                        
                except Exception as e:
                    print(f"❌ Syntax error in generated file: {e}")
                    return False
            else:
                print(f"❌ Output file not created: {output_file}")
                return False
        else:
            print(f"❌ MinionS generation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ MinionS generation error: {e}")
        return False
    
    print("\n🎉 All tests passed!")
    print("\n📁 Generated files are in the test_output/ directory")
    print("📋 You can now copy these files to your Open WebUI functions directory")
    
    return True

if __name__ == "__main__":
    success = test_generator()
    sys.exit(0 if success else 1)