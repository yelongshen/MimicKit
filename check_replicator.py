#!/usr/bin/env python
"""Check if Isaac Sim Replicator is available and provide installation guidance."""

import sys
import os

print("=" * 70)
print("Isaac Sim Replicator Availability Check")
print("=" * 70)
print()

# Check 1: Isaac Sim packages
print("1. Checking for Isaac Sim packages...")
try:
    import isaacsim
    print(f"   ✓ isaacsim found at: {isaacsim.__file__}")
    isaac_sim_path = os.path.dirname(os.path.dirname(isaacsim.__file__))
    print(f"   ✓ Isaac Sim root: {isaac_sim_path}")
except ImportError as e:
    print(f"   ✗ isaacsim not found: {e}")
    isaac_sim_path = None

print()

# Check 2: Omni modules
print("2. Checking for omni modules...")
try:
    import omni
    print(f"   ✓ omni package found")
    omni_path = omni.__file__ if hasattr(omni, '__file__') else "namespace package"
    print(f"     Path: {omni_path}")
except ImportError:
    print(f"   ✗ omni package not found")

print()

# Check 3: Replicator
print("3. Checking for omni.replicator...")
try:
    import omni.replicator.core as rep
    print(f"   ✓ omni.replicator.core found!")
    print(f"     This installation SUPPORTS video recording")
    sys.exit(0)
except ImportError as e:
    print(f"   ✗ omni.replicator.core not found: {e}")

print()

# Check 4: Look for replicator files
print("4. Searching for replicator extension files...")
if isaac_sim_path:
    exts_path = os.path.join(isaac_sim_path, "exts")
    extscache_path = os.path.join(isaac_sim_path, "extscache")
    
    for search_path in [exts_path, extscache_path]:
        if os.path.exists(search_path):
            print(f"   Searching in: {search_path}")
            for root, dirs, files in os.walk(search_path):
                for d in dirs:
                    if "replicator" in d.lower():
                        found_path = os.path.join(root, d)
                        print(f"   → Found: {found_path}")
                        
print()
print("=" * 70)
print("CONCLUSION:")
print("=" * 70)
print()
print("Replicator is NOT available in this installation.")
print()
print("SOLUTION OPTIONS:")
print()
print("Option 1: Use Isaac Sim GUI to install extensions")
print("  - Launch Isaac Sim with GUI")
print("  - Window → Extensions")
print("  - Search for 'omni.replicator.core'")
print("  - Enable/Install the extension")
print()
print("Option 2: Use Isaac Lab without video recording")
print("  - The simulation will work fine without Replicator")
print("  - Video recording is an optional feature")
print("  - You can use screen recording tools instead")
print()
print("Option 3: Manual screen recording")
print("  - Run with --visualize true")
print("  - Use OBS Studio, SimpleScreenRecorder, or similar")
print()

sys.exit(1)
