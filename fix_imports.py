#!/usr/bin/env python3
"""
Helper script to fix import path issues for CSWin Transformer models.
Run this script before running your notebooks to set up the correct paths.
"""

import sys
import os

# Get the current directory (where this script is located)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to Python path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add the parent directory to Python path (in case you need to import from parent)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(f"Added to Python path:")
print(f"  - Current directory: {current_dir}")
print(f"  - Parent directory: {parent_dir}")
print(f"Current Python path: {sys.path[:3]}...")  # Show first 3 entries

# Test the import
try:
    from models.cswinmodified import CSWin_96_24322_base_384
    print("✓ Successfully imported CSWin_96_24322_base_384")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("Available modules in models directory:")
    models_dir = os.path.join(current_dir, 'models')
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.py'):
                print(f"  - {file}") 