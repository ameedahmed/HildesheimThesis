#!/usr/bin/env python3
import pickle
import os

print("Testing pickle file creation...")

# Test data
test_data = {"test": "value", "numbers": [1, 2, 3]}

# Try to save
test_file = "test.pkl"
try:
    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"✓ Successfully created {test_file}")
    
    # Check if file exists
    if os.path.exists(test_file):
        print(f"✓ File exists: {os.path.getsize(test_file)} bytes")
        
        # Try to load
        with open(test_file, 'rb') as f:
            loaded_data = pickle.load(f)
        print(f"✓ Successfully loaded data: {loaded_data}")
        
        # Cleanup
        os.remove(test_file)
        print("✓ Cleaned up test file")
    else:
        print("✗ File was not created")
        
except Exception as e:
    print(f"✗ Error: {e}")

print("Test completed.")
