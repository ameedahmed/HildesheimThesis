#!/usr/bin/env python3
"""
Test script to verify multiprocessing is working
"""
import multiprocessing as mp
import time
import os
import torch

def test_worker(args):
    """Simple test worker function"""
    worker_id, device_id = args
    time.sleep(1)  # Simulate some work
    return {
        'worker_id': worker_id,
        'process_id': os.getpid(),
        'device_id': device_id,
        'success': True
    }

def test_multiprocessing():
    """Test multiprocessing functionality"""
    print("Testing multiprocessing...")
    print(f"CPU count: {mp.cpu_count()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # Create test data
    num_workers = 4
    test_data = [(i, i % 2) for i in range(num_workers)]
    
    print(f"Starting {num_workers} workers...")
    start_time = time.time()
    
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(test_worker, test_data)
    
    end_time = time.time()
    
    print(f"Completed in {end_time - start_time:.2f} seconds")
    print("Results:")
    for result in results:
        print(f"  Worker {result['worker_id']}: PID {result['process_id']}, Device {result['device_id']}")
    
    print("Multiprocessing test completed successfully!")

if __name__ == "__main__":
    # Set multiprocessing start method
    try:
        import torch.multiprocessing as torch_mp
        torch_mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    test_multiprocessing() 