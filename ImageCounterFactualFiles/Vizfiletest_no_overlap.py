#!/usr/bin/env python3
"""
Modified Vizfiletest.py that only processes files from no_overlap_files.csv
"""

import sys
import os
import subprocess

# Add the base path to sys.path
base_path = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main"
sys.path.append(base_path)

# Import the original script's functions
from ImageCounterfactualExplanations.Vizfiletest import (
    ModelConfig, save_attention_analysis, signal_handler, cleanup_on_exit
)

# List of files to process (from no_overlap_files.csv)
FILES_TO_PROCESS = []

def main():
    print("Starting analysis of no_overlap_files only...")
    print(f"Total files to process: {len(FILES_TO_PROCESS)}")
    
    # Initialize configuration for portrait model
    config = ModelConfig("portrait")
    
    # Load model (this will be done by save_attention_analysis)
    # We'll use the same model loading logic as the original script
    
    # Process each file
    for i, file_path in enumerate(FILES_TO_PROCESS):
        print(f"Processing {i+1}/{len(FILES_TO_PROCESS)}: {os.path.basename(file_path)}")
        
        try:
            # Create a simple args object with required attributes
            class SimpleArgs:
                def __init__(self):
                    self.output = None
                    self.skip_existing = True
                    self.force_reprocess = False
                    self.attention_threshold = 0.2
                    self.verbose = True
            
            args = SimpleArgs()
            
            # Process the file
            result = save_attention_analysis(file_path, None, config, None, args)
            
            if result:
                print(f"  ✓ Completed: {result['image_name']}")
                print(f"    True: {result['true_class']}, Predicted: {result['predicted_class']}")
                print(f"    Correct: {result['is_correct']}, Confidence: {result['confidence']:.3f}")
            else:
                print(f"  ✗ Failed to process: {os.path.basename(file_path)}")
                
        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(file_path)}: {str(e)}")
            continue
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
