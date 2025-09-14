#!/usr/bin/env python3
"""
Test script to demonstrate visualization capabilities of the overlap analysis.
"""

import sys
from pathlib import Path
import os

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from calculate_attention_counterfactual_overlap import AttentionCounterfactualOverlap

def test_visualization():
    """Test the visualization capabilities with a single class."""
    base_path = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations"
    calculator = AttentionCounterfactualOverlap(base_path)
    
    # Test with Aubry, Peter II class
    test_class = "Aubry, Peter II"
    min_attention = 0.2
    
    print(f"Testing visualization for class: {test_class}")
    print(f"Attention threshold: >= {min_attention}")
    
    try:
        # Process single class with visualizations
        results = calculator.process_class(test_class, min_attention, 
                                        create_visualizations=True, 
                                        viz_save_dir=os.path.join(base_path, "test_visualizations"))
        
        if results:
            print(f"\nSuccessfully processed {len(results)} images for {test_class}")
            
            # Show first few results
            for i, (image_name, image_results) in enumerate(results.items()):
                if i >= 3:  # Only show first 3 for testing
                    break
                metrics = image_results['overlap_metrics']
                print(f"  {image_name}: Jaccard = {metrics['jaccard_similarity']:.4f}")
            
            print(f"\nVisualizations created and saved to: {os.path.join(base_path, 'test_visualizations')}")
            
        else:
            print(f"No results obtained for {test_class}")
            
    except Exception as e:
        print(f"Error testing {test_class}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualization()
