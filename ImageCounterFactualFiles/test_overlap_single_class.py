#!/usr/bin/env python3
"""
Test script to verify overlap calculation with a single class.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from calculate_attention_counterfactual_overlap import AttentionCounterfactualOverlap

def test_single_class():
    """Test the overlap calculation with a single class."""
    base_path = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations"
    calculator = AttentionCounterfactualOverlap(base_path)
    
    # Test with Aubry, Peter II class
    test_class = "Aubry, Peter II"
    min_attention = 0.2  # Threshold for attention values
    
    print(f"Testing overlap calculation for class: {test_class}")
    print(f"Attention threshold: >= {min_attention}")
    
    try:
        # Process single class
        results = calculator.process_class(test_class, min_attention)
        
        if results:
            print(f"\nSuccessfully processed {len(results)} images for {test_class}")
            
            # Calculate class-specific averages
            class_metrics = []
            for image_name, image_results in results.items():
                metrics = image_results['overlap_metrics']
                class_metrics.append(metrics['jaccard_similarity'])
                print(f"  {image_name}: Jaccard = {metrics['jaccard_similarity']:.4f}")
            
            avg_jaccard = sum(class_metrics) / len(class_metrics)
            print(f"\nAverage Jaccard similarity for {test_class}: {avg_jaccard:.4f}")
            
        else:
            print(f"No results obtained for {test_class}")
            
    except Exception as e:
        print(f"Error testing {test_class}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_class()
