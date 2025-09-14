# -*- coding: utf-8 -*-
"""
Example usage of PyTorch-optimized SEDC implementation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sedc_pytorch import sedc_pytorch, sedc_pytorch_batch


def create_simple_classifier():
    """Create a simple CNN classifier for demonstration"""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10)
    )
    return model


def create_sample_data():
    """Create sample image and segmentation data"""
    # Create a simple image with distinct regions
    image = torch.zeros(1, 3, 64, 64)
    
    # Add some patterns to make it interesting
    image[0, 0, :32, :32] = 1.0  # Red top-left
    image[0, 1, :32, 32:] = 1.0  # Green top-right
    image[0, 2, 32:, :32] = 1.0  # Blue bottom-left
    image[0, :, 32:, 32:] = 0.5  # Mixed bottom-right
    
    # Create segmentation map
    segments = np.zeros((64, 64), dtype=int)
    segments[:32, :32] = 0  # Top-left segment
    segments[:32, 32:] = 1  # Top-right segment
    segments[32:, :32] = 2  # Bottom-left segment
    segments[32:, 32:] = 3  # Bottom-right segment
    
    return image, segments


def visualize_results(image, segments, explanation, perturbation, mode):
    """Visualize the SEDC results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    img_np = image.detach().cpu().numpy()[0].transpose(1, 2, 0)
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Segmentation
    axes[0, 1].imshow(segments, cmap='tab10')
    axes[0, 1].set_title('Segmentation Map')
    axes[0, 1].axis('off')
    
    # Explanation
    axes[0, 2].imshow(explanation)
    axes[0, 2].set_title('Explanation Mask')
    axes[0, 2].axis('off')
    
    # Perturbed image
    pert_np = perturbation.detach().cpu().numpy()[0].transpose(1, 2, 0)
    axes[1, 0].imshow(pert_np)
    axes[1, 0].set_title(f'Perturbed Image ({mode})')
    axes[1, 0].axis('off')
    
    # Difference
    diff = np.abs(img_np - pert_np)
    axes[1, 1].imshow(diff)
    axes[1, 1].set_title('Difference')
    axes[1, 1].axis('off')
    
    # Overlay
    overlay = img_np.copy()
    overlay[explanation.sum(axis=2) > 0] = [1, 0, 0]  # Highlight explanation in red
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Explanation Overlay')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'sedc_example_{mode}.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main example function"""
    print("PyTorch-Optimized SEDC Example")
    print("=" * 40)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data
    print("Creating sample data...")
    image, segments = create_sample_data()
    classifier = create_simple_classifier()
    
    # Move to device
    image = image.to(device)
    classifier = classifier.to(device)
    
    # Test different modes
    modes = ['mean', 'blur', 'random']
    
    for mode in modes:
        print(f"\nTesting mode: {mode}")
        print("-" * 20)
        
        try:
            # Run SEDC
            explanation, segments_in_explanation, perturbation, new_class, predicted_class_index = sedc_pytorch(
                image, classifier, segments, mode=mode, device=device
            )
            
            print(f"Original predicted class: {predicted_class_index}")
            print(f"New predicted class: {new_class}")
            print(f"Segments in explanation: {segments_in_explanation}")
            
            # Visualize results
            visualize_results(image, segments, explanation, perturbation, mode)
            
        except Exception as e:
            print(f"Error with mode {mode}: {e}")
    
    # Test batch processing
    print(f"\nTesting batch processing with mode: mean")
    print("-" * 40)
    
    try:
        explanation, segments_in_explanation, perturbation, new_class, predicted_class_index = sedc_pytorch_batch(
            image, classifier, segments, mode='mean', device=device, batch_size=4
        )
        
        print(f"Batch processing completed successfully!")
        print(f"Original predicted class: {predicted_class_index}")
        print(f"New predicted class: {new_class}")
        print(f"Segments in explanation: {segments_in_explanation}")
        
    except Exception as e:
        print(f"Error with batch processing: {e}")


if __name__ == "__main__":
    main() 