# -*- coding: utf-8 -*-
"""
Performance comparison between original SEDC and PyTorch-optimized version
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sedc import sedc
from sedc_pytorch import sedc_pytorch, sedc_pytorch_batch


def create_dummy_model(input_size=(3, 224, 224), num_classes=1000):
    """Create a dummy classifier for testing"""
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(64, num_classes)
    )
    return model


def create_dummy_data(batch_size=1, channels=3, height=224, width=224):
    """Create dummy image and segmentation data"""
    # Create dummy image
    image = torch.randn(batch_size, channels, height, width)
    
    # Create dummy segmentation (random segments)
    segments = np.random.randint(0, 10, (height, width))
    
    return image, segments


def benchmark_sedc_implementations(image, classifier, segments, mode='mean', device='cpu', num_runs=5):
    """
    Benchmark different SEDC implementations
    
    Args:
        image: Input image tensor
        classifier: PyTorch model
        segments: Segmentation map
        mode: Perturbation mode
        device: Target device
        num_runs: Number of runs for averaging
    
    Returns:
        Dictionary with timing results
    """
    results = {}
    
    # Test original SEDC
    print("Testing original SEDC implementation...")
    times_original = []
    for i in range(num_runs):
        start_time = time.time()
        try:
            explanation, segments_in_explanation, perturbation, new_class, predicted_class_index = sedc(
                image, classifier, segments, mode, device
            )
            end_time = time.time()
            times_original.append(end_time - start_time)
        except Exception as e:
            print(f"Original SEDC failed on run {i}: {e}")
            times_original.append(float('inf'))
    
    results['original'] = {
        'mean_time': np.mean(times_original),
        'std_time': np.std(times_original),
        'min_time': np.min(times_original),
        'max_time': np.max(times_original)
    }
    
    # Test PyTorch-optimized SEDC
    print("Testing PyTorch-optimized SEDC implementation...")
    times_pytorch = []
    for i in range(num_runs):
        start_time = time.time()
        try:
            explanation, segments_in_explanation, perturbation, new_class, predicted_class_index = sedc_pytorch(
                image, classifier, segments, mode, device
            )
            end_time = time.time()
            times_pytorch.append(end_time - start_time)
        except Exception as e:
            print(f"PyTorch SEDC failed on run {i}: {e}")
            times_pytorch.append(float('inf'))
    
    results['pytorch'] = {
        'mean_time': np.mean(times_pytorch),
        'std_time': np.std(times_pytorch),
        'min_time': np.min(times_pytorch),
        'max_time': np.max(times_pytorch)
    }
    
    # Test batch-optimized SEDC
    print("Testing batch-optimized SEDC implementation...")
    times_batch = []
    for i in range(num_runs):
        start_time = time.time()
        try:
            explanation, segments_in_explanation, perturbation, new_class, predicted_class_index = sedc_pytorch_batch(
                image, classifier, segments, mode, device, batch_size=4
            )
            end_time = time.time()
            times_batch.append(end_time - start_time)
        except Exception as e:
            print(f"Batch SEDC failed on run {i}: {e}")
            times_batch.append(float('inf'))
    
    results['batch'] = {
        'mean_time': np.mean(times_batch),
        'std_time': np.std(times_batch),
        'min_time': np.min(times_batch),
        'max_time': np.max(times_batch)
    }
    
    return results


def plot_performance_comparison(results):
    """Plot performance comparison results"""
    implementations = list(results.keys())
    mean_times = [results[impl]['mean_time'] for impl in implementations]
    std_times = [results[impl]['std_time'] for impl in implementations]
    
    # Calculate speedup
    original_time = results['original']['mean_time']
    speedups = [original_time / results[impl]['mean_time'] for impl in implementations]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Timing comparison
    bars1 = ax1.bar(implementations, mean_times, yerr=std_times, capsize=5, alpha=0.7)
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_yscale('log')
    
    # Add value labels on bars
    for bar, time_val in zip(bars1, mean_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.3f}s', ha='center', va='bottom')
    
    # Speedup comparison
    bars2 = ax2.bar(implementations, speedups, alpha=0.7, color=['red', 'green', 'blue'])
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Speedup vs Original Implementation')
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar, speedup in zip(bars2, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('sedc_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def print_performance_summary(results):
    """Print a summary of performance results"""
    print("\n" + "="*60)
    print("SEDC PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    
    original_time = results['original']['mean_time']
    
    for impl, metrics in results.items():
        speedup = original_time / metrics['mean_time']
        print(f"\n{impl.upper()} IMPLEMENTATION:")
        print(f"  Mean time: {metrics['mean_time']:.4f} ± {metrics['std_time']:.4f} seconds")
        print(f"  Min time:  {metrics['min_time']:.4f} seconds")
        print(f"  Max time:  {metrics['max_time']:.4f} seconds")
        print(f"  Speedup:   {speedup:.2f}x")
    
    print("\n" + "="*60)


def main():
    """Main benchmarking function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy data
    print("Creating dummy data...")
    image, segments = create_dummy_data(batch_size=1, channels=3, height=224, width=224)
    classifier = create_dummy_model()
    
    # Move to device
    image = image.to(device)
    classifier = classifier.to(device)
    
    # Test different modes
    modes = ['mean', 'blur', 'random']
    
    for mode in modes:
        print(f"\n{'='*20} TESTING MODE: {mode.upper()} {'='*20}")
        
        # Run benchmarks
        results = benchmark_sedc_implementations(
            image, classifier, segments, mode=mode, device=device, num_runs=3
        )
        
        # Print summary
        print_performance_summary(results)
        
        # Plot results
        plot_performance_comparison(results)
        
        print(f"Performance comparison plot saved as 'sedc_performance_comparison_{mode}.png'")


if __name__ == "__main__":
    main() 