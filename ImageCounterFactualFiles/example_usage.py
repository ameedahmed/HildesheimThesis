#!/usr/bin/env python3
"""
Example usage script for the updated Vizfiletest.py global attention analysis tool.

This script demonstrates how to use the attention analysis functionality for:
1. Single image analysis (global attention only)
2. Batch analysis of multiple images
3. Different model types (artstyle, century, portrait)
4. Various CLI options for customization
"""

import os
import subprocess
import sys

def run_single_image_analysis():
    """Example: Analyze a single image with global attention only"""
    print("=== Single Image Analysis Example ===")

    # Basic command for single image analysis
    cmd = [
        "python", "Vizfiletest.py",
        "--model_type", "artstyle",
        "--image_path", "016.jpg",
        "--verbose"
    ]

    print(f"Running command: {' '.join(cmd)}")
    print("This will:")
    print("- Load the artstyle model")
    print("- Analyze the image '016.jpg'")
    print("- Generate global attention maps only")
    print("- Save results to default ImageCounterfactualExplanations directory")
    print("- Enable verbose output")
    print()

    # Advanced command with custom options
    cmd_advanced = [
        "python", "Vizfiletest.py",
        "--model_type", "artstyle",
        "--image_path", "016.jpg",
        "--output", "./custom_output",
        "--attention_threshold", "0.8",
        "--dpi", "300",
        "--device", "cuda",
        "--verbose",
        "--csv_format", "summary"
    ]

    print(f"Advanced command: {' '.join(cmd_advanced)}")
    print("This will:")
    print("- Use custom output directory './custom_output'")
    print("- Set attention threshold to 0.8 (higher sensitivity)")
    print("- Save images at 300 DPI for high quality")
    print("- Force CUDA device usage")
    print("- Save summary CSV format")
    print()

    # Uncomment the line below to actually run the basic command
    # subprocess.run(cmd)

def run_batch_analysis():
    """Example: Analyze multiple images in a directory"""
    print("=== Batch Analysis Example ===")

    # Basic batch analysis command
    cmd = [
        "python", "Vizfiletest.py",
        "--model_type", "portrait",
        "--data_dir", "/path/to/your/test/images",
        "--verbose"
    ]

    print(f"Basic batch command: {' '.join(cmd)}")
    print("This will:")
    print("- Load the portrait model")
    print("- Process all images in the specified directory")
    print("- Save results to default ImageCounterfactualExplanations directory")
    print("- Organize output by class and classification correctness")
    print("- Generate summary CSV with all results")
    print()

    # Advanced batch analysis with custom options
    cmd_advanced = [
        "python", "Vizfiletest.py",
        "--model_type", "century",
        "--data_dir", "/path/to/your/test/images",
        "--output", "./century_analysis_results",
        "--attention_threshold", "0.6",
        "--dpi", "200",
        "--device", "auto",
        "--verbose",
        "--csv_format", "detailed"
    ]

    print(f"Advanced batch command: {' '.join(cmd_advanced)}")
    print("This will:")
    print("- Use century model for analysis")
    print("- Set lower attention threshold (0.6) for more regions")
    print("- Save images at 200 DPI (faster processing)")
    print("- Auto-detect best device (GPU/CPU)")
    print("- Save detailed CSV format")
    print()

    # Uncomment the line below to actually run the basic command
    # subprocess.run(cmd)

def show_available_models():
    """Show available model configurations"""
    print("=== Available Model Types ===")
    print("1. artstyle:")
    print("   - Classes: 6 art style categories")
    print("   - Image size: 384x384")
    print("   - Model: CSWin_96_24322_base_384")
    print("   - Stage 4: 2 blocks with global attention")
    print()

    print("2. century:")
    print("   - Classes: 3 century categories")
    print("   - Image size: 384x384")
    print("   - Model: CSWin_96_24322_base_384")
    print("   - Stage 4: 2 blocks with global attention")
    print()

    print("3. portrait:")
    print("   - Classes: 9 artist categories")
    print("   - Image size: 384x384")
    print("   - Model: CSWin_96_24322_base_384")
    print("   - Stage 4: 2 blocks with global attention")
    print()

def show_cli_options():
    """Show all available CLI options"""
    print("=== Available CLI Options ===")
    print()
    
    print("Required Arguments:")
    print("  --model_type: Model type (artstyle, century, portrait)")
    print("  --image_path OR --data_dir: Input data (mutually exclusive)")
    print()
    
    print("Optional Arguments:")
    print("  --output: Custom output directory")
    print("  --attention_threshold: Threshold for high attention (default: 0.7)")
    print("  --dpi: Image DPI (default: 150)")
    print("  --device: Device selection (auto, cpu, cuda)")
    print("  --verbose: Enable verbose output")
    print("  --save_raw_only: Save only raw attention maps")
    print("  --csv_format: CSV format (detailed, summary)")
    print()

def show_output_structure():
    """Show the output directory structure"""
    print("=== Output Directory Structure ===")
    print("The tool creates the following structure:")
    print()
    print("ImageCounterfactualExplanations/")
    print("├── correct_classification_exp&newimg/")
    print("│   ├── class_name_1/")
    print("│   │   ├── AttentionAnalysis_image1.png")
    print("│   │   ├── RawAttentionMap_image1.png")
    print("│   │   ├── HighAttentionCoordinates_image1.csv")
    print("│   │   └── AllAttentionCoordinates_image1.png")
    print("│   └── class_name_2/")
    print("│       └── ...")
    print("├── wrong_classification_exp&newimg/")
    print("│   ├── class_name_1/")
    print("│   │   └── ...")
    print("│   └── class_name_2/")
    print("│       └── ...")
    print("└── AnalysisSummary_*.csv (batch processing)")
    print()
    print("Files generated for each image:")
    print("- AttentionAnalysis_*.png: Subplot with original + global attention map")
    print("- RawAttentionMap_*.png: Raw global attention map with colorbar")
    print("- HighAttentionCoordinates_*.csv: Coordinates of high attention regions")
    print("- AllAttentionCoordinates_*.csv: All attention coordinates")
    print()
    print("Note: Only global attention maps are generated (no vertical/horizontal)")

def show_technical_details():
    """Show technical implementation details"""
    print("=== Technical Implementation Details ===")
    print()
    print("Attention Processing:")
    print("- Focuses only on global attention (stage 4)")
    print("- Handles 2 blocks in stage 4 for CSWin base 384")
    print("- Enhanced gradient scaling for small gradient values")
    print("- Automatic gradient magnitude detection and scaling")
    print()
    print("Model Architecture:")
    print("- CSWin_96_24322_base_384: 96x96 patches, 24322 attention heads")
    print("- Stage 4 contains 2 transformer blocks")
    print("- Global attention only (no vertical/horizontal attention)")
    print("- Automatic device detection (GPU/CPU)")
    print()
    print("Image Processing:")
    print("- Input size: 384x384 pixels")
    print("- Normalization: ImageNet mean/std values")
    print("- High-quality output with configurable DPI")
    print("- Automatic patch grid size calculation")

def main():
    """Main function to demonstrate usage"""
    print("CSWin Transformer Global Attention Analysis Tool")
    print("=" * 60)
    print()

    show_available_models()
    print()

    show_cli_options()
    print()

    show_output_structure()
    print()

    show_technical_details()
    print()

    run_single_image_analysis()
    print()

    run_batch_analysis()
    print()

    print("=== Usage Notes ===")
    print("- Make sure you have the required model checkpoints in the specified paths")
    print("- The tool automatically detects GPU/CPU and uses the best available device")
    print("- For batch processing, images should be organized in class-specific folders")
    print("- Attention threshold can be adjusted based on sensitivity requirements")
    print("- Only global attention maps are generated (stage 4)")
    print("- Enhanced gradient scaling handles very small gradient values")
    print()

    print("=== Example Commands ===")
    print("# Single image analysis (basic)")
    print("python Vizfiletest.py --model_type artstyle --image_path path/to/image.jpg")
    print()
    print("# Single image analysis (advanced)")
    print("python Vizfiletest.py --model_type portrait --image_path image.jpg --output ./results --attention_threshold 0.8 --verbose")
    print()
    print("# Batch analysis (basic)")
    print("python Vizfiletest.py --model_type century --data_dir path/to/test/folder")
    print()
    print("# Batch analysis (advanced)")
    print("python Vizfiletest.py --model_type artstyle --data_dir /path/to/images --output ./custom_output --dpi 300 --csv_format summary --verbose")

if __name__ == "__main__":
    main()
