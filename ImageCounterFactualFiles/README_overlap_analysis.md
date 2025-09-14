# Attention-Counterfactual Overlap Analysis

This script calculates the overlap between attention heatmaps and evidence counterfactuals for portrait classification models.

## Overview

The script analyzes how well the attention mechanisms in your model align with the evidence identified by counterfactual explanations. It:

1. **Filters attention data** by attention values ≥ 0.2 and correct classifications
2. **Creates binary masks** from attention coordinates and counterfactual images
3. **Calculates overlap metrics** including Jaccard similarity, coverage ratios
4. **Processes all classes** and calculates overall averages
5. **Creates comprehensive visualizations** for each step of the analysis
6. **Saves detailed results** to JSON format and high-quality visualizations

## Files

- `calculate_attention_counterfactual_overlap.py` - Main analysis script with visualizations
- `test_overlap_single_class.py` - Test script for single class verification
- `test_visualization.py` - Test script for visualization capabilities
- `overlap_requirements.txt` - Required Python packages
- `README_overlap_analysis.md` - This documentation

## Requirements

Install the required packages:

```bash
pip install -r overlap_requirements.txt
```

## Usage

### 1. Test with Single Class

First, test the functionality with a single class:

```bash
cd ImageCounterfactualExplanations
python test_overlap_single_class.py
```

This will test the overlap calculation for "Aubry, Peter II" class and show you the results.

### 2. Test Visualizations

Test the visualization capabilities with a single class:

```bash
python test_visualization.py
```

This will create detailed visualizations for the first few images in "Aubry, Peter II" class.

### 3. Run Full Analysis with Visualizations

Run the complete analysis for all classes with comprehensive visualizations:

```bash
python calculate_attention_counterfactual_overlap.py
```

This will process all classes and create visualizations for every image, plus a global summary.

## Output

The script generates:

1. **Console output** showing progress and summary statistics
2. **JSON results file** (`attention_counterfactual_overlap_results.json`) with:
   - Overall averages across all classes
   - Detailed results for each class and image
   - Overlap metrics (Jaccard similarity, coverage ratios)
3. **Comprehensive visualizations** for each image showing:
   - Original image with attention points overlaid
   - Attention mask visualization
   - Counterfactual evidence mask
   - Overlap analysis with color coding
   - Metrics summary and attention value distribution
4. **Global summary visualization** showing:
   - Jaccard similarity by class
   - Coverage metrics comparison
   - Overall distribution analysis
   - Performance rankings

## Metrics Explained

- **Attention Coverage**: Percentage of attention pixels that overlap with counterfactual evidence
- **Counterfactual Coverage**: Percentage of counterfactual evidence pixels that overlap with attention
- **Jaccard Similarity**: Intersection over union (IoU) - measures overall overlap quality
- **Intersection Pixels**: Number of pixels where both attention and counterfactual evidence exist

## Visualization Features

### Per-Image Analysis (6-panel visualization)
1. **Original Image + Attention Points**: Shows the input image with attention coordinates overlaid, colored by attention value
2. **Attention Mask**: Binary mask showing where the model pays attention
3. **Counterfactual Evidence**: Binary mask showing the evidence identified by counterfactual analysis
4. **Overlap Analysis**: Color-coded visualization showing:
   - Red: Attention only
   - Blue: Counterfactual evidence only  
   - Purple: Overlapping regions
5. **Metrics Summary**: Detailed statistics and pixel counts
6. **Attention Distribution**: Histogram of attention values with threshold line

### Global Summary (4-panel visualization)
1. **Class Performance**: Bar chart ranking classes by average Jaccard similarity
2. **Coverage Comparison**: Side-by-side comparison of attention vs. counterfactual coverage
3. **Distribution Analysis**: Histogram of Jaccard similarity across all images
4. **Summary Statistics**: Overall performance metrics and rankings

## File Structure Expected

The script expects the following directory structure:

```
ImageCounterfactualExplanations/
├── ParallelDivision/PortraitResultsForCF/correct_classification_exp&newimg/
│   ├── Aubry, Peter II/
│   ├── Bock, Christoph Wilhelm/
│   └── ... (other classes)
├── HeatMapAnalysis/ArtistHeatmaps/
│   ├── Aubry__PeterII/
│   ├── Bock__ChristophWilhelm/
│   └── ... (other classes)
└── HeatmapOriginal/HeatmapPortrait/Class_*_Coordinates/
    ├── AllAttentionCoordinates_Aubry__Peter_II.csv
    └── ... (other CSV files)
```

## Customization

You can modify the script to:

- Change the attention threshold (currently ≥ 2.0)
- Adjust the counterfactual binarization threshold
- Add additional overlap metrics
- Modify the output format

## Troubleshooting

1. **No results obtained**: Check file paths and ensure CSV files exist
2. **Image loading errors**: Verify image files exist and are readable
3. **Memory issues**: Process classes one at a time for large datasets
4. **Coordinate mismatches**: Ensure attention coordinates match image dimensions

## Example Output

```
OVERLAP ANALYSIS SUMMARY
============================================================

Total classes processed: 9
Total images processed: 45

Average Metrics:
  Attention Coverage: 0.2345 ± 0.1234
  Counterfactual Coverage: 0.3456 ± 0.2345
  Jaccard Similarity: 0.1567 ± 0.0987

Per-Class Summary:
  Aubry, Peter II: 0.1234 (5 images)
  Bock, Christoph Wilhelm: 0.2345 (6 images)
  ...
```
