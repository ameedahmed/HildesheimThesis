#!/usr/bin/env python3
"""
Script to calculate overlap between attention heatmaps and evidence counterfactuals.
Filters attention values >= 2.0 and only considers correct classifications.
"""

import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

class AttentionCounterfactualOverlap:
    def __init__(self, base_path: str):
        """
        Initialize the overlap calculator.
        
        Args:
            base_path: Base path to the ImageCounterfactualExplanations directory
        """
        self.base_path = Path(base_path)
        self.counterfactual_path = self.base_path / "ParallelDivision" / "PortraitResultsForCF" / "correct_classification_exp&newimg"
        self.heatmap_path = self.base_path / "HeatMapAnalysis" / "ArtistHeatmaps"
        self.coordinates_path = self.base_path / "HeatmapOriginal" / "HeatmapPortrait"
        
        # Class name mapping between counterfactual and heatmap directories
        self.class_mapping = {
            "Aubry, Peter II": "Aubry__PeterII",
            "Bock, Christoph Wilhelm": "Bock__ChristophWilhelm", 
            "Kilian, Lucas": "Kilian__Lucas",
            "Romstet, Christian": "Romstet__Christian",
            "Kilian, Wolfgang Philipp": "Kilian__WolfgangPhilipp",
            "Stimmer, Tobias": "Stimmer__Tobias",
            "Bernigeroth, Martin": "Bernigeroth__Martin",
            "Fennitzer, Georg": "Fennitzer__Georg",
            "Graff, Anton": "Graff__Anton"
        }
        
    def load_attention_coordinates(self, class_name: str, min_attention: float = 2.0) -> pd.DataFrame:
        """
        Load attention coordinates for a specific class, filtered by attention value and correct classification.
        
        Args:
            class_name: Name of the class
            min_attention: Minimum attention value threshold
            
        Returns:
            DataFrame with filtered attention coordinates
        """
        # Find the CSV file for this class
        # Convert class name to match CSV file naming pattern
        csv_name = class_name.replace(', ', '__').replace(' ', '_')
        csv_pattern = f"*{csv_name}*.csv"
        csv_files = list(self.coordinates_path.glob(f"**/{csv_pattern}"))
        
        if not csv_files:
            print(f"Warning: No CSV file found for class {class_name}")
            return pd.DataFrame()
        
        # Prefer AllAttentionCoordinates over HighAttentionCoordinates
        csv_file = None
        for file in csv_files:
            if "AllAttentionCoordinates" in str(file):
                csv_file = file
                break
        
        if csv_file is None:
            csv_file = csv_files[0]  # Fallback to first file found
        print(f"Loading coordinates from: {csv_file}")
        
        # Load and filter the data
        df = pd.read_csv(csv_file)
        
        # Filter by attention value >= min_attention and correct classification
        filtered_df = df[
            (df['attention_value'] >= min_attention) & 
            (df['true_class'] == df['predicted_class']) &
            (df['is_correct'] == True)
        ].copy()
        
        print(f"Loaded {len(filtered_df)} attention points with value >= {min_attention}")
        return filtered_df
    
    def load_counterfactual_image(self, class_name: str, image_name: str) -> Optional[np.ndarray]:
        """
        Load a counterfactual image for a specific class and image.
        
        Args:
            class_name: Name of the class
            image_name: Name of the image
            
        Returns:
            Loaded image as numpy array or None if not found
        """
        class_dir = self.counterfactual_path / class_name
        if not class_dir.exists():
            print(f"Warning: Counterfactual directory not found: {class_dir}")
            return None
        
        # Look for the counterfactual image - specifically ReasonForOriginalClassification files
        cf_pattern = f"ReasonForOriginalClassification_{image_name}*.png"
        cf_files = list(class_dir.glob(cf_pattern))
        
        # If no ReasonForOriginalClassification file found, abort
        if not cf_files:
            print(f"Warning: No counterfactual image found for {image_name} in {class_name}")
            return None
        
        cf_file = cf_files[0]
        print(f"Loading counterfactual: {cf_file}")
        
        # Load image
        img = cv2.imread(str(cf_file))
        if img is None:
            print(f"Error: Could not load image {cf_file}")
            return None
        
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def load_heatmap_image(self, class_name: str, image_name: str) -> Optional[np.ndarray]:
        """
        Load a heatmap image for a specific class and image.
        
        Args:
            class_name: Name of the class
            image_name: Name of the image
            
        Returns:
            Loaded heatmap as numpy array or None if not found
        """
        mapped_class = self.class_mapping.get(class_name)
        if not mapped_class:
            print(f"Warning: No mapping found for class {class_name}")
            return None
        
        class_dir = self.heatmap_path / mapped_class
        if not class_dir.exists():
            print(f"Warning: Heatmap directory not found: {class_dir}")
            return None
        
        # Look for the heatmap image
        hm_pattern = f"*{image_name}*OverlayHeatmap.png"
        hm_files = list(class_dir.glob(hm_pattern))
        
        if not hm_files:
            print(f"Warning: No heatmap image found for {image_name} in {mapped_class}")
            return None
        
        hm_file = hm_files[0]
        print(f"Loading heatmap: {hm_file}")
        
        # Load image
        img = cv2.imread(str(hm_file))
        if img is None:
            print(f"Error: Could not load image {hm_file}")
            return None
        
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def load_original_image(self, class_name: str, image_name: str) -> Optional[np.ndarray]:
        """
        Load the original input image for visualization purposes.
        
        Args:
            class_name: Name of the class
            image_name: Name of the image
            
        Returns:
            Loaded original image as numpy array or None if not found
        """
        # Try to find original image in various locations
        possible_paths = [
            self.base_path / "img" / f"{image_name}.jpg",
            self.base_path / "img" / f"{image_name}.png",
            self.base_path / "img" / f"{image_name}.jpeg",
            self.base_path / "chihuahua_test" / f"{image_name}.jpg",
            self.base_path / "chihuahua_test" / f"{image_name}.png"
        ]
        
        for img_path in possible_paths:
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # If no original image found, return None
        return None
    
    def create_attention_mask(self, coordinates: pd.DataFrame, image_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Create a binary mask from attention coordinates.
        
        Args:
            coordinates: DataFrame with x, y coordinates
            image_shape: Shape of the target image (height, width, channels)
            
        Returns:
            Binary mask where attention points are marked
        """
        mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
        
        for _, row in coordinates.iterrows():
            x, y = int(row['x']), int(row['y'])
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                mask[y, x] = 1
        
        return mask
    
    def create_counterfactual_mask(self, counterfactual_img: np.ndarray, threshold: int = 128) -> np.ndarray:
        """
        Create a binary mask from counterfactual image by thresholding.
        
        Args:
            counterfactual_img: Counterfactual image as numpy array
            threshold: Threshold value for binarization
            
        Returns:
            Binary mask where counterfactual evidence is marked
        """
        # Convert to grayscale
        if len(counterfactual_img.shape) == 3:
            gray = cv2.cvtColor(counterfactual_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = counterfactual_img
        
        # Apply threshold to create binary mask
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        mask = (mask > 0).astype(np.uint8)
        
        return mask
    
    def calculate_overlap(self, attention_mask: np.ndarray, counterfactual_mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate overlap metrics between attention and counterfactual masks.
        
        Args:
            attention_mask: Binary attention mask
            counterfactual_mask: Binary counterfactual mask
            
        Returns:
            Dictionary with overlap metrics
        """
        # Ensure masks have the same shape
        if attention_mask.shape != counterfactual_mask.shape:
            # Resize attention mask to match counterfactual mask
            attention_mask = cv2.resize(attention_mask, (counterfactual_mask.shape[1], counterfactual_mask.shape[0]))
        
        # Calculate intersection and union
        intersection = np.logical_and(attention_mask, counterfactual_mask).sum()
        union = np.logical_or(attention_mask, counterfactual_mask).sum()
        
        # Calculate overlap metrics
        attention_pixels = attention_mask.sum()
        counterfactual_pixels = counterfactual_mask.sum()
        
        if attention_pixels == 0:
            attention_coverage = 0.0
        else:
            attention_coverage = intersection / attention_pixels
        
        if counterfactual_pixels == 0:
            counterfactual_coverage = 0.0
        else:
            counterfactual_coverage = intersection / counterfactual_pixels
        
        if union == 0:
            jaccard_similarity = 0.0
        else:
            jaccard_similarity = intersection / union
        
        return {
            'intersection_pixels': int(intersection),
            'attention_pixels': int(attention_pixels),
            'counterfactual_pixels': int(counterfactual_pixels),
            'attention_coverage': float(attention_coverage),
            'counterfactual_coverage': float(counterfactual_coverage),
            'jaccard_similarity': float(jaccard_similarity)
        }
    
    def create_visualization(self, 
                           original_image: np.ndarray,
                           attention_coordinates: pd.DataFrame,
                           attention_mask: np.ndarray,
                           counterfactual_mask: np.ndarray,
                           overlap_metrics: Dict[str, float],
                           image_name: str,
                           class_name: str,
                           save_path: Optional[str] = None) -> None:
        """
        Create comprehensive visualization of the overlap analysis.
        
        Args:
            original_image: Original input image
            attention_coordinates: DataFrame with attention coordinates
            attention_mask: Binary attention mask
            counterfactual_mask: Binary counterfactual mask
            overlap_metrics: Calculated overlap metrics
            image_name: Name of the image
            class_name: Name of the class
            save_path: Optional path to save visualization
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Attention-Counterfactual Overlap Analysis\n{class_name} - {image_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Original image with attention points overlaid
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image + Attention Points', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Plot attention points with color based on attention value
        if not attention_coordinates.empty:
            attention_values = attention_coordinates['attention_value'].values
            x_coords = attention_coordinates['x'].values
            y_coords = attention_coordinates['y'].values
            
            # Normalize attention values for color mapping
            norm_attention = (attention_values - attention_values.min()) / (attention_values.max() - attention_values.min() + 1e-8)
            
            # Create scatter plot with attention values as colors
            scatter = axes[0, 0].scatter(x_coords, y_coords, c=norm_attention, 
                                        cmap='hot', alpha=0.7, s=20, edgecolors='white', linewidth=0.5)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[0, 0], shrink=0.8)
            cbar.set_label('Attention Value (Normalized)', rotation=270, labelpad=15)
        
        # 2. Attention mask
        axes[0, 1].imshow(attention_mask, cmap='Reds', alpha=0.8)
        axes[0, 1].set_title(f'Attention Mask\n({overlap_metrics["attention_pixels"]} pixels)', fontweight='bold')
        axes[0, 1].axis('off')
        
        # 3. Counterfactual mask
        axes[0, 2].imshow(counterfactual_mask, cmap='Blues', alpha=0.8)
        axes[0, 2].set_title(f'Counterfactual Evidence\n({overlap_metrics["counterfactual_pixels"]} pixels)', fontweight='bold')
        axes[0, 2].axis('off')
        
        # 4. Overlap visualization
        overlap_vis = np.zeros_like(attention_mask, dtype=np.uint8)
        overlap_vis[attention_mask == 1] = 1  # Red for attention
        overlap_vis[counterfactual_mask == 1] = 2  # Blue for counterfactual
        overlap_vis[np.logical_and(attention_mask, counterfactual_mask)] = 3  # Purple for overlap
        
        # Create custom colormap for overlap
        colors = ['white', 'red', 'blue', 'purple']
        cmap = LinearSegmentedColormap.from_list('overlap', colors, N=4)
        
        axes[1, 0].imshow(overlap_vis, cmap=cmap, alpha=0.8)
        axes[1, 0].set_title(f'Overlap Analysis\n(Intersection: {overlap_metrics["intersection_pixels"]} pixels)', fontweight='bold')
        axes[1, 0].axis('off')
        
        # Add legend for overlap
        legend_elements = [
            patches.Patch(color='red', label='Attention Only'),
            patches.Patch(color='blue', label='Counterfactual Only'),
            patches.Patch(color='purple', label='Overlap')
        ]
        axes[1, 0].legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # 5. Metrics summary
        axes[1, 1].axis('off')
        metrics_text = f"""OVERLAP METRICS

Jaccard Similarity: {overlap_metrics['jaccard_similarity']:.4f}
Attention Coverage: {overlap_metrics['attention_coverage']:.4f}
Counterfactual Coverage: {overlap_metrics['counterfactual_coverage']:.4f}

PIXEL COUNTS
Attention Pixels: {overlap_metrics['attention_pixels']:,}
Counterfactual Pixels: {overlap_metrics['counterfactual_pixels']:,}
Intersection: {overlap_metrics['intersection_pixels']:,}

COORDINATES
Total Attention Points: {len(attention_coordinates):,}
Attention Threshold: ≥ 0.2"""
        
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # 6. Attention value distribution
        if not attention_coordinates.empty:
            attention_values = attention_coordinates['attention_value'].values
            axes[1, 2].hist(attention_values, bins=30, color='red', alpha=0.7, edgecolor='black')
            axes[1, 2].axvline(x=0.2, color='green', linestyle='--', linewidth=2, label='Threshold (0.2)')
            axes[1, 2].set_title('Attention Value Distribution', fontweight='bold')
            axes[1, 2].set_xlabel('Attention Value')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def create_summary_visualization(self, all_results: Dict[str, Dict], averages: Dict[str, float], 
                                   save_path: Optional[str] = None) -> None:
        """
        Create summary visualization across all classes and images.
        
        Args:
            all_results: Results from process_all_classes
            averages: Average metrics
            save_path: Optional path to save visualization
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Attention-Counterfactual Overlap Analysis Summary', fontsize=16, fontweight='bold')
        
        # 1. Jaccard similarity by class
        class_names = list(all_results.keys())
        class_jaccards = []
        
        for class_name in class_names:
            class_results = all_results[class_name]
            class_metrics = [result['overlap_metrics']['jaccard_similarity'] 
                           for result in class_results.values()]
            class_jaccards.append(np.mean(class_metrics))
        
        # Sort by Jaccard similarity
        sorted_indices = np.argsort(class_jaccards)[::-1]
        sorted_classes = [class_names[i] for i in sorted_indices]
        sorted_jaccards = [class_jaccards[i] for i in sorted_indices]
        
        bars = axes[0, 0].bar(range(len(sorted_classes)), sorted_jaccards, 
                              color='skyblue', edgecolor='navy', alpha=0.8)
        axes[0, 0].set_title('Average Jaccard Similarity by Class', fontweight='bold')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Jaccard Similarity')
        axes[0, 0].set_xticks(range(len(sorted_classes)))
        axes[0, 0].set_xticklabels(sorted_classes, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, sorted_jaccards)):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Coverage comparison
        coverage_data = {
            'Attention Coverage': averages['average_attention_coverage'],
            'Counterfactual Coverage': averages['average_counterfactual_coverage']
        }
        
        colors = ['red', 'blue']
        bars = axes[0, 1].bar(coverage_data.keys(), coverage_data.values(), 
                              color=colors, alpha=0.8, edgecolor='black')
        axes[0, 1].set_title('Average Coverage Metrics', fontweight='bold')
        axes[0, 1].set_ylabel('Coverage Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, coverage_data.values()):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Jaccard distribution across all images
        all_jaccards = []
        for class_results in all_results.values():
            for image_results in class_results.values():
                all_jaccards.append(image_results['overlap_metrics']['jaccard_similarity'])
        
        axes[1, 0].hist(all_jaccards, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=np.mean(all_jaccards), color='red', linestyle='--', 
                           linewidth=2, label=f'Mean: {np.mean(all_jaccards):.4f}')
        axes[1, 0].set_title('Jaccard Similarity Distribution (All Images)', fontweight='bold')
        axes[1, 0].set_xlabel('Jaccard Similarity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""GLOBAL SUMMARY

Total Classes: {len(all_results)}
Total Images: {averages['total_images_processed']}

AVERAGE METRICS
Jaccard Similarity: {averages['average_jaccard_similarity']:.4f} ± {averages['std_jaccard_similarity']:.4f}
Attention Coverage: {averages['average_attention_coverage']:.4f} ± {averages['std_attention_coverage']:.4f}
Counterfactual Coverage: {averages['average_counterfactual_coverage']:.4f} ± {averages['std_counterfactual_coverage']:.4f}

TOP PERFORMING CLASSES
1. {sorted_classes[0]}: {sorted_jaccards[0]:.4f}
2. {sorted_classes[1]}: {sorted_jaccards[1]:.4f}
3. {sorted_classes[2]}: {sorted_jaccards[2]:.4f}

BOTTOM PERFORMING CLASSES
1. {sorted_classes[-1]}: {sorted_jaccards[-1]:.4f}
2. {sorted_classes[-2]}: {sorted_jaccards[-2]:.4f}
3. {sorted_classes[-3]}: {sorted_jaccards[-3]:.4f}"""
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary visualization saved to: {save_path}")
        
        plt.show()
    
    def process_class(self, class_name: str, min_attention: float = 2.0, 
                     create_visualizations: bool = False, viz_save_dir: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Process all images for a specific class.
        
        Args:
            class_name: Name of the class to process
            min_attention: Minimum attention value threshold
            create_visualizations: Whether to create visualizations for each image
            viz_save_dir: Directory to save visualizations (if None, only display)
            
        Returns:
            Dictionary with results for each image
        """
        print(f"\nProcessing class: {class_name}")
        
        # Load attention coordinates
        coordinates_df = self.load_attention_coordinates(class_name, min_attention)
        if coordinates_df.empty:
            return {}
        
        # Group by image
        image_groups = coordinates_df.groupby('image_name')
        results = {}
        
        for image_name, group_coords in image_groups:
            print(f"  Processing image: {image_name}")
            
            # Load images
            counterfactual_img = self.load_counterfactual_image(class_name, image_name)
            heatmap_img = self.load_heatmap_image(class_name, image_name)
            
            if counterfactual_img is None or heatmap_img is None:
                print(f"    Skipping {image_name} - could not load images")
                continue
            
            # Create masks
            attention_mask = self.create_attention_mask(group_coords, counterfactual_img.shape)
            counterfactual_mask = self.create_counterfactual_mask(counterfactual_img)
            
            # Calculate overlap
            overlap_metrics = self.calculate_overlap(attention_mask, counterfactual_mask)
            
            # Store results
            results[image_name] = {
                'coordinates_count': len(group_coords),
                'overlap_metrics': overlap_metrics
            }
            
            print(f"    Attention pixels: {overlap_metrics['attention_pixels']}")
            print(f"    Counterfactual pixels: {overlap_metrics['counterfactual_pixels']}")
            print(f"    Intersection: {overlap_metrics['intersection_pixels']}")
            print(f"    Jaccard similarity: {overlap_metrics['jaccard_similarity']:.4f}")
            
            # Create visualization if requested
            if create_visualizations:
                # Try to load original image for visualization
                original_img = self.load_original_image(class_name, image_name)
                if original_img is not None:
                    viz_save_path = None
                    if viz_save_dir:
                        os.makedirs(viz_save_dir, exist_ok=True)
                        viz_save_path = os.path.join(viz_save_dir, f"{class_name.replace(', ', '_')}_{image_name}_overlap_analysis.png")
                    
                    self.create_visualization(
                        original_img, group_coords, attention_mask, counterfactual_mask,
                        overlap_metrics, image_name, class_name, viz_save_path
                    )
        
        return results
    
    def process_all_classes(self, min_attention: float = 2.0, 
                          create_visualizations: bool = False, 
                          viz_save_dir: Optional[str] = None) -> Dict[str, Dict]:
        """
        Process all available classes.
        
        Args:
            min_attention: Minimum attention value threshold
            create_visualizations: Whether to create visualizations for each image
            viz_save_dir: Directory to save visualizations (if None, only display)
            
        Returns:
            Dictionary with results for all classes
        """
        all_results = {}
        
        for class_name in self.class_mapping.keys():
            try:
                class_results = self.process_class(class_name, min_attention, 
                                                create_visualizations, viz_save_dir)
                if class_results:
                    all_results[class_name] = class_results
            except Exception as e:
                print(f"Error processing class {class_name}: {e}")
                continue
        
        return all_results
    
    def calculate_averages(self, all_results: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calculate average overlap metrics across all classes and images.
        
        Args:
            all_results: Results from process_all_classes
            
        Returns:
            Dictionary with average metrics
        """
        total_attention_coverage = []
        total_counterfactual_coverage = []
        total_jaccard_similarity = []
        total_images = 0
        
        for class_name, class_results in all_results.items():
            for image_name, image_results in class_results.items():
                metrics = image_results['overlap_metrics']
                total_attention_coverage.append(metrics['attention_coverage'])
                total_counterfactual_coverage.append(metrics['counterfactual_coverage'])
                total_jaccard_similarity.append(metrics['jaccard_similarity'])
                total_images += 1
        
        if total_images == 0:
            return {}
        
        return {
            'average_attention_coverage': np.mean(total_attention_coverage),
            'average_counterfactual_coverage': np.mean(total_counterfactual_coverage),
            'average_jaccard_similarity': np.mean(total_jaccard_similarity),
            'total_images_processed': total_images,
            'std_attention_coverage': np.std(total_attention_coverage),
            'std_counterfactual_coverage': np.std(total_counterfactual_coverage),
            'std_jaccard_similarity': np.std(total_jaccard_similarity)
        }
    
    def save_results(self, all_results: Dict[str, Dict], averages: Dict[str, float], output_file: str):
        """
        Save results to a JSON file.
        
        Args:
            all_results: Results from process_all_classes
            averages: Average metrics
            output_file: Output file path
        """
        output_data = {
            'averages': averages,
            'detailed_results': all_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    def print_summary(self, all_results: Dict[str, Dict], averages: Dict[str, float]):
        """
        Print a summary of the results.
        
        Args:
            all_results: Results from process_all_classes
            averages: Average metrics
        """
        print("\n" + "="*60)
        print("OVERLAP ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nTotal classes processed: {len(all_results)}")
        print(f"Total images processed: {averages.get('total_images_processed', 0)}")
        
        print(f"\nAverage Metrics:")
        print(f"  Attention Coverage: {averages.get('average_attention_coverage', 0):.4f} ± {averages.get('std_attention_coverage', 0):.4f}")
        print(f"  Counterfactual Coverage: {averages.get('average_counterfactual_coverage', 0):.4f} ± {averages.get('std_counterfactual_coverage', 0):.4f}")
        print(f"  Jaccard Similarity: {averages.get('average_jaccard_similarity', 0):.4f} ± {averages.get('std_jaccard_similarity', 0):.4f}")
        
        print(f"\nPer-Class Summary:")
        for class_name, class_results in all_results.items():
            class_metrics = []
            for image_results in class_results.values():
                class_metrics.append(image_results['overlap_metrics']['jaccard_similarity'])
            
            if class_metrics:
                avg_jaccard = np.mean(class_metrics)
                print(f"  {class_name}: {avg_jaccard:.4f} ({len(class_metrics)} images)")


def main():
    """Main function to run the overlap analysis."""
    # Initialize the overlap calculator
    base_path = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations"
    calculator = AttentionCounterfactualOverlap(base_path)
    
    # Set minimum attention threshold
    min_attention = 0.2
    
    # Visualization options
    create_visualizations = True  # Set to True to create visualizations
    viz_save_dir = os.path.join(base_path, "overlap_visualizations")  # Directory to save visualizations
    
    print(f"Starting overlap analysis with attention threshold >= {min_attention}")
    if create_visualizations:
        print(f"Creating visualizations and saving to: {viz_save_dir}")
    
    # Process all classes
    all_results = calculator.process_all_classes(min_attention, create_visualizations, viz_save_dir)
    
    if not all_results:
        print("No results obtained. Check file paths and data availability.")
        return
    
    # Calculate averages
    averages = calculator.calculate_averages(all_results)
    
    # Print summary
    calculator.print_summary(all_results, averages)
    
    # Create summary visualization
    summary_viz_path = os.path.join(base_path, "overlap_analysis_summary.png")
    calculator.create_summary_visualization(all_results, averages, summary_viz_path)
    
    # Save results
    output_file = os.path.join(base_path, "attention_counterfactual_overlap_results.json")
    calculator.save_results(all_results, averages, output_file)
    
    print(f"\nAnalysis complete! Results saved to: {output_file}")
    print(f"Summary visualization saved to: {summary_viz_path}")


if __name__ == "__main__":
    main()
