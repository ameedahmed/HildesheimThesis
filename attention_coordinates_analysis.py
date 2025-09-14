#!/usr/bin/env python3
"""
Comprehensive Analysis of Attention Map Coordinates
==================================================

This script provides multiple approaches to analyze attention map coordinates:
1. DBSCAN Clustering for spatial grouping
2. Attention pattern analysis
3. Spatial distribution analysis
4. Statistical analysis
5. Visualization techniques
6. Image-specific analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class AttentionCoordinatesAnalyzer:
    def __init__(self, csv_file_path):
        """
        Initialize the analyzer with the CSV file path.
        
        Args:
            csv_file_path (str): Path to the CSV file containing attention coordinates
        """
        self.csv_file_path = csv_file_path
        self.df = None
        self.images = None
        self.attention_types = None
        
    def load_data(self, sample_size=None):
        """
        Load the attention coordinates data.
        
        Args:
            sample_size (int, optional): Number of rows to sample for faster processing
        """
        print("Loading data...")
        if sample_size:
            # Sample data for faster processing
            self.df = pd.read_csv(self.csv_file_path, nrows=sample_size)
            print(f"Loaded {len(self.df)} rows (sampled from original)")
        else:
            self.df = pd.read_csv(self.csv_file_path)
            print(f"Loaded {len(self.df)} rows")
        
        # Filter to only include rows where predicted class matches true class
        initial_count = len(self.df)
        self.df = self.df[self.df['predicted_class'] == self.df['true_class']]
        filtered_count = len(self.df)
        
        print(f"Filtered to correct predictions: {filtered_count} rows (removed {initial_count - filtered_count} incorrect predictions)")
        
        # Basic data info
        self.classes = self.df['true_class'].unique()
        self.attention_types = self.df['attention_type'].unique()
        
        print(f"Number of unique classes: {len(self.classes)}")
        print(f"Classes: {list(self.classes)}")
        print(f"Attention types: {self.attention_types}")
        print(f"Coordinate range: X({self.df['x'].min()}-{self.df['x'].max()}), Y({self.df['y'].min()}-{self.df['y'].max()})")
        
        return self.df
    
    def basic_statistics(self):
        """Calculate basic statistics for the attention coordinates."""
        print("\n=== BASIC STATISTICS ===")
        
        # Attention value statistics
        attention_stats = self.df['attention_value'].describe()
        print("\nAttention Value Statistics:")
        print(attention_stats)
        
        # Coordinate statistics
        print(f"\nX-coordinate range: {self.df['x'].min()} to {self.df['x'].max()}")
        print(f"Y-coordinate range: {self.df['y'].min()} to {self.df['y'].max()}")
        
        # Class-level statistics
        class_stats = self.df.groupby('true_class').agg({
            'attention_value': ['mean', 'std', 'min', 'max', 'count'],
            'x': ['min', 'max'],
            'y': ['min', 'max']
        }).round(4)
        
        print(f"\nClass-level statistics:")
        print(class_stats)
        
        return attention_stats, class_stats
    
    def dbscan_clustering(self, eps=20, min_samples=50, class_name=None):
        """
        Perform DBSCAN clustering on attention coordinates.
        
        Args:
            eps (float): The maximum distance between two samples for one to be considered 
                        as in the neighborhood of the other (in original coordinate units)
            min_samples (int): The number of samples in a neighborhood for a point to be 
                             considered as a core point
            class_name (str, optional): Specific class to analyze
        
        Returns:
            dict: Clustering results and statistics
        """
        print(f"\n=== DBSCAN CLUSTERING (eps={eps}, min_samples={min_samples}) ===")
        
        if class_name:
            data = self.df[self.df['true_class'] == class_name]
            print(f"Analyzing class: {class_name}")
        else:
            data = self.df
            print("Analyzing all classes")
        
        # Prepare coordinates for clustering - use original scales
        coordinates = data[['x', 'y']].values
        
        print(f"Coordinate range: X({coordinates[:, 0].min()}-{coordinates[:, 0].max()}), Y({coordinates[:, 1].min()}-{coordinates[:, 1].max()})")
        print(f"Total points for clustering: {len(coordinates)}")
        
        # For very large datasets, we might need to adjust parameters
        if len(coordinates) > 100000:
            # Scale min_samples based on dataset size
            adjusted_min_samples = max(min_samples, len(coordinates) // 10000)
            if adjusted_min_samples != min_samples:
                print(f"Adjusting min_samples from {min_samples} to {adjusted_min_samples} for large dataset")
                min_samples = adjusted_min_samples
        
        # Perform DBSCAN clustering on original coordinates
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(coordinates)
        
        # Add cluster labels to data
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels
        
        # Clustering statistics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        print(f"Percentage of points clustered: {((len(cluster_labels) - n_noise) / len(cluster_labels) * 100):.2f}%")
        
        # Cluster sizes
        if n_clusters > 0:
            cluster_sizes = [list(cluster_labels).count(i) for i in range(n_clusters)]
            print(f"Cluster sizes: {cluster_sizes}")
            print(f"Average cluster size: {np.mean(cluster_sizes):.2f}")
            print(f"Largest cluster size: {max(cluster_sizes)}")
            print(f"Smallest cluster size: {min(cluster_sizes)}")
        
        # Attention value statistics per cluster
        cluster_attention_stats = data_with_clusters.groupby('cluster')['attention_value'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        
        print(f"\nAttention value statistics per cluster:")
        print(cluster_attention_stats)
        
        return {
            'data_with_clusters': data_with_clusters,
            'cluster_labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_attention_stats': cluster_attention_stats,
            'dbscan_model': dbscan,
            'scaler': None  # No scaler used
        }
    
    def tune_dbscan_parameters(self, eps_range=[10, 20, 30, 50, 100], min_samples_range=[50, 100, 200, 500], class_name=None):
        """
        Test different DBSCAN parameters to find optimal clustering.
        
        Args:
            eps_range (list): List of epsilon values to test
            min_samples_range (list): List of min_samples values to test
            class_name (str, optional): Specific class to analyze
        """
        print(f"\n=== DBSCAN PARAMETER TUNING ===")
        
        if class_name:
            data = self.df[self.df['true_class'] == class_name]
            print(f"Analyzing class: {class_name}")
        else:
            data = self.df
            print("Analyzing all classes")
        
        coordinates = data[['x', 'y']].values
        print(f"Total points: {len(coordinates)}")
        
        results = []
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                print(f"\nTesting eps={eps}, min_samples={min_samples}")
                
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    cluster_labels = dbscan.fit_predict(coordinates)
                    
                    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    n_noise = list(cluster_labels).count(-1)
                    clustered_percentage = ((len(cluster_labels) - n_noise) / len(cluster_labels) * 100)
                    
                    # Calculate cluster size statistics
                    if n_clusters > 0:
                        cluster_sizes = [list(cluster_labels).count(i) for i in range(n_clusters)]
                        avg_cluster_size = np.mean(cluster_sizes)
                        max_cluster_size = max(cluster_sizes)
                        min_cluster_size = min(cluster_sizes)
                    else:
                        avg_cluster_size = max_cluster_size = min_cluster_size = 0
                    
                    result = {
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'clustered_percentage': clustered_percentage,
                        'avg_cluster_size': avg_cluster_size,
                        'max_cluster_size': max_cluster_size,
                        'min_cluster_size': min_cluster_size
                    }
                    
                    results.append(result)
                    
                    print(f"  Clusters: {n_clusters}, Noise: {n_noise}, Clustered: {clustered_percentage:.1f}%")
                    if n_clusters > 0:
                        print(f"  Cluster sizes - Avg: {avg_cluster_size:.1f}, Max: {max_cluster_size}, Min: {min_cluster_size}")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    continue
        
        # Sort results by number of clusters (descending) and then by clustered percentage
        results.sort(key=lambda x: (x['n_clusters'], x['clustered_percentage']), reverse=True)
        
        print(f"\n=== PARAMETER TUNING RESULTS (sorted by clusters, then by clustered %) ===")
        print(f"{'eps':<8} {'min_samples':<12} {'clusters':<10} {'noise':<8} {'clustered%':<12} {'avg_size':<10} {'max_size':<10} {'min_size':<10}")
        print("-" * 90)
        
        for result in results:
            print(f"{result['eps']:<8} {result['min_samples']:<12} {result['n_clusters']:<10} {result['n_noise']:<8} "
                  f"{result['clustered_percentage']:<12.1f} {result['avg_cluster_size']:<10.1f} "
                  f"{result['max_cluster_size']:<10} {result['min_cluster_size']:<10}")
        
        return results
    
    def attention_based_clustering(self, attention_threshold=0.8, eps=20, min_samples=10, class_name=None):
        """
        Perform clustering on high attention points only, which should give more meaningful clusters.
        
        Args:
            attention_threshold (float): Minimum attention value to include in clustering
            eps (float): DBSCAN epsilon parameter
            min_samples (int): DBSCAN min_samples parameter
            class_name (str, optional): Specific class to analyze
        
        Returns:
            dict: Clustering results and statistics
        """
        print(f"\n=== ATTENTION-BASED CLUSTERING (threshold={attention_threshold}, eps={eps}, min_samples={min_samples}) ===")
        
        if class_name:
            data = self.df[self.df['true_class'] == class_name]
            print(f"Analyzing class: {class_name}")
        else:
            data = self.df
            print("Analyzing all classes")
        
        # Filter for high attention points
        high_attention_data = data[data['attention_value'] >= attention_threshold]
        print(f"Total points: {len(data)}")
        print(f"High attention points (≥{attention_threshold}): {len(high_attention_data)}")
        print(f"Percentage of high attention points: {(len(high_attention_data) / len(data) * 100):.2f}%")
        
        if len(high_attention_data) < min_samples:
            print(f"Warning: Not enough high attention points ({len(high_attention_data)}) for clustering with min_samples={min_samples}")
            return None
        
        # Prepare coordinates for clustering
        coordinates = high_attention_data[['x', 'y']].values
        
        print(f"Coordinate range: X({coordinates[:, 0].min()}-{coordinates[:, 0].max()}), Y({coordinates[:, 1].min()}-{coordinates[:, 1].max()})")
        
        # Perform DBSCAN clustering on high attention points
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(coordinates)
        
        # Add cluster labels to data
        high_attention_data_with_clusters = high_attention_data.copy()
        high_attention_data_with_clusters['cluster'] = cluster_labels
        
        # Clustering statistics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        print(f"Percentage of high attention points clustered: {((len(cluster_labels) - n_noise) / len(cluster_labels) * 100):.2f}%")
        
        # Cluster sizes and locations
        if n_clusters > 0:
            cluster_sizes = [list(cluster_labels).count(i) for i in range(n_clusters)]
            print(f"Cluster sizes: {cluster_sizes}")
            print(f"Average cluster size: {np.mean(cluster_sizes):.2f}")
            print(f"Largest cluster size: {max(cluster_sizes)}")
            print(f"Smallest cluster size: {min(cluster_sizes)}")
            
            # Show cluster locations
            print("\nCluster locations:")
            for i in range(n_clusters):
                cluster_data = high_attention_data_with_clusters[high_attention_data_with_clusters['cluster'] == i]
                x_min, x_max = cluster_data['x'].min(), cluster_data['x'].max()
                y_min, y_max = cluster_data['y'].min(), cluster_data['y'].max()
                print(f"  Cluster {i}: X({x_min}-{x_max}), Y({y_min}-{y_max}), Size: {len(cluster_data)}")
        
        # Attention value statistics per cluster
        if n_clusters > 0:
            cluster_attention_stats = high_attention_data_with_clusters.groupby('cluster')['attention_value'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(4)
            
            print(f"\nAttention value statistics per cluster:")
            print(cluster_attention_stats)
        
        return {
            'high_attention_data': high_attention_data,
            'data_with_clusters': high_attention_data_with_clusters,
            'cluster_labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_attention_stats': cluster_attention_stats if n_clusters > 0 else None,
            'dbscan_model': dbscan,
            'attention_threshold': attention_threshold
        }
    
    def test_dbscan_small_sample(self, sample_size=1000, eps=30, min_samples=50):
        """
        Test DBSCAN clustering on a small sample to verify it works.
        
        Args:
            sample_size (int): Number of rows to sample
            eps (float): DBSCAN epsilon parameter
            min_samples (int): DBSCAN min_samples parameter
        """
        print(f"\n=== TESTING DBSCAN ON SMALL SAMPLE ({sample_size} rows) ===")
        
        # Load small sample
        self.load_data(sample_size)
        
        # Run DBSCAN
        results = self.dbscan_clustering(eps=eps, min_samples=min_samples)
        
        print(f"\nTest completed successfully!")
        print(f"Found {results['n_clusters']} clusters")
        print(f"Noise points: {results['n_noise']}")
        
        return results
    
    def spatial_analysis(self, class_name=None):
        """
        Perform spatial analysis of attention coordinates.
        
        Args:
            class_name (str, optional): Specific class to analyze
        """
        print(f"\n=== SPATIAL ANALYSIS ===")
        
        if class_name:
            data = self.df[self.df['true_class'] == class_name]
            print(f"Analyzing class: {class_name}")
        else:
            data = self.df
            print("Analyzing all classes")
        
        # Filter for high attention points only (>= 0.5) to avoid memory issues
        high_attention_data = data[data['attention_value'] >= 0.5]
        print(f"Total points: {len(data)}")
        print(f"High attention points (≥0.5): {len(high_attention_data)}")
        print(f"Percentage of high attention points: {(len(high_attention_data) / len(data) * 100):.2f}%")
        
        if len(high_attention_data) == 0:
            print("No high attention points found. Skipping spatial analysis.")
            return {
                'area': 0,
                'density': 0,
                'min_distance': None,
                'mean_distance': None,
                'max_distance': None,
                'x_std': 0,
                'y_std': 0
            }
        
        # Calculate distances between points using high attention data only
        coordinates = high_attention_data[['x', 'y']].values
        
        # Spatial density analysis
        x_range = high_attention_data['x'].max() - high_attention_data['x'].min()
        y_range = high_attention_data['y'].max() - high_attention_data['y'].min()
        area = x_range * y_range
        density = len(high_attention_data) / area if area > 0 else 0
        
        print(f"Spatial area: {area:.2f}")
        print(f"Point density: {density:.4f} points per unit area")
        
        # Nearest neighbor analysis (only for high attention points)
        if len(coordinates) > 1:
            # Safety check for large datasets
            if len(coordinates) > 10000:
                print(f"Large dataset detected ({len(coordinates)} points). Using sample for distance analysis.")
                # Sample 10000 points for distance analysis to avoid memory issues
                sample_indices = np.random.choice(len(coordinates), 10000, replace=False)
                sample_coordinates = coordinates[sample_indices]
                distances = pdist(sample_coordinates)
            else:
                distances = pdist(coordinates)
            
            min_distance = np.min(distances)
            mean_distance = np.mean(distances)
            max_distance = np.max(distances)
            
            print(f"Minimum distance between points: {min_distance:.4f}")
            print(f"Mean distance between points: {mean_distance:.4f}")
            print(f"Maximum distance between points: {max_distance:.4f}")
        else:
            min_distance = mean_distance = max_distance = None
        
        # Spatial distribution analysis
        x_std = high_attention_data['x'].std()
        y_std = high_attention_data['y'].std()
        print(f"X-coordinate standard deviation: {x_std:.4f}")
        print(f"Y-coordinate standard deviation: {y_std:.4f}")
        
        # Check for spatial patterns
        if x_std < y_std * 0.5:
            print("Horizontal clustering detected (points are more spread horizontally)")
        elif y_std < x_std * 0.5:
            print("Vertical clustering detected (points are more spread vertically)")
        else:
            print("Relatively uniform spatial distribution")
        
        return {
            'area': area,
            'density': density,
            'min_distance': min_distance,
            'mean_distance': mean_distance,
            'max_distance': max_distance,
            'x_std': x_std,
            'y_std': y_std
        }
    
    def attention_pattern_analysis(self, class_name=None):
        """
        Analyze attention patterns and their distribution.
        
        Args:
            class_name (str, optional): Specific class to analyze
        """
        print(f"\n=== ATTENTION PATTERN ANALYSIS ===")
        
        if class_name:
            data = self.df[self.df['true_class'] == class_name]
            print(f"Analyzing image: {class_name}")
        else:
            data = self.df
            print("Analyzing all classes")
        
        # Filter for high attention points only (>= 0.5) to focus on meaningful patterns
        high_attention_data = data[data['attention_value'] >= 0.5]
        print(f"Total points: {len(data)}")
        print(f"High attention points (≥0.5): {len(high_attention_data)}")
        print(f"Percentage of high attention points: {(len(high_attention_data) / len(data) * 100):.2f}%")
        
        if len(high_attention_data) == 0:
            print("No high attention points found. Skipping attention pattern analysis.")
            return {
                'attention_percentiles': {},
                'high_attention_threshold': 0.5,
                'high_attention_points': pd.DataFrame(),
                'attention_distribution': pd.Series(),
                'high_attention_center': None,
                'high_attention_spread': None
            }
        
        # Attention value distribution for high attention points
        attention_values = high_attention_data['attention_value']
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        attention_percentiles = np.percentile(attention_values, percentiles)
        
        print("Attention value percentiles (high attention points only):")
        for p, val in zip(percentiles, attention_percentiles):
            print(f"  {p}th percentile: {val:.6f}")
        
        # High attention regions (top 10% of high attention points)
        high_attention_threshold = np.percentile(attention_values, 90)
        very_high_attention_points = high_attention_data[high_attention_data['attention_value'] >= high_attention_threshold]
        
        print(f"\nVery high attention points (≥90th percentile of high attention): {len(very_high_attention_points)}")
        print(f"Very high attention threshold: {high_attention_threshold:.6f}")
        
        # Attention value clustering
        attention_bins = pd.cut(attention_values, bins=10)
        attention_distribution = attention_bins.value_counts().sort_index()
        
        print(f"\nAttention value distribution (10 bins, high attention points only):")
        print(attention_distribution)
        
        # Spatial attention patterns
        high_attention_coords = high_attention_data[['x', 'y']].values
        if len(high_attention_coords) > 0:
            high_attention_center = np.mean(high_attention_coords, axis=0)
            high_attention_spread = np.std(high_attention_coords, axis=0)
            
            print(f"\nHigh attention center: ({high_attention_center[0]:.2f}, {high_attention_center[1]:.2f})")
            print(f"High attention spread: ({high_attention_spread[0]:.2f}, {high_attention_spread[1]:.2f})")
        else:
            high_attention_center = high_attention_spread = None
        
        return {
            'attention_percentiles': dict(zip(percentiles, attention_percentiles)),
            'high_attention_threshold': high_attention_threshold,
            'high_attention_points': very_high_attention_points,
            'attention_distribution': attention_distribution,
            'high_attention_center': high_attention_center,
            'high_attention_spread': high_attention_spread
        }
    
    def class_comparison_analysis(self, class_names=None, max_classes=5):
        """
        Compare attention patterns across different classes.
        
        Args:
            class_names (list, optional): Specific classes to compare
            max_classes (int): Maximum number of classes to compare
        """
        print(f"\n=== CLASS COMPARISON ANALYSIS ===")
        
        if class_names is None:
            # Get top classes by number of high attention points
            high_attention_counts = {}
            for class_name in self.df['true_class'].unique():
                class_data = self.df[self.df['true_class'] == class_name]
                high_attention_count = len(class_data[class_data['attention_value'] >= 0.5])
                high_attention_counts[class_name] = high_attention_count
            
            # Sort by high attention count and take top classes
            sorted_classes = sorted(high_attention_counts.items(), key=lambda x: x[1], reverse=True)
            class_names = [class_name for class_name, count in sorted_classes[:max_classes]]
        
        print(f"Comparing classes: {class_names}")
        
        comparison_results = {}
        
        for class_name in class_names:
            if class_name not in self.df['true_class'].unique():
                print(f"Warning: Class '{class_name}' not found in dataset")
                continue
            
            class_data = self.df[self.df['true_class'] == class_name]
            high_attention_data = class_data[class_data['attention_value'] >= 0.5]
            
            if len(high_attention_data) == 0:
                print(f"Class '{class_name}': No high attention points (≥0.5)")
                continue
            
            print(f"\nClass: {class_name}")
            print(f"  Total points: {len(class_data)}")
            print(f"  High attention points (≥0.5): {len(high_attention_data)}")
            print(f"  Percentage high attention: {(len(high_attention_data) / len(class_data) * 100):.2f}%")
            
            # High attention statistics
            high_attention_values = high_attention_data['attention_value']
            print(f"  High attention - Mean: {high_attention_values.mean():.4f}, Std: {high_attention_values.std():.4f}")
            print(f"  High attention - Min: {high_attention_values.min():.4f}, Max: {high_attention_values.max():.4f}")
            
            # Spatial distribution of high attention points
            if len(high_attention_data) > 0:
                x_range = high_attention_data['x'].max() - high_attention_data['x'].min()
                y_range = high_attention_data['y'].max() - high_attention_data['y'].min()
                print(f"  High attention spatial range: X({x_range:.1f}), Y({y_range:.1f})")
            
            comparison_results[class_name] = {
                'total_points': len(class_data),
                'high_attention_points': len(high_attention_data),
                'high_attention_percentage': len(high_attention_data) / len(class_data) * 100,
                'high_attention_mean': high_attention_values.mean() if len(high_attention_values) > 0 else 0,
                'high_attention_std': high_attention_values.std() if len(high_attention_values) > 0 else 0
            }
        
        return comparison_results
    
    def create_visualizations(self, class_name=None, save_plots=True):
        """
        Create visualizations for the attention coordinates.
        
        Args:
            class_name (str, optional): Specific class to visualize
            save_plots (bool): Whether to save plots to files
        """
        print(f"\n=== CREATING VISUALIZATIONS ===")
        
        if class_name:
            data = self.df[self.df['true_class'] == class_name]
            title_suffix = f" - {class_name}"
            print(f"Visualizing class: {class_name}")
        else:
            data = self.df
            title_suffix = ""
            print("Visualizing all classes")
        
        # Filter for high attention points only (>= 0.5) to focus on meaningful patterns
        high_attention_data = data[data['attention_value'] >= 0.5]
        print(f"Total points: {len(data)}")
        print(f"High attention points (≥0.5): {len(high_attention_data)}")
        print(f"Percentage of high attention points: {(len(high_attention_data) / len(data) * 100):.2f}%")
        
        if len(high_attention_data) == 0:
            print("No high attention points found. Skipping visualizations.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: High attention points only
        scatter1 = axes[0].scatter(high_attention_data['x'], high_attention_data['y'], 
                                  c=high_attention_data['attention_value'], 
                                  cmap='viridis', alpha=0.7, s=20)
        axes[0].set_xlabel('X Coordinate')
        axes[0].set_ylabel('Y Coordinate')
        axes[0].set_title(f'High Attention Points (≥0.5){title_suffix}')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='Attention Value')
        
        # Plot 2: Attention value distribution for high attention points
        axes[1].hist(high_attention_data['attention_value'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1].set_xlabel('Attention Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Attention Value Distribution (High Attention Points){title_suffix}')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics text
        mean_attn = high_attention_data['attention_value'].mean()
        std_attn = high_attention_data['attention_value'].std()
        axes[1].text(0.02, 0.98, f'Mean: {mean_attn:.3f}\nStd: {std_attn:.3f}', 
                     transform=axes[1].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"attention_visualization{'_' + class_name if class_name else ''}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved visualization: {filename}")
        
        plt.show()
        
        # 2. Interactive Plotly visualization
        if len(high_attention_data) <= 10000:  # Plotly works better with smaller datasets
            # fig = px.scatter(data, x='x', y='y', color='attention_value',
            #                title=f'Interactive Attention Coordinates{title_suffix}',
            #                color_continuous_scale='viridis')
            # fig.show()
            print("Plotly visualization skipped due to missing dependencies.")
        
        print("Visualizations completed!")
    
    def create_dbscan_visualization(self, dbscan_results, class_name=None, save_plots=True):
        """
        Create visualization for DBSCAN clustering results.
        
        Args:
            dbscan_results (dict): Results from dbscan_clustering method
            class_name (str, optional): Class name for title
            save_plots (bool): Whether to save plots
        """
        print(f"\n=== CREATING DBSCAN VISUALIZATION ===")
        
        data_with_clusters = dbscan_results['data_with_clusters']
        n_clusters = dbscan_results['n_clusters']
        
        # Create color map for clusters
        colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters + 1, 2)))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Original data colored by attention value
        scatter1 = axes[0].scatter(data_with_clusters['x'], data_with_clusters['y'], 
                                  c=data_with_clusters['attention_value'], 
                                  cmap='viridis', alpha=0.6, s=1)
        axes[0].set_xlabel('X Coordinate')
        axes[0].set_ylabel('Y Coordinate')
        axes[0].set_title('Original Attention Coordinates')
        plt.colorbar(scatter1, ax=axes[0], label='Attention Value')
        
        # Plot 2: DBSCAN clustering results
        for cluster_id in range(-1, n_clusters):
            if cluster_id == -1:
                # Noise points
                mask = data_with_clusters['cluster'] == -1
                label = 'Noise'
                color = 'black'
                alpha = 0.3
            else:
                # Clustered points
                mask = data_with_clusters['cluster'] == cluster_id
                label = f'Cluster {cluster_id}'
                color = colors[cluster_id]
                alpha = 0.7
            
            cluster_data = data_with_clusters[mask]
            if len(cluster_data) > 0:
                axes[1].scatter(cluster_data['x'], cluster_data['y'], 
                               c=[color], alpha=alpha, s=1, label=label)
        
        axes[1].set_xlabel('X Coordinate')
        axes[1].set_ylabel('Y Coordinate')
        axes[1].set_title(f'DBSCAN Clustering Results ({n_clusters} clusters)')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"dbscan_clustering{'_' + class_name if class_name else ''}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved DBSCAN plot: {filename}")
        
        plt.show()
    
    def export_analysis_results(self, dbscan_results=None, output_file="attention_analysis_results.txt"):
        """
        Export analysis results to a text file.
        
        Args:
            dbscan_results (dict, optional): DBSCAN clustering results
            output_file (str): Output file name
        """
        print(f"\n=== EXPORTING ANALYSIS RESULTS ===")
        
        with open(output_file, 'w') as f:
            f.write("ATTENTION COORDINATES ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            f.write("BASIC STATISTICS:\n")
            f.write("-" * 20 + "\n")
            attention_stats, class_stats = self.basic_statistics()
            f.write(f"Total rows: {len(self.df)}\n")
            f.write(f"Number of classes: {len(self.df['true_class'].unique())}\n")
            f.write(f"Attention types: {', '.join(self.attention_types)}\n")
            f.write(f"Attention value range: {attention_stats['min']:.6f} to {attention_stats['max']:.6f}\n")
            f.write(f"Mean attention value: {attention_stats['mean']:.6f}\n")
            f.write(f"Standard deviation: {attention_stats['std']:.6f}\n\n")
            
            # Spatial analysis
            f.write("SPATIAL ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            spatial_results = self.spatial_analysis()
            for key, value in spatial_results.items():
                if value is not None:
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Attention pattern analysis
            f.write("ATTENTION PATTERN ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            pattern_results = self.attention_pattern_analysis()
            f.write(f"High attention threshold (90th percentile): {pattern_results['high_attention_threshold']:.6f}\n")
            f.write(f"Number of high attention points: {len(pattern_results['high_attention_points'])}\n")
            f.write("\n")
            
            # DBSCAN results if available
            if dbscan_results:
                f.write("DBSCAN CLUSTERING RESULTS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Number of clusters: {dbscan_results['n_clusters']}\n")
                f.write(f"Number of noise points: {dbscan_results['n_noise']}\n")
                f.write(f"Percentage clustered: {((len(dbscan_results['cluster_labels']) - dbscan_results['n_noise']) / len(dbscan_results['cluster_labels']) * 100):.2f}%\n")
                f.write("\nCluster attention statistics:\n")
                f.write(dbscan_results['cluster_attention_stats'].to_string())
                f.write("\n")
        
        print(f"Results exported to: {output_file}")
    
    def run_comprehensive_analysis(self, sample_size=None, dbscan_eps=30, dbscan_min_samples=100, tune_parameters=True):
        """
        Run a comprehensive analysis of the attention coordinates.
        
        Args:
            sample_size (int, optional): Number of rows to sample
            dbscan_eps (float): DBSCAN epsilon parameter
            dbscan_min_samples (int): DBSCAN min_samples parameter
            tune_parameters (bool): Whether to run parameter tuning before clustering
        """
        print("=" * 60)
        print("COMPREHENSIVE ATTENTION COORDINATES ANALYSIS")
        print("=" * 60)
        
        # Load data
        self.load_data(sample_size)
        
        # Basic statistics
        self.basic_statistics()
        
        # Spatial analysis
        self.spatial_analysis()
        
        # Attention pattern analysis
        self.attention_pattern_analysis()
        
        # Class comparison
        self.class_comparison_analysis()
        
        # Parameter tuning if requested
        if tune_parameters:
            print("\n" + "=" * 60)
            print("DBSCAN PARAMETER TUNING")
            print("=" * 60)
            tuning_results = self.tune_dbscan_parameters()
            
            # Use best parameters if tuning found better results
            if tuning_results:
                best_result = tuning_results[0]  # First result has most clusters
                if best_result['n_clusters'] > 1:  # Only use if we found multiple clusters
                    dbscan_eps = best_result['eps']
                    dbscan_min_samples = best_result['min_samples']
                    print(f"\nUsing tuned parameters: eps={dbscan_eps}, min_samples={dbscan_min_samples}")
        
        # DBSCAN clustering
        print("\n" + "=" * 60)
        print("FINAL DBSCAN CLUSTERING")
        print("=" * 60)
        dbscan_results = self.dbscan_clustering(eps=dbscan_eps, min_samples=dbscan_min_samples)
        
        # Create visualizations
        self.create_visualizations()
        self.create_dbscan_visualization(dbscan_results)
        
        # Export results
        self.export_analysis_results(dbscan_results)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)


def main():
    """Main function to run the analysis."""
    
    # File path
    csv_file = "ImageCounterfactualExplanations/HeatmapOriginal/HeatmapPortrait/Class_Aubry__Peter_II_Coordinates/AllAttentionCoordinates_Aubry__Peter_II.csv"
    
    # Initialize analyzer
    analyzer = AttentionCoordinatesAnalyzer(csv_file)
    
    # For testing, you can use a smaller sample first
    # Uncomment the line below to test with a smaller dataset
    # sample_size = 10000  # Test with 10K rows first
    
    # Optional: Test DBSCAN on small sample first
    # analyzer.test_dbscan_small_sample(sample_size=1000, eps=30, min_samples=50)
    
    # Run comprehensive analysis with parameter tuning
    analyzer.run_comprehensive_analysis(
        # sample_size=sample_size,  # Uncomment to use sample
        dbscan_eps=30,
        dbscan_min_samples=100,
        tune_parameters=True  # This will automatically find better parameters
    )


if __name__ == "__main__":
    main()
