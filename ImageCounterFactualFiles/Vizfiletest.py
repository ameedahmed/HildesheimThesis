import sys
import os
import argparse
import signal
import csv
import pickle
import time
import atexit

base_path = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main"
sys.path.append(base_path)  # Add current directory to path

import torch
from timm.models import create_model
from models.cswinmodified import CSWin_96_24322_base_384, CSWin_64_12211_tiny_224
import torchvision.transforms as T
import json
import urllib.request
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import argparse
from datetime import datetime

# Global variables for progress tracking and signal handling
progress_file = None
processed_files = set()
current_image_index = 0
total_images = 0
is_processing = False
model = None
config = None
args = None

class ProgressTracker:
    """Class to track and save progress of image processing"""
    
    def __init__(self, progress_file_path):
        self.progress_file_path = progress_file_path
        self.processed_files = set()
        self.current_index = 0
        self.total_images = 0
        self.start_time = None
        self.last_save_time = time.time()
        self.save_interval = 10  # Save progress every 10 seconds
        
    def load_progress(self):
        """Load progress from file if it exists"""
        if os.path.exists(self.progress_file_path):
            try:
                with open(self.progress_file_path, 'rb') as f:
                    data = pickle.load(f)
                    self.processed_files = data.get('processed_files', set())
                    self.current_index = data.get('current_index', 0)
                    self.total_images = data.get('total_images', 0)
                    self.start_time = data.get('start_time', None)
                    print(f"Loaded progress: {len(self.processed_files)} files processed, "
                          f"resuming from index {self.current_index}/{self.total_images}")
                    return True
            except Exception as e:
                print(f"Warning: Could not load progress file: {e}")
                return False
        return False
    
    def save_progress(self, force=False):
        """Save current progress to file"""
        current_time = time.time()
        if force or (current_time - self.last_save_time) >= self.save_interval:
            try:
                data = {
                    'processed_files': self.processed_files,
                    'current_index': self.current_index,
                    'total_images': self.total_images,
                    'start_time': self.start_time,
                    'last_save': current_time
                }
                with open(self.progress_file_path, 'wb') as f:
                    pickle.dump(data, f)
                self.last_save_time = current_time
                if force:
                    print(f"Progress saved: {len(self.processed_files)} files processed, "
                          f"current index: {self.current_index}/{self.total_images}")
            except Exception as e:
                print(f"Warning: Could not save progress: {e}")
    
    def mark_file_processed(self, file_path):
        """Mark a file as processed"""
        self.processed_files.add(file_path)
        self.current_index += 1
    
    def is_file_processed(self, file_path):
        """Check if a file has already been processed"""
        return file_path in self.processed_files
    
    def get_remaining_files(self, all_files):
        """Get list of files that still need to be processed"""
        return [f for f in all_files if not self.is_file_processed(f)]
    
    def get_progress_stats(self):
        """Get current progress statistics"""
        if self.total_images == 0:
            return "No progress"
        
        processed = len(self.processed_files)
        remaining = self.total_images - processed
        progress_pct = (processed / self.total_images) * 100
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            if processed > 0:
                avg_time_per_file = elapsed / processed
                eta = remaining * avg_time_per_file
                eta_str = f"ETA: {eta/60:.1f} minutes"
            else:
                eta_str = "ETA: Unknown"
        else:
            eta_str = "ETA: Unknown"
        
        return f"Progress: {processed}/{self.total_images} ({progress_pct:.1f}%) - {eta_str}"

def signal_handler(signum, frame):
    """Handle process termination signals gracefully"""
    global is_processing, progress_tracker
    
    print(f"\nReceived signal {signum}. Saving progress and shutting down gracefully...")
    
    if progress_tracker:
        progress_tracker.save_progress(force=True)
    
    if is_processing:
        print("Process was interrupted during processing. Progress has been saved.")
        print("Restart the script to resume from where you left off.")
    
    sys.exit(0)

def cleanup_on_exit():
    """Cleanup function called when script exits"""
    global progress_tracker
    try:
        if progress_tracker:
            progress_tracker.save_progress(force=True)
            print("Progress saved on exit.")
    except (NameError, AttributeError):
        # progress_tracker not defined yet, which is fine
        pass

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_on_exit)

# Configuration class based on imgthybrid
class ModelConfig:
    def __init__(self, model_type="artstyle"):
        self.base_path = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main"
        
        self.configs = {
            "artstyle": {
                "model_path": f"{self.base_path}/dataset_resized_train_val_test_combi/TrainValTestModelResult/finetune/20250726-093555-CSWin_96_24322_base_384-384/model_best.pth.tar",
                "labels_path": f"{self.base_path}/dataset_resized_train_val_test_combi/trainvaltestcombi_artstyle_class_index.json",
                "test_data": f"{self.base_path}/dataset_resized_train_val_test_combi/test",
                "num_classes": 6,
                "label_suffix": "_preprocessed_384_dataset",
                "label_prefix": "train_",
                "output_suffix": "artstyle",
                "model_class": CSWin_96_24322_base_384,
                "img_size": 384
            },
            "century": {
                "model_path": f"{self.base_path}/PortraitDataset/CenturyWiseLabelsInsteadofArtists/ResultModelTrainedCenturyWise/finetune/20250727-105348-CSWin_96_24322_base_384-384/model_best.pth.tar",
                "labels_path": f"{self.base_path}/PortraitDataset/CenturyWiseLabelsInsteadofArtists/century_class_index.json",
                "test_data": f"{self.base_path}/PortraitDataset/CenturyWiseLabelsInsteadofArtists/test_century",
                "num_classes": 3,
                "label_suffix": "_century_resized_dataset",
                "label_prefix": "",
                "output_suffix": "century",
                "model_class": CSWin_96_24322_base_384,
                "img_size": 384
            },
            "portrait": {
                "model_path": f"{self.base_path}/outputSmoothingLossPLScheduler/NPortrait32B/finetune/20250713-125113-CSWin_96_24322_base_384-384/model_best.pth.tar",
                "labels_path": f"{self.base_path}/PortraitDataset/artist_class_index.json",
                "test_data": f"{self.base_path}/PortraitDataset/test",
                "num_classes": 9,
                "label_suffix": "_ogdataset",
                "label_prefix": "",
                "output_suffix": "portrait",
                "model_class": CSWin_96_24322_base_384,
                "img_size": 384
            }
        }
        
        if model_type not in self.configs:
            raise ValueError(f"Model type '{model_type}' not supported. Choose from: {list(self.configs.keys())}")
        
        self.config = self.configs[model_type]
        self.model_type = model_type

def avg_heads(cam, grad):
    """
    Enhanced avg_heads function with gradient scaling to handle very small gradient values.
    
    Args:
        cam: attention map
        grad: gradient of the attention
    
    Returns:
        Scaled attention map
    """
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    
    # Check if gradients are very small and scale them up
    grad_magnitude = torch.abs(grad).mean()
    if grad_magnitude < 1e-6:
        # Scale up gradients if they're too small
        grad = grad * (1e-3 / grad_magnitude)
        print(f"Warning: Gradients were very small ({grad_magnitude:.2e}), scaled up by {(1e-3 / grad_magnitude):.2e}")
    
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition

def upsample_relevance_nearest(R, old_tokens, new_tokens):
    R = R.reshape(1, 1, old_tokens, old_tokens)
    R = torch.nn.functional.interpolate(R, size=(new_tokens, new_tokens), mode='bilinear')
    R = R.reshape(new_tokens, new_tokens)
    return R

def generate_relevance(model, input, index=None):
    """
    Generate relevance maps focusing only on global attention (stage 4).
    Handles the 2 blocks in stage 4 for CSWin base 384.
    """
    result = model(input, register_hook=True, return_attentions=True)
    if len(result) == 5:
        output, global_attention, vertical_attention, horizontal_attention, overall_attention = result
    else:
        output, global_attention = result
    
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)
    
    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)
 
    # Get number of tokens from stage 4
    num_tokens = model.stage4[0].attns[0].get_attention_map().shape[-1]

    # Initialize global relevance matrix
    R_global = torch.eye(num_tokens, num_tokens)
    
    # Process both blocks in stage 4 (CSWin base 384 has 2 blocks)
    for blk_idx, blk in enumerate(model.stage4):
        grad_g = blk.attns[0].get_attn_gradients()
        cam_g = blk.attns[0].get_attention_map()
        
        if grad_g is not None and cam_g is not None:
            # Check if gradients are meaningful
            # Use gradient-weighted attention when gradients are meaningful
            cam_g_processed = avg_heads(cam_g, grad_g)
            print(f"Stage 4 block {blk_idx + 1}: Using gradient-weighted attention")
            
            # Apply self-attention rules for this block
            R_global += apply_self_attention_rules(R_global, cam_g_processed)
        else:
            print(f"Stage 4 block {blk_idx + 1}: Skipping - missing gradients or attention maps")
        
        print(f"Processed stage 4 block {blk_idx + 1}/2")

    # Return only global attention relevance
    return R_global.sum(dim=0)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def get_patch_grid_size(model, img_size):
    # Try to get patch size from model, fallback to 4 if not present
    patch_size = getattr(model, 'patch_size', 4)
    if isinstance(patch_size, tuple):
        patch_h, patch_w = patch_size
    else:
        patch_h = patch_w = patch_size
    grid_h = img_size // patch_h
    grid_w = img_size // patch_w
    return grid_h, grid_w

def generate_visualization(model, original_image, img_size, class_index=None):
    """
    Generate visualization focusing only on global attention.
    Returns only global attention map and raw global attention.
    """
    # Get relevance map for global attention only
    R_global = generate_relevance(model, original_image.unsqueeze(0), index=class_index)
    grid_h, grid_w = get_patch_grid_size(model, img_size)
    
    # Process relevance for visualization
    def process_relevance(R):
        R = R.detach()
        numel = R.numel()
        # Try to find two factors (h, w) such that h * w == numel and h <= w
        best_h, best_w = 1, numel
        min_diff = numel
        for h in range(1, int(np.sqrt(numel)) + 1):
            if numel % h == 0:
                w = numel // h
                if abs(w - h) < min_diff:
                    best_h, best_w = h, w
                    min_diff = abs(w - h)
        # If not a perfect rectangle, pad to next (h, w)
        if best_h * best_w != numel:
            next_square = int(np.ceil(np.sqrt(numel))) ** 2
            pad_len = next_square - numel
            R = torch.nn.functional.pad(R, (0, pad_len))
            best_h = best_w = int(np.sqrt(next_square))
        R = R.reshape(1, 1, best_h, best_w)
        R = torch.nn.functional.interpolate(R, size=(img_size, img_size), mode='bilinear')
        R = R.reshape(img_size, img_size).data.cpu().numpy()
        R = (R - R.min()) / (R.max() - R.min())
        return R

    global_map = process_relevance(R_global)
    image_np = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    vis_global = show_cam_on_image(image_np, global_map)
    vis_global = np.uint8(255 * vis_global)
    vis_global = cv2.cvtColor(np.array(vis_global), cv2.COLOR_RGB2BGR)

    # Return only global attention visualization and raw map
    return vis_global, global_map

def extract_high_attention_coordinates(attention_map, threshold=0.7):
    """Extract coordinates of high attention values"""
    high_attention_coords = []
    height, width = attention_map.shape
    
    for y in range(height):
        for x in range(width):
            if attention_map[y, x] >= threshold:
                high_attention_coords.append({
                    'x': x,
                    'y': y,
                    'attention_value': float(attention_map[y, x])
                })
    
    return high_attention_coords

def sanitize_class_name(class_name):
    """Remove or replace problematic characters in class names for safe file/directory naming"""
    # Replace commas, spaces, and other problematic characters
    sanitized = class_name.replace(',', '_').replace(' ', '_').replace('/', '_').replace('\\', '_')
    # Remove any other potentially problematic characters
    sanitized = ''.join(c for c in sanitized if c.isalnum() or c in '_-')
    return sanitized

def save_attention_analysis(image_path, model, config, output_base_dir, args):
    """Main function to analyze attention and save results"""
    
    # Check if output files already exist (skip if already processed)
    if args.output:
        base_output = args.output
    else:
        base_output = os.path.join(config.base_path, "ImageCounterfactualExplanations")
    
    # Get true label from folder structure first to determine output path
    parent_folder = os.path.basename(os.path.dirname(image_path))
    
    # Load labels to determine true class
    with open(config.config['labels_path'], 'r') as f:
        temp_label = json.load(f)
    
    labels = {}
    label_list = []
    for i in range(config.config['num_classes']):
        full_path = temp_label[str(i)][0]
        folder_name = os.path.basename(full_path)
        style_name = folder_name.replace(config.config['label_prefix'], '').replace(config.config['label_suffix'], '')
        labels[style_name] = i
        label_list.append(style_name)
    
    # Determine true label (simplified version for path checking)
    true_label_idx = None
    if parent_folder in labels:
        true_label_idx = labels[parent_folder]
    else:
        # Use the actual configuration values for consistent processing
        clean_parent = parent_folder
        if config.config['label_prefix']:
            clean_parent = clean_parent.replace(config.config['label_prefix'], '')
        if config.config['label_suffix']:
            clean_parent = clean_parent.replace(config.config['label_suffix'], '')
        if clean_parent in labels:
            true_label_idx = labels[clean_parent]
        else:
            # Try to find best match
            for label_name, idx in labels.items():
                # Use the actual configuration values for consistent processing
                clean_label = label_name
                if config.config['label_prefix']:
                    clean_label = clean_label.replace(config.config['label_prefix'], '')
                if config.config['label_suffix']:
                    clean_label = clean_label.replace(config.config['label_suffix'], '')
                if clean_label.lower() == clean_parent.lower():
                    true_label_idx = idx
                    break
            if true_label_idx == None:
                true_label_idx = 0  # Default
    
    # Check if output files already exist (only if skip_existing is enabled)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    class_name = label_list[true_label_idx]
    
    if args.skip_existing and not args.force_reprocess:
        # Check for existing output files
        attention_file = os.path.join(base_output, "correct_classification_exp&newimg", class_name, f'AttentionAnalysis_{base_filename}.png')
        raw_file = os.path.join(base_output, "correct_classification_exp&newimg", class_name, f'RawAttentionMap_{base_filename}.png')
        
        # If files exist, return early with existing data
        if os.path.exists(attention_file) and os.path.exists(raw_file):
            if args.verbose:
                print(f"Output files already exist for {base_filename}, skipping processing")
            
            # Return basic info for tracking
            return {
                'image_name': base_filename,
                'true_class': class_name,
                'predicted_class': class_name,  # We don't know the actual prediction
                'is_correct': True,  # Assume correct for now
                'confidence': 1.0,  # Placeholder
                'high_attention_count': 0,
                'all_coordinates': [],
                'high_attention_coordinates': []
            }
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize(config.config['img_size']),
        T.CenterCrop(config.config['img_size']),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(img).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1)
        predicted_class_index = pred.item()
        confidence = torch.softmax(output, dim=1).max().item()

    # Debug: Print available labels
    if args.verbose:
        print(f"Available labels: {list(labels.keys())}")
        print(f"Parent folder: {parent_folder}")
        print(f"True label index: {true_label_idx}")
        print(f"Predicted class index: {predicted_class_index}")

    # Check if classification is correct
    is_correct_classification = (true_label_idx == predicted_class_index)
    
    # Create output directory structure matching imgthybrid
    classification_folder = "correct_classification_exp&newimg" if is_correct_classification else "wrong_classification_exp&newimg"
    
    output_dir = os.path.join(base_output, classification_folder, class_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations (only global attention)
    vis_global, raw_global = generate_visualization(
        model, transform(img), config.config['img_size'], class_index=predicted_class_index
    )
    
    # Get filename
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save raw attention map separately
    plt.figure(figsize=(10, 8))
    plt.imshow(raw_global, cmap='hot')
    plt.title('Raw Global Attention Map')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'RawAttentionMap_{base_filename}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save subplot with original image and global attention map
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axs[0].imshow(img)
    axs[0].set_title(f'Original\nTrue: {label_list[true_label_idx]}\nPred: {label_list[predicted_class_index]}')
    axs[0].axis('off')
    
    # Global attention
    axs[1].imshow(vis_global)
    axs[1].set_title('Global Attention')
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'AttentionAnalysis_{base_filename}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Extract high attention coordinates
    high_attention_coords = extract_high_attention_coordinates(raw_global, threshold=args.attention_threshold)
    
    # Prepare all attention coordinates data for return (instead of saving individual CSV)
    all_coords = []
    for y in range(raw_global.shape[0]):
        for x in range(raw_global.shape[1]):
            all_coords.append({
                'image_name': base_filename,
                'true_class': label_list[true_label_idx],
                'predicted_class': label_list[predicted_class_index],
                'is_correct': is_correct_classification,
                'confidence': confidence,
                'attention_type': 'global',
                'x': x,
                'y': y,
                'attention_value': float(raw_global[y, x])
            })
    
    return {
        'image_name': base_filename,
        'true_class': label_list[true_label_idx],
        'predicted_class': label_list[predicted_class_index],
        'is_correct': is_correct_classification,
        'confidence': confidence,
        'high_attention_count': len(high_attention_coords),
        'all_coordinates': all_coords,
        'high_attention_coordinates': high_attention_coords
    }

def main():
    global progress_tracker, is_processing, model, config, args
    
    # Comprehensive CLI argument parsing
    parser = argparse.ArgumentParser(description="CSWin Transformer Global Attention Analysis Tool")
    
    # Required arguments
    parser.add_argument("--model_type", 
                        help="Model type: artstyle, century, or portrait", 
                        type=str, required=True, 
                        choices=["artstyle", "century", "portrait"])
    
    # Data input options (optional - can run without specifying input)
    parser.add_argument("--image_path", 
                        help="Path to single image for analysis", 
                        type=str, default=None)
    parser.add_argument("--data_dir", 
                        help="Directory containing images for batch analysis", 
                        type=str, default=None)
    
    # Optional arguments
    parser.add_argument("--output", 
                       help="Output directory (default: ImageCounterfactualExplanations)", 
                       type=str, default=None)
    parser.add_argument("--attention_threshold", 
                       help="Threshold for high attention coordinates (default: 0.6)", 
                       type=float, default=0.2)
    parser.add_argument("--dpi", 
                       help="DPI for output images (default: 150)", 
                       type=int, default=150)
    parser.add_argument("--device", 
                       help="Device to use: 'auto', 'cpu', or 'cuda' (default: 'cpu')", 
                       type=str, default='cpu', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument("--verbose", 
                       help="Enable verbose output", 
                       action='store_true')
    parser.add_argument("--save_raw_only", 
                       help="Save only raw attention maps (no overlays)", 
                       action='store_true')
    parser.add_argument("--csv_format", 
                       help="CSV format: 'detailed' or 'summary' (default: 'detailed')", 
                       type=str, default='detailed', choices=['detailed', 'summary'])
    parser.add_argument("--resume", 
                       help="Resume processing from previous run", 
                       action='store_true')
    parser.add_argument("--progress_file", 
                       help="Custom progress file path (default: auto-generated)", 
                       type=str, default=None)
    parser.add_argument("--skip_existing", 
                       help="Skip processing if output files already exist", 
                       action='store_true', default=True)
    parser.add_argument("--force_reprocess", 
                       help="Force reprocessing even if output files exist", 
                       action='store_true')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle skip_existing vs force_reprocess
    if args.force_reprocess:
        args.skip_existing = False
    
    if args.verbose:
        print(f"Arguments: {args}")
    
    # Initialize configuration
    config = ModelConfig(args.model_type)
    
    # Set device
    global device
    if args.device == 'auto':
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    if args.verbose:
        print(f"Using device: {device}")
        print(f"Model type: {args.model_type}")
        print(f"Image size: {config.config['img_size']}")
        print(f"Number of classes: {config.config['num_classes']}")
    
    # Load model
    if args.verbose:
        print(f"Loading {config.model_type} model...")
    
    model = config.config['model_class'](pretrained=False, num_classes=config.config['num_classes'], img_size=config.config['img_size'])
    
    checkpoint = torch.load(config.config['model_path'], map_location='cpu', weights_only=False)
    
    # Extract state_dict
    if 'state_dict_ema' in checkpoint:
        state_dict = checkpoint['state_dict_ema']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    
    # Load weights
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing and args.verbose:
        print(f"Missing keys: {missing}")
    if unexpected and args.verbose:
        print(f"Unexpected keys: {unexpected}")
    
    model.eval()
    model = model.to(device)
    
    if args.verbose:
        print(f"Model loaded successfully on {device}")
    
    # Check if any input data is provided
    if not args.image_path and not args.data_dir:
        # Use the configured test_data directory if no command-line input is specified
        if 'test_data' in config.config and os.path.exists(config.config['test_data']):
            print(f"No command-line input specified. Using configured test data directory: {config.config['test_data']}")
            args.data_dir = config.config['test_data']
        else:
            print("No input data specified. The model has been loaded successfully.")
            print("To analyze images, use --image_path or --data_dir")
            print("Available model configurations:")
            print(f"  - Model type: {args.model_type}")
            print(f"  - Image size: {config.config['img_size']}")
            print(f"  - Number of classes: {config.config['num_classes']}")
            print(f"  - Model path: {config.config['model_path']}")
            if 'test_data' in config.config:
                print(f"  - Configured test data: {config.config['test_data']}")
            return
    
    # Process single image or batch
    if args.image_path:
        # Single image analysis
        if not os.path.exists(args.image_path):
            print(f"Error: Image path {args.image_path} does not exist")
            return
        
        if args.verbose:
            print(f"Analyzing single image: {args.image_path}")
        
        result = save_attention_analysis(args.image_path, model, config, None, args)
        
        if args.verbose:
            print(f"Analysis complete for {result['image_name']}")
            print(f"True class: {result['true_class']}, Predicted: {result['predicted_class']}")
            print(f"Correct classification: {result['is_correct']}")
            print(f"High attention regions: {result['high_attention_count']}")
        
    elif args.data_dir:
        # Batch analysis
        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory {args.data_dir} does not exist")
            return
        
        if args.verbose:
            print(f"Starting batch analysis of directory: {args.data_dir}")
        
        # Initialize progress tracking
        if args.progress_file:
            progress_file_path = args.progress_file
        else:
            # Auto-generate progress file name based on data directory and model type
            data_dir_name = os.path.basename(os.path.normpath(args.data_dir))
            progress_file_path = f"progress_{args.model_type}_{data_dir_name}_{datetime.now().strftime('%Y%m%d')}.pkl"
        
        progress_tracker = ProgressTracker(progress_file_path)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for root, dirs, files in os.walk(args.data_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        # Load previous progress if resuming
        if args.resume:
            if progress_tracker.load_progress():
                # Filter out already processed files
                remaining_files = progress_tracker.get_remaining_files(image_files)
                if len(remaining_files) == 0:
                    print("All files have already been processed!")
                    return
                image_files = remaining_files
                print(f"Resuming from previous run. {len(remaining_files)} files remaining out of {len(remaining_files) + len(progress_tracker.processed_files)} total.")
            else:
                print("No previous progress found. Starting fresh.")
        
        # Initialize progress tracker with current file list
        progress_tracker.total_images = len(image_files) + len(progress_tracker.processed_files)
        progress_tracker.start_time = time.time()
        
        if args.verbose:
            print(f"Found {len(image_files)} images to process")
            print(f"Progress file: {progress_file_path}")
            print(progress_tracker.get_progress_stats())
        
        # Process each image
        results = []
        # Dictionary to collect only high attention coordinates by class (reduced memory footprint)
        class_high_attention_coords = {}
        
        # Initialize CSV files for each class at the start (streaming approach)
        class_csv_files = {}
        
        # First, determine all possible classes from the directory structure
        possible_classes = set()
        for image_path in image_files:
            parent_folder = os.path.basename(os.path.dirname(image_path))
            # Clean the folder name to match what save_attention_analysis will return
            # Use the actual configuration values for consistent processing
            clean_class_name = parent_folder
            if config.config['label_prefix']:
                clean_class_name = clean_class_name.replace(config.config['label_prefix'], '')
            if config.config['label_suffix']:
                clean_class_name = clean_class_name.replace(config.config['label_suffix'], '')
            possible_classes.add(clean_class_name)
        
        # Initialize CSV files for each class
        for class_name in possible_classes:
            class_output_dir = os.path.join(args.output or os.path.join(config.base_path, "ImageCounterfactualExplanations"), 
                                         f'Class_{sanitize_class_name(class_name)}_Coordinates')
            os.makedirs(class_output_dir, exist_ok=True)
            
            # Create CSV files and write headers
            all_coords_path = os.path.join(class_output_dir, f'AllAttentionCoordinates_{sanitize_class_name(class_name)}.csv')
            high_coords_path = os.path.join(class_output_dir, f'HighAttentionCoordinates_{sanitize_class_name(class_name)}.csv')
            
            class_csv_files[class_name] = {
                'all_coords_file': open(all_coords_path, 'w', newline=''),
                'high_coords_file': open(high_coords_path, 'w', newline=''),
                'all_coords_writer': None,
                'high_coords_writer': None
            }
            
            # Write headers
            headers = ['image_name', 'true_class', 'predicted_class', 'is_correct', 'confidence', 'attention_type', 'x', 'y', 'attention_value']
            class_csv_files[class_name]['all_coords_writer'] = csv.DictWriter(class_csv_files[class_name]['all_coords_file'], fieldnames=headers)
            class_csv_files[class_name]['high_coords_writer'] = csv.DictWriter(class_csv_files[class_name]['high_coords_file'], fieldnames=headers)
            class_csv_files[class_name]['all_coords_writer'].writeheader()
            class_csv_files[class_name]['high_coords_writer'].writeheader()
            
            # Initialize high attention coordinates collection for this class
            class_high_attention_coords[class_name] = []
        
        if args.verbose:
            print(f"Initialized CSV files for classes: {list(class_csv_files.keys())}")
        
        try:
            is_processing = True
            for i, image_path in enumerate(image_files):
                try:
                    # Check if file has already been processed
                    if progress_tracker.is_file_processed(image_path):
                        if args.verbose:
                            print(f"Skipping already processed file: {os.path.basename(image_path)}")
                        continue
                    
                    if args.verbose:
                        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
                        print(progress_tracker.get_progress_stats())
                    
                    result = save_attention_analysis(image_path, model, config, None, args)
                    results.append(result)
                    
                    # Mark file as processed and save progress
                    progress_tracker.mark_file_processed(image_path)
                    progress_tracker.save_progress()
                    
                    # Get the true class name
                    true_class = result['true_class'].replace(config.config['label_prefix'], '').replace(config.config['label_suffix'], '').replace("_resizeddataset", "")
                    
                    
                    if args.verbose:
                        print(f"Result true_class: '{true_class}'")
                        print(f"Available CSV keys: {list(class_csv_files.keys())}")
                    
                    # Stream all coordinates directly to CSV file (immediate writing, no memory storage)
                    if result['all_coordinates']:
                        if true_class in class_csv_files:
                            class_csv_files[true_class]['all_coords_writer'].writerows(result['all_coordinates'])
                            # Flush to ensure data is written immediately and free memory
                            class_csv_files[true_class]['all_coords_file'].flush()
                            # Clear the large coordinate data immediately to free memory
                            del result['all_coordinates']
                        else:
                            print(f"Warning: No CSV file found for class '{true_class}'. Available classes: {list(class_csv_files.keys())}")
                            del result['all_coordinates']
                    
                    # Store only high attention coordinates in memory (reduced memory footprint)
                    if result['high_attention_coordinates']:
                        if true_class in class_high_attention_coords:
                            class_high_attention_coords[true_class].extend(result['high_attention_coordinates'])
                            # Clear high attention coordinates from result to free memory
                            del result['high_attention_coordinates']
                        else:
                            print(f"Warning: No high attention collection found for class '{true_class}'")
                            del result['high_attention_coordinates']
                    
                    # Force garbage collection every 50 images to free memory
                    if (i + 1) % 50 == 0:
                        import gc
                        gc.collect()
                        if args.verbose:
                            print(f"Memory cleanup at image {i+1}")
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    # Clear any data that might be in memory from failed processing
                    if 'result' in locals() and hasattr(result, 'all_coordinates'):
                        del result['all_coordinates']
                    if 'result' in locals() and hasattr(result, 'high_attention_coordinates'):
                        del result['high_attention_coordinates']
                    continue
        finally:
            # Ensure all CSV files are closed even if there's an error
            for class_name, csv_data in class_csv_files.items():
                try:
                    csv_data['all_coords_file'].close()
                    csv_data['high_coords_file'].close()
                    if args.verbose:
                        print(f"Closed CSV files for class: {class_name}")
                except Exception as e:
                    print(f"Warning: Error closing CSV files for class {class_name}: {str(e)}")
            
            # Mark processing as complete and save final progress
            is_processing = False
            if progress_tracker:
                progress_tracker.save_progress(force=True)
                print(f"\nFinal progress saved: {len(progress_tracker.processed_files)} files processed")
        
        # Save consolidated high attention coordinates CSV files (only high attention, much smaller)
        if class_high_attention_coords:
            for class_name, high_coords in class_high_attention_coords.items():
                if high_coords:
                    # Create output directory for this class
                    class_output_dir = os.path.join(args.output or os.path.join(config.base_path, "ImageCounterfactualExplanations"), 
                                                 f'Class_{sanitize_class_name(class_name)}_Coordinates')
                    os.makedirs(class_output_dir, exist_ok=True)
                    
                    # Save high attention coordinates for this class
                    high_coords_df = pd.DataFrame(high_coords)
                    high_coords_csv_path = os.path.join(class_output_dir, f'HighAttentionCoordinates_{sanitize_class_name(class_name)}.csv')
                    high_coords_df.to_csv(high_coords_csv_path, index=False)
                    if args.verbose:
                        print(f"Saved high attention coordinates for class '{class_name}' to: {high_coords_csv_path}")
        
        # Save summary CSV
        if results:
            summary_df = pd.DataFrame(results)
            
            # Choose CSV format based on user preference
            if args.csv_format == 'summary':
                # Save only essential information
                summary_csv_path = os.path.join(args.output or os.path.join(config.base_path, "ImageCounterfactualExplanations"), 
                                             f'AnalysisSummary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                summary_df.to_csv(summary_csv_path, index=False)
            else:
                # Save detailed information
                summary_csv_path = os.path.join(args.output or os.path.join(config.base_path, "ImageCounterfactualExplanations"), 
                                             f'DetailedAnalysisSummary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                summary_df.to_csv(summary_csv_path, index=False)
            
            if args.verbose:
                print(f"Summary saved to: {summary_csv_path}")
                
                # Print summary statistics
                correct_count = sum(1 for r in results if r['is_correct'])
                total_count = len(results)
                print(f"\nAnalysis Summary:")
                print(f"Total images processed: {total_count}")
                print(f"Correct classifications: {correct_count}")
                print(f"Incorrect classifications: {total_count - correct_count}")
                print(f"Accuracy: {correct_count/total_count*100:.2f}%")
                
                # Print class-wise coordinate statistics
                print(f"\nClass-wise Coordinate Statistics:")
                for class_name, high_coords in class_high_attention_coords.items():
                    high_count = len(high_coords)
                    print(f"  {class_name}: {high_count} high attention coordinates")
                
                # Print final progress information
                if progress_tracker:
                    print(f"\nFinal Progress Information:")
                    print(f"Total files processed: {len(progress_tracker.processed_files)}")
                    print(f"Progress file: {progress_tracker.progress_file_path}")
                    print("You can resume processing later using: --resume")

if __name__ == "__main__":
    main()