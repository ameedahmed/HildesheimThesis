#!/usr/bin/env python3
"""
Script to create attention analysis for no-overlap files and save results to nooverlapfolder
"""

import os
import csv
import shutil
import torch
import numpy as np
import cv2
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
        ano_overlap_attention_file = os.path.join(base_output, "correct_classification_exp&newimg", class_name, f'AttentionAnalysis_{base_filename}.png')
        ano_overlap_raw_file = os.path.join(base_output, "correct_classification_exp&newimg", class_name, f'RawAttentionMap_{base_filename}.png')
        
        # If files exist, return early with existing data
        if os.path.exists(ano_overlap_attention_file) and os.path.exists(ano_overlap_raw_file):
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
    # Paths
    base_path = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main"
    no_overlap_csv = os.path.join(base_path, "ImageCounterfactualExplanations/no_overlap_files/Bernigeroth, Martin/no_overlap_files.csv")
    original_script = os.path.join(base_path, "ImageCounterfactualExplanations/Vizfiletest.py")
    modified_script = os.path.join(base_path, "ImageCounterfactualExplanations/Vizfiletest_no_overlap.py")
    test_data_dir = os.path.join(base_path, "PortraitDataset/test")
    
    # Read the no_overlap_files.csv
    print("Reading no_overlap_files.csv...")
    no_overlap_files = []
    try:
        with open(no_overlap_csv, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Extract just the filename from the path
                    filename = os.path.basename(line)
                    no_overlap_files.append(filename)
        print(f"Found {len(no_overlap_files)} files to process")
    except FileNotFoundError:
        print(f"Error: Could not find {no_overlap_csv}")
        return
    
    # Create the full paths for these files
    full_paths = []
    for filename in no_overlap_files:
        # The files are in the Bernigeroth, Martin subdirectory
        filename = filename.replace("ReasonForOriginalClassification_","").replace(".jpg.png",".jpg")
        full_path = os.path.join(test_data_dir, "Bernigeroth, Martin_ogdataset", filename)
        