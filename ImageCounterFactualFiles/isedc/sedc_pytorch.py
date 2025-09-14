# -*- coding: utf-8 -*-
"""
PyTorch-optimized SEDC (Search for Explanations of Deep Classifiers) implementation
Created for improved performance by minimizing CPU-GPU transfers and using PyTorch operations
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, List, Union


def sedc_pytorch(image: torch.Tensor, classifier: torch.nn.Module, segments: np.ndarray, 
                  mode: str, device: torch.device) -> Tuple[np.ndarray, List[int], torch.Tensor, int, int]:
    """
    PyTorch-optimized SEDC implementation for faster counterfactual explanations.
    
    Args:
        image: Input image tensor (B, C, H, W)
        classifier: PyTorch model for classification
        segments: Segmentation map as numpy array
        mode: Perturbation mode ('mean', 'blur', 'random', 'inpaint')
        device: Target device (cuda/cpu)
    
    Returns:
        explanation: Explanation mask as numpy array
        segments_in_explanation: List of segment indices in explanation
        perturbation: Best perturbed image tensor
        new_class: Predicted class for perturbed image
        predicted_class_index: Original predicted class
    """
    
    # Ensure tensors are on correct device
    image = image.to(device)
    classifier = classifier.to(device)
    
    # Get original prediction
    with torch.no_grad():
        result = classifier(image)
        predicted_class_index = torch.argmax(result).item()
        original_score = result[0, predicted_class_index].item()
    
    # Get unique segments
    unique_segments = np.unique(segments)
    
    # Create perturbation mask based on mode
    perturbed_image = create_perturbation_mask(image, segments, mode, device)
    
    # Initialize lists for tracking explanations
    R = []  # list of explanations
    I = []  # corresponding perturbed images
    C = []  # corresponding new classes
    P = []  # corresponding scores for original class
    sets_to_expand_on = []
    P_sets_to_expand_on = []
    
    # Test single segment perturbations
    for j in unique_segments:
        test_image = create_perturbed_image(image, perturbed_image, segments, [j], device)
        
        with torch.no_grad():
            result = classifier(test_image)
            c_new = torch.argmax(result).item()
            p_new = result[0, predicted_class_index].item()
        
        if c_new != predicted_class_index:
            R.append([j])
            I.append(test_image)
            C.append(c_new)
            P.append(p_new)
        else:
            sets_to_expand_on.append([j])
            P_sets_to_expand_on.append(original_score - p_new)
    
    # If no single segment changes the prediction, try combinations
    while len(R) == 0 and len(sets_to_expand_on) > 0:
        # Find the combination with highest score reduction
        combo_idx = np.argmax(P_sets_to_expand_on)
        combo = sets_to_expand_on[combo_idx]
        
        # Generate new combinations by adding one segment at a time
        combo_set = []
        for j in unique_segments:
            if j not in combo:
                combo_set.append(combo + [j])
        
        # Remove the processed combination
        del sets_to_expand_on[combo_idx]
        del P_sets_to_expand_on[combo_idx]
        
        # Test new combinations
        for cs in combo_set:
            test_image = create_perturbed_image(image, perturbed_image, segments, cs, device)
            
            with torch.no_grad():
                result = classifier(test_image)
                c_new = torch.argmax(result).item()
                p_new = result[0, predicted_class_index].item()
            
            if c_new != predicted_class_index:
                R.append(cs)
                I.append(test_image)
                C.append(c_new)
                P.append(p_new)
            else:
                sets_to_expand_on.append(cs)
                P_sets_to_expand_on.append(original_score - p_new)
    
    # Select best explanation: highest score reduction
    if len(R) > 0:
        best_explanation = np.argmax([original_score - p for p in P])
        segments_in_explanation = R[best_explanation]
        perturbation = I[best_explanation]
        new_class = C[best_explanation]
        
        # Create explanation mask
        explanation = create_explanation_mask(image, segments, segments_in_explanation, device)
        
        return explanation, segments_in_explanation, perturbation, new_class, predicted_class_index
    else:
        # No explanation found
        return np.zeros_like(segments), [], image, predicted_class_index, predicted_class_index


def create_perturbation_mask(image: torch.Tensor, segments: np.ndarray, mode: str, 
                           device: torch.device) -> torch.Tensor:
    """
    Create perturbation mask based on the specified mode.
    
    Args:
        image: Input image tensor
        segments: Segmentation map
        mode: Perturbation mode
        device: Target device
    
    Returns:
        Perturbation mask tensor
    """
    # Convert image to numpy for processing
    image_np = image.detach().cpu().numpy()
    if len(image_np.shape) == 4:
        image_np = image_np[0]  # Remove batch dimension
    image_np = np.transpose(image_np, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    
    if mode == 'mean':
        perturbed_image = np.zeros_like(image_np)
        perturbed_image[:, :, 0] = np.mean(image_np[:, :, 0])
        perturbed_image[:, :, 1] = np.mean(image_np[:, :, 1])
        perturbed_image[:, :, 2] = np.mean(image_np[:, :, 2])
        
    elif mode == 'blur':
        perturbed_image = cv2.GaussianBlur(image_np, (31, 31), 0)
        
    elif mode == 'random':
        perturbed_image = np.random.random(image_np.shape)
        
    elif mode == 'inpaint':
        perturbed_image = np.zeros_like(image_np)
        for j in np.unique(segments):
            image_absolute = (image_np * 255).astype('uint8')
            mask = np.full([image_absolute.shape[0], image_absolute.shape[1]], 0)
            mask[segments == j] = 255
            mask = mask.astype('uint8')
            image_segment_inpainted = cv2.inpaint(image_absolute, mask, 3, cv2.INPAINT_NS)
            perturbed_image[segments == j] = image_segment_inpainted[segments == j] / 255.0
    
    # Convert back to tensor and move to device
    perturbed_tensor = torch.from_numpy(perturbed_image).float()
    perturbed_tensor = perturbed_tensor.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
    return perturbed_tensor.to(device)


def create_perturbed_image(original_image: torch.Tensor, perturbed_image: torch.Tensor, 
                          segments: np.ndarray, segment_indices: List[int], 
                          device: torch.device) -> torch.Tensor:
    """
    Create a perturbed image by replacing specified segments with perturbation.
    
    Args:
        original_image: Original image tensor
        perturbed_image: Perturbation tensor
        segments: Segmentation map
        segment_indices: List of segment indices to perturb
        device: Target device
    
    Returns:
        Perturbed image tensor
    """
    # Create a copy of the original image
    test_image = original_image.clone()
    
    # Create mask for segments to perturb
    mask = torch.zeros_like(test_image)
    for seg_idx in segment_indices:
        segment_mask = (segments == seg_idx)
        mask[0, :, segment_mask] = 1
    
    # Apply perturbation only to selected segments
    test_image = test_image * (1 - mask) + perturbed_image * mask
    
    return test_image


def create_explanation_mask(image: torch.Tensor, segments: np.ndarray, 
                           segments_in_explanation: List[int], 
                           device: torch.device) -> np.ndarray:
    """
    Create explanation mask highlighting the segments in the explanation.
    
    Args:
        image: Original image tensor
        segments: Segmentation map
        segments_in_explanation: List of segment indices in explanation
        device: Target device
    
    Returns:
        Explanation mask as numpy array
    """
    # Convert image to numpy for final processing
    image_np = image.detach().cpu().numpy()
    if len(image_np.shape) == 4:
        image_np = image_np[0]  # Remove batch dimension
    image_np = np.transpose(image_np, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    
    # Create explanation mask
    explanation = np.zeros_like(image_np)
    for seg_idx in segments_in_explanation:
        explanation[segments == seg_idx] = image_np[segments == seg_idx]
    
    return explanation


def sedc_pytorch_batch(image: torch.Tensor, classifier: torch.nn.Module, segments: np.ndarray,
                       mode: str, device: torch.device, batch_size: int = 8) -> Tuple[np.ndarray, List[int], torch.Tensor, int, int]:
    """
    Batch-optimized version of SEDC for even better performance when testing multiple perturbations.
    
    Args:
        image: Input image tensor
        classifier: PyTorch model for classification
        segments: Segmentation map
        mode: Perturbation mode
        device: Target device
        batch_size: Number of perturbations to test simultaneously
    
    Returns:
        Same as sedc_pytorch
    """
    # Ensure tensors are on correct device
    image = image.to(device)
    classifier = classifier.to(device)
    
    # Get original prediction
    with torch.no_grad():
        result = classifier(image)
        predicted_class_index = torch.argmax(result).item()
        original_score = result[0, predicted_class_index].item()
    
    # Get unique segments
    unique_segments = np.unique(segments)
    
    # Create perturbation mask
    perturbed_image = create_perturbation_mask(image, segments, mode, device)
    
    # Initialize tracking variables
    R = []
    I = []
    C = []
    P = []
    sets_to_expand_on = []
    P_sets_to_expand_on = []
    
    # Test single segment perturbations in batches
    for i in range(0, len(unique_segments), batch_size):
        batch_segments = unique_segments[i:i + batch_size]
        batch_images = []
        
        for j in batch_segments:
            test_image = create_perturbed_image(image, perturbed_image, segments, [j], device)
            batch_images.append(test_image)
        
        # Stack batch images
        batch_tensor = torch.cat(batch_images, dim=0)
        
        # Get predictions for batch
        with torch.no_grad():
            results = classifier(batch_tensor)
            c_new_batch = torch.argmax(results, dim=1)
            p_new_batch = results[:, predicted_class_index]
        
        # Process results
        for idx, j in enumerate(batch_segments):
            c_new = c_new_batch[idx].item()
            p_new = p_new_batch[idx].item()
            
            if c_new != predicted_class_index:
                R.append([j])
                I.append(batch_images[idx])
                C.append(c_new)
                P.append(p_new)
            else:
                sets_to_expand_on.append([j])
                P_sets_to_expand_on.append(original_score - p_new)
    
    # Continue with combination testing (similar to original but with batching)
    while len(R) == 0 and len(sets_to_expand_on) > 0:
        combo_idx = np.argmax(P_sets_to_expand_on)
        combo = sets_to_expand_on[combo_idx]
        
        # Generate new combinations
        combo_set = []
        for j in unique_segments:
            if j not in combo:
                combo_set.append(combo + [j])
        
        # Remove processed combination
        del sets_to_expand_on[combo_idx]
        del P_sets_to_expand_on[combo_idx]
        
        # Test combinations in batches
        for i in range(0, len(combo_set), batch_size):
            batch_combos = combo_set[i:i + batch_size]
            batch_images = []
            
            for cs in batch_combos:
                test_image = create_perturbed_image(image, perturbed_image, segments, cs, device)
                batch_images.append(test_image)
            
            batch_tensor = torch.cat(batch_images, dim=0)
            
            with torch.no_grad():
                results = classifier(batch_tensor)
                c_new_batch = torch.argmax(results, dim=1)
                p_new_batch = results[:, predicted_class_index]
            
            for idx, cs in enumerate(batch_combos):
                c_new = c_new_batch[idx].item()
                p_new = p_new_batch[idx].item()
                
                if c_new != predicted_class_index:
                    R.append(cs)
                    I.append(batch_images[idx])
                    C.append(c_new)
                    P.append(p_new)
                else:
                    sets_to_expand_on.append(cs)
                    P_sets_to_expand_on.append(original_score - p_new)
    
    # Select best explanation
    if len(R) > 0:
        best_explanation = np.argmax([original_score - p for p in P])
        segments_in_explanation = R[best_explanation]
        perturbation = I[best_explanation]
        new_class = C[best_explanation]
        
        explanation = create_explanation_mask(image, segments, segments_in_explanation, device)
        
        return explanation, segments_in_explanation, perturbation, new_class, predicted_class_index
    else:
        return np.zeros_like(segments), [], image, predicted_class_index, predicted_class_index 