# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:49:20 2020

@author: TVermeire
"""

import numpy as np
import cv2
from time import time
import torch

# BEST-FIRST: difference between target class score and predicted class score

def sedc_t2_fast(image, classifier, segments, target_class, mode, max_time=600,device=None):

    init_time = time()
    image = image.to(device)
    classifier = classifier.to(device)
    
    result = classifier(image)
    result = result.cpu().detach().numpy()
    image_numpy = image.clone().detach().cpu().numpy()
    
    # Convert from (batch, channels, height, width) to (height, width, channels) for OpenCV
    if len(image_numpy.shape) == 4:
        image_numpy = image_numpy[0]  # Remove batch dimension
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  # (C, H, W) -> (H, W, C)    

    c = np.argmax(result)
    p = result[0, target_class]
    R = [] #list of explanations
    I = [] #corresponding perturbed images
    C = [] #corresponding new classes
    P = [] #corresponding scores for target class
    sets_to_expand_on = []
    P_sets_to_expand_on = np.array([])

    if mode == 'mean':
        perturbed_image = np.zeros((image_numpy.shape[0], image_numpy.shape[1], 3))
        perturbed_image[:,:,0] = np.mean(image_numpy[:,:,0])
        perturbed_image[:,:,1] = np.mean(image_numpy[:,:,1])
        perturbed_image[:,:,2] = np.mean(image_numpy[:,:,2])
    elif mode == 'blur':
        perturbed_image = cv2.GaussianBlur(image_numpy, (31,31), 0)
    elif mode == 'random':
        perturbed_image = np.random.random((image_numpy.shape[0], image_numpy.shape[1], 3))
    elif mode == 'inpaint':
        perturbed_image = np.zeros((image_numpy.shape[0], image_numpy.shape[1], 3))
        for j in np.unique(segments):
            image_absolute = (image_numpy*255).astype('uint8')
            mask = np.full([image_absolute.shape[0],image_absolute.shape[1]],0)
            mask[segments == j] = 255
            mask = mask.astype('uint8')
            image_segment_inpainted = cv2.inpaint(image_absolute, mask, 3, cv2.INPAINT_NS)
            perturbed_image[segments == j] = image_segment_inpainted[segments == j]/255.0

    cf_candidates = []
    for j in np.unique(segments):
        test_image = image.clone().detach().cpu().numpy()
        if len(test_image.shape) == 4:
            test_image = test_image[0]  # Remove batch dimension
        test_image = np.transpose(test_image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        test_image[segments == j] = perturbed_image[segments == j]
        
        cf_candidates.append(test_image)

    cf_candidates = np.array(cf_candidates)
    
    # Process in smaller batches to avoid GPU memory issues
    batch_size = 64  # Adjust this based on your GPU memory
    cf_candidates = np.transpose(cf_candidates, (0, 3, 1, 2)) # Convert to (batch, channels, height, width)
    
    results_list = []
    for i in range(0, len(cf_candidates), batch_size):
        batch = cf_candidates[i:i+batch_size]
        batch_tensor = torch.tensor(batch).to(device)
        classifier = classifier.to(device)
        
        with torch.no_grad():
            batch_results = classifier(batch_tensor)
            batch_results = batch_results.cpu().detach().numpy()
            results_list.append(batch_results)
    
    results = np.vstack(results_list)
    c_new_list = np.argmax(results, axis=1)
    p_new_list = results[:, target_class]

    if target_class in c_new_list:
        R = [[x] for x in np.where(c_new_list == target_class)[0]]

        target_class_idxs = np.array(R).reshape(1, -1)[0]

        I = cf_candidates[target_class_idxs]
        C = c_new_list[target_class_idxs]
        P = p_new_list[target_class_idxs]

    sets_to_expand_on = [[x] for x in np.where(c_new_list != target_class)[0]]
    P_sets_to_expand_on = p_new_list[np.where(c_new_list != target_class)[0]]-results[np.where(c_new_list != target_class)[0], c]

    combo_set = [0]
    
    while len(R) == 0 and len(combo_set) > 0:

        combo = np.argmax(P_sets_to_expand_on)
        combo_set = []
        for j in np.unique(segments):
            if j not in sets_to_expand_on[combo]:
                combo_set.append(np.append(sets_to_expand_on[combo],j))
        
        # Make sure to not go back to previous node
        del sets_to_expand_on[combo]
        P_sets_to_expand_on = np.delete(P_sets_to_expand_on, combo)

        cf_candidates = []

        for cs in combo_set:
            test_image = image.clone().detach().cpu().numpy()
            if len(test_image.shape) == 4:
                test_image = test_image[0]  # Remove batch dimension
            test_image = np.transpose(test_image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
            for k in cs:
                test_image[segments == k] = perturbed_image[segments == k]

            cf_candidates.append(test_image)
        cf_candidates = np.array(cf_candidates)

        # Process in smaller batches to avoid GPU memory issues
        if len(cf_candidates) > 0:
            cf_candidates = np.transpose(cf_candidates, (0, 3, 1, 2)) # Convert to (batch, channels, height, width)
            
            results_list = []
            for i in range(0, len(cf_candidates), batch_size):
                batch = cf_candidates[i:i+batch_size]
                batch_tensor = torch.tensor(batch).to(device)
                classifier = classifier.to(device)
                
                with torch.no_grad():
                    batch_results = classifier(batch_tensor)
                    batch_results = batch_results.cpu().detach().numpy()
                    results_list.append(batch_results)
            
            results = np.vstack(results_list)
            c_new_list = np.argmax(results, axis=1)
            p_new_list = results[:, target_class]
        else:
            results = np.array([])
            c_new_list = np.array([])
            p_new_list = np.array([])

        if target_class in c_new_list:
            selected_idx = np.where(c_new_list == target_class)[0]

            R = np.array(combo_set)[selected_idx].tolist()
            I = cf_candidates[selected_idx]
            C = c_new_list[selected_idx]
            P = p_new_list[selected_idx]

        sets_to_expand_on += np.array(combo_set)[np.where(c_new_list != target_class)[0]].tolist()
        P_sets_to_expand_on = np.append(P_sets_to_expand_on, p_new_list[np.where(c_new_list != target_class)[0]] - results[np.where(c_new_list != target_class)[0], c])

    # Select best explanation: highest target score increase

    if len(R) > 0:
        best_explanation = np.argmax(P - p)
        segments_in_explanation = R[best_explanation]
        explanation = np.full([image_numpy.shape[0],image_numpy.shape[1],image_numpy.shape[2]],0/255.0)
        for i in R[best_explanation]:
            explanation[segments == i] = image_numpy[segments == i]
        perturbation = I[best_explanation]
        new_class = C[best_explanation]


        return explanation, segments_in_explanation, perturbation, new_class


    print('No CF found on the requested parameters')
    return None, None, None, c
