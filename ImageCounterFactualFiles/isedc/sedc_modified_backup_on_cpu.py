# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:58:26 2020

@author: TVermeire
"""

def sedc(image, classifier, segments, mode, device):
    
    import numpy as np
    import cv2
    import torch
    
    image = image.to(device)
    classifier = classifier.to(device)
    
    result = classifier(image)
    predicted_class_index = np.argmax(result.cpu().detach().numpy())
    result = result.cpu().detach().numpy() ## Convert to numpy array
    image_numpy = image.clone().detach().cpu().numpy()
    
    # Convert from (batch, channels, height, width) to (height, width, channels) for OpenCV
    if len(image_numpy.shape) == 4:
        image_numpy = image_numpy[0]  # Remove batch dimension
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    
    c = np.argmax(result)
    p = result[0,c]
    R = [] #list of explanations
    I = [] #corresponding perturbed images
    C = [] #corresponding new classes
    P = [] #corresponding scores for original class
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
            print(image_absolute.shape)
            mask = np.full([image_absolute.shape[0],image_absolute.shape[1]],0)
            mask[segments == j] = 255
            mask = mask.astype('uint8')
            image_segment_inpainted = cv2.inpaint(image_absolute, mask, 3, cv2.INPAINT_NS)
            perturbed_image[segments == j] = image_segment_inpainted[segments == j]/255.0
    
    
    for j in np.unique(segments):
        test_image = image.clone().detach().cpu().numpy()
        if len(test_image.shape) == 4:
            test_image = test_image[0]  # Remove batch dimension
        test_image = np.transpose(test_image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        test_image[segments == j] = perturbed_image[segments == j]
        test_image = np.transpose(test_image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        test_image = torch.tensor(test_image).unsqueeze(0) ## Convert to tensor for prediction
        test_image = test_image.to(device)
        classifier = classifier.to(device)
        result = classifier(test_image)
        result = result.cpu().detach().numpy() ## Convert to numpy array
        c_new = np.argmax(result)
        p_new = result[0,c]
        
        if c_new != c:
            R.append([j])
            I.append(test_image)
            C.append(c_new)
            P.append(p_new)

        else: 
            sets_to_expand_on.append([j])
            P_sets_to_expand_on = np.append(P_sets_to_expand_on,p-p_new)
    
    
    while len(R) == 0:
        combo = np.argmax(P_sets_to_expand_on)
        combo_set = []
        for j in np.unique(segments):
            if j not in sets_to_expand_on[combo]:
                combo_set.append(np.append(sets_to_expand_on[combo],j))
        
        # Make sure to not go back to previous node
        del sets_to_expand_on[combo]
        P_sets_to_expand_on = np.delete(P_sets_to_expand_on,combo)
        
        for cs in combo_set: 
            
            test_image = image.clone().detach().cpu().numpy()
            if len(test_image.shape) == 4:
                test_image = test_image[0]  # Remove batch dimension
            test_image = np.transpose(test_image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
            for k in cs: 
                test_image[segments == k] = perturbed_image[segments == k]
            
            test_image = np.transpose(test_image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            test_image = torch.from_numpy(test_image).unsqueeze(0) ##Convert to tensor for classifier
            test_image = test_image.to(device)
            classifier = classifier.to(device)
            result = classifier(test_image)
            result = result.cpu().detach().numpy() ## Convert to numpy array
            c_new = np.argmax(result)
            p_new = result[0,c]
                
            if c_new != c:
                R.append(cs)
                I.append(test_image)
                C.append(c_new)
                P.append(p_new)

            else: 
                sets_to_expand_on.append(cs)
                P_sets_to_expand_on = np.append(P_sets_to_expand_on,p-p_new)
    
    # Select best explanation: highest score reduction
    
    image_numpy_final = image.clone().detach().cpu().numpy()##image numpy array conversion from tensor
    if len(image_numpy_final.shape) == 4:
        image_numpy_final = image_numpy_final[0]  # Remove batch dimension
    image_numpy_final = np.transpose(image_numpy_final, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    
    best_explanation = np.argmax(p - P) 
    segments_in_explanation = R[best_explanation]
    explanation = np.full([image_numpy_final.shape[0],image_numpy_final.shape[1],image_numpy_final.shape[2]],0/255.0)
    for i in R[best_explanation]:
        explanation[segments == i] = image_numpy_final[segments == i]
    perturbation = I[best_explanation]
    new_class = C[best_explanation]
              
    return explanation, segments_in_explanation, perturbation, new_class, predicted_class_index             
                