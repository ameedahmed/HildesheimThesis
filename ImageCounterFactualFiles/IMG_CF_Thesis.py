import sys
import numpy as np
import PIL.Image as Image
import os
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import re
import csv

# Method 1: Add parent directory to Python path

current_dir = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main"
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from models.cswinmodified import CSWin_96_24322_base_384

# Now you can import from models

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import torch
from timm.models import create_model
from PIL import Image
import torchvision.transforms as T
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd
import numpy as np
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import cv2
from datetime import datetime
import time
import signal



# Define a timeout handler
def handler(signum, frame):
    raise TimeoutError("Operation timed out")


def sanitize_filename(filename):
    """
    Sanitize filename by removing special characters and spaces
    """
    # Extract just the filename from the full path
    if isinstance(filename, tuple):
        filename = filename[0]  # If it's a tuple, take the first element
    
    # Get just the filename without the path
    filename = os.path.basename(filename)
    
    # Remove or replace problematic characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Remove multiple underscores
    filename = re.sub(r'_+', '_', filename)
    
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    
    # Limit length
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:95] + ext
    
    return filename

class ImageFolderWithPaths(ImageFolder):
    # override the __getitem__ method. this is the only change.
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super().__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        return original_tuple + (path,)

# Set up validation DataLoader
test_transforms = T.Compose([
            T.Resize(384),
            T.CenterCrop(384),
            T.ToTensor(),  # This was missing!,
        ])

artstyle = True
century = False

if artstyle:    
    #artstyle model
    model_path = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/dataset_resized_train_val_test_combi/TrainValTestModelResult/finetune/20250726-093555-CSWin_96_24322_base_384-384/model_best.pth.tar"
    num_classes = 6  # Artstyle model has 6 classes
    labels_path = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/dataset_resized_train_val_test_combi/trainvaltestcombi_artstyle_class_index.json"
    temp_label = json.load(open(labels_path, 'r'))
    labels = {}
    label_list = []
    
    for i in range(6):
        full_path = temp_label[str(i)][0]
        folder_name = os.path.basename(full_path)
        # Extract style name:
        style_name = folder_name.replace('train_', '').replace('_preprocessed_384_dataset', '')#.replace('_dataset', '')
        
        labels[style_name] = i
        label_list.append(style_name)

    test_dir = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/dataset_resized_train_val_test_combi/test"
    test_dataset = ImageFolderWithPaths(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    class_names = test_dataset.classes
    print(f'Found {len(test_dataset)} images in {len(class_names)} classes.')
elif century:
    model_path = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/PortraitDataset/CenturyWiseLabelsInsteadofArtists/ResultModelTrainedCenturyWise/finetune/20250727-105348-CSWin_96_24322_base_384-384/model_best.pth.tar"
    num_classes = 3
    labels_path = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/PortraitDataset/CenturyWiseLabelsInsteadofArtists/century_class_index.json"

    temp_label = json.load(open(labels_path, 'r'))
    labels = {}
    label_list = []
    for i in range(3):
        full_path = temp_label[str(i)][0]
        folder_name = os.path.basename(full_path)
        # Extract style name:
        style_name = folder_name.replace('_century_resized_dataset', '')
        
        labels[style_name] = i
        label_list.append(style_name)
    test_dir = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/PortraitDataset/CenturyWiseLabelsInsteadofArtists/test_century"
    test_dataset = ImageFolderWithPaths(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    class_names = test_dataset.classes
    print(f'Found {len(test_dataset)} images in {len(class_names)} classes.')
else:
    model_path = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/outputSmoothingLossPLScheduler/NPortrait32B/finetune/20250713-125113-CSWin_96_24322_base_384-384/model_best.pth.tar"
    num_classes = 9  # Portrait model has 9 classes
    labels_path = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/PortraitDataset/artist_class_index.json"
    labels = {}
    label_list = []
    temp_label = json.load(open(labels_path, 'r'))
    for i in range(9):
        full_path = temp_label[str(i)][0]
        folder_name = os.path.basename(full_path)
        # Extract style name:
        style_name = folder_name.replace('_resizeddataset', '')
        labels[style_name] = i
        label_list.append(style_name)
    test_dir = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/PortraitDataset/test"
    test_dataset = ImageFolderWithPaths(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    class_names = test_dataset.classes
    print(f'Found {len(test_dataset)} images in {len(class_names)} classes.')

# Import model
# Import model
model = CSWin_96_24322_base_384(pretrained=False,num_classes=num_classes,img_size=384)
checkpoint = torch.load(model_path,weights_only=False)

# Extract the state_dict (handles both plain and wrapped checkpoints)
if 'state_dict_ema' in checkpoint:
    state_dict = checkpoint['state_dict_ema']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# 4. Remove 'module.' prefix if present (for DataParallel checkpoints)
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '') if k.startswith('module.') else k
    new_state_dict[new_key] = v

# 5. Load weights into the model

missing, unexpected = model.load_state_dict(new_state_dict, strict=False)  # strict=False ignores non-matching keys
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

# 6. Set model to evaluation mode
model.eval()

# 5. (Optional) Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define image shape
IMAGE_SHAPE = (384, 384)
classifier = model

from_factual__class_to_another_class = True

if from_factual__class_to_another_class:
    #%%time
    # CREATE RANDOM CFs - CHANGE UNTIL THE FACTUAL CLASS IS FLIPPED
    from isedc.sedc_pytorch import sedc_pytorch # SEDC makes that
    print(f"beginning sedc loop" )
    # Create segments
    with torch.no_grad():
        for batch_idx, (images, labels, filenames) in enumerate(test_loader):
            if images.max() > 1.0:  # Assuming image is in [0, 255] range
                images = images / 255.0
            image = (images.cpu())
            image_tensor = image[0].clone().detach().permute(1, 2, 0)
            image = image[0].permute(1, 2, 0).numpy()
            ##Predict the output of the model
#            output = classifier(image_tensor)  # Don't call unsqueeze again!
 #           predicted_class = np.argmax(output.cpu().detach().numpy())
            label_scalar = labels.item()
   #         print('Predicted Class: ' + label_list[predicted_class])
            print(f"beginning segmentation")
            ## Once you have the output of the model, segment the images into different chunks for sedc-t implementation
            segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
            #Now convert the image to numpy for sedct
            image_tensor = torch.tensor(image).permute(2, 0, 1).float()
            image_tensor = image_tensor.unsqueeze(0)
            # Sanitize filename
            safe_filename = sanitize_filename(filenames)
            print(f"Starting SEDC Function for {safe_filename}")
            #Now perform sedct on the image
            signal.signal(signal.SIGALRM,handler)
            signal.alarm(540)
            try:
                start = time.time()
                explanation, segments_in_explanation, perturbation, new_class, predicted_class_index = sedc_pytorch(image_tensor, classifier, segments, 'blur', device)
                signal.alarm(0)

                #Log the true label, the prediction by the model and the change in 
                print(f"True Label: {label_list[labels.item()]}, Predicted Class {label_list[predicted_class_index]}, New Class: {label_list[new_class]}")
                time_end = time.time()
                print(f"Time taken for sedc function is: {(time_end-start)/60} mins for file {safe_filename}")

                #Store the label, prediction, new label and filename into an excel file so that it can be later sorted for object detection model
                csv_filename = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/Counterfactual_log.csv"
                csv_filename_wrong = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/WrongClassificationCounterfactual_log.csv"

                # Check if the file exists to decide whether to write the header
                file_exists = os.path.isfile(csv_filename)
                file_exists_wrong = os.path.isfile(csv_filename_wrong)
                            
                if label_list[labels.item()]==label_list[predicted_class_index]:
                    print(f"Prediction correct: Writing into the correct csv file for file: {safe_filename}")
                    with open(csv_filename, mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        if not file_exists:
                            writer.writerow(["Filename", "True Label", "Predicted Class", "New Class", "Time Taken (s)"])
                        
                        writer.writerow([
                            safe_filename,
                            label_list[labels.item()],
                            label_list[predicted_class_index],
                            label_list[new_class],
                            round(time_end - start, 4)
                        ])

                    print(f"OG Label:{label_list[labels.item()]} Predicted Label {label_list[predicted_class_index]}")
                    save_dir = f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/correct_classification_exp&newimg/{label_list[labels.item()]}"
                    os.makedirs(save_dir, exist_ok=True)
                    to_pil = transforms.ToPILImage()
                    pil_image = to_pil(images[0])
                    pil_image.rotate(-90, expand=True)
                    print(f"Saving Correctly Classified Input Image: {safe_filename}")
                    plt.imshow(pil_image)
                    plt.savefig(os.path.join(save_dir,f'Input_{safe_filename}.png'))
                    plt.clf()
                    print(f"Saving Explanation For Correctly Classified Input Image: {safe_filename}")
                    plt.imshow(explanation) 
                    plt.savefig(os.path.join(save_dir,f'ReasonForOriginalClassification_{safe_filename}.png'))
                    plt.clf()
                    # Convert perturbation tensor from GPU to CPU and then to numpy for visualization
                    perturbation_cpu = perturbation.cpu().detach().numpy()
                    if len(perturbation_cpu.shape) == 4:
                        perturbation_cpu = perturbation_cpu[0]  # Remove batch dimension
                    perturbation_cpu = np.transpose(perturbation_cpu, (1, 2, 0))  # (C, H, W) -> (H, W, C)
                    print(f"Saving NewClassNewImage For Correctly Classified Input Image: {safe_filename}")
                    plt.imshow(perturbation_cpu)
                    plt.savefig(os.path.join(save_dir,f'WhatHappensWhenWeBlurVitalParts_{safe_filename}.png'))
                    plt.clf()
                    print(label_list[new_class])
                else:
                    print(f"Wrong classification detected for {safe_filename}: True={label_list[labels.item()]}, Predicted={label_list[predicted_class_index]}")
                    print(f"Prediction wrong: Writing into the wrong csv file for file: {safe_filename}")
                    # Log wrong classification to a separate file
                    with open(csv_filename_wrong, mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        if not file_exists_wrong:
                            writer.writerow(["Filename", "True Label", "Predicted Class", "New Class", "Time Taken (s)"])
                        
                        writer.writerow([
                            safe_filename,
                            label_list[labels.item()],
                            label_list[predicted_class_index],
                            label_list[new_class],
                            round(time_end - start, 4)
                        ])
                    
                    # Still save the input image for reference
                    save_dir = f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/wrong_classification_exp&newimg/{label_list[labels.item()]}"
                    os.makedirs(save_dir, exist_ok=True)
                    
                    to_pil = transforms.ToPILImage()
                    pil_image = to_pil(images[0])
                    pil_image.rotate(-90, expand=True)
                    
                    # Save input image only
                    plt.imshow(pil_image)
                    print(f"Saving the wrongly classified input image {safe_filename}")
                    plt.savefig(os.path.join(save_dir, f'Input_WrongClassification_{safe_filename}.png'))
                    plt.clf()
                    ##Save the explanation of wrong classification
                    print(f"Saving the explanation wrongly classified image {safe_filename}")
                    plt.imshow(explanation) 
                    plt.savefig(os.path.join(save_dir,f'ReasonForOriginalClassification_{safe_filename}.png'))
                    perturbation_cpu = perturbation.cpu().detach().numpy()
                    if len(perturbation_cpu.shape) == 4:
                        perturbation_cpu = perturbation_cpu[0]  # Remove batch dimension
                    perturbation_cpu = np.transpose(perturbation_cpu, (1, 2, 0))  # (C, H, W) -> (H, W, C)
                    print(f"Saving the pertubration/NewClass for wrongly classified input image {safe_filename}")
                    plt.imshow(perturbation_cpu)
                    plt.savefig(os.path.join(save_dir,f'WhatHappensWhenWeBlurVitalParts_{safe_filename}.png'))
                    plt.clf()
            except TimeoutError:
                print(f"Skipped: {safe_filename} took too long")
                csv_filename_timeout = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/Timeout_log.csv"
                # Check if the file exists to decide whether to write the header
                file_exists = os.path.isfile(csv_filename_timeout)
                with open(csv_filename_timeout, mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        if not file_exists:
                            writer.writerow(["OGFileName", "CleanedFileName"])
                        
                        writer.writerow([
                            filenames,
                            safe_filename
                        ])
            finally:
                signal.alarm(0)

                
                
        

###For the else part just save the file
"""                print("Class is different, initiating sedc with targeted CF")
                from isedc.sedc_t2 import sedc_t2
                save_dir = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/wrong_classification_exp&newimg"
                os.makedirs(save_dir, exist_ok=True)
                target_class = labels.item()
                start_time_sedct2 = time.time()
                print("Starting sedct2")
                explanation, segments_in_explanation, perturbation, new_class = sedc_t2(
                image_tensor, 
                classifier, 
                segments, 
                target_class, 
                'blur', 
                device=device)
                print(f"End time for sedct2: {(time.time() - start_time_sedct2)}")
                to_pil = transforms.ToPILImage()
                pil_image = to_pil(images[0])
                pil_image.rotate(-90, expand=True)
                plt.imshow(pil_image)
                plt.savefig(os.path.join(save_dir,f'Input_{safe_filename}.png'))
                plt.clf()
                plt.imshow(explanation)
                plt.savefig(os.path.join(save_dir,f"ICReasonForOriginalClassification_{safe_filename}.png"))
                plt.clf()
                perturbation_cpu = perturbation.cpu().detach().numpy()
                if len(perturbation_cpu.shape) == 4:
                    perturbation_cpu = perturbation_cpu[0]  # Remove batch dimension
                perturbation_cpu = np.transpose(perturbation_cpu, (1, 2, 0))  # (C, H, W) -> (H, W, C)
                plt.imshow(perturbation_cpu)
                plt.savefig(os.path.join(save_dir,f"ICWhatHappensWhenWeBlurVitalParts_{safe_filename}.png"))
                plt.clf()
                print(label_list[new_class])"""
"""else:
    # CREATE TARGETED CF WITH OPTIMIZED FUNCTION
    from isedc.sedc_t2 import sedc_t2

    # Create segments
    segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)

    # Try to generate CF targeting a specific art style class
    # You can change the target class by modifying the index (0-5 for artstyle, 0-8 for portrait)
    image_tensor = torch.tensor(image).permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0)
    target_class = 1  # Change this to target different classes
    explanation, segments_in_explanation, perturbation, new_class = sedc_t2(
        image_tensor, 
        classifier, 
        segments, 
        target_class, 
        'blur', 
        device=device)

    # Show explanation image
    # Show explanation image 
    plt.imshow(explanation)
    plt.savefig('test_explanation_t2.png')
    # Convert perturbation tensor from GPU to CPU and then to numpy for visualization
    perturbation_cpu = perturbation.cpu().detach().numpy()
    if len(perturbation_cpu.shape) == 4:
        perturbation_cpu = perturbation_cpu[0]  # Remove batch dimension
    perturbation_cpu = np.transpose(perturbation_cpu, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    plt.imshow(perturbation_cpu)
    plt.savefig('test_perturbation_t2.png')
    print(f"Target class: {label_list[target_class]}")
    print(f"Achieved class: {label_list[new_class]}")"""
    
    

"""# Create output directory for storing results
output_dir = f"counterfactual_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "original_images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "counterfactual_images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "after_blurring"), exist_ok=True)

# Store results
results = []

## Perform the predictions for the images, and then for the right predictions, perform EDC on it
all_preds = []
all_labels = []
all_filenames = []
all_images = []

print("Running predictions on test set...")
# Class Prediction example - FIXED VERSION
with torch.no_grad():
    for batch_idx, (inputs, labels, filenames) in enumerate(test_loader):
        model, inputs, labels = model.to(device), inputs.to(device), labels.to(device)
        model.eval()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())  # Fixed the bug here
        all_filenames.extend(filenames)
        all_images.extend(inputs.cpu())  # Store the images for later use
        
        if batch_idx % 10 == 0:
            print(f"Processed batch {batch_idx}/{len(test_loader)}")
        
acc = np.mean(np.array(all_preds) == np.array(all_labels))            
print(f'Test Accuracy: {acc*100:.2f}%')

# Separate correct and incorrect predictions
    # Check if image is already a tensor
    if isinstance(image, torch.Tensor):
        # If it's already a tensor, just apply normalization and ensure correct shape
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension if needed
        # Apply normalization if not already normalized
        if image.max() > 1.0:  # Assuming image is in [0, 255] range
            image = image / 255.0
        # Apply ImageNet normalization
    else:
        # If it's a PIL Image, apply the full transform pipeline
        transform = T.Compose([
            T.Resize(384),
            T.CenterCrop(384),
            T.ToTensor(),  # This was missing!,
        ])
        image = transform(image).unsqueeze(0)
    
    image = image.to(device)
    model = model.to(device)
    # Remove the extra unsqueeze here - image already has batch dimension
    output = classifier(image)  # Don't call unsqueeze again!
    predicted_class = np.argmax(output.cpu().detach().numpy())
    labels = {y:x for x,y in labels.items()}
    print('Class: ' + labels[predicted_class])
    
    # Create segments
image = (image.cpu())
image_tensor = image[0].clone().detach().permute(1, 2, 0)
image = image[0].permute(1, 2, 0).numpy()"""