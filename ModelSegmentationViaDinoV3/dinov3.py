import sys
import numpy as np
REPO_DIR = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/ParallelDivision/ObjectDetectionResults/dinov3"
sys.path.append(REPO_DIR)
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import colormaps
from functools import partial
from dinov3.eval.segmentation.inference import make_inference
import os

def make_transform(resize_size: int | list[int] = 768):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])
ADE20K_CLASSES = [
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass',
    'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair',
    'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field',
    'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion',
    'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace',
    'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway',
    'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench',
    'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel',
    'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
    'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet',
    'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool',
    'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball',
    'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher',
    'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan',
    'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag'
]
def visualize_segments_with_labels(img, segmentation_map, segment_info, file_dir, files):
    """
    Visualize the original image and segmentation with labels
    """
    plt.figure(figsize=(18, 8))
    
    # Original image
    plt.subplot(131)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")
    
    # Segmentation map
    plt.subplot(132)
    plt.imshow(segmentation_map, cmap=colormaps["Spectral"])
    plt.title("Segmentation Map")
    plt.axis("off")
    
    # Legend with class names
    plt.subplot(133)
    plt.axis('off')
    
    # Sort by percentage for better readability
    sorted_segments = sorted(segment_info.items(), 
                           key=lambda x: x[1]['percentage'], 
                           reverse=True)
    
    legend_text = "Detected Segments:\n\n"
    for class_id, info in sorted_segments:
        if info['percentage'] > 0.1:  # Only show segments with >0.1% coverage
            legend_text += f"ID {class_id}: {info['name']}\n"
            legend_text += f"  Coverage: {info['percentage']}%\n\n"
    
    plt.text(0, 0.5, legend_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    plt.title("Segment Labels")
    
    plt.tight_layout()
    plt.savefig(f"{file_dir}/{files}")
    plt.close()

def get_segment_labels(segmentation_map, class_names=ADE20K_CLASSES):
    """
    Get labels for each segment in the segmentation map
    
    Args:
        segmentation_map: torch.Tensor of shape (H, W) with class indices
        class_names: list of class names corresponding to indices
    
    Returns:
        dict: mapping of class_id -> (class_name, pixel_count, percentage)
    """
    # Convert to numpy if it's a tensor
    if isinstance(segmentation_map, torch.Tensor):
        seg_array = segmentation_map.cpu().numpy()
    else:
        seg_array = segmentation_map
    
    # Get unique class indices and their counts
    unique_classes, counts = np.unique(seg_array, return_counts=True)
    
    total_pixels = seg_array.size
    segment_info = {}
    
    for class_id, count in zip(unique_classes, counts):
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"unknown_class_{class_id}"
        
        percentage = (count / total_pixels) * 100
        segment_info[int(class_id)] = {
            'name': class_name.replace("sculpture","person"),
            'pixel_count': int(count),
            'percentage': round(percentage, 2)
        }
    
    return segment_info


##First go through the original inputs and then classify them into different segments

for correct_classification_folder in os.listdir(f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/ParallelDivision/PortraitResultsForCF"):
    if correct_classification_folder.startswith("correct"):
        for classification in os.listdir(f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/ParallelDivision/PortraitResultsForCF/{correct_classification_folder}"):
            for files in os.listdir(f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/ParallelDivision/PortraitResultsForCF/{correct_classification_folder}/{classification}"):
                try:
                    ##make the folder of classification directory if it does not exist
                    os.makedirs(f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/ParallelDivision/ObjectDetectionResults/dinov3/CommonLabelPercentage/PortraitResultsForCF/{classification}", exist_ok=True)
                    ##Only process the files that are not already processed
                    if files.startswith("Input") and files not in os.listdir(f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/ParallelDivision/ObjectDetectionResults/dinov3/CommonLabelPercentage/PortraitResultsForCF/{classification}"):
                        print(f"Processing {files}")
                        input_img_directory = f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/ParallelDivision/PortraitResultsForCF/{correct_classification_folder}/{classification}/{files}"
                        reason_img_directory = f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/ParallelDivision/PortraitResultsForCF/{correct_classification_folder}/{classification}/{files.replace('Input', 'ReasonForOriginalClassification')}"
                        segmentor = torch.hub.load(REPO_DIR, model='dinov3_vit7b16_ms', source="local", 
                        segmentor_weights="/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/ParallelDivision/ObjectDetectionResults/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth", 
                        backbone_weights="/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/ParallelDivision/ObjectDetectionResults/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth")
                        input_img = Image.open(input_img_directory)
                        reason_img = Image.open(reason_img_directory)
                        img_size = 384
                        transform = make_transform(img_size)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            batch_img = transform(input_img)[None]
                            input_pred_vit7b = segmentor(batch_img)
                            ##make segmentation maps of input image
                            input_segmentation_map_vit7b = make_inference(
                                batch_img,
                                segmentor,
                                inference_mode="slide",
                                decoder_head_type="m2f",
                                rescale_to=(input_img.size[-1], input_img.size[-2]),
                                n_output_channels=150,
                                crop_size=(img_size, img_size),
                                stride=(img_size, img_size),
                                output_activation=partial(torch.nn.functional.softmax, dim=1),
                            ).argmax(dim=1, keepdim=True)
                            ##now make segmentation map of reason image
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            reason_batch_img = transform(reason_img)[None]
                            reason_pred_vit7b = segmentor(reason_batch_img)
                            reason_segmentation_map_vit7b = make_inference(
                                reason_batch_img,
                                segmentor,
                                inference_mode="slide",
                                decoder_head_type="m2f",
                                rescale_to=(input_img.size[-1], input_img.size[-2]),
                                n_output_channels=150,
                                crop_size=(img_size, img_size),
                                stride=(img_size, img_size),
                                output_activation=partial(torch.nn.functional.softmax, dim=1),
                            ).argmax(dim=1, keepdim=True)
                        ##Now get the segmentation maps of both the images
                        input_segment_info = get_segment_labels(input_segmentation_map_vit7b[0, 0])
                        reason_segment_info = get_segment_labels(reason_segmentation_map_vit7b[0, 0])
                        ##Now get the common labels between the two segmentation maps / dictionaries
                        common_label = input_segment_info.keys() & reason_segment_info.keys()
                        #After getting the common label, find the percentage of the common label for both photographs
                        if common_label is not None:
                            ##Convert the common label into a dictionary
                            input_common_pc = {key: input_segment_info[key] for key in common_label}
                            reason_common_pc = {key: reason_segment_info[key] for key in common_label}
                            #Extract the percentage of the common labels for both photographs
                            ##Write the information to a csv file for analysis:
                            file_dir = f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/ParallelDivision/ObjectDetectionResults/dinov3/CommonLabelPercentage/PortraitResultsForCF/{classification}"
                            if not os.path.exists(file_dir):
                                os.makedirs(file_dir)
                            if not os.path.isfile(os.path.join(file_dir, "common_label_percentage.csv")):
                                with open(os.path.join(file_dir, "common_label_percentage.csv"), "w") as f:
                                    ##Write header
                                    f.write(f"file_name,classification,input_common_pc,reason_common_pc\n")
                                    f.write(f"{files},{classification},{input_common_pc},{reason_common_pc}\n")
                            else:
                                with open(os.path.join(file_dir, "common_label_percentage.csv"), "a") as f:
                                    f.write(f"{files},{classification},{input_common_pc},{reason_common_pc}\n")
                            ##Save the segmentation maps of the input and reason images
                            visualize_segments_with_labels(input_img, input_segmentation_map_vit7b[0,0].cpu(), input_segment_info, file_dir, files)
                            visualize_segments_with_labels(reason_img, reason_segmentation_map_vit7b[0,0].cpu(), reason_segment_info, file_dir, f"Reason_{files}")
                        else:
                            ##If no common pathway exists then save the images and the results to a seperate folder
                            no_common_file_dir = f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/ParallelDivision/ObjectDetectionResults/dinov3/CommonLabelPercentage/PortraitResultsForCF/{classification}/no_common_labels"
                            ##If the directory does not exist, make it
                            if not os.path.exists(no_common_file_dir):
                                os.makedirs(no_common_file_dir)
                            ##After mamking the directory, make a csv file and save the labels there
                            if not os.path.isfile(os.path.join(no_common_file_dir,"no_common_labe_percentage.csv")):
                                with open(os.path.join(no_common_file_dir, "no_common_label_percentage.csv"), "w") as f:
                                    ##Write header
                                    f.write(f"file_name,classification,input_common_pc,reason_common_pc\n")
                                    f.write(f"{files},{classification},{input_common_pc},{reason_common_pc}\n")
                            else:
                                with open(os.path.join(no_common_file_dir, "no_common_label_percentage.csv"), "a") as f:
                                    f.write(f"{files},{classification},{input_common_pc},{reason_common_pc}\n")
                            ##Save the segmentation maps of the input and reason images
                            #Save the images
                            visualize_segments_with_labels(input_img, input_segmentation_map_vit7b[0,0].cpu(), input_segment_info, no_common_file_dir, files)
                            visualize_segments_with_labels(reason_img, reason_segmentation_map_vit7b[0,0].cpu(), reason_segment_info, no_common_file_dir, f"Reason_{files}")
                except Exception as e:
                    print(f"Error processing {files}: {e}")
                    continue