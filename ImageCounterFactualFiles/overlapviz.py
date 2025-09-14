import numpy as np 
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
import pandas as pd
import os
import csv
import difflib

def find_close_match(target, candidates, cutoff=0.6):
    """Find the closest match from a list of candidates"""
    matches = difflib.get_close_matches(target, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None

##Now do it for artstyle
for class_name in os.listdir("/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/ParallelDivision/Artstyle/correct_classification_exp&newimg"):
    if class_name not in ["academicism","baroque","neoclassicism"]:
        for filename in os.listdir(f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/ParallelDivision/Artstyle/correct_classification_exp&newimg/{class_name}"):
            if filename.startswith("ReasonForOriginalClassification_"):
                if not os.path.exists(f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/OverlapAnalysisViz/Artstyle/{class_name.replace('Ukiyo_e','Ukiyo')}"):
                    os.makedirs(f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/OverlapAnalysisViz/Artstyle/{class_name.replace('Ukiyo_e','Ukiyo')}")
                close_match_files_already_done = find_close_match(filename, os.listdir(f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/OverlapAnalysisViz/Artstyle/{class_name.replace('Ukiyo_e','Ukiyo')}"), 0.6)
                if close_match_files_already_done is not None:
                    continue
                else:
                    print(f"Processing {filename} for {class_name}")
                    counterfactual_img = Image.open(f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/ParallelDivision/Artstyle/correct_classification_exp&newimg/{class_name}/{filename}")
                    cleaned_filename = filename.split("ReasonForOriginalClassification_")[1].split(".jpg")[0]
                    counterfactual = np.array(counterfactual_img)
                    cf_mask = np.any(counterfactual>0,axis=-1).astype(float)
                    attention_values = pd.read_csv(f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/NewAttentionMap/artstyle/{class_name.replace('Ukiyo_e','Ukiyo')}/{class_name}_attention_coordinates.csv")
                    attention_values = attention_values[attention_values['is_correct']==True]
                    matched_filename = find_close_match(cleaned_filename, attention_values['filename'].unique(), 0.6)
                    attention_values = attention_values[attention_values['filename']==matched_filename]
                    attention_map = np.zeros(cf_mask.shape)
                    points_added = 0
                    for _, row in attention_values.iterrows():
                        x = int(row['x'])
                        y = int(row['y'])
                        attention_val = row['attention_value']
                        
                        if 0 <= y < attention_map.shape[0] and 0 <= x < attention_map.shape[1]:
                            attention_map[y, x] = attention_val
                            points_added += 1

                    # Calculate overlap score
                    total_attention = np.sum(attention_map)
                    if total_attention > 0:
                        overlap_score = (np.sum(attention_map * cf_mask) / total_attention) * 100
                    else:
                        overlap_score = 0

                        print(f"Overlap score: {overlap_score:.2f}%")

                    
                    # =================== METHOD 1: OVERLAY WITH TRANSPARENCY ===================
                    plt.figure(figsize=(12, 5))

                    # Single combined plot
                    plt.subplot(1, 2, 1)
                    # Show attention map as base
                    im = plt.imshow(attention_map, cmap='YlOrRd', alpha=0.8)
                    # Overlay CF mask with transparency
                    plt.imshow(cf_mask, cmap='Blues', alpha=0.4)
                    plt.title(f'Combined View: Attention + CF Mask\nOverlap: {overlap_score:.2f}%', fontsize=12, fontweight='bold')
                    plt.colorbar(label='Attention Values')
                    plt.axis('off')

                    # Show just the overlap region
                    plt.subplot(1, 2, 2)
                    overlap_region = attention_map * cf_mask
                    plt.imshow(overlap_region, cmap='plasma')
                    plt.title('Overlap Region Only', fontsize=12, fontweight='bold')
                    plt.colorbar(label='Overlap Values')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
                        
                    print(f'Attention Heatmap with CF Mask Contours\nOverlap Score: {overlap_score:.2f}%')
                    plt.title(f'Attention Heatmap with CF Mask Contours\nOverlap Score: {overlap_score:.2f}%', fontsize=14, fontweight='bold')
                    plt.colorbar(im, label='Attention Values') 
                    plt.axis('off')
                    directory = f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/OverlapAnalysisViz/Artstyle/{class_name}"
                    if not os.path.exists(directory):
                        os.makedirs(directory, exist_ok=True)
                    plt.savefig(f"{directory}/{cleaned_filename}_Overlay_Transparency_Heatmap.png", dpi=300)
                    plt.close()

                    # =================== METHOD 2: SIDE-BY-SIDE COMPARISON ===================
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                    # Original attention map
                    im1 = axes[0,0].imshow(attention_map, cmap='YlOrRd')
                    axes[0,0].set_title('Attention Map', fontsize=14, fontweight='bold')
                    axes[0,0].axis('off')
                    plt.colorbar(im1, ax=axes[0,0])

                    # CF mask
                    im2 = axes[0,1].imshow(cf_mask, cmap='Blues')
                    axes[0,1].set_title('Counterfactual Mask', fontsize=14, fontweight='bold')
                    axes[0,1].axis('off')
                    plt.colorbar(im2, ax=axes[0,1])

                    # Combined overlay
                    axes[1,0].imshow(attention_map, cmap='YlOrRd', alpha=0.7)
                    axes[1,0].imshow(cf_mask, cmap='Blues', alpha=0.5)
                    axes[1,0].set_title(f'Combined Overlay\nOverlap: {overlap_score:.2f}%', fontsize=14, fontweight='bold')
                    axes[1,0].axis('off')

                    # Overlap region
                    overlap_region = attention_map * cf_mask
                    im4 = axes[1,1].imshow(overlap_region, cmap='plasma')
                    axes[1,1].set_title('Overlap Region Only', fontsize=14, fontweight='bold')
                    axes[1,1].axis('off')
                    plt.colorbar(im4, ax=axes[1,1])
                    plt.savefig(f"{directory}/{cleaned_filename}_Side_by_Side_Comparison.png", dpi=300)
                    plt.close()

                    # =================== METHOD 3: RGB COMPOSITE ===================
                    plt.figure(figsize=(10, 4))

                    # Create RGB composite where:
                    # Red channel = Attention map
                    # Blue channel = CF mask
                    # Green channel = Overlap
                    rgb_composite = np.zeros((*cf_mask.shape, 3))
                    rgb_composite[:,:,0] = attention_map / np.max(attention_map) if np.max(attention_map) > 0 else 0  # Red for attention
                    rgb_composite[:,:,2] = cf_mask  # Blue for CF mask
                    rgb_composite[:,:,1] = (attention_map * cf_mask) / np.max(attention_map * cf_mask) if np.max(attention_map * cf_mask) > 0 else 0  # Green for overlap

                    plt.subplot(1, 2, 1)
                    plt.imshow(rgb_composite)
                    plt.title(f'RGB Composite\nRed=Attention, Blue=CF Mask, Green=Overlap\nOverlap Score: {overlap_score:.2f}%', fontsize=12, fontweight='bold')
                    plt.axis('off')

                    # Legend/explanation
                    plt.subplot(1, 2, 2)
                    legend_img = np.zeros((100, 300, 3))
                    legend_img[10:30, 10:50, 0] = 1  # Red square
                    legend_img[40:60, 10:50, 2] = 1  # Blue square  
                    legend_img[70:90, 10:50, 1] = 1  # Green square
                    plt.imshow(legend_img)
                    plt.text(60, 20, 'Attention Map', fontsize=12, color='white', fontweight='bold')
                    plt.text(60, 50, 'CF Mask', fontsize=12, color='white', fontweight='bold')
                    plt.text(60, 80, 'Overlap', fontsize=12, color='white', fontweight='bold')
                    plt.title('Color Legend', fontsize=12, fontweight='bold')
                    plt.axis('off')
                    plt.savefig(f"{directory}/{cleaned_filename}_RGB_Composite.png", dpi=300)
                    plt.close()

                    # =================== METHOD 4: CONTOUR OVERLAY ===================
                    plt.figure(figsize=(8, 6))

                    # Show attention map as heatmap - store the image object
                    im = plt.imshow(attention_map, cmap='YlOrRd', alpha=0.8)

                    # Add CF mask as contour lines
                    if np.max(cf_mask) > 0:
                        plt.contour(cf_mask, levels=[0.5], colors='blue', linewidths=2, alpha=0.8)
                        
                    plt.title(f'Attention Heatmap with CF Mask Contours\nOverlap Score: {overlap_score:.2f}%', fontsize=14, fontweight='bold')
                    plt.colorbar(im, label='Attention Values')  # Pass the image object to colorbar
                    plt.axis('off')
                    plt.savefig(f"{directory}/{cleaned_filename}_Contour_Overlay.png", dpi=300)
                    plt.close()
                        
