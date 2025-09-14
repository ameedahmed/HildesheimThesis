import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns
from PIL import Image
from scipy.interpolate import griddata
import os
from scipy.ndimage import gaussian_filter
import difflib

def find_close_match(target, candidates, cutoff=0.6):
    """Find the closest match from a list of candidates"""
    matches = difflib.get_close_matches(target, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None

test_dataset_folder_names_og_names = os.listdir("/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/dataset_resized_train_val_test_combi/test")
test_dataset_folder_full_names = [test_folder for test_folder in test_dataset_folder_names_og_names]

test_dataset_folder_names = [test_folder.replace("_ogdataset","") for test_folder in test_dataset_folder_names_og_names]

for t_folder in test_dataset_folder_names:
        for folders in os.listdir(f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/NewAttentionMap/artstyle/{t_folder.replace('train_','').replace('_preprocessed_384_dataset','')}"): ##Class_Aubry__Peter_II_Coordinates
            if folders.endswith("attention_coordinates.csv"):
                print(folders)
                class_name = folders.replace("_attention_coordinates.csv","")
                class_name = class_name.replace("_","")
                coords = pd.read_csv(f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/NewAttentionMap/artstyle/{t_folder.replace('train_','').replace('_preprocessed_384_dataset','')}/{folders}")
                coords = coords[coords["is_correct"]==True]
                coords = coords[coords["attention_value"]>=0.15]
                print(coords["filename"].unique())
                print(len(coords["filename"].unique()))
                base_dir = f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/NewHeatmapAnalysisPlasma/{t_folder.replace('train_','').replace('_preprocessed_384_dataset','')}"
                os.makedirs(base_dir, exist_ok=True)
                """if f"{class_name}_Overall_Attention_Heatmap.png" not in os.listdir(base_dir):
                    fig = px.scatter(coords, x="x", y="y", color="attention_value", 
                                    color_continuous_scale="viridis")
                    fig.update_yaxes(autorange="reversed")  # This flips the y-axis
                    fig.update_layout(yaxis_range=[0,384],xaxis_range=[0,384])
                    fig.show()
                    fig.write_image(f"{base_dir}/{t_folder}_Overall_Attention_Heatmap.png")
                    fig = []
                else:
                    pass"""
                coord_name_with_jpg = [f"{file}.jpg" for file in coords["filename"].unique()]
                for att_file in os.listdir(f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/NewAttentionMap/artstyle/{t_folder.replace('train_','').replace('_preprocessed_384_dataset','')}"):
                    if att_file.startswith("AttentionAnalysis"):
                        att_file = att_file.replace("AttentionAnalysis_","").replace(".png",".jpg")    
                        matched_filename = find_close_match(att_file, coord_name_with_jpg, 0.7)
                        print(f"matched_filename: {matched_filename}, for att_file: {att_file}")
                        if matched_filename:
                            print(att_file)
                            img = Image.open(f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/dataset_resized_train_val_test_combi/test/train_{class_name.replace('Ukiyoe','Ukiyo')}_preprocessed_384_dataset/{att_file}") ##Open the original file
                            background = img
                            background = background.convert("RGBA")
                            # Create heatmap array
                            heatmap = np.zeros((384, 384))
                            for _, row in coords[coords["filename"]==matched_filename.replace(".jpg","")].iterrows():
                                x, y = int(row['x']), int(row['y'])
                                if 0 <= x < 384 and 0 <= y < 384:
                                    heatmap[y, x] = row['attention_value']

                            # Apply Gaussian blur for smooth overlay
                            from scipy.ndimage import gaussian_filter
                            heatmap_smooth = gaussian_filter(heatmap, sigma=2)

                            # Create overlay
                            plt.figure(figsize=(8, 8))
                            plt.imshow(background, extent=[0, 384, 384, 0])

                            # Mask zero values to make them transparent
                            heatmap_masked = np.ma.masked_where(heatmap_smooth == 0, heatmap_smooth)
                            plt.imshow(heatmap_masked, extent=[0, 384, 384, 0], 
                                    cmap='viridis', alpha=0.7, interpolation='bilinear')

                            plt.xlim(0, 384)
                            plt.ylim(384, 0)
                            plt.axis('off')
                            plt.colorbar(label='Attention Value')
                            plt.savefig(f"{base_dir}/{att_file.replace('.jpg','')}_OverlayHeatmap.png", dpi=300)
                            plt.close()
