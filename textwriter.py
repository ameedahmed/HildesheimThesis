import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import cv2
from tqdm import tqdm
from torchvision import transforms

#folders = ['train','val']


artist_names = os.listdir(f'/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/dataset_resized_train_val_test_combi/val_folder_train_val_test_combi')
artist_names = [art.replace("val_", "").replace("_preprocessed_384_dataset", "") for art in artist_names]

for name in artist_names:
    input_dir = f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/dataset_resized_train_val_test_combi/val_folder_train_val_test_combi/val_{name}_preprocessed_384_dataset"
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            with open(f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/dataset_resized_train_val_test_combi/val.txt", "a") as train_file:
                train_file.write(f"{input_path}\n")