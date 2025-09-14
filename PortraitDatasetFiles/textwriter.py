import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import cv2
from tqdm import tqdm
from torchvision import transforms


folders = ['train','val','test']

for name in folders:
    input_dir = f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/PortraitDataset/CenturyWiseLabelsInsteadofArtists/{name}"
    print(input_dir)
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            with open(f"/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/PortraitDataset/train_val_combi.txt", "a") as train_file:
                train_file.write(f"{input_path}\n")#