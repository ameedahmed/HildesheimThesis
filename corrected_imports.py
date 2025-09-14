# Corrected import section for CSWin Transformer notebooks
import sys
import os

# Method 1: Add the CSWin_Transformer_main directory to Python path
current_dir = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main"
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Method 2: Alternative - use relative path from notebook location
# __file__ = "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main"
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Now you can import from models
from models.cswinmodified import CSWin_96_24322_base_384

# Other imports
import matplotlib.pylab as plt
import numpy as np
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import torch
from timm.models import create_model
import torchvision.transforms as T

print("✓ All imports successful!") 