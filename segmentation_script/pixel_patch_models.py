#### 1- Importation

import sys, torch, os, glob, re, tkinter as tk, numpy as np
from tkinter import filedialog
from PIL import Image
from scipy.ndimage import gaussian_filter,distance_transform_edt
from skimage.morphology import remove_small_objects


from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.rnn import FlexibleGRU
from utils import (
    load_model,
    segment_series
)

#### 2- Loading of the different models configurations 
"""
This part is for loading the different model configurations and their weights. There are three possible configurations  :
1. Model Pixel : takes an input_size of 1 + 2 layers of 64 hidden neurons , trained on single pixel's intensity vectors
2. Model Patch : takes an input_size of 9 + 2 layers of 64 hidden neurons , trained on pixel and 8-closest neighbors intensity vectors
3. Model Distance :  takes an input_size of 2 + 2 layers of 64 hidden neurons , trained on single pixel's intensity vectors and a distance map
associated to the serie. To be used, it needs at least one reference segmentation for the image serie which is
often the segmentation of the last image.
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
pixel_model_path = "models/pixel_model.pt"
patch_model__path = "models/patch_model.pt"
distance_model_path = "models/distance_model.pt"

model_pixel = load_model(pixel_model_path, model_class=FlexibleGRU, device=device,
                   input_size=1, hidden_size=64, num_layers=2)
model_patch = load_model(patch_model__path, model_class=FlexibleGRU, device=device,
                   input_size=9, hidden_size=64, num_layers=2)
model_distance = load_model(distance_model_path, model_class=FlexibleGRU, device=device,
                   input_size=2, hidden_size=64, num_layers=2 )


#### 3- Segmentation function call

outdir = segment_series(
    model=model_pixel,        
    model_type="pixel",       # "pixel" / "patch"
    T=16,
    device="cpu",          
    threshold=0.5,
    min_duration=2,
    min_size=20,
    sigma=0.7,
    smooth_temporal=False,
    batch_size= 128,
    seg_prefix="seg_",
    folder_name="seg_pixel_model",
    seg_mode = "L"
    )

print("Binary masks saved in folder :", outdir)