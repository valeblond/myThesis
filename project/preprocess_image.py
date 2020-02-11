import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torchvision.datasets as dset
import torchvision.transforms as transforms

#make a list of our files
subdir_path = "ragno/"
files = []
files += [os.path.join(subdir_path, f.name) for f in os.scandir(subdir_path) if f.is_file()]

# Remove not images from dataset
not_jpg = [f for f in files if not (f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png"))]
files = [file for file in files if file not in not_jpg]


for idx, i in enumerate(files):
    img = Image.open(i)
    width, height = img.size
    if (width > height):
        image3_transformer = transforms.Compose([
                    transforms.CenterCrop(height*80/100),
                    transforms.Resize((64,64))
        ])
    else:
        image3_transformer = transforms.Compose([
                    transforms.CenterCrop(width*80/100),
                    transforms.Resize((64,64))
        ])
    image3 = image3_transformer(img)
    if (i.endswith(".jpg")):
        image3.save("spiders/"+ str(idx) + ".jpg")
    elif (i.endswith(".jpeg")):
        image3.save("spiders/"+ str(idx) + ".jpeg")
    elif(i.endswith(".png")):
        image3.save("spiders/"+ str(idx) + ".png")
