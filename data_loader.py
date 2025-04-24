import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder # Stream data from images stored in folders
from tqdm import tqdm

import os # Allows to access files
import numpy as np 
from PIL import Image # Allows us to Load Images
from collections import Counter # Utility function to give us the counts of unique items in an iterable

### Lets Add this to our Dataset Class
class ClassOrNOT(Dataset):
    def __init__(self, path_to_folder, transforms, class1, class2): ### Our INIT Now Accepts a Transforms
        
        ### PREVIOUS CODE ###
        path_to_class1 = os.path.join(path_to_folder, class1)
        path_to_class2 = os.path.join(path_to_folder, class2)
        class1_files = os.listdir(path_to_class1)
        class2_files = os.listdir(path_to_class2) 
        path_to_class1_files = [os.path.join(path_to_class1, file) for file in class1_files] 
        path_to_class2_files = [os.path.join(path_to_class2, file) for file in class2_files] 
        self.training_files = path_to_class1_files + path_to_class2_files 

        self.class1_label, self.class2_label = 1, 0

        self.class1 = class1
        self.class2 = class2

        self.transform = transforms
        
    def __len__(self):
        return len(self.training_files) # The number of samples we have is just the number of training files we have

    def __getitem__(self, idx):
        ### PREVIOUS CODE ###
        path_to_image = self.training_files[idx] # Grab file path at the sampled index      
        if self.class1 in path_to_image and self.class2 not in path_to_image: # If the class1 is in the filepath, then set the label to 1
            label = self.class1_label
        else:
            label = self.class2_label # Otherwise set the label to 0
        image = Image.open(path_to_image) # Open Image with PIL to create a PIL Image
        
        ### UPDATED CODE ###
        image = self.transform(image) # Image now will go through series of transforms indicated in self.transform
        return image, label, os.path.basename(path_to_image).split('.')[0]
    
class TETS_DATA(Dataset):
    def __init__(self, path_to_folder, transforms):
        self.path_to_folder = path_to_folder
        self.transform = transforms
        self.files = os.listdir(path_to_folder)
        self.files = [os.path.join(path_to_folder, file) for file in self.files]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path_to_image = self.files[idx]
        image = Image.open(path_to_image)
        image = self.transform(image)
        return image, os.path.basename(path_to_image).split('.')[0]