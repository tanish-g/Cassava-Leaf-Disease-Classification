import os
import torch
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader

class leafDataset(Dataset):
    def __init__(self,df,im_path,transforms=None,is_test=False):
        self.df = df
        self.im_path = im_path
        self.transforms = transforms
        self.is_test = is_test
        
    def __getitem__(self,idx):
        img_path = self.df.iloc[idx]['image_id']
        path = os.path.join(self.im_path,img_path)
        img = Image.open(path)
        if self.transforms:
            img = self.transforms(**{"image": np.array(img)})["image"] 
        if self.is_test:
             return img
        target = self.df.iloc[idx]['label']
        return img,torch.tensor(target,dtype=torch.int64)
    def __len__(self):
        return self.df.shape[0]
