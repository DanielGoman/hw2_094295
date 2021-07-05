import os 

import pandas as pd
import numpy as np
import cv2

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

class FacemaskRecognitionDataset(Dataset):
    
    def __init__(self, dataframe, image_dir, mode='train', transforms=None):
        super().__init__()
        
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.mode = mode
        
    def __getitem__(self, index: int):
        #Retrive Image name and its records (x1, y1, x2, y2, classname) from df
        image_name = self.df['file_name'][index]
        records = self.df[self.df["file_name"] == image_name]
        
        #Loading Image
        image = cv2.imread(self.image_dir + image_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        if self.mode == 'train':
            
            #Get bounding box co-ordinates for each box
            boxes = records[['x1', 'y1', 'x2', 'y2']].values.astype(np.float32)

            #Getting labels for each box
            temp_labels = records[['label']].values
            labels = []
            for label in temp_labels:
                label = int(label == 'True')#1 if label == 'True' else 0
                labels.append(label)


            #Converting boxes & labels into torch tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            #Creating target
            target = {}
            target['boxes'] = boxes
            target['labels'] = labels

            #Transforms
            if self.transforms:
                image = self.transforms(image)


            return image, target, image_name
        
        elif self.mode == 'test':

            if self.transforms:
                image = self.transforms(image)

            return image, image_name
    
    def __len__(self):
        return len(self.df)



def is_valid(x, y, w, h):
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return False
    else:
        return True

def dir_to_df(image_dir):
    files = os.listdir(image_dir)
    df = pd.DataFrame(columns=['file_name', 'x1', 'y1', 'x2', 'y2', 'label'])

    for image_name in files:
        
        _, axes, label = image_name.split('.')[0].split('__')
        axes = [int(val) for val in axes[1: -1].split(',')]


        if is_valid(axes[0], axes[1], axes[2], axes[3]):
            x1 = axes[0]
            x2 = x1 + axes[2]
            y1 = axes[1]
            y2 = y1 + axes[3]

            row = {'file_name': image_name,'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2,'label': label}
            df = df.append(row, ignore_index=True)

    return df