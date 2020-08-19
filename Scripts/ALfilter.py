import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from sklearn.utils import shuffle

import cv2
import pydicom

from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision.models as models

from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

# Define DataLoader
    
def imP3(im):
    im = im[200:,:]
    out = cv2.equalizeHist(im)
    fG = cv2.resize(out,(103,128), interpolation = cv2.INTER_AREA)
    imt = np.zeros((103,128,3))
    for i in range(3): imt[:,:,i] = fG
    imt = cv2.rotate(imt, cv2.ROTATE_180)
    return imt

class DataLoader(Dataset):  
    
    def __init__(self, df, trans, train = 1):
        
        super(DataLoader,self)
        self.df = df
        self.path = '../../rsna-pneumonia-detection-challenge/stage_2_train_images/'        
        self.transforms = trans
        self.train = train
        
    def __getitem__(self,i):
        
        self.image = imP3(pydicom.read_file(self.path+self.df['patientId'][i]+'.dcm').pixel_array)
        self.image = Variable(torch.FloatTensor(self.image)).cuda()
        self.labels = Variable(torch.FloatTensor(self.df[self.df.columns[1:]].iloc[i].values.astype(float))).cuda()
        return self.image.view(1,3,103,128),self.labels
        
    def __len__(self):
        return len(self.df)

def dl(df_train):
        return DataLoader(df_train,transforms.Compose([transforms.Normalize, transforms.RandomCrop, transforms.RandomSizedCrop]))

## Loss Functions to be used:
'''
1. CAC
2. WGCAC
3. Negative Log Likelihood
4. 
'''


class unlosses:
  
  def __init__(self, in, tar):
    
    
