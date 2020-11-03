import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import cv2
import pydicom
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision.models as models

from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

def reindex(df):
    df['Index'] = np.arange(len(df))
    return df.set_index('Index')
def imP(im):
    im = cv2.resize(cv2.equalizeHist(im[200:,:]),(103,128), interpolation = cv2.INTER_AREA)
    return cv2.cvtColor(im*(gaussian_filter(im,sigma=25)>50),cv2.COLOR_GRAY2RGB)
  
class DataLoader(Dataset):  
    
    def __init__(self, df, trans, train = 1):
        
        super(DataLoader,self)
        self.df = df
        self.path = '../input/rsna-pneumonia-detection-challenge/stage_2_train_images/'        
        self.transforms = trans
        self.train = train
        
    def __getitem__(self,i):
        
        self.image = Variable(torch.FloatTensor(imP(pydicom.read_file(self.path+self.df['filename'][i]+'.dcm').pixel_array))).cuda().view(1,3,103,128)
        self.labels = Variable(torch.FloatTensor(self.df[self.df.columns[1:]].iloc[i].values.astype(float))).cuda()
        return self.image.view(1,3,103,128),self.labels
        
    def __len__(self):
        return len(self.df)

def dl(df_train):
        return DataLoader(df_train,transforms.Compose([transforms.Normalize, transforms.RandomCrop, transforms.RandomSizedCrop]))
  
class ResNet50(nn.Module):

    def __init__(self, classCount, isTrained):
     
        super(ResNet50, self).__init__()
        self.net = models.resnet50(pretrained=True)
        kernelCount = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x = F.relu(self.net(x))
        return x

def train(Model,n_epochs):
    
    elosses_mse = []
    elosses_wgcac = []
    optimizer = torch.optim.Adam(Model.parameters(), lr=0.05,weight_decay=0.001) 
    #optimizer = optim.SGD(Model.parameters(), lr=0.05, momentum=0.9)

    criterion = Loss1(0.8,0.8)
    #criterion = nn.MSELoss()
    mse = nn.MSELoss()
    
    for i in range(n_epochs):
    
        print()
        print('Starting epoch: ', i+1)
        print()
        eloss_wgcac = 0
        eloss_mse = 0
        
        for j in range(15):
            
            bloss_wgcac = 0
            bloss_mse = 0

            #print('Batch Number: ',u)
            
            optimizer.zero_grad()
            
            ux = np.random.randint(1,DL.__len__()-1,25)

            for u in ux:

                #print('Batch Index: ', batch_index)
                image,Labels = DL.__getitem__(u)
                output = Model(image)[0]                
                bloss_wgcac += criterion(output, Labels) 
                bloss_mse += mse(output, Labels).item()
            
            avg_bloss_wgcac = bloss_wgcac/25
            avg_bloss_mse = bloss_mse/25
               
            avg_bloss_wgcac.backward()
            optimizer.step()
            
            eloss_wgcac += avg_bloss_wgcac.item()
            eloss_mse += avg_bloss_mse
            
        elosses_mse.append(eloss_mse/15)
        elosses_wgcac.append(eloss_wgcac/15)
        print()
        print('Epoch MSE: ', eloss_mse/15) 
        print('Epoch Wgcac Loss ',eloss_wgcac/15)
        print()
        
    return elosses_wgcac, elosses_mse

def dla(yp,yt):
    return F.mse_loss(yp,yt)*len(yp)
    
def sumc(l):
    u = 0*l[0]
    for i in l:
        u+=i
    return u

def genc(n):
    z = []
    for i in range(n):
        u = torch.zeros(n).cuda()
        u[i]=1
        z.append(u)
    return z

def genw(df_train):
    s = max([len(df_train[df_train[i]==1]) for i in df_train.columns[1:]])
    W = (np.array([len(df_train[df_train[i]==1]) for i in df_train.columns[1:]])/s)**(-1)
    W = W/np.max(W)
    wd = dict(zip(np.arange(len(df_train.columns[1:])),W))
    return wd

class Loss1(nn.Module):
    
    def __init__(self,alf,lam,n=2,sig=1.5,a=1.5):
      
        super(Loss1, self).__init__()
        self.C = genc(n)
        self.alf = alf
        self.lam = lam
        self.sig = sig
        self.a = a
        self.w0 = genw(df1)
        
    def forward(self,yp,yt):
        
        La = dla(yp,self.alf*yt)
        #La = torch.sum(self.a*torch.exp( ((yp - yt)/self.sig)**2 ))
        Lt = torch.log(1+sumc([torch.exp( La - dla(yp,self.alf*i)) for i in self.C ]))
        L = Lt + self.lam*La
        #L = L*self.w0[torch.argmax(yt).item()]
        if(L<1.5):
            L = L*1.8
        return 1.5*L

Model = ResNet50(2,True).cuda() 
wg, m = train(Model,100)      
L = pd.DataFrame()
L['m'] = m
L['wg'] = wg
L.to_csv('../LossTrends/' + model_title + 'Losses.csv')
torch.save(Net1.state_dict(), 'Mod1' + '.pth.tar')
print('Saved at: ', 'Mod1' + '.pth.tar')
