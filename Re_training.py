import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# import torchvision.transforms as transforms
from dataset.Re_training_loading import dataload
from torch.autograd import Variable
from model.Unet import Unet
from model.Simple_FCN import F_CN
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import MultipleLocator
import numpy as np
from torch import Tensor
import torch.nn.functional as F
import random
from scipy import ndimage
from scipy import optimize
from scipy.stats import norm
import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve

def normalization(data):
    # _range = np.max(data) - np.min(data)
    # return (data - np.min(data)) / _range
    data = data/np.max(data)
    return data

def sp_normalization(data):
    data_norm = data / np.max(data)
    return data_norm, np.max(data)

cycler = 0
len_start = 0 # KM=0
len_size = 714 # KM=532
modellr = 1e-1
BATCH_SIZE = 1
EPOCHS = 500
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()
model = torch.load("model_Unet.pth")
optimizer = torch.optim.Adam(model.parameters(), lr=modellr)
dataset_train = dataload('data/re_train', train=True)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 50))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew

def train(model, device, optimizer, epoch):
    global BATCH_SIZE
    model.train()#套路
    sum_loss = 0
    sum_num = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.reshape(1, data.shape[0], data.shape[1]) #data_type: batch_size x embedding_size x text_len
        data = data.permute(1, 0, 2)
        data = torch.as_tensor(data, dtype=torch.float32)
        target = target.reshape(1, target.shape[0], target.shape[1]) # data_type: batch_size x embedding_size x text_len
        target = target.permute(1, 0, 2)
        target = torch.as_tensor(target, dtype=torch.float32)
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss



for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, optimizer, epoch)
    print('epoch:', epoch)
torch.save(model, 'model_Unet_retrain.pth')