import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from scipy.linalg import hankel
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
from scipy import ndimage
from scipy import optimize
from scipy.stats import norm

def normalization(data):
    data = data/np.max(data)
    return data

def sp_normalization(data):
    data_norm = data / np.max(data)
    return data_norm, np.max(data)

def read(filename):
    len_start = 0
    len_size = 784 # size change
    file = open(filename, encoding='utf-8')
    data_lines = file.readlines()
    file.close
    orign_keys = []
    orign_values = []
    for data_line in data_lines[len_start:len_size]:
        pair = data_line.split()
        key = float(pair[0])
        value = float(pair[1])
        orign_keys.append(key)
        orign_values.append(value)
    orign_keys = np.array(orign_keys)
    orign_values = np.array(orign_values)
    # orign_values = orign_values - min(orign_values)#基线归零
    return orign_keys, orign_values

class dataload(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):#输出所有文件的具体路径
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        self.transforms = transforms
        root_input = root+'/input'
        root_output = root + '/output'
        imgs_input = [os.path.join(root_input, img) for img in os.listdir(root_input)]
        imgs_output = [os.path.join(root_output, img) for img in os.listdir(root_output)]
        self.imgs_output = imgs_output
        self.imgs_input = imgs_input

    def __getitem__(self, index):#给每个文件打标签并且读取
        """
        一次返回一张图片的数据
        """
        k = random.randint(0, len(self.imgs_input)-1)
        j = random.randint(0, len(self.imgs_output)-1)
        keys1, input = read(self.imgs_input[k])
        keys2, output = read(self.imgs_output[j])
        input = normalization(input)
        output = normalization(output)
        a = random.uniform(0.2, 0.4)
        input = input*a+output*(1-a)
        input, max_input = sp_normalization(input)
        output = output*(1-a)/max_input
        print(a)
        return input, output

    def __len__(self):
        return len(self.imgs_input)
