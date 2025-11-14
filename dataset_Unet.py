# coding:utf8
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

def distance_Reciprocal(a,b):
    d_all=a**2+b**2
    d=d_all**0.5
    d=1/d
    return d


def closest(mylist, Number):
    answer = []
    for i in mylist:
        answer.append(abs(Number-i))
    return answer.index(min(answer))

def located_normalization(data, g, h):
    import  numpy as np
    max1 = max(data[g:h])
    data_new = [g / max1 for g in data]
    return np.array(data_new)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def derivative(data):
    result = [data[i] - data[i - 1] for i in range(1, len(data))]
    result.insert(0, result[0])
    result = np.array(result)
    return result

def f_1(x, A, B):
    return A * x + B

def add_Gau_peaks(spectrum, a, b, c):#a=10,b=40,c=0.5
    spectrum = normalization(spectrum)
    lenth = len(spectrum)
    y_tot = [0] * lenth
    for g in range(random.randint(1, a)):
        a1 = random.randint(0, lenth)
        b1 = random.randint(1, lenth // b)
        keys_ = range(lenth)
        c1 = random.uniform(0, c)
        gauss = norm(loc=a1, scale=b1)
        y = gauss.pdf(keys_)
        y = normalization(y)
        y = y * c1
        y_tot = y_tot + y
    # y_tot = normalization(y_tot)#考虑是否需要额外归一化，当c=1时
    spectrum = spectrum + y_tot
    spectrum = normalization(spectrum)
    return spectrum

def random_transform(spectrum):
    spectrum1 = add_Gau_peaks(spectrum, 10, 40, 0.3)
    spectrum2 = add_Gau_peaks(spectrum, 10, 40, 0.3)
    return spectrum1, spectrum2

class dataload(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):#输出所有文件的具体路径
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        self.transforms = transforms
        if os.listdir(root)[0].endswith(".txt"):#文件中无子文件夹
            imgs = [os.path.join(root, img) for img in os.listdir(root)]
        else:#文件中有子文件夹
            imgs=[]
            dirs = [os.path.join(root, dirs) for dirs in os.listdir(root)]
            for l in dirs:
                imgs = imgs + [os.path.join(l, imgs) for imgs in os.listdir(l)]
        for k in range(len(imgs)):
            imgs[k] = imgs[k].replace('\\','/')
        self.imgs = imgs

    def __getitem__(self, index):#给每个文件打标签并且读取
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        spectrum = np.loadtxt(img_path, skiprows=0)
        data, target = random_transform(spectrum)
        return data, target

    def __len__(self):
        return len(self.imgs)
