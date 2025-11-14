import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# import torchvision.transforms as transforms
from dataset.dataset_Unet import dataload
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

def read(filename):
    file = open(filename, encoding='utf-8')
    data_lines = file.readlines()
    file.close
    orign_keys = []
    orign_values = []
    for data_line in data_lines:
        pair = data_line.split()
        key = float(pair[0])
        value = float(pair[1])
        orign_keys.append(key)
        orign_values.append(value)
    orign_keys = np.array(orign_keys)
    orign_values = np.array(orign_values)
    orign_values = orign_values - min(orign_values)#基线归零
    return orign_keys, orign_values

def write(filename, files, values):
    file = open(filename, 'w')
    for k, v in zip(files, values):
        file.write(str(k) + " " + str(v) + "\n")
    file.close()

def add_composite_baseline(spectrum):
    """添加复合基线漂移：快速下降的指数背景 + 宽高斯峰隆起的背景"""
    n = len(spectrum)
    x = np.linspace(0, 1, n)
    max_spectrum = np.max(spectrum)
    wavelength = np.linspace(0, n, n)

    # 1. 快速下降的指数背景（模拟荧光背景）
    exp_amplitude = np.random.uniform(0.6, 0.9)  # 高振幅
    exp_decay = np.random.uniform(5.0, 10.0)  # 快速衰减
    exp_offset = np.random.uniform(0.05, 0.15)
    exponential_baseline = exp_amplitude * np.exp(-exp_decay * x)

    # 2. 宽高斯峰隆起的背景（模拟仪器漂移或基质背景）
    # 高斯峰位置（偏向右侧）
    gauss_center = np.random.uniform(0.6, 0.9) * n
    # 高斯峰宽度（很宽，覆盖大部分光谱）
    gauss_width = np.random.uniform(0.2, 0.4) * n
    # 高斯峰高度（较低，形成微微隆起）
    gauss_height = np.random.uniform(0.1, 0.3)

    # 创建宽高斯峰
    gaussian_baseline = gauss_height * norm.pdf(wavelength, loc=gauss_center, scale=gauss_width)
    # 归一化并调整形状
    gaussian_baseline = gaussian_baseline / np.max(gaussian_baseline) * gauss_height
    # 确保最小值在左侧
    gaussian_baseline = gaussian_baseline - np.min(gaussian_baseline)

    # 3. 组合两种基线
    composite_baseline = exponential_baseline

    # 4. 调整基线幅度（原始光谱最大值的50%-90%）
    baseline_amplitude = np.random.uniform(3, 9) * max_spectrum
    composite_baseline = composite_baseline * baseline_amplitude

    spectrum = spectrum + composite_baseline
    spectrum, max_ = sp_normalization(spectrum)
    composite_baseline = composite_baseline/max_
    return spectrum, composite_baseline

def add_Gau_peaks(spectrum, a, b, c):   #a=10,b=40,c=0.5
    spectrum = normalization(spectrum)
    # print(max(spectrum))
    lenth = len(spectrum)
    y_tot = [0] * lenth
    for g in range(random.randint(10, a)):
        b1 = random.randint(2, lenth // b)
        # print(b1)
        a1 = random.randint(0, lenth)
        keys_ = range(lenth)
        c1 = random.uniform(0.1, c)
        gauss = norm(loc=a1, scale=b1)
        y = gauss.pdf(keys_)
        y = normalization(y)
        y = y * c1
        y_tot = y_tot + y    #将生成的随机高斯峰叠加到y_tot上
    y_tot = np.array(y_tot)
    spectrum_sum = spectrum + y_tot
    spectrum_sum, max_sum = sp_normalization(spectrum_sum)
    y_tot = y_tot / max_sum
    return spectrum_sum, y_tot

def normalization(data):
    data = data/np.max(data)
    return data

def sp_normalization(data):
    data_norm = data / np.max(data)
    return data_norm, np.max(data)

def derivative(data):
    result = [data[i] - data[i - 1] for i in range(1, len(data))]
    result.insert(0, result[0])
    return np.array(result)

def de_derivative(data):
    result = [sum(data[:i+1]) for i in range(len(data))]
    return np.array(result)

# 定义训练过程
# 设置全局参数
len_start = 0 # KM=0 耦联：516
len_size = 649 # KM=649，抗生素=734 PAHs=693 耦联：1010
modellr = 1e-2
BATCH_SIZE = 1
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
top_dir = ('data/train')
files = os.listdir(top_dir)
# spectrum = normalization(spectrum)*2
# dataset_train = dataload('data/train', train=True, spectrum=spectrum)
# 读取数据

# 导入数据
# train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE)
# 实例化模型并且移动到GPU
criterion = nn.MSELoss()#损失函数MSELoss
# model = Unet(len(spectrum))#模型导入
model = Unet(len_size-len_start)
# model = torch.load("model_Unet.pth")
model.to(DEVICE)
# 选择简单暴力的Adam优化器，学习率调低s
optimizer = torch.optim.Adam(model.parameters(), lr=modellr)#求解
# optimizer = torch.optim.SGD(model.parameters(), lr=modellr, weight_decay=10)#求解


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 25))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


def train(model, device, optimizer, epoch, spectrum, write_loss):
    model.train()#套路
    sum_loss = 0
    sum_num = 0
    GS_peak_intensity = 1
    # total_num = len(train_loader.dataset)#train_loader.dataset拥有两个张量，前为数据，后为标签，可以直接等于复制
    for i in range(1000):
        sum_num = sum_num+1
        spectrum_raw = spectrum
        data, backgroud = add_Gau_peaks(spectrum_raw, 25, 20, GS_peak_intensity)
        # data, backgroud = add_composite_baseline(spectrum)
        # write('G:/OneDrive/2022/娃娃机-CNN/data/nayixia.txt', data, backgroud)
        data = torch.as_tensor(data, dtype=torch.float32)
        data = data.reshape(1, data.shape[0])
        data = data.reshape(1, data.shape[0], data.shape[1])#data_type: batch_size x embedding_size x text_len
        data = data.permute(1, 0, 2)
        # print(data.shape)
        target = backgroud.copy()
        target = torch.as_tensor(target, dtype=torch.float32)
        target = target.reshape(1, target.shape[0])
        target = target.reshape(1, target.shape[0], target.shape[1])  # data_type: batch_size x embedding_size x text_len
        target = target.permute(1, 0, 2)
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss
    ave_loss = sum_loss / sum_num
    print('epoch:{},loss:{}, enhance_num:{}'.format(epoch, ave_loss, sum_num))
    write_loss.append(ave_loss)

# 验证过程
def val(model, device, spectrum):
    model.eval()#套路
    test_loss = 0
    # total_num = len(test_loader.dataset)
    # print(total_num, len(test_loader))
    GS_peak_intensity = 1
    with torch.no_grad():
        spectrum_raw = spectrum.copy()
        spectrum, backgroud = add_Gau_peaks(spectrum, 25, 20, GS_peak_intensity)
        # spectrum, backgroud = add_composite_baseline(spectrum)
        spectrum = torch.as_tensor(spectrum, dtype=torch.float32)
        spectrum = spectrum.reshape(1, spectrum.shape[0])
        spectrum = spectrum.reshape(1, spectrum.shape[0], spectrum.shape[1])  # data_type: batch_size x embedding_size x text_len
        spectrum = spectrum.permute(1, 0, 2)
        spectrum = Variable(spectrum).to(device)
        output = model(spectrum)
        pl_output = np.array(output.cpu().numpy()[0, 0, :])
        pl_spectrum = np.array(spectrum.cpu().numpy()[0, 0, :])
        plt.plot(keys, (normalization(spectrum_raw) + 3))
        plt.plot(keys, (pl_spectrum + 1))
        plt.plot(keys, (normalization(pl_spectrum-pl_output)+2)) # 这里设置颜色为红色，也可以设置其他颜色
        plt.plot(keys, pl_output)
        plt.plot(keys, backgroud)
        write('add_peaks.txt', pl_spectrum, pl_output)
        # write(os.path.join('G:\\OneDrive\\2022\\娃娃机-CNN', 'Compared with stimula spectral.txt'), backgroud, pl_output)
        plt.title('Raman spectrum after differece')
        plt.savefig('./differece' + '+epoch'+str(epoch) + '.jpg')
        plt.close()
        loss = criterion(output, spectrum)
        print_loss = loss.data.item()
        test_loss += print_loss
    # avgloss = test_loss / len(test_loader)
    # print('\nVal set: Average loss: {:.4f}\n'.format(
    #     avgloss, len(test_loader.dataset)))]

####data augmentation####
def random_add(path, len_size):
    files = os.listdir(path)
    aug_spectrum = len_size*[0]
    for file in files:
        file = os.path.join(path, file)
        file = file.replace('\\', '/')
        keys, spectrum = read(file)
        a = random.uniform(0, 1)
        # print(file, a)
        aug_spectrum = aug_spectrum+spectrum * a
    aug_spectrum = normalization(aug_spectrum)
    return keys, aug_spectrum

write_loss = []
for epoch in range(1, EPOCHS + 1):
    random_ = random.randint(0, len(files) - 1)
    file = files[random_]  # 随机读取一个训练集的文件
    print(file)
    file = os.path.join(top_dir, file)
    file = file.replace('\\', '/')
    keys, spectrum = read(file)
    spectrum = list(spectrum)
    spectrum = np.array(spectrum)
    adjust_learning_rate(optimizer, epoch)
    train(model, DEVICE, optimizer, epoch, spectrum, write_loss)
    val(model, DEVICE, spectrum)
torch.save(model, 'model_Unet.pth')
np.savetxt('./Loss_converge.txt', write_loss)