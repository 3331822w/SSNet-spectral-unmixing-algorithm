import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from model.Unet import Unet
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from dataset.dataset_Unet import dataload
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
from model.Simple_FCN import F_CN
from scipy import ndimage
from torch.nn import functional as F
import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from scipy import spatial
import time
import random
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier

# 设置全局参数
modellr = 1e-3
EPOCHS = 5
BATCH_SIZE = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
len_start = 0 # KM=0 耦联：516
len_size = 649 # KM=649 抗生素=734 耦联：1010
import scipy.stats as stats

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

# def normalization(data):
#     data = np.array(data)
#     data = data/max(data)
#     return data

# def normalization(data):
#     data = np.array(data)
#     data = data/max(data)
#     return data

def sp_normalization(data):
    # _range = np.max(data) - np.min(data)
    # return (data - np.min(data)) / _range,  np.min(data), _range
    data_norm = data / np.max(data)
    return data_norm, np.max(data)

def normalization_0_1(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def cos_sim(data1,data2):
    cos_sim = 1 - spatial.distance.cosine(data1, data2)
    return cos_sim

def normalization(data):
    data = np.array(data)
    # _range = np.max(data) - np.min(data)
    # return (data - np.min(data)) / _range
    data = data/max(data)
    return data

def derivative(data):
    result = [data[i] - data[i - 1] for i in range(1, len(data))]
    result.insert(0, result[0])
    return result

def num2color(values, cmap):
    import matplotlib as mpl
    """将数值映射为颜色"""
    norm = mpl.colors.Normalize(vmin=np.min(values), vmax=np.max(values))
    cmap = mpl.cm.get_cmap(cmap)
    return [cmap(norm(val)) for val in values]

def vis_tree(filename ,module):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    key, data = read(filename)
    Max = np.max(data)
    Min = np.min(data)
    data = (data-Min) / (Max-Min)
    xx = max(data)##################################要改得地方
    data = [x/xx for x in data]
    color_im = num2color(normalization(module.feature_importances_),'Oranges')
    plt.figure(figsize=(10, 5))
    y_major_locator = MultipleLocator(1)
    ax=plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0, 1.1)
    plt.xlim(key[0], key[len(key)-1])
    plt.xticks(size=20, family='Times New Roman')
    plt.yticks(size=20, family='Times New Roman')
    plt.bar(key, data, key[1]-key[0], color=color_im)#cm即colormap，c=y表示颜色随y变化
    plt.plot(key, data, color='k')
    figpath = filename+'vis_tree.png'
    plt.savefig(figpath, dpi=500)
    plt.close()
    return figpath

def tree_similarity(stand, input_data, filename):
    positive = []
    negative = []
    tot = 0
    for r in range(4000):
        lenth = len(stand)
        stand_new = normalization(stand)
        y_tot = [0]*lenth
        for g in range(0, random.randint(1, 10)):
            a = random.randint(0, lenth)
            b = random.randint(1, lenth//40)
            keys_ = range(lenth)
            c = random.uniform(0, 1)
            gauss = norm(loc=a, scale=b)
            y = gauss.pdf(keys_)
            y = normalization(y)
            y = y*c
            y_tot = y_tot+y
        y_tot = normalization(y_tot)
        stand_new = stand_new+y_tot
        stand_new = normalization(stand_new)
        tot = tot+1
        stand_new = derivative(stand_new)
        y_tot = derivative(y_tot)
        positive = np.concatenate((positive, stand_new), axis=0)
        negative = np.concatenate((negative, y_tot), axis=0)
    positive = np.array(positive).reshape(tot, lenth)
    negative = np.array(negative).reshape(tot, lenth)
    positive_label = np.array([1]*tot)
    negative_label = np.array([0]*tot)
    X_train = np.concatenate((positive, negative), axis=0)
    y_train = np.concatenate((positive_label, negative_label), axis=0)
    rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    none = vis_tree(filename, rf_clf)
    input_data = normalization(input_data)
    input_data = derivative(input_data)
    input_data = np.array(input_data).reshape(1, lenth)
    y_pred = rf_clf.predict_proba(input_data)
    return y_pred

def test(file_name, device,spectrum_stand):
    keys, spectrum = read(file_name)
    spectrum, max_ = sp_normalization(spectrum)
    print(file_name, max_)
    spectrum_raw = spectrum.copy()
    spectrum = list(spectrum)
    start = spectrum[0]
    end = spectrum[len(spectrum) - 1]
    cycler = 0
    for i in range(cycler):
        spectrum.insert(0, 0)
        spectrum.append(0)
    spectrum = np.array(spectrum)
    spectrum = torch.tensor(spectrum)
    spectrum = spectrum.reshape(1, spectrum.shape[0])
    spectrum = spectrum.reshape(1, spectrum.shape[0],
                                spectrum.shape[1])  # data_type: batch_size x embedding_size x text_len
    spectrum = torch.as_tensor(spectrum, dtype=torch.float32)
    spectrum = spectrum.permute(1, 0, 2)
    spectrum = Variable(spectrum).to(device)
    output = model(spectrum)
    pl_output = np.array(output.cpu().detach().numpy()[0, 0, :])
    repeat_times = 0
    print(file_name, '背景相似度_start', stats.pearsonr(normalization(spectrum_stand), normalization(pl_output)))
    # print(file_name, '余弦相似度_背景相似度_start',cos_sim(normalization_0_1(spectrum_stand), normalization_0_1(pl_output)))
    while list(stats.pearsonr(normalization(spectrum_stand), pl_output))[0] > 0.6:
        # pl_output, max_pl = sp_normalization(pl_output)
        pl_output = torch.tensor(pl_output)
        pl_output = pl_output.reshape(1, pl_output.shape[0])
        pl_output = pl_output.reshape(1, pl_output.shape[0],
                                    pl_output.shape[1])  # data_type: batch_size x emb  edding_size x text_len
        pl_output = torch.as_tensor(pl_output, dtype=torch.float32)
        pl_output = pl_output.permute(1, 0, 2)
        pl_output = Variable(pl_output).to(device)
        output_add = model(pl_output)
        pl_output = np.array(output_add.cpu().detach().numpy()[0, 0, :])
        # print(pl_output)
        # pl_output = pl_output/max_pl
        repeat_times = repeat_times + 1
        print(repeat_times)
    # print(file_name, '背景相似度_end', stats.pearsonr(normalization_0_1(spectrum_stand), normalization_0_1(pl_output)))
    # print(file_name, '余弦相似度_背景相似度_start', cos_sim(normalization_0_1(spectrum_stand), normalization_0_1(pl_output)))
    final = (spectrum_raw - pl_output) * max_
    # final = final - airPLS(final)
    plt.plot(keys, (normalization(spectrum_stand)) + 3)
    plt.plot(keys, (normalization(spectrum_raw) + 2))
    plt.plot(keys, normalization(final) + 1)
    plt.plot(keys, pl_output)  # 模型输出背景
    plt.title('Raman spectrum after differece')
    files_tot.append(file_name)
    simi_tree = tree_similarity(normalization(spectrum_stand), normalization(final), file_name)
    simi_tot.append(simi_tree)
    # print(file_name, '差谱相似度', stats.pearsonr(normalization(spectrum_stand), normalization(final)))
    print(file_name,  'SCC相似度', simi_tree)
    # print(file_name, '余弦相似度_差谱相似度_start', cos_sim(normalization_0_1(spectrum_stand), normalization_0_1(final)))
    tou, wei = os.path.split(file_name)
    tou, wei = os.path.splitext(wei)
    plt.savefig('./data/test/' + tou + '.jpg')
    plt.close()
    pl_output = pl_output * max_
    return keys, final, pl_output

#标谱载入
stand = 'C:/Users/52772/OneDrive/2022/娃娃机-CNN/data/train'
files = os.listdir(stand)
file = files[0]
file = os.path.join(stand, file)
file = file.replace('\\', '/')
keys, spectrum_stand = read(file)
#模型载入
criterion = nn.MSELoss()#交叉熵
model = torch.load("model_Unet.pth")#model_Unet_retrain.pth
model.to(DEVICE)
# print(model)
# 读取数据
top_dir = 'C:/Users/52772/OneDrive/2022/娃娃机-CNN/data/test'
files = os.listdir(top_dir)
files_tot = []
simi_tot = []
for i in files:
    file = os.path.join(top_dir, i)
    file = file.replace('\\', '/')
    time_start = time.time()
    keys, values, background = test(file, DEVICE, spectrum_stand)
    time_end = time.time()
    print('Cost time:', time_end - time_start)
    q = 'background_'+i
    i = 'diff_'+i
    diff_filename = os.path.join(top_dir, i)
    background_filename = os.path.join(top_dir,q)
    write(diff_filename, keys, values)
    write(background_filename, keys, background)
write(os.path.join(top_dir, 'similarity.txt'), files_tot, simi_tot)




