from __future__ import print_function, division
import torch.nn as nn
import torch

# 定义SE注意力机制的类
class se_block(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接下降通道的倍数
    def __init__(self, in_channel, ratio=4):
        # 继承父类初始化方法
        super(se_block, self).__init__()

        # 属性分配
        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        # 第一个全连接层将特征图的通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # relu激活
        self.relu = nn.ReLU()
        # 第二个全连接层恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
        # sigmoid激活函数，将权值归一化到0-1
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)

    # 前向传播
    def forward(self, inputs):  # inputs 代表输入特征图

        # 获取输入特征图的shape
        b, c, l = inputs.shape
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # 维度调整 [b,c,1]==>[b,c]
        x = x.view([b, c])

        # 第一个全连接下降通道 [b,c]==>[b,c//4]
        x = self.fc1(x)
        x = self.relu(x)
        # 第二个全连接上升通道 [b,c//4]==>[b,c]
        x = self.fc2(x)
        # 对通道权重归一化处理
        x = self.sigmoid(x)

        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1])

        # 将输入特征图和通道权重相乘
        outputs = x * inputs
        return outputs


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=15, stride=1, padding='same', bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            # nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=1, padding='same', bias=True),
            # nn.BatchNorm1d(out_ch),
            # nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, size):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(size=size[0]),  # 使用明确的目标大小
            nn.Conv1d(in_ch, out_ch, kernel_size=15, stride=1, padding='same', bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, size=2500):
        super(U_Net, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]

        # 计算每一层的大小
        sizes = []
        current_size = size
        sizes.append([current_size])

        # 计算下采样每层的大小
        for i in range(5):  # 5次下采样
            current_size = current_size // 2
            sizes.append([current_size])

        self.Maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Maxpool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.5)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        self.Conv6 = conv_block(filters[4], filters[5])

        # 创建上采样块
        self.Up6 = up_conv(filters[5], filters[4], sizes[4])
        self.Up_conv6 = conv_block(filters[5], filters[4])
        self.Se_Bk6 = se_block(filters[5])

        self.Up5 = up_conv(filters[4], filters[3], sizes[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])
        self.Se_Bk5 = se_block(filters[4])
        self.CBAM5 = CBAMLayer(filters[4])

        self.Up4 = up_conv(filters[3], filters[2], sizes[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])
        self.Se_Bk4 = se_block(filters[3])
        self.CBAM4 = CBAMLayer(filters[3])

        self.Up3 = up_conv(filters[2], filters[1], sizes[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])
        self.Se_Bk3 = se_block(filters[2])
        self.CBAM3 = CBAMLayer(filters[2])

        self.Up2 = up_conv(filters[1], filters[0], sizes[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])
        self.Se_Bk2 = se_block(filters[1])
        self.CBAM2 = CBAMLayer(filters[1])

        self.Conv = nn.Conv1d(filters[0], out_ch, kernel_size=1, stride=1, padding='same')

        # self.active = nn.LeakyReLU
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.uniform_(m.weight, a=0, b=1)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        e5 = self.dropout(e5)

        e6 = self.Maxpool5(e5)
        e6 = self.Conv6(e6)

        # 使用nn.functional.interpolate明确调整尺寸，确保上采样后的特征图尺寸与下采样阶段对应层的特征图尺寸匹配
        d6 = self.Up6(e6)

        # 确保d6的长度与e5相同
        if d6.shape[2] != e5.shape[2]:
            d6 = nn.functional.interpolate(d6, size=e5.shape[2])

        d6 = torch.cat((e5, d6), dim=1)
        d6 = self.Se_Bk6(d6)
        d6 = self.Up_conv6(d6)

        d5 = self.Up5(d6)
        # 确保d5的长度与e4相同
        if d5.shape[2] != e4.shape[2]:
            d5 = nn.functional.interpolate(d5, size=e4.shape[2])

        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Se_Bk5(d5)
        # d5 = self.CBAM5(d5)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        # 确保d4的长度与e3相同
        if d4.shape[2] != e3.shape[2]:
            d4 = nn.functional.interpolate(d4, size=e3.shape[2])

        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Se_Bk4(d4)
        # d4 = self.CBAM4(d4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        # 确保d3的长度与e2相同
        if d3.shape[2] != e2.shape[2]:
            d3 = nn.functional.interpolate(d3, size=e2.shape[2])

        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Se_Bk3(d3)
        # d3 = self.CBAM3(d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        # 确保d2的长度与e1相同
        if d2.shape[2] != e1.shape[2]:
            d2 = nn.functional.interpolate(d2, size=e1.shape[2])

        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Se_Bk2(d2)
        # d2 = self.CBAM2(d2)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out


def Unet(size):
    return U_Net(1, 1, size)