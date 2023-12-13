import torch
import torch.nn as nn
import torch.nn.functional as F
# 这个文件定义了常用的模块


# Hardswish激活函数的自定义实现
class Hardswish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.relu6(x + 3.0) / 6.0


# Conv类定义了一个包含卷积、批归一化和激活函数的模块
class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, leaky=True):
        super(Conv, self).__init__()
        # 卷积层包括卷积、批归一化和激活函数（默认为LeakyReLU）
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True) if leaky else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


# SAM类定义了一个空间注意力模块
class SAM(nn.Module):
    """ Parallel CBAM """
    def __init__(self, in_ch):
        super(SAM, self).__init__()
        # 空间注意力模块，利用1x1卷积和Sigmoid函数
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.Sigmoid()           
        )

    def forward(self, x):
        # Spatial Attention Module，对输入应用空间注意力
        x_attention = self.conv(x)

        return x * x_attention


# SPP类定义了一个空间金字塔池化模块
class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        # 空间金字塔池化模块，利用不同大小的池化核池化输入
        x_1 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = torch.nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = torch.nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)

        return x


# Copy from yolov5
# Bottleneck类定义了一个标准的瓶颈块，包含了残差连接
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c_, c2, k=3, p=1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# Copy from yolov5
 # CSP瓶颈块定义，利用了跨阶段部分网络结构
class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = nn.Conv2d(c1, c_, kernel_size=1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, kernel_size=1, bias=False)
        self.cv4 = Conv(2 * c_, c2, k=1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
