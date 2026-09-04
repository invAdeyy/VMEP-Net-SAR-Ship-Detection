import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmdet.models.necks.fpn import FPN


class HFSFModule(nn.Module):
    """
    高频散射聚焦模块 (High-Frequency Scattering Focus Module)
    """

    def __init__(self, in_channels):
        super().__init__()
        # 1. 3x3 深度可分离卷积 (Depthwise Conv)
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        # 2. 1x1 卷积用于生成掩码
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        # 3. Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()
        # 4. 零初始化的可学习参数 alpha (Zero-Initialization)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 局部高通滤波器解耦高频特征 D: D = Conv_dw(X) - X
        dw_out = self.dwconv(x)
        d = dw_out - x

        # 散射感知门控掩码 A: A = Sigmoid(Conv_1x1(D))
        a = self.sigmoid(self.conv1x1(d))

        # 特征调制与残差注入: Y = X + alpha * (X * A)
        y = x + self.alpha * (x * a)
        return y


@MODELS.register_module()
class HFSFEnhancedFPN(FPN):
    """
    集成了 HFSF 模块的增强型 FPN (Enhanced FPN with HFSF)
    """

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, **kwargs):
        # 初始化标准的 FPN
        super().__init__(in_channels, out_channels, num_outs, start_level, end_level, **kwargs)

        # 为传入的每一个特征层级实例化一个 HFSF 模块
        self.hfsf_modules = nn.ModuleList()
        for c in in_channels:
            self.hfsf_modules.append(HFSFModule(c))

    def forward(self, inputs):
        """
        前向传播
        """
        # 1. 骨干网络的特征先经过 HFSF 模块提纯
        hfsf_inputs = []
        for i, x in enumerate(inputs):
            hfsf_inputs.append(self.hfsf_modules[i](x))

        # 2. 将提纯后的特征送入标准的 FPN 进行自顶向下的融合
        return super().forward(tuple(hfsf_inputs))