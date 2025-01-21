import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 添加SE注意力模块
        self.se = SELayer(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)  # 应用SE注意力
        return out + residual

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ColorAttention(nn.Module):
    def __init__(self, channels):
        super(ColorAttention, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 获取输入特征图的尺寸
        b, c, h, w = x.size()
        
        # 生成颜色注意力权重
        attention = self.conv(x)  # [b, c, h, w]
        attention = self.sigmoid(attention)  # [b, c, h, w]
        
        # 直接应用注意力权重
        return x * attention

class ColorAdjustment(nn.Module):
    def __init__(self, reference_weights=None, reference_bias=None):
        super(ColorAdjustment, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        
        # 保存参考权重和偏置作为类属性
        self.reference_weights = reference_weights
        self.reference_bias = reference_bias
        
        # 固定的颜色转换矩阵，专注于蓝到绿的转换
        init_weights = torch.tensor([
            [0.85, 0.10, 0.05],  # R通道基本保持
            [0.15, 0.75, 0.35],  # G通道增强，部分来自B
            [0.05, 0.10, 0.60]   # B通道降低
        ]).view(3, 3, 1, 1)
        
        self.conv.weight = nn.Parameter(init_weights)  # 使用nn.Parameter包装
        self.bias.data = torch.tensor([[0.0], [0.15], [-0.15]]).view(1, 3, 1, 1)
        
        # 禁用权重和偏置的梯度更新
        self.conv.weight.requires_grad = False
        self.bias.requires_grad = False
    
    def forward(self, x):
        # 直接应用颜色转换，不使用constrain_parameters
        color_adjusted = self.conv(x) + self.bias
        # 使用固定的混合比例
        return torch.clamp(0.3 * x + 0.7 * color_adjusted, 0, 1)

class ColorTransformerNetwork(nn.Module):
    def __init__(self, reference_weights=None, reference_bias=None):
        super(ColorTransformerNetwork, self).__init__()
        
        # 增加初始特征通道数
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 减少ResidualBlock数量，避免过度变形
        self.color_transform = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ColorAttention(128)
        )
        
        # 解码器对应修改
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 3, kernel_size=7, padding=3),
            nn.Sigmoid()  # 改用Sigmoid替代Tanh
        )
        
        self.color_adjust = ColorAdjustment(reference_weights, reference_bias)
        
    def forward(self, x):
        # 添加全局残差连接
        enc = self.encoder(x)
        trans = self.color_transform(enc)
        dec = self.decoder(trans)
        out = self.color_adjust(dec)
        return out