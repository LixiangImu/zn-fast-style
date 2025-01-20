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
    def __init__(self):
        super(ColorAdjustment, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        
    def forward(self, x, target='green'):
        # 根据目标颜色调整权重
        if target == 'green':
            color_weights = torch.tensor([0.8, 1.2, 0.8]).view(1, 3, 1, 1).to(x.device)
            self.bias.data = torch.tensor([0.0, 0.1, 0.0]).view(1, 3, 1, 1).to(x.device)
        
        x = self.conv(x) * color_weights + self.bias
        return torch.clamp(x, 0, 1)

class ColorTransformerNetwork(nn.Module):
    def __init__(self):
        super(ColorTransformerNetwork, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 颜色转换模块
        self.color_transform = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ColorAttention(256)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
        # 添加全局颜色调整模块
        self.color_adjust = ColorAdjustment()
        
    def forward(self, x):
        enc = self.encoder(x)
        trans = self.color_transform(enc)
        dec = self.decoder(trans)
        out = self.color_adjust(dec, target='green')
        return out