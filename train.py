import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import os
import random
import utils
from transformer import ColorTransformerNetwork
from vgg import VGG16

# 修改配置参数
TRAIN_IMAGE_SIZE = 256
DATASET_PATH = "VOCdevkit/trainB"  # 源域（绿色风格）
CONTENT_PATH = "VOCdevkit/trainA"  # 目标域（蓝色图像）
NUM_EPOCHS = 20  # 增加训练轮数
BATCH_SIZE = 16  # 4090显存充足，可以增加批量大小
CONTENT_WEIGHT = 2.0       # 保持内容权重
STYLE_WEIGHT = 5e1         # 进一步降低风格权重
COLOR_WEIGHT = 2.0         # 调整颜色权重
TARGET_GREEN = 0.6         # 目标绿色通道值
TARGET_RED = 0.4          # 目标红色通道值
TARGET_BLUE = 0.4         # 目标蓝色通道值
ADAM_LR = 1e-4            # 提高学习率

# 添加颜色转换相关的权重
BLUE_TO_GREEN_WEIGHT = 10.0  # 蓝到绿的转换权重
COLOR_PRESERVE_WEIGHT = 0.5  # 其他颜色保持权重

SAVE_MODEL_PATH = "models/"
SAVE_IMAGE_PATH = "images/out/"
SAVE_MODEL_EVERY = 20

# 数据集类
class StyleContentDataset(data.Dataset):
    def __init__(self, content_dir, style_dir, image_size):
        super(StyleContentDataset, self).__init__()
        self.content_paths = [os.path.join(content_dir, x) for x in os.listdir(content_dir)]
        self.style_paths = [os.path.join(style_dir, x) for x in os.listdir(style_dir)]
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # 这会自动将值范围转换到[0,1]
        ])
        
    def __len__(self):
        return len(self.content_paths)
    
    def __getitem__(self, index):
        style_path = random.choice(self.style_paths)
        content_path = self.content_paths[index]
        
        content_img = Image.open(content_path).convert('RGB')
        style_img = Image.open(style_path).convert('RGB')
        
        content_tensor = self.transform(content_img)
        style_tensor = self.transform(style_img)
        
        return content_tensor, style_tensor

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    os.makedirs(SAVE_IMAGE_PATH, exist_ok=True)
    
    # 加载网络
    transformer = ColorTransformerNetwork().to(device)
    vgg = VGG16().to(device)
    
    # 优化器和学习率调度器
    optimizer = optim.Adam(transformer.parameters(), lr=ADAM_LR, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[8, 12],
        gamma=0.85
    )
    
    # 数据加载器
    dataset = StyleContentDataset(CONTENT_PATH, DATASET_PATH, TRAIN_IMAGE_SIZE)
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 用于验证的固定内容图片
    val_content, _ = dataset[0]
    val_content = val_content.unsqueeze(0).to(device)
    
    print("开始训练...")
    best_loss = float('inf')
    
    def style_loss_fn(features, style_features, style_weights=None):
        if style_weights is None:
            # 为不同层设置不同的权重
            style_weights = {
                'relu1_2': 1.0,
                'relu2_2': 1.0,
                'relu3_3': 1.0,
                'relu4_3': 1.0
            }
        
        style_loss = 0
        for key in style_features.keys():
            if key in style_weights:
                # 计算 Gram 矩阵
                g1 = utils.gram(features[key])
                g2 = utils.gram(style_features[key])
                style_loss += style_weights[key] * nn.MSELoss()(g1, g2)
        return style_loss
    
    def color_transfer_loss(source, target, content):
        # 分离RGB通道
        s_r, s_g, s_b = torch.split(source, 1, dim=1)
        t_r, t_g, t_b = torch.split(target, 1, dim=1)
        
        # 计算目标颜色分布
        target_r_mean = torch.mean(t_r)
        target_g_mean = torch.mean(t_g)
        target_b_mean = torch.mean(t_b)
        
        # 颜色目标损失
        color_target_loss = (
            torch.abs(torch.mean(s_r) - TARGET_RED) +
            torch.abs(torch.mean(s_g) - TARGET_GREEN) +
            torch.abs(torch.mean(s_b) - TARGET_BLUE)
        ) * 2.0
        
        # 颜色比例损失
        color_ratio_loss = (
            torch.abs(torch.mean(s_g) / (torch.mean(s_r) + 1e-6) - 1.5) +  # 绿色应该比红色多50%
            torch.abs(torch.mean(s_g) / (torch.mean(s_b) + 1e-6) - 1.5)    # 绿色应该比蓝色多50%
        ) * 1.5
        
        # 亮度和对比度损失
        brightness_loss = torch.abs(
            torch.mean(source) - 0.5  # 保持适中的亮度
        ) * 2.0
        
        contrast_loss = torch.abs(
            torch.std(source) - torch.std(content)  # 保持原始对比度
        ) * 2.0
        
        # 内容保留损失
        content_preserve = torch.mean(torch.abs(source - content)) * 2.0
        
        # 值域约束
        value_range_loss = (
            torch.mean(torch.relu(0.2 - source)) +  # 最小值不低于0.2
            torch.mean(torch.relu(source - 0.8))    # 最大值不超过0.8
        ) * 3.0
        
        # 局部结构保留
        def local_structure(x, kernel_size=3):
            pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size//2)
            return pool(x)
        
        structure_loss = torch.mean(torch.abs(
            local_structure(source) - local_structure(content)
        )) * 2.0
        
        # 组合损失
        total_structure_loss = (
            content_preserve +
            structure_loss +
            brightness_loss +
            contrast_loss +
            value_range_loss
        )
        
        total_color_loss = color_target_loss + color_ratio_loss
        
        return total_structure_loss, total_color_loss
    
    for epoch in range(NUM_EPOCHS):
        transformer.train()
        epoch_loss = 0.0  # 初始化epoch损失
        
        for batch_id, (content_batch, style_batch) in enumerate(dataloader):
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)
            
            # 生成图像并确保值域
            generated_batch = transformer(content_batch)
            generated_batch = torch.clamp(generated_batch, 0.2, 0.8)
            
            # 提取特征
            content_features = vgg(content_batch)
            style_features = vgg(style_batch)
            generated_features = vgg(generated_batch)
            
            # 计算内容损失 (确保是标量)
            content_loss = CONTENT_WEIGHT * torch.mean(
                nn.MSELoss()(
                    generated_features['relu2_2'],
                    content_features['relu2_2']
                )
            )
            
            # 计算风格损失 (确保是标量)
            style_loss = 0
            for key in style_features.keys():
                style_loss += torch.mean(nn.MSELoss()(
                    utils.gram(generated_features[key]),
                    utils.gram(style_features[key])
                ))
            style_loss = style_loss * STYLE_WEIGHT
            
            # 计算损失
            content_loss = content_loss * CONTENT_WEIGHT
            style_loss = style_loss * STYLE_WEIGHT
            structure_loss, color_loss = color_transfer_loss(
                generated_batch, style_batch, content_batch
            )
            
            # 动态调整权重
            if epoch < NUM_EPOCHS // 3:
                # 前期更注重内容保留
                color_loss = color_loss * 0.5
            elif epoch > NUM_EPOCHS * 2 // 3:
                # 后期加强颜色转换
                color_loss = color_loss * 1.2
            
            total_loss = content_loss + style_loss + structure_loss + color_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 累积epoch损失
            epoch_loss += total_loss.item()
            
            # 打印详细信息
            if batch_id % 5 == 0:
                r, g, b = torch.split(generated_batch, 1, dim=1)
                print(f"Epoch: {epoch+1}/{NUM_EPOCHS} Batch: {batch_id}/{len(dataloader)}")
                print(f"R mean: {torch.mean(r):.4f}, G mean: {torch.mean(g):.4f}, B mean: {torch.mean(b):.4f}")
                print(f"G/R ratio: {torch.mean(g)/torch.mean(r):.4f}, G/B ratio: {torch.mean(g)/torch.mean(b):.4f}")
                print(f"Value range: {torch.min(generated_batch):.4f} to {torch.max(generated_batch):.4f}")
                print(f"Losses - Total: {total_loss.item():.4f}, Content: {content_loss.item():.4f}, "
                      f"Style: {style_loss.item():.4f}, Color: {color_loss.item():.4f}\n")
            
            # 保存模型和示例图片
            if batch_id % SAVE_MODEL_EVERY == 0:
                transformer.eval()  # 切换到评估模式
                with torch.no_grad():
                    # 保存模型
                    torch.save(
                        transformer.state_dict(), 
                        os.path.join(SAVE_MODEL_PATH, f'transformer_epoch_{epoch}_batch_{batch_id}.pth')
                    )
                    
                    # 保存示例图片
                    generated_image = utils.ttoi(generated_batch[0].detach())
                    print(f"Generated image range: {generated_image.min():.2f} to {generated_image.max():.2f}")
                    utils.saveimg(generated_image, os.path.join(SAVE_IMAGE_PATH, f'sample_epoch_{epoch}_batch_{batch_id}.jpg'))
                transformer.train()  # 切换回训练模式
        
        # 计算平均损失
        valid_batches = len(dataloader)
        epoch_loss = epoch_loss / valid_batches if valid_batches > 0 else float('inf')
        print(f"Epoch {epoch+1} 平均损失: {epoch_loss:.4f}")
        
        # 学习率调整
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"学习率调整为: {current_lr}")
        
        # 验证
        transformer.eval()  # 切换到评估模式
        with torch.no_grad():
            val_generated = transformer(val_content)
            generated_image = utils.ttoi(val_generated[0].detach())
            print(f"Validation image range: {generated_image.min():.2f} to {generated_image.max():.2f}")
            utils.saveimg(
                generated_image,
                os.path.join(SAVE_IMAGE_PATH, f'validation_epoch_{epoch}.jpg')
            )
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(
                    transformer.state_dict(),
                    os.path.join(SAVE_MODEL_PATH, 'transformer_best.pth')
                )
        
        transformer.train()  # 切换回训练模式
    
    print("训练完成!")
    print(f"最佳损失值: {best_loss:.4f}")

if __name__ == "__main__":
    main()
