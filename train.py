import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import os
import random
import utils
from transformer import TransformerNetwork
from vgg import VGG16

# 修改配置参数
TRAIN_IMAGE_SIZE = 256
DATASET_PATH = "VOCdevkit/trainB"
CONTENT_PATH = "VOCdevkit/trainA"
NUM_EPOCHS = 8        # 增加训练轮数
BATCH_SIZE = 8
CONTENT_WEIGHT = 0.8    # 降低内容权重
STYLE_WEIGHT = 5e5      # 显著增加风格权重
ADAM_LR = 2e-4         # 增大学习率
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
    transformer = TransformerNetwork().to(device)
    vgg = VGG16().to(device)
    
    # 优化器和学习率调度器
    optimizer = optim.Adam(transformer.parameters(), lr=ADAM_LR, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[3, 5, 7],  # 在这些epoch降低学习率
        gamma=0.7              # 更大的学习率下降幅度
    )
    
    # 数据加载器
    dataset = StyleContentDataset(CONTENT_PATH, DATASET_PATH, TRAIN_IMAGE_SIZE)
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 用于验证的固定内容图片
    val_content, _ = dataset[0]
    val_content = val_content.unsqueeze(0).to(device)
    
    print("开始训练...")
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        transformer.train()
        
        # 风格权重随训练进行而增加
        current_style_weight = STYLE_WEIGHT * (1 + epoch * 0.3)  # 增加权重增长率
        
        for batch_id, (content_batch, style_batch) in enumerate(dataloader):
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)
            
            # 检查输入值范围
            if batch_id == 0:
                print(f"Content batch range: {content_batch.min():.2f} to {content_batch.max():.2f}")
                print(f"Style batch range: {style_batch.min():.2f} to {style_batch.max():.2f}")
            
            # 生成图像
            generated_batch = transformer(content_batch)
            
            # 计算特征
            content_features = vgg(content_batch)
            style_features = vgg(style_batch)
            generated_features = vgg(generated_batch)
            
            # 内容损失
            content_loss = CONTENT_WEIGHT * nn.MSELoss()(
                generated_features['relu2_2'], 
                content_features['relu2_2']
            )
            
            # 风格损失
            style_loss = 0
            for key in style_features.keys():
                style_loss += nn.MSELoss()(
                    utils.gram(generated_features[key]),
                    utils.gram(style_features[key])
                )
            
            # 计算损失时使用更激进的权重
            content_loss = content_loss * CONTENT_WEIGHT
            style_loss = style_loss * current_style_weight
            
            # 确保style loss不会太小
            if style_loss.item() < content_loss.item() * 10:
                style_loss = style_loss * 1.5
                
            total_loss = content_loss + style_loss
            
            # 检查损失值
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("警告：检测到无效损失值，跳过此批次")
                continue
                
            epoch_loss += total_loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 使用梯度裁剪
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 打印详细的损失信息
            if batch_id % 5 == 0:
                print(f"Epoch: {epoch+1}/{NUM_EPOCHS} "
                      f"Batch: {batch_id}/{len(dataloader)} "
                      f"Total Loss: {total_loss.item():.4f} "
                      f"Content Loss: {content_loss.item():.4f} "
                      f"Style Loss: {style_loss.item():.4f}")
            
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
