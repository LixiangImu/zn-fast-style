import torch
import utils
from transformer import TransformerNetwork
import os
from PIL import Image, ImageEnhance
import numpy as np
from torchvision import transforms

def stylize_folder():
    # 设置路径
    model_path = "models/transformer_best.pth"
    content_dir = "VOCdevkit/trainA"
    output_dir = "VOCdevkit/stylized_output"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    transformer = TransformerNetwork().to(device)
    transformer.load_state_dict(torch.load(model_path))
    transformer.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    
    # 处理每张图片
    for img_name in os.listdir(content_dir):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            try:
                # 加载图片
                content_path = os.path.join(content_dir, img_name)
                content_image = Image.open(content_path).convert('RGB')
                original_size = content_image.size
                
                # 转换为tensor
                content_tensor = transform(content_image).unsqueeze(0).to(device)
                
                # 生成风格化图片
                with torch.no_grad():
                    output = transformer(content_tensor)
                
                # 转换为numpy数组并调整范围
                output_image = output[0].cpu().detach().numpy()
                output_image = output_image.transpose(1, 2, 0)
                
                # 确保值在正确范围内
                output_image = np.clip(output_image, 0, 255)
                
                # 增强亮度和对比度
                output_image = output_image * 1.3  # 提高亮度
                output_image = np.clip(output_image, 0, 255).astype(np.uint8)
                
                # 转换为PIL图像进行进一步处理
                output_image = Image.fromarray(output_image)
                
                # 亮度增强
                enhancer = ImageEnhance.Brightness(output_image)
                output_image = enhancer.enhance(1.2)  # 增加亮度
                
                # 对比度增强
                enhancer = ImageEnhance.Contrast(output_image)
                output_image = enhancer.enhance(1.2)  # 增加对比度
                
                # 色彩增强
                enhancer = ImageEnhance.Color(output_image)
                output_image = enhancer.enhance(1.1)  # 轻微增加饱和度
                
                # 锐度增强
                enhancer = ImageEnhance.Sharpness(output_image)
                output_image = enhancer.enhance(1.1)  # 轻微增加锐度
                
                # 调整回原始尺寸
                output_image = output_image.resize(original_size, Image.LANCZOS)
                
                # 保存结果
                output_path = os.path.join(output_dir, img_name)
                output_image.save(output_path, quality=95)
                
                print(f"处理完成: {img_name}")
                
            except Exception as e:
                print(f"处理 {img_name} 时出错: {str(e)}")
                continue

if __name__ == "__main__":
    stylize_folder()