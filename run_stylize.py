import torch
from torchvision import transforms
from PIL import Image
import os
from transformer import ColorTransformerNetwork
import glob

def load_image(image_path, size=None):
    image = Image.open(image_path).convert('RGB')
    if size is not None:
        image = image.resize((size, size), Image.LANCZOS)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return transform(image).unsqueeze(0)

def save_image(tensor, save_path):
    # 确保值在[0,1]范围内
    tensor = torch.clamp(tensor, 0, 1)
    
    # 转换为PIL图像
    image = transforms.ToPILImage()(tensor.squeeze(0))
    
    # 保存图像
    image.save(save_path, 'PNG')
    print(f"已保存图像到: {save_path}")

def main():
    # 配置参数
    LOAD_SIZE = 256
    MODEL_PATH = "models/transformer_best.pth"  # 训练好的模型文件路径
    INPUT_DIR = "VOCdevkit/trainA"  # trainA数据集路径
    OUTPUT_DIR = "VOCdevkit/trainA_converted"  # 转换后的图像保存路径
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 检查输入目录
    if not os.path.exists(INPUT_DIR):
        print(f"错误：找不到输入目录 '{INPUT_DIR}'")
        return
    
    # 获取所有输入图像
    input_images = glob.glob(os.path.join(INPUT_DIR, "*.*"))
    if len(input_images) == 0:
        print(f"错误：输入目录 '{INPUT_DIR}' 中没有找到图像文件")
        return
    
    print(f"找到 {len(input_images)} 个图像文件")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 检查并加载模型
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 '{MODEL_PATH}'")
        return
        
    # 加载模型
    transformer = ColorTransformerNetwork().to(device)
    transformer.load_state_dict(torch.load(MODEL_PATH))
    transformer.eval()
    
    # 处理每个图像
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
    processed_count = 0
    
    with torch.no_grad():
        for idx, image_path in enumerate(input_images):
            # 检查文件格式
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in supported_formats:
                print(f"跳过不支持的文件格式: {image_path}")
                continue
            
            try:
                # 加载并处理图像
                content_image = load_image(image_path, LOAD_SIZE).to(device)
                generated_image = transformer(content_image)
                
                # 保存生成的图像
                output_name = os.path.basename(image_path)
                output_path = os.path.join(OUTPUT_DIR, output_name)
                save_image(generated_image, output_path)
                
                processed_count += 1
                
                # 显示进度
                if (idx + 1) % 10 == 0:
                    print(f"已处理: {idx + 1}/{len(input_images)} 图像")
                
                # 每100张图片显示一次颜色分析
                if (idx + 1) % 100 == 0:
                    r, g, b = torch.split(generated_image, 1, dim=1)
                    print(f"\n第 {idx + 1} 张图像的颜色分析:")
                    print(f"R mean: {torch.mean(r):.4f}")
                    print(f"G mean: {torch.mean(g):.4f}")
                    print(f"B mean: {torch.mean(b):.4f}")
                    print(f"G/R ratio: {torch.mean(g)/torch.mean(r):.4f}")
                    print(f"G/B ratio: {torch.mean(g)/torch.mean(b):.4f}")
                
            except Exception as e:
                print(f"处理图像时出错 {image_path}: {str(e)}")
                continue
    
    print(f"\n处理完成！")
    print(f"成功处理 {processed_count} 张图像")
    print(f"转换后的图像保存在: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()