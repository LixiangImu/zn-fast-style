import os
import torch
from torchvision import transforms
from PIL import Image
import utils
from transformer import TransformerNetwork

def batch_stylize():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载训练好的模型
    transform_net = TransformerNetwork().to(device)
    transform_net.load_state_dict(torch.load("models/transformer_weight.pth"))
    transform_net.eval()
    
    # 输入输出路径
    input_dir = "VOCdevkit/trainA"
    output_dir = "VOCdevkit/stylized_trainA"
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历处理所有图片
    for img_name in os.listdir(input_dir):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            # 加载并预处理图片
            content_image = utils.load_image(os.path.join(input_dir, img_name))
            content_tensor = utils.itot(content_image).to(device)
            
            # 风格转换
            with torch.no_grad():
                generated_tensor = transform_net(content_tensor)
                generated_image = utils.ttoi(generated_tensor.detach())
            
            # 保存结果
            utils.saveimg(generated_image, os.path.join(output_dir, img_name))
            print(f"Processed {img_name}")

if __name__ == "__main__":
    batch_stylize() 