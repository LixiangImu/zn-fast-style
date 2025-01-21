import os
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from torchvision import transforms

def calculate_iou(pred_img, target_img):
    """基于绿色通道强度计算IoU"""
    # 提取绿色通道并归一化
    pred_green = pred_img[1].numpy()
    target_green = target_img[1].numpy()
    
    # 计算高绿色区域的mask
    pred_mask = pred_green > np.mean(pred_green)
    target_mask = target_green > np.mean(target_green)
    
    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    
    return intersection / (union + 1e-6)

def calculate_color_metrics(pred_img, target_img):
    """计算基于颜色分布的指标"""
    # 提取绿色通道
    pred_green = pred_img[1]
    target_green = target_img[1]
    
    # 计算绿色通道的分布特征
    pred_mean = pred_green.mean()
    target_mean = target_green.mean()
    
    # 将图像分为高绿色和低绿色区域
    pred_high = (pred_green > pred_mean).float()
    target_high = (target_green > target_mean).float()
    
    # 计算precision和recall
    tp = (pred_high * target_high).sum()
    fp = (pred_high * (1 - target_high)).sum()
    fn = ((1 - pred_high) * target_high).sum()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return precision.item(), recall.item(), f1.item()

def calculate_metrics(generated_dir, target_dir):
    """计算评估指标"""
    if not os.path.exists(generated_dir) or not os.path.exists(target_dir):
        raise ValueError("目录不存在")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 统一图像大小
        transforms.ToTensor(),
    ])
    
    generated_files = sorted(os.listdir(generated_dir))
    target_files = sorted(os.listdir(target_dir))
    
    print(f"找到 {len(generated_files)} 个生成图像和 {len(target_files)} 个目标图像")
    
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_ious = []
    
    num_files = min(len(generated_files), len(target_files))
    for i in range(num_files):
        gen_path = os.path.join(generated_dir, generated_files[i])
        tar_path = os.path.join(target_dir, target_files[i])
        
        gen_img = transform(Image.open(gen_path).convert('RGB'))
        tar_img = transform(Image.open(tar_path).convert('RGB'))
        
        # 计算IoU
        iou = calculate_iou(gen_img, tar_img)
        all_ious.append(iou)
        
        # 计算颜色指标
        precision, recall, f1 = calculate_color_metrics(gen_img, tar_img)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        
        if i % 10 == 0:
            print(f"已处理 {i+1}/{num_files} 张图像")
    
    # 计算平均值
    mean_precision = np.mean(all_precisions)
    mean_recall = np.mean(all_recalls)
    mean_f1 = np.mean(all_f1s)
    mean_iou = np.mean(all_ious)
    
    # 打印结果
    print("\n评估结果:")
    print(f"Precision: {mean_precision:.4f}")
    print(f"Recall: {mean_recall:.4f}")
    print(f"F1 Score: {mean_f1:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    
    return {
        'precision': mean_precision,
        'recall': mean_recall,
        'f1': mean_f1,
        'iou': mean_iou
    }

def main():
    generated_dir = "VOCdevkit/trainA_converted"
    target_dir = "VOCdevkit/trainB"
    
    try:
        metrics = calculate_metrics(generated_dir, target_dir)
        
        with open('evaluation_results.txt', 'w') as f:
            f.write("评估结果:\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"Mean IoU: {metrics['iou']:.4f}\n")
        
        print("\n结果已保存到 evaluation_results.txt")
        
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")

if __name__ == "__main__":
    main()
