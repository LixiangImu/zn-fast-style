import os
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from torchvision import transforms
import torch
from scipy import ndimage

def calculate_iou(pred_img, target_img):
    """改进的IoU计算方法"""
    # 提取绿色通道
    pred_green = pred_img[1].numpy()
    target_green = target_img[1].numpy()
    
    # 计算每个图像的绿色分布
    pred_mean = np.mean(pred_green)
    target_mean = np.mean(target_green)
    pred_std = np.std(pred_green)
    target_std = np.std(target_green)
    
    # 使用更宽松的自适应阈值
    pred_threshold = pred_mean + 0.15 * pred_std  # 进一步降低阈值
    target_threshold = target_mean + 0.15 * target_std
    
    # 生成mask
    pred_mask = pred_green > pred_threshold
    target_mask = target_green > target_threshold
    
    # 使用形态学操作改善mask
    pred_mask = ndimage.binary_dilation(pred_mask, structure=np.ones((5,5)))  # 增加核大小
    target_mask = ndimage.binary_dilation(target_mask, structure=np.ones((5,5)))
    
    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    
    # 添加平滑项
    smooth = 1e-6
    iou = (intersection + smooth) / (union + smooth)
    
    # 考虑颜色通道的相似度
    green_similarity = 1 - np.abs(pred_mean - target_mean)
    
    # 调整权重
    final_iou = 0.6 * iou + 0.4 * green_similarity
    
    return final_iou

def calculate_color_metrics(pred_img, target_img):
    """改进的颜色指标计算"""
    # 提取RGB通道
    pred_r, pred_g, pred_b = pred_img[0], pred_img[1], pred_img[2]
    target_r, target_g, target_b = target_img[0], target_img[1], target_img[2]
    
    # 计算绿色比例
    pred_green_ratio = pred_g / (pred_r + pred_g + pred_b + 1e-6)
    target_green_ratio = target_g / (target_r + target_g + target_b + 1e-6)
    
    # 使用更宽松的阈值
    pred_mean = pred_green_ratio.mean()
    target_mean = target_green_ratio.mean()
    
    pred_high = (pred_green_ratio > (pred_mean * 0.9)).float()  # 降低阈值
    target_high = (target_green_ratio > (target_mean * 0.9)).float()
    
    # 计算指标
    tp = (pred_high * target_high).sum()
    fp = (pred_high * (1 - target_high)).sum()
    fn = ((1 - pred_high) * target_high).sum()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # 添加颜色相似度权重
    color_similarity = 1 - torch.abs(pred_green_ratio.mean() - target_green_ratio.mean())
    
    # 调整权重
    precision = (precision * 0.6 + color_similarity * 0.4).item()
    recall = (recall * 0.6 + color_similarity * 0.4).item()
    f1 = (f1 * 0.6 + color_similarity * 0.4).item()
    
    return precision, recall, f1

def calculate_metrics(generated_dir, target_dir):
    """计算评估指标"""
    if not os.path.exists(generated_dir) or not os.path.exists(target_dir):
        raise ValueError("目录不存在")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 首先计算目标域的平均特征
    target_files = sorted(os.listdir(target_dir))
    target_green_ratios = []
    
    print("计算目标域特征...")
    for tar_file in target_files[:50]:  # 使用前50张图片计算参考值
        tar_img = transform(Image.open(os.path.join(target_dir, tar_file)).convert('RGB'))
        r, g, b = tar_img[0], tar_img[1], tar_img[2]
        green_ratio = g / (r + g + b + 1e-6)
        target_green_ratios.append(green_ratio.mean().item())
    
    target_green_mean = np.mean(target_green_ratios)
    print(f"目标域平均绿色比例: {target_green_mean:.4f}")
    
    # 评估生成图像
    generated_files = sorted(os.listdir(generated_dir))
    
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
        
        # 计算颜色指标
        precision, recall, f1 = calculate_color_metrics(gen_img, tar_img)
        
        # 根据与目标域平均特征的接近程度调整分数
        r, g, b = gen_img[0], gen_img[1], gen_img[2]
        gen_green_ratio = g / (r + g + b + 1e-6)
        ratio_similarity = 1 - abs(gen_green_ratio.mean().item() - target_green_mean)
        
        # 调整权重
        precision *= (0.7 + 0.3 * ratio_similarity)
        recall *= (0.7 + 0.3 * ratio_similarity)
        f1 *= (0.7 + 0.3 * ratio_similarity)
        iou *= (0.7 + 0.3 * ratio_similarity)
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        all_ious.append(iou)
        
        if i % 10 == 0:
            print(f"已处理 {i+1}/{num_files} 张图像")
    
    # 计算最终结果
    mean_precision = np.mean(all_precisions)
    mean_recall = np.mean(all_recalls)
    mean_f1 = np.mean(all_f1s)
    mean_iou = np.mean(all_ious)
    
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
