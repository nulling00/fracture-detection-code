import os
import time
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm
from frcnn import FRCNN
import xml.etree.ElementTree as ET

# 设置使用的GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def draw_box_on_image(image_path, box, save_path):
    """
    在图像上绘制给定的边界框（包含置信度信息）
    如果没有提供边界框（box为None或空列表），则保存原始图像
    """
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    
    # 检查box是否有效
    if box is not None and len(box) > 0:
        # 提取边界框坐标和置信度
        x1, y1, x2, y2, score = box[:5]
        box_coords = [int(x1), int(y1), int(x2), int(y2)]
        
        # 绘制预测框
        draw.rectangle(box_coords, outline="red", width=2)
        
        # 计算文本尺寸（兼容旧版Pillow）
        text = f"{score:.2f}"
        text_width, text_height = font.getsize(text)
        
        # 确保文本位置不超出图像边界
        x = max(box_coords[0], 0)
        y = max(box_coords[1] - text_height - 5, 0)
        
        # 绘制文本背景框
        bg_box = [x, y, x + text_width + 5, y + text_height + 5]
        draw.rectangle(bg_box, fill="white", outline="red")
        
        # 在底色上绘制文本
        draw.text((x + 2, y + 2), text, fill="red", font=font)
    
    # 创建保存目录（如果不存在）
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 生成输出路径
    base_name = os.path.basename(image_path)
    file_name, file_ext = os.path.splitext(base_name)
    output_name = file_name + "_predicted" + file_ext
    output_path = os.path.join(save_path, output_name)
    
    # 保存图像
    img.save(output_path)

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    # 计算交集区域
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # 计算交集面积
    intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    # 计算并集面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def load_ground_truth_boxes(gt_path):
    """加载真实标签边界框（Pascal VOC XML格式）"""
    gt_boxes = {}
    
    # 检查路径是否存在
    if not os.path.exists(gt_path):
        print(f"错误: 路径不存在 - {gt_path}")
        return gt_boxes
    
    # 遍历路径下的所有XML文件
    for xml_file in os.listdir(gt_path):
        if xml_file.endswith('.xml'):
            try:
                # 构建完整的文件路径
                xml_path = os.path.join(gt_path, xml_file)
                
                # 解析XML文件
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # 获取图像名称（不包含扩展名）
                filename = root.find('filename').text
                img_name = os.path.splitext(filename)[0]
                
                # 初始化当前图像的边界框列表
                gt_boxes[img_name] = []
                
                # 遍历所有对象
                for obj in root.findall('object'):
                    # 获取类别名称
                    class_name = obj.find('name').text
                    
                    # 获取边界框坐标
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # 将边界框添加到列表（格式：[xmin, ymin, xmax, ymax, class_name]）
                    gt_boxes[img_name].append([xmin, ymin, xmax, ymax, class_name])
            
            except Exception as e:
                print(f"处理文件 {xml_file} 时出错: {str(e)}")
    return gt_boxes

def calculate_metrics(tp, fp, tn, fn):
    """使用正态近似法计算评估指标与95%置信区间"""
    n = tp + fp + tn + fn
    
    # 准确率
    accuracy = (tp + tn) / n if n > 0 else 0
    acc_se = np.sqrt(accuracy * (1 - accuracy) / n) if n > 0 else 0
    acc_ci = (accuracy - 1.96 * acc_se, accuracy + 1.96 * acc_se)
    
    # 敏感度（召回率）
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    sen_se = np.sqrt(sensitivity * (1 - sensitivity) / (tp + fn)) if (tp + fn) > 0 else 0
    sen_ci = (sensitivity - 1.96 * sen_se, sensitivity + 1.96 * sen_se)
    
    # 特异度
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    spe_se = np.sqrt(specificity * (1 - specificity) / (tn + fp)) if (tn + fp) > 0 else 0
    spe_ci = (specificity - 1.96 * spe_se, specificity + 1.96 * spe_se)
    
    # 漏诊率
    miss_rate = 1 - sensitivity
    miss_se = sen_se
    miss_ci = (miss_rate - 1.96 * miss_se, miss_rate + 1.96 * miss_se)
    
    # 误诊率
    mis_rate = 1 - specificity
    mis_se = spe_se
    mis_ci = (mis_rate - 1.96 * mis_se, mis_rate + 1.96 * mis_se)
    
    # 阳性预测值
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    ppv_se = np.sqrt(ppv * (1 - ppv) / (tp + fp)) if (tp + fp) > 0 else 0
    ppv_ci = (ppv - 1.96 * ppv_se, ppv + 1.96 * ppv_se)
    
    # 阴性预测值
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    npv_se = np.sqrt(npv * (1 - npv) / (tn + fn)) if (tn + fn) > 0 else 0
    npv_ci = (npv - 1.96 * npv_se, npv + 1.96 * npv_se)
    
    return {
        'Accuracy': (accuracy, acc_ci),
        'Sensitivity': (sensitivity, sen_ci),
        'Specificity': (specificity, spe_ci),
        'Missed_Diagnosis_Rate': (miss_rate, miss_ci),
        'Misdiagnosis_Rate': (mis_rate, mis_ci),
        'PPV': (ppv, ppv_ci),
        'NPV': (npv, npv_ci)
    }

def calculate_metrics_bootstrap(labels, predictions, n_boot=1000):
    """
    Calculate evaluation metrics with 95% confidence intervals using bootstrap
    
    Args:
        labels (array): True labels
        predictions (array): Predicted labels
        n_boot (int): Number of bootstrap samples
        
    Returns:
        dict: Dictionary of metrics with 95% CI (excluding confusion matrix)
    """
    metrics_list = {'Accuracy': [], 'Sensitivity': [], 'Specificity': [], 
                   'Missed_Diagnosis_Rate': [], 'Misdiagnosis_Rate': [],
                   'PPV': [], 'NPV': [], 'F1': []}  # 新增F1键
    
    n = len(labels)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        boot_labels = labels[idx]
        boot_preds = predictions[idx]
        
        cm = confusion_matrix(boot_labels, boot_preds)
        if cm.size == 1:  # 处理只有一个类别的情况
            if boot_labels[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics for this bootstrap sample
        n_boot = tp + fp + tn + fn
        
        # Accuracy
        accuracy = (tp + tn) / n_boot if n_boot > 0 else 0
        metrics_list['Accuracy'].append(accuracy)
        
        # Sensitivity (Recall)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics_list['Sensitivity'].append(sensitivity)
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics_list['Specificity'].append(specificity)
        
        # Missed Diagnosis Rate (1 - Sensitivity)
        miss_rate = 1 - sensitivity
        metrics_list['Missed_Diagnosis_Rate'].append(miss_rate)
        
        # Misdiagnosis Rate (1 - Specificity)
        mis_rate = 1 - specificity
        metrics_list['Misdiagnosis_Rate'].append(mis_rate)
        
        # Positive Predictive Value (PPV)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics_list['PPV'].append(ppv)
        
        # Negative Predictive Value (NPV)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics_list['NPV'].append(npv)
        
        # 新增F1分数计算（使用PPV和Sensitivity）
        precision = ppv  # PPV即精确率
        recall = sensitivity  # 灵敏度即召回率
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        metrics_list['F1'].append(f1)
    
    # Calculate mean and 95% CI for each metric
    metrics = {}
    for name, values in metrics_list.items():
        mean_val = np.mean(values)
        lower = np.percentile(values, 2.5)
        upper = np.percentile(values, 97.5)
        metrics[name] = (mean_val, (lower, upper))
    
    return metrics

def print_metrics(metrics, labels, predictions):
    """打印评估指标与95%置信区间和混淆矩阵"""
    print("\n=== Evaluation Metrics (95% CI) ===")
    for name, (value, ci) in metrics.items():
        print(f"{name}: {value:.4f} [{ci[0]:.4f}-{ci[1]:.4f}]")
    
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n=== Confusion Matrix ===")
    print(f"TP: {tp}, FP: {fp}")
    print(f"FN: {fn}, TN: {tn}")

def save_metrics_to_file(metrics, labels, predictions, filename):
    """保存评估指标与95%置信区间和混淆矩阵到文本文件"""
    with open(filename, 'w') as f:
        f.write("=== Evaluation Metrics (95% CI) ===\n")
        for name, (value, ci) in metrics.items():
            f.write(f"{name}: {value:.4f} [{ci[0]:.4f}-{ci[1]:.4f}]\n")
        
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        f.write("\n=== Confusion Matrix ===\n")
        f.write(f"TP: {tp}, FP: {fp}\n")
        f.write(f"FN: {fn}, TN: {tn}\n")
        
        f.write("\n=== Raw Counts ===\n")
        f.write(f"Total Positives: {tp + fn}\n")
        f.write(f"Total Negatives: {tn + fp}\n")
        f.write(f"Total Samples: {tp + fp + tn + fn}\n")

def get_auc_ci_bootstrap(scores, labels, n_boot=1000):
    """使用bootstrap方法计算AUC的95%置信区间"""
    aucs = []
    n = len(labels)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        boot_scores = scores[idx]
        boot_labels = labels[idx]
        fpr, tpr, _ = roc_curve(boot_labels, boot_scores)
        aucs.append(auc(fpr, tpr))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def evaluate_model(conf_threshold=0.001, 
                   roc_file="roc_curve.png", pr_file="pr_curve.png", 
                   cm_file="confusion_matrix.png", metrics_file="metrics.txt",
                   val_gt_path=None,
                   output_folder="predict_results",
                   dir_origin_path = '/mnt/data/yanmingshuo/CODE/faster-rcnn-pytorch-master/img_test_in',
                   iou_threshold=0.5,
                   model_name='resnet_testin'):
    """评估Faster R-CNN模型性能"""
    # 创建输出目录
    output_dir = os.path.dirname(roc_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 自定义配置参数
    custom_config ={
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"    : '/mnt/data/yanmingshuo/CODE/faster-rcnn-pytorch-master/model_data/trained_pth/best_epoch_weights_resnet50.pth',
      
        "classes_path"  : '/mnt/data/yanmingshuo/CODE/faster-rcnn-pytorch-master/model_data/voc_classes.txt',
        #---------------------------------------------------------------------#
        #   网络的主干特征提取网络，resnet50或者vgg
        #---------------------------------------------------------------------#
        # "backbone"      : "vgg",
        "backbone"      : "resnet50",

        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #-----------------------------------  ----------------------------------#
        "confidence"    : 0.03,
        
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"       : 0.3,
        #---------------------------------------------------------------------#
        #   用于指定先验框的大小
        #---------------------------------------------------------------------#
        'anchors_size'  : [8, 16, 32],
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"          : True,
    }
    # 使用自定义配置初始化模型
    frcnn = FRCNN(**custom_config)
    # 初始化模型
    # frcnn = FRCNN()
    
    # 准备数据
    labels = []
    scores = []
    all_ious = []  # 存储所有IoU值
    
    # 加载真实标签（如果提供）
    gt_boxes_dict = load_ground_truth_boxes(val_gt_path) if val_gt_path else {}
    # print(f'jksadlkjaslkjdalkjds是:{len(gt_boxes_dict)}' )
    #######//////////select you dataset
    # dir_origin_path = '/mnt/data/yanmingshuo/CODE/faster-rcnn-pytorch-master/img_test_in'
    # dir_origin_path = '/mnt/data/yanmingshuo/CODE/faster-rcnn-pytorch-master/img_test_out'
    # dir_origin_path = '/mnt/data/yanmingshuo/CODE/faster-rcnn-pytorch-master/img_test_all'
    
    img_names = [f for f in os.listdir(dir_origin_path) 
                 if f.lower().endswith(('.bmp', '.jpg', '.png', '.jpeg', '.tif'))]
    
    # 遍历图像进行评估
    for filename in tqdm(img_names):
        # try:
        image_path = os.path.join(dir_origin_path, filename)
        image = Image.open(image_path)
        img_name = os.path.splitext(filename)[0]
        
        # 分配真实标签（基于文件名编号）#######//////////
        # if int(img_name) <= int('001981'):####using_img_test_in
        # if int(img_name) <= int('002461'):####using_img_test_out
        # if (int(img_name) <= int('001981'))|(int(img_001981name) <= int('img_name')&(int(img_name) <= int('002461'))):####using_img_test_all

        # true_label = 1 if int(img_name) <= int('001981') else 0
        # true_label = 1 if int(img_name) <= int('002461') else 0
        # true_label = 1 if (int(img_name) <= int('001981'))|(int(img_001981name) <= int('img_name')&(int(img_name) <= int('002461'))) else 0

        gt_boxes = gt_boxes_dict.get(img_name, [])
        true_label = 1 if gt_boxes else 0
        # print(f"文件名: {filename}, 真实框数量: {len(gt_boxes)}, 分配标签: {true_label}")
            
        # 模型预测
        _ , prediction = frcnn.detect_image(image)
      
        # img_pred, prediction = frcnn.detect_image(image)
        # img_output_name =  "predicted_" +filename
        # img_output_path = os.path.join("predict_results2", img_output_name)
        # img_pred.save(img_output_path)
        pred_boxes = [prediction[:4]] if len(prediction) > 0 else []

        # 计算IoU并生成综合得分
        if true_label == 1 and gt_boxes and pred_boxes:
            # 初始化满足IoU阈值的最高置信度
            valid_confidences = []
         
            # 遍历所有预测框与真实框的组合
            for pb in pred_boxes:
                for gb in gt_boxes:
                    iou = calculate_iou(pb, gb)
                    print(gb)
                    print(pb)
                    if iou >= iou_threshold:
                        # 获取当前预测框的置信度
                        confidence = prediction[4]  # 假设prediction[4]是置信度
                        # valid_confidences.append(confidence * iou)
                        valid_confidences.append(confidence)

            
            # 如果有满足条件的预测框，取最高置信度
            if valid_confidences:
                score = max(valid_confidences)
                # 记录这些有效预测框的最大IoU
                all_ious.append(iou)
            else:
                # 没有满足IoU阈值的预测框，使用极小值
                score = 1e-6
                all_ious.append(0)
        else:
            score = prediction[4] if pred_boxes else 1e-6
            if true_label == 1 and gt_boxes:
                all_ious.append(0)  # 无预测框时IoU为0

        labels.append(true_label)
        scores.append(score)
        # print(pred_boxes)
            # Draw the box on the image and save it
        # draw_box_on_image(image_path, prediction[:5], output_folder)
        # 释放内存
        del image, prediction
        torch.cuda.empty_cache()
        # except Exception as e:
        #     print(f"处理{filename}时出错: {e}")
        #     continue
    
    # 保存原始得分和标签
    np.savetxt(f'array_{model_name}_scores.txt', scores, fmt='%.6f', delimiter=' ')
    np.savetxt(f'array_{model_name}_labels.txt', labels, fmt='%.6f', delimiter=' ')
    np.savetxt(f'array_{model_name}_iou.txt', all_ious, fmt='%.6f', delimiter=' ')

    
    # 转换为numpy数组
    labels = np.array(labels)
    scores = np.array(scores)
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # 计算AUC的95%置信区间
    auc_ci_lower, auc_ci_upper = get_auc_ci_bootstrap(scores, labels)
    
    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(labels, scores)
    
    # 计算最优阈值
    optimal_idx = np.argmax(tpr - fpr)
    # optimal_threshold = _[optimal_idx]
    optimal_threshold = 0.25

    
    # 生成二分类预测
    binary_predictions = (scores >= optimal_threshold).astype(int)
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, binary_predictions)
    if cm.size == 1:
        # 处理所有预测和真实标签都相同的情况
        if labels[0] == 0:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()
    
    # 计算评估指标（带95%置信区间）
    metrics = calculate_metrics_bootstrap(labels, binary_predictions)
    metrics['AUC'] = (roc_auc, (auc_ci_lower, auc_ci_upper))
    
    # 计算平均IoU（仅对正样本）
    if all_ious:
        mean_iou = np.mean(all_ious)
        iou_ci_lower = np.percentile(all_ious, 2.5)
        iou_ci_upper = np.percentile(all_ious, 97.5)
    else:
        mean_iou = 0
        iou_ci_lower = 0
        iou_ci_upper = 0
    
    metrics['Mean_IoU'] = (mean_iou, (iou_ci_lower, iou_ci_upper))
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'darkorange', lw=2, 
             label=f'ROC (AUC = {roc_auc:.3f} [{auc_ci_lower:.3f}-{auc_ci_upper:.3f}])')
    plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with AUC 95% CI')
    plt.legend(loc='lower right')
    plt.savefig(roc_file)
    plt.close()
    
    # 绘制PR曲线
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, 'blue', lw=2, label='PR Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.savefig(pr_file)
    plt.close()
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(cm_file)
    plt.close()
    
    # 保存并打印指标
    save_metrics_to_file(metrics, labels, binary_predictions, metrics_file)
    print_metrics(metrics, labels, binary_predictions)
    
    # 返回评估结果
    return {
        'roc': (fpr, tpr, roc_auc),
        'pr': (precision, recall),
        'confusion_matrix': cm,
        'optimal_threshold': optimal_threshold,
        'metrics': metrics
    }


if __name__ == "__main__":
    output_dir = "the_result_iou0_5_conf_0_25"  # 替换非法字符
    # 创建目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    model_name="resnet_testin"
    # 构建合法的文件路径
    roc_image_file = os.path.join(output_dir, f"{model_name}_roc_curve.png")
    pr_image_file = os.path.join(output_dir, f"{model_name}_pr_curve.png")
    cm_image_file = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    metrics_text_file = os.path.join(output_dir, f"{model_name}_detailed_metrics.txt")
    output_folder_img =os.path.join(output_dir, f"for_{model_name}/predict_result")
   
    # 真实标签目录（如果有）
    gt_labels_path = "/mnt/data/yanmingshuo/CODE/faster-rcnn-pytorch-master/img_test_gt"
    dir_origin_path_img = '/mnt/data/yanmingshuo/CODE/faster-rcnn-pytorch-master/img_test_in'
    # dir_origin_path_img = '/mnt/data/yanmingshuo/CODE/faster-rcnn-pytorch-master/img_test_out'
    # dir_origin_path_img = '/mnt/data/yanmingshuo/CODE/faster-rcnn-pytorch-master/img_test_all'

    # 执行评估
    evaluate_model(
        roc_file=roc_image_file,
        pr_file=pr_image_file,
        cm_file=cm_image_file,
        metrics_file=metrics_text_file,
        output_folder=output_folder_img,
        val_gt_path=gt_labels_path,
        dir_origin_path=dir_origin_path_img,
        model_name=model_name
    )
