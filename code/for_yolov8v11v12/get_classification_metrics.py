import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score , f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Specify GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def evaluate_model(model_path, val_dataset_path, conf_threshold=0.25, 
                   roc_file="roc_curve.png", pr_file="pr_curve.png", 
                   cm_file="confusion_matrix.png", metrics_file="metrics.txt",
                   output_folder="/mnt/data/ningling/CODE/YOLO/for_yolov8v11v12/predict_result",
                   iou_threshold=0.5,
                   model_name='yolov8_testin'):
    """
    Evaluate YOLOv8 model with comprehensive metrics and confidence intervals
    
    Args:
        model_path (str): Path to model weights
        val_dataset_path (str): Path to validation dataset
        conf_threshold (float): Confidence threshold for detection
        roc_file (str): File name for ROC curve image
        pr_file (str): File name for PR curve image
        cm_file (str): File name for confusion matrix image
        metrics_file (str): File name for metrics text file
        output_folder (str): Folder to save prediction results
        iou_threshold (float): IoU threshold for considering a prediction as positive
    """
    model = YOLO(model_path)
    labels, scores = [], []
    
    # Process all images in the dataset
    for img_name in os.listdir(os.path.join(val_dataset_path, 'images')):
        img_path = os.path.join(val_dataset_path, 'images', img_name)
        
        # Get the corresponding label file
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(val_dataset_path, 'labels', label_name)
        
        # Determine if the image contains any objects based on the label file
        has_objects = os.path.exists(label_path) and os.path.getsize(label_path) > 0
        true_label = 1 if has_objects else 0
        
        # Get the ground truth bounding boxes if they exist
        gt_boxes = []
        if has_objects:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    # YOLO format: class x_center y_center width height (normalized)
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x_center, y_center = float(parts[1]), float(parts[2])
                        width, height = float(parts[3]), float(parts[4])
                        
                        # Convert to absolute coordinates (assuming image size is known)
                        img_width, img_height = get_image_size(img_path)
                        x1 = (x_center - width/2) * img_width
                        y1 = (y_center - height/2) * img_height
                        x2 = (x_center + width/2) * img_width
                        y2 = (y_center + height/2) * img_height
                        
                        gt_boxes.append([x1, y1, x2, y2])
        
        # Run prediction
        results = model.predict(img_path, conf=conf_threshold)
        pred_boxes = results[0].boxes
        
        # Determine the prediction score based on IoU
        if len(pred_boxes) > 0:
            # For images with ground truth objects
            if has_objects:
                max_score = 0
                # For each predicted box, check if it overlaps with any ground truth box
                for i, pred_box in enumerate(pred_boxes.xyxy.cpu().numpy()):
                    for gt_box in gt_boxes:
                        iou = calculate_iou(pred_box, gt_box)
                        if iou > iou_threshold:
                            score = pred_boxes.conf.cpu().numpy()[i] 
                            if score > max_score:
                                max_score = score
                
                # If no box has IoU > threshold, use a very low score
                if max_score == 0:
                    max_score = 1e-6
            else:
                # For images without ground truth objects, use the highest prediction score
                max_score = pred_boxes.conf.cpu().max().item()
                # For images without ground truth objects, use the highest prediction score
                # max_score = 1 
                # For images without ground truth objects, use the highest prediction score
                # max_score = pred_boxes.conf.cpu().max().item() * 4
        else:
            # No predictions, use a very low score
            max_score = 1e-6
        
        labels.append(true_label)
        scores.append(max_score)
        
        # Draw the box on the image and save it
        # draw_box_on_image(img_path, pred_boxes, output_folder)

    # Save scores and labels
    np.savetxt(f'array_numpy_{model_name}_scores.txt', scores, fmt='%.6f', delimiter=' ')
    np.savetxt(f'array_numpy_{model_name}_labels.txt', labels, fmt='%.6f', delimiter=' ')

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate 95% CI for AUC using bootstrap
    labels = np.array(labels)
    scores = np.array(scores)
    auc_ci_lower, auc_ci_upper = get_auc_ci_bootstrap(scores, labels)
    
    # Calculate PR curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    
    # 计算AP（平均精度）
    ap = average_precision_score(labels, scores)
    ap_ci_lower, ap_ci_upper = get_ap_ci_bootstrap(scores, labels)
    # Calculate optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    print(optimal_idx)
    optimal_threshold = 0.25
    # optimal_threshold = thresholds[optimal_idx]
    print(optimal_threshold)

    # Generate binary predictions
    binary_predictions = (np.array(scores) >= optimal_threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, binary_predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics with 95% CI using bootstrap
    metrics = calculate_metrics_bootstrap(labels, binary_predictions)

    # Add AUC with CI to metrics
    metrics['AUC'] = (roc_auc, (auc_ci_lower, auc_ci_upper))
    
    # Plot ROC curve with CI in legend
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f} [{auc_ci_lower:.3f}-{auc_ci_upper:.3f}])')
    plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with AUC 95% CI')
    plt.legend(loc='lower right')
    plt.savefig(roc_file)
    plt.close()  
    
    # Plot PR curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, 'blue', lw=2, label=f'PR Curve (AP = {ap:.3f} [{ap_ci_lower:.3f}-{ap_ci_upper:.3f}])')

    # plt.plot(recall, precision, 'blue', lw=2, label='PR Curve')
    plt.xlim([-0.05, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.savefig(pr_file)
    plt.close()  
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(cm_file)
    plt.close()  
    
    # Save metrics to file and print
    save_metrics_to_file(metrics, labels, binary_predictions, metrics_file)
    print_metrics(metrics, labels, binary_predictions)
    
    return {
        'roc': (fpr, tpr, roc_auc),
        'pr': (precision, recall),
        'confusion_matrix': cm,
        'optimal_threshold': optimal_threshold,
        'metrics': metrics
    }

def draw_box_on_image(image_path, boxes, save_path):
    """
    Draw the highest scoring bounding box on the image and save it with score background.
    If no boxes are found, save the original image.
    """
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()  # 加载默认字体
    
    if len(boxes) > 0:
        # Move boxes.conf to CPU before finding the max score index
        max_score_idx = np.argmax(boxes.conf.cpu().numpy())
        # Move boxes.xyxy to CPU before converting to numpy
        box = boxes.xyxy.cpu().numpy()[max_score_idx].astype(int)
        score = boxes.conf.cpu().numpy()[max_score_idx]
        
        # 绘制预测框
        draw.rectangle(box.tolist(), outline="red", width=2)
        
        # 计算文本尺寸并绘制底色矩形
        text = f"{score:.2f}"
        
        # 使用textbbox替代textsize
        # textbbox返回 (left, top, right, bottom)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]  # right - left
        text_height = bbox[3] - bbox[1]  # bottom - top
        
        x, y = box[0], box[1] - text_height - 5  # 文本上方留出空间
        bg_box = [x, y, x + text_width + 5, y + text_height + 5]
        draw.rectangle(bg_box, fill="white", outline="red")  # 白色背景，红色边框
        
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

def get_image_size(image_path):
    """获取图像的宽度和高度"""
    img = Image.open(image_path)
    return img.size

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    # box格式: [x1, y1, x2, y2]
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集的坐标
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # 计算交集面积
    if x1_i < x2_i and y1_i < y2_i:
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
    else:
        intersection = 0
    
    # 计算两个框的面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 计算并集面积
    union = area1 + area2 - intersection
    
    # 计算IoU
    if union > 0:
        iou = intersection / union
    else:
        iou = 0
    
    return iou

# 其他函数保持不变

def calculate_metrics(tp, fp, tn, fn):
    """
    Calculate evaluation metrics with 95% confidence intervals using normal approximation
    
    Args:
        tp (int): True Positives
        fp (int): False Positives
        tn (int): True Negatives
        fn (int): False Negatives
        
    Returns:
        dict: Dictionary of metrics with 95% CI (excluding confusion matrix)
    """
    n = tp + fp + tn + fn
    
    # Accuracy
    accuracy = (tp + tn) / n
    acc_se = np.sqrt(accuracy * (1 - accuracy) / n) if n > 0 else 0
    acc_ci = (accuracy - 1.96 * acc_se, accuracy + 1.96 * acc_se)
    
    # Sensitivity (Recall)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    sen_se = np.sqrt(sensitivity * (1 - sensitivity) / (tp + fn)) if (tp + fn) > 0 else 0
    sen_ci = (sensitivity - 1.96 * sen_se, sensitivity + 1.96 * sen_se)
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    spe_se = np.sqrt(specificity * (1 - specificity) / (tn + fp)) if (tn + fp) > 0 else 0
    spe_ci = (specificity - 1.96 * spe_se, specificity + 1.96 * spe_se)
    
    # Missed Diagnosis Rate (1 - Sensitivity)
    miss_rate = 1 - sensitivity
    miss_se = sen_se
    miss_ci = (miss_rate - 1.96 * miss_se, miss_rate + 1.96 * miss_se)
    
    # Misdiagnosis Rate (1 - Specificity)
    mis_rate = 1 - specificity
    mis_se = spe_se
    mis_ci = (mis_rate - 1.96 * mis_se, mis_rate + 1.96 * mis_se)
    
    # Positive Predictive Value (PPV)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    ppv_se = np.sqrt(ppv * (1 - ppv) / (tp + fp)) if (tp + fp) > 0 else 0
    ppv_ci = (ppv - 1.96 * ppv_se, ppv + 1.96 * ppv_se)
    
    # Negative Predictive Value (NPV)
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
    """Print metrics with 95% confidence intervals and confusion matrix"""
    print("\n=== Evaluation Metrics (95% CI) ===")
    for name, (value, ci) in metrics.items():
        print(f"{name}: {value:.4f} [{ci[0]:.4f}-{ci[1]:.4f}]")
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Print confusion matrix
    print("\n=== Confusion Matrix ===")
    print(f"TP: {tp}, FP: {fp}")
    print(f"FN: {fn}, TN: {tn}")

def save_metrics_to_file(metrics, labels, predictions, filename):
    """Save metrics with 95% confidence intervals and confusion matrix to a text file"""
    with open(filename, 'w') as f:
        f.write("=== Evaluation Metrics (95% CI) ===\n")
        for name, (value, ci) in metrics.items():
            f.write(f"{name}: {value:.4f} [{ci[0]:.4f}-{ci[1]:.4f}]\n")
        
        # Calculate confusion matrix
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Save confusion matrix
        f.write("\n=== Confusion Matrix ===\n")
        f.write(f"TP: {tp}, FP: {fp}\n")
        f.write(f"FN: {fn}, TN: {tn}\n")
        
        # Save raw counts
        f.write("\n=== Raw Counts ===\n")
        f.write(f"Total Positives: {tp + fn}\n")
        f.write(f"Total Negatives: {tn + fp}\n")
        f.write(f"Total Samples: {tp + fp + tn + fn}\n")

def get_auc_ci_bootstrap(scores, labels, n_boot=1000):
    aucs = []
    n = len(labels)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        boot_scores = scores[idx]
        boot_labels = labels[idx]
        fpr, tpr, _ = roc_curve(boot_labels, boot_scores)
        aucs.append(auc(fpr, tpr))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def get_ap_ci_bootstrap(scores, labels, n_boot=1000, ci_level=0.95):
    """
    使用自助法(Bootstrap)计算AP(Average Precision)的置信区间
    
    参数:
    - scores: 模型预测的分数
    - labels: 真实标签(0或1)
    - n_boot: 自助抽样次数，默认为1000次
    - ci_level: 置信水平，默认为0.95 (对应95%置信区间)
    
    返回:
    - 元组(ci_lower, ci_upper): 置信区间的下限和上限
    """
    # 确保输入是numpy数组
    scores = np.array(scores)
    labels = np.array(labels)
    
    # 检查输入长度是否一致
    if len(scores) != len(labels):
        raise ValueError("scores和labels的长度必须一致")
    
    # 检查是否包含正负两类样本
    if len(np.unique(labels)) < 2:
        raise ValueError("标签中必须同时包含正类和负类样本")
    
    aps = []
    n = len(labels)
    alpha = (1 - ci_level) / 2
    
    # 进行Bootstrap抽样
    for i in range(n_boot):
        # 显示进度
        if i % 100 == 0 and i > 0:
            print(f"Bootstrap进度: {i}/{n_boot}")
            
        # 有放回地随机抽样
        idx = np.random.choice(n, n, replace=True)
        boot_scores = scores[idx]
        boot_labels = labels[idx]
        
        # 检查抽样后是否仍包含两类样本
        if len(np.unique(boot_labels)) < 2:
            continue  # 如果抽样后只有一类样本，则跳过
        
        # 计算AP
        ap = average_precision_score(boot_labels, boot_scores)
        aps.append(ap)
    
    # 确保有足够的AP值来计算置信区间
    if len(aps) < 10:
        raise ValueError("由于抽样问题，未能获得足够的有效AP值")
    
    # 计算置信区间
    ci_lower = np.percentile(aps, alpha * 100)
    ci_upper = np.percentile(aps, (1 - alpha) * 100)
    
    return ci_lower, ci_upper


def get_auc_ci_normal_approximation(scores, labels, alpha=0.95):
    """
    使用正态近似法计算AUC95%置信区间
    
    参数:
        scores (array): 模型预测得分
        labels (array): 真实标签(0或1)
        alpha (float): 置信水平(默认0.95)
        
    返回:
        tuple: 置信区间下限和上限
    """
    # 分离正负样本得分
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    
    # 计算AUC
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    total_pairs = n_pos * n_neg
    
    # 计算AUC的分子部分
    auc_numerator = 0
    for pos in pos_scores:
        for neg in neg_scores:
            if pos > neg:
                auc_numerator += 1
            elif pos == neg:
                auc_numerator += 0.5  # 平局时得0.5分
                
    auc = auc_numerator / total_pairs
    
    # 计算AUC的方差 (Hanley-McNeil方法简化版)
    # 计算p0: 一个随机正样本得分高于一个随机负样本的概率
    p0 = auc
    
    # 计算p1: 两个随机正样本得分都高于一个随机负样本的概率
    p1 = 0
    for pos1 in pos_scores:
        for pos2 in pos_scores:
            if pos1 != pos2:
                count = 0
                for neg in neg_scores:
                    if pos1 > neg and pos2 > neg:
                        count += 1
                p1 += count / (n_neg * (n_pos - 1))
    p1 /= n_pos
    
    # 计算p2: 两个随机负样本得分都低于一个随机正样本的概率
    p2 = 0
    for neg1 in neg_scores:
        for neg2 in neg_scores:
            if neg1 != neg2:
                count = 0
                for pos in pos_scores:
                    if neg1 < pos and neg2 < pos:
                        count += 1
                p2 += count / (n_pos * (n_neg - 1))
    p2 /= n_neg
    
    # 计算方差
    variance = (p0 * (1 - p0) + 
                (n_pos - 1) * (p1 - p0**2) + 
                (n_neg - 1) * (p2 - p0**2)) / (n_pos * n_neg)
    
    # 计算标准误差和置信区间
    se = np.sqrt(variance)
    z_score = 1.96  # 对应95%置信水平
    lower = max(0, auc - z_score * se)
    upper = min(1, auc + z_score * se)
    
    return lower, upper

if __name__ == "__main__":
    # model_path = '/mnt/data/ningling/CODE/YOLO/for_yolov8v11v12/yolo12s_2.pt'
    model_path = '/mnt/data/ningling/CODE/YOLO/for_yolov8v11v12/result0618_4/my_yolov12/yolo12s_2/weights/best.pt'
    
    val_dataset_path = "/mnt/data/ningling/DATASET/dataset_yolo_all_fracture/test_out"
    model_name='yolov12s_testout'

    output_dir = "the_result_yolo_iou0_5_conf_0.25"  # 替换非法字符

    # 创建目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 构建合法的文件路径
    
    roc_image_file = os.path.join(output_dir, f"{model_name}_roc_curve.png")
    pr_image_file = os.path.join(output_dir,f"{model_name}_pr_curve.png")
    cm_image_file = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    metrics_text_file = os.path.join(output_dir, f"{model_name}__detailed_metrics.txt")
    output_folder_img =os.path.join(output_dir, f"for_{model_name}/predict_result")
    evaluate_model(
        model_path, 
        val_dataset_path,
        roc_file=roc_image_file,
        pr_file=pr_image_file,
        cm_file=cm_image_file,
        metrics_file=metrics_text_file,
        output_folder=output_folder_img,
        model_name=model_name
    )