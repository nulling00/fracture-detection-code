

from ultralytics import YOLO
import os

def evaluate_yolo_model(model_path, data_path, save_name, save_dir="evaluation_results", conf=0.25, iou=0.5):
    """简化版 YOLO 模型评估脚本"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载模型
    model = YOLO(model_path)
    print(f"已加载模型: {model_path}")
    
    # 执行评估
    print("正在评估模型...")
    results = model.val(
        data=data_path,
        task="detect",
        conf=conf,
        iou=iou,
        verbose=True
    )
    
    # 提取关键指标
    metrics = {
        "mAP@0.5": results.box.map50,
        "mAP@0.5:0.95": results.box.map,
        "precision": results.box.p.mean(),  # 直接访问平均精度
        "recall": results.box.r.mean(),     # 直接访问平均召回率
        "F1-score": 2 * (results.box.p.mean() * results.box.r.mean()) / 
                  (results.box.p.mean() + results.box.r.mean() + 1e-10)
    }
    
    # 保存指标到TXT文件
    metrics_path = os.path.join(save_dir, save_name+"_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("=== 评估结果 ===\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    # 打印评估结果
    print("\n=== 评估结果 ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
        
    print(f"\n评估结果已保存至: {metrics_path}")
    return metrics

if __name__ == "__main__":
    # 配置参数
    model_path = '/mnt/data/ningling/CODE/YOLO/for_yolov8v11v12/result0618_4/my_yolov8/yolov8s_2/weights/best.pt'
    data_path = "/mnt/data/ningling/CODE/YOLO/for_yolov8v11v12/data_for_test.yaml"
    
    # 执行评估
    metrics = evaluate_yolo_model(
        model_path=model_path,
        data_path=data_path,
        save_name="yolov8s_testin",
        save_dir="evaluation_results_detection/"
    )