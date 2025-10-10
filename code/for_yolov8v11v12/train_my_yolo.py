import warnings
import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Training Script with Command Line Arguments')
    
    # 模型相关参数
    parser.add_argument('--model-index', type=int, default=2, 
                      help='Index of model configuration in my_model list (0, 1, or 2)')
    parser.add_argument('--weight-index', type=int, default=3, 
                      help='Index of weight file in the selected model list (0, 1, 2, or 3)')
    parser.add_argument('--custom-weight', type=str, default=None, 
                      help='Path to custom weight file (overrides weight-index if provided)')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=300, 
                      help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, 
                      help='Image size for training')
    parser.add_argument('--lr0', type=float, default=0.002, 
                      help='Initial learning rate')
    parser.add_argument('--batch', type=int, default=32, 
                      help='Batch size')
    parser.add_argument('--device', type=str, default='4', 
                      help='GPU device indices (comma-separated, e.g., "4,5")')
    parser.add_argument('--project', type=str, default='result0618_4', 
                      help='Project directory for saving results')
    
    return parser.parse_args()


if __name__ == '__main__':  
    warnings.filterwarnings('ignore')
    
    # 解析命令行参数
    args = parse_args()
    
    # 模型配置和权重路径列表（保持原样）
    my_model = [
                ['./my_yolov8.yaml','./yolov8s.pt','./yolov8m.pt','./yolov8s_2.pt'],
                ['./my_yolov11.yaml','./yolo11s.pt','./yolo11m.pt','./yolo11s_2.pt'],
                ['./my_yolov12.yaml','./yolo12s.pt','./yolo12m.pt','./yolo12s_2.pt']
            ]
    
    # 根据命令行参数选择模型配置和权重
    model_name = my_model[args.model_index][0]
    
    # 如果提供了自定义权重路径，则使用它
    if args.custom_weight:
        model_pt = args.custom_weight
    else:
        model_pt = my_model[args.model_index][args.weight_index]
    
    # 加载模型
    model = YOLO(model_name).load(model_pt)
    
    # 准备设备参数
    devices = [int(d) for d in args.device.split(',')]
    
    # 开始训练
    model.train(
        data='data.yaml',
        imgsz=args.imgsz,
        lr0=args.lr0,
        epochs=args.epochs,
        single_cls=True,
        batch=args.batch,
        workers=16,
        device=devices,
        project=f'{args.project}/{model_name[:-5]}',
        name=model_pt[0:-3],
        amp=True,
        cache=True,
        save_period=10
    )
