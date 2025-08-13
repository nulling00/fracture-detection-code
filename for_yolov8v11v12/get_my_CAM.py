# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import os
# import torch
# import cv2
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from ultralytics import YOLO
# from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# # 设置中文显示
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


# def setup_device():
#     """设置计算设备"""
#     return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def load_model(weights_path, device):
#     """加载 YOLOv11 模型"""
#     try:
#         model = YOLO(weights_path)
#         model.to(device).eval()
#         print(f"成功加载模型: {weights_path}")
#         return model
#     except Exception as e:
#         print(f"加载模型失败: {e}")
#         exit(1)


# def preprocess_image(img_path, input_size=(640, 640)):
#     """预处理图像"""
#     img = cv2.imread(img_path)
#     if img is None:
#         raise FileNotFoundError(f"无法读取图像: {img_path}")
    
#     # 调整图像大小并保持纵横比
#     img = letterbox(img, new_shape=input_size)[0]
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # 归一化并转换为张量
#     img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
#     img_tensor = img_tensor.unsqueeze(0)
    
#     return img, img_tensor


# def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
#     """调整图像大小并添加填充（YOLO 风格）"""
#     shape = img.shape[:2]  # 当前形状 [高度, 宽度]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)
    
#     # 缩放比例 (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # 只缩小，不放大（用于测试）
#         r = min(r, 1.0)
    
#     # 计算填充
#     ratio = r, r  # 宽高缩放比例
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 宽高填充量
#     if auto:  # 最小矩形
#         dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # 填充为32的倍数
#     elif scaleFill:  # 拉伸
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽高缩放比例
    
#     dw /= 2  # 将填充量分为左右两侧
#     dh /= 2  # 将填充量分为上下两侧
    
#     if shape[::-1] != new_unpad:  # 缩放图像
#         img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边界填充
    
#     return img, ratio, (dw, dh)


# def get_target_layers(model, layer_idx):
#     """获取目标层"""
#     try:
#         # 对于 YOLOv11，通常最后几层是检测头，之前的是特征提取层
#         target_layers = [model.model.model[layer_idx]]
#         print(f"选择目标层: {model.model.model[layer_idx]}")
#         return target_layers
#     except IndexError:
#         print(f"错误: 层索引 {layer_idx} 超出范围")
#         print(f"模型总层数: {len(model.model.model)}")
#         exit(1)

# def generate_heatmap(model, target_layers, img_tensor, method, target_category=None, device='cpu'):
#     """生成热力图，针对YOLO模型进行优化"""
#     # 确保模型处于评估模式
#     model.model.eval()
    
#     # 创建一个用于获取特定输出的辅助函数
#     class YOLOWrapper(torch.nn.Module):
#         def __init__(self, model):
#             super().__init__()
#             self.model = model
            
#         def forward(self, x):
#             # 获取模型输出
#             outputs = self.model(x)
#             # 对于YOLO，通常取最后一层输出
#             # 这里可能需要根据实际模型结构调整
#             return outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    
#     wrapped_model = YOLOWrapper(model)
    
#     # 选择热力图生成方法
#     if method == 'gradcam':
#         cam = GradCAM(model=wrapped_model, target_layers=target_layers)
#     elif method == 'gradcam++':
#         cam = GradCAMPlusPlus(model=wrapped_model, target_layers=target_layers)
#     else:
#         raise ValueError(f"不支持的热力图方法: {method}")
    
#     # 设置目标类别（如果指定）
#     targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
    
#     # 确保输入张量需要梯度
#     img_tensor = img_tensor.to(device).requires_grad_(True)
    
#     try:
#         # 计算热力图
#         grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
#     finally:
#         # 手动释放资源
#         if hasattr(cam, 'activations_and_grads'):
#             cam.activations_and_grads.release()
    
#     return grayscale_cam[0, :]  # 返回第一张图像的热力图

# def visualize_and_save(model, img_path, img, img_tensor, grayscale_cam, output_path, show_boxes=False, conf_threshold=0.25):
#     """可视化热力图并保存结果"""
#     # 创建输出文件夹
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
#     # 将热力图叠加到原图
#     rgb_img = np.float32(img) / 255.0
#     visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
#     # 如果需要，在热力图上绘制检测框
#     if show_boxes:
#         results = model(img_path, conf=conf_threshold)
#         boxes = results[0].boxes.cpu().numpy()
#         names = model.names
        
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = box.conf[0]
#             cls = int(box.cls[0])
#             label = f"{names[cls]} {conf:.2f}"
            
#             # 在热力图上绘制边界框
#             cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(visualization, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
#     # 保存结果
#     cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
#     print(f"热力图已保存至: {output_path}")
    
#     # 显示结果（如果在交互式环境中）
#     if os.environ.get('DISPLAY') or True:
#         plt.figure(figsize=(12, 6))
#         plt.subplot(1, 2, 1)
#         plt.imshow(img)
#         plt.title('原始图像')
#         plt.axis('off')
        
#         plt.subplot(1, 2, 2)
#         plt.imshow(visualization)
#         plt.title(f'热力图 ({method})')
#         plt.axis('off')
        
#         plt.tight_layout()
#         plt.savefig(output_path.replace('.jpg', '_combined.jpg'), dpi=300, bbox_inches='tight')
#         plt.close()


# def process_single_image(model, target_layers, device, img_path, output_folder, method, target_class, show_boxes, conf_threshold):
#     """处理单张图像"""
#     try:
#         img, img_tensor = preprocess_image(img_path)
#         img_tensor = img_tensor.to(device)
        
#         # 生成热力图
#         grayscale_cam = generate_heatmap(
#             model=model,
#             target_layers=target_layers,
#             img_tensor=img_tensor,
#             method=method,
#             target_category=target_class,
#             device=device
#         )
        
#         # 生成输出路径
#         img_name = os.path.basename(img_path)
#         output_name = f"heatmap_{method}_{img_name}"
#         if target_class is not None:
#             output_name = f"heatmap_{method}_class{target_class}_{img_name}"
#         output_path = os.path.join(output_folder, output_name)
        
#         # 可视化并保存
#         visualize_and_save(
#             model=model,
#             img_path=img_path,
#             img=img,
#             img_tensor=img_tensor,
#             grayscale_cam=grayscale_cam,
#             output_path=output_path,
#             show_boxes=show_boxes,
#             conf_threshold=conf_threshold
#         )
        
#     except Exception as e:
#         print(f"处理图像 {img_path} 时出错: {e}")


# def process_image_folder(model, target_layers, device, image_folder, output_folder, method, target_class, show_boxes, conf_threshold):
#     """处理图像文件夹"""
#     for img_name in os.listdir(image_folder):
#         if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
#             img_path = os.path.join(image_folder, img_name)
#             process_single_image(
#                 model=model,
#                 target_layers=target_layers,
#                 device=device,
#                 img_path=img_path,
#                 output_folder=output_folder,
#                 method=method,
#                 target_class=target_class,
#                 show_boxes=show_boxes,
#                 conf_threshold=conf_threshold
#             )


# if __name__ == "__main__":
#     # 在这里配置所有参数
#     config = {
#         # 模型配置
#         "weights_path": '/mnt/data/ningling/CODE/YOLO/for_yolov8v11v12/result0618_4/my_yolov11/yolo11s_2/weights/best.pt',  # 替换为你的模型权重路径
        
#         # 输入配置（选择处理单张图像或整个文件夹）
#         # "image_path": "/mnt/data/ningling/DATASET/dataset_yolo_all_fracture/test/images/000001.jpg",  # 单张图像路径
#         "image_path": None,
#         "image_folder": "/mnt/data/ningling/DATASET/dataset_yolo_all_fracture/test/images",  # 图像文件夹路径（如果处理多张图像）
        
#         # 输出配置
#         "output_folder": "./heatmaps",  # 热力图输出文件夹
        
#         # 热力图配置
#         "method": "gradcam++",  # 热力图方法: gradcam 或 gradcam++
#         "target_layer": -4,  # 目标层索引（通常为倒数第4层，即检测头前的特征层）
#         "target_class": 0,  # 指定类别ID（如0表示person），None表示所有类别
        
#         # 可视化配置
#         "show_boxes": True,  # 是否在热力图上显示检测框
#         "conf_threshold": 0.25,  # 检测阈值
        
#         # 设备配置
#         "use_cuda": True  # 是否使用GPU
#     }
    
#     # 检查输入配置
#     if not config["image_path"] and not config["image_folder"]:
#         raise ValueError("请配置 image_path 或 image_folder")
    
#     # 设置设备
#     device = setup_device()
#     if not config["use_cuda"]:
#         device = torch.device('cpu')
#     print(f"使用设备: {device}")
    
#     # 加载模型
#     model = load_model(config["weights_path"], device)
    
#     # 获取目标层
#     target_layers = get_target_layers(model, config["target_layer"])
    
#     # 处理图像
#     if config["image_path"]:
#         process_single_image(
#             model=model,
#             target_layers=target_layers,
#             device=device,
#             img_path=config["image_path"],
#             output_folder=config["output_folder"],
#             method=config["method"],
#             target_class=config["target_class"],
#             show_boxes=config["show_boxes"],
#             conf_threshold=config["conf_threshold"]
#         )
#     else:
#         process_image_folder(
#             model=model,
#             target_layers=target_layers,
#             device=device,
#             image_folder=config["image_folder"],
#             output_folder=config["output_folder"],
#             method=config["method"],
#             target_class=config["target_class"],
#             show_boxes=config["show_boxes"],
#             conf_threshold=config["conf_threshold"]
#         )




# import os
# from PIL import Image
# import torch
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from ultralytics import YOLO

# def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):
#     '''
#     绘制 Class Activation Map
#     :param model: 加载好权重的Pytorch model
#     :param img_path: 测试图片路径
#     :param save_path: CAM结果保存路径
#     :param transform: 输入图像预处理方法
#     :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
#     :return:
#     '''
#     # 图像加载&预处理
#     img = Image.open(img_path).convert('RGB')
#     if transform:
#         img = transform(img)
#     img = img.unsqueeze(0)
 
#     # 获取模型输出的feature/score
#     model.eval()
#     features = model.features(img)
#     output = model.classifier(features)
 
#     # 为了能读取到中间梯度定义的辅助函数
#     def extract(g):
#         global features_grad
#         features_grad = g
 
#     # 预测得分最高的那一类对应的输出score
#     pred = torch.argmax(output).item()
#     pred_class = output[:, pred]
 
#     features.register_hook(extract)
#     pred_class.backward() # 计算梯度
 
#     grads = features_grad   # 获取梯度
 
#     pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
 
#     # 此处batch size默认为1，所以去掉了第0维（batch size维）
#     pooled_grads = pooled_grads[0]
#     features = features[0]
#     # 512是最后一层feature的通道数
#     for i in range(512):
#         features[i, ...] *= pooled_grads[i, ...]
 
#     # 以下部分同Keras版实现
#     heatmap = features.detach().numpy()
#     heatmap = np.mean(heatmap, axis=0)
 
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap)
 
#     # 可视化原始热力图
#     if visual_heatmap:
#         plt.matshow(heatmap)
#         plt.show()

#     img = cv2.imread(img_path)  # 用cv2加载原始图像
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
#     heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
#     superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
#     cv2.imwrite(save_path, superimposed_img)

# if __name__ == "__main__":
#     model_path='/mnt/data/ningling/CODE/YOLO/for_yolov8v11v12/result0618_4/my_yolov11/yolo11s_2/weights/best.pt'  # 替换为你的模型权重路径

#     model = YOLO(model_path)
#     draw_CAM(
#         model,
#         img_path="/mnt/data/ningling/DATASET/dataset_yolo_all_fracture/test/images/000001.jpg",  # 单张图像路径
#         save_path= "./heatmaps",  # 热力图输出文件夹

#     )

import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """调整图像大小并添加填充（YOLO风格）"""
    shape = img.shape[:2]  # 当前形状 [高度, 宽度]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # 缩放比例 (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只缩小，不放大（用于测试）
        r = min(r, 1.0)
    
    # 计算填充
    ratio = r, r  # 宽高缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 宽高填充量
    if auto:  # 最小矩形
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # 填充为32的倍数
    elif scaleFill:  # 拉伸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽高缩放比例
    
    dw /= 2  # 将填充量分为左右两侧
    dh /= 2  # 将填充量分为上下两侧
    
    if shape[::-1] != new_unpad:  # 缩放图像
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边界填充
    
    return img, ratio, (dw, dh)

def draw_CAM(model, img_path, save_path, target_layer=-4, visual_heatmap=False, input_size=(640, 640)):
    '''
    为YOLO模型绘制 Class Activation Map
    :param model: 加载好权重的YOLO model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param target_layer: 目标层索引
    :param visual_heatmap: 是否可视化原始heatmap
    :param input_size: 模型输入尺寸
    :return:
    '''
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 图像加载&预处理
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    
    # 使用letterbox函数调整图像大小，保持纵横比并填充
    img_processed, ratio, (dw, dh) = letterbox(img, new_shape=input_size, auto=False)
    img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
    
    # 转换为张量
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).requires_grad_(True)
    
    # 获取目标层
    try:
        target_layers = [model.model.model[target_layer]]
        print(f"选择目标层: {target_layers[0]}")
    except IndexError:
        print(f"错误: 层索引 {target_layer} 超出范围")
        print(f"模型总层数: {len(model.model.model)}")
        return
    
    # 为了能读取到中间梯度定义的辅助函数
    features = []
    gradients = []
    
    def forward_hook(module, input, output):
        features.append(output)
    
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    
    # 注册钩子
    hook_handle = target_layers[0].register_forward_hook(forward_hook)
    hook_handle_backward = target_layers[0].register_full_backward_hook(backward_hook)  # 使用full backward hook
    
    try:
        # 设置为评估模式
        model.model.eval()
        
        # 前向传播
        try:
            # 使用model.model直接进行前向传播，获取原始特征
            outputs = model.model(img_tensor)
            
            # 打印输出信息用于调试
            print(f"模型输出类型: {type(outputs)}")
            if isinstance(outputs, (list, tuple)):
                print(f"输出列表长度: {len(outputs)}")
                for i, out in enumerate(outputs):
                    if hasattr(out, 'shape'):
                        print(f"输出[{i}]形状: {out.shape}")
                    else:
                        print(f"输出[{i}]类型: {type(out)}")
            elif hasattr(outputs, 'shape'):
                print(f"输出形状: {outputs.shape}")
            else:
                print(f"输出类型: {type(outputs)}")
        
        except RuntimeError as e:
            print(f"前向传播错误: {e}")
            print(f"输入张量形状: {img_tensor.shape}")
            # 尝试调整输入尺寸为32的倍数
            h, w = input_size
            h = (h // 32) * 32
            w = (w // 32) * 32
            print(f"尝试使用调整后的输入尺寸: ({h}, {w})")
            
            # 重新处理图像
            img_processed, ratio, (dw, dh) = letterbox(img, new_shape=(h, w), auto=False)
            img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).requires_grad_(True)
            
            outputs = model.model(img_tensor)
        
        # 处理模型输出，找到合适的目标值进行梯度计算
        detections = model(img_processed, conf=0.25)[0]
        
        if len(detections.boxes) > 0:
            # 如果有检测结果，使用最高置信度的检测框
            max_conf_idx = torch.argmax(detections.boxes.conf)
            box = detections.boxes[max_conf_idx]
            cls_idx = int(box.cls)
            conf = box.conf[0]
            print(f"使用检测结果: 类别={model.names[cls_idx]}, 置信度={conf:.4f}")
            
            # 尝试自动确定输出格式
            if isinstance(outputs, (list, tuple)):
                # 查找最可能包含检测结果的输出
                target_output = None
                for out in outputs:
                    if hasattr(out, 'shape') and out.dim() >= 3:
                        target_output = out
                        break
                
                if target_output is None:
                    raise ValueError("无法找到合适的输出张量")
                
                # 确定输出维度
                output_dims = target_output.dim()
                print(f"目标输出维度: {output_dims}")
                
                if output_dims == 3:  # [batch, boxes, 5+classes]
                    # 找到最接近的box
                    x, y, w, h = box.xywh[0]
                    grid_size = int(np.sqrt(target_output.shape[1]))  # 假设为正方形网格
                    grid_x = int(x / input_size[0] * grid_size)
                    grid_y = int(y / input_size[1] * grid_size)
                    box_idx = grid_y * grid_size + grid_x
                    
                    # 确保索引不越界
                    box_idx = min(box_idx, target_output.shape[1] - 1)
                    
                    # 创建目标张量
                    target = torch.zeros_like(target_output)
                    target[0, box_idx, 4 + cls_idx] = 1.0
                    
                    # 计算损失
                    loss = torch.sum(target_output * target)
                    
                elif output_dims >= 4:  # [batch, channels, height, width] 或更多维度
                    # 假设最后两个维度是空间维度
                    grid_h, grid_w = target_output.shape[-2:]
                    x, y, w, h = box.xywh[0]
                    grid_x = int(x / input_size[0] * grid_w)
                    grid_y = int(y / input_size[1] * grid_h)
                    
                    # 确保索引不越界
                    grid_x = min(grid_x, grid_w - 1)
                    grid_y = min(grid_y, grid_h - 1)
                    
                    # 对于不同维度的处理
                    if output_dims == 4:  # [batch, channels, height, width]
                        # 假设类别信息在channels维度
                        if target_output.shape[1] >= (5 + len(model.names)):
                            # 创建目标张量
                            target = torch.zeros_like(target_output)
                            target[0, 4 + cls_idx, grid_y, grid_x] = 1.0
                            
                            # 计算损失
                            loss = torch.sum(target_output * target)
                        else:
                            # 可能是特征图，使用最大值作为目标
                            loss = target_output[0, :, grid_y, grid_x].sum()
                    
                    elif output_dims == 5:  # [batch, anchors, channels, height, width]
                        # 假设第一个维度是anchor
                        loss = target_output[0, 0, 4 + cls_idx, grid_y, grid_x]
                    
                    else:
                        raise ValueError(f"不支持的输出维度: {output_dims}")
                else:
                    raise ValueError(f"不支持的输出维度: {output_dims}")
            
            else:
                # 对于单张量输出
                if outputs.dim() == 3:  # [batch, boxes, 5+classes]
                    # 找到最接近的box
                    x, y, w, h = box.xywh[0]
                    grid_size = int(np.sqrt(outputs.shape[1]))  # 假设为正方形网格
                    grid_x = int(x / input_size[0] * grid_size)
                    grid_y = int(y / input_size[1] * grid_size)
                    box_idx = grid_y * grid_size + grid_x
                    
                    # 确保索引不越界
                    box_idx = min(box_idx, outputs.shape[1] - 1)
                    
                    # 创建目标张量
                    target = torch.zeros_like(outputs)
                    target[0, box_idx, 4 + cls_idx] = 1.0
                    
                    # 计算损失
                    loss = torch.sum(outputs * target)
                
                else:
                    raise ValueError(f"不支持的输出维度: {outputs.dim()}")
        
        else:
            # 如果没有检测到目标，使用第一个预测的置信度
            print("未检测到目标，使用第一个预测作为目标")
            
            # 尝试自动确定输出格式
            if isinstance(outputs, (list, tuple)):
                # 查找最可能的输出
                target_output = None
                for out in outputs:
                    if hasattr(out, 'shape') and out.dim() >= 3:
                        target_output = out
                        break
                
                if target_output is None:
                    raise ValueError("无法找到合适的输出张量")
                
                # 假设第一个检测的置信度
                if target_output.dim() == 3:  # [batch, boxes, 5+classes]
                    loss = target_output[0, 0, 4]  # 置信度分数
                elif target_output.dim() >= 4:  # [batch, channels, height, width]
                    loss = target_output[0, 4, 0, 0]  # 假设第4个通道是置信度
                else:
                    raise ValueError(f"不支持的输出维度: {target_output.dim()}")
            else:
                if outputs.dim() == 3:
                    loss = outputs[0, 0, 4]
                else:
                    raise ValueError(f"不支持的输出维度: {outputs.dim()}")
        
        print(f"使用损失值: {loss.item()}")
        
        # 反向传播
        model.model.zero_grad()
        loss.backward()
        
        # 获取特征图和梯度
        if not features or not gradients:
            raise RuntimeError("无法获取特征图或梯度")
        
        feature_map = features[0].detach()
        grads = gradients[0].detach()
        
        print(f"特征图形状: {feature_map.shape}")
        print(f"梯度形状: {grads.shape}")
        
        # 计算权重（全局平均池化梯度）
        pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
        print(f"池化后梯度形状: {pooled_grads.shape}")
        
        # 修复广播问题：直接进行矩阵运算，无需循环
        # 特征图形状: [1, 256, 20, 20]
        # pooled_grads形状: [1, 256, 1, 1]
        # 直接相乘，利用广播机制
        feature_map = feature_map * pooled_grads  # 自动广播
        
        # 生成热力图（在通道维度上求平均）
        heatmap = torch.mean(feature_map, dim=1).squeeze().cpu().numpy()
        print(f"热力图原始形状: {heatmap.shape}")
        
        # ReLU激活
        heatmap = np.maximum(heatmap, 0)
        
        # 归一化
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        
        # 可视化原始热力图
        if visual_heatmap:
            plt.matshow(heatmap)
            plt.show()
        
        # 调整热力图大小以匹配原图
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # 将热力图转换为RGB格式
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 叠加热力图到原图
        superimposed_img = heatmap * 0.4 + img
        
        # 如果需要，在热力图上绘制检测框
        if len(detections.boxes) > 0:
            for box in detections.boxes.cpu().numpy():
                # 调整检测框坐标回原图尺寸
                x1, y1, x2, y2 = box.xyxy[0]
                x1 = int((x1 - dw) / ratio[0])
                y1 = int((y1 - dh) / ratio[1])
                x2 = int((x2 - dw) / ratio[0])
                y2 = int((y2 - dh) / ratio[1])
                
                conf = box.conf[0]
                cls = int(box.cls)
                label = f"{model.names[cls]} {conf:.2f}"
                
                # 绘制边界框
                cv2.rectangle(superimposed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(superimposed_img, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存结果
        cv2.imwrite(save_path, superimposed_img)
        print(f"热力图已保存至: {save_path}")
        
    except Exception as e:
        print(f"生成热力图时出错: {e}")
        # 打印更详细的错误信息
        import traceback
        traceback.print_exc()
    finally:
        # 移除钩子
        hook_handle.remove()
        hook_handle_backward.remove()

if __name__ == "__main__":
    # 配置参数
    model_path = '/mnt/data/ningling/CODE/YOLO/for_yolov8v11v12/result0618_4/my_yolov11/yolo11s_2/weights/best.pt'
    img_path = "/mnt/data/ningling/DATASET/dataset_yolo_all_fracture/test/images/000001.jpg"
    output_folder = "./heatmaps"
    input_size = (640, 640)  # 模型输入尺寸，确保是32的倍数
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 生成保存路径（添加文件名）
    img_name = os.path.basename(img_path)
    save_path = os.path.join(output_folder, f"cam_{img_name}")
    
    # 加载模型
    model = YOLO(model_path)
    
    # 尝试获取模型的输入尺寸
    try:
        # 检查模型配置
        if hasattr(model, 'model') and hasattr(model.model, 'yaml'):
            if 'imgsz' in model.model.yaml:
                input_size = model.model.yaml['imgsz']
                if isinstance(input_size, int):
                    input_size = (input_size, input_size)
                print(f"使用模型配置的输入尺寸: {input_size}")
    except:
        print(f"使用默认输入尺寸: {input_size}")
    
    # 打印模型结构信息
    print(f"模型层数: {len(model.model.model)}")
    
    # 生成热力图
    draw_CAM(
        model=model,
        img_path=img_path,
        save_path=save_path,
        target_layer=-4,  # 可尝试不同层，如-6, -8等
        visual_heatmap=False,
        input_size=input_size
    )