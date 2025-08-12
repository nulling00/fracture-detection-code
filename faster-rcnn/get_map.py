import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from frcnn import FRCNN
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map

# 配置参数
map_mode = 0
classes_path = 'model_data/voc_classes.txt'
MINOVERLAP = 0.5
map_vis = False
VOCdevkit_path = '/mnt/data/yanmingshuo/CODE/faster-rcnn-pytorch-master/VOCdevkit'
data_for_test='test.txt'
map_out_path = 'map_in_res'
# model_path   = '/mnt/data/yanmingshuo/CODE/faster-rcnn-pytorch-master/logs_for_vgg_4/ep005-loss0.572-val_loss0.573.pth'
model_path   = '/mnt/data/yanmingshuo/CODE/faster-rcnn-pytorch-master/model_data/trained_pth/best_epoch_weights_resnet50.pth'
# backbone     = "vgg"
backbone     = "resnet50"

# 加载类别
with open(classes_path) as f:
    class_names = [line.strip() for line in f]

# 准备输出目录
os.makedirs(map_out_path, exist_ok=True)
os.makedirs(f'{map_out_path}/ground-truth', exist_ok=True)
os.makedirs(f'{map_out_path}/detection-results', exist_ok=True)
if map_vis:
    os.makedirs(f'{map_out_path}/images-optional', exist_ok=True)

# 读取测试图像ID
with open(f'{VOCdevkit_path}/VOC2007/ImageSets/Main/'+data_for_test) as f:
    image_ids = [line.strip() for line in f]

# 预测结果生成
if map_mode in [0, 1]:
    print("Loading model...")
    frcnn = FRCNN(confidence=0.01, nms_iou=0.5,model_path = model_path,backbone = backbone)  # 需确保FRCNN类已正确导入
    print("Generating predictions...")
    for image_id in tqdm(image_ids):
        img_path = f'{VOCdevkit_path}/VOC2007/JPEGImages/{image_id}.jpg'
        img = Image.open(img_path)
        if map_vis:
            img.save(f'{map_out_path}/images-optional/{image_id}.jpg')
        frcnn.get_map_txt(image_id, img, class_names, map_out_path)

# 真实标签生成
if map_mode in [0, 2]:
    print("Generating ground truth...")
    for image_id in tqdm(image_ids):
        with open(f'{map_out_path}/ground-truth/{image_id}.txt', 'w') as f:
            root = ET.parse(f'{VOCdevkit_path}/VOC2007/Annotations/{image_id}.xml').getroot()
            for obj in root.findall('object'):
                if obj.find('difficult') is not None and obj.find('difficult').text == '1':
                    difficult = True
                else:
                    difficult = False
                    
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                    
                bbox = obj.find('bndbox')
                coords = [bbox.find(coord).text for coord in ['xmin', 'ymin', 'xmax', 'ymax']]
                
                line = f"{obj_name} {' '.join(coords)}"
                f.write(f"{line} difficult\n" if difficult else f"{line}\n")

# mAP计算
if map_mode in [0, 3]:
    print(f"Calculating mAP (IoU@{MINOVERLAP})...")
    get_map(MINOVERLAP, True, score_threhold=0.50, path=map_out_path)  # 需确保get_map函数已正确导入

if map_mode == 4:
    print("Calculating COCO mAP...")
    get_coco_map(class_names=class_names, path=map_out_path)  # 需确保get_coco_map函数已正确导入