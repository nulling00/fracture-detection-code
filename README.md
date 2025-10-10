# fracture-detection-code

Official PyTorch Implementation of Deep Learning Model for Automated Detection of Pediatric Femoral Neck Fractures on Hip Radiographs:A Multicenter Study with Clinical Utility Assessment

## code

The code is divided into two parts:

- faster-rcnn: This is the implementation of the Faster-RCNN model ( resnet-50 backbone and vgg16 backbone ) for fracture detection.
- for_yolov8v11v12: This is the implementation of the YOLOv8, YOLOv11, and YOLOv12 models for fracture detection.

The structure of the code directory is as follows:
```shell
code/
|---- faster-rcnn/
|---- ---- train.py
|---- ---- test.py
|---- for_yolov8v11v12/
|---- ---- train.py
|---- ---- test.py
```

Environment:

for faster-rcnn:
```
cd ./code/faster-rcnn
pip install -r requirements.txt
```

for for_yolov8v11v12:
```
cd ./code/for_yolov8v11v12
pip install -r requirements.txt
```

## Datasets
The datasets strcutre for YOLOv8, YOLOv11, and YOLOv12 is as follows:
```shell
data/
|---- train/
|---- ---- images/
|---- ---- labels/
|---- val/
|---- ---- images/
|---- ---- labels/
|---- test/
|---- ---- images/
|---- ---- labels/
```

The datasets strcutre for Faster-RCNN is as follows:
```shell
VOC2007/
|---- Annotations/
|---- ---- *.xml
|---- ImageSets/
|---- ---- Main/
|---- *.txt
|---- JPEGImages/
|---- ---- *.jpg
```

## Environment
This codes is tested on Python 3.8.
To install requirements:

```setup
cd ./code/faster-rcnn
pip install-r requirements.txt
```

```setup
cd ./code/for_yolov8v11v12
pip install-r requirements.txt
```


### Usage
To train fasterrcnn model, you can set your parammers（such as the type of model, the number of epochs, and the learning rate and so on） in ***train_vgg.py*** and ***train_res50.py*** ,and then run:


```train
cd ./code/faster-rcnn
python train_vgg.py
python train_res50.py
```

To test fasterrcnn modelin the test set, please set your parammers in ***get_classification_metrics.py*** , run:

```test
cd ./code/faster-rcnn
python test.py
```

To train the YOLOv8, YOLOv11, and YOLOv12 models,you can set your parammers（such as the type of model, the number of epochs, and the learning rate and so on） in ***train.py*** and then run:

```train
cd ./code/for_yolov8v11v12
python train.py
```

To get the results of the YOLOv8, YOLOv11, and YOLOv12 models in the test set, please set your parammers in ***get_classification_metrics.py*** , run:

```test
cd ./code/for_yolov8v11v12
python get_classification_metrics.py*
``` 

### Result
The results of the YOLOv8, YOLOv11, YOLOv12 and Faster-RCNN models in the test set are as follows:


<!-- <table>
    <tr>
        <th> </th>
        <th>Method</th>
        <th>AUROC</th>
        <th>P</th>
        <th>R</th>
        <th>F1</th>
    </tr>
    <tr>
        <th rowspan=3>Image-level<br>Detection</th>
        <td>w/o pre</td>
        <td>99.21/99.48/99.39</td>
        <td>98.44/98.72/99.17</td>
        <td>99.21/99.14/99.21</td>
        <td>0.99/0.99/0.99</td>
    </tr>
        <td>w/ pre</td>
        <td>99.24/99.60/99.47</td>
        <td>98.44/99.14/99.44</td>
        <td>99.21/99.14/99.17</td>
        <td>0.99/0.99/0.99</td>
    </tr>
        <td>Upper Bound</td>
        <td>100.0/100.0/100.0</td>
        <td>100.0/100.0/100.0</td>
        <td>100.0/100.0/100.0</td>
        <td>1.00/1.00/1.00</td>
    </tr>
        <tr>
        <th rowspan=3>Pixel-level<br>Localization</th>
        <td>w/o pre</td>
        <td>95.59/98.18/97.13</td>
        <td>58.05/66.59/63.58</td>
        <td>72.15/77.31/75.51</td>
        <td>0.63/0.70/0.68</td>
    </tr>
        <td>w/ pre</td>
        <td>95.70/98.29/97.37</td>
        <td>58.06/66.53/63.52</td>
        <td>71.67/77.15/75.26</td>
        <td>0.63/0.70/0.68</td>
    </tr>
        <td>Upper Bound</td>
        <td>96.94/99.49/98.59</td>
        <td>59.38/68.73/65.42</td>
        <td>75.20/79.34/77.90</td>
        <td>0.65/0.73/0.70</td>
    </tr>
</table> -->

