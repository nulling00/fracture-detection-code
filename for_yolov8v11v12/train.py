import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':  
    # model = YOLO('yolov8m.pt')
    my_model = [
                ['./my_yolov8.yaml','./yolov8s.pt','./yolov8m.pt','./yolov8s_2.pt'],
                ['./my_yolov11.yaml','./yolo11s.pt','./yolo11m.pt','./yolo11s_2.pt'],
                ['./my_yolov12.yaml','./yolo12s.pt','./yolo12m.pt','./yolo12s_2.pt']
            ]
    model_name = my_model[2][0]
    model_pt = my_model[2][3]
    # model_pt = '/mnt/data/ningling/CODE/YOLO/for_yolov8v11v12/result/my_yolov11/yolo11s2/weights/best.pt'

    # model_pt='./'
    model=YOLO(model_name).load(model_pt)
    

    # 指定保存路径为 result 文件夹
    model.train(
        data='data.yaml',  # 直接使用字典配置
        # model='my_yolov8.yaml',
        # model='my_yolov11.yaml',
        
        imgsz=640,
        lr0=0.002,
        # epochs=300,
        # epochs=50,
        # epochs=100,
        epochs=300,


        single_cls=True,  # 将所有类视为一个类别

        batch=32,
        workers=16,
        device=[4],  # 只使用第5,6块GPU（索引4）
        project = 'result0618_4/'+model_name[:-5],
        name = model_pt[0:-3],
        amp = True,
        cache = True,

        save_period=10      # 每25个epoch保存一次检查点
    )