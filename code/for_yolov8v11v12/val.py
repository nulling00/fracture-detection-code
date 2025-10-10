import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    my_model = [
        ["./my_yolov8.yaml", "./yolov8s.pt", "./yolov8m.pt"],
        ["./my_yolov11.yaml", "./yolov11s.pt", "./yolov11m.pt"],
        ["./my_yolov12.yaml", "./yolov12s.pt", "./yolov12m.pt"],
    ]
    model_name = my_model[0][0]
    model_pt = my_model[0][1]

    model = YOLO("result/" + model_name[2:-5] + model_pt[1:-3] + "/weights/best.pt")
    model.val(
        data="data.yaml",
        imgsz=640,
        batch=16,
        split="val",
        workers=10,
        device="4",
    )
