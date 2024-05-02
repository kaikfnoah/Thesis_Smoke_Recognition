import sys
from ultralytics import YOLO
    
def main(argv):
    model = YOLO('/projects/0/prjs1005/back-end/data/pretrained_models/yolov8n-cls.pt')
    model.train(data='/projects/0/prjs1005/back-end/data/yolo',
                batch=16,
                imgsz=1088,
                epochs=100,
                device=[0, 1, 2, 3],
                seed=0)

if __name__ == "__main__":
    main(sys.argv)