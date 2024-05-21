import sys
from ultralytics import YOLO

def main(argv):
    model_path = '../../back-end/data/pretrained_models/yolov8x-seg.pt'
    data_path = '../data/annotated_data/RISE/data.yaml'
    
    model = YOLO(model_path)
    model.train(data=data_path,
                batch=32,
                epochs=100,
                device=[0, 1, 2, 3],
                seed=0)

if __name__ == "__main__":
    main(sys.argv)