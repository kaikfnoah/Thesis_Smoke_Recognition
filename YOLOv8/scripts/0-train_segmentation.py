import sys
from ultralytics import YOLO

def main(argv):
    if len(argv) < 2:
        print("Usage: python 0-train_segmentation.py [ijmond/rise]")
        return
    name = argv[1]
    if name not in ('ijmond', 'rise'):
        print("Usage: python 0-train_segmentation.py [ijmond/rise]")
        return
    
    model_path = '../data/pre_trained_models/yolov8x.pt'
    data_path = f'../data/{name}/annotated/data.yaml'
    
    model = YOLO(model_path)
    model.train(data=data_path,
                batch=32,
                epochs=250,
                device=[0, 1, 2, 3],
                seed=0)

if __name__ == "__main__":
    main(sys.argv)