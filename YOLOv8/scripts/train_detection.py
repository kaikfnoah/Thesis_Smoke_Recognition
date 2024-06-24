import os
import sys
import shutil
from ultralytics import YOLO

def save_model(name, model_path):
    dir_path = f'../data/saved_models'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    
    # Save to folder
    SAVE_PATH = f'../data/saved_models/detection_{name}/best.pt'
    shutil.copy(model_path, SAVE_PATH)

def main(argv):
    if len(argv) < 2:
        print("Usage: python train_detection.py [ijmond/rise]")
        return
    name = argv[1]
    if name not in ('ijmond', 'rise'):
        print("Usage: python train_detection.py [ijmond/rise]")
        return
    
    model_path = '../data/pre_trained_models/yolov8x.pt'
    data_path = f'../data/{name}/annotated/data.yaml'
    
    model = YOLO(model_path)
    model.train(data=data_path,
                batch=32,
                epochs=250,
                # device=[0, 1, 2, 3],
                seed=0)
    
    save_model(name, 'runs/detect/train/weights/best.pt')

if __name__ == "__main__":
    main(sys.argv)