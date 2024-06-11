import os
import sys
import json
from ultralytics import YOLO

def main(argv):
    if len(argv) < 2:
        print("Usage: python 3-train_classifier.py [ijmond/rise]")
        return
    name = argv[1]
    if name not in ('ijmond', 'rise'):
        print("Usage: python 3-train_classifier.py [ijmond/rise]")
        return
    
    dir_path = f'../data/{name}/splits_yolo'
    model_path = '../data/pre_trained_models/yolov8x-cls.pt'

    for split in os.listdir(dir_path):
        data_path = os.path.join(dir_path, split)  
        
        model = YOLO(model_path)  
        model.train(data=data_path,
                    batch=-1,
                    epochs=10,
                    imgsz=640,
                    weight_decay=0.001,
                    single_cls=True,
                    dropout=0.5,
                    device = [0, 1],
                    seed=0)

if __name__ == "__main__":
    main(sys.argv)