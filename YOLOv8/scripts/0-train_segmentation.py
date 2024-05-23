import sys
from ultralytics import YOLO

def main(argv):
    if len(argv) < 2:
        print("Usage: python 0-select_frames.py [ijmond/rise]")
        return
    name = argv[1]
    if name not in ('ijmond', 'rise'):
        print("Usage: python 0-select_frames.py [ijmond/rise]")
        return
    
    model_path = '../data/pre_trained_models/yolov8x.pt'
    data_path = f'../data/{name}/annotated/data.yaml'
    
    model = YOLO(model_path)
    model.train(data=data_path,
                batch=32,
                epochs=100,
                device=[0, 1, 2, 3],
                seed=0)

if __name__ == "__main__":
    main(sys.argv)
    
# model = YOLO('runs/detect/train3/weights/best.pt')
# prediction = model('../data/rise/annotated/test/images/vid-36-0-frame35_jpg.rf.bdcf6df5ff9729fc8b3c47ef8eccde61.jpg')
# for pred in prediction:
#     print(pred.boxes.conf)
#     print(pred.boxes.cls)
#     pred.save(filename="resul1.jpg")