import sys
from ultralytics import YOLO

def main(argv):
    if len(argv) < 2:
        print("Usage: python 5-fine_tune.py [0-shot/1-shot/3-shot/6-shot/9-shot]")
        return
    name = argv[1]
    if name not in ('0-shot', '1-shot', '3-shot', '6-shot', '9-shot'):
        print("Usage: python 4-test.py [0-shot/1-shot/3-shot/6-shot/9-shot]")
        return

    dir_path = '../data/saved_models/0-shot' 
    data_path = f'../data/ijmond/n-shot/{name}'

    model_path = dir_path + '/' + 'S5.pt'
    model = YOLO(model_path)
    
    model.train(data=data_path,
                batch=-1,
                epochs=30,
                seed=0)

if __name__ == "__main__":
    main(sys.argv)