import os
import shutil
from ultralytics import YOLO

def save_model(n_shot, model_path, n_config):
    dir_path = f'../data/saved_models/{n_shot}'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    
    # Save to folder
    SAVE_PATH = f'../data/saved_models/{n_shot}/best.pt'
    shutil.copy(model_path, SAVE_PATH)
    
    # Rename 
    os.rename(f'../data/saved_models/{n_shot}/best.pt',
              f'../data/saved_models/{n_shot}/{n_config}.pt')
    
def remove_folders():    
    shutil.rmtree('runs/classify/')

def main():    
    dir_path = f'../data/rise/splits_yolo'
    model_path = '../data/pre_trained_models/yolov8x-cls.pt'

    for split in os.listdir(dir_path):
        data_path = os.path.join(dir_path, split)  
        
        model = YOLO(model_path)  
        model.train(data=data_path,
                    batch=-1,
                    epochs=50,
                    imgsz=640,
                    weight_decay=0.001,
                    single_cls=True,
                    dropout=0.5,
                    # device=[0, 1],
                    seed=0)

        if split == 'by_date':
            save_model('0-shot', 'runs/classify/train/weights/best.pt', 'S3')
        elif split == '2_by_camera':
            save_model('0-shot', 'runs/classify/train/weights/best.pt', f'S{str(int(split[0]) + 1)}')
        elif split == '3_by_camera':
            save_model('0-shot', 'runs/classify/train/weights/best.pt', f'S{str(int(split[0]) + 1)}')
        elif split == '4_by_camera':
            save_model('0-shot', 'runs/classify/train/weights/best.pt', f'S{str(int(split[0]) + 1)}')
        else:
            save_model('0-shot', 'runs/classify/train/weights/best.pt', f'S{str(split[0])}')
        remove_folders()

if __name__ == "__main__":
    main()