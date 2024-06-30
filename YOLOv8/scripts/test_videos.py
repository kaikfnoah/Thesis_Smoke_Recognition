import os
import sys
import shutil
import json
import numpy as np

from ultralytics import YOLO
from sklearn.metrics import f1_score, precision_score, recall_score


def create_dir(n, n_shot):
    # Load configs
    with open(f'../data/ijmond/setups/{n_shot}.json') as g:
        config = json.load(g)
    config = config[n][f'Config {n}']
    
    shot_path = f'../data/ijmond/n-shot/{n_shot}'
    if not os.path.exists(shot_path):
        os.makedirs(shot_path)
    
    # Setup folders
    for type in ('test', 'train', 'val'):
        
        for name in ('positive', 'negative'):
            out_path = os.path.join(shot_path, type, name)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            
            files = config[type][0][name]
            for file in files:
                source_file = f'../data/ijmond/videos/' + file + '.mp4'
                SAVE_PATH = out_path + '/' + file + '.mp4'
                shutil.copy(source_file, SAVE_PATH)
                
def remove_folders(n_shot):    
    shutil.rmtree(f'../data/ijmond/n-shot/{n_shot}')
                
def main(argvs):
    if len(argvs) < 2:
        print("Usage: python3 test_videos.py [0-shot/1-shot/3-shot/6-shot/9-shot]")
        return
    n_shot = argvs[1]
    if n_shot not in ('0-shot', '1-shot', '3-shot', '6-shot', '9-shot'):
        print("Usage: python3 test_videos.py [0-shot/1-shot/3-shot/6-shot/9-shot]")
        return
    
    # Create n-shot dir
    create_dir(2, n_shot)

    # Best model
    model = YOLO(f'../data/saved_models/{n_shot}/2.pt')

    y_true, y_pred = np.array([]), np.array([])
    for type in ('positive', 'negative'):
        
        PATH = f'../data/ijmond/n-shot/{n_shot}/test/{type}'
        for file in os.listdir(PATH):
            result = model(PATH + '/' + file, verbose=False, stream=True)
            
            n_pos = 0
            for res in result:
                if res.probs.top1 == 1:
                    n_pos += 1
            
            if type == 'positive':
                y_true = np.append(y_true, 1)
            else:
                y_true = np.append(y_true, 0)
                
            if n_pos > 0:
                y_pred = np.append(y_pred, 1)
            else:
                y_pred = np.append(y_pred, 0)
                
            
    print('##################################')
    print(f'Precision: {precision_score(y_true, y_pred, average="weighted")}')
    print(f'Recall: {recall_score(y_true, y_pred, average="weighted")}')
    print(f'F1-score: {f1_score(y_true, y_pred, average="weighted")}')
    print('##################################\n')
    
    # Remove folder
    remove_folders(n_shot)

if __name__ == "__main__":
    main(sys.argvs)