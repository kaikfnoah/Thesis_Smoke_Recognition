import os
import sys
import json
import shutil
import random
import numpy as np

from ultralytics import YOLO
from sklearn.metrics import f1_score, precision_score, recall_score

GOLD_NEG = ['TUZbSqcLVtU-1', 'wg6C4QySFD4-2', 'Dj2I-b0zTWU-1', '5IUJvUa_eso-1', 'zgTCcwKiEXQ-2', 'S0pXH97m558-2', 
            'ILO3p3GdlQg-0', 'rPBe7y54Yxs-1', 'whmQf-ud0RY-2', 'Ajsd1Q08DmA-0', 'S0pXH97m558-0', 'zuX9sR4wYKc-0', 
            'R0esnWDfaP8-0', '93aaWg7BzaA-1', '1NKYOpxE90A-1', '4CA-G8uUtyA-1', 'HgNO44L2INY-0', '8Y9cfUKvqvU-0', 
            'aINMnqmwSUg-3', 'cS2Fk8M2Pj4-0', 'ubv9JZ-k4bU-3', 'j81VqAaG_3I-0', 'EACU2K9A44M-2', 'zuX9sR4wYKc-3', 
            'q1RdTphU5yU-2', 'A9W8G55JucU-2', '5ljnXgkRfaM-2', 'rPBe7y54Yxs-0', '23x-vGYMbec-2', 'crAL3CrEt3s-3', 
            'M7arDQZyAG4-0', '_JzCaMSLzK0-0', 'kSuSGf9_dqA-0', 'Cjs_IIDUDVM-2', 'A9W8G55JucU-1', 'bu7USw70eXs-0', 
            'kpOrtl8MizM-0', '6_j636Zv6fs-0', 'FfRfRhkohqs-0', 'NcY7dtGpyus-2', '1NKYOpxE90A-3', 'VQ5OZbuwXgg-2', 
            'nKIyHSJDzds-2', 'rsBRGyFrPwM-2', 'dQbSLpPnyXU-2', '_JzCaMSLzK0-1', 'zJpGtu3jf_w-0', 'FfRfRhkohqs-2', 
            'A9W8G55JucU-0']

GOLD_POS = ['5nGAnsaKbA8-0', 'TuF5Qnk3TkI-2', 'Qv9-nS5BloI-2', 'b2_XE0-ZufQ-2', 'fD_2Qr7HN4c-3', 'DoNtmf5UBkY-2', 
            'p5klKMpdREI-3', 'z-MaBhPlmz0-0', 'A-nnAPBqYcs-1', 'mP45MTdzdoI-4', 'fD_2Qr7HN4c-1', '5X1jbx4Hii4-3', 
            'ygySU_Bdvok-3', '5PurGkmy0aw-3', 'WarrpDCyYmY-3', 'TUZbSqcLVtU-4', 'omlxGgMOjz8-3', '5CiaQ_ppyHo-3', 
            '5ay369iRsi0-1', 'TuF5Qnk3TkI-3', '4CA-G8uUtyA-2', '5ljnXgkRfaM-3', 'S1Juid0Wi8M-3', 'Hvk6xygq27o-3', 
            'UgBeBBYzTbQ-1', 'DoIBg8LAfWI-3', '3P0bkMMGSCc-0', 'wP78ukOa9W8-1', 'glLcsYu7mbM-3', 'j7QUpjlW1x4-1', 
            'qG2GLIybaRQ-1', 'FGRFWwvDBTo-2', 'zgTCcwKiEXQ-3', 'bTuGDZ7RVUI-1', 'lG7hIctXNiI-0', 'qG2GLIybaRQ-0', 
            'FYZNq4uWCxM-1', 'TUZbSqcLVtU-3', 'VPlcWn5Xkyg-1', 'lL9ungbqqCU-0', 'aBA6JSWKW3c-0', 'Q1yTGkLuAc0-0', 
            'NfEhg5sAXPI-0', 'gLXdasSk8JU-3', 'LXB7SXlDvq8-0', 'OCgjZK8LUmE-1', 'INYCPw7nzLc-1', 'yeP5MvlMCbM-3']


random.seed(1)


def create_config_file(n_shot, n_configs):
    n = int(n_shot[0])
    gp, gn = GOLD_POS.copy(), GOLD_NEG.copy()
    
    with open('../data/ijmond/splits/metadata.json', 'r') as f:
        data = json.load(f)
    all_file_names = [x['file_name'] for x in data]
    
    config_n_shot = []
    for i in range(n_configs):
        
        # Define the training samples for this config
        train_pos = random.sample(gp, n)
        train_neg = random.sample(gn, n)
        gp, gn = [x for x in gp if x not in train_pos], [x for x in gn if x not in train_neg]
        
        # Find the remaining files for the test and val split
        remaining_files = [x for x in all_file_names if (x not in train_pos and x not in train_neg)]
        remaining_neg, remaining_pos = [], []
        for row in data:
            if row['file_name'] in remaining_files and row['label'] == 0:
                remaining_neg.append(row['file_name'])
            elif row['file_name'] in remaining_files and row['label'] == 1:
                remaining_pos.append(row['file_name'])
        
        # Define the split for testing and validation
        split = 0.7
        test_pos = random.sample(remaining_pos, int(len(remaining_pos) * split))
        test_neg = random.sample(remaining_neg, int(len(remaining_neg) * split))
        
        val_pos = [x for x in remaining_pos if (x not in test_pos and x not in train_pos)]
        val_neg = [x for x in remaining_neg if (x not in test_neg and x not in train_neg)]
        
        config_n_shot.append(
            {
                f'Config {i}': 
                    {'train': [
                        {'positive': train_pos,
                         'negative': train_neg}
                        ],
                     'test': [
                         {'positive': test_pos,
                          'negative': test_neg}
                         ],
                     'val' : [
                         {'positive': val_pos, 
                          'negative': val_neg}
                         ]
                    }
                }
            )
    
    path = f"../data/ijmond/setups/{n_shot}.json"
    
    # Serializing json
    json_object = json.dumps(config_n_shot, indent=4)
    with open(path, "w") as outfile:
        outfile.write(json_object)


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
                source_file = f'../data/ijmond/frames/' + file + '.jpg'
                SAVE_PATH = out_path + '/' + file + '.jpg'
                shutil.copy(source_file, SAVE_PATH)


def train_models(n_shot):
    dir_path = '../data/saved_models/0-shot'
    data_path = f'../data/ijmond/n-shot/{n_shot}'

    # Enter what model you want to run n-shot with
    model_path = dir_path + '/S0.pt'
    model = YOLO(model_path)
    
    model.train(data=data_path,
                batch=-1,
                epochs=20,
                imgsz=640,
                seed=0,
                single_cls=True,
                verbose=True)


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
    
    
def test_models(n_shot, n_config=None):
    if n_shot != '0-shot':
        model_path = 'runs/classify/train/weights/best.pt'
        model = YOLO(model_path)
        
        y_true, y_pred = np.array([]), np.array([])
        for type in ('positive', 'negative'):

            results = model(f'../data/ijmond/n-shot/{n_shot}/test/{type}/', verbose=False)
            for result in results:
                
                if type == 'positive':
                    y_true = np.append(y_true, 1)
                else:
                    y_true = np.append(y_true, 0)
                y_pred = np.append(y_pred, result.probs.top1)
        
        # Save model
        save_model(n_shot, model_path, n_config)
        
        print('##################################')
        print(f'Precision: {precision_score(y_true, y_pred, average="weighted")}')
        print(f'Recall: {recall_score(y_true, y_pred, average="weighted")}')
        print(f'F1-score: {f1_score(y_true, y_pred, average="weighted")}')
        print('##################################\n')
        
        return precision_score(y_true, y_pred, average='weighted'), recall_score(y_true, y_pred, average='weighted'), f1_score(y_true, y_pred, average='weighted')

    else:
        avg_p, avg_r, avg_f1, n_models = 0, 0, 0, 0
        for m in os.listdir('../data/saved_models/0-shot'):
            model_path = '../data/saved_models/0-shot' + f'/{m}'
            model = YOLO(model_path)
        
            y_true, y_pred = np.array([]), np.array([])
            for type in ('positive', 'negative'):

                results = model(f'../data/ijmond/n-shot/{n_shot}/test/{type}/', verbose=False)
                for result in results:
                    
                    if type == 'positive':
                        y_true = np.append(y_true, 1)
                    else:
                        y_true = np.append(y_true, 0)
                    y_pred = np.append(y_pred, result.probs.top1)
                    
            avg_p += precision_score(y_true, y_pred, average="weighted")
            avg_r += recall_score(y_true, y_pred, average="weighted")
            avg_f1 += f1_score(y_true, y_pred, average="weighted")
            n_models += 1

            print('##################################')
            print(f'Precision: {precision_score(y_true, y_pred, average="weighted")}')
            print(f'Recall: {recall_score(y_true, y_pred, average="weighted")}')
            print(f'F1-score: {f1_score(y_true, y_pred, average="weighted")}')
            print('##################################\n')
            
        print('##################################')
        print('AVERAGES')
        print(f'Precision: {avg_p / n_models}')
        print(f'Recall: {avg_r / n_models}')
        print(f'F1-score: {avg_f1 / n_models}')
        print('##################################\n')


def remove_folders(n_shot):    
    shutil.rmtree('runs/classify/')
    shutil.rmtree(f'../data/ijmond/n-shot/{n_shot}')
    
    
def save_results(n_shot, i):
    dir_path = f'../data/saved_results/{n_shot}/c{i}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    SAVE_PATH = f'../data/saved_results/{n_shot}/c{i}/results.csv'
    shutil.copy('runs/classify/train/results.csv', SAVE_PATH)


def main(argv):
    if len(argv) < 2:
        print("Usage: python3 experiments.py [0-shot/1-shot/3-shot/6-shot/9-shot]")
        return
    n_shot = argv[1]
    if n_shot not in ('0-shot', '1-shot', '3-shot', '6-shot', '9-shot'):
        print("Usage: python3 experiments.py [0-shot/1-shot/3-shot/6-shot/9-shot]")
        return

    if n_shot == '0-shot':
        # Test the model
        print(f'Starting testing')
        test_models(n_shot)

    else:
        # Create config file for n-shot
        n_configs = 5    
        create_config_file(n_shot, n_configs)
        
        avg_p, avg_r, avg_f1 = 0, 0, 0
        for i in range(n_configs):
            
            print(f'Starting config {i}')
            
            # Create temp n-shot dir
            create_dir(i, n_shot)
            print('Finished creating directories.')

            # Train the model on all IJMD splits
            train_models(n_shot)
            print('Finished training all models.')
            
            # Test and save best
            print(f'Starting testing')
            p, r, f1 = test_models(n_shot, i)
            
            avg_p += p
            avg_r += r
            avg_f1 += f1
            
            # Save loss and lr
            save_results(n_shot, i)
            
            # Remove other folders
            remove_folders(n_shot, i)
            print('Finished removing folders\n\n\n\n')
        
        # Print averages
        print('##################################')
        print('AVERAGES')
        print(f'Precision: {avg_p / n_configs}')
        print(f'Recall: {avg_r / n_configs}')
        print(f'F1-score: {avg_f1 / n_configs}')
        print('##################################\n')

if __name__ == "__main__":
    main(sys.argv)