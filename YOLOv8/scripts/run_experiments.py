import os
import sys
import json
import shutil
import random

from ultralytics import YOLO

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

    for m in os.listdir(dir_path):
        model_path = dir_path + '/' + m
        model = YOLO(model_path)
        
        model.train(data=data_path,
                    batch=-1,
                    epochs=30,
                    seed=0,
                    verbose=False)


def save_best_model(n_shot, best_model_path, n_config):
    dir_path = f'../data/saved_models/{n_shot}'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    
    SAVE_PATH = f'../data/saved_models/{n_shot}/best.pt'
    shutil.copy(best_model_path, SAVE_PATH)
    
    # Rename 
    os.rename(f'../data/saved_models/{n_shot}/best.pt',
              f'../data/saved_models/{n_shot}/{n_config}.pt')


def test_models(n_shot, n_config):
    dir_path = 'runs/classify'    
    best_f1, best_p, best_r = 0, 0, 0
    avg_f1, avg_p, avg_r = 0, 0, 0
    
    best_model = ''

    for fldr in os.listdir(dir_path):
        model_path = os.path.join(dir_path, fldr) + '/weights/best.pt'
        model = YOLO(model_path)
        
        tp, tn, fp, fn = 0, 0, 0, 0
        for type in ('positive', 'negative'):
            results = model(f'../data/ijmond/n-shot/{n_shot}/test/{type}/', verbose=False)

            for result in results:
                y_pred = result.probs.top1
                if type == 'positive' and y_pred == 1:
                    tp += 1
                elif type == 'positive' and y_pred == 0:
                    fn += 1
                elif type == 'negative' and y_pred == 1:
                    fp += 1
                else:
                    tn += 1
        try:
            p = tp / (tp + fp)
        except ZeroDivisionError:
            p = 0
        try:
            r = tp / (tp + fn)
        except ZeroDivisionError:
            r = 0
        try:
            f1 = 2 * ((p * r) / (p + r))
        except ZeroDivisionError:
            f1 = 0
            
        if f1 > best_f1:
            best_f1 = f1
            best_p = p
            best_r = r
            best_model = model_path
        
        avg_p += p
        avg_r += r
        avg_f1 += f1

    print('##################################')
    print(f'Best model: {best_model}')
    print(f'Precision: {best_p}')
    print(f'Recall: {best_r}')
    print(f'F1-score: {best_f1}')
    print('Averages:')
    print(f'Precision: {avg_p / 6}')
    print(f'Recall: {avg_r / 6}')
    print(f'F1-score: {avg_f1 / 6}')
    print('##################################\n')
    
    # Save the best model
    save_best_model(n_shot, best_model, n_config)


def remove_folders(n_shot):
    shutil.rmtree('runs/classify/')
    shutil.rmtree(f'../data/ijmond/n-shot/{n_shot}')


def main(argv):
    if len(argv) < 2:
        print("Usage: python run_experiments.py [0-shot/1-shot/3-shot/6-shot/9-shot]")
        return
    n_shot = argv[1]
    if n_shot not in ('0-shot', '1-shot', '3-shot', '6-shot', '9-shot'):
        print("Usage: python run_experiments.py [0-shot/1-shot/3-shot/6-shot/9-shot]")
        return

    # Create config file for n-shot
    n_configs = 5    
    create_config_file(n_shot, n_configs)
    
    for i in range(n_configs):
        
        print(f'Starting config {i}')
        
        # Create temp n-shot dir
        create_dir(i, n_shot)
        print('Finished creating directories.')

        # Train all models
        train_models(n_shot)
        print('Finished training all models.')
        
        # Test models and save best
        print(f'Starting testing')
        test_models(n_shot, i)
        
        # Remove other folders
        remove_folders(n_shot)
        print('Finished removing folders\n\n\n\n')
            

if __name__ == "__main__":
    main(sys.argv)