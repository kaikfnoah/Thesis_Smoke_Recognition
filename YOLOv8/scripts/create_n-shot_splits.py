import os
import cv2
import sys
import json
import shutil

def create_test_set(data, dir, gold_pos, gold_neg ):
    # Create test
    SPLIT_DIR = os.path.join(dir, 'test')
    if not os.path.exists(SPLIT_DIR):
        os.mkdir(SPLIT_DIR)
    
    # Create positive, negative
    for i in ('positive', 'negative'):
        OUT_DIR = os.path.join(SPLIT_DIR, i)
        if not os.path.exists(OUT_DIR):
            os.mkdir(OUT_DIR)

    for row in data:
        file_name = row['file_name']
        
        if row['label_state_admin'] == 47:
            gold_pos.add(file_name)
        if row['label_state_admin'] == 32:
            gold_neg.add(file_name)
        # # you have to open the source  file in binary mode with 'rb'
        # source_file = f'../data/ijmond/frames/' + file_name + '.jpg'

        # if row['label'] == 0:
        #     SAVE_PATH = os.path.join(SPLIT_DIR, 'negative/') + file_name + '.jpg'
        #     shutil.copy(source_file, SAVE_PATH)
        # else:
        #     SAVE_PATH = os.path.join(SPLIT_DIR, 'positive/') + file_name + '.jpg'
        #     shutil.copy(source_file, SAVE_PATH)
    return gold_pos, gold_neg


def main(argvs):
    SPLITS_DIR = f'../data/ijmond/n-shot/1-shot'
    
    gold_pos, gold_neg = set(), set()
    if not os.path.exists(SPLITS_DIR):
        os.mkdir(SPLITS_DIR)
    
    for type in ('train', 'test', 'validation'):
    
        # Load data
        with open(f'../data/ijmond/splits/metadata_{type}_split_0_by_camera.json') as f:
            data = json.load(f)
        
        # Run yolo data structure creation
        gold_pos, gold_neg = create_test_set(data, SPLITS_DIR, gold_pos, gold_neg)
    print(gold_neg)
    print(gold_pos)
            

if __name__ == "__main__":
    main(sys.argv)