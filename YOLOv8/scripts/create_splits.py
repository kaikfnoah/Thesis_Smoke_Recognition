import os
import json
import shutil

def create_yolo_struture(data, type, name, YOLO_DIR):
    if type == 'validation':
        type = 'val'
    
    # Create train, test, val
    SPLIT_DIR = os.path.join(YOLO_DIR, type)
    if not os.path.exists(SPLIT_DIR):
        os.mkdir(SPLIT_DIR)
    
    # Create positive, negative
    for i in ('positive', 'negative'):
        OUT_DIR = os.path.join(SPLIT_DIR, i)
        if not os.path.exists(OUT_DIR):
            os.mkdir(OUT_DIR)
            
    for row in data:
        file_name = row['file_name']
        source_file = f'../data/{name}/frames/' + file_name + '.jpg'

        if row['label'] == 0:
            SAVE_PATH = os.path.join(SPLIT_DIR, 'negative/') + file_name + '.jpg'
            shutil.copy(source_file, SAVE_PATH)
        else:
            SAVE_PATH = os.path.join(SPLIT_DIR, 'positive/') + file_name + '.jpg'
            shutil.copy(source_file, SAVE_PATH)


def main():
    SPLITS_DIR = f'../data/rise/splits_yolo'
    
    if not os.path.exists(SPLITS_DIR):
        os.mkdir(SPLITS_DIR)
    
    SPLITS = ['0_by_camera', '1_by_camera', '2_by_camera', '3_by_camera', '4_by_camera', 'by_date']

    for split in SPLITS:
        
        print(f'######### CREATING SPLIT {split}')
        YOLO_DIR = os.path.join(SPLITS_DIR, split)
        if not os.path.exists(YOLO_DIR):
            os.mkdir(YOLO_DIR)
        
        for type in ('train', 'test', 'validation'):
        
            # Load data
            with open(f'../data/rise/splits/metadata_{type}_split_{split}.json') as f:
                data = json.load(f)
            
            # Run yolo data structure creation
            create_yolo_struture(data, type, 'rise', YOLO_DIR)
            print(f'#### FINISHED {type}')
            

if __name__ == "__main__":
    main()