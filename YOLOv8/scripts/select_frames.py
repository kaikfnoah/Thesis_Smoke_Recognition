import os
import sys
import cv2
import json
import torch
import numpy as np

from ultralytics import YOLO


def conf_score_frame(model, frame):    
    results = model(frame, verbose=False)
    
    classes = results[0].boxes.cls
    pos_positions = torch.where(classes == 0)[0]
    
    # Check non-empty
    if len(pos_positions) > 0:
        return max(results[0].boxes.conf[pos_positions])
    else:
        return 0


def main(argvs):
    if len(argvs) < 2:
        print("Usage: python3 select_frames.py [ijmond/rise]")
        return
    name = argvs[1]
    if name not in ('ijmond', 'rise'):
        print("Usage: python3 select_frames.py [ijmond/rise]")
        return
    
    model_path = f'../data/saved_models/detection_{name}.pt'
    model = YOLO(model_path)
    
    # All video file names
    video_dir = f'../data/{name}/videos'
    all_video_filenames = np.array(os.listdir(video_dir))
    
    # Check if dir exists
    if not os.path.exists(f'../data/{name}/frames/'):
        os.mkdir(f'../data/{name}/frames/')
    
    # Videos with smoke
    metadata_dir = f'../data/{name}/splits'
    pos_video_filenames = []
    for file in os.listdir(os.fsencode(metadata_dir)):
        filename = os.fsdecode(file)
        
        if filename in ('metadata_test_split_0_by_camera.json',
                        'metadata_train_split_0_by_camera.json',
                        'metadata_validation_split_0_by_camera.json'):
            with open(os.path.join(metadata_dir, filename), 'r') as file:
                data = json.load(file)
                
            # Iterate over each dictionary in the list
            for item in data:
                if item['label'] == 1:
                    pos_video_filenames.append(item['file_name'] + '.mp4')
    pos_video_filenames = np.array(pos_video_filenames)
    
    # Loop through all IJmond videos
    print("#################### STARTING VIDEO FRAME SELECION ####################")
    print(f"Using model: {model_path}")
    
    no_pos_detection = []
    for filename in all_video_filenames:
        file_path = os.path.join(video_dir, filename)
        vidcap = cv2.VideoCapture(file_path)
        
        # Check pos or neg
        success, frame = vidcap.read()
        if filename in pos_video_filenames:
            largest_area, count = 0, 1
            best_frame = frame
            while success:
                area = conf_score_frame(model=model, frame=frame)
                
                if area > largest_area:
                    best_frame = frame
                    largest_area = area
                if count // 2 == 18:
                    middle_frame = frame
                success, frame = vidcap.read()
                count += 1
            
            # Chech if there are images (in positive dir) without smoke detection
            if largest_area == 0:
                no_pos_detection.append(filename)
                cv2.imwrite(f"../data/{name}/frames/{filename[:-4]}.jpg", middle_frame)
            else:
                cv2.imwrite(f"../data/{name}/frames/{filename[:-4]}.jpg", best_frame)  
            print(f'Finished POSITIVE video {filename}')
        else:
            # Select the first frame
            cv2.imwrite(f"../data/{name}/frames/{filename[:-4]}.jpg", frame)  
            print(f'Finished NEGATIVE video {filename}')
            
        vidcap.release()
    print(f"There are {len(no_pos_detection)} videos in which no smoke was detected (while there should have been).")
    print(no_pos_detection)

if __name__ == "__main__":
    main(sys.argv)