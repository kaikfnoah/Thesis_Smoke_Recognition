import os
import sys
from util import *
import numpy as np
from optical_flow.optical_flow import OpticalFlow
from multiprocessing import Pool

thread = "1"
os.environ["MKL_NUM_THREADS"] = thread
os.environ["NUMEXPR_NUM_THREADS"] = thread
os.environ["OMP_NUM_THREADS"] = thread
os.environ["VECLIB_MAXIMUM_THREADS"] = thread
os.environ["OPENBLAS_NUM_THREADS"] = thread
import cv2 as cv
cv.setNumThreads(12)

# Process videos into rgb frame files and optical flow files
# The file format is numpy.array
def main(argv):
    _, split, type = argv
    
    if type == 'val':
        metadata_path = f'../data/split/metadata_validation_split_{split}_by_camera.json'
    else:
        metadata_path = f'../data/split/metadata_{type}_split_{split}_by_camera.json'
        
    rgb_dir = f'../data/yolo/{type}/'
    num_workers = 1

    # Check for saving directories and create if they don't exist
    check_and_create_dir(rgb_dir)

    metadata = load_json(metadata_path)
    p = Pool(num_workers)
    for video_data in metadata:
        p.apply(compute_and_save_flow, args=(video_data, type))
    p.close()
    p.join()
    print("Done process_videos.py")
    

def extract_frames(video_path, output_dir, filename, num_frames=32):
    # Open the video file
    cap = cv.VideoCapture(video_path)

    # Get the total number of frames in the video
    num_frames_total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # Calculate the step size for the sliding window based on the desired number of frames
    step_size = max(num_frames_total // num_frames, 1)

    # Initialize the current frame number and the frame counter
    frame_num = 0
    count = 0

    # Loop through the video frames and extract a frame every step_size frames
    while True:
        # Read the current frame
        ret, frame = cap.read()

        # If we've reached the end of the video, break out of the loop
        if not ret:
            break

        # If the current frame number is a multiple of the step size, save the frame
        if frame_num % step_size == 0:
            # Construct the output filename
            unique_filename = str(uuid.uuid4()) + ".jpg"
            output_path = os.path.join(output_dir, unique_filename)

            # Save the frame to disk
            cv.imwrite(output_path, frame)

            # Increment the frame counter
            count += 1

        # Increment the current frame number
        frame_num += 1

    # Release the video file
    cap.release()

    # Return the total number of frames extracted
    return count


def compute_and_save_flow(video_data, type):
    video_dir = "../data/videos/"
    file_name = video_data["file_name"]
    
    label = video_data['label']
    if label == 0:
        out_dir = f'../data/yolo/{type}/negative/'
    else:
        out_dir = f'../data/yolo/{type}/positive/'

    vid_dir = str(video_dir + file_name + ".mp4")

    if not is_file_here(vid_dir):
        return

    # Check if the output dir exists and create if not
    check_and_create_dir(out_dir)

    # Extract frames from the video file
    num_frames = 36
    extract_frames(vid_dir, out_dir, num_frames)


if __name__ == "__main__":
    main(sys.argv)
