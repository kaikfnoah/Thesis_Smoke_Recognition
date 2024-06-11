import sys
import urllib.request
from util import *


# Download all videos in the metadata json file
# This script is specific for the url generated by the thumbnail server
# IMPORTANT: this script should not be used for the smoke recognition pipeline
def main(argv):
    vm = load_json("../data/metadata.json")["data"]
    video_root_path = "../data/videos_all/"
    rp_180 = video_root_path + "180/"
    rp_320 = video_root_path + "320/"
    check_and_create_dir(rp_180)
    check_and_create_dir(rp_320)
    problem_videos = []
    for v in vm:
        fn_180 = v["file_name"]
        fn_320 = fn_180.replace("-180-180-", "-320-320-")
        fp_180 = rp_180 + fn_180 + ".mp4"
        fp_320 = rp_320 + fn_320 + ".mp4"
        url_180 = v["url_root"] + v["url_part"]
        url_320 = url_180.replace("width=180", "width=320")
        url_320 = url_320.replace("height=180", "height=320")
        if not is_file_here(fp_180):
            print("Download video", fn_180)
            try:
                urllib.request.urlretrieve(url_180, fp_180)
            except:
                print("\tError downloading video", fn_180)
                problem_videos.append(fn_180)
        if not is_file_here(fp_320):
            print("Download video", fn_320)
            try:
                urllib.request.urlretrieve(url_320, fp_320)
            except:
                print("\tError downloading video", fn_320)
                problem_videos.append(fn_320)
    print("Done download_all_videos.py")
    if len(problem_videos) > 0:
        print("The following videos were not downloaded due to errors:")
        for v in problem_videos:
            print("\tv\n")


if __name__ == "__main__":
    main(sys.argv)
