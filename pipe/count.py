import cv2
import os

video_dir = "mmpose-main/Demonstration Videos"
video_files = []

# List all files in the directory
for file in os.listdir(video_dir):
    # Check if the file is a video file (e.g., .mp4, .avi, .mov, etc.)
    if file.endswith(".mp4") or file.endswith(".avi") or file.endswith(".mov"):
        video_files.append(os.path.join(video_dir, file))
def count_frames(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames
# Print the list of video files
for video_file in video_files:
    print(video_file)
    frame_count = count_frames(video_file)
    print("Total frames:", frame_count)


