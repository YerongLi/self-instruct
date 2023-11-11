import cv2

def count_frames(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

# Usage
input_video_path = 'mmpose-main/Demonstration Videos/Phase II Lesson 1 Elaboration Identifying Additional Element.mp4'
frame_count = count_frames(input_video_path)
print("Total frames:", frame_count)