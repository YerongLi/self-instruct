import cv2
import os
import argparse

def save_frame_at_timestamp(input_video_path, timestamp, output_folder):
    cap = cv2.VideoCapture(input_video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)  # Convert timestamp to milliseconds
    success, frame = cap.read()
    if success:
        # Find the smallest number that does not exist in the output folder
        # i = 1
        # while os.path.exists(os.path.join(output_folder, f"{i:03d}.jpg")):
        #     i += 1
        output_file_path = os.path.join(output_folder, f"{input_video_path.split('/')[-1]}_{frame}.jpg")
        cv2.imwrite(output_file_path, frame)
        print(f"Frame saved successfully as {output_file_path}.")
    else:
        print("Failed to save frame.")
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save a frame from a video at a specific timestamp.")
    parser.add_argument("input_video_path", help="Path to the input video file")
    parser.add_argument("timestamp", type=float, help="Timestamp in seconds")
    parser.add_argument("--output_folder", default="img", help="Path to the output folder to save the images")

    args = parser.parse_args()

    save_frame_at_timestamp(args.input_video_path, args.timestamp, args.output_folder)