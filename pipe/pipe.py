import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
def process_video(input_video_path):
    # Create a face detector instance with the video mode
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path='/path/to/model.task'),
        running_mode=VisionRunningMode.VIDEO
    )
    with FaceDetector(options) as detector:
        cap = cv2.VideoCapture(input_video_path)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1
            print(f"Processing frame {frame_count}")

            # Convert frame to RGB (MediaPipe requires RGB input)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to MediaPipe's Image object
            mp_image = mp.solutions.drawing_utils._image_from_image(frame_rgb)

            # Get timestamp for the frame (in milliseconds)
            frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Perform face detection on the provided single image
            face_detector_result = detector.process(mp_image, frame_timestamp_ms)

            # Perform operations with the face detector results (e.g., drawing bounding boxes)
            # For example:
            # for detection in face_detector_result.detections:
            # Retrieve and process the face detection data

            # Show or save annotated frames
            # ...

        cap.release()

# Process the video
input_video_path = 'your_video.mp4'
process_video(input_video_path)
