# https://github.com/google/mediapipe.git
# !pip install -q mediapipe==0.10.0
# !wget -q -O efficientdet.tflite -q https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
#@markdown We implemented some functions to visualize the object detection results. <br/> Run the following cell to activate the functions.
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image
# !wget -q -O image.jpg https://storage.googleapis.com/mediapipe-tasks/object_detector/cat_and_dog.jpg

def process_video(input_video_path, output_path):
    # STEP 2: Create an ObjectDetector object.
    base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)

    # Read video file
    cap = cv2.VideoCapture(input_video_path)
    frame_count = 0

    while cap.isOpened():
        if frame_count <= 10 or frame_count > 100: 
            frame_count+= 1
            continue

        # ret, frame = cap.read()
        ret, frame = cap.read()
        image = mp.Image(data=frame, image_format=mp.ImageFormat.SRGB)
        # if not ret:
        #     break

        frame_count += 1
        print(frame_count)
        if frame_count > 100: break

        detection_result = detector.detect(image)

        # STEP 5: Process the detection result. In this case, visualize it.
        image_copy = np.copy(image.numpy_view())
        annotated_image = visualize(image_copy, detection_result)
        # print(f"Processing frame {frame_count}")
        print(annotated_image)
        # # Convert frame to RGB
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # # Create MediaPipe Image
        # image = mp.Image(
        #     data=frame_rgb
        # )

        # # Detect objects in the frame.
        # detection_result = detector.detect(image)

        # Process the detection result and save the annotated frame.
        output_file_path = f"{output_path}/frame_{frame_count:04d}.png"
        cv2.imwrite(output_file_path, annotated_image)

    cap.release()



# Process the video and save annotated frames
input_video_path = 'Phase II Lesson 1 Elaboration Identifying Additional Element.mp4'
output_directory = 'annotated_frames'
process_video(input_video_path, output_directory)
