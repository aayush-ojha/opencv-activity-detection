import cv2
import tensorflow as tf
import numpy as np
import threading
import tensorflow_hub as hub
from activities import recognize_activity
from datetime import datetime
import sys

class Config:
    MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    FRAME_SKIP = 15
    CONFIDENCE_THRESHOLD = 0.5
    WINDOW_NAME = 'MoveNet Motion Tracking'

try:
    model = hub.load(Config.MODEL_URL)
    movenet = model.signatures['serving_default']
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)

try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Cannot open camera")
except Exception as e:
    print(f"Camera initialization failed: {e}")
    sys.exit(1)

lock = threading.Lock()
shared_keypoints = []
activity = "Unknown"

def check_movement(frame):
    global shared_keypoints
    try:
        input_data = cv2.resize(frame, (192, 192))
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0)
        input_data = tf.convert_to_tensor(input_data, dtype=tf.int32)
        
        outputs = movenet(input_data)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]

        with lock:
            shared_keypoints = keypoints.tolist()
    except Exception as e:
        print(f"Error in movement detection: {e}")

def cam_runner():
    global activity
    count = 0
    last_time = datetime.now()
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera.")
                break

            current_time = datetime.now()
            if (current_time - last_time).seconds >= 1:
                fps = count
                count = 0
                last_time = current_time

            if count % Config.FRAME_SKIP == 0:
                threading.Thread(target=check_movement, args=(frame.copy(),), daemon=True).start()
            
            with lock:
                prev_x, prev_y = None, None
                for keypoint in shared_keypoints:
                    y, x, confidence = keypoint[:3]
                    if confidence > Config.CONFIDENCE_THRESHOLD:
                        x_coord = int(x * frame.shape[1])
                        y_coord = int(y * frame.shape[0])
                        cv2.circle(frame, (x_coord, y_coord), 5, (0, 255, 0), -1)
                        if prev_x is not None and prev_y is not None:
                            cv2.line(frame, (prev_x, prev_y), (x_coord, y_coord), (255, 0, 0), 2)
                        prev_x, prev_y = x_coord, y_coord

                activity = recognize_activity(shared_keypoints)
                cv2.putText(frame, f"Activity: {activity}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"FPS: {fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if activity != "Unknown":
                    print(activity)

            count += 1
            cv2.imshow(Config.WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Gracefully shutting down...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    cam_runner()

