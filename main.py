import cv2
import tensorflow as tf
import numpy as np
import threading
import tensorflow_hub as hub
from activities import recognize_activity
from datetime import datetime
import sys
import logging
import json
import queue
from pathlib import Path
import yaml
import atexit

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logging.error(f"GPU setup error: {e}")

class Config:
    def __init__(self):
        self.load_config()

    def load_config(self):
        config_path = Path(__file__).parent / 'config.yaml'
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                self.__dict__.update(config)
        except FileNotFoundError:
            self.set_defaults()

    def set_defaults(self):
        self.MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
        self.FRAME_SKIP = 15
        self.CONFIDENCE_THRESHOLD = 0.5
        self.WINDOW_NAME = 'MoveNet Motion Tracking'
        self.ACTIVITY_HISTORY_SIZE = 30
        self.LOG_LEVEL = 'INFO'

config = Config()

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('activity_detection.log'),
        logging.StreamHandler()
    ]
)

class ActivityTracker:
    def __init__(self, history_size=30):
        self.history = queue.Queue(maxsize=history_size)
        self.current_activity = "Unknown"
        self.confidence = 0.0

    def update(self, activity, confidence):
        try:
            if self.history.full():
                self.history.get()
            self.history.put((activity, confidence))
            self.current_activity = self._get_most_common_activity()
        except Exception as e:
            logging.error(f"Error updating activity: {e}")

    def _get_most_common_activity(self):
        activities = list(self.history.queue)
        if not activities:
            return "Unknown"
        return max(set(a[0] for a in activities), key=lambda x: sum(1 for a in activities if a[0] == x))

try:
    model = hub.load(config.MODEL_URL)
    movenet = model.signatures['serving_default']
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    sys.exit(1)

try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Cannot open camera")
except Exception as e:
    logging.error(f"Camera initialization failed: {e}")
    sys.exit(1)

class MotionDetector:
    def __init__(self):
        self.lock = threading.Lock()
        self.shared_keypoints = []
        self.activity_tracker = ActivityTracker(config.ACTIVITY_HISTORY_SIZE)
        self.frame_buffer = queue.Queue(maxsize=2)
        self.running = True
        atexit.register(self.cleanup)
        self.frame_queue = queue.Queue(maxsize=config.FRAME_BUFFER_SIZE)
        self.result_queue = queue.Queue(maxsize=config.FRAME_BUFFER_SIZE)
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()

    def cleanup(self):
        self.running = False
        with self.lock:
            self.shared_keypoints = []
        logging.info("MotionDetector cleanup completed")

    def process_frame(self, frame):
        if not self.running:
            return [], "Unknown"
        try:
            input_data = cv2.resize(frame, (192, 192))
            input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = tf.convert_to_tensor(input_data, dtype=tf.int32)
            
            outputs = movenet(input_data)
            keypoints = outputs['output_0'].numpy()[0, 0, :, :]

            with self.lock:
                self.shared_keypoints = keypoints.tolist()
            
            activity, confidence = recognize_activity(self.shared_keypoints)
            self.activity_tracker.update(activity, confidence)
            return self.shared_keypoints, self.activity_tracker.current_activity
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return [], "Unknown"

    def _process_queue(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                result = self.process_frame(frame)
                self.result_queue.put(result)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Processing error: {e}")

def cam_runner():
    detector = MotionDetector()
    stats = {'fps': 0, 'processed_frames': 0, 'dropped_frames': 0}
    count = 0
    last_time = datetime.now()
    fps = 0

    try:
        while detector.running:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture frame from camera.")
                break

            current_time = datetime.now()
            if (current_time - last_time).seconds >= 1:
                fps = count
                count = 0
                last_time = current_time

            if count % config.FRAME_SKIP == 0:
                threading.Thread(target=detector.process_frame, args=(frame.copy(),), daemon=True).start()
            
            with detector.lock:
                prev_x, prev_y = None, None
                for keypoint in detector.shared_keypoints:
                    y, x, confidence = keypoint[:3]
                    if confidence > config.CONFIDENCE_THRESHOLD:
                        x_coord = int(x * frame.shape[1])
                        y_coord = int(y * frame.shape[0])
                        cv2.circle(frame, (x_coord, y_coord), 5, (0, 255, 0), -1)
                        if prev_x is not None and prev_y is not None:
                            cv2.line(frame, (prev_x, prev_y), (x_coord, y_coord), (255, 0, 0), 2)
                        prev_x, prev_y = x_coord, y_coord

                activity = detector.activity_tracker.current_activity
                cv2.putText(frame, f"Activity: {activity}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"FPS: {fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Dropped: {stats['dropped_frames']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if activity != "Unknown":
                    logging.info(activity)

            count += 1
            stats['processed_frames'] += 1
            if not detector.frame_buffer.full():
                detector.frame_buffer.put(frame)
            else:
                stats['dropped_frames'] += 1

            cv2.imshow(config.WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        logging.info("Gracefully shutting down...")
    except Exception as e:
        logging.error(f"Fatal error in cam_runner: {str(e)}", exc_info=True)
    finally:
        detector.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        logging.info(f"Session stats: {stats}")

if __name__ == '__main__':
    cam_runner()

