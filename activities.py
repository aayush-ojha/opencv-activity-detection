import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
import logging

@dataclass
class ActivityThresholds:
    STANDING_THRESHOLD = 0.15
    SITTING_THRESHOLD = 0.15
    RAISING_HANDS_THRESHOLD = 0.2
    JUMPING_THRESHOLD = 0.25
    RUNNING_THRESHOLD = 0.2
    WALKING_THRESHOLD = 0.15
    TYPING_THRESHOLD = 0.15
    WAVING_THRESHOLD = 0.2
    CLAPPING_THRESHOLD = 0.2

class ActivityConfidence:
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4

class ActivityRecognizer:
    def __init__(self):
        self.thresholds = ActivityThresholds()
        self.previous_positions = []
        self.smoothing_window = 5
        self.confidence_history = []
        self.confidence_window = 10
        self._setup_logging()

    def _setup_logging(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def smooth_keypoints(self, keypoints: List[List[float]]) -> np.ndarray:
        try:
            self.previous_positions.append(keypoints)
            if len(self.previous_positions) > self.smoothing_window:
                self.previous_positions.pop(0)
            return np.mean(self.previous_positions, axis=0)
        except Exception as e:
            self.logger.error(f"Error smoothing keypoints: {e}")
            return np.array(keypoints)

    def smooth_confidence(self, confidence: float) -> float:
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > self.confidence_window:
            self.confidence_history.pop(0)
        return np.mean(self.confidence_history)

def recognize_activity(keypoints: List[List[float]]) -> Tuple[str, float]:
    NOSE = 0
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    if len(keypoints) < 17:
        return "Unknown", 0.0

    try:
        smoothed_keypoints = ActivityRecognizer().smooth_keypoints(keypoints)
        
        # Calculate velocity vectors for dynamic activities
        if len(ActivityRecognizer.previous_positions) >= 2:
            velocity = np.linalg.norm(
                smoothed_keypoints - ActivityRecognizer.previous_positions[-2], 
                axis=1
            )
        else:
            velocity = np.zeros(len(keypoints))

        nose = smoothed_keypoints[NOSE]
        left_wrist = smoothed_keypoints[LEFT_WRIST]
        right_wrist = smoothed_keypoints[RIGHT_WRIST]
        left_hip = smoothed_keypoints[LEFT_HIP]
        right_hip = smoothed_keypoints[RIGHT_HIP]
        left_knee = smoothed_keypoints[LEFT_KNEE]
        right_knee = smoothed_keypoints[RIGHT_KNEE]
        left_ankle = smoothed_keypoints[LEFT_ANKLE]
        right_ankle = smoothed_keypoints[RIGHT_ANKLE]

        min_confidence = 0.3
        activity_confidence = 0.0

        if all(keypoint[2] > min_confidence for keypoint in [nose, left_hip, right_hip, left_knee, right_knee]):
            hip_height = (left_hip[1] + right_hip[1]) / 2
            knee_height = (left_knee[1] + right_knee[1]) / 2
            height_diff = abs(hip_height - knee_height)

            if hip_height < knee_height:
                activity_confidence = min(1.0, height_diff * 2)
                if activity_confidence > ActivityConfidence.MEDIUM:
                    return "Standing", activity_confidence

            if hip_height > knee_height:
                activity_confidence = min(1.0, height_diff * 2)
                if activity_confidence > ActivityConfidence.MEDIUM:
                    return "Sitting", activity_confidence

            if (left_wrist[2] > min_confidence and 
                right_wrist[2] > min_confidence and 
                left_wrist[1] < nose[1] and 
                right_wrist[1] < nose[1]):
                return "Raising Hands", activity_confidence
            

            if (left_ankle[2] > min_confidence and 
                right_ankle[2] > min_confidence and 
                left_ankle[1] < knee_height and 
                right_ankle[1] < knee_height):
                return "Jumping", activity_confidence


            if (left_ankle[2] > min_confidence and 
                right_ankle[2] > min_confidence):
                leg_diff = abs(left_ankle[1] - right_ankle[1])
                if leg_diff > 0.2:
                    arm_diff = abs(left_wrist[1] - right_wrist[1])
                    if arm_diff > 0.2:
                        return "Running", activity_confidence
                    return "Walking", activity_confidence

        # Add velocity-based confidence
        if activity == "Running" or activity == "Walking":
            activity_confidence *= min(1.0, np.mean(velocity) * 2)
        
        return "Unknown", 0.0
            
    except Exception as e:
        logging.error(f"Error in activity recognition: {e}")
        return "Unknown", 0.0

