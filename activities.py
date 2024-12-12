class ActivityConfidence:
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4

def recognize_activity(keypoints):
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
        return "Unknown"

    nose = keypoints[NOSE]
    left_wrist = keypoints[LEFT_WRIST]
    right_wrist = keypoints[RIGHT_WRIST]
    left_hip = keypoints[LEFT_HIP]
    right_hip = keypoints[RIGHT_HIP]
    left_knee = keypoints[LEFT_KNEE]
    right_knee = keypoints[RIGHT_KNEE]
    left_ankle = keypoints[LEFT_ANKLE]
    right_ankle = keypoints[RIGHT_ANKLE]

    min_confidence = 0.3
    activity_confidence = 0.0

    if all(keypoint[2] > min_confidence for keypoint in [nose, left_hip, right_hip, left_knee, right_knee]):
        hip_height = (left_hip[1] + right_hip[1]) / 2
        knee_height = (left_knee[1] + right_knee[1]) / 2
        height_diff = abs(hip_height - knee_height)

        if hip_height < knee_height:
            activity_confidence = min(1.0, height_diff * 2)
            if activity_confidence > ActivityConfidence.MEDIUM:
                return "Standing"

        if hip_height > knee_height:
            activity_confidence = min(1.0, height_diff * 2)
            if activity_confidence > ActivityConfidence.MEDIUM:
                return "Sitting"

        if (left_wrist[2] > min_confidence and 
            right_wrist[2] > min_confidence and 
            left_wrist[1] < nose[1] and 
            right_wrist[1] < nose[1]):
            return "Raising Hands"
        

        if (left_ankle[2] > min_confidence and 
            right_ankle[2] > min_confidence and 
            left_ankle[1] < knee_height and 
            right_ankle[1] < knee_height):
            return "Jumping"


        if (left_ankle[2] > min_confidence and 
            right_ankle[2] > min_confidence):
            leg_diff = abs(left_ankle[1] - right_ankle[1])
            if leg_diff > 0.2:
                arm_diff = abs(left_wrist[1] - right_wrist[1])
                if arm_diff > 0.2:
                    return "Running"
                return "Walking"

    return "Unknown"

