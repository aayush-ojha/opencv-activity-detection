import unittest
import numpy as np
from activities import ActivityRecognizer, recognize_activity

class TestActivityRecognition(unittest.TestCase):
    def setUp(self):
        self.recognizer = ActivityRecognizer()

    def test_recognize_standing(self):
        # Mock keypoints for standing position
        keypoints = np.zeros((17, 3))
        # Set nose position
        keypoints[0] = [0.5, 0.2, 0.9]
        # Set hip positions
        keypoints[11] = [0.4, 0.5, 0.9]
        keypoints[12] = [0.6, 0.5, 0.9]
        # Set knee positions
        keypoints[13] = [0.4, 0.8, 0.9]
        keypoints[14] = [0.6, 0.8, 0.9]

        activity, confidence = recognize_activity(keypoints)
        self.assertEqual(activity, "Standing")
        self.assertGreater(confidence, 0.6)

    def test_invalid_keypoints(self):
        keypoints = np.zeros((5, 3))  # Invalid keypoint count
        activity, confidence = recognize_activity(keypoints)
        self.assertEqual(activity, "Unknown")
        self.assertEqual(confidence, 0.0)

if __name__ == '__main__':
    unittest.main()
