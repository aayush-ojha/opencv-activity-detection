import unittest
import time
import numpy as np
from main import MotionDetector
from activities import recognize_activity

class PerformanceTest(unittest.TestCase):
    def setUp(self):
        self.detector = MotionDetector()
        self.sample_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_processing_speed(self):
        iterations = 100
        start_time = time.time()
        
        for _ in range(iterations):
            self.detector.process_frame(self.sample_frame)
            
        elapsed_time = time.time() - start_time
        fps = iterations / elapsed_time
        
        self.assertGreater(fps, 15, "Processing speed below 15 FPS")

if __name__ == '__main__':
    unittest.main()
