# OpenCV Activity Detection

Real-time human activity recognition using OpenCV and TensorFlow's MoveNet model. This project provides a robust solution for detecting and classifying human activities through computer vision.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Features

- Real-time human pose detection using MoveNet
- Multiple activity recognition (Standing, Sitting, Walking, Running, etc.)
- GPU acceleration support
- Performance optimization with multi-threading
- Configurable parameters through YAML
- Comprehensive logging system
- Unit testing and performance benchmarks

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, for GPU acceleration)
- Webcam or video input device

## Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/aayush-ojha/opencv-activity-detection.git
   cd opencv-activity-detection
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python main.py
   ```

## Configuration

The project can be customized through `config.yaml`:

```yaml
# Core settings
MODEL_URL: "https://tfhub.dev/google/movenet/singlepose/lightning/4"
FRAME_SKIP: 15
CONFIDENCE_THRESHOLD: 0.5

# Activity thresholds
STANDING_THRESHOLD: 0.15
SITTING_THRESHOLD: 0.15
WALKING_THRESHOLD: 0.15
# ... other thresholds

# Performance settings
ENABLE_GPU: true
THREAD_POOL_SIZE: 4
```

## Supported Activities

- Standing
- Sitting
- Walking
- Running
- Jumping
- Raising Hands
- Typing
- Waving
- Clapping

## Architecture

```
opencv-activity-detection/
├── main.py              # Application entry point
├── activities.py        # Activity recognition logic
├── config.yaml         # Configuration file
├── requirements.txt    # Project dependencies
├── tests/             # Test suite
│   ├── test_activities.py
│   └── performance_test.py
└── logs/              # Application logs
```

## Development

### Running Tests
```bash
# Run unit tests
pytest tests/

# Run performance tests
pytest tests/performance_test.py

# Generate coverage report
pytest --cov=./ tests/
```

### Code Style
The project follows PEP 8 guidelines. Format code using:
```bash
black .
```

## Performance Optimization

The application includes several optimization features:
- GPU acceleration (when available)
- Frame skipping for better performance
- Multi-threaded frame processing
- Batch processing support
- Keypoint smoothing

## Troubleshooting

Common issues and solutions:

1. **Camera not detected**
   ```python
   cv2.error: OpenCV(4.X.X) Error...
   ```
   Solution: Ensure your webcam is properly connected and not in use by another application.

2. **GPU Memory Issues**
   ```
   tensorflow.python.framework.errors_impl.ResourceExhaustedError
   ```
   Solution: Adjust `BATCH_SIZE` in config.yaml or disable GPU acceleration.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow team for the MoveNet model
- OpenCV community
- All contributors to this project

## Contact

Your Name - [@yourusername](https://twitter.com/yourusername)
Project Link: [https://github.com/yourusername/opencv-activity-detection](https://github.com/yourusername/opencv-activity-detection)

