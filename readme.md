# OpenCV Activity Detection

This project uses TensorFlow and OpenCV to detect human activities in real-time using the MoveNet model from TensorFlow Hub.

## Requirements

- Python 3.12
- OpenCV
- TensorFlow
- TensorFlow Hub
- NumPy

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/aayush-ojha/opencv-activity-detection.git
    cd opencv-activity-detection
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Run the `main.py` script to start the activity detection:
```sh
python3 main.py
```

## Project Structure

- `main.py`: Main script to run the activity detection.
- `activities.py`: Contains the `recognize_activity` function to classify activities based on keypoints.

## Key Functions

### `main.py`

- **`check_movement(frame)`**: Processes a video frame to detect keypoints using the MoveNet model.
- **`cam_runner()`**: Captures video frames from the camera, runs the `check_movement` function, and displays the results.

### `activities.py`

- **`recognize_activity(keypoints)`**: Classifies activities based on detected keypoints.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

