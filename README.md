# Study Attention Monitor

A Python desktop application that uses your webcam to monitor your attention during study sessions. It detects sleep (closed eyes), phone usage, and looking away, providing a visual and audible alarm.

## Features
- **Face & Eye Tracking**: Uses MediaPipe to calculate Eye Aspect Ratio (EAR) for sleep detection and head pose estimation for distraction.
- **Phone Detection**: Uses YOLOv8 Nano to detect if a mobile phone is in the frame.
- **Threaded Alarm**: Uses Windows `winsound` to play a loud beep without freezing the video feed.
- **Session Logging**: Logs events (sleep, distraction, phone usage) to `study_log.csv` with timestamps.
- **Real-time UI**: Displays bounding boxes, EAR values, a study timer, and distraction counters directly on the video feed.

## Prerequisites
- Python 3.8+
- Windows OS (due to the `winsound` module used for the alarm)
- A working webcam

## Installation

1. Open a Command Prompt or PowerShell in this directory.
2. Install the required Python libraries:
   ```cmd
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```cmd
   python main.py
   ```
2. **First Run Note**: The first time you run the script, the `ultralytics` library will automatically download the lightweight `yolov8n.pt` model (around 6MB).
3. The webcam window will open. Ensure your face is clearly visible.
4. The monitor will track your attention. If you close your eyes for more than 3 seconds, look away for more than 3 seconds, or hold up a phone, the alarm will sound.
5. Press **`Q`** on your keyboard while the video window is focused to safely quit the application.

## Logs
Check the `study_log.csv` file generated in the same folder to see a timestamped history of your study session events.
