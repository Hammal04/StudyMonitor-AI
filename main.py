import cv2
import time
import csv
from datetime import datetime
from detector import FaceDetector, PhoneDetector
from utils import AlarmManager

# --- Configuration Settings ---
EAR_THRESHOLD = 0.20     # Eye Aspect Ratio threshold (below this means eyes are closed)
SLEEP_TIME_THRESH = 1.0    # Seconds of eyes closed before alarm triggers
DISTRACTION_TIME_THRESH = 1.0 # Seconds of looking away before alarm triggers
LOG_FILE = "study_log.csv"

def init_logger():
    """Initialize the CSV log file with headers if it doesn't exist"""
    try:
        with open(LOG_FILE, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "EventType", "Details"])
    except FileExistsError:
        pass

def log_event(event_type, details=""):
    """Append an event to the CSV log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, event_type, details])

def main():
    init_logger()
    log_event("Session Start", "Study session started")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    # Initialize detectors and alarm
    print("Initializing models... (This might take a moment if downloading YOLO for the first time)")
    face_detector = FaceDetector()
    phone_detector = PhoneDetector()
    alarm_manager = AlarmManager()

    # State variables for tracking durations
    sleep_start_time = None
    distraction_start_time = None
    
    # Counters for session statistics
    distraction_count = 0
    sleep_count = 0
    phone_count = 0
    
    # Session start time
    session_start = time.time()

    print("Starting monitoring. Press 'Q' to quit.")

    # Static variable to track phone presence and avoid logging spam
    phone_active = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for a mirror-like experience (intuitive for users)
        frame = cv2.flip(frame, 1)

        # 1. Process Frame with Detectors
        phone_detected, phone_bboxes = phone_detector.detect_phone(frame)
        ear, looking_away, direction, face_bbox = face_detector.process_frame(frame)

        # 2. Evaluate Logic & Timers
        current_time = time.time()
        
        warning_msg = ""
        warning_color = (0, 0, 255) # Red in BGR

        # -- Phone Detection Logic --
        if phone_detected:
            warning_msg = "PHONE DETECTED!"
            alarm_manager.trigger_alarm()
            
            if not phone_active:
                phone_count += 1
                log_event("Phone Usage", "Phone detected in frame")
                phone_active = True
        else:
            phone_active = False

        # -- Sleep Detection Logic (Eyes closed) --
        if ear is not None and ear < EAR_THRESHOLD:
            if sleep_start_time is None:
                sleep_start_time = current_time
            elif current_time - sleep_start_time > SLEEP_TIME_THRESH:
                warning_msg = "SLEEP DETECTED!"
                alarm_manager.trigger_alarm()
        else:
            # If eyes are open, check if we just woke up from a >3s sleep to log it
            if sleep_start_time is not None:
                if current_time - sleep_start_time > SLEEP_TIME_THRESH:
                    sleep_count += 1
                    log_event("Sleep", "Eyes closed for >3s")
            sleep_start_time = None

        # -- Distraction Detection Logic (Looking away) --
        if looking_away:
            if distraction_start_time is None:
                distraction_start_time = current_time
            elif current_time - distraction_start_time > DISTRACTION_TIME_THRESH:
                if not warning_msg: # Don't override a more severe phone/sleep warning
                    warning_msg = f"DISTRACTION: Looking {direction}"
                alarm_manager.trigger_alarm()
        else:
            # If looking center, check if we just recovered from a >3s distraction to log it
            if distraction_start_time is not None:
                if current_time - distraction_start_time > DISTRACTION_TIME_THRESH:
                    distraction_count += 1
                    log_event("Distraction", f"Looking {direction} for >3s")
            distraction_start_time = None

        # 3. Draw UI Overlays
        
        # Draw Phone Bounding Boxes
        if phone_detected:
            for (x1, y1, x2, y2) in phone_bboxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
        # Draw Face Bounding Box and EAR
        if face_bbox:
            x, y, w, h = face_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # EAR value overlay
            cv2.putText(frame, f"EAR: {ear:.2f}" if ear else "EAR: N/A", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw Warning Message (Top Center-ish)
        if warning_msg:
            cv2.putText(frame, warning_msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, warning_color, 3)

        # Draw Stats overlay (Bottom Left)
        session_time = int(current_time - session_start)
        minutes, seconds = divmod(session_time, 60)
        
        # Semi-transparent black background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, frame.shape[0] - 120), (250, frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        cv2.putText(frame, f"Timer: {minutes:02d}:{seconds:02d}", (20, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Sleeps: {sleep_count}", (20, frame.shape[0] - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Distractions: {distraction_count}", (20, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Phone Checks: {phone_count}", (20, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 4. Render Frame
        cv2.imshow("Study Attention Monitor", frame)

        # 5. Handle Quit (Press 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Session Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    
    # Log final session stats
    log_event("Session End", f"Total Sleeps: {sleep_count}, Distractions: {distraction_count}, Phones: {phone_count}")
    print("Session Ended. Logs saved to study_log.csv")

if __name__ == "__main__":
    main()
