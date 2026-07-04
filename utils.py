import threading
import winsound
import numpy as np

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_ear(eye_points):
    """
    Calculate the Eye Aspect Ratio (EAR)
    eye_points: list of 6 (x, y) coordinates representing the eye
    """
    # Vertical eye landmarks
    v1 = euclidean_distance(eye_points[1], eye_points[5])
    v2 = euclidean_distance(eye_points[2], eye_points[4])
    
    # Horizontal eye landmarks
    h = euclidean_distance(eye_points[0], eye_points[3])
    
    # EAR formula
    ear = (v1 + v2) / (2.0 * h)
    return ear

def estimate_head_pose(landmarks, frame_width, frame_height):
    """
    Simple heuristic to estimate if the user is looking away
    based on the relative position of the nose to the sides of the face.
    Returns: bool (True if looking away, False otherwise), string (direction)
    """
    # Key landmarks: nose tip (1), left cheek (234), right cheek (454)
    # top of head (10), chin (152)
    nose = landmarks[1]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    
    # Calculate ratios
    face_width = right_cheek.x - left_cheek.x
    if face_width == 0: return False, "Unknown"
    
    # How far is the nose from the left side, relative to face width?
    nose_x_ratio = (nose.x - left_cheek.x) / face_width
    
    # Looking left/right thresholds (approximate)
    if nose_x_ratio < 0.3:
        return True, "Looking Right" # Flipped due to mirrored webcam
    elif nose_x_ratio > 0.7:
        return True, "Looking Left"
        
    # How about up/down?
    top = landmarks[10]
    chin = landmarks[152]
    face_height = chin.y - top.y
    if face_height == 0: return False, "Unknown"
    
    nose_y_ratio = (nose.y - top.y) / face_height
    if nose_y_ratio < 0.4:
        return True, "Looking Up"
    elif nose_y_ratio > 0.7:
        return True, "Looking Down"
        
    return False, "Center"

class AlarmManager:
    """Manages playing an alarm sound in a background thread"""
    def __init__(self):
        self.is_playing = False

    def _play_sound(self):
        # Play a loud 2500Hz beep for 1000ms
        winsound.Beep(2500, 1000)
        self.is_playing = False

    def trigger_alarm(self):
        """Starts the alarm sound if it is not already playing"""
        if not self.is_playing:
            self.is_playing = True
            # Use daemon=True so the thread dies when the main program exits
            threading.Thread(target=self._play_sound, daemon=True).start()
