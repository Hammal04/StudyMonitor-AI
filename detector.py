import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import numpy as np
from utils import calculate_ear, estimate_head_pose

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.face_mesh = vision.FaceLandmarker.create_from_options(options)
        
        # Indices for right and left eyes in MediaPipe Face Mesh
        # These represent the 6 points around each eye
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    def _get_eye_points(self, landmarks, indices, width, height):
        """Extract (x, y) coordinates for specific landmark indices"""
        points = []
        for idx in indices:
            lm = landmarks[idx]
            x, y = int(lm.x * width), int(lm.y * height)
            points.append((x, y))
        return points

    def process_frame(self, frame):
        """
        Process a BGR frame and return EAR, head pose string, and bounding box
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.face_mesh.detect(mp_image)
        
        h, w = frame.shape[:2]
        
        ear = None
        looking_away = False
        direction = "Center"
        bbox = None
        
        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            
            # 1. Calculate EAR
            right_eye = self._get_eye_points(landmarks, self.RIGHT_EYE_INDICES, w, h)
            left_eye = self._get_eye_points(landmarks, self.LEFT_EYE_INDICES, w, h)
            
            right_ear = calculate_ear(right_eye)
            left_ear = calculate_ear(left_eye)
            ear = (right_ear + left_ear) / 2.0
            
            # 2. Estimate Head Pose
            looking_away, direction = estimate_head_pose(landmarks, w, h)
            
            # 3. Calculate bounding box for the face
            x_min = min([lm.x for lm in landmarks]) * w
            x_max = max([lm.x for lm in landmarks]) * w
            y_min = min([lm.y for lm in landmarks]) * h
            y_max = max([lm.y for lm in landmarks]) * h
            bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
            
        return ear, looking_away, direction, bbox

class PhoneDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Initialize YOLO model. This will download the weights if not present
        self.model = YOLO(model_path)
        
    def detect_phone(self, frame):
        """
        Run YOLO detection on the frame.
        Returns True if a 'cell phone' is detected, and its bounding boxes
        """
        # Run inference. Use verbose=False to avoid spamming the console
        results = self.model(frame, verbose=False)
        
        phone_detected = False
        phone_bboxes = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Class 67 in COCO dataset is 'cell phone'
                cls_id = int(box.cls[0])
                if cls_id == 67:
                    # Filter by confidence
                    conf = float(box.conf[0])
                    if conf > 0.4:  # Adjust threshold if needed
                        phone_detected = True
                        x1, y1, x2, y2 = box.xyxy[0]
                        phone_bboxes.append((int(x1), int(y1), int(x2), int(y2)))
                        
        return phone_detected, phone_bboxes
