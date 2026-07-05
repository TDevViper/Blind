import cv2
import torch
from ultralytics import YOLO
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import calculate_iou
from model_manager import ModelHotSwapper

# ==========================================
# Scoped PyTorch Security Bypass for Older Checkpoints
# ==========================================
class ScopedLegacyLoad:
    """Temporarily sets weights_only=False during model initialization without weakening global process security."""
    def __enter__(self):
        self.orig = torch.load
        def _legacy(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return self.orig(*args, **kwargs)
        torch.load = _legacy
    def __exit__(self, *args):
        torch.load = self.orig

# ==========================================
# Vision Core Engine
# ==========================================
class VisionPipeline:
    """
    Handles Object Detection (YOLO) and Motion Segmentation (Background Subtraction).
    Fuses the results into a single list of detected entities.
    """
    def __init__(self, yolo_model_path="BlindAssistant/yolov8n.pt"):
        with ScopedLegacyLoad():
            self.model = YOLO(yolo_model_path)
        self.model_manager = ModelHotSwapper(default_model_path=yolo_model_path)
        # Background subtractor to catch unknown moving objects outside YOLO's vocabulary
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
        self.prev_gray = None
        self.last_raw_detections = []

    def process_frame(self, frame):
        """
        Runs YOLO and Motion Detection on a frame and fuses the results.
        Returns a list of tuples: ( [x, y, w, h], label )
        """
        # Check for model updates (zero-downtime hot-swap)
        new_model_path = self.model_manager.check_for_updates()
        if new_model_path:
            with ScopedLegacyLoad():
                self.model = YOLO(new_model_path)

        # Stream A: YOLO Detections (Predefined objects, evaluated from conf 0.25 for active learning)
        yolo_outputs = self.model(frame, verbose=False, conf=0.25, imgsz=320)[0]
        detected_entities = [] 
        raw_detections = []
        
        for box in yolo_outputs.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = self.model.names[int(box.cls[0])]
            
            raw_detections.append({
                "box": [x1, y1, x2, y2],
                "score": conf,
                "label": label
            })
            
            if conf >= 0.55:
                detected_entities.append(([x1, y1, x2-x1, y2-y1], label))
                
        self.last_raw_detections = raw_detections
            
        # Check for camera ego-motion (user walking/turning causes whole-frame movement)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ego_motion_detected = False
        if self.prev_gray is not None:
            # Calculate global frame displacement/difference
            frame_diff = cv2.absdiff(gray, self.prev_gray)
            if cv2.mean(frame_diff)[0] > 14.0:
                ego_motion_detected = True
        self.prev_gray = gray
            
        # Stream B: Motion Detection (Only reliable if camera is not undergoing violent ego-motion)
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadow pixel values (shadows are marked as 127 when detectShadows=True)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Clean up the mask: Open to remove speckle noise, Dilate & Close to group object fragments
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        refined_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_DILATE, kernel_close)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # If >20% of the screen is foreground, MOG2 is reacting to background ego-motion
        fg_ratio = cv2.countNonZero(refined_mask) / float(frame.shape[0] * frame.shape[1])
        if fg_ratio > 0.20:
            ego_motion_detected = True
        
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Stream C: Fusion Engine (Only add MOG2 unknown obstacles if ego-motion is minimal)
        if not ego_motion_detected:
            for contour in contours:
                if cv2.contourArea(contour) < 800:  
                    continue
                    
                m_box = cv2.boundingRect(contour)
                
                # Check if this motion overlaps with an object YOLO already found
                is_predefined = False
                for y_box, _ in detected_entities:
                    if calculate_iou(m_box, y_box) > 0.15:
                        is_predefined = True
                        break
                
                # If YOLO missed it, it's an unknown moving hazard (e.g. stray dog, suitcase, thrown object)
                if not is_predefined:
                    detected_entities.append((m_box, "Moving Obstacle"))

        return detected_entities
