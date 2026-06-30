import cv2
import torch
from ultralytics import YOLO

# ==========================================
# PyTorch Security Bypass for Older Checkpoints
# ==========================================
_original_torch_load = torch.load
def _legacy_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _legacy_torch_load

# ==========================================
# Helper Functions
# ==========================================
def calculate_iou(box_a, box_b):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    xA, yA, wA, hA = box_a
    xB, yB, wB, hB = box_b
    
    x1 = max(xA, xB)
    y1 = max(yA, yB)
    x2 = min(xA + wA, xB + wB)
    y2 = min(yA + hA, yB + hB)
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box_a_area = wA * hA
    box_b_area = wB * hB
    
    union_area = float(box_a_area + box_b_area - inter_area)
    return inter_area / union_area if union_area > 0 else 0

# ==========================================
# Vision Core Engine
# ==========================================
class VisionPipeline:
    """
    Handles Object Detection (YOLO) and Motion Segmentation (Background Subtraction).
    Fuses the results into a single list of detected entities.
    """
    def __init__(self, yolo_model_path="yolov8n.pt"):
        self.model = YOLO(yolo_model_path)
        # Background subtractor to catch unknown moving objects
        # Increased varThreshold to 150 to make it extremely robust against shadows/lighting noise
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=150, detectShadows=False)

    def process_frame(self, frame):
        """
        Runs YOLO and Motion Detection on a frame and fuses the results.
        Returns a list of tuples: ( [x, y, w, h], label )
        """
        # Stream A: YOLO Detections (Predefined objects)
        # Added conf=0.55 to prevent YOLO from hallucinating objects like "fridge" or "ball" in noisy moving curtains
        yolo_outputs = self.model(frame, verbose=False, conf=0.55)[0]
        detected_entities = [] 
        
        for box in yolo_outputs.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.model.names[int(box.cls[0])]
            detected_entities.append(([x1, y1, x2-x1, y2-y1], label))
            
        # Stream B: Motion Detection (Undefined objects)
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        refined_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Stream C: Fusion Engine
        for contour in contours:
            # Increased minimum area from 5000 to 12000 to ignore fluttering curtains and fans
            if cv2.contourArea(contour) < 12000:  
                continue
                
            m_box = cv2.boundingRect(contour)
            
            # Check if this motion overlaps with an object YOLO already found
            is_predefined = False
            for y_box, _ in detected_entities:
                if calculate_iou(m_box, y_box) > 0.3:
                    is_predefined = True
                    break
            
            # If YOLO missed it, it's an unknown moving object
            if not is_predefined:
                detected_entities.append((m_box, "Moving Obstacle"))

        return detected_entities
