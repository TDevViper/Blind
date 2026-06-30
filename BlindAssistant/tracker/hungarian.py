import numpy as np
from scipy.optimize import linear_sum_assignment
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import calculate_iou

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.15):
    """
    Assigns detections to tracked objects using the Hungarian Algorithm (Munkres) based on IoU.
    This guarantees mathematically optimal assignments, preventing ID switching.
    
    detections: list of [ [x, y, w, h], label ]
    trackers: list of MovingObjectTracker instances
    
    Returns:
        matches: list of tuples (detection_idx, tracker_idx)
        unmatched_detections: list of detection_idx
    """
    if len(trackers) == 0:
        return [], list(range(len(detections)))
        
    if len(detections) == 0:
        return [], []

    # Build IoU cost matrix
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = calculate_iou(det[0], trk.get_current_box())
            
    # scipy's linear_sum_assignment minimizes cost, so we maximize IoU by minimizing -IoU
    cost_matrix = -iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_detections = []
    
    # Filter out matches that don't meet the minimum IoU threshold
    matched_det_indices = set()
    
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append((r, c))
            matched_det_indices.add(r)
            
    # Any detection not in a valid match is considered "unmatched" and will spawn a new tracker
    for d in range(len(detections)):
        if d not in matched_det_indices:
            unmatched_detections.append(d)
            
    return matches, unmatched_detections
