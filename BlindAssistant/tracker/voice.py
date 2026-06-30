import subprocess
import threading
import time
import sys
import os

# --- CONFIGURABLE CONSTANTS ---
FOCAL_LENGTH = 650.0 
ASSUMED_REAL_WIDTH = 0.4 
SAFETY_BUFFER = 0.8 

class AudioFeedbackManager:
    def __init__(self):
        self.last_queued_time = 0
        self.speaker_script = os.path.join(os.path.dirname(__file__), "speaker.py")

    def start(self):
        # Kept for compatibility with main.py
        pass

    def speak(self, text):
        """Spawns a separate process to handle TTS, bypassing Windows thread freezing."""
        subprocess.Popen([sys.executable, self.speaker_script, text])
        self.last_queued_time = time.time()

    def stop(self):
        # Kept for compatibility
        pass
    
    def join(self):
        pass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ASSUMED_REAL_WIDTH, FOCAL_LENGTH, FPS_ASSUMPTION, estimate_distance, calculate_horizontal_deviation
from tracker.collision import predict_collision
from tracker.risk import assess_risk, RISK_CRITICAL, RISK_HIGH
from tracker.safe_path import calculate_avoidance_instruction
from tracker.occupancy_grid import OccupancyGrid

def evaluate_and_instruct(trackers, frame_width, tts_manager):
    """
    Evaluates all active trackers to find the highest risk obstacle.
    Then, generates and speaks a contextual navigation instruction for that object.
    """
    if time.time() - tts_manager.last_queued_time < 4.0:
        return

    highest_risk_score = -1
    best_instruction = None
    best_debug_info = None

    # Risk mapping for prioritization
    risk_weights = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

    # Build the global occupancy grid to map all obstacles
    grid = OccupancyGrid(frame_width)
    grid.build_grid(trackers)

    for tracker in trackers:
        box = tracker.get_current_box()
        x, y, w, h = box
        label = tracker.label
        
        if w <= 0:
            continue

        distance_z = estimate_distance(w)
        if distance_z > 5.0:
            continue

        cx = x + (w / 2)
        horizontal_deviation_x = calculate_horizontal_deviation(cx, frame_width, distance_z)
        
        # Extract velocity
        vx_mps, vz_mps = tracker.get_velocity_mps(FOCAL_LENGTH, ASSUMED_REAL_WIDTH, FPS_ASSUMPTION)
        
        # Predict Collision
        will_collide, ttc, intersect_x = predict_collision(horizontal_deviation_x, distance_z, vx_mps, vz_mps)
        
        # Assess Risk
        risk_level = assess_risk(label, distance_z, ttc, will_collide)
        risk_weight = risk_weights[risk_level]
        
        # We only care about high and critical risks for interrupting the user
        if risk_weight < 2:
            continue
            
        # If this is the most dangerous thing we've seen so far
        if risk_weight > highest_risk_score or (risk_weight == highest_risk_score and distance_z < 2.0):
            highest_risk_score = risk_weight
            
            # Determine Action using the global occupancy grid
            action, dist = calculate_avoidance_instruction(horizontal_deviation_x, ASSUMED_REAL_WIDTH, vx_mps, distance_z, grid)
            
            if action == 'none':
                continue # Path is technically free, no instruction needed
            
            # Format Natural Language
            obj_name = label if label != "Moving Obstacle" else "Unknown object"
            
            # Convert meters to steps
            steps = max(1, int(round(dist / 0.6)))
            step_word = "step" if steps == 1 else "steps"
            
            if action == 'stop':
                best_instruction = f"{obj_name} approaching. Left and right blocked. Stop immediately."
            elif action == 'move_left':
                best_instruction = f"{obj_name} ahead. Move {steps} {step_word} left."
            elif action == 'move_right':
                best_instruction = f"{obj_name} ahead. Move {steps} {step_word} right."
                
            best_debug_info = f"[AI DEBUG] {label} | Risk: {risk_level} | TTC: {ttc:.1f}s | Action: {action} ({steps} steps)"

    if best_instruction:
        print(best_debug_info)
        tts_manager.speak(best_instruction)
