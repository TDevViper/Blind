import subprocess
import threading
import time
import sys
import os

# --- CONFIGURABLE CONSTANTS ---
FOCAL_LENGTH = 650.0 
ASSUMED_REAL_WIDTH = 0.4 
SAFETY_BUFFER = 0.8 

from tracker.safe_path import calculate_avoidance_instruction
from tracker.occupancy_grid import OccupancyGrid
from tracker.llm_reasoner import SpatialLLMReasoner

class AudioFeedbackManager:
    def __init__(self):
        self.last_queued_time = 0
        self.speaker_script = os.path.join(os.path.dirname(__file__), "speaker.py")
        self.current_process = None
        self.last_risk_score = -1
        self.current_instruction = None
        self.llm_reasoner = SpatialLLMReasoner()

    def start(self):
        # Kept for compatibility with main.py
        pass

    def speak(self, text, risk_score=0):
        """Spawns a separate process to handle TTS, bypassing Windows thread freezing."""
        self.current_instruction = text
        if self.current_process and self.current_process.poll() is None:
            # If higher risk alert comes in, terminate previous speech process
            if risk_score > self.last_risk_score:
                try:
                    self.current_process.terminate()
                except Exception:
                    pass
        self.current_process = subprocess.Popen([sys.executable, self.speaker_script, text])
        self.last_queued_time = time.time()
        self.last_risk_score = risk_score

    def stop(self):
        if self.current_process and self.current_process.poll() is None:
            try:
                self.current_process.terminate()
            except Exception:
                pass
    
    def join(self):
        if self.current_process:
            try:
                self.current_process.wait(timeout=2.0)
            except Exception:
                pass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ASSUMED_REAL_WIDTH, FOCAL_LENGTH, FPS_ASSUMPTION, estimate_distance, calculate_horizontal_deviation, get_real_width
from tracker.collision import predict_collision
from tracker.risk import assess_risk, RISK_CRITICAL, RISK_HIGH, RISK_MEDIUM, RISK_LOW
from tracker.safe_path import calculate_avoidance_instruction
from tracker.occupancy_grid import OccupancyGrid

def evaluate_and_instruct(trackers, frame_width, tts_manager, fps=FPS_ASSUMPTION):
    """
    Evaluates all active trackers to find the highest risk obstacle.
    Then, generates and speaks a contextual navigation instruction for that object.
    Includes distance and approaching velocity in spoken alerts.
    """
    highest_risk_score = -1
    best_distance_z = float('inf')
    best_ttc = float('inf')
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

        real_w = get_real_width(label)
        distance_z = estimate_distance(w, label)
        if distance_z > 5.0:
            continue

        cx = x + (w / 2)
        horizontal_deviation_x = calculate_horizontal_deviation(cx, frame_width, distance_z)
        
        # Annotate tracker attributes for LLM reasoner and downstream analytics
        tracker.distance = distance_z
        if horizontal_deviation_x < -0.4:
            tracker.zone = "LEFT"
        elif horizontal_deviation_x > 0.4:
            tracker.zone = "RIGHT"
        else:
            tracker.zone = "CENTER"
        
        # Extract velocity using actual dynamic FPS and class width
        vx_mps, vz_mps = tracker.get_velocity_mps(FOCAL_LENGTH, real_w, fps)
        tracker.velocity = (vx_mps, vz_mps)
        
        # If moving away from the camera significantly, no harm! Ignore from critical voice alarm
        if vz_mps <= -0.1 and distance_z > 1.0:
            continue
        
        # Predict Collision
        will_collide, ttc, intersect_x = predict_collision(horizontal_deviation_x, distance_z, vx_mps, vz_mps)
        tracker.ttc = ttc
        
        # Assess Risk
        risk_level = assess_risk(label, distance_z, ttc, will_collide)
        risk_weight = risk_weights.get(risk_level, 0)
        
        # We only care about high and critical risks for interrupting the user
        if risk_weight < 2:
            continue
            
        # Tie-breaker: if same risk weight, closest distance wins (best_distance_z)
        if risk_weight > highest_risk_score or (risk_weight == highest_risk_score and distance_z < best_distance_z):
            highest_risk_score = risk_weight
            best_distance_z = distance_z
            best_ttc = ttc
            
            # Determine Action using the global occupancy grid
            action, dist = calculate_avoidance_instruction(horizontal_deviation_x, real_w, vx_mps, distance_z, grid)
            
            if action == 'none':
                continue # Path is technically free, no instruction needed
            
            # Format Natural Language with distance and approaching velocity cleanly
            obj_name = label if label != "Moving Obstacle" else "Unknown obstacle"
            dist_str = f"at {distance_z:.1f} meters"
            speed_str = f" approaching at {vz_mps:.1f} meters per second" if vz_mps > 0.1 else ""
            
            # Convert meters to steps
            steps = max(1, int(round(dist / 0.6)))
            step_word = "step" if steps == 1 else "steps"
            
            if action == 'stop':
                best_instruction = f"Warning! {obj_name} {dist_str}{speed_str}. Stop immediately!"
            elif action == 'move_left':
                best_instruction = f"Caution! {obj_name} {dist_str}{speed_str}. Move {steps} {step_word} left."
            elif action == 'move_right':
                best_instruction = f"Caution! {obj_name} {dist_str}{speed_str}. Move {steps} {step_word} right."
                
            best_debug_info = f"[AI DEBUG] {label} | Dist: {distance_z:.1f}m | Speed: {vz_mps:.1f}m/s | Risk: {risk_level} | TTC: {ttc:.1f}s | Action: {action}"

    if best_instruction:
        if hasattr(tts_manager, 'llm_reasoner') and tts_manager.llm_reasoner.enabled:
            for t in trackers:
                if not hasattr(t, 'distance'):
                    box = t.get_current_box()
                    w = box[2] if box else 0
                    t.distance = estimate_distance(w, t.label) if w > 0 else 99.0
                if not hasattr(t, 'velocity'):
                    t.velocity = t.get_velocity_mps(FOCAL_LENGTH, get_real_width(t.label), fps)
                if not hasattr(t, 'zone'):
                    t.zone = "CENTER"
                if not hasattr(t, 'ttc'):
                    t.ttc = 99.0
            try:
                best_instruction = tts_manager.llm_reasoner.generate_instruction(trackers, best_instruction)
            except Exception as e:
                print(f"[WARNING] Spatial LLM Reasoner fallback triggered due to error/timeout: {e}")
        if hasattr(tts_manager, 'current_instruction'):
            tts_manager.current_instruction = best_instruction
        elapsed = time.time() - getattr(tts_manager, 'last_queued_time', 0)
        last_score = getattr(tts_manager, 'last_risk_score', -1)
        last_ttc = getattr(tts_manager, 'last_ttc', float('inf'))
        last_dist = getattr(tts_manager, 'last_distance', float('inf'))
        
        # Cooldown check: speak if 4.0s elapsed OR if escalating risk score OR escalating severity within Critical tier (shorter TTC or closer distance)
        critical_escalated = (highest_risk_score == 3 and (best_ttc < last_ttc * 0.6 or best_distance_z < last_dist - 1.0 or best_ttc < 1.0))
        if elapsed >= 4.0 or (highest_risk_score >= 3 and highest_risk_score > last_score) or critical_escalated:
            print(best_debug_info)
            if hasattr(tts_manager, 'speak'):
                try:
                    tts_manager.speak(best_instruction, risk_score=highest_risk_score)
                except TypeError:
                    tts_manager.speak(best_instruction)
            if hasattr(tts_manager, 'last_risk_score'):
                tts_manager.last_risk_score = highest_risk_score
            tts_manager.last_ttc = best_ttc
            tts_manager.last_distance = best_distance_z

def evaluate_all_trackers_telemetry(trackers, frame_width, fps=FPS_ASSUMPTION):
    """
    Evaluates all active trackers to generate structured telemetry for UI analytics.
    Ranks objects by collision urgency (Time-To-Collision and distance) to determine
    which object will hit first.
    """
    telemetry_list = []
    
    for idx, tracker in enumerate(trackers):
        box = tracker.get_current_box()
        x, y, w, h = box
        label = tracker.label
        
        if w <= 0:
            continue
            
        real_w = get_real_width(label)
        distance_z = estimate_distance(w, label)
        cx = x + (w / 2)
        horizontal_deviation_x = calculate_horizontal_deviation(cx, frame_width, distance_z)
        vx_mps, vz_mps = tracker.get_velocity_mps(FOCAL_LENGTH, real_w, fps)
        will_collide, ttc, intersect_x = predict_collision(horizontal_deviation_x, distance_z, vx_mps, vz_mps)
        risk_level = assess_risk(label, distance_z, ttc, will_collide)
        
        # Determine direction status ("moving away then no harm")
        if vz_mps > 0.1:
            direction_status = f"Approaching ({vz_mps:.1f} m/s)"
        elif vz_mps <= -0.1:
            direction_status = f"Moving Away (No harm)"
            if risk_level in [RISK_HIGH, RISK_CRITICAL]:
                risk_level = RISK_LOW  # Downgrade risk if receding safely
        else:
            direction_status = "Stationary"
            
        telemetry_list.append({
            "id": idx + 1,
            "label": label if label != "Moving Obstacle" else "Unknown Obstacle",
            "distance": round(float(distance_z), 2),
            "velocity_z": round(float(vz_mps), 2),
            "velocity_x": round(float(vx_mps), 2),
            "direction": direction_status,
            "ttc": round(float(ttc), 2) if ttc != float('inf') else 999.0,
            "ttc_display": f"{ttc:.1f}s" if ttc != float('inf') else "N/A",
            "risk": risk_level,
            "will_collide": will_collide
        })
        
    # Sort by collision urgency: first finite TTC ascending (will hit first), then by distance
    telemetry_list.sort(key=lambda item: (not item["will_collide"], item["ttc"], item["distance"]))
    
    # Assign ranking ("which object will hit first")
    for i, item in enumerate(telemetry_list):
        if item["will_collide"] and item["ttc"] < 999.0:
            if i == 0:
                item["rank"] = "#1 Impact Threat (Will hit first!)"
            else:
                item["rank"] = f"#{i+1} Collision Risk"
        elif item["risk"] in [RISK_HIGH, RISK_CRITICAL]:
            item["rank"] = f"#{i+1} Proximity Hazard"
        else:
            item["rank"] = "Safe / Receding"
            
    return telemetry_list
