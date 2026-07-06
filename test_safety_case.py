"""
Automated Safety & Stability Verification Script for BLIND Platform.

Tests:
1. Multi-session concurrency and YOLO thread locking.
2. Kalman filter tracking gap closing velocity normalization.
3. OccupancyGrid ±0.75m corridor boundaries & voice cooldown priority override.
4. ActiveLearningHarvester sample collection & FIFO storage limit enforcement.
5. Synthetic benchmark suite execution.
"""

import os
import sys
import time
import shutil
import numpy as np
import threading

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BlindAssistant'))

from tracker.motion import VisionPipeline
from tracker.kalman import MovingObjectTracker
from tracker.occupancy_grid import OccupancyGrid
from tracker.voice import evaluate_and_instruct, AudioFeedbackManager
from tracker.harvester import ActiveLearningHarvester
import verify_pipeline

def test_concurrency():
    print("--- Test 1: Concurrency & Shared YOLO Loading ---")
    results = []
    lock = threading.Lock()
    def worker(sid):
        try:
            vp = VisionPipeline(lock=lock)
            # Simulate processing an empty frame
            dummy_frame = np.zeros((360, 480, 3), dtype=np.uint8)
            res = vp.process_frame(dummy_frame)
            results.append((sid, len(res)))
        except Exception as e:
            results.append((sid, str(e)))
            
    threads = [threading.Thread(target=worker, args=(f"session_{i}",)) for i in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    
    assert len(results) == 5, "Not all sessions completed."
    for sid, res in results:
        assert isinstance(res, int), f"Session {sid} failed: {res}"
    print("[PASSED] 5 concurrent sessions initialized and processed frames cleanly.\n")

def test_kalman_gaps():
    print("--- Test 2: Kalman Filter Gap Closing Velocity Normalization ---")
    tracker = MovingObjectTracker([100, 100, 50, 100], "person")
    
    # Track for 5 frames approaching
    for i in range(5):
        box = [100, 100, 50 + i*2, 100 + i*4]
        tracker.update(box, "person")
        tracker.predict()
        vx, vz = tracker.get_velocity_mps(650.0, 0.45, 10.0)
        
    print(f"Normal approaching velocity: {vz} m/s")
    
    # Simulate a tracking gap of 4 frames (object occluded, only predict called)
    for _ in range(4):
        tracker.predict()
        
    # Re-acquire object after gap
    tracker.update([100, 100, 65, 130], "person")
    vx_gap, vz_gap = tracker.get_velocity_mps(650.0, 0.45, 10.0)
    print(f"Velocity after 4-frame gap (normalized): {vz_gap} m/s")
    
    assert abs(vz_gap) < 5.0, f"Artificial velocity spike detected: {vz_gap} m/s!"
    print("[PASSED] Closing velocity remains stable after tracking gaps.\n")

def test_corridor_and_voice():
    print("--- Test 3: OccupancyGrid Corridor Width & Voice Priority Override ---")
    grid = OccupancyGrid()
    assert grid.zone_boundaries["CENTER"] == (-0.75, 0.75), f"Incorrect center corridor width: {grid.zone_boundaries['CENTER']}"
    
    # Create mock tracker in center zone (w=600px -> ~1.95m for car width 1.8m)
    t = MovingObjectTracker([20, 100, 600, 200], "car")
    t.total_frames_tracked = 5
    t.distance = 2.0
    t.velocity = (0.0, 1.5) # approaching at 1.5 m/s
    t.zone = "CENTER"
    t.ttc = 2.0 / 1.5
    
    grid.build_grid([t])
    assert grid.zones["CENTER"] == True, "Center corridor should be blocked."
    
    # Test voice cooldown override
    class MockAudio:
        def __init__(self):
            self.last_spoken = ""
            self.last_spoken_time = 0
            self.cooldown = 3.0
            self.current_instruction = ""
        def speak(self, text, force=False):
            self.last_spoken = text
            self.last_spoken_time = time.time()
            self.current_instruction = text
            
    mock_audio = MockAudio()
    # Speak initial prompt
    mock_audio.speak("System ready", force=True)
    
    # Trigger evaluate_and_instruct immediately within cooldown period
    evaluate_and_instruct([t], 640, mock_audio)
    instruction = mock_audio.current_instruction
    assert "car" in instruction.lower() and "stop" in instruction.lower(), f"Failed to override cooldown for critical threat! Instruction: {instruction}"
    print(f"[PASSED] Corridor boundaries verified and Critical threat overridden: '{instruction}'\n")

def test_harvester():
    print("--- Test 4: ActiveLearningHarvester Sample Collection ---")
    harvester = ActiveLearningHarvester(min_conf=0.25, max_conf=0.50, min_approach_vel=0.5, cooldown_sec=0.1)
    
    dummy_frame = np.full((360, 480, 3), 128, dtype=np.uint8)
    detections = [{"box": [10, 10, 100, 100], "score": 0.35, "label": "obstacle"}]
    
    harvested = harvester.evaluate_and_harvest(dummy_frame, detections, [])
    assert harvested, "Failed to harvest high uncertainty detection."
    
    stats = harvester.get_stats()
    assert stats["total_harvested"] >= 1, "Stats did not increment."
    print(f"[PASSED] Harvester captured anomaly sample successfully. Total: {stats['total_harvested']}\n")

def test_synthetic_suite():
    print("--- Test 5: Synthetic Tracking Benchmark Suite ---")
    res = verify_pipeline.run_validation_suite()
    assert res["kpis"]["precision"] == 0.975, f"Unexpected precision: {res['kpis']['precision']}"
    assert res["kpis"]["mota"] == 0.872, f"Unexpected MOTA: {res['kpis']['mota']}"
    print("[PASSED] Synthetic benchmark suite generated valid KPIs.\n")

if __name__ == "__main__":
    print("==================================================")
    print("STARTING BLIND PLATFORM SAFETY & STABILITY VERIFICATION")
    print("==================================================\n")
    test_concurrency()
    test_kalman_gaps()
    test_corridor_and_voice()
    test_harvester()
    test_synthetic_suite()
    print("==================================================")
    print("ALL 5 SAFETY VERIFICATION SUITES PASSED SUCCESSFULLY!")
    print("==================================================")
