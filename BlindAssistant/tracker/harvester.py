"""
Active Learning Data Harvester for BLIND Assistive Navigation Platform.

Captures high-uncertainty detections and unclassified motion anomalies from live
video streams and logs them in JSONL format for on-device fine-tuning via FineTuneKit.
Tailored for global blind mobility standards across diverse walking environments.
"""

import os
import json
import time
import cv2
import numpy as np
from pathlib import Path

# Paths to shared registry
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SHARED_REGISTRY_DIR = ROOT_DIR / "shared_registry"
DATASETS_DIR = SHARED_REGISTRY_DIR / "datasets"
IMAGES_DIR = DATASETS_DIR / "images"
JSONL_PATH = DATASETS_DIR / "active_learning.jsonl"

# Ensure directories exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
if not JSONL_PATH.exists():
    JSONL_PATH.touch()

class ActiveLearningHarvester:
    def __init__(self, min_conf=0.25, max_conf=0.50, min_approach_vel=0.5, cooldown_sec=2.0):
        """
        Initialize the active learning harvester.
        
        Args:
            min_conf (float): Minimum detection confidence to trigger harvesting.
            max_conf (float): Maximum detection confidence (uncertainty window).
            min_approach_vel (float): Velocity threshold (m/s) for approaching motion hazards (positive means approaching).
            cooldown_sec (float): Minimum seconds between harvesting similar objects to avoid flooding.
        """
        self.min_conf = min_conf
        self.max_conf = max_conf
        self.min_approach_vel = min_approach_vel
        self.cooldown_sec = cooldown_sec
        self.last_harvest_time = 0.0
        self.harvest_count = self._count_existing_samples()

    def _count_existing_samples(self):
        try:
            if not JSONL_PATH.exists():
                return 0
            with open(JSONL_PATH, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0

    def get_stats(self):
        """Return harvesting statistics for frontend telemetry."""
        return {
            "total_harvested": self.harvest_count,
            "jsonl_path": str(JSONL_PATH),
            "images_dir": str(IMAGES_DIR),
            "last_harvest_time": self.last_harvest_time
        }

    def evaluate_and_harvest(self, frame, detections, trackers):
        """
        Evaluate current frame detections and trackers for harvesting criteria.
        
        Args:
            frame (np.ndarray): Current raw BGR image frame.
            detections (list): List of dicts with keys 'box', 'score', 'class_id', 'label'.
            trackers (list): List of MovingObjectTracker instances.
            
        Returns:
            bool: True if a sample was harvested in this frame, False otherwise.
        """
        now = time.time()
        if now - self.last_harvest_time < self.cooldown_sec:
            return False

        if frame is None or frame.size == 0:
            return False

        should_harvest = False
        harvest_reason = ""
        target_box = None
        target_label = "Unknown Obstacle"

        # 1. Check for high-uncertainty YOLO detections
        for det in detections:
            score = det.get("score", 1.0)
            label = det.get("label", "Unknown")
            box = det.get("box")
            if self.min_conf <= score <= self.max_conf:
                should_harvest = True
                harvest_reason = f"Low Confidence YOLO ({score:.2f}) on {label}"
                target_box = box
                target_label = label
                break

        # 2. Check for unclassified MOG2 motion hazards or fast approaching objects
        if not should_harvest:
            for t in trackers:
                box = t.get_current_box() if hasattr(t, "get_current_box") else getattr(t, "box", [0, 0, 10, 10])
                hits = getattr(t, "total_frames_tracked", 0)
                if t.label == "Moving Obstacle" and hits >= 5:
                    should_harvest = True
                    harvest_reason = f"Unclassified MOG2 Motion Hazard (tracked {hits} frames)"
                    target_box = box
                    target_label = "Moving Obstacle"
                    break
                else:
                    vel_z = 0.0
                    dist_z = 999.0
                    if hasattr(t, 'velocity'):
                        vel_z = t.velocity[1]
                    elif hasattr(t, 'get_velocity_mps'):
                        from utils import get_real_width, FOCAL_LENGTH
                        w = box[2] if len(box) >= 3 else 0
                        dist_z = (get_real_width(t.label) * FOCAL_LENGTH) / w if w > 0 else 999.0
                        _, vel_z = t.get_velocity_mps(FOCAL_LENGTH, get_real_width(t.label), 15.0)
                    if hasattr(t, 'distance'):
                        dist_z = t.distance
                        
                    if vel_z > self.min_approach_vel and dist_z < 4.0:
                        should_harvest = True
                        harvest_reason = f"Fast Approaching Hazard ({vel_z:.2f} m/s)"
                        target_box = box
                        target_label = t.label
                        break

        if should_harvest and target_box is not None:
            self._save_sample(frame, target_box, target_label, harvest_reason)
            self.last_harvest_time = now
            return True

        return False

    def _enforce_storage_limits(self, max_mb=500, max_files=2000):
        """Enforces FIFO storage limits to prevent disk exhaustion."""
        try:
            if not IMAGES_DIR.exists():
                return
            files = sorted(IMAGES_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
            total_size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
            
            while len(files) > max_files or total_size_mb > max_mb:
                oldest = files.pop(0)
                size_mb = oldest.stat().st_size / (1024 * 1024)
                oldest.unlink(missing_ok=True)
                total_size_mb -= size_mb
                print(f"[HARVESTER CLEANUP] Removed oldest sample {oldest.name} to free disk space.")
        except Exception as e:
            print(f"[HARVESTER ERROR] Storage cleanup failed: {e}")

    def _save_sample(self, frame, box, label, reason):
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = box
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))

            if x2 <= x1 or y2 <= y1:
                return

            crop = frame[y1:y2, x1:x2]
            timestamp = int(time.time() * 1000)
            filename = f"anomaly_{timestamp}_{label.replace(' ', '_')}.jpg"
            img_path = IMAGES_DIR / filename

            cv2.imwrite(str(img_path), crop)

            # Format for FineTuneKit JSONL
            sample = {
                "instruction": f"Identify and assess navigation risk for detected obstacle image: {filename}",
                "output": f"Obstacle classified as '{label}'. Reason: {reason}. Recommended action: Proceed with caution or evade.",
                "metadata": {
                    "image_file": str(img_path),
                    "label": label,
                    "reason": reason,
                    "timestamp": timestamp,
                    "box": [x1, y1, x2, y2]
                }
            }

            with open(JSONL_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample) + "\n")

            self.harvest_count += 1
            print(f"[HARVESTER] Sample logged: {filename} ({reason})")
            self._enforce_storage_limits()
        except Exception as e:
            print(f"[HARVESTER ERROR] Failed to save sample: {e}")
