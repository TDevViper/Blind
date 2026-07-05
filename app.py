import sys
import os

# Only use eventlet on Linux/Render production. On Windows (local dev), use standard threading
if not sys.platform.startswith('win'):
    try:
        import eventlet
        eventlet.monkey_patch(socket=True, select=True)
        async_mode = 'eventlet'
    except ImportError:
        async_mode = 'threading'
else:
    async_mode = 'threading'

import time
import base64
import traceback
import cv2
import numpy as np
from flask import Flask, request
from flask_socketio import SocketIO, emit

# Ensure we can import from the BlindAssistant package
sys.path.append(os.path.join(os.path.dirname(__file__), 'BlindAssistant'))

from tracker.motion import VisionPipeline
from tracker.kalman import MovingObjectTracker
from tracker.hungarian import associate_detections_to_trackers
from tracker.voice import evaluate_and_instruct, evaluate_all_trackers_telemetry
from tracker.metrics import get_validation_metrics
from tracker.harvester import ActiveLearningHarvester
from tracker.llm_reasoner import SpatialLLMReasoner

app = Flask(__name__)
# Allow CORS configuration from environment variable, fallback to '*' for local testing
cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",") if "CORS_ORIGINS" in os.environ else "*"
socketio = SocketIO(app, cors_allowed_origins=cors_origins, async_mode=async_mode,
                    ping_timeout=60, ping_interval=25)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

print("[INFO] Initializing Vision Pipeline (YOLO)...")
# Check inside BlindAssistant first, then project root, then let ultralytics download
project_root = os.path.dirname(os.path.abspath(__file__))
yolo_path = os.path.join(project_root, 'BlindAssistant', 'yolov8n.pt')
if not os.path.exists(yolo_path):
    yolo_path = os.path.join(project_root, 'yolov8n.pt')
if not os.path.exists(yolo_path):
    yolo_path = "yolov8n.pt"  # let ultralytics download it

print(f"[INFO] Using YOLO model at: {yolo_path}")
vision_engine = VisionPipeline(yolo_path)
harvester = ActiveLearningHarvester()
global_llm_enabled = False

# Per-session state management keyed by request.sid
user_sessions = {}

class WebTTSManager:
    """Mock TTS Manager that captures text to send to the web client."""
    def __init__(self):
        self.last_queued_time = 0
        self.current_instruction = None
        self.last_risk_score = -1
        self.llm_reasoner = SpatialLLMReasoner()
        self.llm_reasoner.enabled = global_llm_enabled

    def speak(self, text, risk_score=0):
        self.current_instruction = text
        self.last_queued_time = time.time()
        self.last_risk_score = risk_score

@app.route('/')
def index():
    return {
        "service": "BlindAssistant Vision & AI Server",
        "status": "online",
        "version": "2.0.0-nextjs",
        "frontend": "http://localhost:3000",
        "websocket": "ws://localhost:5000",
        "endpoints": ["/health", "/api/metrics", "/api/harvested_stats", "/api/llm_mode"]
    }, 200

@app.route('/health')
def health():
    return {'status': 'ok'}, 200

@app.route('/api/metrics')
def metrics():
    return get_validation_metrics(), 200

@app.route('/api/harvested_stats')
def harvested_stats():
    return harvester.get_stats(), 200

@app.route('/api/llm_mode', methods=['GET', 'POST'])
def llm_mode():
    global global_llm_enabled
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        global_llm_enabled = bool(data.get('enabled', not global_llm_enabled))
        for sid, sdata in user_sessions.items():
            if 'tts_manager' in sdata and hasattr(sdata['tts_manager'], 'llm_reasoner'):
                sdata['tts_manager'].llm_reasoner.enabled = global_llm_enabled
    return {'enabled': global_llm_enabled}, 200

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    user_sessions[sid] = {
        'active_trackers': [],
        'is_processing_frame': False,
        'tts_manager': WebTTSManager(),
        'last_frame_time': time.time(),
        'fps': 15.0
    }
    print(f"[INFO] Client connected via WebSocket (sid: {sid})")

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    user_sessions.pop(sid, None)
    print(f"[INFO] Client disconnected (sid: {sid})")

@socketio.on('video_frame')
def handle_video_frame(data):
    sid = request.sid
    session_data = user_sessions.get(sid)
    if not session_data:
        session_data = {
            'active_trackers': [],
            'is_processing_frame': False,
            'tts_manager': WebTTSManager(),
            'last_frame_time': time.time(),
            'fps': 15.0
        }
        user_sessions[sid] = session_data

    if session_data['is_processing_frame']:
        # Server is busy executing YOLO inference on the current frame for this user.
        return

    session_data['is_processing_frame'] = True
    try:
        now = time.time()
        dt = now - session_data['last_frame_time']
        if dt > 0.001:
            raw_fps = 1.0 / dt
            session_data['fps'] = 0.7 * session_data['fps'] + 0.3 * min(30.0, max(5.0, raw_fps))
        session_data['last_frame_time'] = now
        current_fps = session_data['fps']

        active_trackers = session_data['active_trackers']
        tts_manager = session_data['tts_manager']

        # 1. Decode incoming frame
        try:
            image_data = data['image'].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[ERROR] Frame decode failed for sid {sid}: {e}")
            return

        if frame is None:
            return
            
        frame_width = frame.shape[1]
        
        # 2. Vision Pipeline
        detected_entities = vision_engine.process_frame(frame)
        
        # 3. Predict & Associate
        for tracker in active_trackers:
            tracker.predict()
            
        matches, unmatched_detections = associate_detections_to_trackers(detected_entities, active_trackers, iou_threshold=0.15)
        
        for d_idx, t_idx in matches:
            det_box, label = detected_entities[d_idx]
            active_trackers[t_idx].update(det_box, label)
            
        for d_idx in unmatched_detections:
            det_box, label = detected_entities[d_idx]
            active_trackers.append(MovingObjectTracker(det_box, label))
            
        # 4. Cleanup
        active_trackers = [t for t in active_trackers if t.frames_without_update < 8]
        session_data['active_trackers'] = active_trackers
        
        # Harvest active learning anomalies for FineTuneKit
        harvester.evaluate_and_harvest(frame, getattr(vision_engine, 'last_raw_detections', []), active_trackers)
        
        # 5. Evaluate Risk & Get Instruction
        tts_manager.current_instruction = None # Reset for this frame
        evaluate_and_instruct(active_trackers, frame_width, tts_manager, fps=current_fps)
        
        # 5b. Generate comprehensive telemetry and collision rankings
        telemetry = evaluate_all_trackers_telemetry(active_trackers, frame_width, fps=current_fps)
        
        # 6. Draw Bounding Boxes with Distance & Ranking
        for idx, tracker in enumerate(active_trackers):
            box = tracker.get_current_box()
            x, y, w, h = box
            color = (0, 165, 255) if tracker.label == "Moving Obstacle" else (0, 255, 0)
            
            # Highlight #1 Impact Threat in bright RED
            dist_str = ""
            for item in telemetry:
                if item["id"] == idx + 1:
                    dist_str = f" | {item['distance']}m"
                    if "Impact Threat" in item["rank"]:
                        color = (0, 0, 255) # Red for primary collision hazard
                    elif item["risk"] in ["High", "Critical"]:
                        color = (0, 140, 255) # Orange for proximity hazard
                    break
                    
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{tracker.label}{dist_str}", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # 7. Encode back to base64 with optimized compression
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        data_url = 'data:image/jpeg;base64,' + encoded_image
        
        # Send back to client with full real-time telemetry
        emit('processed_frame', {
            'image': data_url,
            'instruction': tts_manager.current_instruction,
            'telemetry': telemetry
        })
    except Exception as e:
        print(f"[ERROR] Frame processing failed for sid {sid}: {e}")
        traceback.print_exc()
        emit('processed_frame', {
            'image': '',
            'instruction': 'Server error: Frame processing failed. Please check client video stream.'
        })
    finally:
        if sid in user_sessions:
            user_sessions[sid]['is_processing_frame'] = False

if __name__ == '__main__':
    print("[INFO] Starting Flask-SocketIO Server...")
    # Cloud providers like Render supply a PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
