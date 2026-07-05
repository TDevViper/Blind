import cv2
from tracker.motion import VisionPipeline
from tracker.kalman import MovingObjectTracker
from tracker.voice import AudioFeedbackManager, evaluate_and_instruct
from utils import calculate_iou

def main():
    # Initialize hardware and sub-systems
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access the camera. Check your permissions.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    vision_engine = VisionPipeline("yolov8n.pt")
    tts_manager = AudioFeedbackManager()
    tts_manager.start()
    
    active_trackers = []
    print("[INFO] Blind Assistive Tracker Started. Press 'q' on the video window to exit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Do not mirror frame; forward-facing mobility camera must preserve true physical left/right
            

            # 1. Get raw detections (YOLO + Motion Fusion)
            detected_entities = vision_engine.process_frame(frame)

            # 2. Predict next locations for existing trackers
            for tracker in active_trackers:
                tracker.predict()
                
            # 3. Associate new detections with existing trackers
            from tracker.hungarian import associate_detections_to_trackers
            matches, unmatched_detections = associate_detections_to_trackers(detected_entities, active_trackers, iou_threshold=0.15)
            
            # Update matched trackers
            for d_idx, t_idx in matches:
                det_box, label = detected_entities[d_idx]
                active_trackers[t_idx].update(det_box, label)
                
            # Spawn new trackers for unmatched detections
            for d_idx in unmatched_detections:
                det_box, label = detected_entities[d_idx]
                active_trackers.append(MovingObjectTracker(det_box, label))

            # 4. Remove dead trackers (objects that disappeared for > 8 frames)
            active_trackers = [t for t in active_trackers if t.frames_without_update < 8]
            
            # 5. Evaluate all trackers and give audio instructions for the closest one
            evaluate_and_instruct(active_trackers, frame_width, tts_manager)
            
            # 6. Draw bounding boxes for visual debugging
            for tracker in active_trackers:
                box = tracker.get_current_box()
                x, y, w, h = box
                
                # Use a different color for Unknown Moving Obstacles vs Predefined Objects
                color = (0, 165, 255) if tracker.label == "Moving Obstacle" else (0, 255, 0)
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, tracker.label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Multi-Object Assistive Tracking Prototype", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        print("[INFO] Shutting down...")
        cap.release()
        cv2.destroyAllWindows()
        tts_manager.stop()
        tts_manager.join()

if __name__ == "__main__":
    main()
