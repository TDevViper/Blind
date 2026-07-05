try:
    from gunicorn.workers.geventlet import EventletWorker
except ImportError as e:
    # Fallback for local Windows development or Gunicorn versions where EventletWorker was removed (v25+)
    class EventletWorker:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Failed to load EventletWorker from Gunicorn. Note: Gunicorn removed EventletWorker in v25.0+. "
                "Please ensure gunicorn is pinned (e.g., gunicorn==21.2.0 as in requirements_prod.txt) "
                f"or run locally without Gunicorn. Original error: {e}"
            )
        def run(self):
            pass
        def init_process(self):
            pass

import eventlet

class CustomEventletWorker(EventletWorker):
    """
    Custom Eventlet worker for Gunicorn that only monkey patches socket and select.
    Patching thread or os breaks PyTorch C++ multithreading and Ultralytics cpuinfo module,
    causing deadlocks or crashes during YOLO model inference on cloud platforms like Render.
    """
    def patch(self):
        eventlet.monkey_patch(socket=True, select=True)
