import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ASSUMED_REAL_WIDTH, FOCAL_LENGTH, FPS_ASSUMPTION, estimate_distance, calculate_horizontal_deviation, get_real_width

class OccupancyGrid:
    """
    A 1D Egocentric spatial map representing the world in front of the user.
    Divides the horizontal space into sectors (e.g. Left, Center, Right) to determine free-space paths.
    """
    def __init__(self, frame_width):
        self.frame_width = frame_width
        # We define 3 basic spatial zones: Left, Center (Safe Path), Right
        self.zones = {
            "LEFT": False,   # True if blocked
            "CENTER": False,
            "RIGHT": False
        }
        
        self.zone_boundaries = {
            "LEFT": (-float('inf'), -0.4),
            "CENTER": (-0.4, 0.4),
            "RIGHT": (0.4, float('inf'))
        }

    def build_grid(self, trackers):
        """
        Projects all active trackers onto the occupancy grid.
        Marks a zone as blocked if an object is dangerously close or on a collision course.
        """
        for tracker in trackers:
            box = tracker.get_current_box()
            x, y, w, h = box
            
            if w <= 0:
                continue

            real_w = get_real_width(tracker.label)
            distance_z = estimate_distance(w, tracker.label)
            # Match voice.py risk evaluation horizon: objects out to 5.0 meters are considered
            if distance_z > 5.0:
                continue

            cx = x + (w / 2)
            horizontal_deviation_x = calculate_horizontal_deviation(cx, self.frame_width, distance_z)
            
            # The object spans from (center - width/2) to (center + width/2)
            obj_left_edge = horizontal_deviation_x - (real_w / 2)
            obj_right_edge = horizontal_deviation_x + (real_w / 2)
            
            # Check which zones are blocked by this object
            for zone_name, (z_left, z_right) in self.zone_boundaries.items():
                # A zone is blocked if the object's physical span overlaps with the zone
                if obj_right_edge > z_left and obj_left_edge < z_right:
                    self.zones[zone_name] = True
                    
    def get_free_path(self, target_avoidance_direction=None):
        """
        Calculates the safest free zone to move into.
        If target_avoidance_direction is provided ('left' or 'right'), tries to prioritize that.
        Returns: 'left', 'right', 'center', or 'blocked'
        """
        if not self.zones["CENTER"]:
            return "center"
            
        left_free = not self.zones["LEFT"]
        right_free = not self.zones["RIGHT"]
        
        if left_free and right_free:
            # Both sides are free, pick the preferred one if provided
            if target_avoidance_direction == 'left':
                return "left"
            elif target_avoidance_direction == 'right':
                return "right"
            else:
                return "right" # Default to right side step
                
        elif left_free:
            return "left"
        elif right_free:
            return "right"
            
        return "blocked"
