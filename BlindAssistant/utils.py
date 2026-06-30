# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================

# Camera Intrinsic Assumptions
FOCAL_LENGTH = 650.0  # Approx focal length in pixels for a 720p webcam
ASSUMED_REAL_WIDTH = 0.4  # Assumed real-world width of an object in meters (used for depth estimation)

# Pipeline Config
FPS_ASSUMPTION = 10.0  # Assumed FPS if we can't calculate it dynamically
MIN_TRACKING_FRAMES = 3  # Minimum frames an object must be tracked before we consider its velocity valid

# ==========================================
# UTILITY MATH FUNCTIONS
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

def estimate_distance(pixel_width):
    """
    Estimates the Z-distance (depth) to the object using Monocular Vision.
    Z = (Focal_Length * Real_Width) / Pixel_Width
    """
    if pixel_width <= 0:
        return float('inf')
    return (ASSUMED_REAL_WIDTH * FOCAL_LENGTH) / pixel_width

def calculate_horizontal_deviation(center_x, frame_width, distance_z):
    """
    Calculates how far left (-) or right (+) the object is from the camera center in real-world meters.
    """
    image_center_x = frame_width / 2.0
    # X = (x_pixel_diff * Z) / Focal_Length
    return ((center_x - image_center_x) * distance_z) / FOCAL_LENGTH
