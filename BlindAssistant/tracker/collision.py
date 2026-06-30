def calculate_ttc(distance_z, velocity_z_mps):
    """
    Calculates Time-To-Collision (TTC).
    distance_z: Current depth in meters.
    velocity_z_mps: Velocity in the Z-axis (meters/second). Positive means moving towards the camera.
    """
    if velocity_z_mps <= 0.1:  # Not moving towards the user significantly
        return float('inf')
        
    ttc = distance_z / velocity_z_mps
    return ttc

def predict_collision(horizontal_deviation_x, distance_z, velocity_x_mps, velocity_z_mps, time_horizon=2.0):
    """
    Predicts if the object will intersect the user's walking path.
    Assumes user is walking straight and occupies X between -0.4m and +0.4m.
    
    Returns: (will_collide, collision_time, intersection_x)
    """
    # If the object is not moving, check if it's already in the user's path
    if abs(velocity_x_mps) < 0.1 and abs(velocity_z_mps) < 0.1:
        if abs(horizontal_deviation_x) < 0.4 and distance_z < 2.0:
            return True, 0.0, horizontal_deviation_x
        return False, float('inf'), horizontal_deviation_x
        
    ttc = calculate_ttc(distance_z, velocity_z_mps)
    
    # We only care about collisions happening within our time horizon (e.g. next 2 seconds)
    if ttc > time_horizon:
        return False, ttc, 0.0
        
    # Project the X position at the Time-To-Collision
    # If the object takes `ttc` seconds to reach Z=0, where will it be on the X-axis?
    predicted_x = horizontal_deviation_x + (velocity_x_mps * ttc)
    
    # Check if the predicted X is within the user's bodily width (approx 0.8m wide, so -0.4 to 0.4)
    if abs(predicted_x) < 0.4:
        return True, ttc, predicted_x
        
    return False, ttc, predicted_x
