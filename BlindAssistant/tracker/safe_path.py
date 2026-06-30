def calculate_avoidance_instruction(horizontal_deviation_x, object_real_width, velocity_x_mps, distance_z, occupancy_grid):
    """
    Calculates the best avoidance path by checking the global OccupancyGrid for free space.
    Returns: (action, distance_to_move)
        action: 'stop', 'move_left', 'move_right', 'step_back', 'wait'
        distance_to_move: in meters
    """
    speed_x = abs(velocity_x_mps)
    dynamic_buffer = 0.8 + (speed_x * 0.5) 
    
    # 1. Determine naive ideal avoidance direction based on the primary threat
    preferred_direction = 'right' if horizontal_deviation_x < 0 else 'left'
    
    # 2. Check the global occupancy grid to see if the preferred direction is actually safe
    actual_safe_path = occupancy_grid.get_free_path(preferred_direction)
    
    if actual_safe_path == "blocked":
        # All paths (left, right, center) are blocked!
        # The only safe action is to stop moving completely.
        return 'stop', 0.0
        
    if speed_x > 1.5: 
        # Object moving rapidly across, don't try to dodge, just stop and wait
        return 'stop', 0.0

    if actual_safe_path == "left":
        object_left_edge = horizontal_deviation_x - (object_real_width / 2)
        required_move = abs(object_left_edge) + dynamic_buffer
        return 'move_left', round(required_move, 1)
        
    elif actual_safe_path == "right":
        object_right_edge = horizontal_deviation_x + (object_real_width / 2)
        required_move = abs(object_right_edge) + dynamic_buffer
        return 'move_right', round(required_move, 1)
        
    elif actual_safe_path == "center":
        # No object is actually blocking the center path right now
        return 'none', 0.0
        
    return 'stop', 0.0
