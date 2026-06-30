# Risk Categories
RISK_LOW = "Low"
RISK_MEDIUM = "Medium"
RISK_HIGH = "High"
RISK_CRITICAL = "Critical"

LETHAL_CLASSES = {"car", "bus", "truck", "motorcycle", "train", "airplane"}

def assess_risk(label, distance_z, time_to_collision, will_collide):
    """
    Categorizes the danger level based on distance, TTC, and object lethality.
    """
    is_lethal = label in LETHAL_CLASSES
    
    if not will_collide:
        # Even if not on a direct collision course, dangerous objects very close are High risk
        if distance_z < 1.5 and is_lethal:
            return RISK_HIGH
        if distance_z < 1.0:
            return RISK_MEDIUM
        return RISK_LOW
    
    if is_lethal:
        # Lethal kinetic objects (cars, buses) require much more reaction time
        if time_to_collision < 4.0:
            return RISK_CRITICAL
        return RISK_HIGH
        
    # Standard objects (people, chairs)
    if time_to_collision < 1.5:
        return RISK_CRITICAL
    elif time_to_collision < 3.0:
        return RISK_HIGH
    else:
        return RISK_MEDIUM
