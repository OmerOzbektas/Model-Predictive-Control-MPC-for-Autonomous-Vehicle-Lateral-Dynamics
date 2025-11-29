import numpy as np

def generate_reference_path(t_total, dt, vx, path_type='double_lane_change'):
    """
    Generates the reference trajectory (Y and Psi) for the vehicle to track.
    """
    t = np.arange(0, t_total + dt, dt)
    x = vx * t
    
    if path_type == 'double_lane_change':
        # Smooth double lane change using hyperbolic tangent
        y_ref = 4.0 * np.tanh(t - t_total/2)
    else:
        # Simple sinusoidal path
        y_ref = 4.0 * np.sin(0.5 * t)

    # Calculate Yaw reference (Psi_ref) from the derivative of Y
    # psi ~ arctan(dy/dx)
    dy = np.gradient(y_ref, x)
    psi_ref = np.arctan(dy)
    
    # Stack into a matrix [N, 2] -> [[psi_0, y_0], [psi_1, y_1]...]
    refs = np.vstack([psi_ref, y_ref]).T
    
    return t, x, refs