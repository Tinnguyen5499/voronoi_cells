import numpy as np

def global_to_rel_state(xA, xB):
    """Convert global states (robot A and B) to the relative state of B w.r.t A."""
    xA = np.array(xA)
    xB = np.array(xB)
    xrel = xB - xA

    # Wrap theta to [-pi, pi]
    xrel[2] = (xrel[2] + np.pi) % (2 * np.pi)
    if xrel[2] > np.pi:
        xrel[2] -= 2 * np.pi

    # Rotation matrix to rotate into A's frame
    rot_mat = np.array([
        [np.cos(xA[2]), np.sin(xA[2])],
        [-np.sin(xA[2]), np.cos(xA[2])]
    ])
    
    xrel[:2] = rot_mat @ xrel[:2]

    return xrel

# Example test
xA = [1, 1, np.pi/4]
xB = [1.5, 1.2, np.pi/2]

xrel = global_to_rel_state(xA, xB)
print("Relative state of B w.r.t A:")
print(f"x_rel: {xrel}")
