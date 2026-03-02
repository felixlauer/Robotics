import numpy as np

# --- 1. SYSTEM CONSTANTS ---
# Robot geometry used to calculate Z height from d3
H1, L0 = 0.4, 1.1
L2, H4, L3 = 0.1, 0.1, 0.1
# Formula: Z = (H1 + L0) - (d3 + L2 + H4 + L3)
# Z_base = 1.5 - 0.3 = 1.2
# z_low (d3=0.15) = 1.2 - 0.15 = 1.05 m
# z_high (Lift 2cm) = 1.05 + 0.02 = 1.07 m

Z_LOW = 1.05
Z_HIGH = 1.07

# Pizza Circle Definition
CENTER = np.array([2.3, 0.0]) # meters
RADIUS = 0.15                 # meters

# Angles for Points (in radians relative to Center)
# B = (2.3, -0.15) -> Angle -pi/2 (-90 deg)
# C = (2.15, 0.0)  -> Angle -pi   (-180 deg)
# D = (2.3, 0.15)  -> Angle -3pi/2 (-270 deg) [Moving "left" around the circle]

ANGLE_B = -np.pi / 2
ANGLE_D = -3 * np.pi / 2

# --- 2. GENERATE TRAJECTORY ---
trajectory_cartesian = []

# Step 1: Start at B (Low)
# (2.3, -0.15, 1.05)
trajectory_cartesian.append([CENTER[0], CENTER[1] - RADIUS, Z_LOW])

# Step 2: Lift to B (High)
# (2.3, -0.15, 1.07)
trajectory_cartesian.append([CENTER[0], CENTER[1] - RADIUS, Z_HIGH])

# Step 3: Arc High (B -> C -> D)
# Generates 10 points along the arc at Z_HIGH
angles_high = np.linspace(ANGLE_B, ANGLE_D, 10)
for theta in angles_high:
    x = CENTER[0] + RADIUS * np.cos(theta)
    y = CENTER[1] + RADIUS * np.sin(theta)
    # We skip the first point if it's identical to Step 2, 
    # but for safety in trajectory planning, repeating it is fine (zero velocity start).
    # Here we include it to ensure the arc shape is fully defined with 10 samples.
    trajectory_cartesian.append([x, y, Z_HIGH])

# Step 4: Lower at D (High -> Low)
# Last point of Step 3 was D_High. Now we add D_Low.
# (2.3, 0.15, 1.05)
trajectory_cartesian.append([CENTER[0], CENTER[1] + RADIUS, Z_LOW])

# Step 5: Arc Low (Reverse: D -> C -> B)
# Generates 10 points along the arc at Z_LOW
# Note: Reverse order of angles (ANGLE_D to ANGLE_B)
angles_low = np.linspace(ANGLE_D, ANGLE_B, 10)
for theta in angles_low:
    x = CENTER[0] + RADIUS * np.cos(theta)
    y = CENTER[1] + RADIUS * np.sin(theta)
    trajectory_cartesian.append([x, y, Z_LOW])

# --- 3. OUTPUT ---
print(f"{'#':<3} | {'X (m)':<8} | {'Y (m)':<8} | {'Z (m)':<8} | {'Description'}")
print("-" * 55)

labels = ["Start (B_Low)", "Lift (B_High)"] + \
         ["Arc High"] * 10 + \
         ["Lower (D_Low)"] + \
         ["Arc Low"] * 10

for i, (point, label) in enumerate(zip(trajectory_cartesian, labels)):
    # Simple logic to label start/end of arcs for clarity
    if i == 2: label = "Arc High (Start)"
    if i == 11: label = "Arc High (End)"
    if i == 13: label = "Arc Low (Start)"
    if i == 22: label = "Arc Low (End)"
    
    print(f"{i+1:<3} | {point[0]:<8.3f} | {point[1]:<8.3f} | {point[2]:<8.3f} | {label}")