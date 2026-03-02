import numpy as np
import plotly.graph_objects as go
import os

# --- 1. SYSTEM PARAMETERS ---
H1, L0 = 0.4, 1.1
L1 = 0.3
L2 = 0.1
H4, L3 = 0.1, 0.1
L4 = 0.5
A1 = 0.1  # Added Lateral Offset

# --- 2. FORWARD KINEMATICS (Updated with A1) ---
def forward_kinematics(q):
    """
    Computes Cartesian Position (x, y, z) from Joint Angles.
    Updated to include A1 offset based on the robot configuration.
    """
    t1, d2, d3, t4 = q
    phi = t1 - t4
    
    # Position calculations including Lateral Offset A1
    px = L4 * np.sin(phi) + A1 * np.cos(t1) + (d2 + L1) * np.sin(t1)
    py = -L4 * np.cos(phi) + A1 * np.sin(t1) - (d2 + L1) * np.cos(t1)
    pz = (H1 + L0) - (d3 + L2 + H4 + L3)
    
    return [px, py, pz]

# --- 3. DEFINE KEYFRAMES (Pizza Turning Maneuver) ---
# Joints: [Theta1 (rad), d2 (m), d3 (m), Theta4 (rad)]
keyframes = [
    # 1-4: Approach and pick up
    {'q': [0, 0.9, 0.15, 0], 'label': '1.Start'},
    {'q': [0, 1.2, 0.15, 0], 'label': '2.Slide'},
    {'q': [0, 1.2, 0.10, 0], 'label': '3.Lift'},
    {'q': [0, 0.3, 0.10, 0], 'label': '4.Retract'},
    
    # 5: Rotate 90 degrees to Oven Axis
    {'q': [np.pi/2, 0.3, 0.10, 0], 'label': '5.Rotate'},
    
    # 6-8: Insert and Drop
    {'q': [np.pi/2, 1.5, 0.10, 0], 'label': '6.Extend'},
    {'q': [np.pi/2, 1.5, 0.10, np.deg2rad(30)], 'label': '7.Align'},
    {'q': [np.pi/2, 1.5, 0.15, np.deg2rad(30)], 'label': '8.Drop'},
    
    # 9-10: Flick/Slide off
    {'q': [np.pi/2, 1.5, 0.15, np.deg2rad(-30)], 'label': '9.SlideOff'},
    {'q': [np.pi/2, 1.5, 0.10, np.deg2rad(-30)], 'label': '10.LiftClear'},
    
    # 11-12: Reset
    {'q': [np.pi/2, 1.5, 0.10, 0], 'label': '11.Reset'},
    {'q': [np.pi/2, 0.3, 0.10, 0], 'label': '12.Home'},
]

# --- 4. OUTPUT COORDINATES ---
print(f"{'STEP':<15} | {'X (m)':<8} | {'Y (m)':<8} | {'Z (m)':<8}")
print("-" * 50)

for step in keyframes:
    pos = forward_kinematics(step['q'])
    step['pos'] = pos # Store for plotting
    print(f"{step['label']:<15} | {pos[0]:<8.3f} | {pos[1]:<8.3f} | {pos[2]:<8.3f}")

# --- 5. GENERATE TRAJECTORY LINES ---
path_x, path_y, path_z = [], [], []
key_x, key_y, key_z, key_labels = [], [], [], []

def interpolate_points(q_start, q_end, steps=15):
    qs = []
    # Linear interpolation in Joint Space
    for i in range(steps):
        alpha = i / (steps - 1)
        q = np.array(q_start) * (1 - alpha) + np.array(q_end) * alpha
        qs.append(q)
    return qs

for i in range(len(keyframes)):
    # Current Point
    p_curr = keyframes[i]['pos']
    key_x.append(p_curr[0]); key_y.append(p_curr[1]); key_z.append(p_curr[2])
    key_labels.append(keyframes[i]['label'])
    
    # Draw Line from Previous Point
    if i > 0:
        q_curr = keyframes[i]['q']
        q_prev = keyframes[i-1]['q']
        
        # Interpolate to create smooth path in Cartesian space
        interp_qs = interpolate_points(q_prev, q_curr)
        for q in interp_qs:
            p = forward_kinematics(q)
            path_x.append(p[0]); path_y.append(p[1]); path_z.append(p[2])
    else:
        # Start point
        path_x.append(p_curr[0]); path_y.append(p_curr[1]); path_z.append(p_curr[2])

# --- 6. PLOTLY VISUALIZATION & AUTOSAVE ---
fig = go.Figure()

# A. Trajectory
fig.add_trace(go.Scatter3d(
    x=path_x, y=path_y, z=path_z,
    mode='lines', line=dict(color='blue', width=4), name='Trajectory'
))

# B. Keyframes
fig.add_trace(go.Scatter3d(
    x=key_x, y=key_y, z=key_z,
    mode='markers+text',
    marker=dict(size=6, color='red'),
    text=key_labels, textposition="top center", name='Keyframes'
))

# C. Origin Reference
fig.add_trace(go.Scatter3d(
    x=[0, 0], y=[0, 0], z=[0, 1.5],
    mode='lines', line=dict(color='black', dash='dash', width=2), name='Z-Axis'
))

fig.update_layout(
    title="Pizza Robot: Task Space Trajectory",
    scene=dict(
        xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
        aspectmode='data', camera=dict(eye=dict(x=1.5, y=-1.5, z=1.5))
    ),
    width=1000, height=800
)

# Autosave Logic
filename_viz = "pizza_task_trajectory.html"
fig.write_html(filename_viz)
print(f"\nSaved 3D Task Trajectory to {filename_viz}")

fig.show()