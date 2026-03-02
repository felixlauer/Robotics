import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- 1. SYSTEM PARAMETERS ---
H1, L0 = 0.4, 1.1
L1 = 0.3
L2 = 0.1
H4, L3 = 0.1, 0.1
L4 = 0.5
A1 = 0.1 

# --- 2. KINEMATICS & JACOBIAN ---
def forward_kinematics(q):
    """
    Computes Cartesian Position (x, y, z) from Joint Angles.
    """
    t1, d2, d3, t4 = q
    phi = t1 - t4
    
    # Position calculations including Lateral Offset A1
    px = L4 * np.sin(phi) + A1 * np.cos(t1) + (d2 + L1) * np.sin(t1)
    py = -L4 * np.cos(phi) + A1 * np.sin(t1) - (d2 + L1) * np.cos(t1)
    pz = (H1 + L0) - (d3 + L2 + H4 + L3)
    
    return np.array([px, py, pz, phi])

def jacobian(q):
    """
    Calculates the 6x4 Jacobian Matrix.
    """
    t1, d2, d3, t4 = q
    s1, c1 = np.sin(t1), np.cos(t1)
    # phi = t1 - t4
    s_phi, c_phi = np.sin(t1 - t4), np.cos(t1 - t4)
    
    J = np.zeros((6, 4))
    
    # --- Linear Velocity Part (Rows 0-2) ---
    
    # Row 0: Vx
    # dx/dt1
    J[0, 0] = L4 * c_phi - A1 * s1 + (d2 + L1) * c1
    # dx/dd2
    J[0, 1] = s1
    # dx/dt4
    J[0, 3] = -L4 * c_phi
    
    # Row 1: Vy
    # dy/dt1
    J[1, 0] = L4 * s_phi + A1 * c1 + (d2 + L1) * s1
    # dy/dd2
    J[1, 1] = -c1
    # dy/dt4
    J[1, 3] = -L4 * s_phi 
    
    # Row 2: Vz
    # dz/dd3
    J[2, 2] = -1
    
    # --- Angular Velocity Part (Rows 3-5) ---
    
    # Row 5: Wz
    J[5, 0] = 1
    J[5, 3] = -1
    
    return J

# --- 3. TRAJECTORY GENERATOR (LSPB with Custom Times) ---
def generate_lspb_custom_times(waypoints, segment_times, acc_limit=1.5):
    dt = 0.02 # 50Hz resolution
    
    # Calculate arrival times based on the segment durations
    t_arrival = np.concatenate(([0], np.cumsum(segment_times)))
    total_time = t_arrival[-1]
    
    # Calculate Linear Velocities for each segment (Distance / Time)
    linear_velocities = np.diff(waypoints, axis=0) / np.array(segment_times)[:, None]
    
    # Generate Time Vector
    t_total = np.arange(0, total_time, dt)
    
    # Storage arrays
    q_traj = np.zeros((len(t_total), 4))
    qd_traj = np.zeros((len(t_total), 4))
    qdd_traj = np.zeros((len(t_total), 4))
    
    for j in range(4): # Joint Loop
        for k, t in enumerate(t_total):
            # 1. Identify Segment
            seg_idx = np.searchsorted(t_arrival, t, side='right') - 1
            if seg_idx >= len(linear_velocities): seg_idx = len(linear_velocities) - 1
            
            # 2. Get Targets
            v_target = linear_velocities[seg_idx, j]
            time_to_next = t_arrival[seg_idx+1] - t
            time_from_prev = t - t_arrival[seg_idx]
            
            # 3. Dynamic Blend Calculation
            if seg_idx < len(linear_velocities) - 1:
                v_next = linear_velocities[seg_idx+1, j]
            else:
                v_next = 0
            
            # Blend time tb = |Delta V| / a
            # Use a safety clamp so blending doesn't exceed the segment duration
            tb = abs(v_next - v_target) / acc_limit
            tb = min(tb, segment_times[seg_idx]*0.9) 

            # --- APPLY BLENDS ---
            # A. Start Blend
            if seg_idx == 0 and time_from_prev < abs(v_target)/acc_limit:
                 acc = np.sign(v_target) * acc_limit
                 qdd_traj[k, j] = acc
                 qd_traj[k, j] = acc * time_from_prev
                 q_traj[k, j] = waypoints[0, j] + 0.5 * acc * time_from_prev**2
                 
            # B. Transition Blend (Corner) - Entering
            elif time_to_next < 0.5 * tb:
                 acc = np.sign(v_next - v_target) * acc_limit
                 dt_blend = (0.5 * tb) - time_to_next
                 qdd_traj[k, j] = acc
                 qd_traj[k, j] = v_target + acc * dt_blend
                 if k > 0: q_traj[k, j] = q_traj[k-1, j] + qd_traj[k, j] * dt
                 
            # C. Transition Blend (Corner) - Leaving
            elif seg_idx > 0 and time_from_prev < 0.5 * (abs(linear_velocities[seg_idx,j] - linear_velocities[seg_idx-1,j])/acc_limit):
                 v_prev = linear_velocities[seg_idx-1, j]
                 acc = np.sign(v_target - v_prev) * acc_limit
                 qdd_traj[k, j] = acc
                 if k > 0: 
                     qd_traj[k, j] = qd_traj[k-1, j] + acc * dt
                     q_traj[k, j] = q_traj[k-1, j] + qd_traj[k, j] * dt

            # D. Linear Cruise
            else:
                 qdd_traj[k, j] = 0
                 qd_traj[k, j] = v_target
                 if k > 0: q_traj[k, j] = q_traj[k-1, j] + v_target * dt
                 else: q_traj[k, j] = waypoints[0, j]

    return t_total, q_traj, qd_traj, qdd_traj

# --- 4. DEFINE WAYPOINTS ---
waypoints = np.array([
    [0, 0.9, 0.15, 0],              # 1. Start
    [0, 1.2, 0.15, 0],              # 2. Slide
    [0, 1.2, 0.10, 0],              # 3. Lift
    [0, 0.3, 0.10, 0],              # 4. Retract
    [np.pi/2, 0.3, 0.10, 0],        # 5. Rotate
    [np.pi/2, 1.5, 0.10, 0],        # 6. Extend
    [np.pi/2, 1.5, 0.10, np.deg2rad(30)], # 7. Align
    [np.pi/2, 1.5, 0.15, np.deg2rad(30)], # 8. Drop
    [np.pi/2, 1.5, 0.15, np.deg2rad(-30)],# 9. SlideOff
    [np.pi/2, 1.5, 0.10, np.deg2rad(-30)],# 10. LiftClear
    [np.pi/2, 1.5, 0.10, 0],        # 11. Reset
    [np.pi/2, 0.3, 0.10, 0]         # 12. Home
])

# --- 5. TIME SCALING (Goal: 60s Total) ---

# A. Define Max Velocities per Joint
# [Theta1 (rad/s), d2 (m/s), d3 (m/s), Theta4 (rad/s)]
v_limits = np.array([1.0, 0.5, 0.05, 1.5]) 
# Note: d3 set to 0.05 m/s (5cm/s) for safety/smoothness

# B. Calculate Minimum Time for each Segment
min_segment_times = []
for i in range(len(waypoints) - 1):
    dist_per_joint = np.abs(waypoints[i+1] - waypoints[i])
    
    # Time required for each joint to complete its move
    time_per_joint = dist_per_joint / v_limits
    
    # The segment must take as long as the slowest joint
    seg_time = np.max(time_per_joint)
    
    # Enforce a small floor (0.5s) to avoid divide-by-zero on tiny adjustments
    seg_time = max(seg_time, 0.5)
    
    min_segment_times.append(seg_time)

# C. Scale to exactly 60 seconds
current_total = sum(min_segment_times)
target_total = 60.0
scaling_factor = target_total / current_total

final_segment_times = [t * scaling_factor for t in min_segment_times]

print(f"--- TIMING STRATEGY ---")
print(f"Min Total Time needed: {current_total:.2f} s")
print(f"Scaling Factor:        {scaling_factor:.2f} x")
print(f"Final Total Time:      {sum(final_segment_times):.2f} s")
print("-" * 30)

# --- 6. RUN GENERATOR ---
# Using a low acceleration limit creates smoother blends
traj_time, traj_q, traj_qd, traj_qdd = generate_lspb_custom_times(
    waypoints, final_segment_times, acc_limit=0.5
)

# --- 7. COMPUTE CARTESIAN DATA ---
traj_x, traj_xd, traj_xdd = [], [], []
for k in range(len(traj_time)):
    q_curr = traj_q[k]
    qd_curr = traj_qd[k]
    pose = forward_kinematics(q_curr) 
    traj_x.append(pose[:3])
    J = jacobian(q_curr)
    vel_spatial = J @ qd_curr 
    traj_xd.append(vel_spatial[:3]) 

traj_x = np.array(traj_x)
traj_xd = np.array(traj_xd)
# Numerical Acceleration
traj_xdd = np.gradient(traj_xd, axis=0) / 0.02 

# --- 8. PLOTTING & AUTOSAVE ---

# Joint Space
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
joint_names = ['Theta1', 'd2', 'd3', 'Theta4']
for j in range(4): axs[0].plot(traj_time, traj_q[:, j], label=joint_names[j])
axs[0].set_ylabel('Pos'); axs[0].set_title(f'Joint Space (Total T={traj_time[-1]:.1f}s)'); axs[0].legend()
axs[0].grid(True)
for j in range(4): axs[1].plot(traj_time, traj_qd[:, j], label=joint_names[j])
axs[1].set_ylabel('Vel'); axs[1].grid(True)
for j in range(4): axs[2].plot(traj_time, traj_qdd[:, j], label=joint_names[j])
axs[2].set_ylabel('Acc'); axs[2].set_xlabel('Time (s)'); axs[2].grid(True)

# Autosave Joint Space
filename_joint = "lspb_jointspace.png"
plt.savefig(filename_joint)
print(f"Saved Joint Space plot to {filename_joint}")
plt.show()

# Cartesian Space
fig2, axs2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
axis_names = ['X', 'Y', 'Z']
for k in range(3): axs2[0].plot(traj_time, traj_x[:, k], label=axis_names[k])
axs2[0].set_ylabel('Pos (m)'); axs2[0].set_title('Cartesian Space'); axs2[0].legend(); axs2[0].grid(True)
for k in range(3): axs2[1].plot(traj_time, traj_xd[:, k], label=axis_names[k])
axs2[1].set_ylabel('Vel (m/s)'); axs2[1].grid(True)
for k in range(3): axs2[2].plot(traj_time, traj_xdd[:, k], label=axis_names[k])
axs2[2].set_ylabel('Acc (m/s^2)'); axs2[2].set_xlabel('Time (s)'); axs2[2].grid(True)

# Autosave Cartesian Space
filename_cart = "lspb_cartesianspace.png"
plt.savefig(filename_cart)
print(f"Saved Cartesian Space plot to {filename_cart}")
plt.show()

# 3D Path
fig3 = go.Figure()
fig3.add_trace(go.Scatter3d(x=traj_x[:,0], y=traj_x[:,1], z=traj_x[:,2],
    mode='lines', line=dict(color='blue', width=4), name='Continuous Path'))
wx, wy, wz = [], [], []
for wp in waypoints:
    p = forward_kinematics(wp)
    wx.append(p[0]); wy.append(p[1]); wz.append(p[2])
fig3.add_trace(go.Scatter3d(x=wx, y=wy, z=wz, mode='markers', marker=dict(size=5, color='red'), name='Via Points'))
fig3.update_layout(title="Resulting 3D Trajectory", scene=dict(aspectmode='data'))

# Autosave 3D Path
filename_3d = "lspb_trajectory_3d.html" 
fig3.write_html(filename_3d)
print(f"Saved 3D Path plot to {filename_3d}")
fig3.show()