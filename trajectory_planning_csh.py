import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- 1. SYSTEM PARAMETERS (Kept same) ---
H1, L0 = 0.4, 1.1
L1 = 0.3
L2 = 0.1
H4, L3 = 0.1, 0.1
L4 = 0.5

# --- 2. KINEMATICS & JACOBIAN (Kept same) ---
def forward_kinematics(q):
    t1, d2, d3, t4 = q
    phi = t1 - t4
    wx = (d2 + L1) * np.sin(t1)
    wy = -(d2 + L1) * np.cos(t1)
    wz = (H1 + L0) - (d3 + L2 + H4 + L3)
    px = wx + L4 * np.sin(phi)
    py = wy - L4 * np.cos(phi)
    pz = wz
    return np.array([px, py, pz, phi])

def jacobian(q):
    t1, d2, d3, t4 = q
    s1, c1 = np.sin(t1), np.cos(t1)
    s14, c14 = np.sin(t1 - t4), np.cos(t1 - t4)
    J = np.zeros((6, 4))
    J[0, 0] = (d2 + L1) * c1 + L4 * c14
    J[0, 1] = s1
    J[0, 3] = -L4 * c14
    J[1, 0] = (d2 + L1) * s1 + L4 * s14
    J[1, 1] = -c1
    J[1, 3] = -L4 * s14
    J[2, 2] = -1 
    J[5, 0] = 1
    J[5, 3] = -1
    return J

# --- 3. TRAJECTORY GENERATOR: CUBIC HEURISTIC (Section 6.1.2) ---
def generate_cubic_heuristic(waypoints, segment_times):
    """
    Implements Cubic Polynomials with Via Points using the Tangent Heuristic.
    Ref: Lecture Notes Page 6, Eq 8 and Figure 4.
    """
    num_joints = waypoints.shape[1]
    num_points = waypoints.shape[0]
    num_segments = num_points - 1
    
    # 1. Calculate Velocities at Via Points (Heuristic)
    # Ref: Page 6, "Heuristic choice of velocities based on tangents" [cite: 137]
    via_velocities = np.zeros((num_points, num_joints))
    
    for j in range(num_joints):
        for i in range(num_points):
            if i == 0 or i == num_points - 1:
                # Constrain initial and final velocities to zero [cite: 66, 67]
                via_velocities[i, j] = 0.0
            else:
                # Calculate slopes of connecting lines 
                dist_prev = waypoints[i, j] - waypoints[i-1, j]
                t_prev = segment_times[i-1]
                slope_in = dist_prev / t_prev
                
                dist_next = waypoints[i+1, j] - waypoints[i, j]
                t_next = segment_times[i]
                slope_out = dist_next / t_next
                
                # Heuristic Logic:
                # If slope changes sign, set velocity to zero 
                if np.sign(slope_in) != np.sign(slope_out):
                    via_velocities[i, j] = 0.0
                else:
                    # Else, average the two slopes [cite: 140]
                    via_velocities[i, j] = 0.5 * (slope_in + slope_out)

    # 2. Generate Time Vector
    # We need high resolution (e.g. 50Hz = 0.02s)
    total_duration = sum(segment_times)
    t_total = np.arange(0, total_duration, 0.02)
    
    # Pre-allocate trajectory arrays
    q_traj = np.zeros((len(t_total), num_joints))
    qd_traj = np.zeros((len(t_total), num_joints))
    qdd_traj = np.zeros((len(t_total), num_joints))
    
    # Calculate arrival times for indexing
    arrival_times = np.concatenate(([0], np.cumsum(segment_times)))
    
    # 3. Compute Cubic Coefficients per Segment [cite: 126]
    for k, t_curr in enumerate(t_total):
        # Find which segment 'k' belongs to
        seg_idx = np.searchsorted(arrival_times, t_curr, side='right') - 1
        if seg_idx >= num_segments: seg_idx = num_segments - 1
        
        # Local time within the segment (t from 0 to tf)
        tf = segment_times[seg_idx]
        t_local = t_curr - arrival_times[seg_idx]
        
        for j in range(num_joints):
            # Parameters for Eq 8 [cite: 124]
            theta_0 = waypoints[seg_idx, j]
            theta_f = waypoints[seg_idx+1, j]
            dtheta_0 = via_velocities[seg_idx, j]
            dtheta_f = via_velocities[seg_idx+1, j]
            
            # Coefficients (Equation 8, Page 6) [cite: 124]
            a0 = theta_0
            a1 = dtheta_0
            a2 = (3/(tf**2)) * (theta_f - theta_0) - (2/tf)*dtheta_0 - (1/tf)*dtheta_f
            a3 = -(2/(tf**3)) * (theta_f - theta_0) + (1/(tf**2))*(dtheta_f + dtheta_0)
            
            # Evaluate Cubic Polynomial (Equation 2 & 3, Page 4) [cite: 70, 73]
            q_traj[k, j] = a0 + a1*t_local + a2*(t_local**2) + a3*(t_local**3)
            qd_traj[k, j] = a1 + 2*a2*t_local + 3*a3*(t_local**2)
            qdd_traj[k, j] = 2*a2 + 6*a3*t_local

    return t_total, q_traj, qd_traj, qdd_traj

# --- 4. DEFINE WAYPOINTS (Kept same) ---
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

# --- 5. TIME SCALING (Kept same logic as requested) ---
v_limits = np.array([1.0, 0.5, 0.05, 1.5]) 

min_segment_times = []
for i in range(len(waypoints) - 1):
    dist_per_joint = np.abs(waypoints[i+1] - waypoints[i])
    time_per_joint = dist_per_joint / v_limits
    seg_time = np.max(time_per_joint)
    seg_time = max(seg_time, 0.5)
    min_segment_times.append(seg_time)

current_total = sum(min_segment_times)
target_total = 60.0
scaling_factor = target_total / current_total
final_segment_times = [t * scaling_factor for t in min_segment_times]

print(f"--- TIMING STRATEGY ---")
print(f"Total Time: {sum(final_segment_times):.2f} s")
print("-" * 30)

# --- 6. RUN CUBIC GENERATOR ---
traj_time, traj_q, traj_qd, traj_qdd = generate_cubic_heuristic(
    waypoints, final_segment_times
)

# --- 7. COMPUTE CARTESIAN DATA (Kept same) ---
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
traj_xdd = np.gradient(traj_xd, axis=0) / 0.02 

# --- 8. PLOTTING (Kept same) ---
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
joint_names = ['Theta1', 'd2', 'd3', 'Theta4']
for j in range(4): axs[0].plot(traj_time, traj_q[:, j], label=joint_names[j])
axs[0].set_ylabel('Pos'); axs[0].set_title(f'Joint Space (Cubic Heuristic)'); axs[0].legend(); axs[0].grid(True)
for j in range(4): axs[1].plot(traj_time, traj_qd[:, j], label=joint_names[j])
axs[1].set_ylabel('Vel'); axs[1].grid(True)
for j in range(4): axs[2].plot(traj_time, traj_qdd[:, j], label=joint_names[j])
axs[2].set_ylabel('Acc'); axs[2].set_xlabel('Time (s)'); axs[2].grid(True)
plt.show()

fig2, axs2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
axis_names = ['X', 'Y', 'Z']
for k in range(3): axs2[0].plot(traj_time, traj_x[:, k], label=axis_names[k])
axs2[0].set_ylabel('Pos (m)'); axs2[0].set_title('Cartesian Space'); axs2[0].legend(); axs2[0].grid(True)
for k in range(3): axs2[1].plot(traj_time, traj_xd[:, k], label=axis_names[k])
axs2[1].set_ylabel('Vel (m/s)'); axs2[1].grid(True)
for k in range(3): axs2[2].plot(traj_time, traj_xdd[:, k], label=axis_names[k])
axs2[2].set_ylabel('Acc (m/s^2)'); axs2[2].set_xlabel('Time (s)'); axs2[2].grid(True)
plt.show()

fig3 = go.Figure()
fig3.add_trace(go.Scatter3d(x=traj_x[:,0], y=traj_x[:,1], z=traj_x[:,2],
    mode='lines', line=dict(color='blue', width=4), name='Continuous Path'))
wx, wy, wz = [], [], []
for wp in waypoints:
    p = forward_kinematics(wp)
    wx.append(p[0]); wy.append(p[1]); wz.append(p[2])
fig3.add_trace(go.Scatter3d(x=wx, y=wy, z=wz, mode='markers', marker=dict(size=5, color='red'), name='Via Points'))
fig3.update_layout(title="Resulting 3D Trajectory (Cubic)", scene=dict(aspectmode='data'))
fig3.show()