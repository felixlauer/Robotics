import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

# --- 1. SYSTEM PARAMETERS ---
H1, L0 = 0.4, 1.1
L1 = 0.3
L2 = 0.1
H4, L3 = 0.1, 0.1
L4 = 0.5
A1 = 0.1  # Added Lateral Offset

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
    J[0, 0] = L4 * c_phi - A1 * s1 + (d2 + L1) * c1
    J[0, 1] = s1
    J[0, 3] = -L4 * c_phi
    
    J[1, 0] = L4 * s_phi + A1 * c1 + (d2 + L1) * s1
    J[1, 1] = -c1
    J[1, 3] = -L4 * s_phi 
    
    J[2, 2] = -1
    
    # --- Angular Velocity Part (Rows 3-5) ---
    J[5, 0] = 1
    J[5, 3] = -1
    
    return J

def inverse_kinematics_analytical(target_pos, target_phi=0.0):
    """
    Analytical IK based on geometric derivation with Offset A1.
    """
    px, py, pz = target_pos
    theta_14 = target_phi 
    
    # 1. Vertical (d3)
    d3 = (H1 + L0 - L2 - H4 - L3) - pz
    
    # 2. Wrist Position (Decouple Tool)
    x_wrist = px - L4 * np.sin(theta_14)
    y_wrist = py + L4 * np.cos(theta_14)
    
    # 3. Base Angle (Theta1) & Extension (d2)
    R_sq = x_wrist**2 + y_wrist**2
    if R_sq < A1**2:
        return None # Unreachable
    
    term_sqrt = np.sqrt(R_sq - A1**2)
    d2 = term_sqrt - L1
    
    beta = np.arctan2(x_wrist, -y_wrist)
    delta = np.arctan2(A1, term_sqrt)
    theta1 = beta - delta
    
    # 4. Tool Angle (Theta4)
    theta4 = theta1 - theta_14
    
    return np.array([theta1, d2, d3, theta4])

# --- 3. TRAJECTORY GENERATOR: CUBIC HEURISTIC ---
def generate_cubic_heuristic(waypoints, segment_times):
    """
    Implements Cubic Polynomials with Via Points using the Tangent Heuristic.
    """
    num_joints = waypoints.shape[1]
    num_points = waypoints.shape[0]
    num_segments = num_points - 1
    
    # 1. Calculate Velocities at Via Points (Heuristic)
    via_velocities = np.zeros((num_points, num_joints))
    
    for j in range(num_joints):
        for i in range(num_points):
            if i == 0 or i == num_points - 1:
                via_velocities[i, j] = 0.0
            else:
                dist_prev = waypoints[i, j] - waypoints[i-1, j]
                t_prev = segment_times[i-1]
                slope_in = dist_prev / t_prev
                
                dist_next = waypoints[i+1, j] - waypoints[i, j]
                t_next = segment_times[i]
                slope_out = dist_next / t_next
                
                if np.sign(slope_in) != np.sign(slope_out):
                    via_velocities[i, j] = 0.0
                else:
                    via_velocities[i, j] = 0.5 * (slope_in + slope_out)

    # 2. Generate Time Vector
    total_duration = sum(segment_times)
    t_total = np.arange(0, total_duration, 0.02)
    
    q_traj = np.zeros((len(t_total), num_joints))
    qd_traj = np.zeros((len(t_total), num_joints))
    qdd_traj = np.zeros((len(t_total), num_joints))
    
    arrival_times = np.concatenate(([0], np.cumsum(segment_times)))
    
    # 3. Compute Cubic Coefficients per Segment
    for k, t_curr in enumerate(t_total):
        seg_idx = np.searchsorted(arrival_times, t_curr, side='right') - 1
        if seg_idx >= num_segments: seg_idx = num_segments - 1
        
        tf = segment_times[seg_idx]
        t_local = t_curr - arrival_times[seg_idx]
        
        for j in range(num_joints):
            theta_0 = waypoints[seg_idx, j]
            theta_f = waypoints[seg_idx+1, j]
            dtheta_0 = via_velocities[seg_idx, j]
            dtheta_f = via_velocities[seg_idx+1, j]
            
            a0 = theta_0
            a1 = dtheta_0
            a2 = (3/(tf**2)) * (theta_f - theta_0) - (2/tf)*dtheta_0 - (1/tf)*dtheta_f
            a3 = -(2/(tf**3)) * (theta_f - theta_0) + (1/(tf**2))*(dtheta_f + dtheta_0)
            
            q_traj[k, j] = a0 + a1*t_local + a2*(t_local**2) + a3*(t_local**3)
            qd_traj[k, j] = a1 + 2*a2*t_local + 3*a3*(t_local**2)
            qdd_traj[k, j] = 2*a2 + 6*a3*t_local

    return t_total, q_traj, qd_traj, qdd_traj

# --- 4. PIZZA COORDINATES & PLAN ---
cartesian_list = np.array([
    [2.300, -0.050, 1.050], # 1. Start (Under Pizza)
    [2.300, -0.050, 1.070], # 2. Lift
    [2.300, -0.050, 1.070], # 3. Arc High Start
    [2.249, -0.041, 1.070], # 4
    [2.204, -0.015, 1.070], # 5
    [2.170,  0.025, 1.070], # 6
    [2.152,  0.074, 1.070], # 7
    [2.152,  0.126, 1.070], # 8
    [2.170,  0.175, 1.070], # 9
    [2.204,  0.215, 1.070], # 10
    [2.249,  0.241, 1.070], # 11
    [2.300,  0.250, 1.070], # 12. Arc High End (Rotated)
    [2.300,  0.250, 1.050], # 13. Lower (Place)
    [2.300,  0.250, 1.050], # 14. Arc Low Start (Return)
    [2.249,  0.241, 1.050], # 15
    [2.204,  0.215, 1.050], # 16
    [2.170,  0.175, 1.050], # 17
    [2.152,  0.126, 1.050], # 18
    [2.152,  0.074, 1.050], # 19
    [2.170,  0.025, 1.050], # 20
    [2.204, -0.015, 1.050], # 21
    [2.249, -0.041, 1.050], # 22
    [2.300, -0.050, 1.050]  # 23. Arc Low End (Reset)
])

# Convert to Joint Space using Analytical IK
joint_waypoints = []
desired_phi = 0.0 
for pt in cartesian_list:
    q_sol = inverse_kinematics_analytical(pt, desired_phi)
    if q_sol is None:
        print(f"Warning: Point {pt} unreachable with offset {A1}")
    else:
        joint_waypoints.append(q_sol)
joint_waypoints = np.array(joint_waypoints)

# Time Scaling (0.5s per segment)
segment_times = [0.5] * (len(joint_waypoints) - 1)

# Generate Trajectory
traj_time, traj_q, traj_qd, traj_qdd = generate_cubic_heuristic(joint_waypoints, segment_times)

# Compute Cartesian Data for Plotting
traj_x, traj_xd, traj_xdd = [], [], []
dt = 0.02
for k in range(len(traj_time)):
    q_curr = traj_q[k]
    qd_curr = traj_qd[k]
    
    # Cartesian Pos
    pose = forward_kinematics(q_curr) 
    traj_x.append(pose[:3])
    
    # Cartesian Vel (J * qd)
    J = jacobian(q_curr)
    vel_spatial = J @ qd_curr 
    traj_xd.append(vel_spatial[:3]) 

traj_x = np.array(traj_x)
traj_xd = np.array(traj_xd)
# Cartesian Acc (Numerical Gradient)
traj_xdd = np.gradient(traj_xd, axis=0) / dt 

# --- 5. PLOTTING & AUTOSAVE ---

# Joint Space Plot
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
joint_names = ['Theta1', 'd2', 'd3', 'Theta4']
for j in range(4): axs[0].plot(traj_time, traj_q[:, j], label=joint_names[j])
axs[0].set_ylabel('Pos'); axs[0].set_title(f'Joint Space (Cubic Heuristic)'); axs[0].legend(); axs[0].grid(True)
for j in range(4): axs[1].plot(traj_time, traj_qd[:, j], label=joint_names[j])
axs[1].set_ylabel('Vel'); axs[1].grid(True)
for j in range(4): axs[2].plot(traj_time, traj_qdd[:, j], label=joint_names[j])
axs[2].set_ylabel('Acc'); axs[2].set_xlabel('Time (s)'); axs[2].grid(True)

# Autosave Joint Space
filename_joint = "pizza_jointspace.png"
plt.savefig(filename_joint)
print(f"Saved Joint Space plot to {filename_joint}")
plt.show()

# Cartesian Space Plot
fig2, axs2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
axis_names = ['X', 'Y', 'Z']
for k in range(3): axs2[0].plot(traj_time, traj_x[:, k], label=axis_names[k])
axs2[0].set_ylabel('Pos (m)'); axs2[0].set_title('Cartesian Space'); axs2[0].legend(); axs2[0].grid(True)
for k in range(3): axs2[1].plot(traj_time, traj_xd[:, k], label=axis_names[k])
axs2[1].set_ylabel('Vel (m/s)'); axs2[1].grid(True)
for k in range(3): axs2[2].plot(traj_time, traj_xdd[:, k], label=axis_names[k])
axs2[2].set_ylabel('Acc (m/s^2)'); axs2[2].set_xlabel('Time (s)'); axs2[2].grid(True)

# Autosave Cartesian Space
filename_cart = "pizza_cartesianspace.png"
plt.savefig(filename_cart)
print(f"Saved Cartesian Space plot to {filename_cart}")
plt.show()

# 3D Path Check
fig3 = go.Figure()
fig3.add_trace(go.Scatter3d(x=traj_x[:,0], y=traj_x[:,1], z=traj_x[:,2],
    mode='lines', line=dict(color='blue', width=4), name='Continuous Path'))
fig3.add_trace(go.Scatter3d(x=cartesian_list[:,0], y=cartesian_list[:,1], z=cartesian_list[:,2],
    mode='markers', marker=dict(size=4, color='red'), name='Via Points'))
fig3.update_layout(title="Resulting 3D Pizza Trajectory", scene=dict(aspectmode='data'))

# Autosave 3D Path
filename_3d = "pizza_trajectory_3d.html"
fig3.write_html(filename_3d)
print(f"Saved 3D Path plot to {filename_3d}")
fig3.show()