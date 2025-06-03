import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load Earthquake Ground Acceleration Data ---
# Modify file path to your specific path
file_path = r'C:\Users\User\Documents\GitHub\cycu_oop_1132_11022143\地震期末\Kobe.txt'
try:
    # Read data, skipping the first row (header) and specifying column names
    df_ground_accel = pd.read_csv(file_path, sep='\s+', header=None, skiprows=1, names=['Time (s)', 'Acceleration (g)'])
    # Convert acceleration from 'g' to m/s^2 (assuming 1g = 9.81 m/s^2)
    g = 9.81  # Acceleration due to gravity in m/s^2
    df_ground_accel['Acceleration (m/s²)'] = df_ground_accel['Acceleration (g)'] * g
    time_series = df_ground_accel['Time (s)'].values # Time series
    ground_accel = df_ground_accel['Acceleration (m/s²)'].values # Ground surface acceleration
    dt = time_series[1] - time_series[0] # Time step
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please make sure it's in the correct directory.")
    exit()

# --- 2. Define System Parameters ---
# Main Structure (Single Floor) Parameters
ms = 84600000  # KG (mass of the main structure)
# Assuming "自然頻率比0.9174rad/s" means omega_ns = 0.9174 rad/s
omega_ns = 0.9174  # rad/s (natural frequency of the main structure)
zeta_s = 0.01  # (damping ratio of the main structure)

# Tuned Mass Damper (TMD) Parameters
mu = 0.03  # md/ms (damper mass ratio)
alpha = 0.9592  # omega_nd/omega_ns (tuning frequency ratio)
zeta_d = 0.0857  # (damper damping ratio)

# --- 3. Derive Physical Parameters ---
# Main Structure parameters
ks = ms * (omega_ns**2)  # Stiffness of the main structure
cs = 2 * zeta_s * ms * omega_ns  # Damping coefficient of the main structure

# TMD parameters
md = mu * ms  # Mass of the damper
omega_nd = alpha * omega_ns  # Natural frequency of the damper
kd = md * (omega_nd**2)  # Stiffness of the damper
cd = 2 * zeta_d * md * omega_nd  # Damping coefficient of the damper

print(f"Derived Main Structure Parameters:")
print(f"  Stiffness (ks): {ks:.2f} N/m")
print(f"  Damping Coefficient (cs): {cs:.2f} Ns/m")
print(f"\nDerived TMD Parameters:")
print(f"  Damper Mass (md): {md:.2f} KG")
print(f"  Damper Natural Frequency (omega_nd): {omega_nd:.4f} rad/s")
print(f"  Damper Stiffness (kd): {kd:.2f} N/m")
print(f"  Damper Damping Coefficient (cd): {cd:.2f} Ns/m")

# --- 4. Establish 2DOF System Matrices ---
# Degrees of Freedom:
# x[0] = us (displacement of main structure relative to ground)
# x[1] = ud (displacement of damper relative to main structure)

M = np.array([[ms, 0],
              [0, md]])

C = np.array([[cs + cd, -cd],
              [-cd, cd]])

K = np.array([[ks + kd, -kd],
              [-kd, kd]])

# Load vector for ground acceleration
# P(t) = -M * 1 * u_double_dot_g(t)
# Where 1 = {1, 0} for us_relative_to_ground and ud_relative_to_structure
load_matrix = np.array([[1], [0]])

# --- 5. Numerical Integration (Newmark-Beta Method) ---
# Newmark-beta parameters (Average Constant Acceleration Method)
gamma = 0.5
beta = 0.25

num_steps = len(time_series)
# Initialize displacement, velocity, and acceleration vectors
# Responses are stored as columns: us, ud_rel, us_dot, ud_rel_dot, us_double_dot, ud_rel_double_dot
response = np.zeros((num_steps, 6))

# Initial conditions (all zero)
us_0 = 0.0
ud_rel_0 = 0.0
us_dot_0 = 0.0
ud_rel_dot_0 = 0.0

# Calculate initial accelerations
# Initial acceleration vector should be (2,1) shape
initial_accel_vec = np.linalg.solve(M, -M @ load_matrix * ground_accel[0])

response[0, 4] = initial_accel_vec[0, 0] # initial us_double_dot (relative)
response[0, 5] = initial_accel_vec[1, 0] # initial ud_rel_double_dot

# Effective stiffness matrix for Newmark-beta
K_eff = K + (gamma / (beta * dt)) * C + (1 / (beta * dt**2)) * M

# Loop through time steps
for i in range(num_steps - 1):
    # Current values, ensure they are (2,1) column vectors
    u_i = response[i, 0:2].reshape(-1, 1) # us, ud_rel
    v_i = response[i, 2:4].reshape(-1, 1) # us_dot, ud_rel_dot
    a_i = response[i, 4:6].reshape(-1, 1) # us_double_dot, ud_rel_double_dot

    # Effective load vector at t+dt
    P_t_plus_dt = -M @ load_matrix * ground_accel[i+1] # Force due to ground acceleration at t+dt, shape (2,1)

    # Calculate RHS_force_terms
    # Ensure all summed terms are (2,1) shape, use @ for matrix multiplication
    RHS_force_terms = P_t_plus_dt + \
                      M @ ((1/(beta*dt**2))*u_i + (1/(beta*dt))*v_i + (1/(2*beta) - 1)*a_i) + \
                      C @ ((gamma/(beta*dt))*u_i + (gamma/beta - 1)*v_i + (gamma/2 - beta)*dt*a_i)

    # Solve for displacement at t+dt, result will be (2,1)
    u_t_plus_dt = np.linalg.solve(K_eff, RHS_force_terms)

    # Update accelerations and velocities at t+dt, result will be (2,1)
    a_t_plus_dt = (1/(beta*dt**2)) * (u_t_plus_dt - u_i) - (1/(beta*dt)) * v_i - (1/(2*beta) - 1) * a_i
    v_t_plus_dt = v_i + (1 - gamma) * dt * a_i + gamma * dt * a_t_plus_dt

    # Store results into the (2,) slice of the response array, need to flatten (2,1) results to (2,)
    response[i+1, 0:2] = u_t_plus_dt.flatten() # us, ud_rel
    response[i+1, 2:4] = v_t_plus_dt.flatten() # us_dot, ud_rel_dot
    response[i+1, 4:6] = a_t_plus_dt.flatten() # us_double_dot, ud_rel_double_dot

# --- 6. Calculate Absolute Responses and Performance Metrics ---
# us: displacement of main structure relative to ground
# ud_rel: displacement of damper relative to main structure

# Absolute displacement of damper
u_d_abs = response[:, 0] + response[:, 1]

# Absolute acceleration of main structure (relative to ground)
# response[:, 4] is already u_double_dot_s (relative to ground, which is absolute for the main structure)
u_double_dot_s_abs = response[:, 4]

# Absolute acceleration of damper (relative to ground)
u_double_dot_d_abs = response[:, 4] + response[:, 5]

# Convert responses to a DataFrame for easier handling
results_df = pd.DataFrame({
    'Time (s)': time_series,
    'Ground Accel (m/s²)': ground_accel,
    'Floor Disp (m)': response[:, 0],
    'Floor Vel (m/s)': response[:, 2],
    'Floor Accel (m/s²)': u_double_dot_s_abs, # This is the absolute acceleration of the floor
    'Damper Rel Disp (m)': response[:, 1], # Damper displacement relative to floor
    'Damper Rel Vel (m/s)': response[:, 3], # Damper velocity relative to floor
    'Damper Rel Accel (m/s²)': response[:, 5], # Damper acceleration relative to floor
    'Damper Abs Disp (m)': u_d_abs, # Damper absolute displacement
    'Damper Abs Accel (m/s²)': u_double_dot_d_abs # Damper absolute acceleration
})

print("\n--- First 5 rows of Calculated Responses ---")
print(results_df.head())

# --- Save results to a CSV file ---
output_csv_path = r'C:\Users\User\Documents\GitHub\cycu_oop_1132_11022143\地震期末\simulation_results.csv'
try:
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"\nCalculation results successfully saved to: {output_csv_path}")
except Exception as e:
    print(f"\nError saving file: {e}")

# --- 7. Plotting Results ---
plt.figure(figsize=(15, 10))

# Plot Floor Absolute Acceleration
plt.subplot(3, 1, 1)
plt.plot(results_df['Time (s)'], results_df['Floor Accel (m/s²)'], label='Floor Absolute Acceleration')
plt.plot(results_df['Time (s)'], results_df['Ground Accel (m/s²)'], linestyle='--', alpha=0.7, label='Ground Acceleration Input')
plt.title('Floor Absolute Acceleration Response')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.grid(True)
plt.legend()

# Plot Floor Displacement
plt.subplot(3, 1, 2)
plt.plot(results_df['Time (s)'], results_df['Floor Disp (m)'], label='Floor Displacement (relative to ground)')
plt.title('Floor Displacement Response')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.grid(True)
plt.legend()

# Plot Damper Absolute Acceleration
plt.subplot(3, 1, 3)
plt.plot(results_df['Time (s)'], results_df['Damper Abs Accel (m/s²)'], label='Damper Absolute Acceleration')
plt.plot(results_df['Time (s)'], results_df['Damper Rel Accel (m/s²)'], linestyle=':', alpha=0.7, label='Damper Relative Acceleration (relative to floor)')
plt.title('Damper Acceleration Response')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# --- 8. Basic Performance Metrics (Optional) ---
max_ground_accel = np.max(np.abs(ground_accel))
max_floor_accel = np.max(np.abs(results_df['Floor Accel (m/s²)']))
max_floor_disp = np.max(np.abs(results_df['Floor Disp (m)']))
max_damper_abs_accel = np.max(np.abs(results_df['Damper Abs Accel (m/s²)']))
max_damper_rel_disp = np.max(np.abs(results_df['Damper Rel Disp (m)']))

print(f"\n--- Response Summary ---")
print(f"Max Ground Acceleration: {max_ground_accel:.4f} m/s²")
print(f"Max Floor Absolute Acceleration: {max_floor_accel:.4f} m/s²")
print(f"Max Floor Displacement (relative to ground): {max_floor_disp:.4f} m")
print(f"Max Damper Absolute Acceleration: {max_damper_abs_accel:.4f} m/s²")
print(f"Max Damper Relative Displacement (relative to floor): {max_damper_rel_disp:.4f} m")

# To calculate without TMD, you would need to run a separate SDOF analysis.
# This code currently only calculates with TMD.