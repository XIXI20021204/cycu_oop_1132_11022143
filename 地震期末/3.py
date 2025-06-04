import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. Load Earthquake Ground Acceleration Data ---
# Make sure 'Kobe.txt' is in the same directory as this script, or provide the full path.
file_path = r'C:\Users\a0965\OneDrive\文件\GitHub\cycu_oop_1132_11022143\地震期末\Kobe.txt'
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

# --- 2. Define Main Structure Parameters ---
ms = 84600000  # KG (mass of the main structure)
omega_ns = 0.9174  # rad/s (natural frequency of the main structure)
zeta_s = 0.01  # (damping ratio of the main structure)

# --- 2.1 Define TMD Configurations ---
tmd_configurations = [
    {"label": "TMD_Config_1 (mu=0.03, alpha=0.9592, zeta_d=0.0857)", "mu": 0.03, "alpha": 0.9592, "zeta_d": 0.0857},
    {"label": "TMD_Config_2 (mu=0.1, alpha=0.8789, zeta_d=0.1527)", "mu": 0.1, "alpha": 0.8789, "zeta_d": 0.1527},
    {"label": "TMD_Config_3 (mu=0.2, alpha=0.7815, zeta_d=0.2098)", "mu": 0.2, "alpha": 0.7815, "zeta_d": 0.2098},
]

# Create output directory for plots and CSVs
output_dir = 'output_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Dictionary to store results for comparison plots and further analysis
all_results_for_comparison = {}
all_results_dfs = {} # To store full dataframes for individual CSVs and summaries

# --- Loop through each TMD configuration for calculation ---
for config_num, tmd_config in enumerate(tmd_configurations):
    print(f"\n--- Analyzing {tmd_config['label']} ---")

    mu = tmd_config['mu']
    alpha = tmd_config['alpha']
    zeta_d = tmd_config['zeta_d']

    # --- 3. Derive Physical Parameters ---
    ks = ms * (omega_ns**2)
    cs = 2 * zeta_s * ms * omega_ns

    md = mu * ms
    omega_nd = alpha * omega_ns
    kd = md * (omega_nd**2)
    cd = 2 * zeta_d * md * omega_nd

    print(f"Derived Main Structure Parameters (Constant):")
    print(f"  Stiffness (ks): {ks:.2f} N/m")
    print(f"  Damping Coefficient (cs): {cs:.2f} Ns/m")
    print(f"\nDerived TMD Parameters for {tmd_config['label']}:")
    print(f"  Damper Mass (md): {md:.2f} KG")
    print(f"  Damper Natural Frequency (omega_nd): {omega_nd:.4f} rad/s")
    print(f"  Damper Stiffness (kd): {kd:.2f} N/m")
    print(f"  Damper Damping Coefficient (cd): {cd:.2f} Ns/m")

    # --- 4. Establish 2DOF System Matrices ---
    # M = [[ms, 0], [0, md]]
    M = np.array([[ms, 0],
                  [0, md]])

    # K = [[ks + kd, -kd], [-kd, kd]]
    K = np.array([[ks + kd, -kd],
                  [-kd, kd]])

    # C = [[cs + cd, -cd], [-cd, cd]]
    C = np.array([[cs + cd, -cd],
                  [-cd, cd]])

    # Load matrix (influence vector for ground acceleration)
    # This matrix determines how the ground acceleration force is distributed.
    # For a relative displacement formulation (u_s, u_d_rel_s), the force vector is -M @ [1, 0]^T * accel_g
    # meaning ground acceleration primarily excites the main structure.
    load_matrix = np.array([[1], [0]])

    # --- 5. Numerical Integration (Newmark-Beta Method) ---
    gamma = 0.5
    beta = 0.25

    num_steps = len(time_series)
    # response array stores [u_s, u_d_rel_s, v_s, v_d_rel_s, a_s_rel_g, a_d_rel_s]
    # where u_s is main structure displacement relative to ground,
    # and u_d_rel_s is TMD displacement relative to main structure.
    response = np.zeros((num_steps, 6))

    # Initial acceleration calculation: M a_0 + C v_0 + K u_0 = P_0
    # Assuming initial displacements and velocities are zero (u_0 = 0, v_0 = 0).
    # P_0 = -M @ load_matrix * ground_accel[0]
    initial_accel_vec = np.linalg.solve(M, -M @ load_matrix * ground_accel[0])

    response[0, 4] = initial_accel_vec[0, 0] # Initial relative acceleration of main structure (a_s_rel_g)
    response[0, 5] = initial_accel_vec[1, 0] # Initial relative acceleration of TMD w.r.t main structure (a_d_rel_s)

    # Pre-calculate K_eff as it's constant throughout the integration
    K_eff = K + (gamma / (beta * dt)) * C + (1 / (beta * dt**2)) * M

    for i in range(num_steps - 1):
        # Current state vectors (reshaped to column vectors for matrix operations)
        u_i = response[i, 0:2].reshape(-1, 1) # Current relative displacements
        v_i = response[i, 2:4].reshape(-1, 1) # Current relative velocities
        a_i = response[i, 4:6].reshape(-1, 1) # Current relative accelerations

        # External force vector at time t+dt
        P_t_plus_dt = -M @ load_matrix * ground_accel[i+1]

        # Effective load vector (RHS) for Newmark-Beta method
        RHS_force_terms = P_t_plus_dt + \
                          M @ ((1/(beta*dt**2))*u_i + (1/(beta*dt))*v_i + (1/(2*beta) - 1)*a_i) + \
                          C @ ((gamma/(beta*dt))*u_i + (gamma/beta - 1)*v_i + (gamma/2 - beta)*dt*a_i)

        # Solve for displacement at t+dt (u_t_plus_dt)
        u_t_plus_dt = np.linalg.solve(K_eff, RHS_force_terms)

        # Update acceleration and velocity at t+dt using Newmark-Beta formulas
        a_t_plus_dt = (1/(beta*dt**2)) * (u_t_plus_dt - u_i) - (1/(beta*dt)) * v_i - (1/(2*beta) - 1) * a_i
        v_t_plus_dt = v_i + (1 - gamma) * dt * a_i + gamma * dt * a_t_plus_dt

        # Store results for the next time step (flatten to store into 1D slices of response array)
        response[i+1, 0:2] = u_t_plus_dt.flatten() # [u_s_rel_g, u_d_rel_s]
        response[i+1, 2:4] = v_t_plus_dt.flatten() # [v_s_rel_g, v_d_rel_s]
        response[i+1, 4:6] = a_t_plus_dt.flatten() # [a_s_rel_g, a_d_rel_s]

    # --- 6. Calculate Absolute Responses for output and analysis ---
    # Absolute displacement of TMD = (main structure relative to ground) + (TMD relative to main structure)
    u_d_abs = response[:, 0] + response[:, 1]
    # Absolute acceleration of main structure = (main structure relative acceleration) + (ground acceleration)
    u_double_dot_s_abs = response[:, 4] + ground_accel
    # Absolute acceleration of TMD = (TMD relative acceleration to ground) + (ground acceleration)
    # TMD relative acceleration to ground = (main structure relative acceleration) + (TMD relative acceleration to main structure)
    u_double_dot_d_abs = (response[:, 4] + response[:, 5]) + ground_accel


    # Create a DataFrame to store all calculated responses for this configuration
    results_df = pd.DataFrame({
        'Time (s)': time_series,
        'Ground Accel (m/s²)': ground_accel,
        'Floor Disp (m)': response[:, 0],          # Main structure relative displacement to ground
        'Floor Vel (m/s)': response[:, 2],          # Main structure relative velocity to ground
        'Floor Accel (m/s²)': u_double_dot_s_abs,   # Main structure absolute acceleration
        'Damper Rel Disp (m)': response[:, 1],      # TMD relative displacement to main structure
        'Damper Rel Vel (m/s)': response[:, 3],      # TMD relative velocity to main structure
        'Damper Rel Accel (m/s²)': response[:, 5],  # TMD relative acceleration to main structure
        'Damper Abs Disp (m)': u_d_abs,             # TMD absolute displacement
        'Damper Abs Accel (m/s²)': u_double_dot_d_abs # TMD absolute acceleration
    })

    print(f"\n--- First 5 rows of Calculated Responses for {tmd_config['label']} ---")
    print(results_df.head())

    # Store full DataFrame for individual CSV saving and summary later
    all_results_dfs[tmd_config['label']] = results_df

    # Store relevant data for comparison plots (only Floor Disp is needed for the requested plot)
    all_results_for_comparison[tmd_config['label']] = {
        'Floor Disp': results_df['Floor Disp (m)']
    }

    # --- Save individual results to a CSV file ---
    # Clean up label for filename by replacing spaces and parentheses
    output_csv_filename = f"{tmd_config['label'].replace(' ', '_').replace('(', '').replace(')', '')}_simulation_results.csv"
    output_csv_path = os.path.join(output_dir, output_csv_filename)
    try:
        results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"Calculation results for {tmd_config['label']} successfully saved to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving file for {tmd_config['label']}: {e}")

    # --- 8. Basic Performance Metrics for individual config ---
    max_ground_accel = np.max(np.abs(ground_accel))
    max_floor_accel = np.max(np.abs(results_df['Floor Accel (m/s²)']))
    max_floor_disp = np.max(np.abs(results_df['Floor Disp (m)']))
    max_damper_abs_accel = np.max(np.abs(results_df['Damper Abs Accel (m/s²)']))
    max_damper_rel_disp = np.max(np.abs(results_df['Damper Rel Disp (m)']))

    print(f"\n--- Response Summary for {tmd_config['label']} ---")
    print(f"Max Ground Acceleration: {max_ground_accel:.4f} m/s²")
    print(f"Max Floor Absolute Acceleration: {max_floor_accel:.4f} m/s²")
    print(f"Max Floor Displacement (relative to ground): {max_floor_disp:.4f} m")
    print(f"Max Damper Absolute Acceleration: {max_damper_abs_accel:.4f} m/s²")
    print(f"Max Damper Relative Displacement (relative to floor): {max_damper_rel_disp:.4f} m")

print("\n--- All individual simulations completed. Generating comparison plots ---")

# --- 7. Plotting Comparison Results (Only Floor Displacement as requested) ---
plt.figure(figsize=(12, 6))
for label, data in all_results_for_comparison.items():
    plt.plot(time_series, data['Floor Disp'], label=label)
plt.title('Main Structure Displacement Response Comparison (Relative to Ground)')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.grid(True)
plt.legend()
comparison_plot_path_disp = os.path.join(output_dir, 'Main_Structure_Displacement_Comparison.png')
plt.savefig(comparison_plot_path_disp)
print(f"Main structure displacement comparison plot saved to: {comparison_plot_path_disp}")
plt.show()

# --- Calculate and print Mean, RMS, Peak values for Floor Displacement ---
print("\n--- Performance Metrics for Main Structure Displacement (Relative to Ground) ---")
print("{:<45} {:<15} {:<15} {:<15}".format("TMD Configuration", "Mean (m)", "RMS (m)", "Peak (m)"))
print("-" * 90)

for label, df in all_results_dfs.items():
    floor_disp = df['Floor Disp (m)']
    mean_disp = np.mean(floor_disp)
    rms_disp = np.sqrt(np.mean(floor_disp**2))
    peak_disp = np.max(np.abs(floor_disp)) # Use absolute value for peak

    print(f"{label:<45} {mean_disp:<15.6f} {rms_disp:<15.6f} {peak_disp:<15.6f}")

print("\n--- All simulations and calculations completed ---")