import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. Load Earthquake Ground Acceleration Data ---
# Make sure 'Kobe.txt' is in the same directory as this script, or provide the full path.
# IMPORTANT: Please verify this file path is correct on your system!
file_path = r'C:\Users\a0965\OneDrive\文件\GitHub\cycu_oop_1132_11022143\地震期末\Kobe.txt'
try:
    # Read data, skipping the first row (header) and specifying column names
    df_ground_accel = pd.read_csv(file_path, sep='\s+', header=None, skiprows=1, names=['Time (s)', 'Acceleration (g)'])
    # Convert acceleration from 'g' to m/s^2 (assuming 1g = 9.81 m/s^2)
    g = 9.81  # Acceleration due to gravity in m/s^2
    df_ground_accel['Acceleration (m/s²)'] = df_ground_accel['Acceleration (g)'] * g
    time_series = df_ground_accel['Time (s)'].values  # Time series
    ground_accel = df_ground_accel['Acceleration (m/s²)'].values  # Ground surface acceleration
    dt = time_series[1] - time_series[0]  # Time step
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please make sure it's in the correct directory.")
    exit()

# --- 2. Define Main Structure Parameters (Constant for both SDOF and 2DOF analyses) ---
ms = 84600000  # KG (mass of the main structure)
omega_ns = 0.9174  # rad/s (natural frequency of the main structure)
zeta_s = 0.01  # (damping ratio of the main structure)

# Create output directory for plots and CSVs
output_dir = 'output_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Dictionary to store results for comparison plots and further analysis
all_results_for_comparison_plots = {}
all_results_dfs = {} # To store full dataframes for individual CSVs and summaries

# ==============================================================================
---
## PART 1: Simulate Single Floor Structure (No TMD)
---
# ==============================================================================
print("\n" + "="*50)
print("--- Analyzing Single Floor Structure (No TMD) ---")
print("="*50)

# Derive physical parameters for the SDOF system
ks_sdof = ms * (omega_ns**2)
cs_sdof = 2 * zeta_s * ms * omega_ns

print(f"Derived Main Structure Parameters:")
print(f"   Mass (m): {ms:.2f} KG")
print(f"   Stiffness (k): {ks_sdof:.2f} N/m")
print(f"   Damping Coefficient (c): {cs_sdof:.2f} Ns/m")
print(f"   Natural Frequency (omega_n): {omega_ns:.4f} rad/s")
print(f"   Damping Ratio (zeta): {zeta_s:.4f}")

# --- SDOF Numerical Integration (Newmark-Beta Method) ---
gamma = 0.5
beta = 0.25

num_steps = len(time_series)
# response array stores [u, v, a] for the single DOF
response_sdof = np.zeros((num_steps, 3))

# Initial acceleration calculation for SDOF: m*a_0 + c*v_0 + k*u_0 = -m*a_g0
# Assuming initial displacement u_0 = 0, initial velocity v_0 = 0
initial_accel_sdof = -ground_accel[0]

response_sdof[0, 2] = initial_accel_sdof # Initial relative acceleration

# Pre-calculate K_eff for SDOF
K_eff_sdof = ks_sdof + (gamma / (beta * dt)) * cs_sdof + (1 / (beta * dt**2)) * ms

for i in range(num_steps - 1):
    # Current state
    u_i_sdof = response_sdof[i, 0]
    v_i_sdof = response_sdof[i, 1]
    a_i_sdof = response_sdof[i, 2] # Relative acceleration

    # External force at t+dt for SDOF: P_t_plus_dt = -m * ground_accel[i+1]
    P_t_plus_dt_sdof = -ms * ground_accel[i+1]

    # Effective load (RHS) for SDOF
    RHS_sdof = P_t_plus_dt_sdof + \
               ms * ((1/(beta*dt**2))*u_i_sdof + (1/(beta*dt))*v_i_sdof + (1/(2*beta) - 1)*a_i_sdof) + \
               cs_sdof * ((gamma/(beta*dt))*u_i_sdof + (gamma/beta - 1)*v_i_sdof + (gamma/2 - beta)*dt*a_i_sdof)

    # Solve for displacement at t+dt
    u_t_plus_dt_sdof = RHS_sdof / K_eff_sdof

    # Update acceleration and velocity at t+dt
    a_t_plus_dt_sdof = (1/(beta*dt**2)) * (u_t_plus_dt_sdof - u_i_sdof) - (1/(beta*dt)) * v_i_sdof - (1/(2*beta) - 1) * a_i_sdof
    v_t_plus_dt_sdof = v_i_sdof + (1 - gamma) * dt * a_i_sdof + gamma * dt * a_t_plus_dt_sdof

    # Store results
    response_sdof[i+1, 0] = u_t_plus_dt_sdof # Relative displacement
    response_sdof[i+1, 1] = v_t_plus_dt_sdof # Relative velocity
    response_sdof[i+1, 2] = a_t_plus_dt_sdof # Relative acceleration

# --- Prepare SDOF Results DataFrame ---
results_df_sdof = pd.DataFrame({
    'Time (s)': time_series,
    'Ground Accel (m/s²)': ground_accel,
    'Floor Disp (m)': response_sdof[:, 0],  # Relative displacement to ground
    'Floor Vel (m/s)': response_sdof[:, 1],  # Relative velocity to ground
    'Floor Accel (m/s²)': response_sdof[:, 2] + ground_accel # Absolute acceleration
})

print(f"\n--- Single Floor (No TMD) Calculated Responses (First 5 rows) ---")
print(results_df_sdof.head())

# Save single floor results to CSV
output_csv_filename_sdof = "Single_Floor_No_TMD_simulation_results.csv"
output_csv_path_sdof = os.path.join(output_dir, output_csv_filename_sdof)
try:
    results_df_sdof.to_csv(output_csv_path_sdof, index=False, encoding='utf-8')
    print(f"Calculation results for Single Floor (No TMD) successfully saved to: {output_csv_path_sdof}")
except Exception as e:
    print(f"Error saving file for Single Floor (No TMD): {e}")

# Store single floor displacement results for comparison
all_results_for_comparison_plots['Single Floor (No TMD)'] = {
    'Floor Disp': results_df_sdof['Floor Disp (m)']
}
all_results_dfs['Single Floor (No TMD)'] = results_df_sdof # Store full dataframe

# Calculate Displacement Statistics for SDOF (Base values for relative error)
floor_disp_sdof = results_df_sdof['Floor Disp (m)']
mean_disp_sdof_base = np.mean(floor_disp_sdof)
rms_disp_sdof_base = np.sqrt(np.mean(floor_disp_sdof**2))
peak_disp_sdof_base = np.max(np.abs(floor_disp_sdof)) # Peak value (absolute)

print("\n--- Single Floor (No TMD) Displacement Statistics (Base Values) ---")
print(f"Average Displacement: {mean_disp_sdof_base:.6f} m")
print(f"RMS Displacement: {rms_disp_sdof_base:.6f} m")
print(f"Peak Displacement: {peak_disp_sdof_base:.6f} m")

print("\n--- Single Floor Simulation Completed ---")

# ==============================================================================
---
## PART 2: Simulate Main Structure with TMD Configurations
---
# ==============================================================================
print("\n\n" + "="*50)
print("--- Analyzing Main Structure with TMD ---")
print("="*50)

tmd_configurations = [
    {"label": "TMD_Config_1 (mu=0.03, alpha=0.9592, zeta_d=0.0857)", "mu": 0.03, "alpha": 0.9592, "zeta_d": 0.0857},
    {"label": "TMD_Config_2 (mu=0.1, alpha=0.8789, zeta_d=0.1527)", "mu": 0.1, "alpha": 0.8789, "zeta_d": 0.1527},
    {"label": "TMD_Config_3 (mu=0.2, alpha=0.7815, zeta_d=0.2098)", "mu": 0.2, "alpha": 0.7815, "zeta_d": 0.2098},
]

# --- Loop through each TMD configuration for calculation ---
for config_num, tmd_config in enumerate(tmd_configurations):
    print(f"\n--- Analyzing {tmd_config['label']} ---")

    mu = tmd_config['mu']
    alpha = tmd_config['alpha']
    zeta_d = tmd_config['zeta_d']

    # --- Derive Physical Parameters ---
    # Main structure parameters (using ms, omega_ns, zeta_s from SDOF)
    ks = ms * (omega_ns**2)
    cs = 2 * zeta_s * ms * omega_ns

    # TMD parameters
    md = mu * ms
    omega_nd = alpha * omega_ns
    kd = md * (omega_nd**2)
    cd = 2 * zeta_d * md * omega_nd

    print(f"Derived Main Structure Parameters (Constant):")
    print(f"   Stiffness (ks): {ks:.2f} N/m")
    print(f"   Damping Coefficient (cs): {cs:.2f} Ns/m")
    print(f"\nDerived TMD Parameters for {tmd_config['label']}:")
    print(f"   Damper Mass (md): {md:.2f} KG")
    print(f"   Damper Natural Frequency (omega_nd): {omega_nd:.4f} rad/s")
    print(f"   Damper Stiffness (kd): {kd:.2f} N/m")
    print(f"   Damper Damping Coefficient (cd): {cd:.2f} Ns/m")

    # --- Establish 2DOF System Matrices ---
    M = np.array([[ms, 0],
                  [0, md]])

    K = np.array([[ks + kd, -kd],
                  [-kd, kd]])

    C = np.array([[cs + cd, -cd],
                  [-cd, cd]])

    # Load matrix (influence vector for ground acceleration)
    # For a relative displacement formulation (u_s, u_d_rel_s), the force vector is -M @ [1, 0]^T * accel_g
    load_matrix = np.array([[1], [0]])

    # --- 2DOF Numerical Integration (Newmark-Beta Method) ---
    # gamma and beta values are the same as SDOF
    response_2dof = np.zeros((num_steps, 6))
    # response_2dof array stores [u_s, u_d_rel_s, v_s, v_d_rel_s, a_s_rel_g, a_d_rel_s]

    # Initial acceleration calculation: M a_0 + C v_0 + K u_0 = P_0
    # Assuming initial displacements and velocities are zero (u_0 = 0, v_0 = 0).
    initial_accel_vec_2dof = np.linalg.solve(M, -M @ load_matrix * ground_accel[0])

    response_2dof[0, 4] = initial_accel_vec_2dof[0, 0] # Initial relative acceleration of main structure (a_s_rel_g)
    response_2dof[0, 5] = initial_accel_vec_2dof[1, 0] # Initial relative acceleration of TMD w.r.t main structure (a_d_rel_s)

    # Pre-calculate K_eff (constant throughout integration)
    K_eff_2dof = K + (gamma / (beta * dt)) * C + (1 / (beta * dt**2)) * M

    for i in range(num_steps - 1):
        # Current state vectors (reshaped to column vectors for matrix operations)
        u_i = response_2dof[i, 0:2].reshape(-1, 1) # Current relative displacements
        v_i = response_2dof[i, 2:4].reshape(-1, 1) # Current relative velocities
        a_i = response_2dof[i, 4:6].reshape(-1, 1) # Current relative accelerations

        # External force vector at time t+dt
        P_t_plus_dt = -M @ load_matrix * ground_accel[i+1]

        # Effective load vector (RHS) for Newmark-Beta method
        RHS_force_terms = P_t_plus_dt + \
                          M @ ((1/(beta*dt**2))*u_i + (1/(beta*dt))*v_i + (1/(2*beta) - 1)*a_i) + \
                          C @ ((gamma/(beta*dt))*u_i + (gamma/beta - 1)*v_i + (gamma/2 - beta)*dt*a_i)

        # Solve for displacement at t+dt (u_t_plus_dt)
        u_t_plus_dt = np.linalg.solve(K_eff_2dof, RHS_force_terms)

        # Update acceleration and velocity at t+dt
        a_t_plus_dt = (1/(beta*dt**2)) * (u_t_plus_dt - u_i) - (1/(beta*dt)) * v_i - (1/(2*beta) - 1) * a_i
        v_t_plus_dt = v_i + (1 - gamma) * dt * a_i + gamma * dt * a_t_plus_dt

        # Store results for the next time step (flatten to store into 1D slices of response array)
        response_2dof[i+1, 0:2] = u_t_plus_dt.flatten() # [u_s_rel_g, u_d_rel_s]
        response_2dof[i+1, 2:4] = v_t_plus_dt.flatten() # [v_s_rel_g, v_d_rel_s]
        response_2dof[i+1, 4:6] = a_t_plus_dt.flatten() # [a_s_rel_g, a_d_rel_s]

    # --- Calculate Absolute Responses for output and analysis ---
    # Absolute displacement of TMD = (main structure relative to ground) + (TMD relative to main structure)
    u_d_abs = response_2dof[:, 0] + response_2dof[:, 1]
    # Absolute acceleration of main structure = (main structure relative acceleration) + (ground acceleration)
    u_double_dot_s_abs = response_2dof[:, 4] + ground_accel
    # Absolute acceleration of TMD = (TMD relative acceleration to ground) + (ground acceleration)
    u_double_dot_d_abs = (response_2dof[:, 4] + response_2dof[:, 5]) + ground_accel

    # Create a DataFrame to store all calculated responses for this configuration
    results_df_tmd = pd.DataFrame({
        'Time (s)': time_series,
        'Ground Accel (m/s²)': ground_accel,
        'Floor Disp (m)': response_2dof[:, 0],          # Main structure relative displacement to ground
        'Floor Vel (m/s)': response_2dof[:, 2],          # Main structure relative velocity to ground
        'Floor Accel (m/s²)': u_double_dot_s_abs,      # Main structure absolute acceleration
        'Damper Rel Disp (m)': response_2dof[:, 1],       # TMD relative displacement to main structure
        'Damper Rel Vel (m/s)': response_2dof[:, 3],      # TMD relative velocity to main structure
        'Damper Rel Accel (m/s²)': response_2dof[:, 5],  # TMD relative acceleration to main structure
        'Damper Abs Disp (m)': u_d_abs,                 # TMD absolute displacement
        'Damper Abs Accel (m/s²)': u_double_dot_d_abs # TMD absolute acceleration
    })

    print(f"\n--- {tmd_config['label']} Calculated Responses (First 5 rows) ---")
    print(results_df_tmd.head())

    # Store full DataFrame for individual CSV saving and summary later
    all_results_dfs[tmd_config['label']] = results_df_tmd

    # Store relevant data for comparison plots (only Floor Disp is needed for the requested plot)
    all_results_for_comparison_plots[tmd_config['label']] = {
        'Floor Disp': results_df_tmd['Floor Disp (m)']
    }

    # --- Save individual results to a CSV file ---
    # Clean up label for filename by replacing spaces and parentheses
    output_csv_filename_tmd = f"{tmd_config['label'].replace(' ', '_').replace('(', '').replace(')', '')}_simulation_results.csv"
    output_csv_path_tmd = os.path.join(output_dir, output_csv_filename_tmd)
    try:
        results_df_tmd.to_csv(output_csv_path_tmd, index=False, encoding='utf-8')
        print(f"Calculation results for {tmd_config['label']} successfully saved to: {output_csv_path_tmd}")
    except Exception as e:
        print(f"Error saving file for {tmd_config['label']}: {e}")

    # --- Basic Performance Metrics for individual config ---
    max_ground_accel = np.max(np.abs(ground_accel))
    max_floor_accel = np.max(np.abs(results_df_tmd['Floor Accel (m/s²)']))
    max_floor_disp = np.max(np.abs(results_df_tmd['Floor Disp (m)']))
    max_damper_abs_accel = np.max(np.abs(results_df_tmd['Damper Abs Accel (m/s²)']))
    max_damper_rel_disp = np.max(np.abs(results_df_tmd['Damper Rel Disp (m)']))

    print(f"\n--- {tmd_config['label']} Response Summary ---")
    print(f"Max Ground Acceleration: {max_ground_accel:.4f} m/s²")
    print(f"Max Floor Absolute Acceleration: {max_floor_accel:.4f} m/s²")
    print(f"Max Floor Displacement (relative to ground): {max_floor_disp:.4f} m")
    print(f"Max Damper Absolute Acceleration: {max_damper_abs_accel:.4f} m/s²")
    print(f"Max Damper Relative Displacement (relative to floor): {max_damper_rel_disp:.4f} m")

print("\n--- All individual simulations completed. Generating comparison plots ---")

# ==============================================================================
---
## FINAL: Plotting All Comparison Results & Relative Error Analysis
---
# ==============================================================================
print("\n" + "="*50)
print("--- Generating All Structure Main Structure Displacement Comparison Plot ---")
print("="*50)

plt.figure(figsize=(14, 8)) # Adjust figure size for better readability

# Plot all results
for label, data in all_results_for_comparison_plots.items():
    plt.plot(time_series, data['Floor Disp'], label=label, linewidth=1.5) # Slightly thicker lines

plt.title('Main Structure Displacement Response Comparison (Relative to Ground)\n'
          'Single Floor (No TMD) vs. Different TMD Configurations', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Displacement (m)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right', fontsize=10) # Adjust legend position and font size
plt.tight_layout() # Automatically adjust layout to prevent overlapping
comparison_plot_path_all_disp = os.path.join(output_dir, 'All_Main_Structure_Displacement_Comparison.png')
plt.savefig(comparison_plot_path_all_disp)
print(f"All Main Structure Displacement Comparison Plot successfully saved to: {comparison_plot_path_all_disp}")
plt.show()

# --- Calculate and print Mean, RMS, Peak values for Floor Displacement with Relative Error ---
print("\n" + "="*80)
print("--- Performance Metrics for Main Structure Displacement (Relative to Ground) with Relative Error ---")
print("="*80)
print("{:<45} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
    "Structure Configuration", "Mean (m)", "RMS (m)", "Peak (m)",
    "Rel. Error (Mean) (%)", "Rel. Error (RMS) (%)", "Rel. Error (Peak) (%)"
))
print("-" * 125) # Extend separator line

for label, df in all_results_dfs.items():
    floor_disp = df['Floor Disp (m)']
    mean_disp = np.mean(floor_disp)
    rms_disp = np.sqrt(np.mean(floor_disp**2))
    peak_disp = np.max(np.abs(floor_disp)) # Use absolute value for peak

    # Calculate relative error only for TMD configurations
    if label == 'Single Floor (No TMD)':
        rel_error_mean = "N/A"
        rel_error_rms = "N/A"
        rel_error_peak = "N/A"
    else:
        # Relative error = ((Value_TMD - Value_NoTMD) / Value_NoTMD) * 100%
        # Or, to show reduction: ((Value_NoTMD - Value_TMD) / Value_NoTMD) * 100%
        # Here we calculate reduction percentage:
        rel_error_mean = ((mean_disp_sdof_base - mean_disp) / mean_disp_sdof_base) * 100
        rel_error_rms = ((rms_disp_sdof_base - rms_disp) / rms_disp_sdof_base) * 100
        rel_error_peak = ((peak_disp_sdof_base - peak_disp) / peak_disp_sdof_base) * 100
        rel_error_mean = f"{rel_error_mean:.2f}"
        rel_error_rms = f"{rel_error_rms:.2f}"
        rel_error_peak = f"{rel_error_peak:.2f}"


    print(f"{label:<45} {mean_disp:<15.6f} {rms_disp:<15.6f} {peak_disp:<15.6f} "
          f"{rel_error_mean:<20} {rel_error_rms:<20} {rel_error_peak:<15}")

print("\n--- All simulations and calculations completed ---")