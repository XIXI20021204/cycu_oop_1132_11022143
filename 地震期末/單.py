    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    # --- 1. Load Earthquake Ground Acceleration Data ---
    # Make sure 'Kobe.txt' is in the same directory as this script, or provide the full path.
    file_path = r'C:\Users\a0965\OneDrive\文件\GitHub\cycu_oop_1132_11022143\地震期末\Kobe.txt'
    try:
        df_ground_accel = pd.read_csv(file_path, sep='\s+', header=None, skiprows=1, names=['Time (s)', 'Acceleration (g)'])
        g = 9.81  # Acceleration due to gravity in m/s^2
        df_ground_accel['Acceleration (m/s²)'] = df_ground_accel['Acceleration (g)'] * g
        time_series = df_ground_accel['Time (s)'].values
        ground_accel = df_ground_accel['Acceleration (m/s²)'].values
        dt = time_series[1] - time_series[0]
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please make sure it's in the correct directory.")
        exit()

    # --- 2. Define Main Structure Parameters (for a single floor) ---
    ms = 84600000  # KG (mass of the main structure)
    omega_ns = 0.9174  # rad/s (natural frequency of the main structure)
    zeta_s = 0.01  # (damping ratio of the main structure)

    # Derive physical parameters for the SDOF system
    ks = ms * (omega_ns**2)
    cs = 2 * zeta_s * ms * omega_ns

    print(f"--- Analyzing Single Floor Structure (No TMD) ---")
    print(f"Derived Main Structure Parameters:")
    print(f"   Mass (m): {ms:.2f} KG")
    print(f"   Stiffness (k): {ks:.2f} N/m")
    print(f"   Damping Coefficient (c): {cs:.2f} Ns/m")
    print(f"   Natural Frequency (omega_n): {omega_ns:.4f} rad/s")
    print(f"   Damping Ratio (zeta): {zeta_s:.4f}")

    # --- 3. Numerical Integration (Newmark-Beta Method) for SDOF system ---
    gamma = 0.5
    beta = 0.25

    num_steps = len(time_series)
    # response array stores [u, v, a] for the single DOF
    response = np.zeros((num_steps, 3))

    # Initial acceleration calculation for SDOF: m*a_0 + c*v_0 + k*u_0 = -m*a_g0
    # Assuming initial displacement u_0 = 0, initial velocity v_0 = 0
    # So, m*a_0 = -m*a_g0  => a_0 = -a_g0
    initial_accel = -ground_accel[0]

    response[0, 2] = initial_accel # Initial relative acceleration

    # Pre-calculate K_eff for SDOF
    # K_eff = k + (gamma / (beta * dt)) * c + (1 / (beta * dt**2)) * m
    K_eff_sdof = ks + (gamma / (beta * dt)) * cs + (1 / (beta * dt**2)) * ms

    for i in range(num_steps - 1):
        # Current state
        u_i = response[i, 0]
        v_i = response[i, 1]
        a_i = response[i, 2] # Relative acceleration

        # External force at t+dt for SDOF: P_t_plus_dt = -m * ground_accel[i+1]
        P_t_plus_dt_sdof = -ms * ground_accel[i+1]

        # Effective load (RHS) for SDOF
        RHS_sdof = P_t_plus_dt_sdof + \
                ms * ((1/(beta*dt**2))*u_i + (1/(beta*dt))*v_i + (1/(2*beta) - 1)*a_i) + \
                cs * ((gamma/(beta*dt))*u_i + (gamma/beta - 1)*v_i + (gamma/2 - beta)*dt*a_i)

        # Solve for displacement at t+dt
        u_t_plus_dt_sdof = RHS_sdof / K_eff_sdof

        # Update acceleration and velocity at t+dt
        a_t_plus_dt_sdof = (1/(beta*dt**2)) * (u_t_plus_dt_sdof - u_i) - (1/(beta*dt)) * v_i - (1/(2*beta) - 1) * a_i
        v_t_plus_dt_sdof = v_i + (1 - gamma) * dt * a_i + gamma * dt * a_t_plus_dt_sdof

        # Store results
        response[i+1, 0] = u_t_plus_dt_sdof # Relative displacement
        response[i+1, 1] = v_t_plus_dt_sdof # Relative velocity
        response[i+1, 2] = a_t_plus_dt_sdof # Relative acceleration

    # --- 4. Prepare Results DataFrame ---
    results_df_sdof = pd.DataFrame({
        'Time (s)': time_series,
        'Ground Accel (m/s²)': ground_accel,
        'Floor Disp (m)': response[:, 0],           # Relative displacement to ground
        'Floor Vel (m/s)': response[:, 1],           # Relative velocity to ground
        'Floor Accel (m/s²)': response[:, 2] + ground_accel # Absolute acceleration
    })

    print(f"\n--- First 5 rows of Calculated Responses for Single Floor ---")
    print(results_df_sdof.head())

    # Create output directory if it doesn't exist
    output_dir = 'output_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save results to CSV
    output_csv_filename_sdof = "Single_Floor_No_TMD_simulation_results.csv"
    output_csv_path_sdof = os.path.join(output_dir, output_csv_filename_sdof)
    try:
        results_df_sdof.to_csv(output_csv_path_sdof, index=False, encoding='utf-8')
        print(f"Calculation results for Single Floor (No TMD) successfully saved to: {output_csv_path_sdof}")
    except Exception as e:
        print(f"Error saving file for Single Floor (No TMD): {e}")

    # --- Calculate Displacement Statistics ---
    floor_disp = results_df_sdof['Floor Disp (m)']
    average_disp = np.mean(floor_disp)
    rms_disp = np.sqrt(np.mean(floor_disp**2))
    peak_disp = np.max(np.abs(floor_disp)) # Use absolute for peak value

    print("\n--- Displacement Statistics (Single Floor, No TMD) ---")
    print(f"Average Displacement: {average_disp:.6f} m")
    print(f"RMS Displacement: {rms_disp:.6f} m")
    print(f"Peak Displacement: {peak_disp:.6f} m")


    # --- 5. Plotting Displacement History ---
    plt.figure(figsize=(12, 6))
    plt.plot(time_series, results_df_sdof['Floor Disp (m)'], label='Single Floor Displacement')
    plt.title('Single Floor Displacement History (No Damper)')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.grid(True)
    plt.legend()
    plot_path_sdof = os.path.join(output_dir, 'Single_Floor_No_TMD_Displacement_History.png')
    plt.savefig(plot_path_sdof)
    print(f"Single Floor Displacement History plot saved to: {plot_path_sdof}")
    plt.show()

    print("\n--- Single Floor Simulation Completed ---")