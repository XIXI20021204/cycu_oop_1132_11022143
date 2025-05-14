import numpy as np
import matplotlib.pyplot as plt

# --- Input Parameters ---
T_NS = 0.57
T_EW = 0.31
Xi = 0.05  # Assuming a more typical damping ratio
dt = 0.02
g = 386.4
gamma_nm = 0.5
beta_nm = 0.25
ground_motion_file = r'C:\Users\a0965\Downloads\Northridge_NS.txt'

def load_ground_motion(file_path, dt_target):
    """Loads ground motion data from a file and ensures consistent time steps."""
    time_gm = []
    accel_g_values = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            first_line_processed = False
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        t_val = float(parts[0])
                        acc_val = float(parts[1])
                        time_gm.append(t_val)
                        accel_g_values.append(acc_val)
                        if not first_line_processed and i == 0 and len(parts) >= 3:
                            file_dt = float(parts[2]) # Attempt to read dt from the first line
                            if not np.isclose(file_dt, dt_target, atol=dt_target * 0.1):
                                print(f"Warning: File dt ({file_dt:.4f} s) differs from target dt ({dt_target:.4f} s).")
                            first_line_processed = True
                    except ValueError:
                        print(f"Warning: Skipping line {i+1} due to invalid data: {line.strip()}")
                elif len(parts) == 1 and first_line_processed:
                    try:
                        # Assume constant dt after the first line
                        current_time = time_gm[-1] + dt_target
                        time_gm.append(current_time)
                        accel_g_values.append(float(parts[0]))
                    except ValueError:
                        print(f"Warning: Skipping line {i+1} due to invalid acceleration data: {line.strip()}")
                elif len(parts) >= 3 and not first_line_processed and i == 0:
                    try:
                        file_dt = float(parts[2])
                        if not np.isclose(file_dt, dt_target, atol=dt_target * 0.1):
                            print(f"Warning: File dt ({file_dt:.4f} s) differs from target dt ({dt_target:.4f} s).")
                        first_line_processed = True
                    except ValueError:
                        print("Warning: Could not parse dt from the first line.")

        time_gm = np.array(time_gm)
        accel_g_values = np.array(accel_g_values)

        if not time_gm.size:
            raise ValueError("No ground motion data loaded from the file.")

        # Resample to ensure consistent time steps
        time_resampled = np.arange(time_gm[0], time_gm[-1] + dt_target, dt_target)
        accel_resampled = np.interp(time_resampled, time_gm, accel_g_values, left=accel_g_values[0], right=accel_g_values[-1])

        return time_resampled, accel_resampled * g

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Ground motion file not found at: {file_path}")
    except ValueError as e:
        raise ValueError(f"Error loading ground motion data: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")

def newmark_beta_sdof(T_period, damping_ratio, time_array_input, ground_acceleration_history, dt_calc):
    """
    Calculates the seismic response of a single-degree-of-freedom system using the Newmark-Beta method.
    """
    omega_n = 2 * np.pi / T_period
    m = 1.0
    c = 2 * damping_ratio * omega_n * m
    k = omega_n**2 * m
    n_steps = len(time_array_input)
    time_output = time_array_input
    u = np.zeros(n_steps)
    v = np.zeros(n_steps)
    a_rel = np.zeros(n_steps)
    a_abs = np.zeros(n_steps)

    a0 = 1 / (beta_nm * dt_calc**2)
    a1 = gamma_nm / (beta_nm * dt_calc)
    a2 = 1 / (beta_nm * dt_calc)
    a3 = 1 / (2 * beta_nm) - 1
    a4 = gamma_nm / beta_nm - 1
    a5 = dt_calc / 2 * (gamma_nm / beta_nm - 2)
    a6 = dt_calc * (1 - gamma_nm)
    a7 = dt_calc * gamma_nm

    k_hat = k + c * a1 + m * a0

    if n_steps > 0:
        a_rel[0] = -ground_acceleration_history[0] / m
        a_abs[0] = a_rel[0] + ground_acceleration_history[0]

    for i in range(n_steps - 1):
        p_hat = -m * ground_acceleration_history[i+1] + m * (a0 * u[i] + a2 * v[i] + a3 * a_rel[i]) + c * (a1 * u[i] + a4 * v[i] + a5 * a_rel[i])
        u[i+1] = p_hat / k_hat
        a_rel[i+1] = a0 * (u[i+1] - u[i]) - a2 * v[i] - a3 * a_rel[i]
        v[i+1] = v[i] + a6 * a_rel[i] + a7 * a_rel[i+1]
        a_abs[i+1] = a_rel[i+1] + ground_acceleration_history[i+1]

    return time_output, u, v, a_abs

# --- Analysis and Plotting ---
try:
    time_gm_processed, ground_accel_abs = load_ground_motion(ground_motion_file, dt)
except (FileNotFoundError, ValueError, Exception) as e:
    print(e)
    exit()

directions = {
    "E-W": {"T": T_EW, "color": "red"},
    "N-S": {"T": T_NS, "color": "blue"}
}

if time_gm_processed.size < 2:
    print("Error: Insufficient ground motion data for analysis.")
else:
    for direction_name, params in directions.items():
        T = params["T"]
        color = params["color"]
        print(f"\nAnalyzing {direction_name} direction (T={T:.4f} s)...")
        time_results, rel_disp, rel_vel, abs_accel = newmark_beta_sdof(T, Xi, time_gm_processed, ground_accel_abs, dt)

        if not rel_disp.size:
            print(f"Warning: No results for {direction_name} direction.")
            continue

        plt.figure(figsize=(12, 10))
        plt.suptitle(f'Structural Response ({direction_name} Direction, T={T:.3f} s, $\\xi$={Xi})', fontsize=16)

        plt.subplot(3, 1, 1)
        plt.plot(time_results, rel_disp, label=f'$x(t)$ - {direction_name}', color=color)
        plt.xlabel('Time (s)')
        plt.ylabel('Relative Displacement $x(t)$ (in)')
        plt.grid(True)
        plt.legend()
        plt.title('Relative Displacement vs. Time')

        plt.subplot(3, 1, 2)
        plt.plot(time_results, rel_vel, label=f'$\\dot{{x}}(t)$ - {direction_name}', color=color)
        plt.xlabel('Time (s)')
        plt.ylabel('Relative Velocity $\\dot{{x}}(t)$ (in/s)')
        plt.grid(True)
        plt.legend()
        plt.title('Relative Velocity vs. Time')

        plt.subplot(3, 1, 3)
        plt.plot(time_results, abs_accel / g, label=f'$\\ddot{{x}}_t(t)$ - {direction_name}', color=color)
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Acceleration $\\ddot{{x}}_t(t)$ (g)')
        plt.grid(True)
        plt.legend()
        plt.title('Absolute Acceleration vs. Time')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

print("\nAnalysis completed.")
print("Note:")
print("1. Assumed that the acceleration in 'Northridge_NS (2).txt' is in 'g'.")
print("2. The plotted absolute acceleration $\\ddot{{x}}_t(t)$ is the total acceleration of the mass.")
print("3. Ensure the ground motion file is in the specified path.")
print(f"4. The target time step for analysis is dt = {dt} s.")