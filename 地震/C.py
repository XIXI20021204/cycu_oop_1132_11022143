import numpy as np
import matplotlib.pyplot as plt

<<<<<<< HEAD
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
=======
# 載入地震動資料
def load_ground_motion(file_path):
    """載入地震動資料，假設檔案為兩欄，時間和加速度。"""
    time = []
    accel = []
    try:
        with open(file_path, 'r') as f:
            next(f)  # 跳過標頭行
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    time.append(float(parts[0]))
                    accel.append(float(parts[1]))
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file_path}")
        return None, None
    except ValueError:
        print(f"錯誤：檔案 {file_path} 資料格式不正確")
        return None, None
    return np.array(time), np.array(accel)

# Newmark-Beta 方法
def newmark_beta_sdof(T, xi, time_gm, ground_accel_history, dt):
    omega_n = 2 * np.pi / T
    m = 1.0  # 假設質量為 1
    c = 2 * xi * omega_n * m
    k = omega_n**2 * m

    n_steps = len(time_gm)
    time_output = time_gm
>>>>>>> bcb257d5937b93727817d2a7647a957f69e534cc
    u = np.zeros(n_steps)
    v = np.zeros(n_steps)
    a_rel = np.zeros(n_steps)
    a_abs = np.zeros(n_steps)

<<<<<<< HEAD
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
=======
    beta = 0.25
    gamma = 0.5

    # 初始條件
    a_abs[0] = ground_accel_history[0] - (c * v[0] + k * u[0]) / m
    a_rel[0] = ground_accel_history[0] - a_abs[0]

    for i in range(n_steps - 1):
        delta_t = dt

        # 預測步驟
        u_pred = u[i] + v[i] * delta_t + a_rel[i] * delta_t**2 / 2
        v_pred = v[i] + a_rel[i] * delta_t

        # 計算有效剛度
        k_eff = k + c * gamma / (beta * delta_t) + m / (beta * delta_t**2)

        # 計算有效力增量
        delta_p_eff = -m * (ground_accel_history[i+1] - ground_accel_history[i]) + \
                      c * (gamma / (beta * delta_t) * u[i] + (gamma / beta - 1) * v[i] + (gamma / (2 * beta) - 1) * a_rel[i] * delta_t) + \
                      k * (u[i] + v[i] * delta_t + (0.5 - beta) * a_rel[i] * delta_t**2)

        # 計算位移增量
        delta_u = delta_p_eff / k_eff

        # 更新位移、速度和相對加速度
        u[i+1] = u[i] + delta_u
        v[i+1] = v[i] + gamma / (beta * delta_t) * delta_u - (gamma / beta - 1) * v[i] - (gamma / (2 * beta) - 1) * a_rel[i] * delta_t
        a_rel[i+1] = 1 / (beta * delta_t**2) * delta_u - 1 / (beta * delta_t) * v[i] - (1 / (2 * beta) - 1) * a_rel[i]

        # 計算絕對加速度
        a_abs[i+1] = ground_accel_history[i+1] - a_rel[i+1]

    return time_output, u, v, a_abs

# 繪圖函數
def plot_results(time, rel_disp, rel_vel, abs_accel, direction_name, color):
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'結構響應歷時 ({direction_name}), T={T:.2f}s, ξ={xi*100:.1f}%', fontsize=16)

    # 相對位移
    plt.subplot(3, 1, 1)
    plt.plot(time, rel_disp, label=f'u(t) - {direction_name}', color=color)
    plt.xlabel('時間 (s)')
    plt.ylabel('相對位移 (m)')
    plt.grid(True)
    plt.legend()

    # 相對速度
    plt.subplot(3, 1, 2)
    plt.plot(time, rel_vel, label=r'$\dot{u}(t)$ - ' + f'{direction_name}', color=color)
    plt.xlabel('時間 (s)')
    plt.ylabel('相對速度 (m/s)')
    plt.grid(True)
    plt.legend()

    # 絕對加速度
    plt.subplot(3, 1, 3)
    plt.plot(time, abs_accel, label=r'$\ddot{u}_a(t)$ - ' + f'{direction_name}', color=color)
    plt.xlabel('時間 (s)')
    plt.ylabel('絕對加速度 (g)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --- 主要程式碼 ---
if __name__ == "__main__":
    # 設定參數
    ground_motion_file_NS = r"C:\Users\rsrs1\OneDrive\文件\GitHub\cycu_pop_1132_11022232\Northridge_NS.txt"
    ground_motion_file_EW = r"C:\Users\rsrs1\OneDrive\文件\GitHub\cycu_pop_1132_11022232\Northridge_EW.txt"
    dt = 0.02  # 假設地震動資料的時間間隔為 0.02 秒 (根據註解)
    xi = 0.05  # 阻尼比
    T_NS = 2.0  # N-S 向的自然週期 (秒)
    T_EW = 1.5  # E-W 向的自然週期 (秒)

    directions = {
        "N-S": ("blue", ground_motion_file_NS, T_NS),
        "E-W": ("red", ground_motion_file_EW, T_EW),
    }

    print("開始分析...")

    for direction, (color, file, period) in directions.items():
        time_gm, ground_accel = load_ground_motion(file)
        if time_gm is not None and ground_accel is not None:
            # 將地震動加速度單位從 'g' 轉換為 m/s^2
            ground_accel_ms2 = ground_accel * 9.81
            time_response, rel_disp, rel_vel, abs_accel = newmark_beta_sdof(period, xi, time_gm, ground_accel_ms2, dt)

            # 將絕對加速度單位轉換回 'g' 以便繪圖
            abs_accel_g = abs_accel / 9.81
            plot_results(time_response, rel_disp, rel_vel, abs_accel_g, direction, color)
        else:
            print(f"跳過 {direction} 方向的分析。")

    print("分析完成。")
    print("(註1. 假定 'Northridge_NS (2).txt' 中的第二列加速度是以 'g' 為單位。)")
    print("(註2. $\ddot{x}(t)$ 結構的絕對加速度，其定義為 $-\zeta(2/\omega_n)\dot{x}(t) + \omega_n^2 x(t) + \ddot{x}_g(t)$。)")
    print("(註3. 確保 'Northridge_NS (2).txt' 檔案與 Python 腳本在同一目錄，或提供正確路徑。)")
    print(f"(註4. 計算中使用的時間間隔 dt 為 {dt} 秒。)")
>>>>>>> bcb257d5937b93727817d2a7647a957f69e534cc
