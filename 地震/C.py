import numpy as np
import matplotlib.pyplot as plt

def load_ground_motion(file_path, dt, g):
    """
    載入地震動資料。

    Args:
        file_path (str): 地震動檔案路徑。
        dt (float): 時間間隔。
        g (float): 重力加速度。

    Returns:
        tuple: 包含時間序列 (time)、加速度 (ground_accel)、速度 (ground_vel)、位移 (ground_disp)。
    """
    time = []
    ground_accel_values = []
    first_processed = False
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                parts = line.split()
                if parts:
                    try:
                        current_time = float(parts[0])
                        acc_val = float(parts[1])
                        time.append(current_time)
                        ground_accel_values.append(acc_val * g) # Convert to consistent units
                        if not first_processed:
                            first_processed = True
                            initial_time = current_time
                    except ValueError:
                        if not first_processed:
                            raise ValueError("地震動檔案格式錯誤：前幾行應為數值。")
                        else:
                            continue # Skip non-numeric lines after data starts
    except FileNotFoundError:
        raise FileNotFoundError(f"地震動檔案未找到：{file_path}")
    except Exception as e:
        raise RuntimeError(f"讀取地震動檔案時發生錯誤：{e}")

    time = np.array(time) - initial_time
    ground_accel = np.array(ground_accel_values)
    ground_vel = np.zeros_like(ground_accel)
    ground_disp = np.zeros_like(ground_accel)

    # Integrate to get velocity and displacement
    for i in range(1, len(ground_accel)):
        ground_vel[i] = ground_vel[i-1] + 0.5 * dt * (ground_accel[i] + ground_accel[i-1])
        ground_disp[i] = ground_disp[i-1] + 0.5 * dt * (ground_vel[i] + ground_vel[i-1])

    return time, ground_accel, ground_vel, ground_disp

def newmark_beta_sdof(T_period, damping_ratio, time_array, input_accel, dt_calc, beta=0.25, gamma=0.5):
    """
    使用 Newmark-beta 方法分析單自由度系統。

    Args:
        T_period (float): 單自由度系統的自然週期。
        damping_ratio (float): 單自由度系統的阻尼比。
        time_array (np.ndarray): 時間序列。
        input_accel (np.ndarray): 輸入加速度時程。
        dt_calc (float): 計算的時間間隔。
        beta (float): Newmark-beta 方法的 beta 參數 (預設為 0.25，常數平均加速度法)。
        gamma (float): Newmark-beta 方法的 gamma 參數 (預設為 0.5)。

    Returns:
        tuple: 包含時間序列 (time_output)、位移 (u)、速度 (v)、加速度 (a_abs)。
    """
    omega_n = 2 * np.pi / T_period
    zeta = damping_ratio
    m = 1  # 假設質量為 1 以簡化計算，力與加速度單位相同
    c = 2 * zeta * omega_n * m
    k = omega_n**2 * m

    n_steps = len(time_array)
    time_output = time_array
    u = np.zeros(n_steps)
    v = np.zeros(n_steps)
    a = np.zeros(n_steps)
    a_abs = np.zeros(n_steps)

    delta_t = dt_calc

    a[0] = input_accel[0] - (c * v[0] + k * u[0]) / m # Initial absolute acceleration
    a_abs[0] = a[0]

    a_rel_0 = input_accel[0] - a[0] # Initial relative acceleration (should be 0 if initial conditions are zero)

    # Newmark-beta 係數
    a_rel_beta_dt2 = beta * delta_t**2
    a_rel_gamma_dt = gamma * delta_t

    k_eff = k + gamma / (beta * delta_t) * c + m / (beta * delta_t**2)
    delta_f_eff = m * (input_accel[1:] - input_accel[:-1]) + c * (gamma / (beta * delta_t) * u[:-1] + (gamma / beta - 1) * v[:-1] + delta_t * (gamma / (2 * beta) - 1) * a_rel[:-1]) + k * (u[:-1] + delta_t * v[:-1] + delta_t**2 / 2 * (1 - 2 * beta) * a_rel[:-1])

    delta_u = np.zeros(n_steps - 1)
    delta_v = np.zeros(n_steps - 1)
    delta_a_rel = np.zeros(n_steps - 1)

    for i in range(n_steps - 1):
        delta_u[i] = delta_f_eff[i] / k_eff
        delta_v[i] = gamma / (beta * delta_t) * delta_u[i] - gamma / beta * v[i] + delta_t * (1 - gamma / (2 * beta)) * a_rel[i]
        delta_a_rel[i] = 1 / (beta * delta_t**2) * delta_u[i] - 1 / (beta * delta_t) * v[i] - (1 / (2 * beta) - 1) * a_rel[i]

        u[i+1] = u[i] + delta_u[i]
        v[i+1] = v[i] + delta_v[i]
        a_rel[i+1] = a_rel[i] + delta_a_rel[i]
        a[i+1] = input_accel[i+1] - a_rel[i+1]
        a_abs[i+1] = a[i+1]

    return time_output, u, v, a_abs

def plot_response(directions, time_gm, ground_accel, abs_accel_dict, disp_dict, vel_dict, xi, T_val):
    """
    繪製地震動和結構響應。

    Args:
        directions (dict): 方向名稱和顏色的字典。
        time_gm (np.ndarray): 地震動時間序列。
        ground_accel (np.ndarray): 地震動加速度時程。
        abs_accel_dict (dict): 各方向的絕對加速度響應字典。
        disp_dict (dict): 各方向的位移響應字典。
        vel_dict (dict): 各方向的速度響應字典。
        xi (float): 阻尼比。
        T_val (float): 自然週期。
    """
    plt.figure(figsize=(10, 12))

    plt.subplot(5, 1, 1)
    plt.plot(time_gm, ground_accel)
    plt.ylabel('Ground Acceleration (g)')
    plt.title(f'Ground Motion (ξ={xi}, T={T_val}s)')
    plt.grid(True)

    i = 2
    for direction_name, color in directions.items():
        if direction_name in disp_dict:
            plt.subplot(5, 1, i)
            plt.plot(time_gm, disp_dict[direction_name], label=direction_name, color=color)
            plt.ylabel('Displacement (m)')
            plt.legend()
            plt.grid(True)
            i += 1
        if direction_name in vel_dict:
            plt.subplot(5, 1, i)
            plt.plot(time_gm, vel_dict[direction_name], label=direction_name, color=color)
            plt.ylabel('Velocity (m/s)')
            plt.legend()
            plt.grid(True)
            i += 1
        if direction_name in abs_accel_dict:
            plt.subplot(5, 1, i)
            plt.plot(time_gm, abs_accel_dict[direction_name] / 9.81, label=direction_name, color=color) # Convert back to g for plotting
            plt.ylabel('Absolute Acceleration (g)')
            plt.legend()
            plt.grid(True)
            i += 1

    plt.tight_layout()
    plt.show()

# --- 主要程式碼 ---
if __name__ == "__main__":
    # 設定參數
    file_path_T_NS = r'C:\Users\rsrs1\OneDrive\文件\GitHub\cycu_pop_1132_11022232\地震工程\T-NS.txt'
    file_path_T_EW = r'C:\Users\rsrs1\OneDrive\文件\GitHub\cycu_pop_1132_11022232\地震工程\T-EW.txt'
    dt_gm = 0.01  # 地震動資料的時間間隔 (秒)
    g_val = 9.81  # 重力加速度 (m/s^2)
    dt_analysis = 0.01 # 分析時的時間間隔 (可以與地震動的時間間隔不同)
    damping_ratio = 0.05  # 阻尼比
    T_natural_NS = 1.0  # N-S 向的自然週期 (秒)
    T_natural_EW = 0.8  # E-W 向的自然週期 (秒)

    directions = {
        "T-NS": "tab:blue",
        "T-EW": "tab:orange",
    }

    abs_accel_response = {}
    displacement_response = {}
    velocity_response = {}

    # 載入地震動資料
    try:
        time_NS, ground_accel_NS, ground_vel_NS, ground_disp_NS = load_ground_motion(file_path_T_NS, dt_gm, g_val)
        time_EW, ground_accel_EW, ground_vel_EW, ground_disp_EW = load_ground_motion(file_path_T_EW, dt_gm, g_val)

        ground_motions = {
            "T-NS": (time_NS, ground_accel_NS),
            "T-EW": (time_EW, ground_accel_EW),
        }

        natural_periods = {
            "T-NS": T_natural_NS,
            "T-EW": T_natural_EW,
        }

        # 進行 Newmark-beta 分析並繪圖
        for direction_name, color in directions.items():
            if direction_name in ground_motions and direction_name in natural_periods:
                time_gm_current, ground_accel_current = ground_motions[direction_name]
                T_natural_current = natural_periods[direction_name]

                time_response, u, v, a_abs = newmark_beta_sdof(
                    T_natural_current, damping_ratio, time_gm_current, ground_accel_current, dt_analysis
                )
                abs_accel_response[direction_name] = a_abs
                displacement_response[direction_name] = u
                velocity_response[direction_name] = v

        # 繪製結果
        # 使用 T-NS 的時間作為繪圖的時間軸 (假設兩個地震動的時間長度大致相同)
        if "T-NS" in ground_motions:
            plot_response(directions, time_NS, ground_accel_NS, abs_accel_response, displacement_response, velocity_response, damping_ratio, T_natural_NS)
        elif "T-EW" in ground_motions:
            plot_response(directions, time_EW, ground_accel_EW, abs_accel_response, displacement_response, velocity_response, damping_ratio, T_natural_EW)
        else:
            print("沒有可繪製的地震動資料。")

    except FileNotFoundError as e:
        print(f"錯誤：{e}")
    except ValueError as e:
        print(f"錯誤：{e}")
    except RuntimeError as e:
        print(f"錯誤：{e}")