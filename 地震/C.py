import numpy as np
import matplotlib.pyplot as plt

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
    u = np.zeros(n_steps)
    v = np.zeros(n_steps)
    a_rel = np.zeros(n_steps)
    a_abs = np.zeros(n_steps)

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