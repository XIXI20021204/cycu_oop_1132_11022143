import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. Load Earthquake Ground Acceleration Data ---
# 確保 'Kobe.txt' 在與此腳本相同的目錄中，或提供完整的路徑。
file_path = r'C:\Users\a0965\OneDrive\文件\GitHub\cycu_oop_1132_11022143\地震期末\Kobe.txt'
try:
    # 讀取數據，跳過第一行（標題），並指定列名
    df_ground_accel = pd.read_csv(file_path, sep='\s+', header=None, skiprows=1, names=['Time (s)', 'Acceleration (g)'])
    # 將加速度從 'g' 轉換為 m/s^2 (假設 1g = 9.81 m/s^2)
    g = 9.81  # 重力加速度，單位 m/s^2
    df_ground_accel['Acceleration (m/s²)'] = df_ground_accel['Acceleration (g)'] * g
    time_series = df_ground_accel['Time (s)'].values  # 時間序列
    ground_accel = df_ground_accel['Acceleration (m/s²)'].values  # 地表加速度
    dt = time_series[1] - time_series[0]  # 時間步長
except FileNotFoundError:
    print(f"錯誤：找不到檔案 '{file_path}'。請確保它在正確的目錄中。")
    exit()

# --- 2. Define Main Structure Parameters (Constant for both SDOF and 2DOF analyses) ---
ms = 84600000  # KG (主結構質量)
omega_ns = 0.9174  # rad/s (主結構固有頻率)
zeta_s = 0.01  # (主結構阻尼比)

# 創建輸出目錄
output_dir = 'output_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 儲存所有結果供比較繪圖和進一步分析
all_results_for_comparison_plots = {}
all_results_dfs = {} # 用於儲存完整的數據框，以便個別 CSV 和摘要

# ==============================================================================
# --- PART 1: Simulate Single Floor Structure (No TMD) ---
# ==============================================================================
print("\n" + "="*50)
print("--- 正在分析單層結構 (無 TMD) ---")
print("="*50)

# 導出 SDOF 系統的物理參數
ks_sdof = ms * (omega_ns**2)
cs_sdof = 2 * zeta_s * ms * omega_ns

print(f"導出的主結構參數：")
print(f"   質量 (m): {ms:.2f} KG")
print(f"   剛度 (k): {ks_sdof:.2f} N/m")
print(f"   阻尼係數 (c): {cs_sdof:.2f} Ns/m")
print(f"   固有頻率 (omega_n): {omega_ns:.4f} rad/s")
print(f"   阻尼比 (zeta): {zeta_s:.4f}")

# --- SDOF 數值積分 (Newmark-Beta 方法) ---
gamma = 0.5
beta = 0.25

num_steps = len(time_series)
# response array 儲存 [u, v, a] 給單自由度系統
response_sdof = np.zeros((num_steps, 3))

# 初始加速度計算 (SDOF): m*a_0 + c*v_0 + k*u_0 = -m*a_g0
# 假設初始位移 u_0 = 0, 初始速度 v_0 = 0
# 所以，m*a_0 = -m*a_g0  => a_0 = -a_g0 (這裡的 a_0 是相對加速度)
initial_accel_sdof = -ground_accel[0]

response_sdof[0, 2] = initial_accel_sdof # 初始相對加速度

# 預計算 SDOF 的 K_eff
K_eff_sdof = ks_sdof + (gamma / (beta * dt)) * cs_sdof + (1 / (beta * dt**2)) * ms

for i in range(num_steps - 1):
    # 當前狀態
    u_i_sdof = response_sdof[i, 0]
    v_i_sdof = response_sdof[i, 1]
    a_i_sdof = response_sdof[i, 2] # 相對加速度

    # t+dt 時刻的外部力 (SDOF): P_t_plus_dt = -m * ground_accel[i+1]
    P_t_plus_dt_sdof = -ms * ground_accel[i+1]

    # Newmark-Beta 方法的有效載荷 (RHS) for SDOF
    RHS_sdof = P_t_plus_dt_sdof + \
               ms * ((1/(beta*dt**2))*u_i_sdof + (1/(beta*dt))*v_i_sdof + (1/(2*beta) - 1)*a_i_sdof) + \
               cs_sdof * ((gamma/(beta*dt))*u_i_sdof + (gamma/beta - 1)*v_i_sdof + (gamma/2 - beta)*dt*a_i_sdof)

    # 求解 t+dt 時刻的位移
    u_t_plus_dt_sdof = RHS_sdof / K_eff_sdof

    # 更新 t+dt 時刻的加速度和速度
    a_t_plus_dt_sdof = (1/(beta*dt**2)) * (u_t_plus_dt_sdof - u_i_sdof) - (1/(beta*dt)) * v_i_sdof - (1/(2*beta) - 1) * a_i_sdof
    v_t_plus_dt_sdof = v_i_sdof + (1 - gamma) * dt * a_i_sdof + gamma * dt * a_t_plus_dt_sdof

    # 儲存結果
    response_sdof[i+1, 0] = u_t_plus_dt_sdof # 相對位移
    response_sdof[i+1, 1] = v_t_plus_dt_sdof # 相對速度
    response_sdof[i+1, 2] = a_t_plus_dt_sdof # 相對加速度

# --- 準備 SDOF 結果數據框 ---
results_df_sdof = pd.DataFrame({
    'Time (s)': time_series,
    'Ground Accel (m/s²)': ground_accel,
    'Floor Disp (m)': response_sdof[:, 0],  # 相對於地面的位移
    'Floor Vel (m/s)': response_sdof[:, 1],  # 相對於地面的速度
    'Floor Accel (m/s²)': response_sdof[:, 2] + ground_accel # 絕對加速度
})

print(f"\n--- 單層結構 (無 TMD) 計算結果前 5 行 ---")
print(results_df_sdof.head())

# 儲存單層結構結果到 CSV
output_csv_filename_sdof = "Single_Floor_No_TMD_simulation_results.csv"
output_csv_path_sdof = os.path.join(output_dir, output_csv_filename_sdof)
try:
    results_df_sdof.to_csv(output_csv_path_sdof, index=False, encoding='utf-8')
    print(f"單層結構 (無 TMD) 計算結果成功儲存至：{output_csv_path_sdof}")
except Exception as e:
    print(f"儲存單層結構 (無 TMD) 檔案時發生錯誤：{e}")

# 將單層結構的位移結果加入到比較列表
all_results_for_comparison_plots['Single Floor (No TMD)'] = {
    'Floor Disp': results_df_sdof['Floor Disp (m)']
}
all_results_dfs['Single Floor (No TMD)'] = results_df_sdof # 儲存完整數據框

# 計算單層結構位移統計數據
floor_disp_sdof = results_df_sdof['Floor Disp (m)']
mean_disp_sdof = np.mean(floor_disp_sdof)
rms_disp_sdof = np.sqrt(np.mean(floor_disp_sdof**2))
peak_disp_sdof = np.max(np.abs(floor_disp_sdof)) # 峰值使用絕對值

print("\n--- 單層結構 (無 TMD) 位移統計 ---")
print(f"平均位移: {mean_disp_sdof:.6f} m")
print(f"RMS 位移: {rms_disp_sdof:.6f} m")
print(f"峰值位移: {peak_disp_sdof:.6f} m")

print("\n--- 單層結構模擬完成 ---")

# ==============================================================================
# --- PART 2: Simulate Main Structure with TMD Configurations ---
# ==============================================================================
print("\n\n" + "="*50)
print("--- 正在分析帶有 TMD 的主結構 ---")
print("="*50)

tmd_configurations = [
    {"label": "TMD_Config_1 (mu=0.03, alpha=0.9592, zeta_d=0.0857)", "mu": 0.03, "alpha": 0.9592, "zeta_d": 0.0857},
    {"label": "TMD_Config_2 (mu=0.1, alpha=0.8789, zeta_d=0.1527)", "mu": 0.1, "alpha": 0.8789, "zeta_d": 0.1527},
    {"label": "TMD_Config_3 (mu=0.2, alpha=0.7815, zeta_d=0.2098)", "mu": 0.2, "alpha": 0.7815, "zeta_d": 0.2098},
]

# --- 遍歷每個 TMD 配置進行計算 ---
for config_num, tmd_config in enumerate(tmd_configurations):
    print(f"\n--- 正在分析 {tmd_config['label']} ---")

    mu = tmd_config['mu']
    alpha = tmd_config['alpha']
    zeta_d = tmd_config['zeta_d']

    # --- 導出物理參數 ---
    # 主結構參數 (沿用 SDOF 的 ms, omega_ns, zeta_s)
    ks = ms * (omega_ns**2)
    cs = 2 * zeta_s * ms * omega_ns

    # TMD 參數
    md = mu * ms
    omega_nd = alpha * omega_ns
    kd = md * (omega_nd**2)
    cd = 2 * zeta_d * md * omega_nd

    print(f"導出的主結構參數 (常數)：")
    print(f"   剛度 (ks): {ks:.2f} N/m")
    print(f"   阻尼係數 (cs): {cs:.2f} Ns/m")
    print(f"\n導出的 TMD 參數 (針對 {tmd_config['label']})：")
    print(f"   阻尼器質量 (md): {md:.2f} KG")
    print(f"   阻尼器固有頻率 (omega_nd): {omega_nd:.4f} rad/s")
    print(f"   阻尼器剛度 (kd): {kd:.2f} N/m")
    print(f"   阻尼器阻尼係數 (cd): {cd:.2f} Ns/m")

    # --- 建立 2DOF 系統矩陣 ---
    M = np.array([[ms, 0],
                  [0, md]])

    K = np.array([[ks + kd, -kd],
                  [-kd, kd]])

    C = np.array([[cs + cd, -cd],
                  [-cd, cd]])

    # 載荷矩陣（地面加速度的影響向量）
    # 對於相對位移公式 (u_s, u_d_rel_s)，力向量是 -M @ [1, 0]^T * accel_g
    # 這表示地面加速度主要激發主結構。
    load_matrix = np.array([[1], [0]])

    # --- 2DOF 數值積分 (Newmark-Beta 方法) ---
    # gamma 和 beta 值與 SDOF 相同
    response_2dof = np.zeros((num_steps, 6))
    # response_2dof array 儲存 [u_s, u_d_rel_s, v_s, v_d_rel_s, a_s_rel_g, a_d_rel_s]
    # 其中 u_s 是主結構相對於地面的位移，u_d_rel_s 是 TMD 相對於主結構的位移。

    # 初始加速度計算: M a_0 + C v_0 + K u_0 = P_0
    # 假設初始位移和速度為零 (u_0 = 0, v_0 = 0)。
    # P_0 = -M @ load_matrix * ground_accel[0]
    initial_accel_vec_2dof = np.linalg.solve(M, -M @ load_matrix * ground_accel[0])

    response_2dof[0, 4] = initial_accel_vec_2dof[0, 0] # 主結構初始相對加速度 (a_s_rel_g)
    response_2dof[0, 5] = initial_accel_vec_2dof[1, 0] # TMD 相對於主結構的初始相對加速度 (a_d_rel_s)

    # 預計算 K_eff (在整個積分過程中是常數)
    K_eff_2dof = K + (gamma / (beta * dt)) * C + (1 / (beta * dt**2)) * M

    for i in range(num_steps - 1):
        # 當前狀態向量 (重塑為列向量以進行矩陣操作)
        u_i = response_2dof[i, 0:2].reshape(-1, 1) # 當前相對位移
        v_i = response_2dof[i, 2:4].reshape(-1, 1) # 當前相對速度
        a_i = response_2dof[i, 4:6].reshape(-1, 1) # 當前相對加速度

        # t+dt 時刻的外部力向量
        P_t_plus_dt = -M @ load_matrix * ground_accel[i+1]

        # Newmark-Beta 方法的有效載荷向量 (RHS)
        RHS_force_terms = P_t_plus_dt + \
                          M @ ((1/(beta*dt**2))*u_i + (1/(beta*dt))*v_i + (1/(2*beta) - 1)*a_i) + \
                          C @ ((gamma/(beta*dt))*u_i + (gamma/beta - 1)*v_i + (gamma/2 - beta)*dt*a_i)

        # 求解 t+dt 時刻的位移 (u_t_plus_dt)
        u_t_plus_dt = np.linalg.solve(K_eff_2dof, RHS_force_terms)

        # 使用 Newmark-Beta 公式更新 t+dt 時刻的加速度和速度
        a_t_plus_dt = (1/(beta*dt**2)) * (u_t_plus_dt - u_i) - (1/(beta*dt)) * v_i - (1/(2*beta) - 1) * a_i
        v_t_plus_dt = v_i + (1 - gamma) * dt * a_i + gamma * dt * a_t_plus_dt

        # 儲存結果到下一個時間步長 (攤平以儲存到 response array 的 1D 切片中)
        response_2dof[i+1, 0:2] = u_t_plus_dt.flatten() # [u_s_rel_g, u_d_rel_s]
        response_2dof[i+1, 2:4] = v_t_plus_dt.flatten() # [v_s_rel_g, v_d_rel_s]
        response_2dof[i+1, 4:6] = a_t_plus_dt.flatten() # [a_s_rel_g, a_d_rel_s]

    # --- 計算輸出和分析的絕對響應 ---
    # TMD 絕對位移 = (主結構相對於地面的位移) + (TMD 相對於主結構的位移)
    u_d_abs = response_2dof[:, 0] + response_2dof[:, 1]
    # 主結構絕對加速度 = (主結構相對加速度) + (地面加速度)
    u_double_dot_s_abs = response_2dof[:, 4] + ground_accel
    # TMD 絕對加速度 = (TMD 相對於地面的相對加速度) + (地面加速度)
    # TMD 相對於地面的相對加速度 = (主結構相對加速度) + (TMD 相對於主結構的相對加速度)
    u_double_dot_d_abs = (response_2dof[:, 4] + response_2dof[:, 5]) + ground_accel

    # 創建一個 DataFrame 來儲存此配置的所有計算響應
    results_df_tmd = pd.DataFrame({
        'Time (s)': time_series,
        'Ground Accel (m/s²)': ground_accel,
        'Floor Disp (m)': response_2dof[:, 0],          # 主結構相對於地面的相對位移
        'Floor Vel (m/s)': response_2dof[:, 2],          # 主結構相對於地面的相對速度
        'Floor Accel (m/s²)': u_double_dot_s_abs,      # 主結構絕對加速度
        'Damper Rel Disp (m)': response_2dof[:, 1],       # TMD 相對於主結構的相對位移
        'Damper Rel Vel (m/s)': response_2dof[:, 3],      # TMD 相對於主結構的相對速度
        'Damper Rel Accel (m/s²)': response_2dof[:, 5],  # TMD 相對於主結構的相對加速度
        'Damper Abs Disp (m)': u_d_abs,                 # TMD 絕對位移
        'Damper Abs Accel (m/s²)': u_double_dot_d_abs # TMD 絕對加速度
    })

    print(f"\n--- {tmd_config['label']} 計算結果前 5 行 ---")
    print(results_df_tmd.head())

    # 儲存完整的 DataFrame 以便個別 CSV 和摘要
    all_results_dfs[tmd_config['label']] = results_df_tmd

    # 儲存相關數據以用於比較繪圖 (只需要 Floor Disp)
    all_results_for_comparison_plots[tmd_config['label']] = {
        'Floor Disp': results_df_tmd['Floor Disp (m)']
    }

    # --- 儲存個別結果到 CSV 檔案 ---
    # 清理標籤以用於檔案名，替換空格和括號
    output_csv_filename_tmd = f"{tmd_config['label'].replace(' ', '_').replace('(', '').replace(')', '')}_simulation_results.csv"
    output_csv_path_tmd = os.path.join(output_dir, output_csv_filename_tmd)
    try:
        results_df_tmd.to_csv(output_csv_path_tmd, index=False, encoding='utf-8')
        print(f"針對 {tmd_config['label']} 的計算結果成功儲存至：{output_csv_path_tmd}")
    except Exception as e:
        print(f"儲存 {tmd_config['label']} 檔案時發生錯誤：{e}")

    # --- 基本性能指標 (針對個別配置) ---
    max_ground_accel = np.max(np.abs(ground_accel))
    max_floor_accel = np.max(np.abs(results_df_tmd['Floor Accel (m/s²)']))
    max_floor_disp = np.max(np.abs(results_df_tmd['Floor Disp (m)']))
    max_damper_abs_accel = np.max(np.abs(results_df_tmd['Damper Abs Accel (m/s²)']))
    max_damper_rel_disp = np.max(np.abs(results_df_tmd['Damper Rel Disp (m)']))

    print(f"\n--- {tmd_config['label']} 響應摘要 ---")
    print(f"最大地面加速度: {max_ground_accel:.4f} m/s²")
    print(f"最大樓層絕對加速度: {max_floor_accel:.4f} m/s²")
    print(f"最大樓層位移 (相對於地面): {max_floor_disp:.4f} m")
    print(f"最大阻尼器絕對加速度: {max_damper_abs_accel:.4f} m/s²")
    print(f"最大阻尼器相對位移 (相對於樓層): {max_damper_rel_disp:.4f} m")

print("\n--- 所有個別模擬完成。正在生成比較圖表 ---")

# ==============================================================================
# --- FINAL: Plotting All Comparison Results (Floor Displacement) ---
# ==============================================================================
print("\n" + "="*50)
print("--- 正在生成所有結構的主結構位移比較圖 ---")
print("="*50)

plt.figure(figsize=(14, 8)) # 調整圖表大小以容納更多曲線

# 繪製所有結果
for label, data in all_results_for_comparison_plots.items():
    plt.plot(time_series, data['Floor Disp'], label=label, linewidth=1.5) # 稍微加粗線條

plt.title('Main Structure Displacement Response Comparison (Relative to Ground)\n'
          'Single Floor (No TMD) vs. Different TMD Configurations', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Displacement (m)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right', fontsize=10) # 調整圖例位置和字體大小
plt.tight_layout() # 自動調整佈局以避免重疊
comparison_plot_path_all_disp = os.path.join(output_dir, 'All_Main_Structure_Displacement_Comparison.png')
plt.savefig(comparison_plot_path_all_disp)
print(f"所有主結構位移比較圖成功儲存至：{comparison_plot_path_all_disp}")
plt.show()

# --- 計算並列印主結構位移的平均值、RMS 值、峰值 ---
print("\n" + "="*50)
print("--- 主結構位移性能指標 (相對於地面) ---")
print("="*50)
print("{:<45} {:<15} {:<15} {:<15}".format("結構配置", "平均值 (m)", "RMS (m)", "峰值 (m)"))
print("-" * 90)

for label, df in all_results_dfs.items():
    floor_disp = df['Floor Disp (m)']
    mean_disp = np.mean(floor_disp)
    rms_disp = np.sqrt(np.mean(floor_disp**2))
    peak_disp = np.max(np.abs(floor_disp)) # 峰值使用絕對值

    print(f"{label:<45} {mean_disp:<15.6f} {rms_disp:<15.6f} {peak_disp:<15.6f}")

print("\n--- 所有模擬和計算完成 ---")