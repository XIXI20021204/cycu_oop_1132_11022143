import numpy as np
import matplotlib.pyplot as plt

# 1. 定義系統參數 (與先前相同，但使用 SI 單位)
W = 50  # 重量 [k]
g = 9.81  # 重力加速度 [m/s²]
m = W / g  # 質量 [kg]
k = 100 * 175.127  # 彈簧剛度 [N/m]  (1 k/in = 175.127 N/m)
xi = 0.12  # 阻尼比
c = 2 * m * np.sqrt(k / m) * xi  # 阻尼常數 [N·s/m]
acc_g_peak = 0.25 * g  # 最大地面加速度 [m/s²]

# 2. 定義時間參數
dt = 0.01  # 時間步長 [s]
num_steps = 6  # 計算的時間步數
t = np.arange(0, num_steps * dt, dt)  # 時間向量 [s]

# 3. 定義地震加速度模型 (SI 單位)
acc_g = acc_g_peak * np.sin(np.pi * t / (num_steps * dt))  # 地面加速度 [m/s²]

# 4. 初始化位移、速度和加速度向量
x = np.zeros(num_steps)  # 位移 [m]
v = np.zeros(num_steps)  # 速度 [m/s]
a = np.zeros(num_steps)  # 加速度 [m/s²]
f_eff = np.zeros(num_steps)  # 有效力 [N]

# 5. 中心差分法參數
delta_t = dt  # 時間步長
m_val = m
c_val = c
k_val = k

# 使用文件中的公式 4.28 計算 a, b, k_hat
a = (m_val / delta_t**2) - (c_val / (2 * delta_t))
b = k_val - (2 * m_val / delta_t**2)
k_hat = (m_val / delta_t**2) + (c_val / (2 * delta_t))

# 6. 初始條件 (需要以 SI 單位提供)
x0 = 0.0  # 假設初始位移為 0 [m]
v0 = 0.0  # 假設初始速度為 0 [m/s]

# 使用文件中的公式 4.28 計算 x_minus_1 和 a0
a0 = (-c_val * v0 - k_val * x0 + m_val * acc_g[0]) / m_val  # 從 m*a + c*v + k*x = F => a = (F - c*v - k*x) / m
x_minus_1 = x0 - delta_t * v0 + (delta_t**2 / 2) * a0

x[0] = x0
v[0] = v0
a[0] = a0

# 7. 使用中心差分法進行時間積分
for i in range(1, num_steps):
    # 7.1 計算有效力
    f_eff[i] = -m_val * acc_g[i]  # 有效力 [N]

    # 7.2 使用公式 4.25 計算 x_i+1
    f_hat_j = f_eff[i] - a * x[i - 1] - b * x[i]
    x[i] = f_hat_j / k_hat

    # 7.3 使用公式 4.26 計算 v_i
    v[i] = (x[i] - x[i - 2]) / (2 * delta_t)  #  v_i = (x_i+1 - x_i-1) / 2*dt  ，程式的i 從1開始，所以要改成 x[i] - x[i-2]

    # 7.4 使用公式 4.27 計算 a_i
    a[i] = (x[i + 1] - 2 * x[i] + x[i - 1]) / delta_t**2 if i < num_steps -1 else (f_eff[i] - c_val * v[i] - k_val * x[i]) / m_val # a[i] = (x_i+1 - 2x_i + x_i-1) / dt^2

# 8. 繪製結果
plt.rcParams['font.sans-serif'] = ['DFKai-SB']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# 8.1 繪製有效力
axs[0, 0].plot(t, f_eff, label=r'$F_{eff}(t)$', color='b')
axs[0, 0].set_ylabel("有效力 (N)")
axs[0, 0].legend()
axs[0, 0].grid()
axs[0, 0].set_xlabel("時間 (s)")

# 8.2 繪製位移
axs[0, 1].plot(t, x, label=r'$x(t)$', color='g')
axs[0, 1].set_ylabel("位移 (m)")
axs[0, 1].legend()
axs[0, 1].grid()
axs[0, 1].set_xlabel("時間 (s)")

# 8.3 繪製速度
axs[1, 0].plot(t, v, label=r'$\dot{x}(t)$', color='r')
axs[1, 0].set_ylabel("速度 (m/s)")
axs[1, 0].legend()
axs[1, 0].grid()
axs[1, 0].set_xlabel("時間 (s)")

# 8.4 繪製加速度
axs[1, 1].plot(t, a, label=r'$\ddot{x}(t)$', color='m')
axs[1, 1].set_ylabel("加速度 (m/s²)")
axs[1, 1].set_xlabel("時間 (s)")
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()
plt.show()

# 9. 輸出完成訊息
print("使用中心差分法計算完成，並顯示結果圖。")
