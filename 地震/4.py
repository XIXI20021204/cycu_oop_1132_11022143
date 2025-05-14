import numpy as np
import matplotlib.pyplot as plt

# 已知參數
W = 50   # 重量 [k]
g = 386.1   # 重力加速度 [in/s²]
m = W / g   # 質量 [k·s²/in]
k = 100   # 彈簧剛度 [k/in]
ξ = 0.12   # 阻尼比
c = 2 * m * np.sqrt(k / m) * ξ   # 阻尼常數 [k·s/in]
ẍ_g_peak = 0.25 * g   # 最大地面加速度 [in/s²]

# 時間參數
dt = 0.01   # 時間步長 [s]
num_steps = 6   # 計算前六個時間步
t = np.arange(0, num_steps * dt, dt)   # 時間序列

# 地震加速度模型（使用簡單的三角函數近似）
ẍ_g_t = ẍ_g_peak * np.sin(np.pi * t / (num_steps * dt))

# 初始化數組
x = np.zeros(num_steps)
ẋ = np.zeros(num_steps)
ẍ = np.zeros(num_steps)
F_eff = np.zeros(num_steps) # 將 F_dy 更名為 F_eff

# Wilson θ 法（θ = 1.4）
theta = 1.4
a1 = (theta / dt**2) * m + (theta / dt) * c
a2 = k
a3 = (theta / dt**2) * m
a4 = (theta / dt) * c
a5 = m / (theta * dt**2)
a6 = c / (theta * dt)

# 迴圈計算前六個時間步
for i in range(1, num_steps):
    F_eff[i] = -m * ẍ_g_t[i]   # 計算有效力 (已更名)
    F_dy_temp = F_eff[i] - c * ẋ[i-1] - k * x[i-1]  # 計算動力反應力 (使用臨時變數)

    # Wilson θ 法時間步進行數值積分
    delta_F = F_dy_temp - (F_eff[i-1] - c * ẋ[i-2] - k * x[i-2] if i > 1 else 0) # 使用臨時變數計算 delta_F
    ẍ[i] = (a5 * delta_F - a6 * ẋ[i-1] - ẍ[i-1]) / (1 + a5)
    ẋ[i] = ẋ[i-1] + dt * ((1 - 1/theta) * ẍ[i-1] + (1/theta) * ẍ[i])
    x[i] = x[i-1] + dt * ẋ[i-1] + (dt**2 / 2) * ((1 - 2/theta) * ẍ[i-1] + (2/theta) * ẍ[i])

# 繪製結果
fig, axs = plt.subplots(4, 1, figsize=(8, 10))

axs[0].plot(t, F_eff, label=r'$F_{eff}(t)$', color='b') # 繪製 F_eff
axs[0].set_ylabel("有效力 (k)")
axs[0].legend()
axs[0].grid()

axs[1].plot(t, x, label=r'$x(t)$', color='g')
axs[1].set_ylabel("位移 (in)")
axs[1].legend()
axs[1].grid()

axs[2].plot(t, ẋ, label=r'$\dot{x}(t)$', color='r')
axs[2].set_ylabel("速度 (in/s)")
axs[2].legend()
axs[2].grid()

axs[3].plot(t, ẍ, label=r'$\ddot{x}(t)$', color='m')
axs[3].set_ylabel("加速度 (in/s²)")
axs[3].set_xlabel("時間 (s)")
axs[3].legend()
axs[3].grid()

plt.tight_layout()
plt.show()