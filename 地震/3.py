import numpy as np
import matplotlib.pyplot as plt

# 已知參數
W = 50
g = 386.1
m = W / g
k = 100
ξ = 0.12
c = 2 * m * np.sqrt(k / m) * ξ
ẍ_g_peak = 0.25 * g
t0_earthquake = 0.75

# 時間參數
dt = 0.01
num_steps = 600
t = np.arange(0, num_steps * dt, dt)

# 地震加速度模型
ẍ_g_t = np.zeros_like(t)
for i, time in enumerate(t):
    if time < t0_earthquake:
        ẍ_g_t[i] = 0
    elif time < 2 * t0_earthquake:
        ẍ_g_t[i] = -ẍ_g_peak
    else:
        ẍ_g_t[i] = 0

# 初始化數組
x = np.zeros(num_steps)
ẋ = np.zeros(num_steps)
ẍ = np.zeros(num_steps)
F_eff = np.zeros(num_steps)

# Wilson θ 法 (θ = 1.4)
theta = 1.4
ω = np.sqrt(k / m)

# 迴圈計算
for i in range(1, num_steps):
    F_eff[i] = -m * ẍ_g_t[i]

    x_j = x[i-1]
    ẋ_j = ẋ[i-1]
    ẍ_j = ẍ[i-1]
    F_j_theta = (1 - theta) * (-m * ẍ_g_t[i-1]) + theta * (-m * ẍ_g_t[i])

    ẍ_j_theta = (F_j_theta - c * (ẋ_j + dt/2 * ẍ_j) - k * (x_j + dt * ẋ_j + dt**2 / 6 * ẍ_j)) / (m + c * theta * dt / 2 + k * theta**2 * dt**2 / 6 )

    ẋ[i] = ẋ_j + dt/theta * (ẍ_j_theta - ẍ_j)
    x[i] = x_j + dt * ẋ_j + (dt**2 / (2 * theta)) * ẍ_j + (dt**2 / (2 * theta)) * ẍ_j_theta
    ẍ[i] = ẍ_j + dt/theta * (ẍ_j_theta - ẍ_j)

# 繪製結果
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

axs[0].plot(t, F_eff, label=r'$F_{eff}(t)$', color='b')
axs[0].set_ylabel("Effective Force (k)")
axs[0].legend()
axs[0].grid()

axs[1].plot(t, x, label=r'$x(t)$', color='g')
axs[1].set_ylabel("Displacement (in)")
axs[1].legend()
axs[1].grid()

axs[2].plot(t, ẋ, label=r'$\dot{x}(t)$', color='r')
axs[2].set_ylabel("Velocity (in/s)")
axs[2].legend()
axs[2].grid()

axs[3].plot(t, ẍ, label=r'$\ddot{x}(t)$', color='m')
axs[3].set_ylabel("Acceleration (in/s²)")
axs[3].set_xlabel("Time (s)")
axs[3].legend()
axs[3].grid()

plt.tight_layout()
plt.show()