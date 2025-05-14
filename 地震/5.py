import numpy as np
import matplotlib.pyplot as plt

# Given parameters
W = 50  # Weight [k]
g = 386.1  # Gravitational acceleration [in/s²]
m = W / g  # Mass [k·s²/in]
k = 100  # Spring stiffness [k/in]
ξ = 0.12  # Damping ratio
c = 2 * m * np.sqrt(k / m) * ξ  # Damping constant [k·s/in]
ẍ_g_peak = 0.25 * g  # Peak ground acceleration [in/s²]

# Time parameters
dt = 0.01  # Time step [s]
num_steps = 6  # Compute the first six time steps
t = np.arange(0, num_steps * dt, dt)  # Time sequence

# Ground acceleration model (using a simple trigonometric approximation)
ẍ_g_t = ẍ_g_peak * np.sin(np.pi * t / (num_steps * dt))

# Initialize arrays
x_avg = np.zeros(num_steps)
x_lin = np.zeros(num_steps)
x_wilson = np.zeros(num_steps)
x_center = np.zeros(num_steps)
ẋ_avg = np.zeros(num_steps)
ẋ_lin = np.zeros(num_steps)
ẋ_wilson = np.zeros(num_steps)
ẋ_center = np.zeros(num_steps)
ẍ_avg = np.zeros(num_steps)
ẍ_lin = np.zeros(num_steps)
ẍ_wilson = np.zeros(num_steps)
ẍ_center = np.zeros(num_steps)
F_eff = np.zeros(num_steps)  # Renamed from F_dy

# Wilson θ method (θ = 1.4)
theta = 1.4
a1 = (theta / dt**2) * m + (theta / dt) * c
a2 = k
a3 = (theta / dt**2) * m
a4 = (theta / dt) * c
a5 = m / (theta * dt**2)
a6 = c / (theta * dt)

# Loop to compute the first six time steps
for i in range(1, num_steps):
    F_eff[i] = -m * ẍ_g_t[i]  # Compute effective force
    F_eff[i] -= c * ẋ_avg[i-1] + k * x_avg[i-1]  # Dynamic reaction force

    # Average acceleration method
    ẍ_avg[i] = F_eff[i] / m
    ẋ_avg[i] = ẋ_avg[i-1] + dt * (ẍ_avg[i] + ẍ_avg[i-1]) / 2
    x_avg[i] = x_avg[i-1] + dt * ẋ_avg[i-1] + (dt**2 / 4) * ẍ_avg[i-1]+ (dt**2 / 4) * ẍ_avg[i]

    # Linear acceleration method
    ẍ_lin[i] = F_eff[i] / m
    ẋ_lin[i] = ẋ_lin[i-1] + dt * ẍ_lin[i-1]
    x_lin[i] = x_lin[i-1] + dt * ẋ_lin[i-1] + (dt**2 / 4) * ẍ_lin[i-1]+ (dt**2 / 4) * ẍ_lin[i]

    # Wilson θ 方法
    delta_F = F_eff[i] - F_eff[i-1]  # 计算有效力的增量

    # 计算 delta_F_theta (考虑了时间步内的力变化)
    delta_F_theta = delta_F + theta * m * xg_t[i] - m * xg_t[i-1]

    # 求解 delta_a_wilson (加速度的增量)
    delta_a_wilson = delta_F_theta / (m + theta * dt * c + (theta * dt)**2 * k / 2)  # 分母是修正后的刚度

    a_wilson[i] = a_wilson[i-1] + delta_a_wilson  # 更新加速度
    v_wilson[i] = v_wilson[i-1] + dt * ((1 - 1/theta) * a_wilson[i-1] + (1/theta) * a_wilson[i])  # 更新速度
    x_wilson[i] = x_wilson[i-1] + dt * v_wilson[i-1] + (dt**2 / 2) * ((1 - 2/theta) * a_wilson[i-1] + (2/theta) * a_wilson[i])  # 更新位移

    # 中心差分法
    if i > 1:
        a_center[i] = F_eff[i] / m  # 计算加速度
        x_center[i] = 2 * x_center[i-1] - x_center[i-2] + (dt**2) * a_center[i-1]  # 计算位移
        v_center[i] = (x_center[i] - x_center[i-2]) / (2 * dt)  # 计算速度
# Plot results
fig, axs = plt.subplots(4, 1, figsize=(8, 12))

axs[0].plot(t, F_eff, label=r'$F_{eff}(t)$', color='b')  # Updated label
axs[0].set_ylabel("Effective Force (k)")
axs[0].legend()
axs[0].grid()

axs[1].plot(t, x_avg, label="Average Acceleration Method", color='g')
axs[1].plot(t, x_lin, label="Linear Acceleration Method", color='c')
axs[1].plot(t, x_wilson, label="Wilson θ Method", color='m')
axs[1].plot(t, x_center, label="Central Difference Method", linestyle="dashed", color='r')
axs[1].set_ylabel("Displacement (in)")
axs[1].legend()
axs[1].grid()

axs[2].plot(t, ẋ_avg, label="Average Acceleration Method", color='g')
axs[2].plot(t, ẋ_lin, label="Linear Acceleration Method", color='c')
axs[2].plot(t, ẋ_wilson, label="Wilson θ Method", color='m')
axs[2].plot(t, ẋ_center, label="Central Difference Method", linestyle="dashed", color='r')
axs[2].set_ylabel("Velocity (in/s)")
axs[2].legend()
axs[2].grid()

axs[3].plot(t, ẍ_avg, label="Average Acceleration Method", color='g')
axs[3].plot(t, ẍ_lin, label="Linear Acceleration Method", color='c')
axs[3].plot(t, ẍ_wilson, label="Wilson θ Method", color='m')
axs[3].plot(t, ẍ_center, label="Central Difference Method", linestyle="dashed", color='r')
axs[3].set_ylabel("Acceleration (in/s²)")
axs[3].set_xlabel("Time (s)")
axs[3].legend()
axs[3].grid()

plt.tight_layout()
plt.show()