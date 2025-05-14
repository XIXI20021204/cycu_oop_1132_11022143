import numpy as np
import matplotlib.pyplot as plt

# ===== 參數設定 =====
W = 50  # 重量 [k]
g = 386.1 # 重力加速度 [in/s²]
m = W / g # 質量 [k·s²/in]
k = 100 # 彈簧剛度 [k/in]
xi = 0.12 # 阻尼比
omega_n = np.sqrt(k / m) # 自然頻率 [rad/s]
c = 2 * m * omega_n * xi # 阻尼係數 [k·s/in]
ag_peak = 0.25 * g # 峰值地面加速度
dt = 0.01 # 時間步長 [s]
n_steps = 6 # 前 6 個時間步
t = np.arange(0, n_steps * dt, dt) # 時間向量
ag_t = ag_peak * np.sin(np.pi * t / (n_steps * dt)) # 假設地面加速度

# ===== 初始化陣列 =====
def init_arr(n):
    return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n) # x, x_dot, x_ddot, F_eff

# ===== Wilson-θ 法 =====
def wilson_theta(x, xd, xdd, F_eff, m, c, k, dt, theta, n, ag_t):
    beta = 1 + xi * omega_n * theta * dt + (1/6) * omega_n**2 * theta**2 * dt**2
    gamma = xi * omega_n * theta * dt + (1/3) * omega_n**2 * theta**2 * dt**2
    Aw = np.array([
        [-omega_n**2 * theta**2 * dt**2 / (6 * beta), (dt - xi * omega_n * theta**2 * dt**2 / 3 - omega_n**2 * theta**2 * dt**3 / 6) / beta, (1 - theta - gamma) * theta**2 * dt**2 / (6 * beta)],
        [-omega_n**2 * theta * dt / beta, (1 - 2 * xi * omega_n * theta * dt - omega_n**2 * theta * dt**2) / beta, (beta * theta - theta - gamma) * dt / (beta * theta)],
        [-omega_n**2 * theta / beta, (-2 * xi * omega_n * theta - omega_n**2 * theta * dt) / beta, (-gamma - theta + 1) / beta]
    ])
    Bw = np.array([[theta**2 * dt**2 / (6 * m * beta)], [theta * dt / (2 * m * beta)], [theta / (m * beta)]])
    Xj = np.zeros((3, 1))
    Xj1 = np.zeros((3, 1))
    for i in range(1, n):
        F_eff[i] = -m * ag_t[i]
        Xj[:, 0] = [x[i-1], xd[i-1], xdd[i-1]]
        Xj1 = Aw @ Xj + Bw * F_eff[i]
        x[i], xd[i], xdd[i] = Xj1[:, 0]
    return x, xd, xdd, F_eff

# ===== 中央差分法 =====
def central_diff(x, xd, xdd, F_eff, m, c, k, dt, n, ag_t):
    a = m / dt**2 - c / (2 * dt)
    b = k - 2 * m / dt**2
    k_hat = m / dt**2 + c / (2 * dt)
    x[0] = xd[0] = 0
    xdd[0] = (F_eff[0] - c * xd[0] - k * x[0]) / m
    x_prev = x[0] - dt * xd[0] + (dt**2 / 2) * xdd[0]
    for i in range(1, n):
        F_eff[i] = -m * ag_t[i]
        F_hat = F_eff[i] - a * x_prev - b * x[i-1]
        x_next = F_hat / k_hat
        x_prev = x[i-1]
        x[i] = x_next
        if i < n - 1:
            xd[i] = (x[i+1] - x_prev) / (2 * dt)
        xdd[i] = (x_next - 2 * x[i-1] + x_prev) / dt**2
    return x, xd, xdd, F_eff

# ===== 平均加速度法 =====
def avg_accel(x, xd, xdd, F_eff, m, c, k, dt, n, ag_t):
    for i in range(1, n):
        F_eff[i] = -m * ag_t[i]
        xdd[i] = (F_eff[i] - c * xd[i-1] - k * x[i-1]) / m
        xd[i] = xd[i-1] + dt * (xdd[i] + xdd[i-1]) / 2
        x[i] = x[i-1] + dt * xd[i-1] + (dt**2 / 4) * xdd[i-1] + (dt**2 / 4) * xdd[i]
    return x, xd, xdd, F_eff

# ===== 線性加速度法 =====
def linear_accel(x, xd, xdd, F_eff, m, c, k, dt, n, ag_t):
    for i in range(1, n):
        F_eff[i] = -m * ag_t[i]
        xdd[i] = (F_eff[i] - c * xd[i-1] - k * x[i-1]) / m
        xd[i] = xd[i-1] + dt * xdd[i-1] + (dt / 2) * (xdd[i] - xdd[i-1])
        x[i] = x[i-1] + dt * xd[i-1] + (dt**2 / 6) * (xdd[i] - xdd[i-1]) + (dt**2 / 2) * xdd[i-1]
    return x, xd, xdd, F_eff

# ===== 主要計算 =====
xw, xdw, xddw, Feffw = init_arr(n_steps)
xc, xdc, xddc, Feffc = init_arr(n_steps)
xavg, xdavg, xddavg, Feffavg = init_arr(n_steps)
xlin, xdlin, xddlin, Fefflin = init_arr(n_steps)

theta = 1.4

xw, xdw, xddw, Feffw = wilson_theta(xw, xdw, xddw, Feffw, m, c, k, dt, theta, n_steps, ag_t)
xc, xdc, xddc, Feffc = central_diff(xc, xdc, xddc, Feffc, m, c, k, dt, n_steps, ag_t)
xavg, xdavg, xddavg, Feffavg = avg_accel(xavg, xdavg, xddavg, Feffavg, m, c, k, dt, n_steps, ag_t)
xlin, xdlin, xddlin, Fefflin = linear_accel(xlin, xdlin, xddlin, Fefflin, m, c, k, dt, n_steps, ag_t)

# ===== 繪製結果 =====
fig, axs = plt.subplots(4, 1, figsize=(10, 14))
results = [
    (t, xw, xc, xavg, xlin, 'Displacement x(t) [in]'),
    (t, xdw, xdc, xdavg, xdlin, 'Velocity ẋ(t) [in/s]'),
    (t, xddw, xddc, xddavg, xddlin, 'Acceleration ẍ(t) [in/s²]'),
    (t, Feffw, Feffc, Feffavg, Fefflin, 'Effective Force F_eff(t) [k]')
]
labels = ['Wilson θ', 'Central Difference', 'Average Acceleration', 'Linear Acceleration']
linestyles = ['-', '--', '-.', ':']

for i, (time, w, c, avg, lin, ylabel) in enumerate(results):
    axs[i].plot(time, w, label=labels[0], linestyle=linestyles[0])
    axs[i].plot(time, c, label=labels[1], linestyle=linestyles[1])
    axs[i].plot(time, avg, label=labels[2], linestyle=linestyles[2])
    axs[i].plot(time, lin, label=labels[3], linestyle=linestyles[3])
    axs[i].set_ylabel(ylabel)
    if i == 3:
        axs[i].set_xlabel('Time t [s]')
    axs[i].legend()
    axs[i].grid()

plt.tight_layout()
plt.savefig('numerical_methods_comparison_simplified.jpg')
plt.show()