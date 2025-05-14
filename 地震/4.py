import numpy as np
import matplotlib.pyplot as plt

# ===== Parameter Setup =====
W = 50   # Weight [k]
g = 386.1  # Gravitational acceleration [in/s²]
m = W / g  # Mass [k·s²/in]
k = 100  # Spring stiffness [k/in]
xi = 0.12  # Damping ratio
omega_n = np.sqrt(k / m)   # Natural frequency [rad/s]
c = 2 * m * omega_n * xi   # Damping coefficient [k·s/in]
ẍ_g_peak = 0.25 * g   # Peak ground acceleration
dt = 0.01  # Time step [s]
num_steps = 6   # First 6 time steps
t = np.arange(0, num_steps * dt, dt)   # Time vector
ẍ_g_t = ẍ_g_peak * np.sin(np.pi * t / (num_steps * dt))   # Assumed ground acceleration

# ===== Initialize arrays =====
def init_arrays(num_steps):
    return (
        np.zeros(num_steps),   # displacement x
        np.zeros(num_steps),   # velocity ẋ
        np.zeros(num_steps),   # acceleration ẍ
        np.zeros(num_steps),   # effective force F_eff
    )

# ===== Wilson-θ Method =====
def wilson_theta_method(x, x_dot, x_ddot, F_eff, m, c, k, dt, theta, num_steps, ẍ_g_t):
    beta = 1 + xi * omega_n * theta * dt + (1/6) * omega_n**2 * theta**2 * dt**2
    gamma = xi * omega_n * theta * dt + (1/3) * omega_n**2 * theta**2 * dt**2
    Aw = np.zeros((3, 3))
    Bw = np.zeros((3, 1))

    Aw[0, 0] = (-omega_n**2 * theta**2 * dt**2 / (6 * beta))
    Aw[0, 1] = (dt - xi * omega_n * theta**2 * dt**2 / 3 - omega_n**2 * theta**2 * dt**3 / 6) / beta
    Aw[0, 2] = (1 - theta - gamma) * theta**2 * dt**2 / (6 * beta)
    Aw[1, 0] = (-omega_n**2 * theta * dt / beta)
    Aw[1, 1] = (1 - 2 * xi * omega_n * theta * dt - omega_n**2 * theta * dt**2) / beta
    Aw[1, 2] = (beta * theta - theta - gamma) * dt / (beta * theta)
    Aw[2, 0] = -omega_n**2 * theta / beta
    Aw[2, 1] = (-2 * xi * omega_n * theta - omega_n**2 * theta * dt) / beta
    Aw[2, 2] = (-gamma - theta + 1) / beta

    Bw[0, 0] = theta**2 * dt**2 / (6 * m * beta)
    Bw[1, 0] = theta * dt / (2 * m * beta)
    Bw[2, 0] = theta / (m * beta)

    Xj = np.zeros((3, 1))
    Xj1 = np.zeros((3, 1))

    for i in range(1, num_steps):
        F_eff[i] = -m * ẍ_g_t[i]
        Xj[0, 0] = x[i-1]
        Xj[1, 0] = x_dot[i-1]
        Xj[2, 0] = x_ddot[i-1]

        Xj1 = np.dot(Aw, Xj) + Bw * F_eff[i]

        x[i] = Xj1[0, 0]
        x_dot[i] = Xj1[1, 0]
        x_ddot[i] = Xj1[2, 0]

    return x, x_dot, x_ddot, F_eff

# ===== Central Difference Method =====
def central_difference_method(x, x_dot, x_ddot, F_eff, m, c, k, dt, num_steps, ẍ_g_t):
    a = m / dt**2 - c / (2 * dt)
    b = k - 2 * m / dt**2
    k_hat = m / dt**2 + c / (2 * dt)

    x[0] = 0
    x_dot[0] = 0
    x_ddot[0] = (F_eff[0] - c * x_dot[0] - k * x[0]) / m
    x_prev = x[0] - dt * x_dot[0] + (dt**2 / 2) * x_ddot[0]  # x_{-1}

    for i in range(1, num_steps):
        F_eff[i] = -m * ẍ_g_t[i]
        F_hat = F_eff[i] - a * x_prev - b * x[i-1]
        x_next = F_hat / k_hat

        x_prev = x[i-1]
        x[i] = x_next
        if i < num_steps - 1:
            x_dot[i] = (x[i+1] - x_prev) / (2 * dt)
        x_ddot[i] = (x_next - 2 * x[i-1] + x_prev) / dt**2
    return x, x_dot, x_ddot, F_eff

# ===== Average Acceleration Method =====
def average_acceleration_method(x, x_dot, x_ddot, F_eff, m, c, k, dt, num_steps, ẍ_g_t):
    for i in range(1, num_steps):
        F_eff[i] = -m * ẍ_g_t[i]
        x_ddot[i] = (F_eff[i] - c * x_dot[i-1] - k * x[i-1]) / m
        x_dot[i] = x_dot[i-1] + dt * (x_ddot[i] + x_ddot[i-1]) / 2
        x[i] = x[i-1] + dt * x_dot[i-1] + (dt**2 / 4) * x_ddot[i-1] + (dt**2 / 4) * x_ddot[i]
    return x, x_dot, x_ddot, F_eff

# ===== Linear Acceleration Method =====
def linear_acceleration_method(x, x_dot, x_ddot, F_eff, m, c, k, dt, num_steps, ẍ_g_t):
    for i in range(1, num_steps):
        F_eff[i] = -m * ẍ_g_t[i]
        x_ddot[i] = (F_eff[i] - c * x_dot[i-1] - k * x[i-1]) / m
        x_dot[i] = x_dot[i-1] + dt * x_ddot[i-1] + (dt / 2) * (x_ddot[i] - x_ddot[i-1])
        x[i] = x[i-1] + dt * x_dot[i-1] + (dt**2 / 6) * (x_ddot[i] - x_ddot[i-1]) + (dt**2 / 2) * x_ddot[i-1]
    return x, x_dot, x_ddot, F_eff

# ===== Main Calculation =====

# Initialize arrays
x_w, x_dot_w, x_ddot_w, F_eff_w = init_arrays(num_steps)
x_c, x_dot_c, x_ddot_c, F_eff_c = init_arrays(num_steps)
x_avg, x_dot_avg, x_ddot_avg, F_eff_avg = init_arrays(num_steps)
x_lin, x_dot_lin, x_ddot_lin, F_eff_lin = init_arrays(num_steps)

theta = 1.4

# Perform calculations
x_w, x_dot_w, x_ddot_w, F_eff_w = wilson_theta_method(x_w, x_dot_w, x_ddot_w, F_eff_w, m, c, k, dt, theta, num_steps, ẍ_g_t)
x_c, x_dot_c, x_ddot_c, F_eff_c = central_difference_method(x_c, x_dot_c, x_ddot_c, F_eff_c, m, c, k, dt, num_steps, ẍ_g_t)
x_avg, x_dot_avg, x_ddot_avg, F_eff_avg = average_acceleration_method(x_avg, x_dot_avg, x_ddot_avg, F_eff_avg, m, c, k, dt, num_steps, ẍ_g_t)
x_lin, x_dot_lin, x_ddot_lin, F_eff_lin = linear_acceleration_method(x_lin, x_dot_lin, x_ddot_lin, F_eff_lin, m, c, k, dt, num_steps, ẍ_g_t)


# ===== Plot Results =====
fig, axs = plt.subplots(4, 1, figsize=(10, 14))

# Displacement
axs[0].plot(t, x_w, label='Wilson θ')
axs[0].plot(t, x_c, '--', label='Central Difference')
axs[0].plot(t, x_avg, '-.', label='Average Acceleration')
axs[0].plot(t, x_lin, ':', label='Linear Acceleration')
axs[0].set_ylabel('Displacement x(t) [in]')
axs[0].legend()
axs[0].grid()

# Velocity
axs[1].plot(t, x_dot_w, label='Wilson θ')
axs[1].plot(t, x_dot_c, '--', label='Central Difference')
axs[1].plot(t, x_dot_avg, '-.', label='Average Acceleration')
axs[1].plot(t, x_dot_lin, ':', label='Linear Acceleration')
axs[1].set_ylabel('Velocity ẋ(t) [in/s]')
axs[1].legend()
axs[1].grid()

# Acceleration
axs[2].plot(t, x_ddot_w, label='Wilson θ')
axs[2].plot(t, x_ddot_c, '--', label='Central Difference')
axs[2].plot(t, x_ddot_avg, '-.', label='Average Acceleration')
axs[2].plot(t, x_ddot_lin, ':', label='Linear Acceleration')
axs[2].set_ylabel('Acceleration ẍ(t) [in/s²]')
axs[2].legend()
axs[2].grid()

# Effective Force
axs[3].plot(t, F_eff_w, label='Wilson θ')
axs[3].plot(t, F_eff_c, '--', label='Central Difference')
axs[3].plot(t, F_eff_avg, '-.', label='Average Acceleration')
axs[3].plot(t, F_eff_lin, ':', label='Linear Acceleration')
axs[3].set_ylabel('Effective Force F_eff(t) [k]')
axs[3].set_xlabel('Time t [s]')
axs[3].legend()
axs[3].grid()

plt.tight_layout()
plt.savefig('numerical_methods_comparison.jpg') # 在此行儲存為 JPG 檔案
plt.show()