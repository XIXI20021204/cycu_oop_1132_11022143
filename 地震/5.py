import numpy as np
import matplotlib.pyplot as plt

# ===== Parameter Setup =====
W = 50  # Weight [k]
g = 386.1  # Gravitational acceleration [in/s²]
m = W / g  # Mass [k·s²/in]
k = 100  # Spring stiffness [k/in]
ζ = 0.12  # Damping ratio
c = 2 * m * np.sqrt(k / m) * ζ  # Damping coefficient [k·s/in]
ẍ_g_peak = 0.25 * g  # Peak ground acceleration

dt = 0.01  # Time step [s]
t0 = 0.75  # Duration of pulse [s]
t_full = np.arange(0, 2 * t0 + dt, dt)  # Full time vector
t_plot = np.arange(0, 0.05 + dt, dt) # Time vector for plotting
indices_plot = np.where(t_full <= 0.05)[0]
t = t_full[indices_plot]
ẍ_g_t = np.where(t <= t0, ẍ_g_peak, -ẍ_g_peak)

# Initialize arrays
def init_arrays(num_steps):
    return (
        np.zeros(num_steps),  # displacement x
        np.zeros(num_steps),  # velocity ẋ
        np.zeros(num_steps),  # acceleration ẍ
        np.zeros(num_steps),  # effective force F_eff
    )

num_steps = len(t)

# ===== Wilson-θ Method =====
theta = 1.4
ω_n = np.sqrt(k / m)
x_w, ẋ_w, ẍ_w, F_eff_w = init_arrays(num_steps)
x_w[0] = 0
ẋ_w[0] = 0
ẍ_w[0] = (-c * ẋ_w[0] - k * x_w[0] - m * ẍ_g_t[0]) / m
for i in range(num_steps - 1):
    F_eff_w[i] = -m * ẍ_g_t[i]
    K_hat = k + (c * theta * dt / 2) + (m * 4 / (theta * dt)**2)
    ΔF_hat = (F_eff_w[i+1] - F_eff_w[i]) + m * (4 / (theta * dt) * ẋ_w[i] + 2 * ẍ_w[i]) + c * (theta * dt / 2 * ẍ_w[i])
    Δx = ΔF_hat / K_hat
    ẍ_w[i+1] = (6 / (theta * dt)**2) * Δx - (6 / (theta * dt)) * ẋ_w[i] + (1 - 3 / theta) * ẍ_w[i]
    ẋ_w[i+1] = ẋ_w[i] + (theta * dt / 2) * (ẍ_w[i] + ẍ_w[i+1])
    x_w[i+1] = x_w[i] + theta * dt * ẋ_w[i] + (theta * dt)**2 / 6 * (2 * ẍ_w[i] + ẍ_w[i+1])

# ===== Central Difference Method =====
x_c, ẋ_c, ẍ_c, F_eff_c = init_arrays(num_steps)
x_c[0] = 0
ẋ_c[0] = 0
if dt > 0:
    x_c[1] = x_c[0] + dt * ẋ_c[0] + (dt**2 / 2) * ((-c * ẋ_c[0] - k * x_c[0] - m * ẍ_g_t[0]) / m)
else:
    x_c[1] = 0
for i in range(1, num_steps - 1):
    F_eff_c[i] = -m * ẍ_g_t[i]
    ẍ_c[i] = (F_eff_c[i] - c * (x_c[i] - x_c[i-2]) / (2 * dt) - k * x_c[i-1]) / m
    x_c[i+1] = 2 * x_c[i] - x_c[i-1] + dt**2 * ẍ_c[i]
    ẋ_c[i] = (x_c[i+1] - x_c[i-1]) / (2 * dt)
if num_steps > 2:
    ẋ_c[-1] = (x_c[-1] - x_c[-3]) / (2 * dt)
    ẍ_c[-1] = (F_eff_c[-1] - c * (x_c[-1] - x_c[-3]) / (2 * dt) - k * x_c[-2]) / m

# ===== Average Acceleration Method =====
x_avg, ẋ_avg, ẍ_avg, F_eff_avg = init_arrays(num_steps)
x_avg[0] = 0
ẋ_avg[0] = 0
ẍ_avg[0] = (-c * ẋ_avg[0] - k * x_avg[0] - m * ẍ_g_t[0]) / m
for i in range(num_steps - 1):
    F_eff_avg[i] = -m * ẍ_g_t[i]
    K_eff = k + c / (2 * dt) + m / dt**2
    ΔF = F_eff_avg[i+1] - F_eff_avg[i] + m * (1 / dt * ẋ_avg[i] + 0.5 * ẍ_avg[i]) + c * 0.5 * ẍ_avg[i] * dt
    Δx = ΔF / K_eff
    Δẍ = Δx * 2 / dt**2 - ẍ_avg[i]
    ẋ_avg[i+1] = ẋ_avg[i] + dt / 2 * (ẍ_avg[i] + ẍ_avg[i+1])
    x_avg[i+1] = x_avg[i] + dt * ẋ_avg[i] + dt**2 / 4 * (ẍ_avg[i] + ẍ_avg[i+1])
    ẍ_avg[i+1] = ẍ_avg[i] + Δẍ

# ===== Linear Acceleration Method =====
x_lin, ẋ_lin, ẍ_lin, F_eff_lin = init_arrays(num_steps)
x_lin[0] = 0
ẋ_lin[0] = 0
ẍ_lin[0] = (-c * ẋ_lin[0] - k * x_lin[0] - m * ẍ_g_t[0]) / m
for i in range(num_steps - 1):
    F_eff_lin[i] = -m * ẍ_g_t[i]
    K_eff = k + c / dt + m / dt**2
    ΔF = F_eff_lin[i+1] - F_eff_lin[i] + m * (1 / dt * ẋ_lin[i] + 0.5 * ẍ_lin[i]) + c * 0.5 * ẍ_lin[i] * dt
    Δx = ΔF / K_eff
    Δẍ = Δx * 2 / dt**2 - ẍ_lin[i]
    ẋ_lin[i+1] = ẋ_lin[i] + dt / 2 * (ẍ_lin[i] + ẍ_lin[i+1])
    x_lin[i+1] = x_lin[i] + dt * ẋ_lin[i] + dt**2 / 6 * (2 * ẍ_lin[i] + ẍ_lin[i+1])
    ẍ_lin[i+1] = ẍ_lin[i] + Δẍ

# ===== Plot Results =====
fig, axs = plt.subplots(4, 1, figsize=(10, 14))

# Displacement
axs[0].plot(t, x_w, label='Wilson θ')
axs[0].set_ylabel('Displacement x(t) [in]')
axs[0].set_xlabel('Time t [s]')
axs[0].set_xlim(0, 0.05)
axs[0].legend()
axs[0].grid()

# Velocity
axs[1].plot(t, ẋ_w, label='Wilson θ')
axs[1].set_ylabel('Velocity ẋ(t) [in/s]')
axs[1].set_xlabel('Time t [s]')
axs[1].set_xlim(0, 0.05)
axs[1].legend()
axs[1].grid()

# Acceleration
axs[2].plot(t, ẍ_w, label='Wilson θ')
axs[2].set_ylabel('Acceleration ẍ(t) [in/s²]')
axs[2].set_xlabel('Time t [s]')
axs[2].set_xlim(0, 0.05)
axs[2].legend()
axs[2].grid()

# Effective Force (-m * acceleration)
axs[3].plot(t, -m * ẍ_w, label='Wilson θ')
axs[3].set_ylabel('Effective Force -m*ẍ(t) [k]')
axs[3].set_xlabel('Time t [s]')
axs[3].set_xlim(0, 0.05)
axs[3].legend()
axs[3].grid()

plt.tight_layout()
plt.show()

# ===== Plot Results - Separate Figures =====
fig_separate, axs_separate = plt.subplots(4, 1, figsize=(10, 14))

# Displacement
axs_separate[0].plot(t, x_w, label='Wilson θ')
axs_separate[0].set_ylabel('Displacement x(t) [in]')
axs_separate[0].set_xlabel('Time t [s]')
axs_separate[0].set_xlim(0, 0.05)
axs_separate[0].legend()
axs_separate[0].grid()

axs_separate[1].plot(t[1:], x_c[1:], label='Central Difference')
axs_separate[1].set_ylabel('Displacement x(t) [in]')
axs_separate[1].set_xlabel('Time t [s]')
axs_separate[1].set_xlim(0, 0.05)
axs_separate[1].legend()
axs_separate[1].grid()

axs_separate[2].plot(t, x_avg, label='Average Acceleration')
axs_separate[2].set_ylabel('Displacement x(t) [in]')
axs_separate[2].set_xlabel('Time t [s]')
axs_separate[2].set_xlim(0, 0.05)
axs_separate[2].legend()
axs_separate[2].grid()

axs_separate[3].plot(t, x_lin, label='Linear Acceleration')
axs_separate[3].set_ylabel('Displacement x(t) [in]')
axs_separate[3].set_xlabel('Time t [s]')
axs_separate[3].set_xlim(0, 0.05)
axs_separate[3].legend()
axs_separate[3].grid()

plt.tight_layout()
plt.show()

fig_separate_v, axs_separate_v = plt.subplots(4, 1, figsize=(10, 14))

# Velocity
axs_separate_v[0].plot(t, ẋ_w, label='Wilson θ')
axs_separate_v[0].set_ylabel('Velocity ẋ(t) [in/s]')
axs_separate_v[0].set_xlabel('Time t [s]')
axs_separate_v[0].set_xlim(0, 0.05)
axs_separate_v[0].legend()
axs_separate_v[0].grid()

axs_separate_v[1].plot(t[1:-1], ẋ_c[1:-1], label='Central Difference')
axs_separate_v[1].set_ylabel('Velocity ẋ(t) [in/s]')
axs_separate_v[1].set_xlabel('Time t [s]')
axs_separate_v[1].set_xlim(0, 0.05)
axs_separate_v[1].legend()
axs_separate_v[1].grid()

axs_separate_v[2].plot(t, ẋ_avg, label='Average Acceleration')
axs_separate_v[2].set_ylabel('Velocity ẋ(t) [in/s]')
axs_separate_v[2].set_xlabel('Time t [s]')
axs_separate_v[2].set_xlim(0, 0.05)
axs_separate_v[2].legend()
axs_separate_v[2].grid()

axs_separate_v[3].plot(t, ẋ_lin, label='Linear Acceleration')
axs_separate_v[3].set_ylabel('Velocity ẋ(t) [in/s]')
axs_separate_v[3].set_xlabel('Time t [s]')
axs_separate_v[3].set_xlim(0, 0.05)
axs_separate_v[3].legend()
axs_separate_v[3].grid()

plt.tight_layout()
plt.show()

fig_separate_a, axs_separate_a = plt.subplots(4, 1, figsize=(10, 14))

# Acceleration
axs_separate_a[0].plot(t, ẍ_w, label='Wilson θ')
axs_separate_a[0].set_ylabel('Acceleration ẍ(t) [in/s²]')
axs_separate_a[0].set_xlabel('Time t [s]')
axs_separate_a[0].set_xlim(0, 0.05)
axs_separate_a[0].legend()
axs_separate_a[0].grid()

axs_separate_a[1].plot(t[1:-1], ẍ_c[1:-1], label='Central Difference')
axs_separate_a[1].set_ylabel('Acceleration ẍ(t) [in/s²]')
axs_separate_a[1].set_xlabel('Time t [s]')
axs_separate_a[1].set_xlim(0, 0.05)
axs_separate_a[1].legend()
axs_separate_a[1].grid()

axs_separate_a[2].plot(t, ẍ_avg, label='Average Acceleration')
axs_separate_a[2].set_ylabel('Acceleration ẍ(t) [in/s²]')
axs_separate_a[2].set_xlabel('Time t [s]')
axs_separate_a[2].set_xlim(0, 0.05)
axs_separate_a[2].legend()
axs_separate_a[2].grid()

axs_separate_a[3].plot(t, ẍ_lin, label='Linear Acceleration')
axs_separate_a[3].set_ylabel('Acceleration ẍ(t) [in/s²]')
axs_separate_a[3].set_xlabel('Time t [s]')
axs_separate_a[3].set_xlim(0, 0.05)
axs_separate_a[3].legend()
axs_separate_a[3].grid()

plt.tight_layout()
plt.show()

fig_separate_f, axs_separate_f = plt.subplots(4, 1, figsize=(10, 14))

# Effective Force (-m * acceleration)
axs_separate_f[0].plot(t, -m * ẍ_w, label='Wilson θ')
axs_separate_f[0].set_ylabel('Effective Force -m*ẍ(t) [k]')
axs_separate_f[0].set_xlabel('Time t [s]')
axs_separate_f[0].set_xlim(0, 0.05)
axs_separate_f[0].legend()