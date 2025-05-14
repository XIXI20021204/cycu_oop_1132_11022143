import numpy as np
import matplotlib.pyplot as plt

# Given parameters
W = 50  # Weight [k]
g = 386.1  # Gravitational acceleration [in/s²]
m = W / g  # Mass [k·s²/in]
k = 100  # Spring stiffness [k/in]
xi = 0.12  # Damping ratio
c = 2 * m * np.sqrt(k / m) * xi  # Damping constant [k·s/in]
xg_peak = 0.25 * g  # Peak ground acceleration [in/s²]

# Time parameters
dt = 0.01  # Time step [s]
num_steps = 6  # Compute the first six time steps
t = np.arange(0, num_steps * dt, dt)  # Time sequence

# Ground acceleration model (using a simple trigonometric approximation)
xg_t = xg_peak * np.sin(np.pi * t / (num_steps * dt))

# Initialize arrays
x_avg = np.zeros(num_steps)
x_lin = np.zeros(num_steps)
x_wilson = np.zeros(num_steps)
x_center = np.zeros(num_steps)
x_dot_avg = np.zeros(num_steps)
x_dot_lin = np.zeros(num_steps)
x_dot_wilson = np.zeros(num_steps)
x_dot_center = np.zeros(num_steps)
x_ddot_avg = np.zeros(num_steps)
x_ddot_lin = np.zeros(num_steps)
x_ddot_wilson = np.zeros(num_steps)
x_ddot_center = np.zeros(num_steps)
F_eff = np.zeros(num_steps)  # Renamed from F_dy

# Wilson θ method (θ = 1.4)
theta = 1.4
a1 = (theta / dt**2) * m + (theta / dt) * c
a2 = k
a3 = (theta / dt**2) * m
a4 = (theta / dt) * c
a5 = m / (theta * dt**2)
a6 = c / (theta * dt)

# Initialize the first values, crucial for the central difference method
x_center[0] = 0  # Initial displacement
x_dot_center[0] = 0  # Initial velocity
if num_steps > 1:
    x_center[1] = x_center[0] + dt * x_dot_center[0] + (dt**2 / 2) * (-xg_t[0] - (c/m)*x_dot_center[0] - (k/m)*x_center[0])
    x_ddot_center[0] = (-xg_t[0] - (c/m)*x_dot_center[0] - (k/m)*x_center[0])

# Loop to compute the first six time steps
for i in range(1, num_steps):
    F_eff[i] = -m * xg_t[i]  # Compute effective force
    F_eff[i] -= c * x_dot_avg[i-1] + k * x_avg[i-1]  # Dynamic reaction force

    # Average acceleration method
    x_ddot_avg[i] = F_eff[i] / m
    x_dot_avg[i] = x_dot_avg[i-1] + dt * (x_ddot_avg[i] + x_ddot_avg[i-1]) / 2
    x_avg[i] = x_avg[i-1] + dt * x_dot_avg[i-1] + (dt**2 / 4) * x_ddot_avg[i-1] + (dt**2 / 4) * x_ddot_avg[i]

    # Linear acceleration method
    x_ddot_lin[i] = F_eff[i] / m
    x_dot_lin[i] = x_dot_lin[i-1] + dt * ((x_ddot_lin[i] + x_ddot_lin[i-1]) / 2)
    x_lin[i] = x_lin[i-1] + dt * x_dot_lin[i-1] + (dt**2 / 2) * ((1/2) * x_ddot_lin[i-1] + (1/2) * x_ddot_lin[i])

    # Wilson θ method
    delta_F = F_eff[i] - F_eff[i-1]
    x_ddot_wilson[i] = (a5 * delta_F - a6 * x_dot_wilson[i-1] - x_wilson[i-1]) / (1 + a5)
    x_dot_wilson[i] = x_dot_wilson[i-1] + dt * ((1 - 1/theta) * x_ddot_wilson[i-1] + (1/theta) * x_ddot_wilson[i])
    x_wilson[i] = x_wilson[i-1] + dt * x_dot_wilson[i-1] + (dt**2 / 2) * ((1 - 2/theta) * x_ddot_wilson[i-1] + (2/theta) * x_ddot_wilson[i])

    # Central difference method
    if i > 1:
        x_ddot_center[i-1] = (F_eff[i-1] + m * xg_t[i-1] - c * x_dot_center[i-1] - k * x_center[i-1]) / m
        x_center[i] = (F_eff[i] + m * xg_t[i] + (m/dt**2)*x_center[i-2] - c*(x_center[i-1] - x_center[i-2])/(2*dt)) / (m/dt**2 + c/(2*dt) + k)
        x_dot_center[i] = (x_center[i] - x_center[i-1]) / dt

# Correct the last acceleration for the central difference method
if num_steps > 1:
    x_ddot_center[-1] = (F_eff[-1] + m * xg_t[-1] - c * x_dot_center[-1] - k * x_center[-1]) / m

# Plot results
fig, axs = plt.subplots(4, 1, figsize=(8, 12))

axs[0].plot(t, F_eff, label=r'$F_{eff}(t)$', color='b')  # Updated label
axs[0].set_ylabel("Effective Force (k)")
axs[0].legend()
axs[0].grid()

axs[1].plot(t, x_avg, label="Average Acceleration", color='g')
axs[1].plot(t, x_lin, label="Linear Acceleration", color='c')
axs[1].plot(t, x_wilson, label="Wilson θ", color='m')
axs[1].plot(t, x_center, label="Central Difference", linestyle="dashed", color='r')
axs[1].set_ylabel("Displacement (in)")
axs[1].legend()
axs[1].grid()

axs[2].plot(t, x_dot_avg, label="Average Acceleration", color='g')
axs[2].plot(t, x_dot_lin, label="Linear Acceleration", color='c')
axs[2].plot(t, x_dot_wilson, label="Wilson θ", color='m')
axs[2].plot(t, x_dot_center, label="Central Difference", linestyle="dashed", color='r')
axs[2].set_ylabel("Velocity (in/s)")
axs[2].legend()
axs[2].grid()

axs[3].plot(t, x_ddot_avg, label="Average Acceleration", color='g')
axs[3].plot(t, x_ddot_lin, label="Linear Acceleration", color='c')
axs[3].plot(t, x_ddot_wilson, label="Wilson θ", color='m')
axs[3].plot(t, x_ddot_center, label="Central Difference", linestyle="dashed", color='r')
axs[3].set_ylabel("Acceleration (in/s²)")
axs[3].set_xlabel("Time (s)")
axs[3].legend()
axs[3].grid()

plt.tight_layout()
plt.show()