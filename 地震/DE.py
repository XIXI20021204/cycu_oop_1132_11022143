import numpy as np
from scipy.integrate import odeint

# 載入數據 (請注意：這裡的路徑是硬編碼的，可能需要根據您的實際路徑進行修改)
data = np.loadtxt(r'C:\Users\a0965\Downloads\Northridge_NS.txt')
time = data[:, 0]
acc_g = data[:, 1]

# 結構參數
m = 290 / 386.4  # 質量 (kg)
ksi = 0.05         # 阻尼比
omega_n = 2 * np.pi * 0.57 # 自然頻率 (rad/s)
k = m * omega_n**2 # 勁度 (N/m)
c = 2 * m * omega_n * ksi # 阻尼係數 (Ns/m)

zeta = 0.05        # 另一種阻尼比
omega_EW = 2 * np.pi * 1.02 # EW方向自然頻率 (rad/s)
k_EW = m * omega_EW**2 # EW方向勁度
c_EW = 2 * m * omega_EW * zeta # EW方向阻尼

T = 1 / 0.57       # 週期 (s)
dt = time[1] - time[0] # 時間間隔
L_Columns = 1.68    # 柱長 (m)
L_Braces = 1.4384   # 斜撐長度 (m)

omega_n_NS = np.sqrt(k / m)
omega_n_EW = np.sqrt(k_EW / m)

# 初始條件
z_0 = [0, 0] # [位移, 速度] NS方向
z_EW_0 = [0, 0] # [位移, 速度] EW方向

# 地震加速度插值函數
def acc_g_interp(t, time, acc_g):
    """
    對地震加速度進行線性插值。
    """
    if t < time[0]:
        return acc_g[0]
    elif t > time[-1]:
        return acc_g[-1]
    else:
        i = np.where(time <= t)[0][-1]
        t0 = time[i]
        t1 = time[i+1]
        a0 = acc_g[i]
        a1 = acc_g[i+1]
        return a0 + (a1 - a0) * (t - t0) / (t1 - t0)

# 運動方程式 (NS方向)
def eq_of_motion(z, t, m, c, k, time, acc_g):
    """
    一自由度系統的運動方程式。
    z: [位移, 速度]
    t: 時間
    m: 質量
    c: 阻尼係數
    k: 勁度
    time: 時間序列
    acc_g: 地震加速度序列
    """
    x, x_dot = z
    acc_g_t = acc_g_interp(t, time, acc_g)
    x_ddot = - (c * x_dot + k * x) / m - acc_g_t
    return [x_dot, x_ddot]

# 運動方程式 (EW方向)
def eq_of_motion_EW(z_EW, t, m, c_EW, k_EW, time, acc_g):
    """
    EW方向的一自由度系統運動方程式。
    z_EW: [位移, 速度]
    t: 時間
    m: 質量
    c_EW: 阻尼係數 (EW方向)
    k_EW: 勁度 (EW方向)
    time: 時間序列
    acc_g: 地震加速度序列
    """
    x_EW, x_dot_EW = z_EW
    acc_g_t = acc_g_interp(t, time, acc_g)
    x_ddot_EW = - (c_EW * x_dot_EW + k_EW * x_EW) / m - acc_g_t
    return [x_dot_EW, x_ddot_EW]

# 求解運動方程式
solution_NS = odeint(eq_of_motion, z_0, time, args=(m, c, k, time, acc_g))
x_NS, x_dot_NS = solution_NS[:, 0], solution_NS[:, 1]
x_ddot_NS = np.array([eq_of_motion([x, x_dot], t, m, c, k, time, acc_g)[1] for x, x_dot, t in zip(x_NS, x_dot_NS, time)])

solution_EW = odeint(eq_of_motion_EW, z_EW_0, time, args=(m, c_EW, k_EW, time, acc_g))
x_EW, x_dot_EW = solution_EW[:, 0], solution_EW[:, 1]
x_ddot_EW = np.array([eq_of_motion_EW([x, x_dot], t, m, c_EW, k_EW, time, acc_g)[1] for x, x_dot, t in zip(x_EW, x_dot_EW, time)])

# 計算最大反應
x_max_NS = np.max(np.abs(x_NS))
x_dot_max_NS = np.max(np.abs(x_dot_NS))
x_ddot_max_NS = np.max(np.abs(x_ddot_NS))

x_max_EW = np.max(np.abs(x_EW))
x_dot_max_EW = np.max(np.abs(x_dot_EW))
x_ddot_max_EW = np.max(np.abs(x_ddot_EW))

# 計算最大層剪力 (假設層剪力與結構總水平力成正比，簡化為 m * 最大加速度)
V_max_NS = m * x_ddot_max_NS
V_max_EW = m * x_ddot_max_EW

# 計算最大柱軸力 (簡化模型，假設柱子承受所有垂直載重，不考慮地震引起的額外軸力)
# 這部分需要更詳細的結構模型和載重資訊才能準確計算
M_max_NS = V_max_NS * L_Columns / 2 # 假設彎矩在柱底最大，簡化計算
M_max_EW = V_max_EW * L_Columns / 2 # 假設彎矩在柱底最大，簡化計算

# 打印 NS 方向最大反應
print("NS 方向最大反應:")
print(f"|x(t)|_max = {x_max_NS:.4f} m")
print(f"|x'(t)|_max = {x_dot_max_NS:.4f} m/s")
print(f"|x''(t)|_max = {x_ddot_max_NS:.4f} m/s^2")
print(f"|V(t)|_max = {V_max_NS:.4f} N")
print(f"|M(t)|_max = {M_max_NS:.4f} Nm")

# 打印 EW 方向最大反應
print("\nEW 方向最大反應:")
print(f"|x(t)|_max = {x_max_EW:.4f} m")
print(f"|x'(t)|_max = {x_dot_max_EW:.4f} m/s")
print(f"|x''(t)|_max = {x_ddot_max_EW:.4f} m/s^2")
print(f"|V(t)|_max = {V_max_EW:.4f} N")
print(f"|M(t)|_max = {M_max_EW:.4f} Nm")