import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 讀取地震加速度資料 (假設是兩欄：時間(s), 加速度(g))
data = np.loadtxt('Northridge_NS.txt')
time = data[:, 0]
acc = data[:, 1] * 9.81  # 轉換為 m/s²

# 定義參數
zeta = 0.05  # 阻尼比 5%
periods = np.logspace(-1, 1, 100)  # T from 0.1 to 10 s
omega = 2 * np.pi / periods

# 儲存最大反應
Sd = []  # 最大位移
Sv = []  # 最大速度
Sa = []  # 最大加速度

# 計算 SDOF 反應譜
for w in omega:
    k = 1.0
    m = 1.0
    c = 2 * zeta * m * w
    def sdof_eq(u, t):
        x, xdot = u
        interp_acc = np.interp(t, time, acc)
        xddot = (-c * xdot - k * x) / m - interp_acc
        return [xdot, xddot]

    u0 = [0.0, 0.0]
    sol = odeint(sdof_eq, u0, time)
    x = sol[:, 0]
    xdot = sol[:, 1]
    xddot = -2 * zeta * w * xdot - w**2 * x

    Sd.append(np.max(np.abs(x)))
    Sv.append(np.max(np.abs(xdot)))
    Sa.append(np.max(np.abs(xddot)))

# 畫圖
plt.figure(figsize=(10, 6))
plt.loglog(periods, Sa, label='加速度譜 (Sa)', color='r')
plt.loglog(periods, Sv, label='速度譜 (Sv)', color='g')
plt.loglog(periods, Sd, label='位移譜 (Sd)', color='b')
plt.xlabel('週期 T (秒)')
plt.ylabel('反應 (最大值)')
plt.title('反應譜（Northridge 地震）ζ = 0.05')
plt.grid(which='both')
plt.legend()
plt.tight_layout()
plt.show()
