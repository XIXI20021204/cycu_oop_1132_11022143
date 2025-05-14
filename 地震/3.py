import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 定義系統參數
W = 50       # 重量 [k]
g = 386.1    # 重力加速度 [in/s²]
m = W / g    # 質量 [k·s²/in]
k = 100      # 彈簧剛度 [k/in]
xi = 0.12    # 阻尼比
c = 2 * m * np.sqrt(k / m) * xi  # 阻尼常數 [k·s/in]
acc_g_peak = 0.25 * g  # 最大地面加速度 [in/s²]

# 2. 定義時間參數
dt = 0.01      # 時間步長 [s]
num_steps = 6    # 計算的時間步數
t = np.arange(0, num_steps * dt, dt)  # 時間向量 [s]

# 3. 定義地震加速度模型
# 使用正弦函數模擬地震加速度
acc_g = acc_g_peak * np.sin(np.pi * t / (num_steps * dt))  # 地面加速度 [in/s²]

# 4. 初始化位移、速度和加速度向量
x = np.zeros(num_steps)  # 位移 [in]
v = np.zeros(num_steps)  # 速度 [in/s]
a = np.zeros(num_steps)  # 加速度 [in/s²]
f_eff = np.zeros(num_steps) # 有效力 [k]

# 5. Wilson-theta 法參數 (theta = 1.4)
theta = 1.4
theta_dt_sq = theta / dt**2
theta_dt = theta / dt
a1 = m * theta_dt_sq + c * theta_dt
a2 = k
a3 = m * theta_dt_sq
a4 = c * theta_dt
a5 = m / (theta * dt**2)
a6 = c / (theta * dt)

# 6. 使用 Wilson-theta 法進行時間積分
for i in range(1, num_steps):
    # 6.1 計算有效力
    f_eff[i] = -m * acc_g[i]

    # 6.2 計算時間步內的力增量
    delta_f = f_eff[i] - (f_eff[i-1] - c * v[i-1] - k * x[i-1]) if i > 1 else f_eff[i] - ( - c * v[i-1] - k * x[i-1])

    # 6.3 使用 Wilson-theta 法更新加速度、速度和位移
    a[i] = (a5 * delta_f - a6 * v[i-1] - a[i-1]) / (1 + a5)
    v[i] = v[i-1] + dt * ((1 - 1/theta) * a[i-1] + (1/theta) * a[i])
    x[i] = x[i-1] + dt * v[i-1] + (dt**2 / 2) * ((1 - 2/theta) * a[i-1] + (2/theta) * a[i])

# 7. 繪製結果
# 設定中文字型
plt.rcParams['font.sans-serif'] = ['DFKai-SB']  # 使用標楷體
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 建立 2x2 的子圖格

# 7.1 繪製有效力
axs[0, 0].plot(t, f_eff, label=r'$F_{eff}(t)$', color='b')
axs[0, 0].set_ylabel("有效力 (k)")
axs[0, 0].legend()
axs[0, 0].grid()
axs[0, 0].set_xlabel("時間 (s)")  # 新增 x 軸標籤

# 7.2 繪製位移
axs[0, 1].plot(t, x, label=r'$x(t)$', color='g')
axs[0, 1].set_ylabel("位移 (in)")
axs[0, 1].legend()
axs[0, 1].grid()
axs[0, 1].set_xlabel("時間 (s)")  # 新增 x 軸標籤

# 7.3 繪製速度
axs[1, 0].plot(t, v, label=r'$\dot{x}(t)$', color='r')
axs[1, 0].set_ylabel("速度 (in/s)")
axs[1, 0].legend()
axs[1, 0].grid()
axs[1, 0].set_xlabel("時間 (s)")  # 新增 x 軸標籤

# 7.4 繪製加速度
axs[1, 1].plot(t, a, label=r'$\ddot{x}(t)$', color='m')
axs[1, 1].set_ylabel("加速度 (in/s²)")
axs[1, 1].set_xlabel("時間 (s)")  # 新增 x 軸標籤
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()

# 8. 將繪製的圖形儲存為 JPG 檔案
filename = "dynamic_response_plot.jpg"  # 設定檔名
plt.savefig(filename, format='jpg', dpi=300)  # 儲存為 JPG，設定解析度

# 9. 輸出完成訊息
print(f"圖形已儲存為 {filename}")
