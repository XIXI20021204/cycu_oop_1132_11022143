import numpy as np
import matplotlib.pyplot as plt

# 讀取 Northridge 地震資料（時間, 加速度(g)）
data = np.loadtxt('/mnt/data/Northridge_NS.txt')
time = data[:, 0]
acc_g = data[:, 1]
acc_mps2 = acc_g * 9.81  # 轉換為 m/s²

# 計算最大地震加速度
pga = np.max(np.abs(acc_mps2))

# 畫圖
plt.figure(figsize=(10, 4))
plt.plot(time, acc_mps2, label='地震加速度 $\\ddot{x}_g(t)$ [m/s²]', color='blue')
plt.axhline(pga, color='red', linestyle='--', label=f'Max = {pga:.2f} m/s²')
plt.axhline(-pga, color='red', linestyle='--')
plt.xlabel('時間 (秒)')
plt.ylabel('加速度 (m/s²)')
plt.title('地震加速度歷時圖 (Northridge NS Direction)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
