import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 設定中文字型
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

# 讀取資料，跳過第一行標頭
data = np.loadtxt(r'c:\Users\a0965\OneDrive\文件\GitHub\cycu_oop_1132_11022143\地震期末\Kobe.txt', skiprows=1)
time = data[:, 0]
acc = data[:, 1]

# 計算統計量
mean_acc = np.mean(acc)
rms_acc = np.sqrt(np.mean(acc**2))
peak_acc = np.max(np.abs(acc))

print(f"平均值: {mean_acc:.6f} g")
print(f"均方根值(RMS): {rms_acc:.6f} g")
print(f"尖峰值: {peak_acc:.6f} g")

plt.figure(figsize=(12, 5))
plt.plot(time, acc, label='Kobe Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (g)')
plt.title('日本神戶地震地表加速歷時圖')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('kobe_acc_time_history.jpg', format='jpg', dpi=300)
plt.show()