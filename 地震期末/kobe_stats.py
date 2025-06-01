import numpy as np

# 讀取資料，跳過第一行標頭
data = np.loadtxt(r'c:\Users\a0965\OneDrive\文件\GitHub\cycu_oop_1132_11022143\地震期末\Kobe.txt', skiprows=1)
acc = data[:, 1]

mean_acc = np.mean(acc)
rms_acc = np.sqrt(np.mean(acc**2))
peak_acc = np.max(np.abs(acc))

print(f"平均值: {mean_acc:.6f} g")
print(f"均方根值(RMS): {rms_acc:.6f} g")
print(f"尖峰值: {peak_acc:.6f} g")
