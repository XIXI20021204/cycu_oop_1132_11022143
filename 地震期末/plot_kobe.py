import numpy as np
import matplotlib.pyplot as plt

# 讀取資料，跳過第一行標頭
data = np.loadtxt('Kobe.txt', skiprows=1)
time = data[:, 0]
acc = data[:, 1]

plt.figure(figsize=(12, 5))
plt.plot(time, acc, label='Kobe Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (g)')
plt.title('Kobe Earthquake Acceleration Time History')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
