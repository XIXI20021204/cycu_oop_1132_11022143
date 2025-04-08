import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# 定義參數
mu = 1.5  # μ
sigma = 0.4  # σ

# 計算對數常態分布的 s 和 scale
s = sigma
scale = np.exp(mu)

# 定義 x 軸範圍
x = np.linspace(0.01, 10, 500)

# 計算累積分布函數 (CDF)
cdf = lognorm.cdf(x, s, scale=scale)

# 繪製圖形
plt.figure(figsize=(8, 6))
plt.plot(x, cdf, label='Log-normal CDF', color='blue')
plt.title('Log-normal Cumulative Distribution Function')
plt.xlabel('x')
plt.ylabel('CDF')
plt.grid(True)
plt.legend()

# 儲存為 JPG 檔案
plt.savefig('lognormal_cdf.jpg', format='jpg')
plt.show()