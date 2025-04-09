import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

def plot_lognormal_cdf(mu, sigma, filename='lognormal_cdf.jpg'):
    """
    繪製並儲存對數常態分布的累積分布函數 (CDF)

    參數:
        mu (float): 對數常態分布的 μ
        sigma (float): 對數常態分布的 σ
        filename (str): 圖片儲存檔名（預設為 'lognormal_cdf.jpg'）
    """
    # 對應 scipy 的參數：s = σ, scale = exp(μ)
    s = sigma
    scale = np.exp(mu)

    # 定義 x 軸範圍
    x = np.linspace(0.01, 10, 500)

    # 計算 CDF
    cdf = lognorm.cdf(x, s, scale=scale)

    # 繪圖
    plt.figure(figsize=(8, 6))
    plt.plot(x, cdf, label='Log-normal CDF', color='blue')
    plt.title('Log-normal Cumulative Distribution Function')
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.legend()

    # 儲存圖檔
    plt.savefig(filename, format='jpg')
    plt.show()

# 使用範例
plot_lognormal_cdf(mu=1.5, sigma=0.4)
