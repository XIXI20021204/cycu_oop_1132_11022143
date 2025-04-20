import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_normal_pdf(mu, sigma):
    # 建立 x 軸資料
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    # 使用 norm.pdf 計算 y 軸機率密度
    y = norm.pdf(x, mu, sigma)
    
    # 畫圖
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f'μ={mu}, σ={sigma}')
    plt.title('Normal Distribution PDF')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.legend()
    
    # 儲存圖檔
    plt.savefig('normal_pdf.jpg')
    plt.close()

# 範例呼叫
plot_normal_pdf(mu=0, sigma=1.0)
