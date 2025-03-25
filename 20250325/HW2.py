import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV 檔案
file_path = r'c:\Users\User\Documents\GitHub\cycu_oop_1132_11022143\20250325\gold.csv'
df = pd.read_csv(file_path)

# 選取需要的資料
dates = df['資料日期']
cash_buy = df['現金'].iloc[:, 0]  # 本行買入的現金
cash_sell = df['現金'].iloc[:, 1]  # 本行賣出的現金

# 繪製折線圖
plt.figure(figsize=(10, 5))
plt.plot(dates, cash_buy, label='本行買入現金', marker='o')
plt.plot(dates, cash_sell, label='本行賣出現金', marker='o')

# 設定圖表標題和標籤
plt.title('現金匯率折線圖')
plt.xlabel('資料日期')
plt.ylabel('現金匯率')
plt.xticks(rotation=45)
plt.legend()

# 顯示圖表
plt.tight_layout()
plt.show()