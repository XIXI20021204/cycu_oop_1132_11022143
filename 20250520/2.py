import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('20250520/midterm_scores.csv')

subjects = ['Chinese', 'English', 'Math', 'History', 'Geography', 'Physics', 'Chemistry']

# 定義分數區間，从10开始
bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # 10个边界，9个区间

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']

num_subjects = len(subjects)
bar_width = 0.8 / num_subjects

# x 对应0到8的索引，表示9个区间
x = np.arange(len(bins) -1)

plt.figure(figsize=(14, 8))

# base_x_positions 现在从10开始
base_x_positions = x * 10 + 10 # 10, 20, ..., 90

for i, subject in enumerate(subjects):
    scores = df[subject]
    # 对分数进行分箱，只考虑10分及以上
    counts, _ = np.histogram(scores, bins=bins)
    
    plt.bar(base_x_positions + i * (bar_width * 10), counts, 
            width=bar_width * 10, 
            label=subject, color=colors[i], edgecolor='black')

plt.xlabel('Score')
plt.ylabel('Number of Students')
plt.title('Score Distribution by Subject')

# 新的刻度，从10到100
new_xticks = bins
plt.xticks(new_xticks)

plt.legend(title='Subjects')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlim(5, 105) # 调整X轴范围，从5开始，避免和10太近
plt.tight_layout()

output_path = 'C:/Users/User/Documents/GitHub/cycu_oop_1132_11022143/20250520/score_distribution_10_100_xaxis.png'
plt.savefig(output_path, dpi=300)
print(f"圖表已儲存至 {output_path}")

plt.show()