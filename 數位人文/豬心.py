import jieba
import re
import pandas as pd

# 檔案路徑
file_path = 'fish.txt'

# 讀取檔案內容
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
except FileNotFoundError:
    print(f"檔案 {file_path} 不存在，請確認檔案名稱或路徑是否正確。")
    exit()

# 使用 Jieba 進行斷詞
words = jieba.lcut(text)

# 過濾條件：只保留長度大於1的中文字詞，排除標點、英文、數字等
filtered_words = [
    word for word in words 
    if len(word) > 1 and 
       re.match(r'^[\u4e00-\u9fa5]+$', word)
]

# 建立 DataFrame：每個詞放在第一列不同欄
df = pd.DataFrame([filtered_words])

# 儲存成 Excel
output_file = '詞語清單.xlsx'
df.to_excel(output_file, index=False, header=False)
print(f"詞語清單已儲存為 '{output_file}'")