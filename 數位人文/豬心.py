import jieba
import re
import pandas as pd

# 讀取檔案內容（可以更換為 '豬心.txt'）
with open('羊肉湯.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# jieba 斷詞
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
df.to_excel('詞語清單.xlsx', index=False, header=False)
