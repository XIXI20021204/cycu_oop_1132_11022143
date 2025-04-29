import sqlite3
import csv

# 指定資料庫檔案名和 CSV 檔案名
database_file = 'your_database_file.db'  # 替換為您的 SQLite 資料庫檔案名稱
csv_file = 'output_table.csv'           # 替換為您希望輸出的 CSV 檔案名稱
table_name = 'your_table_name'          # 替換為您的資料表名稱

try:
    # 連接到 SQLite 資料庫
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # 執行查詢以選取資料
    cursor.execute(f"SELECT * FROM {table_name}")
    data = cursor.fetchall()

    # 取得欄位名稱
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [info[1] for info in cursor.fetchall()]

    # 開啟 CSV 檔案以寫入
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 寫入標頭
        writer.writerow(columns)

        # 寫入資料
        writer.writerows(data)

    print(f"成功將表格 '{table_name}' 匯出到 '{csv_file}'")

except sqlite3.Error as e:
    print(f"SQLite 錯誤: {e}")
except Exception as e:
    print(f"發生錯誤: {e}")
finally:
    if conn:
        conn.close()