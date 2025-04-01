import requests
import html
import pandas as pd
from bs4 import BeautifulSoup

# 設定 Pandas 顯示所有列
pd.set_option('display.max_rows', None)  # 顯示所有列
pd.set_option('display.max_columns', None)  # 顯示所有欄位
pd.set_option('display.width', 1000)  # 設定輸出寬度

url = '''https://pda5284.gov.taipei/MQS/route.jsp?rid=10417'''

# 發送 GET 請求
response = requests.get(url)

# 確保請求成功
if response.status_code == 200:
    # 將內容寫入 bus1.html
    with open("bus1.html", "w", encoding="utf-8") as file:
        file.write(response.text)
    print("網頁已成功下載並儲存為 bus1.html")

    # 重新讀取並解碼 HTML
    with open("bus1.html", "r", encoding="utf-8") as file:
        content = file.read()
        decoded_content = html.unescape(content)  # 解碼 HTML 實體
        print(decoded_content)  # 顯示解碼後的內容

    # 使用 BeautifulSoup 解析 HTML
    soup = BeautifulSoup(content, "html.parser")

    # 找到所有表格
    tables = soup.find_all("table")

    # 初始化資料列表
    all_rows = []

    # 遍歷表格
    for table in tables:
        # 找到所有符合條件的 tr 標籤
        for tr in table.find_all("tr", class_=["ttego1", "ttego2"]):
            # 提取站點名稱和連結
            td = tr.find("td")
            if td:
                stop_name = html.unescape(td.text.strip())  # 解碼站點名稱
                stop_link = td.find("a")["href"] if td.find("a") else None
                # 根據 class 區分去程與回程
                if "ttego1" in tr["class"]:
                    stop_type = "去程站點名稱"
                elif "ttego2" in tr["class"]:
                    stop_type = "回程站點名稱"
                else:
                    stop_type = "未知"
                all_rows.append({"類型": stop_type, "站點名稱": stop_name, "連結": stop_link})

    # 將資料分成兩個 DataFrame
    df_go = pd.DataFrame([row for row in all_rows if row["類型"] == "去程站點名稱"])
    df_return = pd.DataFrame([row for row in all_rows if row["類型"] == "回程站點名稱"])

    # 輸出結果
    print("去程 DataFrame:")
    print(df_go)
    print("\n回程 DataFrame:")
    print(df_return)
else:
    print(f"無法下載網頁，HTTP 狀態碼: {response.status_code}")