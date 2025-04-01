import requests
import html
import pandas as pd
from bs4 import BeautifulSoup
import os

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

    # 使用 BeautifulSoup 解析 HTML
    soup = BeautifulSoup(content, "html.parser")

    # 找到所有表格
    tables = soup.find_all("table")

    # 初始化 DataFrame 列表
    dataframes = []

    # 創建資料夾來儲存車站 HTML
    os.makedirs("stations_html", exist_ok=True)

    # 遍歷表格
    for table in tables:
        rows = []
        # 找到所有符合條件的 tr 標籤
        for tr in table.find_all("tr", class_=["ttego1", "ttego2"]):
            # 提取站點名稱和連結
            td = tr.find("td")
            if td:
                stop_name = html.unescape(td.text.strip())  # 解碼站點名稱
                stop_link = td.find("a")["href"] if td.find("a") else None
                rows.append({"站點名稱": stop_name, "連結": stop_link})

                # 如果有連結，下載對應的 HTML
                if stop_link:
                    stop_url = f"https://pda5284.gov.taipei/MQS/{stop_link}"
                    try:
                        stop_response = requests.get(stop_url)
                        stop_response.raise_for_status()
                        # 儲存為 HTML 檔案
                        stop_filename = f"stations_html/{stop_name}.html"
                        with open(stop_filename, "w", encoding="utf-8") as stop_file:
                            stop_file.write(stop_response.text)
                        print(f"已下載並儲存 {stop_name} 的 HTML：{stop_filename}")
                    except requests.exceptions.RequestException as e:
                        print(f"無法下載 {stop_name} 的 HTML：{e}")

        # 如果有資料，轉換為 DataFrame
        if rows:
            df = pd.DataFrame(rows)
            dataframes.append(df)

    # 將兩個 DataFrame 分別命名
    if len(dataframes) >= 2:
        df1, df2 = dataframes[0], dataframes[1]
        print("第一個 DataFrame:")
        print(df1)
        print("\n第二個 DataFrame:")
        print(df2)
    else:
        print("未找到足夠的表格資料。")
else:
    print(f"無法下載網頁，HTTP 狀態碼: {response.status_code}")