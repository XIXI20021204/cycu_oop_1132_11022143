import requests
import html
import pandas as pd
from bs4 import BeautifulSoup

# 設定要抓取的公車路線 ID
rid = "10417"
url = f"https://pda5284.gov.taipei/MQS/route.jsp?rid={rid}"

try:
    # 發送 GET 請求
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # 如果請求失敗，會觸發 Exception

    # 儲存 HTML 檔案
    with open("bus_route.html", "w", encoding="utf-8") as file:
        file.write(response.text)
    print("✅ 網頁已成功下載並儲存為 bus_route.html")

    # 讀取並解析 HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # 初始化 DataFrame 列表
    go_stops = []
    return_stops = []

    # 找到所有表格
    tables = soup.find_all("table")

    for table in tables:
        # 去程站點 (ttego1, ttego2)
        for tr in table.find_all("tr", class_=["ttego1", "ttego2"]):
            td = tr.find("td")
            if td:
                stop_name = html.unescape(td.text.strip())
                stop_link = td.find("a")["href"] if td.find("a") else None
                go_stops.append({"類型": "去程", "站點名稱": stop_name, "連結": stop_link})

        # 回程站點 (tteback1, tteback2)
        for tr in table.find_all("tr", class_=["tteback1", "tteback2"]):
            td = tr.find("td")
            if td:
                stop_name = html.unescape(td.text.strip())
                stop_link = td.find("a")["href"] if td.find("a") else None
                return_stops.append({"類型": "回程", "站點名稱": stop_name, "連結": stop_link})

    # 轉換為 DataFrame
    df_go = pd.DataFrame(go_stops)
    df_return = pd.DataFrame(return_stops)

    # 顯示 DataFrame
    print("\n🚏 **去程站點 DataFrame:**")
    print(df_go)

    print("\n🚏 **回程站點 DataFrame:**")
    print(df_return)

    # 儲存 CSV 檔案
    df_go.to_csv("去程站點.csv", index=False, encoding="utf-8")
    df_return.to_csv("回程站點.csv", index=False, encoding="utf-8")
    print("\n📂 已成功儲存 CSV 檔案：去程站點.csv, 回程站點.csv")

except requests.exceptions.RequestException as e:
    print(f"❌ 無法下載網頁: {e}")
