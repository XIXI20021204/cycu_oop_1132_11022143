import requests
from bs4 import BeautifulSoup
import html
import re  # 用於正則表達式匹配
import time

# 設定 User-Agent 避免被封鎖
HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_bus_stations(rid: str):
    """獲取公車去程站點名稱"""
    url = f"https://pda5284.gov.taipei/MQS/route.jsp?rid={rid}"
    
    response = requests.get(url, headers=HEADERS, timeout=10)
    if response.status_code != 200:
        print(f"❌ 無法連接公車網站，狀態碼：{response.status_code}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 尋找「去程 (往xxx)」的表格
    target_header = soup.find('td', string=re.compile(r'去程'))
    if not target_header:
        print("❌ 找不到去程標題")
        return []

    # 找到最近的表格
    table = target_header.find_next('table')

    # 提取所有車站名稱
    stations = []
    for row in table.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) >= 1:
            station = cells[0].text.strip().replace('...', '')  # 去掉多餘符號
            stations.append(station)

    return stations

def get_real_time_data(rid: str):
    """爬取臺北 5284 公車即時資訊，回傳車站與到站狀態"""
    url = f'https://pda5284.gov.taipei/MQS/route.jsp?rid={rid}'

    response = requests.get(url, headers=HEADERS, timeout=10)
    if response.status_code != 200:
        print(f"❌ 無法獲取即時數據，狀態碼：{response.status_code}")
        return {}

    soup = BeautifulSoup(response.text, "html.parser")

    # 初始化站點即時資訊
    stop_data = {}

    # 找到所有的站點表格
    tables = soup.find_all("table")

    for table in tables:
        for tr in table.find_all("tr", class_=["ttego1", "ttego2", "tteback1", "tteback2"]):
            td_list = tr.find_all("td")

            if len(td_list) >= 2:
                stop_name = html.unescape(td_list[0].text.strip())  # 站點名稱
                arrival_info = td_list[1].text.strip()  # 到站時間資訊

                # 嘗試抓取車牌號碼（可能存在於 <font> 標籤內）
                vehicle_number = td_list[1].find("font")
                vehicle_number = vehicle_number.text.strip() if vehicle_number else "無"

                stop_data[stop_name] = {
                    "預計到達": arrival_info,
                    "車牌號碼": vehicle_number
                }

    return stop_data

if __name__ == "__main__":
    rid = "10417"  # 公車路線 ID

    # 取得所有站點
    stations = get_bus_stations(rid)

    if not stations:
        print("⚠️ 找不到站點資料，請檢查公車路線。")
    else:
        print("\n🚏 **公車站點列表**")
        print("\n".join(stations))

    # 持續更新即時動態
    while True:
        print("\n📡 取得即時到站資訊中...")
        real_time_data = get_real_time_data(rid)

        if not real_time_data:
            print("⚠️ 無法獲取即時數據，請稍後再試。")
        else:
            print("\n🚏 **10417 公車站點即時動態**")
            print("=" * 40)
            for stop_name in stations:
                if stop_name in real_time_data:
                    data = real_time_data[stop_name]
                    print(f"🔹 **站點名稱：{stop_name}**")
                    print(f"   - ⏳ 預計到達：{data['預計到達']}")
                    print(f"   - 🚌 車牌號碼：{data['車牌號碼']}")
                    print("-" * 40)
                else:
                    print(f"🔹 **站點名稱：{stop_name}**（❌ 無即時資訊）")

        # 每 30 秒更新一次
        time.sleep(30)
