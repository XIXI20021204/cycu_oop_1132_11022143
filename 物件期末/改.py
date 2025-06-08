import folium
import random # 依然用於模擬公車位置和時間，因為這需要實時數據
import time
import webbrowser
import pandas as pd # 引入 pandas 處理 DataFrame
import re
import os # 引入 os 處理路徑

# --- 引入 Playwright 相關的庫 ---
from playwright.sync_api import sync_playwright

# --- 引入之前 Selenium 相關的庫 (不再用於站牌抓取，但如果 main 函數的路線列表抓取仍用 Selenium 則保留) ---
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from webdriver_manager.chrome import ChromeDriverManager


# --- 新增 BusRouteInfo 類別 (來自編號 2 的程式碼) ---
class BusRouteInfo:
    def __init__(self, routeid: str, direction: str = 'go'):
        self.rid = routeid
        self.content = None
        self.url = f'https://ebus.gov.taipei/Route/StopsOfRoute?routeid={routeid}'
        self.dataframe = None # To store parsed bus stop data

        if direction not in ['go', 'come']:
            raise ValueError("Direction must be 'go' or 'come'")

        self.direction = direction

        self._fetch_content()
    
    def _fetch_content(self):
        """
        Fetches the webpage content using Playwright.
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(self.url, timeout=60000) # Increased timeout for page loading
                
                if self.direction == 'come':
                    # Wait for the button to be visible and enabled before clicking
                    page.wait_for_selector('a.stationlist-come-go-gray.stationlist-come', state='visible', timeout=10000)
                    page.click('a.stationlist-come-go-gray.stationlist-come')
                    # Give a short delay after clicking to ensure content updates
                    time.sleep(2) 

                # Wait for a specific part of the content to be loaded, e.g., the first station name
                # This is more robust than a fixed timeout
                page.wait_for_selector('span.auto-list-stationlist-place', timeout=10000)
                
                self.content = page.content()
            except Exception as e:
                print(f"Error fetching content for route {self.rid}, direction {self.direction}: {e}")
                self.content = None # Set content to None if fetching fails
            finally:
                browser.close()

    def parse_route_info(self) -> pd.DataFrame:
        """
        Parses the fetched HTML content to extract bus stop data.
        Returns a DataFrame containing all stop details, including latitude and longitude.
        """
        if self.content is None:
            return pd.DataFrame() # Return empty DataFrame if content is not available

        pattern = re.compile(
            r'<li>.*?<span class="auto-list-stationlist-position[^>]*?">(.*?)</span>\s*'
            r'<span class="auto-list-stationlist-number">\s*(\d+)</span>\s*'
            r'<span class="auto-list-stationlist-place">(.*?)</span>.*?'
            r'<input[^>]+name="item\.UniStopId"[^>]+value="(\d+)"[^>]*>.*?'
            r'<input[^>]+name="item\.Latitude"[^>]+value="([\d\.]+)"[^>]*>.*?'
            r'<input[^>]+name="item\.Longitude"[^>]+value="([\d\.]+)"[^>]*>',
            re.DOTALL
        )

        matches = pattern.findall(self.content)
        if not matches:
            return pd.DataFrame() # Return empty DataFrame if no matches

        bus_stops_data = [m for m in matches]
        self.dataframe = pd.DataFrame(
            bus_stops_data,
            columns=["arrival_info", "stop_number", "stop_name", "stop_id", "latitude", "longitude"]
        )

        self.dataframe["stop_number"] = pd.to_numeric(self.dataframe["stop_number"], errors='coerce')
        self.dataframe["stop_id"] = pd.to_numeric(self.dataframe["stop_id"], errors='coerce')
        self.dataframe["latitude"] = pd.to_numeric(self.dataframe["latitude"], errors='coerce')
        self.dataframe["longitude"] = pd.to_numeric(self.dataframe["longitude"], errors='coerce')

        self.dataframe["direction"] = self.direction
        self.dataframe["route_id"] = self.rid

        return self.dataframe

# --- 修改 get_bus_route_stops_from_ebus 函式，使用 Playwright ---
def get_bus_route_stops_from_ebus(route_id, bus_name):
    """
    從台北市公車動態資訊系統抓取指定路線的站牌名稱、緯度和經度。
    使用 Playwright 進行網頁爬取。
    返回一個站牌列表，每個元素是一個字典，包含 'name', 'lat', 'lon'。
    """
    print(f"正在從 ebus.gov.taipei 獲取路線 '{bus_name}' ({route_id}) 的站牌數據 (使用 Playwright)...")
    
    stops_with_coords = []
    
    try:
        # 只抓取去程數據，因為 Folium 繪製路線時通常只需要一個方向的完整路徑
        # 如果需要繪製返程，可以呼叫兩次或根據需求調整
        route_info = BusRouteInfo(route_id, direction="go") 
        df_stops = route_info.parse_route_info()
        
        if not df_stops.empty:
            # 將 DataFrame 轉換為所需的字典列表格式
            stops_with_coords = df_stops[['stop_name', 'latitude', 'longitude']].rename(
                columns={'stop_name': 'name', 'latitude': 'lat', 'longitude': 'lon'}
            ).to_dict(orient='records')
            
            print(f"路線 '{bus_name}' 的站牌數據獲取完成。共 {len(stops_with_coords)} 站。")
        else:
            print(f"未找到路線 {bus_name} 的任何站牌資訊。")

    except Exception as e:
        print(f"[錯誤] 獲取路線 {bus_name} 站牌數據失敗：{e}")
        stops_with_coords = [] # 返回空列表或處理錯誤
            
    return stops_with_coords

# --- display_bus_route_on_map 函式保持不變 ---
def display_bus_route_on_map(route_name, stops_data, bus_location=None, estimated_times=None):
    """
    將公車路線、站牌、預估時間和公車位置顯示在地圖上。
    stops_data: 列表，每個元素是一個字典，包含 'name', 'lat', 'lon'
    bus_location: 字典，包含 'lat', 'lon'，可選
    estimated_times: 字典，鍵為站牌名稱，值為預估時間，可選
    """
    if not stops_data:
        print(f"沒有路線 '{route_name}' 的站牌數據可顯示。")
        return

    print(f"正在為路線 '{route_name}' 生成地圖...")

    # 以第一個站牌為中心創建地圖
    map_center = [stops_data[0]["lat"], stops_data[0]["lon"]]
    m = folium.Map(location=map_center, zoom_start=14) # 稍微放大一點

    # 添加站牌標記和彈出視窗
    for stop in stops_data:
        stop_name = stop["name"]
        coords = [stop["lat"], stop["lon"]]
        
        # 獲取預估時間，如果沒有則顯示「未知」
        est_time_text = estimated_times.get(stop_name, "未知") if estimated_times else "未知"
        popup_html = f"<b>{stop_name}</b><br>預估時間: {est_time_text}"
        
        folium.Marker(
            location=coords,
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

    # 添加公車當前位置標記 (如果提供)
    if bus_location:
        folium.Marker(
            location=[bus_location["lat"], bus_location["lon"]],
            popup=folium.Popup(f"<b>公車位置</b><br>路線: {route_name}", max_width=200),
            icon=folium.Icon(color="red", icon="bus", prefix="fa") # 使用Font Awesome的公車圖標
        ).add_to(m)

    # 繪製路線路徑 (使用實際站牌的順序)
    route_coords_list = [[stop["lat"], stop["lon"]] for stop in stops_data]
    if len(route_coords_list) > 1:
        folium.PolyLine(
            locations=route_coords_list,
            color='green',
            weight=5,
            opacity=0.7,
            tooltip=f"路線: {route_name}"
        ).add_to(m)

    # 將地圖保存為HTML文件並自動打開
    map_filename = f"bus_route_{route_name}_map.html"
    m.save(map_filename)
    print(f"地圖已保存到 '{map_filename}'。")
    print("正在嘗試在瀏覽器中打開地圖...")
    webbrowser.open(map_filename)
    print("✅ 完成！")

if __name__ == "__main__":
    print("歡迎使用公車路線查詢與地圖顯示工具！")
    print("-----------------------------------")

    # 這個區塊負責獲取所有公車路線列表 ( route_id 和 name )
    # 您之前的程式碼是用 Selenium 抓取這個列表，這裡為了簡潔，假設您有一個 CSV 檔案存儲了這些資訊
    # 建議使用您編號 2 程式碼的 `taipei_bus_routes.csv` 作為輸入
    
    # 請確保 'taipei_bus_routes.csv' 存在於相同的目錄或指定路徑
    # 範例：BASE_PATH = r'C:\Users\truck\OneDrive\文件\GitHub\cycu_oop_1132_11022327\期末報告'
    # ROUTE_LIST_CSV = 'taipei_bus_routes.csv'
    # ROUTE_LIST_CSV_PATH = os.path.join(BASE_PATH, ROUTE_LIST_CSV)

    # 這裡我們直接使用一個簡單的預設或從您提供過的 CSV 讀取
    # 如果您沒有 taipei_bus_routes.csv，可以手動建立一個包含 route_id 和 route_name 的 CSV
    # 範例 CSV 內容：
    # route_id,route_name
    # 10565,299
    # 10403,262
    # ...

    all_bus_routes_data = []
    ROUTE_LIST_CSV = 'taipei_bus_routes.csv' # 假設這個檔案存在
    
    # 假設您手動維護這個檔案，或者使用前一個腳本生成
    # 這裡可以根據您的實際路徑設置 BASE_PATH
    BASE_PATH = os.path.dirname(os.path.abspath(__file__)) # 預設為腳本所在目錄
    # 如果您的 taipei_bus_routes.csv 在期末報告資料夾內
    # BASE_PATH = r'C:\Users\truck\OneDrive\文件\GitHub\cycu_oop_1132_11022327\期末報告' 
    ROUTE_LIST_CSV_PATH = os.path.join(BASE_PATH, ROUTE_LIST_CSV)


    try:
        df_routes_base = pd.read_csv(ROUTE_LIST_CSV_PATH, encoding='utf-8-sig')
        if 'route_id' in df_routes_base.columns and 'route_name' in df_routes_base.columns:
            all_bus_routes_data = df_routes_base[['route_name', 'route_id']].to_dict(orient='records')
            print(f"已從 '{ROUTE_LIST_CSV_PATH}' 載入 {len(all_bus_routes_data)} 條公車路線。")
        else:
            print(f"錯誤：'{ROUTE_LIST_CSV_PATH}' 缺少 'route_id' 或 'route_name' 欄位。")
            print("請確認 CSV 檔案格式正確，或手動提供路線數據。")
            exit()
    except FileNotFoundError:
        print(f"錯誤：路線列表檔案 '{ROUTE_LIST_CSV_PATH}' 不存在。")
        print("請確認檔案路徑或手動輸入路線ID。")
        # 如果檔案不存在，則提供一個預設的測試路線
        all_bus_routes_data = [
            {"name": "299", "route_id": "10565"},
            {"name": "262", "route_id": "10403"},
            {"name": "信義幹線", "route_id": "10041"} # 範例路線
        ]
        print("已載入預設測試路線。")
    except Exception as e:
        print(f"載入路線列表時發生錯誤：{e}")
        exit()


    while True:
        route_name_input = input("\n請輸入公車路線號碼 (例如: 299, 262)，或輸入 'exit' 退出: ").strip()

        if route_name_input.lower() == 'exit':
            print("感謝使用，再見！")
            break

        if not route_name_input:
            print("輸入不能為空，請重試。")
            continue

        selected_route = None
        for route in all_bus_routes_data:
            if route["name"] == route_name_input:
                selected_route = route
                break
        
        if not selected_route:
            print(f"找不到路線 '{route_name_input}'，請確認輸入是否正確。")
            continue

        try:
            # 步驟 1: 獲取真實的站牌數據 (使用 Playwright)
            stops_with_coords = get_bus_route_stops_from_ebus(selected_route["route_id"], selected_route["name"])

            if not stops_with_coords:
                print(f"無法獲取路線 '{selected_route['name']}' 的站牌數據，無法繪製地圖。")
                continue

            # 步驟 2: 模擬公車當前位置和預估時間 (這部分還是模擬的，因為實時數據較難獲取)
            # 您需要從台北市公車API獲取實時公車位置和預估到站時間
            # 這裡為了演示地圖，我們仍然模擬這些數據
            bus_location_data = {
                "lat": stops_with_coords[random.randint(0, len(stops_with_coords)-1)]["lat"] + random.uniform(-0.0005, 0.0005),
                "lon": stops_with_coords[random.randint(0, len(stops_with_coords)-1)]["lon"] + random.uniform(-0.0005, 0.0005)
            }
            estimated_times_data = {stop["name"]: f"{random.randint(1, 15)} 分鐘" for stop in stops_with_coords}

            # 步驟 3: 顯示地圖
            display_bus_route_on_map(selected_route["name"], stops_with_coords, bus_location_data, estimated_times_data)

            time.sleep(2) # 延遲一下，避免太快回到輸入提示

        except Exception as e:
            print(f"處理路線 '{route_name_input}' 時發生錯誤：{e}")
            print("請確認您的網路連接或稍後再試。")