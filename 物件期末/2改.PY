import pandas as pd
import re
import os
import time
import folium
import random # 仍用於模擬公車位置和預估時間
import webbrowser

# --- 引入 Selenium 相關的庫 ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import json # 引入 json 模組用於序列化列表

# 您可以選擇是否要將所有抓取到的路線和站點數據保存到 CSV 中
# 如果不需要，可以將以下兩行註釋掉
# OUTPUT_FILENAME = 'all_bus_routes_with_stops_and_map_data.csv'
# BASE_PATH = os.getcwd()
# OUTPUT_FILEPATH = os.path.join(BASE_PATH, OUTPUT_FILENAME)

class BusRouteInfoSelenium:
    """
    用於從台北市公車動態資訊系統抓取指定路線的站牌名稱和經緯度。
    這是從先前的 '優化後的 Selenium 公車資訊抓取腳本 (減少通知)' 整合過來的核心抓取邏輯。
    它會訪問提供完整站點數據的頁面。
    """
    def __init__(self, routeid: str, direction: str = 'go', driver=None):
        self.rid = routeid
        self.driver = driver
        # 使用顯示完整站點列表的頁面，而不是地圖頁面
        self.url = f'https://ebus.gov.taipei/Route/StopsOfRoute?routeid={routeid}'
        self.dataframe = None # 用於儲存解析後的公車站點數據

        if direction not in ['go', 'come']:
            raise ValueError("Direction must be 'go' or 'come'")
        self.direction = direction

        self.wait = WebDriverWait(self.driver, 20) # 增加等待時間到 20 秒

    def _fetch_content_and_switch_direction(self):
        """
        使用 Selenium 抓取網頁內容並處理方向切換。
        """
        try:
            self.driver.get(self.url)

            # 等待頁面主要內容（例如路線名稱或第一個站點）載入
            # 使用 EC.presence_of_element_located 更為穩健，確保元素存在於 DOM 中
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'span.auto-list-stationlist-place')))

            if self.direction == 'come':
                # 等待返程按鈕可見並可點擊
                come_button_selector = 'a.stationlist-come-go-gray.stationlist-come'
                self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, come_button_selector)))
                come_button = self.driver.find_element(By.CSS_SELECTOR, come_button_selector)
                come_button.click()
                # 點擊後需要額外等待，確保新的站點內容完全載入
                # 這裡使用一個較為穩健的等待方式：等待站點列表中的第一個站點出現（或重新載入）
                self.wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'span.auto-list-stationlist-place')))
                time.sleep(2) # 額外給予頁面渲染時間，確保 JS 完成更新

            # 獲取當前頁面的 HTML 內容
            return self.driver.page_source
        except Exception as e:
            print(f"錯誤：在獲取路線 {self.rid}, 方向 {self.direction} 內容時發生問題: {e}")
            return None # 如果抓取失敗，返回 None

    def parse_route_info(self) -> pd.DataFrame:
        """
        解析抓取到的 HTML 內容，提取公車站點數據，包括經緯度。
        """
        page_content = self._fetch_content_and_switch_direction()
        if page_content is None:
            return pd.DataFrame() # 如果內容獲取失敗，返回空 DataFrame

        # 與之前腳本使用相同的正則表達式來解析站點資訊，它能提取經緯度
        pattern = re.compile(
            r'<li>.*?<span class="auto-list-stationlist-position.*?">(.*?)</span>\s*'
            r'<span class="auto-list-stationlist-number">\s*(\d+)</span>\s*'
            r'<span class="auto-list-stationlist-place">(.*?)</span>.*?'
            r'<input[^>]+name="item\.UniStopId"[^>]+value="(\d+)"[^>]*>.*?'
            r'<input[^>]+name="item\.Latitude"[^>]+value="([\d\.]+)"[^>]*>.*?'
            r'<input[^>]+name="item\.Longitude"[^>]+value="([\d\.]+)"[^>]*>',
            re.DOTALL
        )

        matches = pattern.findall(page_content)
        if not matches:
            return pd.DataFrame() # 如果沒有匹配項，返回空 DataFrame

        bus_stops_data = []
        for m in matches:
            # 確保經緯度是浮點數，如果解析失敗則為 None
            try:
                lat = float(m[4])
            except ValueError:
                lat = None
            try:
                lon = float(m[5])
            except ValueError:
                lon = None

            bus_stops_data.append({
                "arrival_info": m[0],
                "stop_number": int(m[1]) if m[1].isdigit() else None,
                "stop_name": m[2],
                "stop_id": int(m[3]) if m[3].isdigit() else None,
                "latitude": lat,
                "longitude": lon
            })

        self.dataframe = pd.DataFrame(bus_stops_data)
        self.dataframe["direction"] = self.direction
        self.dataframe["route_id"] = self.rid

        return self.dataframe

def display_bus_route_on_map(route_name, go_stops_data, come_stops_data, bus_location=None, estimated_times=None):
    """
    將公車路線、站牌、預估時間和公車位置顯示在地圖上。
    go_stops_data: 列表，每個元素是一個字典，包含 'name', 'lat', 'lon' (去程)
    come_stops_data: 列表，每個元素是一個字典，包含 'name', 'lat', 'lon' (返程)
    bus_location: 字典，包含 'lat', 'lon'，可選 (模擬)
    estimated_times: 字典，鍵為站牌名稱，值為預估時間，可選 (模擬)
    """
    all_stops = go_stops_data + come_stops_data
    if not all_stops:
        print(f"沒有路線 '{route_name}' 的站牌數據可顯示。")
        return

    print(f"正在為路線 '{route_name}' 生成地圖...")

    # 以所有站牌的中心點創建地圖，或以去程的第一個站牌為中心
    if go_stops_data:
        map_center = [go_stops_data[0]["lat"], go_stops_data[0]["lon"]]
    elif come_stops_data:
        map_center = [come_stops_data[0]["lat"], come_stops_data[0]["lon"]]
    else: # Fallback if no stops found, although checked above
        map_center = [25.0330, 121.5654] # 台北市中心附近

    m = folium.Map(location=map_center, zoom_start=14)

    # --- 繪製去程路線和站牌 (綠色) ---
    if go_stops_data:
        go_route_coords_list = [[stop["lat"], stop["lon"]] for stop in go_stops_data]
        if len(go_route_coords_list) > 1:
            folium.PolyLine(
                locations=go_route_coords_list,
                color='green',
                weight=5,
                opacity=0.7,
                tooltip=f"路線: {route_name} (去程)"
            ).add_to(m)
        for stop in go_stops_data:
            stop_name = stop["name"]
            coords = [stop["lat"], stop["lon"]]
            est_time_text = estimated_times.get(stop_name, "未知") if estimated_times else "未知"
            popup_html = f"<b>{stop_name} (去程)</b><br>預估時間: {est_time_text}"
            folium.Marker(
                location=coords,
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)

    # --- 繪製返程路線和站牌 (紫色) ---
    if come_stops_data:
        come_route_coords_list = [[stop["lat"], stop["lon"]] for stop in come_stops_data]
        if len(come_route_coords_list) > 1:
            folium.PolyLine(
                locations=come_route_coords_list,
                color='purple',
                weight=5,
                opacity=0.7,
                tooltip=f"路線: {route_name} (返程)"
            ).add_to(m)
        for stop in come_stops_data:
            stop_name = stop["name"]
            coords = [stop["lat"], stop["lon"]]
            est_time_text = estimated_times.get(stop_name, "未知") if estimated_times else "未知"
            popup_html = f"<b>{stop_name} (返程)</b><br>預估時間: {est_time_text}"
            folium.Marker(
                location=coords,
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color="orange", icon="info-sign") # 使用不同顏色區分返程站牌
            ).add_to(m)

    # 添加公車當前位置標記 (如果提供)
    if bus_location:
        folium.Marker(
            location=[bus_location["lat"], bus_location["lon"]],
            popup=folium.Popup(f"<b>公車位置</b><br>路線: {route_name}", max_width=200),
            icon=folium.Icon(color="red", icon="bus", prefix="fa") # 使用Font Awesome的公車圖標
        ).add_to(m)

    # 將地圖保存為HTML文件並自動打開
    map_filename = f"bus_route_{route_name.replace('/', '_')}_map.html" # 修改檔案名稱避免衝突，替換 /
    m.save(map_filename)
    print(f"地圖已保存到 '{map_filename}'。")
    print("正在嘗試在瀏覽器中打開地圖...")
    webbrowser.open(map_filename)
    print("✅ 完成！")

if __name__ == "__main__":
    print("歡迎使用公車路線查詢與地圖顯示工具！")
    print("-----------------------------------")

    # 設置 Selenium WebDriver
    print("正在啟動 Chrome WebDriver...")
    chrome_options = Options()
    chrome_options.add_argument("--headless=new") # 使用新 Headless 模式
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-extensions")
    # chrome_options.add_argument("--blink-settings=imagesEnabled=false") # 不加載圖片，但這可能影響某些動態內容的載入，暫時移除
    chrome_options.page_load_strategy = 'normal' # 將載入策略改回 'normal'，確保所有資源載入
    
    # 添加以下行以處理 WebGL 警告 (可選，僅為抑制控制台訊息)
    # chrome_options.add_argument("--enable-unsafe-swiftshader") 

    driver = None
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        print("WebDriver 已啟動。")

        # 首先訪問公車路線列表頁面以獲取路線ID和名稱
        print("正在獲取所有公車路線列表，請稍候...")
        driver.get("https://ebus.gov.taipei/ebus") # 您的原始起點頁面

        # 使用更長的等待時間，確保所有路線連結載入
        wait_initial = WebDriverWait(driver, 30) # 增加等待時間到 30 秒
        wait_initial.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a[href*='javascript:go']")))

        all_bus_routes_data = []
        for link in driver.find_elements(By.CSS_SELECTOR, "a[href*='javascript:go']"):
            href = link.get_attribute("href")
            name = link.text.strip()
            if href and name:
                # 從 href 提取 route_id
                route_id_match = re.search(r"go\('(\d+)'\)", href)
                if route_id_match:
                    route_id = route_id_match.group(1)
                    all_bus_routes_data.append({"name": name, "route_id": route_id})
                
        if not all_bus_routes_data:
            print("錯誤：未找到任何公車路線連結。請檢查選擇器或網站結構。")
            raise Exception("無法獲取公車路線列表。")

        print(f"已獲取 {len(all_bus_routes_data)} 條公車路線。")

        while True:
            route_name_input = input("\n請輸入公車路線號碼 (例如: 299, 262)，或輸入 'exit' 退出: ").strip()

            if route_name_input.lower() == 'exit':
                print("感謝使用，再見！")
                break

            if not route_name_input:
                print("輸入不能為空，請重試。")
                continue

            # 查找使用者輸入的路線名稱對應的 route_id
            selected_route = None
            for route in all_bus_routes_data:
                if route["name"] == route_name_input:
                    selected_route = route
                    break
            
            if not selected_route:
                print(f"找不到路線 '{route_name_input}'，請確認輸入是否正確。")
                continue

            try:
                print(f"正在抓取路線 '{selected_route['name']}' ({selected_route['route_id']}) 的去程站牌數據...")
                # 使用 BusRouteInfoSelenium 類來抓取去程站牌
                route_info_go = BusRouteInfoSelenium(selected_route["route_id"], direction="go", driver=driver)
                df_stops_go = route_info_go.parse_route_info()
                
                go_stops_for_map = []
                if not df_stops_go.empty:
                    go_stops_for_map = df_stops_go[['stop_name', 'latitude', 'longitude']].rename(
                        columns={'stop_name': 'name', 'latitude': 'lat', 'longitude': 'lon'}
                    ).to_dict(orient='records')
                    print(f"  去程找到 {len(go_stops_for_map)} 個站點。")
                else:
                    print(f"  去程未找到站點數據。")

                time.sleep(1) # 短暫延遲

                print(f"正在抓取路線 '{selected_route['name']}' ({selected_route['route_id']}) 的返程站牌數據...")
                # 使用 BusRouteInfoSelenium 類來抓取返程站牌
                route_info_come = BusRouteInfoSelenium(selected_route["route_id"], direction="come", driver=driver)
                df_stops_come = route_info_come.parse_route_info()

                come_stops_for_map = []
                if not df_stops_come.empty:
                    come_stops_for_map = df_stops_come[['stop_name', 'latitude', 'longitude']].rename(
                        columns={'stop_name': 'name', 'latitude': 'lat', 'longitude': 'lon'}
                    ).to_dict(orient='records')
                    print(f"  返程找到 {len(come_stops_for_map)} 個站點。")
                else:
                    print(f"  返程未找到站點數據。")

                if not go_stops_for_map and not come_stops_for_map:
                    print(f"無法獲取路線 '{selected_route['name']}' 的任何站牌數據，無法繪製地圖。")
                    continue

                # 步驟 2: 模擬公車當前位置和預估時間 (這部分還是模擬的，因為實時數據較難獲取)
                # 您需要從台北市公車API獲取實時公車位置和預估到站時間
                # 這裡為了演示地圖，我們仍然模擬這些數據
                # 結合去程和返程的站點來生成模擬數據
                all_available_stops = go_stops_for_map + come_stops_for_map
                
                bus_location_data = None
                estimated_times_data = {}

                if all_available_stops:
                    # 隨機選取一個站點附近作為公車位置
                    random_stop_for_bus = random.choice(all_available_stops)
                    bus_location_data = {
                        "lat": random_stop_for_bus["lat"] + random.uniform(-0.001, 0.001),
                        "lon": random_stop_for_bus["lon"] + random.uniform(-0.001, 0.001)
                    }
                    # 為所有站點生成模擬預估時間
                    for stop in all_available_stops:
                        estimated_times_data[stop["name"]] = f"{random.randint(1, 15)} 分鐘"


                # 步驟 3: 顯示地圖
                display_bus_route_on_map(selected_route["name"], go_stops_for_map, come_stops_for_map, bus_location_data, estimated_times_data)

                time.sleep(2) # 延遲一下，避免太快回到輸入提示

            except Exception as e:
                print(f"處理路線 '{route_name_input}' 時發生錯誤：{e}")
                print("請確認您的網路連接或稍後再試。")
                
            time.sleep(3) # 每次查詢後的一個通用延遲

    except Exception as main_e:
        print(f"腳本執行期間發生主要錯誤: {main_e}")
    finally:
        if driver:
            driver.quit() # 無論如何都要關閉 WebDriver
            print("WebDriver 已關閉。")

