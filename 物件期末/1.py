import folium
import random
import time
import webbrowser

# --- 引入之前 Selenium 相關的庫 ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import csv # 如果還需要寫入CSV的話

# --- 新增一個函式來獲取公車路線的站牌名稱和經緯度 (這是關鍵的整合點) ---
def get_bus_route_stops_from_ebus(route_id, bus_name):
    """
    從台北市公車動態資訊系統抓取指定路線的站牌名稱，
    並嘗試獲取其經緯度 (這裡需要額外實現經緯度查找邏輯)。
    返回一個站牌列表，每個元素是 (站牌名稱, 緯度, 經度)。
    
    由於台北市公車動態資訊系統的站牌頁面不直接提供經緯度，
    這裡需要您額外實現一個站牌名稱到經緯度的映射或查詢機制。
    為了示例，我會使用一個簡單的模擬或預設值。
    """
    print(f"正在從 ebus.gov.taipei 獲取路線 '{bus_name}' ({route_id}) 的站牌數據...")
    
    # --- Selenium 部分 ---
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-extensions")
    options.add_argument("--blink-settings=imagesEnabled=false")
    options.page_load_strategy = 'normal' # 保持 'normal' 確保穩定性

    driver = None # 初始化為 None
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        detail_url = f"https://ebus.gov.taipei/EBus/VsSimpleMap?routeid={route_id}&gb=1"
        driver.get(detail_url)

        wait = WebDriverWait(driver, 20) # 20 秒等待時間
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#plMapStops .snz > span")))
        
        time.sleep(1.5) # 額外延遲確保渲染

        stop_spans = driver.find_elements(By.CSS_SELECTOR, "#plMapStops .snz > span")
        
        if not stop_spans:
            raise ValueError(f"未找到路線 {bus_name} 的任何站牌資訊。")

        # 獲取站牌名稱列表 (已排序)
        stop_names = [span.text.strip() for span in stop_spans if span.text.strip()]
        
        # --- 經緯度查找邏輯 (這裡需要您實現) ---
        # 這是最難的部分。台北市公車動態系統只給站牌名稱，沒有經緯度。
        # 您可能需要：
        # 1. 預先建立一個「站牌名稱 -> 經緯度」的對照表 (如果資料量不大)。
        # 2. 使用地圖 API (如 Google Maps Geocoding API, OpenStreetMap Nominatim)
        #    來查詢每個站牌的經緯度。這可能涉及 API Key 和使用限制。
        
        # 為了示範，這裡我會使用模擬的經緯度，並確保其順序與站牌名稱一致。
        # 實際應用中，您應該替換這裡的模擬部分。
        stops_with_coords = []
        base_lat = 25.0330 # 台北市中心附近緯度
        base_lon = 121.5654 # 台北市中心附近經度
        
        for i, stop_name in enumerate(stop_names):
            # 模擬經緯度，稍微偏移，讓它們看起來像一條線
            lat = base_lat + (i * 0.001) + random.uniform(-0.0005, 0.0005)
            lon = base_lon + (i * 0.0008) + random.uniform(-0.0005, 0.0005)
            stops_with_coords.append({"name": stop_name, "lat": lat, "lon": lon})
            
    except Exception as e:
        print(f"[錯誤] 獲取路線 {bus_name} 站牌數據失敗：{e}")
        stops_with_coords = [] # 返回空列表或處理錯誤
    finally:
        if driver:
            driver.quit()
            
    print(f"路線 '{bus_name}' 的站牌數據獲取完成。共 {len(stops_with_coords)} 站。")
    return stops_with_coords

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
    map_filename = f"bus_route_{route_name}_map.html" # 修改檔案名稱避免衝突
    m.save(map_filename)
    print(f"地圖已保存到 '{map_filename}'。")
    print("正在嘗試在瀏覽器中打開地圖...")
    webbrowser.open(map_filename)
    print("✅ 完成！")

if __name__ == "__main__":
    print("歡迎使用公車路線查詢與地圖顯示工具！")
    print("-----------------------------------")

    # 為了一致性，預先抓取所有公車路線的名稱和其對應的 route_id
    # 這部分與你之前的爬蟲程式碼相同，用於獲取所有公車路線列表
    print("正在獲取所有公車路線列表，請稍候...")
    all_bus_routes_data = []
    
    options_initial = Options()
    options_initial.add_argument("--headless=new")
    options_initial.add_argument("--disable-gpu")
    options_initial.add_argument("--no-sandbox")
    options_initial.add_argument("--disable-extensions")
    options_initial.add_argument("--blink-settings=imagesEnabled=false")
    options_initial.page_load_strategy = 'normal'

    driver_initial = None
    try:
        driver_initial = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options_initial)
        driver_initial.get("https://ebus.gov.taipei/ebus")
        wait_initial = WebDriverWait(driver_initial, 20) 
        wait_initial.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a[href*='javascript:go']")))

        for link in driver_initial.find_elements(By.CSS_SELECTOR, "a[href*='javascript:go']"):
            href = link.get_attribute("href")
            name = link.text.strip()
            if href and name:
                try:
                    route_id = href.split("('")[1].split("')")[0]
                    all_bus_routes_data.append({"name": name, "route_id": route_id})
                except IndexError:
                    print(f"警告：無法從 {href} 解析 route_id，跳過此連結。")
        print(f"已獲取 {len(all_bus_routes_data)} 條公車路線。")
    except Exception as e:
        print(f"錯誤：無法獲取公車路線列表。原因：{e}")
        print("請檢查您的網路連接或稍後再試。程式將退出。")
        exit()
    finally:
        if driver_initial:
            driver_initial.quit()


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
            # 步驟 1: 獲取真實的站牌數據 (包含經緯度，這裡我用模擬的)
            # 注意：這裡會開啟一個新的 Selenium 瀏覽器實例來抓取站牌
            stops_with_coords = get_bus_route_stops_from_ebus(selected_route["route_id"], selected_route["name"])

            if not stops_with_coords:
                print(f"無法獲取路線 '{selected_route['name']}' 的站牌數據，無法繪製地圖。")
                continue

            # 步驟 2: 模擬公車當前位置和預估時間 (這部分還是模擬的，因為實時數據較難獲取)
            # 您需要從台北市公車API獲取實時公車位置和預估到站時間
            # 這裡為了演示地圖，我們仍然模擬這些數據
            bus_location_data = {
                "lat": stops_with_coords[random.randint(0, len(stops_with_coords)-1)]["lat"] + random.uniform(-0.001, 0.001),
                "lon": stops_with_coords[random.randint(0, len(stops_with_coords)-1)]["lon"] + random.uniform(-0.001, 0.001)
            }
            estimated_times_data = {stop["name"]: f"{random.randint(1, 15)} 分鐘" for stop in stops_with_coords}

            # 步驟 3: 顯示地圖
            display_bus_route_on_map(selected_route["name"], stops_with_coords, bus_location_data, estimated_times_data)

            time.sleep(2) # 延遲一下，避免太快回到輸入提示

        except Exception as e:
            print(f"處理路線 '{route_name_input}' 時發生錯誤：{e}")
            print("請確認您的網路連接或稍後再試。")