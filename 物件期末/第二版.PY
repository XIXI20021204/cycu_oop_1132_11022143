import folium
import random
import time
import webbrowser
import re # 引入 re 模組

# --- 引入 Selenium 相關的庫 ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- 新增一個函式來獲取公車路線的站牌名稱和經緯度 (整合編號1的經緯度獲取方式) ---
def get_bus_route_stops_from_ebus(route_id, bus_name, driver_instance):
    """
    從台北市公車動態資訊系統抓取指定路線的站牌名稱和真實經緯度。
    這整合了之前編號 1 中解析頁面獲取經緯度的方法。
    返回一個站牌列表，每個元素是字典，包含 'name', 'lat', 'lon', 'stop_id'。
    """
    print(f"正在從 ebus.gov.taipei 獲取路線 '{bus_name}' ({route_id}) 的站牌數據...")

    url = f'https://ebus.gov.taipei/Route/StopsOfRoute?routeid={route_id}'
    wait = WebDriverWait(driver_instance, 20) # 20 秒等待時間

    stops_with_coords = []
    try:
        driver_instance.get(url)

        # 等待頁面主要內容（例如路線名稱或第一個站點）載入
        # 注意：這裡使用 auto-list-stationlist-place 作為標誌，與之前編號1的函式一致
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'span.auto-list-stationlist-place')))

        time.sleep(1.5) # 額外延遲確保渲染

        page_content = driver_instance.page_source

        # 與編號 1 腳本使用相同的正則表達式來解析站點資訊
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
            print(f"未在路線 {bus_name} 中找到匹配的站點數據。")
            return []

        for m in matches:
            try:
                lat = float(m[4])
            except ValueError:
                lat = None
            try:
                lon = float(m[5])
            except ValueError:
                lon = None

            if lat is not None and lon is not None: # 只添加有有效經緯度的站點
                stops_with_coords.append({
                    "name": m[2],
                    "lat": lat,
                    "lon": lon,
                    "stop_id": int(m[3]) if m[3].isdigit() else None
                })
            else:
                print(f"警告：站點 '{m[2]}' 經緯度無效，已跳過。")

    except Exception as e:
        print(f"[錯誤] 獲取路線 {bus_name} 站牌數據失敗：{e}")
        stops_with_coords = [] # 返回空列表或處理錯誤

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

    # 以所有站牌的中心點為地圖中心
    avg_lat = sum(s["lat"] for s in stops_data) / len(stops_data)
    avg_lon = sum(s["lon"] for s in stops_data) / len(stops_data)
    map_center = [avg_lat, avg_lon]
    m = folium.Map(location=map_center, zoom_start=14)

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
            icon=folium.Icon(color="red", icon="bus", prefix="fa")
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

    # 設置 Selenium WebDriver，這裡的 driver 將會被多個函式共用
    print("正在啟動 Chrome WebDriver...")
    chrome_options = Options()
    chrome_options.add_argument("--headless=new") # 使用新 Headless 模式
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--blink-settings=imagesEnabled=false") # 禁用圖片載入以加速
    chrome_options.page_load_strategy = 'normal'

    driver = None # 初始化 driver
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        print("WebDriver 已啟動。")

        # 預先抓取所有公車路線的名稱和其對應的 route_id
        print("正在獲取所有公車路線列表，請稍候...")
        all_bus_routes_data = []

        driver.get("https://ebus.gov.taipei/ebus")
        wait_initial = WebDriverWait(driver, 30) # 增加等待時間
        wait_initial.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a[href*='javascript:go']")))

        for link in driver.find_elements(By.CSS_SELECTOR, "a[href*='javascript:go']"):
            href = link.get_attribute("href")
            name = link.text.strip()
            if href and name:
                try:
                    # 使用更寬鬆的正規表達式來提取 route_id
                    # 匹配 'go(' 和 ')' 之間的所有字元，無論是否是數字
                    route_id_match = re.search(r"go\('([^']+)'\)", href)
                    if route_id_match:
                        route_id = route_id_match.group(1)
                        all_bus_routes_data.append({"name": name, "route_id": route_id})
                    else:
                        print(f"警告：無法從 {href} 解析 route_id，跳過此連結。")
                except Exception as e: # 捕獲任何可能發生的異常
                    print(f"處理連結 {href} 時發生錯誤：{e}，跳過此連結。")
        print(f"已獲取 {len(all_bus_routes_data)} 條公車路線。")

    except Exception as e:
        print(f"錯誤：無法獲取公車路線列表或啟動 WebDriver。原因：{e}")
        print("請檢查您的網路連接或稍後再試。程式將退出。")
        if driver:
            driver.quit()
        exit()

    while True:
        route_name_input = input("\n請輸入公車路線號碼 (例如: 299, 0東)，或輸入 'exit' 退出: ").strip()

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
            # 輔助提示：列出部分相似的路線供使用者參考
            if len(all_bus_routes_data) > 0:
                print("以下是一些可用的路線範例：")
                sample_routes = random.sample(all_bus_routes_data, min(5, len(all_bus_routes_data)))
                for sr in sample_routes:
                    print(f"- {sr['name']}")
                print("請輸入完整且正確的路線名稱。")
            continue

        try:
            # 步驟 1: 獲取真實的站牌數據
            # 注意：這裡共用了一開始啟動的 driver 實例
            stops_with_coords = get_bus_route_stops_from_ebus(selected_route["route_id"], selected_route["name"], driver)

            if not stops_with_coords:
                print(f"無法獲取路線 '{selected_route['name']}' 的站牌數據，無法繪製地圖。")
                continue

            # 步驟 2: 模擬公車當前位置和預估時間 (仍為模擬，實時數據需另行獲取)
            if stops_with_coords:
                random_stop = random.choice(stops_with_coords)
                bus_location_data = {
                    "lat": random_stop["lat"] + random.uniform(-0.001, 0.001),
                    "lon": random_stop["lon"] + random.uniform(-0.001, 0.001)
                }
            else:
                bus_location_data = None # 如果沒有站點數據，則不顯示公車位置

            estimated_times_data = {stop["name"]: f"{random.randint(1, 15)} 分鐘" for stop in stops_with_coords}

            # 步驟 3: 顯示地圖
            display_bus_route_on_map(selected_route["name"], stops_with_coords, bus_location_data, estimated_times_data)

            time.sleep(2) # 延遲一下，避免太快回到輸入提示

        except Exception as e:
            print(f"處理路線 '{route_name_input}' 時發生錯誤：{e}")
            print("請確認您的網路連接或稍後再試。")
        finally:
            # 在這裡不關閉 driver，因為我們希望在整個會話中重用它
            pass

    # 主程式結束時才關閉 driver
    if driver:
        driver.quit()
    print("WebDriver 已關閉。")