import folium
import random
import time
import webbrowser
import re
import csv
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def get_bus_route_stops_from_ebus(route_id, bus_name, driver_instance):
    print(f"\n正在從 ebus.gov.taipei 獲取路線 '{bus_name}' ({route_id}) 的站牌數據...")

    url = f'https://ebus.gov.taipei/Route/StopsOfRoute?routeid={route_id}'
    wait = WebDriverWait(driver_instance, 20)

    stops_with_coords = []
    try:
        driver_instance.get(url)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'span.auto-list-stationlist-place')))
        time.sleep(1.5)

        page_content = driver_instance.page_source

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
                lon = float(m[5])
            except ValueError:
                lat = None
                lon = None

            if lat is not None and lon is not None:
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
        stops_with_coords = []

    print(f"路線 '{bus_name}' 的站牌數據獲取完成。共 {len(stops_with_coords)} 站。")
    return stops_with_coords

def display_bus_route_on_map(route_name, stops_data, bus_location=None, estimated_times=None):
    if not stops_data:
        print(f"沒有路線 '{route_name}' 的站牌數據可顯示。")
        return

    print(f"\n正在為路線 '{route_name}' 生成地圖...")

    avg_lat = sum(s["lat"] for s in stops_data) / len(stops_data)
    avg_lon = sum(s["lon"] for s in stops_data) / len(stops_data)
    map_center = [avg_lat, avg_lon]
    m = folium.Map(location=map_center, zoom_start=14)

    for stop in stops_data:
        stop_name = stop["name"]
        coords = [stop["lat"], stop["lon"]]

        # 從傳入的 estimated_times 字典中獲取到站時間
        est_time_text = estimated_times.get(stop_name, "無資料") if estimated_times else "無資料"
        popup_html = f"<b>{stop_name}</b><br>預估時間: {est_time_text}"

        folium.Marker(
            location=coords,
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

    if bus_location:
        folium.Marker(
            location=[bus_location["lat"], bus_location["lon"]],
            popup=folium.Popup(f"<b>公車位置</b><br>路線: {route_name}", max_width=200),
            icon=folium.Icon(color="red", icon="bus", prefix="fa")
        ).add_to(m)

    route_coords_list = [[stop["lat"], stop["lon"]] for stop in stops_data]
    if len(route_coords_list) > 1:
        folium.PolyLine(
            locations=route_coords_list,
            color='green',
            weight=5,
            opacity=0.7,
            tooltip=f"路線: {route_name}"
        ).add_to(m)

    map_filename = f"bus_route_{route_name}_map.html"
    m.save(map_filename)
    print(f"地圖已保存到 '{map_filename}'。")
    print("正在嘗試在瀏覽器中打開地圖...")
    webbrowser.open(map_filename)
    print("✅ 完成！")

def export_stops_to_csv(route_name, stops_data):
    if not stops_data:
        print(f"沒有路線 '{route_name}' 的站牌數據可輸出到 CSV。")
        return

    csv_filename = f"bus_route_{route_name}_stops.csv"
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['站牌名稱', '緯度', '經度', '站牌ID']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for stop in stops_data:
                writer.writerow({
                    '站牌名稱': stop.get('name', ''),
                    '緯度': stop.get('lat', ''),
                    '經度': stop.get('lon', ''),
                    '站牌ID': stop.get('stop_id', '')
                })
        print(f"站牌數據已成功輸出到 '{csv_filename}'。")
    except Exception as e:
        print(f"錯誤：輸出 '{csv_filename}' 時發生問題：{e}")

# 修改後的 get_estimated_times_from_ebus 函數
def get_estimated_times_from_ebus(route_id, driver_instance):
    print(f"正在從 ebus.gov.taipei 獲取路線 '{route_id}' 的預估到站時間...")
    # 仍然使用 VsSimpleMap 頁面來獲取實時到站時間
    url = f"https://ebus.gov.taipei/EBus/VsSimpleMap?routeid={route_id}&amppgb=0"
    estimated_times = {}
    try:
        driver_instance.get(url)
        # 等待足夠的時間讓頁面上的 JavaScript 載入並渲染出到站時間
        # 這裡從 3 秒增加到 5 秒，給予更多載入時間
        WebDriverWait(driver_instance, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.stationlist-item")) # 嘗試尋找站牌列表的通用項目
        )
        time.sleep(2) # 額外等待確保所有動態內容渲染完成

        html = driver_instance.page_source
        soup = BeautifulSoup(html, "html.parser")

        # 根據經驗和您提供的圖片，嘗試更通用或更新的選擇器
        # 假設每個站牌資訊是一個 class 為 'stationlist-item' 的 div
        # 站名可能在其中一個 div 內，到站時間可能在 class 為 'eta_onroad' 的 span 內
        
        # 重新嘗試抓取站牌資訊的區塊
        # 根據ebus的頁面結構，站牌列表通常在一個特定的div內，每個站牌項目前面會有一個block_ID
        # 這裡我們嘗試抓取所有可能包含站牌名稱和到站時間的項目
        
        # 尋找所有包含站牌名稱的元素，例如 class 為 auto-list-stationlist-place 的 span
        # 然後透過其父元素或兄弟元素尋找相對應的到站時間
        
        # 更保守且可能有效的方式是找到包裹所有站牌的 div，然後迭代每個站牌
        # 觀察 ebus.gov.taipei/EBus/VsSimpleMap 的 HTML，通常站牌資訊在一個大的 div 內，每個站是一個子元素
        
        # 假設結構為：
        # <div class="station-block">
        #    <div class="station-name-container" data-stop="站牌名稱">...</div>
        #    <span class="eta_onroad">到站時間</span>
        # </div>
        # 或者
        # <div id="block_...">
        #    <span class="station-name">站牌名稱</span>
        #    <span class="eta_onroad">到站時間</span>
        # </div>

        # 如果 'div[id^="block_"]' 仍無法找到，則可能是其他 class 或更深層的結構
        # 由於沒有直接的 HTML 文本，我們基於常見結構和圖片中暗示的到站時間標籤來調整。

        # 嘗試從更廣泛的範圍尋找包含站牌名稱和到站時間的元素
        # 在 VsSimpleMap 頁面，站牌的資訊通常在 `div.stationlist-item` 或類似的元素中
        station_items = soup.find_all('div', class_='stationlist-item') # 嘗試這個通用 class

        if not station_items:
            # 如果上面沒找到，就回到原始的 div[id^='block_'] 試試，因為它在某些情況下還是有效
            station_items = soup.select("div[id^='block_']")
            if not station_items:
                 print("⚠️ 未找到任何站牌項目 (stationlist-item 或 div[id^='block_'])，請檢查 HTML 結構。")
                 return {}

        for item in station_items:
            stop_name = ""
            eta_time = "無資料"

            # 嘗試從 data-stop 屬性獲取站名 (這是您之前代碼的邏輯)
            if item.has_attr('data-stop'):
                stop_name = item.get('data-stop').strip()
            else:
                # 如果沒有 data-stop，嘗試在 item 內部尋找站名 (例如 h3 或 span)
                # 這裡可能需要根據實際 HTML 調整
                name_tag = item.find(['h3', 'span'], class_=['bus-stop-name', 'auto-list-stationlist-place']) # 假設站名可能在這些標籤或類別中
                if name_tag:
                    stop_name = name_tag.get_text().strip()

            # 尋找到站時間
            eta_tag = item.find('span', class_='eta_onroad') # 到站時間通常在此 class
            if not eta_tag:
                eta_tag = item.find('div', class_='eta_onroad') # 也可能是 div
            
            if eta_tag:
                eta_time = eta_tag.get_text().strip()
                # 對時間字串進行清理，移除不必要的字元
                if "分" in eta_time:
                    eta_time = eta_time.replace("約", "").replace("分", "分鐘")
                elif "即將到站" in eta_time:
                    eta_time = "即將到站"
                elif "尚未發車" in eta_time:
                    eta_time = "尚未發車"
                elif "進站中" in eta_time:
                    eta_time = "進站中"
                elif "--" in eta_time:
                    eta_time = "無資料" # 如果是 "--"，也視為無資料
                # 移除所有空白字元
                eta_time = eta_time.replace(" ", "") 
            
            if stop_name:
                estimated_times[stop_name] = eta_time

    except Exception as e:
        print(f"[錯誤] 獲取預估到站時間失敗：{e}")
        # 如果失敗，返回空字典
        estimated_times = {} 
    print(f"預估到站時間獲取完成。共 {len(estimated_times)} 個站點的資料。")
    return estimated_times


if __name__ == "__main__":
    print("歡迎使用台北市公車路線查詢與地圖顯示工具！")
    print("-----------------------------------")

    print("正在啟動 Chrome WebDriver...")
    chrome_options = Options()
    # 再次將 headless 模式打開，因為您已經排除了單獨測試時的問題
    chrome_options.add_argument("--headless=new") 
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--blink-settings=imagesEnabled=false")
    chrome_options.page_load_strategy = 'normal'
    chrome_options.add_argument("--enable-unsafe-swiftshader")
    chrome_options.add_argument("--log-level=OFF")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

    driver = None
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        print("WebDriver 已啟動。")

        print("正在獲取所有公車路線列表，請稍候...")
        all_bus_routes_data = []

        driver.get("https://ebus.gov.taipei/ebus")
        wait_initial = WebDriverWait(driver, 30)

        wait_initial.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[data-toggle='collapse'][href*='#collapse']")))
        time.sleep(2)

        for i in range(1, 23):
            try:
                collapse_link_selector = f"a[href='#collapse{i}']"
                collapse_link = driver.find_element(By.CSS_SELECTOR, collapse_link_selector)

                if collapse_link.get_attribute("aria-expanded") == "false" or "collapse" in collapse_link.get_attribute("class"):
                    driver.execute_script("arguments[0].click();", collapse_link)
                    print(f"已點擊展開 #collapse{i}...")
                    time.sleep(0.5)

            except Exception as e:
                print(f"點擊 #collapse{i} 失敗或該元素不存在: {e}")

        time.sleep(3)

        bus_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='javascript:go']")
        for link in bus_links:
            href = link.get_attribute("href")
            name = link.text.strip()
            if href and name:
                try:
                    route_id_match = re.search(r"go\('([^']+)'\)", href)
                    if route_id_match:
                        route_id = route_id_match.group(1)
                        all_bus_routes_data.append({"name": name, "route_id": route_id})
                    else:
                        print(f"警告：無法從 {href} 解析 route_id，跳過此連結。")
                except Exception as e:
                    print(f"處理連結 {href} 時發生錯誤：{e}，跳過此連結。")
        print(f"已獲取 {len(all_bus_routes_data)} 條公車路線。")

    except Exception as e:
        print(f"錯誤：無法獲取公車路線列表或啟動 WebDriver。原因：{e}")
        print("請檢查您的網路連接或稍後再試。程式將退出。")
        if driver:
            driver.quit()
        exit()

    if all_bus_routes_data:
        print("\n--- 可查詢的公車路線列表 ---")
        display_count = 20
        if len(all_bus_routes_data) > 2 * display_count:
            print(f"部分路線列表 (共 {len(all_bus_routes_data)} 條):")
            for i in range(display_count):
                print(f"- {all_bus_routes_data[i]['name']}")
            print("...")
            for i in range(len(all_bus_routes_data) - display_count, len(all_bus_routes_data)):
                print(f"- {all_bus_routes_data[i]['name']}")
        else:
            for route in all_bus_routes_data:
                print(f"- {route['name']}")
        print("----------------------------")
    else:
        print("\n警告：未獲取到任何公車路線資訊。")

    while True:
        route_name_input = input("\n請輸入您想查詢的公車路線號碼 (請輸入完整的名稱，例如: 299, 0東)，或輸入 'exit' 退出: ").strip()

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
            print(f"找不到路線 '{route_name_input}'，請確認輸入是否正確，或從上方列表中選擇。")
            continue

        try:
            # 步驟 1: 獲取站牌名稱和經緯度
            stops_with_coords = get_bus_route_stops_from_ebus(selected_route["route_id"], selected_route["name"], driver)

            if not stops_with_coords:
                print(f"無法獲取路線 '{selected_route['name']}' 的站牌數據，無法繪製地圖。")
                continue

            # 詢問是否輸出 CSV
            export_choice = input(f"是否要將路線 '{selected_route['name']}' 的站牌數據輸出為 CSV 檔案？(y/n): ").strip().lower()
            if export_choice == 'y':
                export_stops_to_csv(selected_route["name"], stops_with_coords)

            # 步驟 2: 獲取真實預估到站時間
            # 這裡調用新的 get_estimated_times_from_ebus 函數
            estimated_times_data = get_estimated_times_from_ebus(selected_route["route_id"], driver)
            
            # 模擬公車當前位置 (這部分邏輯保持不變，因為實時位置數據可能需要不同的 API)
            bus_location_data = None
            if stops_with_coords:
                random_stop = random.choice(stops_with_coords)
                bus_location_data = {
                    "lat": random_stop["lat"] + random.uniform(-0.001, 0.001),
                    "lon": random_stop["lon"] + random.uniform(-0.001, 0.001)
                }
            
            # 步驟 3: 顯示地圖，包含站牌、公車位置和預估到站時間
            display_bus_route_on_map(selected_route["name"], stops_with_coords, bus_location_data, estimated_times_data)

            time.sleep(2)

        except Exception as e:
            print(f"處理路線 '{route_name_input}' 時發生錯誤：{e}")
            print("請確認您的網路連接或稍後再試。")
        finally:
            pass

    if driver:
        driver.quit()
    print("WebDriver 已關閉。")