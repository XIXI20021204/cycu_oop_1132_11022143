import folium
import random
import time
import webbrowser
import re
import csv
import asyncio # 雖然目前是 Selenium 程式碼，但因為之前有提到 asyncio，保留導入

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# 將抓取站牌數據的邏輯細分為處理單一方向的數據
def extract_stops_from_soup(soup, direction_type, route_id):
    stops_with_coords = []
    # 修改 estimated_times 的鍵，使其包含方向信息
    estimated_times = {}

    # 根據方向類型找到對應的容器
    if direction_type == "去程":
        direction_container = soup.find('div', id='GoDirectionRoute')
    elif direction_type == "返程":
        direction_container = soup.find('div', id='BackDirectionRoute')
    else:
        print(f"錯誤：未知方向類型 '{direction_type}'。")
        return [], {}

    if not direction_container:
        print(f"未找到 {direction_type} 方向的內容容器。")
        return [], {}

    # 查找每個站牌的 li 元素
    all_stop_list_items = direction_container.find_all('li')

    if not all_stop_list_items:
        print(f"在 {direction_type} 方向中未找到任何站牌列表項目。")
        return [], {}

    for item in all_stop_list_items:
        item_html = str(item)

        stop_name_tag = item.find('span', class_='auto-list-stationlist-place')
        stop_name = stop_name_tag.get_text().strip() if stop_name_tag else "未知站名"

        stop_id_match = re.search(r'<input[^>]+name="item\.UniStopId"[^>]+value="(\d+)"[^>]*>', item_html)
        lat_match = re.search(r'<input[^>]+name="item\.Latitude"[^>]+value="([\d\.]+)"[^>]*>', item_html)
        lon_match = re.search(r'<input[^>]+name="item\.Longitude"[^>]+value="([\d\.]+)"[^>]*>', item_html)

        stop_id = int(stop_id_match.group(1)) if stop_id_match and stop_id_match.group(1).isdigit() else None
        lat = float(lat_match.group(1)) if lat_match else None
        lon = float(lon_match.group(1)) if lon_match else None

        if lat is not None and lon is not None:
            stops_with_coords.append({
                "name": stop_name,
                "lat": lat,
                "lon": lon,
                "stop_id": stop_id,
                "direction": direction_type # 添加方向信息
            })
        else:
            print(f"警告：站點 '{stop_name}' 經緯度無效，已跳過。")

        # 抓取到站狀態/時間
        eta_text = "查無資訊" # 預設值

        # 優先查找 span.eta_onroad (動態顯示的到站時間，如「約X分」、「進站中」)
        eta_tag_onroad = item.find('span', class_='eta_onroad')
        if eta_tag_onroad and eta_tag_onroad.get_text().strip() != '':
            eta_text = eta_tag_onroad.get_text().strip()
        else:
            # 如果沒有找到 eta_onroad 或其內容為空，則嘗試查找 auto-list-stationlist-position-time (靜態顯示的到站時間，如「X分鐘」)
            eta_tag_static = item.find('span', class_='auto-list-stationlist-position-time')
            if eta_tag_static and eta_tag_static.get_text().strip() != '':
                eta_text = eta_tag_static.get_text().strip()
        
        # 使用組合鍵儲存預估時間，以區分去程和返程的同名站牌
        estimated_times[f"{stop_name}_{direction_type}"] = eta_text

    return stops_with_coords, estimated_times

# get_bus_route_stops_from_ebus 不再是 async 函數，移除 async 關鍵字
def get_bus_route_stops_from_ebus(route_id, bus_name, driver_instance):
    print(f"\n正在從 ebus.gov.taipei 獲取路線 '{bus_name}' ({route_id}) 的站牌數據和到站時間...")

    url = f'https://ebus.gov.taipei/Route/StopsOfRoute?routeid={route_id}'
    wait = WebDriverWait(driver_instance, 40) # 延長最長等待時間到 40 秒

    all_stops_data = [] # 儲存所有方向的站牌數據
    all_estimated_times = {} # 儲存所有方向的預估時間

    try:
        driver_instance.get(url)
        
        # 等待「去程/返程」切換按鈕出現，這通常意味著主頁面結構已載入
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'p.stationlist-come-go-c')))
        print("主頁面結構已載入。")

        # --- 處理去程數據 ---
        print("正在獲取去程站牌數據...")
        go_button = driver_instance.find_element(By.CSS_SELECTOR, 'a.stationlist-go')
        go_button.click()
        # 等待去程的站牌列表出現且內容載入
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#GoDirectionRoute li .auto-list-stationlist-place')))
        # 增加等待到站時間顯示的條件
        try:
            wait.until(
                EC.text_to_be_present_in_element((By.CSS_SELECTOR, '#GoDirectionRoute span.eta_onroad'), '分') or
                EC.text_to_be_present_in_element((By.CSS_SELECTOR, '#GoDirectionRoute span.eta_onroad'), '進站中') or
                EC.text_to_be_present_in_element((By.CSS_SELECTOR, '#GoDirectionRoute span.auto-list-stationlist-position-time'), '分鐘')
            )
            print("去程到站時間已載入。")
        except Exception as e:
            print(f"警告：去程到站時間等待超時 ({e})，部分或全部到站時間可能未完全載入。")
            time.sleep(5) # 額外給予一些時間，以防萬一

        go_page_content = driver_instance.page_source
        go_soup = BeautifulSoup(go_page_content, 'html.parser')
        go_stops, go_estimated_times = extract_stops_from_soup(go_soup, "去程", route_id)
        all_stops_data.extend(go_stops)
        all_estimated_times.update(go_estimated_times) # 這裡 now all_estimated_times has keys like "站名_去程"
        print(f"去程數據獲取完成。共 {len(go_stops)} 站。")

        # --- 處理返程數據 ---
        print("正在獲取返程站牌數據...")
        return_button = driver_instance.find_element(By.CSS_SELECTOR, 'a.stationlist-come')
        return_button.click()
        # 等待返程的站牌列表出現且內容載入
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#BackDirectionRoute li .auto-list-stationlist-place')))
        # 增加等待到站時間顯示的條件
        try:
            wait.until(
                EC.text_to_be_present_in_element((By.CSS_SELECTOR, '#BackDirectionRoute span.eta_onroad'), '分') or
                EC.text_to_be_present_in_element((By.CSS_SELECTOR, '#BackDirectionRoute span.eta_onroad'), '進站中') or
                EC.text_to_be_present_in_element((By.CSS_SELECTOR, '#BackDirectionRoute span.auto-list-stationlist-position-time'), '分鐘')
            )
            print("返程到站時間已載入。")
        except Exception as e:
            print(f"警告：返程到站時間等待超時 ({e})，部分或全部到站時間可能未完全載入。")
            time.sleep(5) # 額外給予一些時間，以防萬一

        return_page_content = driver_instance.page_source
        return_soup = BeautifulSoup(return_page_content, 'html.parser')
        return_stops, return_estimated_times = extract_stops_from_soup(return_soup, "返程", route_id)
        all_stops_data.extend(return_stops)
        all_estimated_times.update(return_estimated_times) # 這裡 all_estimated_times has keys like "站名_返程"
        print(f"返程數據獲取完成。共 {len(return_stops)} 站。")

    except Exception as e:
        print(f"[錯誤] 獲取路線 {bus_name} 站牌數據和到站時間失敗：{e}")
        all_stops_data = []
        all_estimated_times = {}

    print(f"路線 '{bus_name}' 的所有站牌數據和到站時間獲取完成。共 {len(all_stops_data)} 站。")
    return all_stops_data, all_estimated_times


def display_bus_route_on_map(route_name, stops_data, bus_location=None, estimated_times=None):
    if not stops_data:
        print(f"沒有路線 '{route_name}' 的站牌數據可顯示。")
        return

    print(f"\n正在為路線 '{route_name}' 生成地圖...")

    # 計算去程和返程的中心點以更好顯示
    go_coords = [[s["lat"], s["lon"]] for s in stops_data if s.get("direction") == "去程"]
    return_coords = [[s["lat"], s["lon"]] for s in stops_data if s.get("direction") == "返程"]

    all_lats = [s["lat"] for s in stops_data]
    all_lons = [s["lon"] for s in stops_data]

    if all_lats and all_lons:
        avg_lat = sum(all_lats) / len(all_lats)
        avg_lon = sum(all_lons) / len(all_lons)
        map_center = [avg_lat, avg_lon]
    else:
        # Fallback to a default center if no coordinates found
        map_center = [25.0330, 121.5654] # Taipei City Hall area
        print("警告：未找到站牌經緯度，地圖中心設為台北市中心。")

    m = folium.Map(location=map_center, zoom_start=13) # 稍微放大地圖以顯示去返程

    # 為不同方向的站牌和路線使用不同顏色
    colors = {"去程": "blue", "返程": "purple"}
    line_colors = {"去程": "darkblue", "返程": "darkpurple"}

    for stop in stops_data:
        stop_name = stop["name"]
        coords = [stop["lat"], stop["lon"]]
        direction = stop.get("direction", "未知方向")

        # 根據站牌名稱和方向從 estimated_times 中獲取正確的預估時間
        # 關鍵修改在這裡
        est_time_key = f"{stop_name}_{direction}"
        est_time_text = estimated_times.get(est_time_key, "查無資訊") if estimated_times else "查無資訊"
        
        popup_html = f"<b>{stop_name}</b><br>方向: {direction}<br>預估時間: {est_time_text}"

        folium.Marker(
            location=coords,
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=colors.get(direction, "gray"), icon="info-sign")
        ).add_to(m)

    if go_coords and len(go_coords) > 1:
        folium.PolyLine(
            locations=go_coords,
            color=line_colors.get("去程"),
            weight=5,
            opacity=0.7,
            tooltip=f"路線: {route_name} (去程)"
        ).add_to(m)

    if return_coords and len(return_coords) > 1:
        folium.PolyLine(
            locations=return_coords,
            color=line_colors.get("返程"),
            weight=5,
            opacity=0.7,
            tooltip=f"路線: {route_name} (返程)"
        ).add_to(m)

    if bus_location:
        folium.Marker(
            location=[bus_location["lat"], bus_location["lon"]],
            popup=folium.Popup(f"<b>公車位置</b><br>路線: {route_name}", max_width=200),
            icon=folium.Icon(color="red", icon="bus", prefix="fa")
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
            # CSV 導出時，經緯度分別輸出
            fieldnames = ['方向', '站牌名稱', '緯度', '經度', '站牌ID'] 
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for stop in stops_data:
                writer.writerow({
                    '方向': stop.get('direction', '未知'),
                    '站牌名稱': stop.get('name', ''),
                    '緯度': stop.get('lat', ''),
                    '經度': stop.get('lon', ''),
                    '站牌ID': stop.get('stop_id', '')
                })
        print(f"站牌數據已成功輸出到 '{csv_filename}'。")
    except Exception as e:
        print(f"錯誤：輸出 '{csv_filename}' 時發生問題：{e}")

if __name__ == "__main__":
    print("歡迎使用台北市公車路線查詢與地圖顯示工具！")
    print("-----------------------------------")

    print("正在啟動 Chrome WebDriver...")
    chrome_options = Options()
    # 保持註解，以便在可見模式下運行並觀察
    # chrome_options.add_argument("--headless=new") 
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.page_load_strategy = 'normal'
    chrome_options.add_argument("--enable-unsafe-swiftshader")
    chrome_options.add_argument("--log-level=OFF")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

    driver = None
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        print("WebDriver 已啟動 (可見模式，用於調試到站時間問題)。")

        print("正在獲取所有公車路線列表，請稍候...")
        all_bus_routes_data = []

        driver.get("https://ebus.gov.taipei/ebus")
        wait_initial = WebDriverWait(driver, 30)

        wait_initial.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[data-toggle='collapse'][href*='#collapse']")))
        time.sleep(2)

        for i in range(1, 23): # 這裡的範圍需要根據實際網站的摺疊選單數量調整
            try:
                collapse_link_selector = f"a[href='#collapse{i}']"
                collapse_link = driver.find_element(By.CSS_SELECTOR, collapse_link_selector)

                if collapse_link.get_attribute("aria-expanded") == "false" or "collapsed" in collapse_link.get_attribute("class"):
                    driver.execute_script("arguments[0].click();", collapse_link)
                    print(f"已點擊展開 #collapse{i}...")
                    time.sleep(0.5)
            except Exception as e:
                pass # 忽略找不到某些摺疊選單的錯誤

        time.sleep(3) # 給予時間讓所有路線連結載入

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
            # get_bus_route_stops_from_ebus 不再是 async 函數，直接呼叫
            stops_with_coords, estimated_times_data = \
                get_bus_route_stops_from_ebus(selected_route["route_id"], selected_route["name"], driver)

            if not stops_with_coords:
                print(f"無法獲取路線 '{selected_route['name']}' 的站牌數據，無法繪製地圖。")
                continue

            export_choice = input(f"是否要將路線 '{selected_route['name']}' 的站牌數據輸出為 CSV 檔案？(y/n): ").strip().lower()
            if export_choice == 'y':
                export_stops_to_csv(selected_route["name"], stops_with_coords)

            bus_location_data = None

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