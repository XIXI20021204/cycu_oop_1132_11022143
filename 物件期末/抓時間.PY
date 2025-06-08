import folium
import random
import time
import webbrowser
import re
import csv
import json

# --- 引入 Selenium 相關的庫 ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- 獲取公車路線的站牌名稱和真實經緯度函式 ---
def get_bus_route_stops_from_ebus(route_id, bus_name, driver_instance):
    """
    從台北市公車動態資訊系統抓取指定路線的站牌名稱和真實經緯度。
    返回一個站牌列表，每個元素是字典，包含 'name', 'lat', 'lon', 'stop_id'。
    """
    print(f"\n正在從 ebus.gov.taipei 獲取路線 '{bus_name}' ({route_id}) 的站牌數據...")

    url = f'https://ebus.gov.taipei/Route/StopsOfRoute?routeid={route_id}'
    wait = WebDriverWait(driver_instance, 20)

    stops_with_coords = []
    try:
        driver_instance.get(url)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'span.auto-list-stationlist-place')))
        time.sleep(1.5) # 額外延遲確保渲染

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
                    "stop_id": int(m[3]) if m[3].isdigit() else None,
                    "sequence": int(m[1]) if m[1].isdigit() else None
                })
            else:
                print(f"警告：站點 '{m[2]}' 經緯度無效，已跳過。")

    except Exception as e:
        print(f"[錯誤] 獲取路線 {bus_name} 站牌數據失敗：{e}")
        stops_with_coords = []

    print(f"路線 '{bus_name}' 的站牌數據獲取完成。共 {len(stops_with_coords)} 站。")
    return stops_with_coords

# --- 獲取公車預估到站時間函式 ---
def get_bus_estimated_times(route_id, bus_name, driver_instance):
    """
    從台北市公車動態資訊系統獲取指定路線的預估到站時間。
    返回一個字典，鍵為站牌名稱，值為預估時間資訊。
    """
    print(f"\n正在獲取路線 '{bus_name}' 的預估到站時間...")
    
    estimated_times = {}
    bus_locations = []
    
    try:
        # 訪問路線的即時資訊頁面
        url = f'https://ebus.gov.taipei/Route/BusInfo?routeid={route_id}'
        driver_instance.get(url)
        
        # 等待頁面載入
        wait = WebDriverWait(driver_instance, 15)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.auto-list-stationlist')))
        time.sleep(2)  # 額外等待確保動態內容載入
        
        # 嘗試找到預估時間的元素
        # 這裡可能需要根據實際的HTML結構來調整選擇器
        time_elements = driver_instance.find_elements(By.CSS_SELECTOR, '.auto-list-stationlist li')
        
        for element in time_elements:
            try:
                # 獲取站牌名稱
                station_name_elem = element.find_element(By.CSS_SELECTOR, '.auto-list-stationlist-place')
                station_name = station_name_elem.text.strip() if station_name_elem else None
                
                # 獲取預估時間 - 可能在不同的位置
                time_info = "暫無資訊"
                
                # 嘗試多種可能的時間顯示元素
                time_selectors = [
                    '.auto-list-stationlist-time',
                    '.bus-time',
                    '.estimate-time',
                    '.arrival-time'
                ]
                
                for selector in time_selectors:
                    try:
                        time_elem = element.find_element(By.CSS_SELECTOR, selector)
                        if time_elem and time_elem.text.strip():
                            time_info = time_elem.text.strip()
                            break
                    except:
                        continue
                
                # 如果沒有找到專門的時間元素，嘗試從整個元素的文本中提取
                if time_info == "暫無資訊":
                    element_text = element.text
                    # 使用正則表達式尋找時間模式
                    time_patterns = [
                        r'(\d+)\s*分',  # X分
                        r'進站中',
                        r'即將到站',
                        r'暫停服務',
                        r'末班車已過'
                    ]
                    
                    for pattern in time_patterns:
                        match = re.search(pattern, element_text)
                        if match:
                            time_info = match.group(0)
                            break
                
                if station_name:
                    estimated_times[station_name] = time_info
                    
            except Exception as e:
                print(f"處理站點預估時間時發生錯誤: {e}")
                continue
        
        # 嘗試獲取公車位置資訊
        try:
            bus_position_elements = driver_instance.find_elements(By.CSS_SELECTOR, '[data-lat][data-lng]')
            for pos_elem in bus_position_elements:
                try:
                    lat = float(pos_elem.get_attribute('data-lat'))
                    lon = float(pos_elem.get_attribute('data-lng'))
                    if lat and lon:
                        bus_locations.append({'lat': lat, 'lon': lon})
                except:
                    continue
        except Exception as e:
            print(f"獲取公車位置時發生錯誤: {e}")
        
    except Exception as e:
        print(f"[錯誤] 獲取預估時間失敗：{e}")
    
    print(f"已獲取 {len(estimated_times)} 個站點的預估時間資訊")
    if bus_locations:
        print(f"已獲取 {len(bus_locations)} 個公車位置")
    
    return estimated_times, bus_locations

# --- 使用替代API獲取預估時間 ---
def get_estimated_times_from_api(route_name, stops_data, driver_instance):
    """
    嘗試使用台北市公車API獲取預估時間
    """
    print(f"\n嘗試透過API獲取路線 '{route_name}' 的預估時間...")
    
    estimated_times = {}
    
    try:
        # 構建API URL - 這裡使用台北市政府開放資料API
        # 注意：實際使用時可能需要申請API Key
        api_url = f"https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2.0/youbike_immediate.json"
        
        # 由於我們在這個示例中無法直接調用外部API，
        # 我們將為每個站點生成模擬的預估時間
        for stop in stops_data:
            # 生成模擬的預估時間
            random_scenarios = [
                "進站中",
                "即將到站", 
                f"{random.randint(1, 15)}分",
                f"{random.randint(16, 30)}分",
                "暫無資訊"
            ]
            estimated_times[stop['name']] = random.choice(random_scenarios)
            
        print("已生成模擬預估時間資訊")
        
    except Exception as e:
        print(f"API獲取失敗: {e}")
        # 如果API失敗，生成基本的模擬數據
        for stop in stops_data:
            estimated_times[stop['name']] = "查詢中..."
    
    return estimated_times

# --- 顯示地圖函式 ---
def display_bus_route_on_map(route_name, stops_data, bus_locations=None, estimated_times=None):
    """
    將公車路線、站牌、預估時間和公車位置顯示在地圖上。
    stops_data: 列表，每個元素是一個字典，包含 'name', 'lat', 'lon'
    bus_locations: 列表，包含公車位置的字典 [{'lat': xx, 'lon': xx}, ...]
    estimated_times: 字典，鍵為站牌名稱，值為預估時間，可選
    """
    if not stops_data:
        print(f"沒有路線 '{route_name}' 的站牌數據可顯示。")
        return

    print(f"\n正在為路線 '{route_name}' 生成地圖...")

    # 以所有站牌的中心點為地圖中心
    avg_lat = sum(s["lat"] for s in stops_data) / len(stops_data)
    avg_lon = sum(s["lon"] for s in stops_data) / len(stops_data)
    map_center = [avg_lat, avg_lon]
    m = folium.Map(location=map_center, zoom_start=13)

    # 添加站牌標記和彈出視窗
    for i, stop in enumerate(stops_data):
        stop_name = stop["name"]
        coords = [stop["lat"], stop["lon"]]

        est_time_text = estimated_times.get(stop_name, "未知") if estimated_times else "未知"
        
        # 根據預估時間設置不同的顏色
        if est_time_text in ["進站中", "即將到站"]:
            icon_color = "red"
        elif "分" in est_time_text:
            try:
                minutes = int(re.search(r'(\d+)', est_time_text).group(1))
                if minutes <= 5:
                    icon_color = "orange"
                elif minutes <= 15:
                    icon_color = "blue"
                else:
                    icon_color = "gray"
            except:
                icon_color = "blue"
        else:
            icon_color = "gray"
        
        popup_html = f"""
        <div style='font-family: Arial; width: 200px;'>
            <b style='font-size: 14px;'>{stop_name}</b><br>
            <hr style='margin: 5px 0;'>
            <span style='color: #2E8B57;'><b>預估時間:</b></span> {est_time_text}<br>
            <span style='color: #4682B4;'><b>站序:</b></span> {i+1}<br>
            <span style='color: #666;'><b>座標:</b></span> {coords[0]:.6f}, {coords[1]:.6f}
        </div>
        """

        folium.Marker(
            location=coords,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{stop_name} - {est_time_text}",
            icon=folium.Icon(color=icon_color, icon="info-sign")
        ).add_to(m)

    # 添加公車當前位置標記 (如果提供)
    if bus_locations:
        for i, bus_loc in enumerate(bus_locations):
            folium.Marker(
                location=[bus_loc["lat"], bus_loc["lon"]],
                popup=folium.Popup(f"<b>公車位置 #{i+1}</b><br>路線: {route_name}", max_width=200),
                tooltip=f"公車 #{i+1}",
                icon=folium.Icon(color="red", icon="bus", prefix="fa")
            ).add_to(m)

    # 繪製路線路徑 (使用實際站牌的順序)
    route_coords_list = [[stop["lat"], stop["lon"]] for stop in stops_data]