# -*- coding: utf-8 -*-
import pandas as pd
import re
from playwright.sync_api import sync_playwright
import os
import time
import sys # 引入 sys 模組用於退出程式

# Install necessary packages
# !pip install playwright pandas
# !playwright install

# Define the base path for input and output files
# 請確認這個路徑是您實際擁有的，並且有寫入權限
BASE_PATH = r'C:\Users\a0965\OneDrive\文件\GitHub\cycu_oop_1132_11022143\物件期末'
ROUTE_LIST_CSV = 'taipei_bus_routes.csv' # 原始路線列表 CSV
OUTPUT_FILENAME = 'all_bus_routes_with_stops.csv' # 單一輸出 CSV 檔名

ROUTE_LIST_CSV_PATH = os.path.join(BASE_PATH, ROUTE_LIST_CSV)
OUTPUT_FILEPATH = os.path.join(BASE_PATH, OUTPUT_FILENAME)

# --- 路徑檢查與權限驗證 (在任何操作前執行) ---
try:
    if not os.path.exists(BASE_PATH):
        print(f"警告：基礎目錄 '{BASE_PATH}' 不存在，嘗試創建。")
        os.makedirs(BASE_PATH, exist_ok=True)
        print(f"已創建基礎目錄: {BASE_PATH}")
    else:
        print(f"基礎目錄已存在: {BASE_PATH}")
    
    # 測試寫入權限
    test_file_path = os.path.join(BASE_PATH, 'temp_write_test.txt')
    try:
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write("Test write access.\n")
        os.remove(test_file_path)
        print(f"對目錄 '{BASE_PATH}' 的寫入權限測試成功。")
    except IOError as e:
        print(f"錯誤：無法寫入目錄 '{BASE_PATH}'。請檢查權限。詳細訊息：{e}")
        print("請確保您的使用者帳戶對此路徑有『完全控制』或『修改』權限。")
        print("您可能需要以管理員身份運行此腳本，或將 BASE_PATH 更改為您有寫入權限的路徑。")
        sys.exit(1) # 退出程式

except Exception as e:
    print(f"初始化路徑時發生意外錯誤：{e}")
    sys.exit(1) # 退出程式

# --- BusRouteInfo 類別保持不變，因為邏輯上沒有問題 ---
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
        No longer saves the rendered HTML to a local file.
        """
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                try:
                    # Increased timeout for page loading
                    page.goto(self.url, timeout=60000) 
                    
                    if self.direction == 'come':
                        # Wait for the button to be visible and enabled before clicking
                        # 增加等待時間以提高穩定性
                        page.wait_for_selector('a.stationlist-come-go-gray.stationlist-come', state='visible', timeout=20000)
                        page.click('a.stationlist-come-go-gray.stationlist-come')
                        # Give a short delay after clicking to ensure content updates
                        time.sleep(2) 

                    # Wait for a specific part of the content to be loaded, e.g., the first station name
                    # This is more robust than a fixed timeout
                    page.wait_for_selector('span.auto-list-stationlist-place', timeout=20000) # 增加等待時間
                    
                    self.content = page.content()
                except Exception as e:
                    print(f"錯誤：抓取路線 {self.rid}, 方向 {self.direction} 內容失敗：{e}")
                    self.content = None # Set content to None if fetching fails
                finally:
                    browser.close()
        except Exception as e:
            print(f"錯誤：啟動 Playwright 瀏覽器失敗：{e}")
            self.content = None # 確保內容為 None

    def parse_route_info(self) -> pd.DataFrame:
        """
        Parses the fetched HTML content to extract bus stop data.
        Returns a DataFrame containing all stop details, but we will only use 'stop_name'.
        """
        if self.content is None:
            return pd.DataFrame() # Return empty DataFrame if content is not available

        pattern = re.compile(
            r'<li>.*?<span class="auto-list-stationlist-position.*?">(.*?)</span>\s*'
            r'<span class="auto-list-stationlist-number">\s*(\d+)</span>\s*'
            r'<span class="auto-list-stationlist-place">(.*?)</span>.*?'
            r'<input[^>]+name="item\.UniStopId"[^>]+value="(\d+)"[^>]*>.*?'
            r'<input[^>]+name="item\.Latitude"[^>]+value="([\d\.]+)"[^>]*>.*?'
            r'<input[^>]+name="item\.Longitude"[^>]+value="([\d\.]+)"[^>]*>',
            re.DOTALL
        )

        matches = pattern.findall(self.content)
        if not matches:
            # 如果沒有找到匹配項，可能是網頁結構改變或載入失敗，印出警告
            print(f"警告：路線 {self.rid}, 方向 {self.direction} 未找到站點數據。")
            return pd.DataFrame() # Return empty DataFrame if no matches

        bus_stops_data = [m for m in matches]
        self.dataframe = pd.DataFrame(
            bus_stops_data,
            columns=["arrival_info", "stop_number", "stop_name", "stop_id", "latitude", "longitude"]
        )

        # Convert appropriate columns to numeric types (optional for this specific request, but good practice)
        self.dataframe["stop_number"] = pd.to_numeric(self.dataframe["stop_number"], errors='coerce')
        self.dataframe["stop_id"] = pd.to_numeric(self.dataframe["stop_id"], errors='coerce')
        self.dataframe["latitude"] = pd.to_numeric(self.dataframe["latitude"], errors='coerce')
        self.dataframe["longitude"] = pd.to_numeric(self.dataframe["longitude"], errors='coerce')

        self.dataframe["direction"] = self.direction
        self.dataframe["route_id"] = self.rid

        return self.dataframe


if __name__ == "__main__":
    # 1. 讀取所有路線代碼和名稱，並處理進度標記
    try:
        # 使用 'utf-8-sig' 確保正確讀取中文
        df_routes_base = pd.read_csv(ROUTE_LIST_CSV_PATH, encoding='utf-8-sig')
        if 'route_id' not in df_routes_base.columns or 'route_name' not in df_routes_base.columns:
            raise ValueError(f"'{ROUTE_LIST_CSV_PATH}' 必須包含 'route_id' 和 'route_name' 欄位。")
        
        # 檢查 'is_processed' 欄位是否存在，如果不存在則新增並初始化為 False
        if 'is_processed' not in df_routes_base.columns:
            df_routes_base['is_processed'] = False
            print(f"已為 '{ROUTE_LIST_CSV_PATH}' 新增 'is_processed' 欄位。")
        else:
            # 確保 'is_processed' 是布林型態
            df_routes_base['is_processed'] = df_routes_base['is_processed'].astype(bool)
            print(f"'{ROUTE_LIST_CSV_PATH}' 已載入。有 {df_routes_base['is_processed'].sum()} 條路線已處理過。")

        print(f"成功從 '{ROUTE_LIST_CSV_PATH}' 讀取 {len(df_routes_base)} 條路線 ID。")
    except FileNotFoundError:
        print(f"錯誤：輸入檔案 '{ROUTE_LIST_CSV_PATH}' 未找到。請確認它是否存在。")
        sys.exit(1) # 如果輸入 CSV 遺失則退出
    except Exception as e:
        print(f"錯誤：讀取 CSV 檔案 '{ROUTE_LIST_CSV_PATH}' 時發生錯誤：{e}")
        sys.exit(1)

    # 判斷輸出 CSV 檔案是否已存在，以決定是否寫入 header
    output_file_exists = os.path.exists(OUTPUT_FILEPATH)
    
    # 2. 遍歷每個路線代碼，抓取站點名稱並合併
    processed_count = 0
    total_routes = len(df_routes_base)
    
    # 為避免重複處理已存在的輸出數據，先讀取已有的數據
    existing_output_df = pd.DataFrame()
    if output_file_exists:
        try:
            existing_output_df = pd.read_csv(OUTPUT_FILEPATH, encoding='utf-8-sig')
            print(f"已從 '{OUTPUT_FILEPATH}' 載入 {len(existing_output_df)} 條現有路線數據。")
        except Exception as e:
            print(f"警告：讀取現有輸出檔案 '{OUTPUT_FILEPATH}' 失敗，將重新建立。錯誤訊息：{e}")
            output_file_exists = False # 強制重新寫入 header

    for index, row in df_routes_base.iterrows():
        route_id = str(row['route_id']) # 確保 route_id 是字串
        route_name = row['route_name']
        is_processed = row['is_processed']
        
        # 檢查該路線是否已在輸出檔案中且已被標記為處理過
        if is_processed and not existing_output_df.empty and \
           (existing_output_df['route_id'] == route_id).any():
            print(f"\n--- 跳過路線: {route_name} ({route_id}) - 已處理且已存在於輸出檔案中。 ---")
            processed_count += 1 # 依然算作處理過
            continue

        print(f"\n--- 正在處理路線: {route_name} ({route_id}) --- ({processed_count + 1}/{total_routes})")
        
        current_route_output_data = {'route_id': route_id, 'route_name': route_name} 

        # --- 處理 'go' (去程) 方向 ---
        go_stop_names = []
        go_latitudes = []
        go_longitudes = []
        go_fetch_success = False
        try:
            route_info_go = BusRouteInfo(route_id, direction="go")
            df_stops_go = route_info_go.parse_route_info()
            
            if not df_stops_go.empty:
                go_stop_names = df_stops_go['stop_name'].tolist()
                go_latitudes = df_stops_go['latitude'].tolist()
                go_longitudes = df_stops_go['longitude'].tolist()
                print(f"  為 '去程' 方向找到 {len(go_stop_names)} 個站點。")
                go_fetch_success = True
            else:
                print(f"  未為 '去程' 方向找到站點數據。")
            
            time.sleep(1) # Add a small delay between requests

        except Exception as e:
            print(f"  處理路線 {route_id} 的 '去程' 方向時發生錯誤：{e}")
            
        for i, stop_name in enumerate(go_stop_names):
            current_route_output_data[f'stop_name_go_{i+1}'] = stop_name
            current_route_output_data[f'latitude_go_{i+1}'] = go_latitudes[i]
            current_route_output_data[f'longitude_go_{i+1}'] = go_longitudes[i]
            
        # --- 處理 'come' (返程) 方向 ---
        come_stop_names = []
        come_latitudes = []
        come_longitudes = []
        come_fetch_success = False
        try:
            route_info_come = BusRouteInfo(route_id, direction="come")
            df_stops_come = route_info_come.parse_route_info()

            if not df_stops_come.empty:
                come_stop_names = df_stops_come['stop_name'].tolist()
                come_latitudes = df_stops_come['latitude'].tolist()
                come_longitudes = df_stops_come['longitude'].tolist()
                print(f"  為 '返程' 方向找到 {len(come_stop_names)} 個站點。")
                come_fetch_success = True
            else:
                print(f"  未為 '返程' 方向找到站點數據。")

            time.sleep(1) # Add a small delay between requests

        except Exception as e:
            print(f"  處理路線 {route_id} 的 '返程' 方向時發生錯誤：{e}")
        
        for i, stop_name in enumerate(come_stop_names):
            current_route_output_data[f'stop_name_come_{i+1}'] = stop_name
            current_route_output_data[f'latitude_come_{i+1}'] = come_latitudes[i]
            current_route_output_data[f'longitude_come_{i+1}'] = come_longitudes[i]
        
        # 將單一路線的數據轉換為 DataFrame，並追加寫入 CSV
        single_route_df = pd.DataFrame([current_route_output_data])
        
        try:
            single_route_df.to_csv(
                OUTPUT_FILEPATH, 
                mode='a', # 追加模式
                index=False, 
                encoding='utf-8-sig', 
                header=not output_file_exists # 只有在檔案不存在時才寫入 header
            )
            # 寫入後，將 output_file_exists 設為 True，確保後續追加不再寫入 header
            output_file_exists = True 
            
            print(f"--- 路線 {route_id} 的數據已追加到 {OUTPUT_FILEPATH}。 ---")

            # 僅當去程或返程至少有一個有數據時，才將該路線標記為已處理
            if go_fetch_success or come_fetch_success: # 如果有成功抓取到任何一個方向的數據
                df_routes_base.loc[index, 'is_processed'] = True # 在 df_routes_base 中更新標記
            else:
                print(f"  警告：路線 {route_id} 沒有成功抓取到任何數據。未標記為已處理。")

        except Exception as e:
            print(f"錯誤：寫入路線 {route_id} 數據到 CSV 失敗：{e}")
            # 如果寫入失敗，不將其標記為已處理
        
        processed_count += 1
        print(f"--- 完成處理路線 {route_id}。狀態已更新。 ---")
        
        # 定期保存進度到原始的路線列表 CSV，以防程式崩潰
        if processed_count % 10 == 0 or processed_count == total_routes: # 每處理10條路線，或全部處理完畢，保存一次
            print(f"正在保存處理進度到 {ROUTE_LIST_CSV_PATH}...")
            try:
                df_routes_base.to_csv(ROUTE_LIST_CSV_PATH, index=False, encoding='utf-8-sig')
                print("進度已保存。")
            except Exception as e:
                print(f"錯誤：保存進度到 {ROUTE_LIST_CSV_PATH} 失敗：{e}")
            
        # 延遲策略
        if processed_count % 10 == 0: # 每處理10條路線，休息更久
            print("正在進行較長休息...")
            time.sleep(10)
        else:
            time.sleep(3) # 一般延遲

    # 3. 程式結束時，最終保存一次更新後的路線列表狀態
    print(f"\n最終保存處理狀態到 {ROUTE_LIST_CSV_PATH}...")
    try:
        df_routes_base.to_csv(ROUTE_LIST_CSV_PATH, index=False, encoding='utf-8-sig')
        print("最終狀態已保存。腳本執行完畢。")
    except Exception as e:
        print(f"錯誤：最終保存狀態到 {ROUTE_LIST_CSV_PATH} 失敗：{e}")

    print(f"\n所有處理過的路線站點數據 (包括新處理和跳過的) 都可以在: {OUTPUT_FILEPATH} 中找到。")