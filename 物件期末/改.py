import pandas as pd
import re
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import json # 引入 json 模組用於序列化列表

# 定義檔案路徑
# 注意：此腳本不會使用 is_processed 欄位，每次執行會重新抓取並覆蓋輸出檔案。
# 如果需要像編號 1 腳本那樣的斷點續傳功能，則需要額外實現。
OUTPUT_FILENAME = 'all_bus_routes_with_stops_selenium_optimized.csv' # 單一輸出 CSV 檔名
BASE_PATH = os.getcwd() # 預設使用當前工作目錄，您也可以設定為 'C:\Users\truck\OneDrive\文件\GitHub\cycu_oop_1132_11022327\期末報告'

OUTPUT_FILEPATH = os.path.join(BASE_PATH, OUTPUT_FILENAME)

class BusRouteInfoSelenium:
    def __init__(self, routeid: str, direction: str = 'go', driver=None):
        self.rid = routeid
        self.driver = driver
        self.url = f'https://ebus.gov.taipei/Route/StopsOfRoute?routeid={routeid}'
        self.dataframe = None # 用於儲存解析後的公車站點數據

        if direction not in ['go', 'come']:
            raise ValueError("Direction must be 'go' or 'come'")
        self.direction = direction

        # 每個實例獨立的等待器，避免多執行緒或多頁面操作問題
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
                print(f"  正在切換到返程 ({self.rid})...")
                # 等待返程按鈕可見並可點擊
                come_button_selector = 'a.stationlist-come-go-gray.stationlist-come'
                self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, come_button_selector)))
                come_button = self.driver.find_element(By.CSS_SELECTOR, come_button_selector)
                come_button.click()
                print(f"  已點擊返程按鈕，等待內容更新...")
                # 點擊後需要額外等待，確保新的站點內容完全載入
                # 可以等待第一個站點的名稱變化，或者等待頁面加載狀態穩定
                # 這裡使用一個較為穩健的等待方式：等待站點列表中的第一個站點出現（或重新載入）
                self.wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'span.auto-list-stationlist-place')))
                time.sleep(2) # 額外給予頁面渲染時間，確保 JS 完成更新

            # 獲取當前頁面的 HTML 內容
            return self.driver.page_source
        except Exception as e:
            print(f"Error fetching content for route {self.rid}, direction {self.direction}: {e}")
            return None # 如果抓取失敗，返回 None

    def parse_route_info(self) -> pd.DataFrame:
        """
        解析抓取到的 HTML 內容，提取公車站點數據。
        """
        page_content = self._fetch_content_and_switch_direction()
        if page_content is None:
            return pd.DataFrame() # 如果內容獲取失敗，返回空 DataFrame

        # 與 Playwright 腳本使用相同的正則表達式來解析站點資訊
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
            print(f"  在路線 {self.rid}, 方向 {self.direction} 中未找到匹配的站點數據。")
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


if __name__ == "__main__":
    # 確保輸出檔案的目錄存在
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
        print(f"已建立基礎目錄: {BASE_PATH}")

    # 設置 Selenium WebDriver
    print("正在啟動 Chrome WebDriver...")
    chrome_options = Options()
    chrome_options.add_argument("--headless=new") # 使用新 Headless 模式
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-extensions")
    # chrome_options.add_argument("--blink-settings=imagesEnabled=false") # 不加載圖片，但這可能影響某些動態內容的載入，暫時移除
    chrome_options.page_load_strategy = 'normal' # 將載入策略改回 'normal'，確保所有資源載入
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    print("WebDriver 已啟動。")

    try:
        # 首先訪問公車路線列表頁面以獲取路線ID和名稱
        print("正在抓取公車路線列表...")
        driver.get("https://ebus.gov.taipei/ebus") # 您的原始起點頁面

        # 使用更長的等待時間，確保所有路線連結載入
        wait_for_links = WebDriverWait(driver, 30) # 增加等待時間到 30 秒
        wait_for_links.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a[href*='javascript:go']")))

        link_data = []
        for link in driver.find_elements(By.CSS_SELECTOR, "a[href*='javascript:go']"):
            href = link.get_attribute("href")
            name = link.text.strip()
            if href and name:
                # 從 href 提取 route_id
                route_id_match = re.search(r"go\('(\d+)'\)", href)
                if route_id_match:
                    route_id = route_id_match.group(1)
                    link_data.append({'route_id': route_id, 'route_name': name})
        
        if not link_data:
            print("錯誤：未找到任何公車路線連結。請檢查選擇器或網站結構。")
            driver.quit()
            exit()

        df_routes_base = pd.DataFrame(link_data)
        print(f"成功從網站獲取 {len(df_routes_base)} 條路線。")

        all_routes_data = []
        total_routes = len(df_routes_base)

        for index, row in df_routes_base.iterrows():
            route_id = row['route_id']
            route_name = row['route_name']

            print(f"\n--- 正在處理路線: {route_name} ({route_id}) --- ({index + 1}/{total_routes})")
            
            current_route_output_data = {'route_id': route_id, 'route_name': route_name}

            # --- 處理 'go' (去程) 方向 ---
            go_stops_list = []
            try:
                route_info_go = BusRouteInfoSelenium(route_id, direction="go", driver=driver)
                df_stops_go = route_info_go.parse_route_info()
                
                if not df_stops_go.empty:
                    # 將 DataFrame 轉換為 JSON 可序列化的列表
                    go_stops_list = df_stops_go[['stop_name', 'latitude', 'longitude', 'stop_id']].to_dict(orient='records')
                    print(f"  為去程找到 {len(go_stops_list)} 個站點。")
                else:
                    print(f"  去程未找到站點數據。")
            except Exception as e:
                print(f"  處理去程路線 {route_id} 時發生錯誤: {e}")
            
            # 將列表轉換為 JSON 字符串，儲存到 DataFrame 中
            current_route_output_data['go_stops'] = json.dumps(go_stops_list, ensure_ascii=False)
            
            time.sleep(1) # 增加小延遲

            # --- 處理 'come' (返程) 方向 ---
            come_stops_list = []
            try:
                route_info_come = BusRouteInfoSelenium(route_id, direction="come", driver=driver)
                df_stops_come = route_info_come.parse_route_info()

                if not df_stops_come.empty:
                    # 將 DataFrame 轉換為 JSON 可序列化的列表
                    come_stops_list = df_stops_come[['stop_name', 'latitude', 'longitude', 'stop_id']].to_dict(orient='records')
                    print(f"  為返程找到 {len(come_stops_list)} 個站點。")
                else:
                    print(f"  返程未找到站點數據。")
            except Exception as e:
                print(f"  處理返程路線 {route_id} 時發生錯誤: {e}")

            current_route_output_data['come_stops'] = json.dumps(come_stops_list, ensure_ascii=False)
            
            all_routes_data.append(current_route_output_data)

            print(f"--- 完成處理路線 {route_id}。---")
            
            # 延遲策略
            if (index + 1) % 10 == 0: # 每處理10條路線，休息更久
                print("正在進行較長休息...")
                time.sleep(10)
            else:
                time.sleep(3) # 一般延遲

    except Exception as main_e:
        print(f"腳本執行期間發生主要錯誤: {main_e}")
    finally:
        driver.quit() # 無論如何都要關閉 WebDriver
        print("WebDriver 已關閉。")

    # 將所有收集到的數據寫入 CSV 檔
    if all_routes_data:
        final_df = pd.DataFrame(all_routes_data)
        final_df.to_csv(OUTPUT_FILEPATH, index=False, encoding='utf-8-sig')
        print(f"\n✅ 完成！所有處理過的路線數據已寫入: {OUTPUT_FILEPATH}")
    else:
        print("\n沒有收集到任何路線數據。")

