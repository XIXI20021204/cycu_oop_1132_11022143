
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time

# 設定 Selenium 的選項
options = Options()
options.add_argument("--headless")  # 不開啟瀏覽器視窗
options.add_argument("--disable-gpu")

# 建立 WebDriver
driver = webdriver.Chrome(options=options)

# 目標公車路線頁面（以公車5路為例）
url = "https://ebus.gov.taipei/EBus/VsSimpleMap?routeid=0100000500&amppgb=0"
driver.get(url)

# 等待頁面載入 JavaScript 資料
time.sleep(3)

# 取得頁面 HTML
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")

# 擷取站牌與到站時間資訊
stations = []
blocks = soup.select("div[id^='block_']")

for block in blocks:
    stop_name = block.get("data-stop", "").strip()
    eta_tag = block.find_next("span", class_="eta_onroad")
    eta_time = eta_tag.text.strip() if eta_tag else "無資料"
    stations.append((stop_name, eta_time))

driver.quit()

# 輸出爬取的結果
for i, (stop, eta) in enumerate(stations, start=1):
    print(f"{i}. {stop} - 到站時間: {eta}")
