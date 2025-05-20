# -*- coding: utf-8 -*-
import os
import csv
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

class BusRouteInfo:
    def __init__(self, routeid: str, direction: str = 'go'):
        self.rid = routeid
        self.content = None
        self.url = f'https://ebus.gov.taipei/Route/StopsOfRoute?routeid={routeid}'

        if direction not in ['go', 'come']:
            raise ValueError("Direction must be 'go' or 'come'")

        self.direction = direction

        self._fetch_content()
        self._parse_and_save_to_csv()

    def _fetch_content(self):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(self.url)
            
            if self.direction == 'come':
                page.click('a.stationlist-come-go-gray.stationlist-come')
            
            page.wait_for_selector('.auto-list-stationlist', timeout=10000)  # 最多等待 10 秒
            self.content = page.content()
            browser.close()

        os.makedirs("data", exist_ok=True)
        with open(f"data/ebus_taipei_{self.rid}.html", "w", encoding="utf-8") as file:
            file.write(self.content)

    def _parse_and_save_to_csv(self):
        soup = BeautifulSoup(self.content, 'html.parser')
        stops = []

        stop_elements = soup.select('.auto-list-stationlist')
        if not stop_elements:
            print("无法找到站点资讯，请检查选择器或网站结构")
            return

        for stop in stop_elements:
            try:
                arrival_info = stop.select_one('.auto-list-stationlist-position-time').text.strip()
                stop_number = stop.select_one('.auto-list-stationlist-number').text.strip()
                stop_name = stop.select_one('.auto-list-stationlist-place').text.strip()
                stop_id = stop.select_one('input[name="item.UniStopId"]')['value']
                latitude = stop.select_one('input[name="item.Latitude"]')['value']
                longitude = stop.select_one('input[name="item.Longitude"]')['value']

                stops.append([arrival_info, stop_number, stop_name, stop_id, latitude, longitude])
            except AttributeError:
                continue

        os.makedirs("data", exist_ok=True)

        csv_filename = f"data/bus_route_{self.rid}_{self.direction}.csv" # 修改文件名以包含方向
        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["公车到达时间", "车站序号", "车站名称", "车站编号", "latitude", "longitude"])
            writer.writerows(stops)

        print(f"资料已储存至 {csv_filename}")

        for stop in stops:
            print(f"{stop[0]}, {stop[1]}, {stop[2]}, {stop[3]}, {stop[4]}, {stop[5]}")

---

## 如何使用新的 `bus_info` 函数

现在，你可以在呼叫 `bus_info` 函数时直接传入想要查询的**公车路线 ID** 和**方向**。

**范例 1：查询去程 (go)**

```python
if __name__ == "__main__":
    bus_info("0161001500", "go")