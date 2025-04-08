# -*- coding: utf-8 -*-
import pandas as pd
import re
from playwright.sync_api import sync_playwright
import sqlite3
import os
import csv


class BusRouteInfo:
    def __init__(self, routeid: str, direction: str = 'go'):
        self.rid = routeid
        self.content = None
        self.url = f'https://ebus.gov.taipei/Route/StopsOfRoute?routeid={routeid}'

        if direction not in ['go', 'come']:
            raise ValueError("Direction must be 'go' or 'come'")

        self.direction = direction

        self._fetch_content()
        self._save_to_csv()

    def _fetch_content(self):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(self.url)
            
            if self.direction == 'come':
                page.click('a.stationlist-come-go-gray.stationlist-come')
            
            page.wait_for_timeout(3000)  # wait for 3 seconds
            self.content = page.content()
            browser.close()

        # Write the rendered HTML to a file
        os.makedirs("data", exist_ok=True)
        with open(f"data/ebus_taipei_{self.rid}.html", "w", encoding="utf-8") as file:
            file.write(self.content)

    def _save_to_csv(self):
        """
        將抓取的 HTML 資料解析並存成 CSV 檔案
        """
        # 模擬解析資料 (這裡假設資料是站點名稱的列表，需根據實際 HTML 調整)
        stops = re.findall(r'<div class="stop-name">([^<]+)</div>', self.content)

        if not stops:
            print("無法解析站點資料，請檢查 HTML 結構")
            return

        # 將資料存成 CSV
        csv_file = f"data/ebus_taipei_{self.rid}.csv"
        with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Stop Name"])  # CSV 標題
            for stop in stops:
                writer.writerow([stop])

        print(f"站點資料已儲存至 {csv_file}")


# 使用範例
if __name__ == "__main__":
    route_id = input("請輸入公車路線 ID (例如 '0100000A00')：")
    direction = input("請輸入方向 ('go' 或 'come')：")
    bus_info = BusRouteInfo(routeid=route_id, direction=direction)