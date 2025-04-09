import pandas as pd
import re
from playwright.sync_api import sync_playwright
import sqlite3
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
    

    def _fetch_content(self):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(self.url)
            
            if self.direction == 'come':
                page.click('a.stationlist-come-go-gray.stationlist-come')
            
            page.wait_for_timeout(3000)  # wait for 1 second
            self.content = page.content()
            browser.close()


        # Write the rendered HTML to a file route_{rid}.html
        with open(f"data/ebus_taipei_{self.rid}.html", "w", encoding="utf-8") as file:
            file.write(self.content)

    def parse_and_save(self):
        soup = BeautifulSoup(self.content, 'html.parser')
        stops = []

        stop_elements = soup.find_all("li")

        for stop in stop_elements:

            # 狀態
            status_tag = stop.find("span", class_=re.compile(r"^auto-list-stationlist-position auto-list-stationlist-position"))
            status = status_tag.text.strip() if status_tag else None

            # 站牌順序
            number_tag = stop.find("span", class_="auto-list-stationlist-number")
            number = number_tag.text.strip() if number_tag else None

            # 站名
            name_tag = stop.find("span", class_="auto-list-stationlist-place")
            name = name_tag.text.strip() if name_tag else None

            # stop_id
            stop_id_tag = stop.find("input", {"id": "item_UniStopId"})
            stop_id = stop_id_tag["value"] if stop_id_tag else None

            # 緯度
            lat_tag = stop.find("input", {"id": "item_Latitude"})
            lat = lat_tag["value"] if lat_tag else None

            # 經度
            lon_tag = stop.find("input", {"id": "item_Longitude"})
            lon = lon_tag["value"] if lon_tag else None

            if all([status, number, name, stop_id, lat, lon]):
                stops.append([status, number, name, stop_id, lat, lon])

        df = pd.DataFrame(stops, columns=["arrival_info", "stop_number", "stop_name", "stop_id", "latitude", "longitude"])
        df.to_csv(f"data/bus_stops_{self.rid}_{self.direction}.csv", index=False, encoding="utf-8-sig")

bus = BusRouteInfo("0100000A00", direction="go") # 你這邊前面可以改你要的公車代碼，後面看你要go還是come
bus.parse_and_save()
