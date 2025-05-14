# -*- coding: utf-8 -*-
"""
This module retrieves bus stop data for a specific route and direction from the Taipei eBus website,
saves the rendered HTML and CSV file, and stores the parsed data in a SQLite database.
"""

import re
import pandas as pd
from playwright.sync_api import sync_playwright
from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


class taipei_route_list:
    """
    Manages fetching, parsing, and storing route data for Taipei eBus.
    """
    def __init__(self, working_directory: str = 'data'):
        self.working_directory = working_directory
        self.url = 'https://ebus.gov.taipei/ebus?ct=all'
        self.content = None
        self._fetch_content()
        self._setup_database()

    def _fetch_content(self):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(self.url)
            page.wait_for_timeout(3000)
            self.content = page.content()
            browser.close()
        html_file_path = f'{self.working_directory}/hermes_ebus_taipei_route_list.html'
        with open(html_file_path, "w", encoding="utf-8") as file:
            file.write(self.content)

    def _setup_database(self):
        Base = declarative_base()
        class bus_route_orm(Base):
            __tablename__ = 'data_route_list'
            route_id = Column(String, primary_key=True)
            route_name = Column(String)
            route_data_updated = Column(Integer, default=0)
        self.orm = bus_route_orm
        self.engine = create_engine(f'sqlite:///{self.working_directory}/hermes_ebus_taipei.sqlite3')
        self.engine.connect()
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def parse_route_list(self) -> pd.DataFrame:
        pattern = r'<li><a href="javascript:go\(\'(.*?)\'\)">(.*?)</a></li>'
        matches = re.findall(pattern, self.content, re.DOTALL)
        if not matches:
            raise ValueError("No data found for route table")
        bus_routes = [(route_id, route_name.strip()) for route_id, route_name in matches]
        self.dataframe = pd.DataFrame(bus_routes, columns=["route_id", "route_name"])
        return self.dataframe

    def save_to_database(self):
        for _, row in self.dataframe.iterrows():
            self.session.merge(self.orm(route_id=row['route_id'], route_name=row['route_name']))
        self.session.commit()

    def read_from_database(self) -> pd.DataFrame:
        query = self.session.query(self.orm)
        self.db_dataframe = pd.read_sql(query.statement, self.session.bind)
        return self.db_dataframe

    def set_route_data_updated(self, route_id: str, route_data_updated: int = 1):
        self.session.query(self.orm).filter_by(route_id=route_id).update({"route_data_updated": route_data_updated})
        self.session.commit()

    def set_route_data_unexcepted(self, route_id: str):
        self.session.query(self.orm).filter_by(route_id=route_id).update({"route_data_updated": 2 })
        self.session.commit()

    def __del__(self):
        self.session.close()
        self.engine.dispose()


class taipei_route_info:
    """
    Manages fetching, parsing, and storing bus stop data for a specified route and direction.
    """
    def __init__(self, route_id: str, direction: str = 'go', working_directory: str = 'data'):
        self.route_id = route_id
        self.direction = direction
        self.content = None
        self.url = f'https://ebus.gov.taipei/Route/StopsOfRoute?routeid={route_id}'
        self.working_directory = working_directory
        if self.direction not in ['go', 'come']:
            raise ValueError("Direction must be 'go' or 'come'")
        self._fetch_content()

    def _fetch_content(self):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(self.url)
            if self.direction == 'come':
                page.click('a.stationlist-come-go-gray.stationlist-come')
            page.wait_for_timeout(3000)
            self.content = page.content()
            browser.close()
        self.html_file = f"{self.working_directory}/ebus_taipei_{self.route_id}.html"

    def parse_route_info(self) -> pd.DataFrame:
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
            raise ValueError(f"No data found for route ID {self.route_id}")
        bus_routes = [m for m in matches]
        self.dataframe = pd.DataFrame(
            bus_routes,
            columns=["arrival_info", "stop_number", "stop_name", "stop_id", "latitude", "longitude"]
        )
        self.dataframe["direction"] = self.direction
        self.dataframe["route_id"] = self.route_id
        return self.dataframe

    def save_to_database(self):
        db_file = f"{self.working_directory}/hermes_ebus_taipei.sqlite3"
        engine = create_engine(f"sqlite:///{db_file}")
        Base = declarative_base()
        class bus_stop_orm(Base):
            __tablename__ = "data_route_info_busstop"
            stop_id = Column(Integer)
            arrival_info = Column(String)
            stop_number = Column(Integer, primary_key=True)
            stop_name = Column(String)
            latitude = Column(Float)
            longitude = Column(Float)
            direction = Column(String, primary_key=True)
            route_id = Column(String, primary_key=True)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        for _, row in self.dataframe.iterrows():
            session.merge(bus_stop_orm(
                stop_id=row["stop_id"],
                arrival_info=row["arrival_info"],
                stop_number=row["stop_number"],
                stop_name=row["stop_name"],
                latitude=row["latitude"],
                longitude=row["longitude"],
                direction=row["direction"],
                route_id=row["route_id"]
            ))
        session.commit()
        session.close()


# homework 1
def get_bus_info_go(bus_id):
    route_info = taipei_route_info(bus_id, direction="go")
    route_info.parse_route_info()
    route_info.save_to_database()
    stop_list = []
    for index, row in route_info.dataframe.iterrows():
        stop_list.append(row['stop_id'])
    return stop_list


if __name__ == "__main__":
    stop_ids = get_bus_info_go('0161000900')
    for stop_id in stop_ids:
        print(f"stop_id: {stop_id}")