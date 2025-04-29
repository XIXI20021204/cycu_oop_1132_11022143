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
        """
        Initializes the taipei_route_list, fetches webpage content,
        configures the ORM, and sets up the SQLite database.

        Args:
            working_directory (str): Directory to store the HTML and database files.
        """
        self.working_directory = working_directory
        self.url = 'https://ebus.gov.taipei/ebus?ct=all'
        self.content = None

        # Fetch webpage content
        self._fetch_content()

        # Setup ORM base and table
        Base = declarative_base()

        class bus_route_orm(Base):
            __tablename__ = 'data_route_list'

            route_id = Column(String, primary_key=True)
            route_name = Column(String)
            route_data_updated = Column(Integer, default=0)

        self.orm = bus_route_orm

        # Create and connect to the SQLite engine
        self.engine = create_engine(f'sqlite:///{self.working_directory}/hermes_ebus_taipei.sqlite3')
        self.engine.connect()
        Base.metadata.create_all(self.engine)

        # Create session
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def _fetch_content(self):
        """
        Fetches the webpage content using Playwright and saves it as a local HTML file.
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(self.url)
            page.wait_for_timeout(3000)  # Wait for the page to load
            self.content = page.content()
            browser.close()

        # Save the rendered HTML to a file for inspection
        html_file_path = f'{self.working_directory}/hermes_ebus_taipei_route_list.html'
        with open(html_file_path, "w", encoding="utf-8") as file:
            file.write(self.content)

    def parse_route_list(self) -> pd.DataFrame:
        """
        Parses bus route data from the fetched HTML content.

        Returns:
            pd.DataFrame: DataFrame containing bus route IDs and names.

        Raises:
            ValueError: If no route data is found.
        """
        pattern = r'<li><a href="javascript:go\(\'(.*?)\'\)">(.*?)</a></li>'
        matches = re.findall(pattern, self.content, re.DOTALL)

        if not matches:
            raise ValueError("No data found for route table")

        bus_routes = [(route_id, route_name.strip()) for route_id, route_name in matches]
        self.dataframe = pd.DataFrame(bus_routes, columns=["route_id", "route_name"])
        return self.dataframe

    def save_to_database(self):
        """
        Saves the parsed bus route data to the SQLite database via SQLAlchemy ORM.
        """
        for _, row in self.dataframe.iterrows():
            self.session.merge(self.orm(route_id=row['route_id'], route_name=row['route_name']))

        self.session.commit()

    def read_from_database(self) -> pd.DataFrame:
        """
        Reads bus route data from the SQLite database.

        Returns:
            pd.DataFrame: DataFrame containing bus route data.
        """
        query = self.session.query(self.orm)
        self.db_dataframe = pd.read_sql(query.statement, self.session.bind)
        return self.db_dataframe

    def set_route_data_updated(self, route_id: str, route_data_updated: int = 1):
        """
        Sets the route_data_updated flag in the database.

        Args:
            route_id (str): The ID of the bus route.
            route_data_updated (bool): The value to set for the route_data_updated flag.
        """
        self.session.query(self.orm).filter_by(route_id=route_id).update({"route_data_updated": route_data_updated})
        self.session.commit()

    def set_route_data_unexcepted(self, route_id: str):
        """
        Sets the route_data_updated flag to 2 for unexpected errors.
        """
        self.session.query(self.orm).filter_by(route_id=route_id).update({"route_data_updated": 2})
        self.session.commit()

    def __del__(self):
        """
        Closes the session and engine when the object is deleted.
        """
        self.session.close()
        self.engine.dispose()


class taipei_route_info:
    """
    Placeholder for taipei_route_info class.
    """
    def __init__(self, route_id: str, direction: str = 'go'):
        self.route_id = route_id
        self.direction = direction

    def parse_route_info(self):
        """
        Placeholder for parsing route info.
        """
        pass

    def save_to_database(self):
        """
        Placeholder for saving route info to database.
        """
        pass


if __name__ == "__main__":
    # Initialize and process route data
    route_list = taipei_route_list()
    route_list.parse_route_list()
    route_list.save_to_database()

    bus1 = '0161000900'  # 承德幹線
    bus2 = '0161001500'  # 基隆幹線

    bus_list = [bus1]

    for route_id in bus_list:
        try:
            route_info = taipei_route_info(route_id, direction="go")
            route_info.parse_route_info()
            route_info.save_to_database()

            # Example output for debugging
            print(f"Processed route: {route_id}")

            route_list.set_route_data_updated(route_id)
            print(f"Route data for {route_id} updated.")

        except Exception as e:
            print(f"Error processing route {route_id}: {e}")
            route_list.set_route_data_unexcepted(route_id)
            continue