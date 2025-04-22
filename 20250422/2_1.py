import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point
import os

def read_route_csv(csv_path):
    """
    Reads a bus route CSV file, creates Point geometries, and returns a GeoDataFrame.
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        return gdf
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {csv_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"警告：檔案 {csv_path} 是空的。")
        return gpd.GeoDataFrame() # 返回一個空的 GeoDataFrame
    except Exception as e:
        print(f"讀取檔案 {csv_path} 時發生錯誤：{e}")
        return None

def create_folium_map_from_csvs(csv_files: list, output_html: str):
    """
    Reads multiple bus route CSV files, converts them to GeoJSON layers,
    and creates an interactive HTML map using Folium.

    :param csv_files: A list of paths to the input CSV files.
    :param output_html: Path to the output HTML file.
    """
    # Create a Folium map centered around the first route's starting point (if available)
    initial_coords = [23.5, 121]  # Default center
    first_gdf = read_route_csv(csv_files[0]) if csv_files else None

    if first_gdf is not None and not first_gdf.empty:
        initial_coords = [first_gdf.iloc[0].geometry.y, first_gdf.iloc[0].geometry.x]
    m = folium.Map(location=initial_coords, zoom_start=12)

    for csv_file in csv_files:
        gdf = read_route_csv(csv_file)
        if gdf is not None and not gdf.empty:
            # Convert GeoDataFrame to GeoJSON
            geojson_data = gdf.to_json()

            # Add GeoJSON data to the map with a name for layers control
            route_name = os.path.basename(csv_file).replace(".csv", "")
            folium.GeoJson(geojson_data, name=route_name).add_to(m)

            # Add markers for each bus stop with the station name as a tooltip
            for index, row in gdf.iterrows():
                folium.Marker([row.geometry.y, row.geometry.x], tooltip=row["車站名稱"]).add_to(m)

    # Add layer control to the map
    folium.LayerControl().add_to(m)

    # Save the map to an HTML file
    m.save(output_html)

if __name__ == "__main__":
    # 指定輸入的 CSV 檔案和輸出的 HTML 檔案
    input_files = [
        "20250422/bus_route_0161000900.csv",
        "20250422/bus_route_0161001500.csv"
    ]
    outputfile = "bus_routes_map.html"

    # 繪製並儲存 Folium 地圖
    create_folium_map_from_csvs(input_files, outputfile)