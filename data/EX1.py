import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
from shapely.geometry import Point, LineString
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sqlalchemy import create_engine
import os

# 設定中文字型（支援中文）
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

def read_route_from_database(db_path, table_name, route_name):
    """
    從 SQLite 資料庫中讀取指定路線的資料，並轉換為 GeoDataFrame。
    """
    engine = create_engine(f"sqlite:///{db_path}")
    query = f"SELECT * FROM {table_name} WHERE route_name = '{route_name}'"
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"無法讀取資料表 {table_name} 或路線 {route_name}：{e}")
        return None

    # 確保資料表中包含必要欄位
    required_columns = {"longitude", "latitude", "station_name", "station_id"}
    if not required_columns.issubset(df.columns):
        print(f"資料表缺少必要欄位：{required_columns - set(df.columns)}")
        return None

    # 將經緯度轉換為幾何點
    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf

def draw_route_with_marker_from_db(db_path, table_name, outputfile: str, route_name: str, station_id: int, icon_path: str):
    """
    繪製指定公車路線，並在指定車站上繪製人形標記。
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    station_found = False  # 用於檢查是否找到指定車站

    # 從資料庫讀取指定路線的資料
    gdf = read_route_from_database(db_path, table_name, route_name)
    if gdf is None or gdf.empty:
        print(f"無法繪製路線 {route_name}，因為資料為空或讀取失敗。")
        return

    # 繪製點
    gdf.plot(ax=ax, color='blue', marker='o', markersize=5, label=f"路線 {route_name}")

    # 將點的 geometry 轉換為 LineString 並繪製線
    line_geometry = LineString(gdf.geometry.tolist())
    line_gdf = gpd.GeoDataFrame([1], geometry=[line_geometry], crs=gdf.crs)
    line_gdf.plot(ax=ax, color='blue', linewidth=1)

    # 顯示每個站名
    for x, y, name, stop_id in zip(gdf.geometry.x, gdf.geometry.y, gdf["station_name"], gdf["station_id"]):
        ax.text(x, y, name, fontsize=6, ha='left', va='center')

        # 如果車站編號匹配，繪製人形標記
        if stop_id == station_id:
            # 加載圖案
            try:
                img = plt.imread(icon_path)
                imagebox = OffsetImage(img, zoom=0.05)  # 調整圖案大小
                # 向左偏移圖案
                offset_x = x - 0.005  # 調整偏移量，單位為經度
                ab = AnnotationBbox(imagebox, (offset_x, y), frameon=False)
                ax.add_artist(ab)
                station_found = True
            except FileNotFoundError:
                print(f"找不到圖案檔案：{icon_path}")
                return

    if not station_found:
        print(f"找不到車站編號：{station_id} 在路線 {route_name} 中")

    ax.set_title(f"公車路線圖 - {route_name}")
    ax.set_xlabel("經度")
    ax.set_ylabel("緯度")
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    plt.savefig(outputfile, dpi=300)
    plt.close()

if __name__ == "__main__":
    db_path = "C:/Users/User/Documents/GitHub/cycu_oop_1132_11022143/20250429/hermes_ebus_taipei.sqlite3"
    table_name = "data_route_list"  # 替換為資料庫中的實際表名稱
    outputfile = "20250429/bus_route_with_marker.png"

    # 手動輸入公車路線名稱與車站編號
    route_name = input("請輸入公車路線名稱：")
    station_id = int(input("請輸入車站編號："))
    icon_path = "C:/Users/User/Documents/GitHub/cycu_oop_1132_11022143/20250429/1.jpg"  # 圖案的完整路徑

    draw_route_with_marker_from_db(db_path, table_name, outputfile, route_name, station_id, icon_path)