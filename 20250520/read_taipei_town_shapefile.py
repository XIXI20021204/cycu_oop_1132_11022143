import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd

# 找出 taipei_town 目錄下的所有 .shp 檔案
shp_dir = "20250520/taipei_town"
shp_files = []
for fname in os.listdir(shp_dir):
    if fname.endswith(".shp"):
        shp_files.append(os.path.join(shp_dir, fname))

if not shp_files:
    print(f"No shapefiles found in {shp_dir}")
else:
    gdfs = []
    for shp_file in shp_files:
        try:
            gdf_temp = gpd.read_file(shp_file)
            gdfs.append(gdf_temp)
            print(f"Successfully loaded: {shp_file}")
        except Exception as e:
            print(f"Error loading {shp_file}: {e}")

    if not gdfs:
        print("No GeoDataFrames could be loaded.")
    else:
        # 统一所有GeoDataFrame的CRS
        crs_to_use = gdfs[0].crs
        aligned_gdfs = []
        for gdf in gdfs:
            if gdf.crs != crs_to_use:
                aligned_gdfs.append(gdf.to_crs(crs_to_use))
            else:
                aligned_gdfs.append(gdf)

        combined_gdf = pd.concat(aligned_gdfs, ignore_index=True)

        # --- 筛选出北北基桃区域 ---
        
        # 定义北北基桃的县市名称列表 (中文和英文映射)
        # 请根据您的shapefile中实际的县市名称进行调整
        # 这里假设 COUNTYNAME 字段的值是 '臺北市', '新北市', '基隆市', '桃園市'
        county_name_map = {
            '臺北市': 'Taipei City',
            '台北市': 'Taipei City', # 兼容另一种写法
            '新北市': 'New Taipei City',
            '基隆市': 'Keelung City',
            '桃園市': 'Taoyuan City',
            '桃园市': 'Taoyuan City' # 兼容另一种写法
        }
        
        # 筛选 GeoDataFrame
        if 'COUNTYNAME' in combined_gdf.columns:
            # 筛选出北北基桃的GeoDataFrame
            # 使用 county_name_map 的 keys 来匹配中文名称
            north_regions_gdf = combined_gdf[
                combined_gdf['COUNTYNAME'].isin(list(county_name_map.keys()))
            ].copy()

            if north_regions_gdf.empty:
                print("No North-North-Keelung-Taoyuan regions found with 'COUNTYNAME' column using specified names.")
                print("Available COUNTYNAMEs:", combined_gdf['COUNTYNAME'].unique())
            else:
                # --- 将 COUNTYNAME 列的中文值替换为英文 ---
                north_regions_gdf['COUNTYNAME_EN'] = north_regions_gdf['COUNTYNAME'].map(county_name_map)
                
                # 绘制北北基桃的 GeoDataFrame
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                
                # 根据新的英文列进行着色和生成图例
                north_regions_gdf.plot(ax=ax, edgecolor='black', linewidth=0.5, 
                                       column='COUNTYNAME_EN', cmap='Paired', legend=True) # 使用 COUNTYNAME_EN
                
                # 设置图表标题为英文
                plt.title("Map of Northern Taiwan (Taipei, New Taipei, Keelung, Taoyuan)")
                plt.xlabel("Longitude") # 添加经度标签
                plt.ylabel("Latitude")  # 添加纬度标签
                plt.axis('equal') # 保持地图的纵横比
                plt.show()

                print("Successfully plotted Northern Taiwan regions with English labels.")
        else:
            print("The GeoDataFrame does not have a 'COUNTYNAME' column. Cannot filter by county name.")
            print("Available columns:", combined_gdf.columns.tolist())
            # 如果没有 COUNTYNAME，打印所有列，以便用户手动检查
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            combined_gdf.plot(ax=ax, edgecolor='black', linewidth=0.5, cmap='viridis')
            plt.title("Combined Shapefiles (No COUNTYNAME column found)")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.axis('equal')
            plt.show()