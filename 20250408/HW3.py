import requests
import csv

def fetch_bus_route_to_csv(route_id, output_file='bus_route.csv'):
    url = f'https://ebus.gov.taipei/Route/StopsOfRoute?routeid={route_id}'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"資料擷取失敗：{e}")
        return

    route_stops = []

    for direction in data:  # 有「去程」和「返程」，這裡以全部資料為例
        for stop in direction.get("Stops", []):
            arrival_info = stop.get("StopStatusName", "無資料")
            stop_number = stop.get("StopSequence", "")
            stop_name = stop.get("StopName", {}).get("Zh_tw", "")
            stop_id = stop.get("StopID", "")
            latitude = stop.get("StopPosition", {}).get("PositionLat", "")
            longitude = stop.get("StopPosition", {}).get("PositionLon", "")

            route_stops.append([
                arrival_info, stop_number, stop_name, stop_id, latitude, longitude
            ])

    # 寫入 CSV 檔案
    with open(output_file, mode='w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["arrival_info", "stop_number", "stop_name", "stop_id", "latitude", "longitude"])
        writer.writerows(route_stops)

    print(f"已將資料儲存至 {output_file}")

# 使用範例：
fetch_bus_route_to_csv('0100000A00')
