import requests
import csv
import json

def fetch_bus_data_modified(route_id, output_file):
    """
    從臺北市公開網站抓取指定公車代碼的資料，正確處理 "Go" 和 "Back" 站點列表，
    並輸出為 CSV 格式。

    :param route_id: 公車代碼 (例如 '0100000A00')
    :param output_file: 輸出的 CSV 檔案名稱
    """
    url = f"https://ebus.gov.taipei/Route/StopsOfRoute?routeid={route_id}"
    try:
        # 發送 GET 請求
        response = requests.get(url)
        response.raise_for_status()  # 檢查請求是否成功

        # 調試用：檢查伺服器返回的內容
        # print("伺服器返回的內容：")
        # print(response.text)

        # 嘗試解析 JSON
        try:
            data = response.json()  # 將回應轉換為 JSON 格式
        except json.JSONDecodeError:
            print("伺服器返回的內容不是 JSON 格式：")
            print(response.text)
            return

        all_stops_data = []

        # 處理去程站點
        go_stops = data.get("Go", [])
        for stop in go_stops:
            arrival_info = stop.get("EstimateTime") if stop.get("EstimateTime") is not None else "進站中" if stop.get("StopStatus") == 1 else "未發車"
            stop_number = stop.get("StopSequence", "未知")
            stop_name = stop.get("StopName", {}).get("Zh_tw", "未知")
            stop_id = stop.get("StopID", "未知")
            latitude = stop.get("Latitude", "未知")
            longitude = stop.get("Longitude", "未知")
            all_stops_data.append([arrival_info, stop_number, stop_name, stop_id, latitude, longitude])

        # 處理回程站點
        back_stops = data.get("Back", [])
        for stop in back_stops:
            arrival_info = stop.get("EstimateTime") if stop.get("EstimateTime") is not None else "進站中" if stop.get("StopStatus") == 1 else "未發車"
            stop_number = stop.get("StopSequence", "未知")
            stop_name = stop.get("StopName", {}).get("Zh_tw", "未知")
            stop_id = stop.get("StopID", "未知")
            latitude = stop.get("Latitude", "未知")
            longitude = stop.get("Longitude", "未知")
            all_stops_data.append([arrival_info, stop_number, stop_name, stop_id, latitude, longitude])

        # 將資料寫入 CSV
        if all_stops_data:
            with open(output_file, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["arrival_info", "stop_number", "stop_name", "stop_id", "latitude", "longitude"])
                writer.writerows(all_stops_data)
            print(f"資料已成功儲存至 {output_file}")
        else:
            print(f"無法取得路線 {route_id} 的站點資訊。")

    except requests.exceptions.RequestException as e:
        print(f"無法取得資料，請檢查網路連線或公車代碼是否正確。錯誤訊息：{e}")
    except Exception as e:
        print(f"發生錯誤：{e}")

# 測試函數
if __name__ == "__main__":
    route_id = input("請輸入公車代碼 (例如 '0100000A00')：")
    output_file = f"{route_id}_bus_data.csv"
    fetch_bus_data_modified(route_id, output_file)