import pandas as pd
import requests
import time

# === 參數設定 ===
API_KEY = "AIzaSyBV4cedY5CjWvMZtF2p7cwh1BLFKKSgLsE"  # <== 請在這裡貼上你的金鑰
INPUT_FILE = "C:/Users/a0965/OneDrive/文件/GitHub/cycu_oop_1132_11022143/人文/twcoupon (1).csv"
ADDRESS_COLUMN = "地址"  # <== 修改為你的地址欄位名稱
OUTPUT_FILE = "twcoupon_geocoded.csv"

# === 讀取資料 ===
df = pd.read_csv(INPUT_FILE)

# 新增欄位
df["Latitude"] = None
df["Longitude"] = None

# === 地理編碼 ===
for idx, row in df.iterrows():
    address = row[ADDRESS_COLUMN]
    if pd.isna(address):
        continue

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": API_KEY
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if data["status"] == "OK":
            location = data["results"][0]["geometry"]["location"]
            df.at[idx, "Latitude"] = location["lat"]
            df.at[idx, "Longitude"] = location["lng"]
            print(f"[✓] 第 {idx + 1} 筆轉換成功")
        else:
            print(f"[!] 第 {idx + 1} 筆轉換失敗：{data['status']}")
    except Exception as e:
        print(f"[x] 發生錯誤：{e}")

    time.sleep(0.1)  # 延遲避免超過速率限制

# === 儲存新檔案 ===
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\n✅ 完成！已儲存至 {OUTPUT_FILE}")
