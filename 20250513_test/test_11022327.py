from cycu11022327.ebus_map import taipei_route_list, taipei_route_info

if __name__ == "__main__":
    stop_ids = get_bus_info_go('0161000900')
    for stop_id in stop_ids:
        print(f"stop_id: {stop_id}")