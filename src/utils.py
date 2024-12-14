import json

def write_list_to_json(file_path, data_list: list):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data_list, file, ensure_ascii=False, indent=4)