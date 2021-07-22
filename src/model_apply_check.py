import json
from os import path


def load_json(file):
    with open(file, "r") as f:
        output = json.load(f)
    return output


class Data:
    def __init__(self, data_path="."):
        self.travel_times = load_json(
            f"{data_path}/model_apply_inputs/new_travel_times.json"
        )
        self.route_id = list(self.travel_times.keys())

    def __len__(self) -> int:
        return len(self.route_id)


if __name__ == "__main__":
    BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
    data = Data(data_path=f"{BASE_DIR}/data/")

    with open(f"{BASE_DIR}/data/model_apply_outputs/TOTAL_NUMBER_OF_ROUTE", "w") as f:
        f.write(str(len(data)))
