import json
import pandas as pd
from os import path
import pickle
from collections import defaultdict


def load_json(file):
    with open(file, "r") as f:
        output = json.load(f)
    return output


def load_pickle(file):
    with open(file, "rb") as f:
        output = pickle.load(f)
    return output


class Train:
    def __init__(self, data_path="."):
        self.route_data = load_json(f"{data_path}/model_build_inputs/route_data.json")
        self.actual_sequences = load_json(
            f"{data_path}/model_build_inputs/actual_sequences.json"
        )
        self.route_id = list(self.route_data.keys())
        self.all_stations = pd.Series([self.get_station(i) for i in range(len(self))])
        self.data_path = data_path

    def get_station(self, idx) -> str:
        route_id = self[idx]
        station = self.route_data[route_id]["station_code"]
        return station

    def __len__(self) -> int:
        return len(self.route_data)

    def __getitem__(self, idx) -> str:
        return self.route_id[idx]

    def _build_history(self) -> None:
        output = {}
        for station in self.all_stations.unique():
            all_result = []
            for idx in self.all_stations[self.all_stations == station].index:
                route_id = self[idx]
                stops = [
                    [k, *v.values()]
                    for k, v in self.route_data[route_id]["stops"].items()
                ]
                for i, k in enumerate(
                    self.actual_sequences[route_id]["actual"].values()
                ):
                    stops[i].append(k)
                stops.sort(key=lambda x: x[-1])
                # old_x = ""
                result = []
                for x in [x[-2] for x in stops if not pd.isna(x[-2])]:
                    # if x != old_x:
                    #     result.append(x)
                    if x not in result:
                        result.append(x)
                    old_x = x
                result = [station] + result
                all_result.append(result)
            output[station] = all_result

        with open(f"{self.data_path}/model_build_outputs/history.json", "w") as f:
            json.dump(output, f)

        # G = defaultdict(lambda: 0)
        # for k, T in output.items():
        #     for s in T:
        #         for i in range(len(s) - 1):
        #             x = s[i].split(".")[0]
        #             y = s[i + 1].split(".")[0]
        #             if x != y:
        #                 G[(k, x, y)] += 1
        # G = dict(G)
        # with open(f"{self.data_path}/model_build_outputs/main_zone_map.pkl", "wb") as f:
        #     pickle.dump(G, f)

    def __call__(self) -> None:
        self._build_history()


if __name__ == "__main__":
    BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
    print(BASE_DIR)
    trainer = Train(data_path=f"{BASE_DIR}/data/")
    trainer()
