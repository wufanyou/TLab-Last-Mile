import json
import pandas as pd
import numpy as np
import os
from os import path
from collections import defaultdict
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import argparse
import pickle

Infinity = int(2 ** 31) - 1


def get_solution(manager, routing, solution):
    index = routing.Start(0)
    plan_output = []
    while not routing.IsEnd(index):
        plan_output.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
    return plan_output


def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return m["distance_matrix"][from_node][to_node]


def check_m_valid(m):
    if m["depot"] < 0:
        return False
    if m["depot"] >= len(m["distance_matrix"]):
        return False
    if len(m["distance_matrix"]) == 0:
        return False
    return True


def load_json(file):
    with open(file, "r") as f:
        output = json.load(f)
    return output


def load_pickle(file):
    with open(file, "rb") as f:
        output = pickle.load(f)
    return output


class Data:
    def __init__(self, data_path="."):

        self.package_data = load_json(
            f"{data_path}/model_apply_inputs/new_package_data.json"
        )
        self.route_data = load_json(
            f"{data_path}/model_apply_inputs/new_route_data.json"
        )
        self.travel_times = load_json(
            f"{data_path}/model_apply_inputs/new_travel_times.json"
        )
        self.history = load_json(f"{data_path}/model_build_outputs/history.json")

        # zone_map = load_pickle(f"{data_path}/model_build_outputs/main_zone_map.pkl")
        # self.zone_map = defaultdict(lambda: 0)
        # self.zone_map.update(zone_map)
        self.route_id = list(self.travel_times.keys())
        self.all_stations = pd.Series([self.get_station(i) for i in range(len(self))])

    def __getitem__(self, idx) -> str:

        return self.route_id[idx]

    def __len__(self) -> int:

        return len(self.route_id)

    # get the depot of given route
    def get_depot(self, idx):

        route_id = self[idx]
        depot = [
            k
            for k, v in self.route_data[route_id]["stops"].items()
            if v["type"] == "Station"
        ][0]

        return depot

    # get station of route
    def get_station(self, idx):

        route_id = self[idx]
        station = self.route_data[route_id]["station_code"]

        return station

    def build_data_model(self, idx) -> dict:

        route_id = self[idx]
        m = {}
        # m['distance_matrix'] = (pd.DataFrame(self.travel_times[route_id]).values*10).astype(int)
        m["num_vehicles"] = 1
        depot = self.get_depot(idx)
        depot = list(self.travel_times[route_id].keys()).index(depot)
        distance_matrix = pd.DataFrame(self.travel_times[route_id]).values * 10
        dummy_matrix = np.zeros(
            [len(distance_matrix) + 1, len(distance_matrix) + 1], dtype=int
        )
        dummy_matrix[-1, :] = Infinity
        dummy_matrix[-1, depot] = 0
        dummy_matrix[-1, -1] = 0
        dummy_matrix[:-1, :-1] = distance_matrix
        m["distance_matrix"] = dummy_matrix
        m["depot"] = len(dummy_matrix) - 1
        return m

    def get_sentences(self, station):

        sentences = None
        if station in self.history:
            sentences = self.history[station]
        return sentences

    # def get_zone_sentences(self, station):
    #
    #     sentences = None
    #     if station in self.zone_history:
    #         sentences = self.zone_history[station]
    #     return sentences

    def build_data_model_fix_end(self, idx) -> dict:

        route_id = self[idx]
        m = {}
        m["distance_matrix"] = (
            pd.DataFrame(self.travel_times[route_id]).values * 10
        ).astype(int)
        m["num_vehicles"] = 1
        depot = [
            k
            for k, v in self.route_data[route_id]["stops"].items()
            if v["type"] == "Station"
        ][0]
        m["depot"] = list(self.travel_times[route_id].keys()).index(depot)
        return m

    def build_zone_data_model(self, idx, depot, stops, end) -> dict:

        route_id = self[idx]
        sequence = [depot] + stops + [end]
        distance_matrix = pd.DataFrame(self.travel_times[route_id])
        distance_matrix = distance_matrix.loc[sequence, sequence].values * 10
        m = {}
        dummy_matrix = np.zeros(
            [len(distance_matrix) + 1, len(distance_matrix) + 1], dtype=int
        )
        dummy_matrix[:-1, :-1] = distance_matrix
        dummy_matrix[-1, :] = Infinity
        dummy_matrix[:, -1] = Infinity
        dummy_matrix[-1, 0] = 0
        dummy_matrix[-2, -1] = 0
        dummy_matrix[-1, -1] = 0
        m["num_vehicles"] = 1
        m["distance_matrix"] = dummy_matrix
        m["depot"] = len(dummy_matrix) - 1  # last dummy node as depot
        m["sequence"] = sequence

        return m

    def build_zone_data_model_last(self, idx, depot, stops) -> dict:

        route_id = self[idx]
        sequence = [depot] + stops
        distance_matrix = pd.DataFrame(self.travel_times[route_id])
        distance_matrix = distance_matrix.loc[sequence, sequence].values * 10

        m = {}
        dummy_matrix = np.zeros(
            [len(distance_matrix) + 1, len(distance_matrix) + 1], dtype=int
        )
        dummy_matrix[:-1, :-1] = distance_matrix
        dummy_matrix[-1, :] = Infinity
        dummy_matrix[-1, 0] = 0
        dummy_matrix[:, -1] = 0
        dummy_matrix[-1, -1] = 0

        m["num_vehicles"] = 1
        m["distance_matrix"] = dummy_matrix
        m["depot"] = len(dummy_matrix) - 1
        m["sequence"] = sequence

        return m

    def get_zone_rank(self, idx):

        route_id = self[idx]
        zone = {}
        station = self.get_station(idx)
        for k, v in self.route_data[self[idx]]["stops"].items():
            if v["type"] == "Dropoff":
                if not pd.isna(v["zone_id"]):
                    zone[k] = v["zone_id"]  # .split('.')[0]
                else:
                    zone[k] = v["zone_id"]
            else:
                zone[k] = station

        sentences = self.get_sentences(station)

        if sentences is not None:
            unique_zone = set(list(np.unique(list(zone.values()))))

            seq_inter = [set(s).intersection(unique_zone) for s in sentences]
            # seq_len = np.array([len(s) for s in sentences])
            seq_sim = np.array([len(s) for s in seq_inter])
            seq_sim_sort = (-1 * seq_sim).argsort()

            current_union = seq_inter[seq_sim_sort[0]]
            max_pattern = [i for i in sentences[seq_sim_sort[0]] if i in current_union]

            for seq_id in seq_sim_sort[1:]:
                if seq_sim[seq_id] < 3:
                    break
                else:
                    new_diff = seq_inter[seq_id].difference(current_union)

                    if len(new_diff) > 1:
                        current_union = current_union.union(new_diff)
                        new_pattern = [i for i in sentences[seq_id] if i in new_diff]
                        if (
                            new_pattern[0].split(".")[0]
                            == max_pattern[-1].split(".")[0]
                        ):
                            max_pattern.extend(new_pattern)
                        elif (
                            new_pattern[-1].split(".")[0]
                            == max_pattern[1].split(".")[0]
                        ):
                            max_pattern = (
                                [max_pattern[0]] + new_pattern + max_pattern[1:]
                            )
                        else:
                            try:
                                x = max_pattern[-1].split(".")[0]
                                y = max_pattern[1].split(".")[0]
                                z = new_pattern[0].split(".")[0]
                                w = new_pattern[-1].split(".")[0]
                                x, x_n = x.split("-")
                                y, y_n = y.split("-")
                                z, z_n = z.split("-")
                                w, w_n = w.split("-")
                                if (x == z) and abs(x_n - z_n) <= 1:
                                    max_pattern.extend(new_pattern)
                                elif (y == w) and abs(y_n - w_n) <= 1:
                                    max_pattern = (
                                        [max_pattern[0]] + new_pattern + max_pattern[1:]
                                    )
                            except:
                                pass

                        # else:
                        #     for i in range(len(new_pattern)):
                        #         x = str(new_pattern[i]).split(".")[0]
                        #         for j in range(len(max_pattern) - 1, 0, -1):
                        #             y = str(max_pattern[j]).split(".")[0]
                        #             if x == y:
                        #                 max_pattern.insert(j + 1, new_pattern[i])
                        #                 break
                        #         else:
                        #             x = str(new_pattern[i]).split(".")[0]
                        #             x, x_n = x.split("-")
                        #             x_n = int(x_n)
                        #             for j in range(len(max_pattern) - 1, 0, -1):
                        #                 y = str(max_pattern[j]).split(".")[0]
                        #                 y, y_n = y.split("-")
                        #                 y_n = int(y_n)
                        #                 if (x == y) and abs(y_n - x_n) <= 2:
                        #                     max_pattern.insert(j + 1, new_pattern[i])
                        #                     break
                        #             else:
                        #                 max_pattern.append(new_pattern[i])

                        # max_pattern.extend(new_pattern)

            rank = defaultdict(lambda: float("nan"))
            rank.update(dict(zip(max_pattern, range(len(max_pattern)))))
            # r = np.arange(len(max_pattern)) #[0,1,2,3...]
            # r[1:]+=1 #[0,2,3,4,]...
            # r = r//2
            # rank.update(dict(zip(max_pattern, r)))

            distance = pd.DataFrame(self.travel_times[route_id])
            for k, v in zone.items():
                if v not in rank:
                    values = (distance[k] + distance.loc[:, k]).copy()
                    values = values.sort_values()
                    for vv in values.index:
                        if (zone[vv] in rank) and (zone[vv] != station):
                            zone[k] = zone[vv]
                            break
        else:
            rank = defaultdict(lambda: float("nan"))
            for k, v in zone.items():
                rank[k] = 0
        return zone, rank


if __name__ == "__main__":

    BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
    os.makedirs(f"{BASE_DIR}/data/model_apply_outputs/route", exist_ok=True)
    data = Data(data_path=f"{BASE_DIR}/data/")

    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "-i",
        "--id",
        default=0,
        type=int,
    )

    args = parser.parse_args()

    for idx in range(args.id, len(data)):
        try:
            zone, zone_rank = data.get_zone_rank(idx)
            route_id = data[idx]
            distance_matrix = pd.DataFrame(data.travel_times[route_id])  # .values*10
            zone = pd.DataFrame(zone, index=["zone_id"]).T
            zone["zone_rank"] = zone.zone_id.map(zone_rank)
            # print(zone,zone_rank)
            zone = zone.reset_index()
            zone_rank = zone.zone_rank.dropna().unique()
            zone_rank.sort()
            if len(zone_rank) > 1:
                result = [zone[zone.zone_rank == 0]["index"].iloc[0]]
                for i in range(1, len(zone_rank) - 1):
                    depot = result[-1]
                    stops = list(zone[zone.zone_rank == zone_rank[i]]["index"].values)
                    optional_ends = zone[zone.zone_rank == zone_rank[i + 1]][
                        "index"
                    ].values
                    end = (
                        distance_matrix.loc[stops, optional_ends]
                        .min()
                        .sort_values()
                        .index[0]
                    )
                    m = data.build_zone_data_model(idx, depot, stops, end)
                    if check_m_valid(m):
                        manager = pywrapcp.RoutingIndexManager(
                            len(m["distance_matrix"]), m["num_vehicles"], m["depot"]
                        )
                        routing = pywrapcp.RoutingModel(manager)
                        transit_callback_index = routing.RegisterTransitCallback(
                            distance_callback
                        )
                        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
                        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                        # GLOBAL_CHEAPEST_ARC 0.4620
                        search_parameters.first_solution_strategy = (
                            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC  #
                        )
                        solution = routing.SolveWithParameters(search_parameters)
                        solution = get_solution(manager, routing, solution)
                        result += list(np.array(m["sequence"])[solution[2:-1]])
                    else:
                        result += m["sequence"][1:-1]

                depot = result[-1]
                stops = list(zone[zone.zone_rank == zone_rank[-1]]["index"].values)

                m = data.build_zone_data_model_last(idx, depot, stops)
                if check_m_valid(m):
                    manager = pywrapcp.RoutingIndexManager(
                        len(m["distance_matrix"]), m["num_vehicles"], m["depot"]
                    )
                    routing = pywrapcp.RoutingModel(manager)
                    transit_callback_index = routing.RegisterTransitCallback(
                        distance_callback
                    )
                    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
                    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                    search_parameters.first_solution_strategy = (
                        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
                    )
                    solution = routing.SolveWithParameters(search_parameters)
                    solution = get_solution(manager, routing, solution)
                    result += list(np.array(m["sequence"])[solution[2:]])
                else:
                    result += m["sequence"][1:]

                sub = {"proposed": dict(zip(result, range(len(result))))}

            else:
                m = data.build_data_model_fix_end(idx)
                if check_m_valid(m):
                    manager = pywrapcp.RoutingIndexManager(
                        len(m["distance_matrix"]), m["num_vehicles"], m["depot"]
                    )
                    routing = pywrapcp.RoutingModel(manager)
                    transit_callback_index = routing.RegisterTransitCallback(
                        distance_callback
                    )
                    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
                    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                    search_parameters.first_solution_strategy = (
                        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
                    )
                    solution = routing.SolveWithParameters(search_parameters)
                    solution = get_solution(manager, routing, solution)
                else:
                    solution = list(range(len(m["distance_matrix"])))
                    solution[m["depot"]] = -1

                rank = np.array(solution).argsort()
                rank = [int(i) for i in rank]

                sub = {
                    "proposed": dict(
                        zip(
                            list(data.route_data[data[idx]]["stops"].keys()),
                            rank,
                        )
                    )
                }

            with open(
                f"{BASE_DIR}/data/model_apply_outputs/route/{route_id}", "wb"
            ) as f:
                pickle.dump(sub, f)

            with open(
                f"{BASE_DIR}/data/model_apply_outputs/CURRENT_PROCESSED_ROUTE", "w"
            ) as f:
                f.write(str(idx))

        # TODO random access one sub
        except:
            pass
