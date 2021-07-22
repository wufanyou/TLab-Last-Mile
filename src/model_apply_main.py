import os
from os import path
import glob
import json
import pickle


def load_pickle(file):
    with open(file, "rb") as f:
        output = pickle.load(f)
    return output


if __name__ == "__main__":
    # Return total of number of route in apply section
    os.system("python src/model_apply_check.py")

    # current_idx+2 is the start point
    # skip the potential broken data
    current_idx = -2
    BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))

    with open(f"{BASE_DIR}/data/model_apply_outputs/TOTAL_NUMBER_OF_ROUTE", "r") as f:
        total_idx = int(f.read())

    while current_idx + 1 < total_idx:
        os.system(f"python src/model_apply.py -i {current_idx+2}")
        try:
            with open(
                f"{BASE_DIR}/data/model_apply_outputs/CURRENT_PROCESSED_ROUTE", "r"
            ) as f:
                current_idx = int(f.read())
        except:
            current_idx += 1

    output = {}
    for f in glob.glob(f"{BASE_DIR}/data/model_apply_outputs/route/*"):
        route_id = f.split("/")[-1]
        try:
            output[route_id] = load_pickle(f)
        except:
            print(route_id)

    with open(f"{BASE_DIR}/data/model_apply_outputs/proposed_sequences.json", "w") as f:
        output = json.dumps(output)
        f.write(output)
