# import glob
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import extract_nested, scaling_law

USE_LCBENCH = True

if USE_LCBENCH:
    LEARNING_CURVE_DATA_PATH = Path(
        "/Users/marcelhofmann/UAM_Deep_Learning/TFM_Implementation/LCBench/data/exported"
    )
else:
    LEARNING_CURVE_DATA_PATH = Path(
        "/Users/marcelhofmann/Deep_Learning_UAM/Thesis/NATS-Bench/curves"
    )

SCALING_LAW_DATA_PATH = Path(
    "/Users/marcelhofmann/UAM_Deep_Learning/TFM_Implementation/scaling_law_derivation/scaling_laws"
)

X_VALUE_KEY = ("architecture", "param_count")
Y_VALUE_KEY = (
    "summary",
    "eval",
    "best_accuracy",
    "value",
)


# build lookup of all available scaling laws
scaling_laws = {}

for sl_path in SCALING_LAW_DATA_PATH.glob("*.npz"):

    # filename:
    # sl__dataset=adult__in=architecture.param_count__out=summary.eval.best_accuracy.value.npz

    stem = sl_path.stem

    parts = stem.split("__")

    dataset = parts[1].split("=", 1)[1]
    input_name = parts[2].split("=", 1)[1]
    output_name = parts[3].split("=", 1)[1]

    data = np.load(sl_path)

    scaling_laws[(dataset, input_name, output_name)] = {
        "popt": data["popt"].tolist(),
        "boot_params": data["boot_params"].tolist(),
    }

# scaling laws have the json paths to x and y in them, reconstruct them
input_name = ".".join(X_VALUE_KEY)
output_name = ".".join(Y_VALUE_KEY)

# walk through all learning curves
for json_path in tqdm(LEARNING_CURVE_DATA_PATH.glob("*.json")):
    # first part divided by "__" is dataset, e.g.: KDDCup09_appetency__config_386_budget_50_seed_2
    dataset_name = json_path.stem.split("__")[0]

    key = (dataset_name, input_name, output_name)

    if key not in scaling_laws:
        print(f"No scaling law found for {key}")
        continue

    with open(json_path, "r") as f:
        sample = json.load(f)

    x_value = extract_nested(sample, X_VALUE_KEY)
    y_inf, A, alpha = scaling_laws[key]["popt"]
    # get mode, e.g. accuracy leads to curve with positive slope
    mode = "max" if "accuracy" in output_name else "min"
    prediction = scaling_law(x_value, y_inf, A, alpha, mode=mode)

    sample.setdefault("scaling_laws", {})
    sample["scaling_laws"][output_name] = {
        "input": input_name,
        "prediction": float(prediction),
    }

    with open(json_path, "w") as f:
        json.dump(sample, f, indent=2)
