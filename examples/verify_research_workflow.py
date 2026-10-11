"""Independent reconstruction of fitted models and every archived prediction.

Uses scalar medians/moments, direct ridge normal equations and distance ordering;
does not call the production fitting, prediction, split or metrics functions.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd


def verify(root):
    spec = json.loads((root / "dataset.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    output = root / "runs" / plan["study_id"]
    features = spec["features"]
    data = pd.read_csv(root / spec["data_file"]).set_index("row_id")
    predictions = pd.read_csv(output / "predictions.csv")
    maximum = 0.0
    checked = 0
    for model_file in sorted((output / "models").glob("*.json")):
        document = json.loads(model_file.read_text())
        model, split = document["model"], document["split"]
        train = data.loc[split["training_rows"]]
        test = data.loc[split["evaluation_rows"]]
        # Independently verify the purge contract from the archived raw timestamps.
        cutoff = pd.Timestamp(split["fit_at"])
        assert all(pd.Timestamp(v) < cutoff for v in train.target_known_at)
        assert all(pd.Timestamp(v) < cutoff for v in train.decision_at)
        assert all(pd.Timestamp(v) >= cutoff for v in test.decision_at)
        columns = []
        medians, centers, scales = [], [], []
        for feature in features:
            median = statistics.median([v for v in train[feature] if not math.isnan(v)])
            column = [median if math.isnan(v) else v for v in train[feature]]
            center = statistics.mean(column)
            scale = statistics.pstdev(column) or 1.0
            medians.append(median)
            centers.append(center)
            scales.append(scale)
            columns.append([(v - center) / scale for v in column])
        z = np.array(columns).T
        y = train.target.to_numpy()
        intercept = statistics.mean(y)
        assert np.allclose(model["median"], medians)
        assert np.allclose(model["center"], centers)
        assert np.allclose(model["scale"], scales)
        if model["kind"] == "ridge":
            coefficients = np.linalg.solve(
                z.T @ z + model["alpha"] * np.eye(len(features)), z.T @ (y - intercept)
            )
            assert np.allclose(model["coefficients"], coefficients)
        reference = []
        for _, row in test.iterrows():
            scaled = [
                (medians[i] if pd.isna(row[f]) else row[f]) - centers[i]
                for i, f in enumerate(features)
            ]
            scaled = [value / scales[i] for i, value in enumerate(scaled)]
            if model["kind"] == "ridge":
                value = intercept + sum(a * b for a, b in zip(scaled, coefficients))
            elif model["kind"] == "knn":
                indices = sorted(
                    range(len(z)), key=lambda i: sum((a - b) ** 2 for a, b in zip(z[i], scaled))
                )[: model["neighbors"]]
                value = statistics.mean([y[i] for i in indices])
            else:
                value = intercept
            reference.append(value)
        actual = (
            predictions[predictions.model_artifact == f"models/{model_file.name}"]
            .set_index("row_id")
            .loc[test.index]
            .prediction.to_numpy()
        )
        error = float(np.max(np.abs(actual - reference)))
        maximum = max(maximum, error)
        assert error < 1e-10
        checked += len(actual)
    return {
        "status": "passed",
        "independently_reconstructed_predictions": checked,
        "maximum_absolute_error": maximum,
        "method": "scalar preprocessing; normal-equation ridge; scalar KNN distances",
        "does_not_establish": "market profitability, PIT provenance or independent forward validity",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
