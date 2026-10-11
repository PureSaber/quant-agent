"""Auditable regression baselines with training-only imputation and scaling.

All fitted state is portable JSON. No executable pickle or hidden global fit.
"""

import numpy as np


def fit_model(values, targets, spec):
    x, y = np.asarray(values, dtype=float), np.asarray(targets, dtype=float)
    if len(x) < 3 or not np.isfinite(y).all() or np.isnan(x).all(axis=0).any():
        raise ValueError("at least three training rows and observed values per feature required")
    kind = spec["kind"]
    if kind not in {"mean", "ridge", "knn"}:
        raise ValueError("model kind must be mean, ridge or knn")
    median = np.nanmedian(x, axis=0)
    filled = np.where(np.isnan(x), median, x)
    center, scale = filled.mean(axis=0), filled.std(axis=0)
    scale[scale == 0] = 1
    z = (filled - center) / scale
    model = {
        "kind": kind,
        "median": median.tolist(),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "train_rows": len(y),
        "intercept": float(y.mean()),
    }
    if kind == "ridge":
        alpha = spec.get("alpha", 1.0)
        if isinstance(alpha, bool) or not np.isfinite(alpha) or alpha < 0:
            raise ValueError("ridge alpha must be finite and nonnegative")
        # Augmented least squares keeps alpha=0 stable even for rank-deficient inputs.
        augmented = np.vstack([z, np.sqrt(alpha) * np.eye(z.shape[1])])
        response = np.r_[y - y.mean(), np.zeros(z.shape[1])]
        model.update(
            alpha=float(alpha),
            coefficients=np.linalg.lstsq(augmented, response, rcond=None)[0].tolist(),
        )
    elif kind == "knn":
        neighbors = spec.get("neighbors", 5)
        if type(neighbors) is not int or not 1 <= neighbors <= len(y):
            raise ValueError("knn neighbors must be an integer within training rows")
        model.update(neighbors=neighbors, training_features=z.tolist(), training_targets=y.tolist())
    return model


def predict_model(model, values):
    x = np.asarray(values, dtype=float)
    x = np.where(np.isnan(x), model["median"], x)
    z = (x - model["center"]) / model["scale"]
    if model["kind"] == "mean":
        return np.full(len(x), model["intercept"])
    if model["kind"] == "ridge":
        return model["intercept"] + z @ np.asarray(model["coefficients"])
    # Stable distance sorting gives deterministic tie handling.
    train = np.asarray(model["training_features"])
    labels = np.asarray(model["training_targets"])
    return np.asarray(
        [
            labels[
                np.argsort(((train - row) ** 2).sum(axis=1), kind="stable")[: model["neighbors"]]
            ].mean()
            for row in z
        ]
    )


def explain_model(model, features):
    result = {
        "kind": model["kind"],
        "features": list(features),
        "preprocessing_fit": "training rows only",
        "interpretation": "descriptive associations, not causal effects",
        "training_rows": model["train_rows"],
    }
    if model["kind"] == "ridge":
        result["standardized_coefficients"] = dict(zip(features, model["coefficients"]))
        result["raw_unit_coefficients"] = dict(
            zip(features, (np.asarray(model["coefficients"]) / model["scale"]).tolist())
        )
    elif model["kind"] == "knn":
        result["method"] = (
            "equal-weight average of nearest training labels in training-standardized space"
        )
        result["neighbors"] = model["neighbors"]
    else:
        result["training_target_mean"] = model["intercept"]
    return result


def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if not len(actual) or not np.isfinite(predicted).all():
        raise ValueError("nonempty finite predictions required")
    residual = actual - predicted
    denominator = ((actual - actual.mean()) ** 2).sum()
    return {
        "rows": len(actual),
        "mse": float(np.mean(residual**2)),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1 - (residual**2).sum() / denominator) if denominator else None,
    }
