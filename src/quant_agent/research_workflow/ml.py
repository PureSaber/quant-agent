"""Frozen chronological model comparison and a single final holdout consumption."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant_agent.research_history import verify_citations

from .contracts import (
    digest,
    file_hash,
    identifier,
    load_dataset,
    read_json,
    seal_archive,
    timestamp,
    write_json,
)
from .models import explain_model, fit_model, predict_model, regression_metrics


def validate_plan(plan):
    if plan.get("schema") != "quant.ml-study/v1":
        raise ValueError("unsupported study schema")
    identifier(plan["study_id"])
    identifier(plan["family_id"])
    if not plan.get("hypothesis") or plan.get("selection_metric") != "mse":
        raise ValueError("hypothesis and validation MSE selection required")
    models = plan["models"]
    ids = [identifier(m["id"]) for m in models]
    if not ids or len({name.casefold() for name in ids}) != len(ids):
        raise ValueError("nonempty unique candidate IDs required")
    for model in models:
        allowed = (
            {"id", "kind", "alpha"}
            if model["kind"] == "ridge"
            else {"id", "kind", "neighbors"}
            if model["kind"] == "knn"
            else {"id", "kind"}
        )
        if set(model) - allowed:
            raise ValueError("unknown model fields")
    embargo = plan.get("embargo_seconds", 0)
    if type(embargo) is not int or embargo < 0:
        raise ValueError("embargo_seconds must be a nonnegative integer")
    train_start = timestamp(plan["train_start"])
    hold_start, hold_end = [timestamp(plan["holdout"][k]) for k in ("start", "end")]
    as_of = timestamp(plan["evaluation_as_of"])
    if not train_start < hold_start < hold_end <= as_of:
        raise ValueError("ordered training, holdout and evaluation timestamps required")
    previous = train_start
    if not plan["folds"]:
        raise ValueError("at least one validation fold required")
    for fold in plan["folds"]:
        start, end = timestamp(fold["start"]), timestamp(fold["end"])
        if not train_start < start < end <= hold_start or start < previous:
            raise ValueError("disjoint ordered validation folds before holdout required")
        previous = end
    canonical_plan = digest(plan)
    return canonical_plan


def split_rows(frame, plan, window, *, holdout=False):
    start, end = timestamp(window["start"]), timestamp(window["end"])
    cutoff = start - pd.Timedelta(seconds=plan.get("embargo_seconds", 0))
    eligible = (frame.decision_at >= timestamp(plan["train_start"])) & (frame.decision_at < cutoff)
    train_mask = eligible & (frame.target_known_at < cutoff)
    evaluation = (frame.decision_at >= start) & (frame.decision_at < end)
    observed = (
        evaluation
        & (frame.target_end <= end)
        & (frame.target_known_at <= timestamp(plan["evaluation_as_of"]))
    )
    # Validation targets must be available before selecting the final model.
    if not holdout:
        observed &= frame.target_known_at < timestamp(plan["holdout"]["start"])
    train, test = frame.loc[train_mask], frame.loc[observed]
    if len(train) < 3 or len(test) < 2:
        raise ValueError("insufficient eligible training/evaluation rows after label purge")
    audit = {
        "fit_at": cutoff.isoformat(),
        "training_rows": train.row_id.tolist(),
        "evaluation_rows": test.row_id.tolist(),
        "purged_training_rows": frame.loc[eligible & ~train_mask, "row_id"].tolist(),
        "unscored_evaluation_rows": frame.loc[evaluation & ~observed, "row_id"].tolist(),
        "max_training_label_known_at": train.target_known_at.max().isoformat(),
    }
    return train, test, audit


def _fit_predict(frame, plan, dataset, window, model_spec, root, label):
    train, test, audit = split_rows(frame, plan, window, holdout=label == "holdout")
    features = dataset["features"]
    fitted = fit_model(train[features], train.target, model_spec)
    predictions = predict_model(fitted, test[features])
    metrics = regression_metrics(test.target, predictions)
    name = f"models/{model_spec['id']}-{label}.json"
    write_json(
        root / name,
        {
            "model": fitted,
            "split": audit,
            "explanation": explain_model(fitted, features),
            "dataset_sha256": dataset["data_sha256"],
            "plan_sha256": digest(plan),
        },
    )
    rows = test[
        [
            "row_id",
            "asset",
            "decision_at",
            "target_start",
            "target_end",
            "target_known_at",
            "target",
        ]
    ].copy()
    rows["prediction"] = predictions
    rows["candidate"] = model_spec["id"]
    rows["partition"] = label
    rows["fit_at"] = audit["fit_at"]
    rows["model_artifact"] = name
    return rows, metrics, audit


def run_study(plan_path, dataset_path, output_root):
    """One study ID per durable output root. Failed/consumed runs cannot be overwritten.

    A new root is a new governance domain; this local guard is not an organization-wide
    anti-research-fraud service. The caller must preserve and review its complete ledger.
    """
    plan = read_json(plan_path)
    plan_sha = validate_plan(plan)
    root = Path(output_root) / plan["study_id"]
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", plan)
    attempts, predictions = [], []
    result = {
        "schema": "quant.ml-result/v1",
        "study_id": plan["study_id"],
        "family_id": plan["family_id"],
        "hypothesis": plan["hypothesis"],
        "plan_sha256": plan_sha,
        "status": "running",
        "execution_authorized": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_design": "batch chronological replay; not independent forward validation",
        "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
        "implementation_sha256": {
            name: file_hash(Path(__file__).parent / name)
            for name in ("ml.py", "models.py", "contracts.py")
        },
        "lab_references": plan.get("lab_references", []),
        "attempts": attempts,
        "holdout": {"status": "unconsumed"},
        "limitations": [
            "regression prediction metrics are not trading returns",
            "PIT/sample-kind provenance is supplied by the data producer",
            "one-use protection is local to the durable output root; never reset it to retune holdout",
        ],
    }
    try:
        if verify_citations(plan.get("lab_references", []))["invalid"]:
            raise ValueError("Lab reference citation verification failed")
        frame, dataset = load_dataset(dataset_path)
        write_json(
            root / "dataset.json",
            {
                **dataset,
                "schema": "quant.ml-dataset-reference/v1",
                "source_manifest": str(Path(dataset_path).resolve()),
            },
        )
        result["sample_kind"] = dataset["sample_kind"]
        result["dataset_sha256"] = dataset["data_sha256"]
        for model_spec in plan["models"]:
            for index, window in enumerate(plan["folds"]):
                attempt = {
                    "candidate": model_spec["id"],
                    "parameters": model_spec,
                    "fold": index,
                    "status": "running",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
                attempt_path = root / f"attempts/{model_spec['id']}-{index}"
                write_json(attempt_path / "started.json", attempt)
                try:
                    rows, metrics, audit = _fit_predict(
                        frame, plan, dataset, window, model_spec, root, f"validation-{index}"
                    )
                    predictions.append(rows)
                    attempt.update(status="completed", metrics=metrics, split=audit)
                except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
                    attempt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                attempts.append(attempt)
                attempt["finished_at"] = datetime.now(timezone.utc).isoformat()
                write_json(attempt_path / "terminal.json", attempt)
        if any(a["status"] != "completed" for a in attempts):
            result["status"] = "incomplete_family"
            result["selection"] = {
                "available": False,
                "reason": "all planned candidates/folds must complete; failures are retained",
            }
        else:
            scores = []
            for model_spec in plan["models"]:
                parts = [a["metrics"] for a in attempts if a["candidate"] == model_spec["id"]]
                count = sum(p["rows"] for p in parts)
                scores.append(
                    {
                        "candidate": model_spec["id"],
                        "validation_mse": sum(p["mse"] * p["rows"] for p in parts) / count,
                        "rows": count,
                    }
                )
            selected = min(scores, key=lambda s: (s["validation_mse"], s["candidate"]))["candidate"]
            result["selection"] = {
                "available": True,
                "selected": selected,
                "scores": scores,
                "basis": "pooled validation MSE only; candidate ID breaks ties",
            }
            # Durable claim precedes the first holdout fit/prediction and includes selection.
            write_json(
                root / "holdout-consumed.json",
                {
                    "plan_sha256": plan_sha,
                    "selection": result["selection"],
                    "policy": "never retry or tune on this holdout",
                },
            )
            result["holdout"] = {"status": "consumed_failed", "selected": selected}
            model_spec = next(m for m in plan["models"] if m["id"] == selected)
            rows, metrics, audit = _fit_predict(
                frame, plan, dataset, plan["holdout"], model_spec, root, "holdout"
            )
            predictions.append(rows)
            result["holdout"].update(status="consumed", metrics=metrics, split=audit)
            result["status"] = "completed"
    except (OSError, ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(root / "predictions.csv", index=False)
    write_json(root / "result.json", result)
    seal_archive(root)
    return result
