"""Fail-closed access to versioned standard run artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StandardArtifacts:
    """Validated contract metadata and the JSON artifacts used by the agent."""

    project: str
    schema_version: str
    profile: str
    strategy_ids: tuple[str, ...]
    config: dict[str, Any]
    metrics: dict[str, Any]
    contract: dict[str, Any]
    is_v2: bool


def _read_validated_json(base: Path, records: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Read a JSON artifact only after quant-lab validated its manifest record."""
    record = records[name]
    payload = json.loads((base / record.path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Validated standard/v2 {name} artifact must be a JSON object")
    return payload


def load_standard_artifacts(run_dir: Path) -> StandardArtifacts | None:
    """Validate the selected standard contract before exposing any consumable payload."""
    run_dir = Path(run_dir)
    v2_dir = run_dir / "standard" / "v2"
    v1_manifest = run_dir / "standard" / "run_manifest.json"
    if not v2_dir.exists() and not v1_manifest.is_file():
        return None

    from quant_lab import load_and_validate_standard_run
    from quant_lab.contracts_v2 import RunManifestV2

    manifest = load_and_validate_standard_run(run_dir)
    contract = {
        "schema_version": manifest.schema_version,
        "project": manifest.project,
        "run_id": manifest.run_id,
        "code_version": manifest.code_version,
        "dataset_snapshots": manifest.dataset_snapshots,
        "profile": getattr(manifest, "profile", "legacy-v1"),
        "validated": True,
    }
    if not isinstance(manifest, RunManifestV2):
        return StandardArtifacts(
            project=manifest.project,
            schema_version=manifest.schema_version,
            profile="legacy-v1",
            strategy_ids=(manifest.strategy,),
            config={},
            metrics={},
            contract=contract,
            is_v2=False,
        )

    records = {record.name: record for record in manifest.artifacts}
    return StandardArtifacts(
        project=manifest.project,
        schema_version=manifest.schema_version,
        profile=manifest.profile,
        strategy_ids=tuple(manifest.strategy_ids),
        config=_read_validated_json(v2_dir, records, "config"),
        metrics=_read_validated_json(v2_dir, records, "metrics"),
        contract=contract,
        is_v2=True,
    )


def metric_rows(metrics: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    """Return a validated list-of-objects metric section."""
    value = metrics.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"standard/v2 metrics.{key} must be a list of objects")
    return [dict(row) for row in value]


def flat_metric_rows(metrics: Mapping[str, Any], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Project selected top-level metrics into one rule-engine row."""
    row = {key: metrics[key] for key in keys if key in metrics}
    return [row] if row else []


def string_list(value: Any, field_name: str) -> list[str]:
    """Validate a JSON string list used by v2 config or manifest projections."""
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"standard/v2 {field_name} must be a list of non-empty strings")
    return list(value)
