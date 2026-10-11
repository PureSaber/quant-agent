"""Versioned read-only input contracts and content-addressed artifact references."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_KINDS = {"synthetic", "retrospective", "historical_pit", "independent_forward"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return (
            hashlib.file_digest(stream, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else hashlib.sha256(stream.read()).hexdigest()
        )


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value):
        raise ValueError("identifier must be 1-100 safe ASCII characters")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10)
    }
    if value.endswith(".") or value.split(".")[0].upper() in reserved:
        raise ValueError("identifier must be portable across supported filesystems")
    return value


def timestamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("an explicit timezone-aware timestamp is required")
    return result.tz_convert("UTC")


def relative_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError("artifact path escapes its root")
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    """Exclusive create: published artifacts are never silently overwritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def load_dataset(manifest_path):
    manifest_path = Path(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    spec = json.loads(manifest_bytes.decode("utf-8-sig"))
    if spec.get("schema") != "quant.ml-dataset/v1":
        raise ValueError("unsupported dataset schema")
    if spec.get("sample_kind") not in SAMPLE_KINDS or not spec.get("provenance"):
        raise ValueError("sample_kind and provenance are required; timestamps are not PIT proof")
    if spec["sample_kind"] == "independent_forward":
        raise ValueError("batch model fitting cannot claim independent_forward validation")
    features = spec.get("features")
    reserved = {
        "row_id",
        "asset",
        "decision_at",
        "target_start",
        "target_end",
        "target_known_at",
        "target",
    }
    if not features or len(set(features)) != len(features):
        raise ValueError("distinct features required")
    for name in features:
        identifier(name)
        if name in reserved or name.endswith("__known_at"):
            raise ValueError("reserved feature name")
    if spec.get("target_kind") != "regression":
        raise ValueError("v1 supports tabular regression targets only")
    path = relative_file(manifest_path.parent, spec["data_file"])
    data_bytes = path.read_bytes()
    if hashlib.sha256(data_bytes).hexdigest() != spec["data_sha256"]:
        raise ValueError("dataset SHA256 mismatch")
    frame = pd.read_csv(io.BytesIO(data_bytes), dtype={"row_id": str, "asset": str})
    needed = reserved | set(features) | {f"{name}__known_at" for name in features}
    if not needed <= set(frame):
        raise ValueError(f"missing dataset columns: {sorted(needed - set(frame))}")
    if frame.empty or frame.row_id.isna().any() or frame.row_id.duplicated().any():
        raise ValueError("nonempty unique row IDs required")
    if frame.asset.isna().any() or (frame.asset.str.strip() == "").any():
        raise ValueError("asset identity required")
    for column in ["decision_at", "target_start", "target_end", "target_known_at"] + [
        f"{f}__known_at" for f in features
    ]:
        frame[column] = pd.to_datetime([timestamp(v) for v in frame[column]], utc=True)
    if frame.duplicated(["asset", "decision_at"]).any():
        raise ValueError("duplicate asset/decision timestamp")
    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="raise")
        if np.isinf(frame[feature]).any():
            raise ValueError("infinite features are not missing values")
        if (frame[f"{feature}__known_at"] > frame.decision_at).any():
            raise ValueError(f"feature {feature} became known after decision")
    if not np.isfinite(frame.target.astype(float)).all():
        raise ValueError("finite observed regression targets required")
    if not (
        (frame.decision_at <= frame.target_start)
        & (frame.target_start < frame.target_end)
        & (frame.target_end <= frame.target_known_at)
    ).all():
        raise ValueError("require decision <= target_start < target_end <= target_known_at")
    frame = frame.sort_values(["decision_at", "asset", "row_id"]).reset_index(drop=True)
    return frame, {**spec, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}


def verify_archive(root):
    root = Path(root)
    manifest = read_json(root / "archive.json")
    expected = manifest["artifacts"]
    actual = {str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file()}
    if actual != set(expected) | {"archive.json"}:
        raise ValueError("archive file inventory mismatch")
    for relative, sha in expected.items():
        if file_hash(relative_file(root, relative)) != sha:
            raise ValueError(f"archive digest mismatch: {relative}")
    return manifest


def seal_archive(root):
    root = Path(root)
    files = {
        str(p.relative_to(root)).replace("\\", "/"): file_hash(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    write_json(root / "archive.json", {"schema": "quant.research-archive/v1", "artifacts": files})
    return file_hash(root / "archive.json")
