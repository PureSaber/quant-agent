from pathlib import Path

import pytest
import yaml

from quant_agent.nodes.pipeline import _resolve_notes_root


def test_resolve_notes_from_workspace_env(tmp_path: Path, monkeypatch) -> None:
    notes = tmp_path / "quant-research-notes"
    notes.mkdir()
    (notes / "experiment-log").mkdir()
    monkeypatch.setenv("QUANT_WORKSPACE_ROOT", str(tmp_path))
    run_dir = tmp_path / "quant-lab" / "runs" / "r1"
    run_dir.mkdir(parents=True)
    assert _resolve_notes_root(run_dir) == notes


def test_resolve_notes_via_quant_workspace_package(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("quant_workspace")
    root = tmp_path
    notes = root / "quant-research-notes"
    notes.mkdir()
    (notes / "experiment-log").mkdir()
    cfg_dir = root / "quant-workspace" / "configs"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "default.workspace.yaml").write_text(
        yaml.safe_dump(
            {
                "root": str(root),
                "projects": {
                    "quant-research-notes": {"repo": "quant-research-notes", "notes": "experiment-log"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("QUANT_WORKSPACE_ROOT", str(root))
    run_dir = root / "a-share-multifactor" / "outputs" / "r1"
    run_dir.mkdir(parents=True)
    assert _resolve_notes_root(run_dir) == notes.resolve()
