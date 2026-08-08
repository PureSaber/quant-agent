from pathlib import Path

from quant_agent.nodes.pipeline import _resolve_notes_root


def test_resolve_notes_from_workspace_env(tmp_path: Path, monkeypatch) -> None:
    notes = tmp_path / "quant-research-notes"
    notes.mkdir()
    (notes / "experiment-log").mkdir()
    monkeypatch.setenv("QUANT_WORKSPACE_ROOT", str(tmp_path))
    run_dir = tmp_path / "quant-lab" / "runs" / "r1"
    run_dir.mkdir(parents=True)
    assert _resolve_notes_root(run_dir) == notes
