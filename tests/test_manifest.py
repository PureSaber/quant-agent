import json
from pathlib import Path

from quant_agent.graph import run_review
from tests.fixtures import write_multifactor_fixture


def test_manifest_has_findings_array(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_multifactor_fixture(run_dir)
    (tmp_path / "experiment-log").mkdir()

    result = run_review(project="multifactor", run_dir=run_dir, offline=True)
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert "findings" in manifest
    assert isinstance(manifest["findings"], list)
    assert manifest["finding_count"] == len(manifest["findings"])
