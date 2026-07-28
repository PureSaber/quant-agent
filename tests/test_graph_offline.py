from pathlib import Path

from quant_agent.graph import run_review
from tests.fixtures import write_multifactor_fixture


def test_graph_offline_writes_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_multifactor_fixture(run_dir)
    log_dir = tmp_path / "experiment-log"
    log_dir.mkdir()

    result = run_review(
        project="multifactor",
        run_dir=run_dir,
        offline=True,
    )

    assert result["rule_passed"] is True
    assert Path(result["manifest_path"]).is_file()
    manifest = Path(result["manifest_path"]).read_text(encoding="utf-8")
    assert "review_manifest" in manifest or "reviewed_at" in manifest
    assert Path(result["report_path"]).is_file()
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "forecast_score" in report
    assert "Offline review" in report or "Rule findings" in report
