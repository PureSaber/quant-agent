import hashlib
import json
from copy import deepcopy
from pathlib import Path

from quant_lab.trials import TrialRegistry

from quant_agent.research_assistant import validate_proposal
from quant_agent.research_history import (
    main,
    research_advice,
    research_history,
    validate_artifacts,
    verify_citations,
)

FIXTURE = Path(__file__).parent / "fixtures" / "research_history_eval.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _history(tmp_path: Path) -> tuple[Path, Path, dict]:
    case = json.loads(FIXTURE.read_text(encoding="utf-8"))
    root = tmp_path / "studies"
    study = root / "ashare"
    completed_id = "completed-attempt"
    failed_id = "failed-attempt"
    completed_dir = study / "attempts" / completed_id
    completed_dir.mkdir(parents=True)
    metrics_path = completed_dir / "metrics.json"
    metrics_path.write_text('{"total_return":0.08,"sharpe":1.1}', encoding="utf-8")
    document = {
        "schema_version": "quant.research-study/v1",
        "study_id": case["template"]["study_id"],
        "recipe": case["template"],
        "results": [
            {
                "attempt_id": completed_id,
                "status": "completed",
                "candidate": {
                    "name": "momentum_lowvol",
                    "factors": case["template"]["factors"],
                    "strategy": case["template"]["strategy"],
                },
                "metrics": {"total_return": 0.08, "sharpe": 1.1},
                "artifacts": {"metrics.json": _digest(metrics_path)},
            },
            {
                "attempt_id": failed_id,
                "status": "failed",
                "candidate": {"name": "monthly_rebalance"},
                "error_type": "ValueError",
                "error": "价格历史不足，无法完成60日预热",
            },
        ],
    }
    study.mkdir(exist_ok=True)
    study_path = study / "study.json"
    study_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return root, study_path, case


def test_realistic_case_has_valid_recipe_and_exact_source_quotes() -> None:
    case = json.loads(FIXTURE.read_text(encoding="utf-8"))
    validated = validate_proposal(case["proposal"], case["source_text"], case["template"])
    assert validated["factors"] == case["proposal"]["factors"]
    assert validated["costs"] == case["template"]["costs"]
    assert validated["risk"] == case["template"]["risk"]


def test_history_verifies_citations_artifacts_failures_and_minimum_control(tmp_path: Path) -> None:
    root, study_path, case = _history(tmp_path)
    original_recipe = deepcopy(case["template"])
    advice = research_advice(
        case["query"],
        [study_path],
        recipe=case["template"],
    )
    assert len(advice["matches"]) == 1
    match = advice["matches"][0]
    assert match["completed_results"][0]["artifact_validation"]["status"] == "verified"
    assert match["failures"][0]["error"] == "价格历史不足，无法完成60日预热"
    assert advice["citation_verification"]["accuracy"] == 1.0
    assert advice["minimum_comparison"]["control"]["attempt_id"] == "completed-attempt"
    assert (
        advice["minimum_comparison"]["locked_fields"]["costs"]["value"] == original_recipe["costs"]
    )
    assert advice["minimum_comparison"]["locked_fields"]["risk"]["value"] == original_recipe["risk"]
    assert advice["boundaries"] == {
        "read_only": True,
        "recipe_fields_modified": [],
        "automatic_run": False,
        "online_model_called": False,
    }
    assert case["template"] == original_recipe
    assert (
        research_history(case["query"], studies_root=root)["matches"][0]["study_id"]
        == match["study_id"]
    )
    assert (
        research_advice(case["query"], studies_root=str(root))["matches"][0]["study_id"]
        == match["study_id"]
    )


def test_tampering_invalidates_artifacts_and_citations(tmp_path: Path) -> None:
    root, study_path, case = _history(tmp_path)
    advice = research_history(case["query"], studies_root=root, recipe=case["template"])
    artifact = root / "ashare" / "attempts" / "completed-attempt" / "metrics.json"
    artifact.write_text('{"total_return":99}', encoding="utf-8")
    changed = research_history(case["query"], studies_root=root, recipe=case["template"])
    validation = changed["matches"][0]["completed_results"][0]["artifact_validation"]
    assert validation["status"] == "invalid"
    assert changed["minimum_comparison"]["control"] is None
    study_path.write_text(study_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    recheck = verify_citations(advice["citations"])
    assert recheck["accuracy"] == 0.0


def test_sqlite_terminal_events_are_read_only_and_cited(tmp_path: Path) -> None:
    case = json.loads(FIXTURE.read_text(encoding="utf-8"))
    database = tmp_path / "experiments.db"
    registry = TrialRegistry(database)
    parameters = {"factors": {"momentum_20d": 1}}
    registry.register(
        "sqlite-momentum",
        {
            "hypothesis": "20日动量成本后收益",
            "parameters": [parameters],
            "code_identity": {"quant-factors": "abc"},
            "selection_rule": "fixed before replay",
        },
    )
    completed = registry.start("sqlite-momentum", parameters)
    attempt_dir = tmp_path / "studies" / "sqlite" / "attempts" / completed
    attempt_dir.mkdir(parents=True)
    artifact = attempt_dir / "metrics.json"
    artifact.write_text('{"sharpe":0.7}', encoding="utf-8")
    registry.finish(
        completed,
        "completed",
        {
            "metrics": {"sharpe": 0.7},
            "candidate": parameters,
            "artifacts": {"metrics.json": _digest(artifact)},
        },
    )
    failed = registry.start("sqlite-momentum", parameters)
    registry.finish(failed, "failed", {"error": "样本覆盖不足", "error_type": "ValueError"})
    before = database.read_bytes()
    advice = research_history(
        "动量成本",
        database=database,
        studies_root=tmp_path / "studies",
        recipe=case["template"],
    )
    assert advice["citation_verification"]["accuracy"] == 1.0
    assert (
        advice["matches"][0]["completed_results"][0]["artifact_validation"]["status"] == "verified"
    )
    assert advice["matches"][0]["failures"][0]["error"] == "样本覆盖不足"
    assert database.read_bytes() == before


def test_artifact_path_cannot_escape_attempt_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    result = validate_artifacts({"../outside.txt": _digest(outside)}, tmp_path / "attempt")
    assert result["status"] == "invalid"
    assert "escapes" in result["items"][0]["reason"]


def test_history_cli_writes_json_without_running_research(tmp_path: Path) -> None:
    root, _, case = _history(tmp_path)
    output = tmp_path / "advice.json"
    assert main([case["query"], "--studies-root", str(root), "--output", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["matches"][0]["study_id"] == case["template"]["study_id"]
    assert result["boundaries"]["automatic_run"] is False
