import json

import pytest
from quant_lab.research import canonical, file_hash
from quant_lab.trials import TrialRegistry

from quant_agent.objective_review import main, review_objective


def setup(tmp_path, *, include_objective=True, evidence=True):
    registry = TrialRegistry(tmp_path / "r.db")
    goal = {
        "mechanism": "cash efficiency",
        "benchmark_id": "cash",
        "primary": {"metric": "net_return", "direction": "at_least", "threshold": 0.01},
        "capital_range": [1000, 10000],
        "max_drawdown": 0.2,
        "max_turnover": 10,
        "max_cost_rate": 0.01,
        "min_capacity": 10000,
        "data_conditions": ["pit_complete"],
        "stop_conditions": ["data_break"],
    }
    definition = {
        "hypothesis": "h",
        "parameters": [{"p": 1}],
        "code_identity": "frozen",
        "selection_rule": "all",
    }
    if include_objective:
        definition["objective"] = goal
    registry.register("s", definition)
    attempt = registry.start("s", {"p": 1}, context={"artifact_root": str(tmp_path)})
    result = {"attempt_id": attempt, "study_id": "s", "artifacts": {}}
    if evidence:
        result["investment_evidence"] = {
            "benchmark_id": "cash",
            "net_return": 0.03,
            "drawdown_magnitude": 0.3,
            "turnover": 5,
            "cost_rate": 0.005,
            "capacity": 50000,
            "capital": 5000,
            "data_conditions": {"pit_complete": True},
            "stop_conditions": {"data_break": False},
        }
    path = tmp_path / "result.json"
    path.write_text(canonical(result), encoding="utf-8")
    registry.finish(attempt, "completed", {"result": "result.json", "sha256": file_hash(path)})
    failed = registry.start("s", {"p": 1})
    registry.finish(failed, "failed", {"reason": "missing quotes"})
    return registry


def test_readonly_explanation_separates_risk_failure_and_keeps_failed_attempts(
    tmp_path, monkeypatch, capsys
):
    registry = setup(tmp_path)
    before = registry.path.read_bytes()
    result = review_objective(registry.path, "s")
    assert registry.path.read_bytes() == before
    assert len(result["reviews"]) == 2
    review = result["reviews"][0]
    assert review["evaluation"]["status"] == "failed"
    assert [r["category"] for r in review["evaluation"]["checks"] if r["passed"] is False] == [
        "risk"
    ]
    assert review["citation"]["result_sha256"]
    assert result["reviews"][1]["attempt_status"] == "failed"
    monkeypatch.setattr(
        "sys.argv", ["quant-objective-review", "--registry", str(registry.path), "--study-id", "s"]
    )
    main()
    assert json.loads(capsys.readouterr().out)["study_id"] == "s"
    (tmp_path / "result.json").write_text("tampered")
    with pytest.raises(ValueError, match="hash"):
        review_objective(registry.path, "s")


@pytest.mark.parametrize(
    "objective,evidence,status",
    [(False, True, "not_preregistered"), (True, False, "insufficient_evidence")],
)
def test_missing_ex_ante_goal_or_evidence_cannot_be_promoted(tmp_path, objective, evidence, status):
    registry = setup(tmp_path, include_objective=objective, evidence=evidence)
    result = review_objective(registry.path, "s")
    actual = result.get("status") or result["reviews"][0]["evaluation"]["status"]
    assert actual == status
    with pytest.raises(ValueError, match="missing"):
        review_objective(registry.path, "unknown")


def test_external_result_reference_rejected(tmp_path):
    registry = TrialRegistry(tmp_path / "r.db")
    # Reuse the valid definition without importing any strategy implementation.
    other = tmp_path / "other"
    other.mkdir()
    spec = setup(other).definition("s")["definition"]
    registry.register("s", spec)
    attempt = registry.start("s", {"p": 1}, context={"artifact_root": str(tmp_path)})
    registry.finish(attempt, "completed", {"result": "../outside.json", "sha256": "a" * 64})
    with pytest.raises(ValueError, match="escapes"):
        review_objective(registry.path, "s")
