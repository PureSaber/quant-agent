from __future__ import annotations

import json
from pathlib import Path

from quant_agent import cli


def test_detect_command_reports_project_and_unknown(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "detect_project", lambda _path: "a-share-multifactor")
    assert cli.main(["detect", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "a-share-multifactor"

    monkeypatch.setattr(cli, "detect_project", lambda _path: None)
    assert cli.main(["detect", str(tmp_path)]) == 1
    assert capsys.readouterr().err.strip() == "unknown"


def test_run_command_requires_a_detectable_project(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "detect_project", lambda _path: None)
    assert cli.main(["run", "--run-dir", str(tmp_path)]) == 1
    assert "Could not detect project" in capsys.readouterr().err


def test_run_command_json_passes_paths_and_llm_mode(tmp_path: Path, monkeypatch, capsys) -> None:
    config = tmp_path / "run.yaml"
    agent_config = tmp_path / "agent.yaml"
    captured = {}

    def fake_review(**kwargs):
        captured.update(kwargs)
        return {
            "report_path": "report.md",
            "manifest_path": "review_manifest.json",
            "rule_passed": True,
            "rule_findings": [{"code": "info"}],
        }

    monkeypatch.setattr(cli, "run_review", fake_review)
    result = cli.main(
        [
            "run",
            "--project",
            "multifactor",
            "--run-dir",
            str(tmp_path),
            "--config",
            str(config),
            "--agent-config",
            str(agent_config),
            "--llm",
            "--json",
        ]
    )

    assert result == 0
    assert captured == {
        "project": "multifactor",
        "run_dir": tmp_path,
        "config_path": config,
        "agent_config_path": agent_config,
        "offline": False,
    }
    assert json.loads(capsys.readouterr().out) == {
        "report_path": "report.md",
        "manifest_path": "review_manifest.json",
        "rule_passed": True,
        "finding_count": 1,
    }


def test_run_command_text_returns_two_for_failed_rules(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "detect_project", lambda _path: "a-share-multifactor")
    monkeypatch.setattr(
        cli,
        "run_review",
        lambda **_kwargs: {
            "report_path": "report.md",
            "manifest_path": "review_manifest.json",
            "rule_passed": False,
            "rule_findings": [],
        },
    )

    assert cli.main(["run", "--run-dir", str(tmp_path), "--offline"]) == 2
    output = capsys.readouterr().out
    assert "Review written: report.md" in output
    assert "Findings: 0 (passed=False)" in output
