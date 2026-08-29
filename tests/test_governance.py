from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import quant_agent

ROOT = Path(__file__).resolve().parents[1]
LAB_COMMIT = "27489d270e132adbec1bced93eb2ae84ad5e1a9b"
WORKSPACE_COMMIT = "1a9134ac329704060a3ae96cc81e31db481a938f"


def test_release_and_workspace_governance_are_declared() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    governance = data["tool"]["quant-workspace"]

    assert project["version"] == quant_agent.__version__ == "0.3.1"
    assert governance == {
        "layer": "orchestration",
        "schemas": [{"id": "standard/v2", "version": "2.0.0"}],
        "lock-files": ["requirements.lock"],
    }
    dependencies = "\n".join(project["dependencies"])
    assert f"quant-lab.git@{LAB_COMMIT}" in dependencies
    assert f"quant-workspace.git@{WORKSPACE_COMMIT}" in dependencies
    assert not re.search(r"git\+[^\s]+@(main|master|latest)(?:\b|$)", dependencies)


def test_lock_covers_internal_and_cross_python_dependencies() -> None:
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")

    assert f"quant-lab.git@{LAB_COMMIT}" in lock
    assert f"quant-workspace.git@{WORKSPACE_COMMIT}" in lock
    assert 'tomli==2.2.1 ; python_version < "3.11"' in lock
    assert "setuptools==" in lock
    assert not re.search(r"git\+[^\s]+@(main|master|latest)(?:\b|$)", lock)
