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
WORKSPACE_COMMIT = "537388a4d9548b612fa1e4b306c482c04b45c433"


def test_release_and_workspace_governance_are_declared() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    governance = data["tool"]["quant-workspace"]

    assert project["version"] == quant_agent.__version__ == "0.3.2"
    assert governance == {
        "layer": "orchestration",
        "schemas": [{"id": "standard/v2", "version": "2.0.0"}],
        "lock-files": ["requirements.lock"],
    }
    dependencies = "\n".join(project["dependencies"])
    assert "quant-lab.git@v0.3.1" in dependencies
    assert "quant-workspace.git@v0.3.1" in dependencies
    assert not re.search(r"git\+[^\s]+@(main|master|latest)(?:\b|$)", dependencies)


def test_lock_covers_internal_and_cross_python_dependencies() -> None:
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")

    assert "quant-lab.git@v0.3.1" in lock
    assert "quant-workspace.git@v0.3.1" in lock
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert LAB_COMMIT in readme and WORKSPACE_COMMIT in readme
    assert re.search(r'tomli==[^\s]+ ; python_version < "3\.11"', lock)
    assert "setuptools==" in lock
    assert not re.search(r"git\+[^\s]+@(main|master|latest)(?:\b|$)", lock)
