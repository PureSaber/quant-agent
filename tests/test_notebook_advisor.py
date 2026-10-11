import hashlib
import json

import pytest

from quant_agent.notebook_advisor import propose_notebook
from tests.test_workbench_assistant import context


def test_proposals_require_source_identity_and_evidence(monkeypatch):
    monkeypatch.setenv("QUANT_AGENT_LLM_OK", "1")
    nb = {"nbformat": 4, "cells": [{"cell_type": "code", "source": "frame['volume']"}]}
    answer = {
        "findings": [{"text": "缺少字段", "citations": ["E1"], "quotes": ["缺少成交量字段"]}],
        "next_steps": ["确认字段后重新运行"],
        "patches": [
            {
                "cell": 0,
                "before_sha256": hashlib.sha256(b"frame['volume']").hexdigest(),
                "source": "assert 'volume' in frame, '需要成交量'",
                "reason": "提前报告缺失",
                "citations": ["E1"],
            }
        ],
    }
    report = propose_notebook(context(), nb, model="test", invoke=lambda *_: json.dumps(answer))
    assert report["patches"] == answer["patches"]
    assert not report["model_invoked"]
    assert nb["cells"][0]["source"] == "frame['volume']"
    answer["patches"][0]["before_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        propose_notebook(context(), nb, model="test", invoke=lambda *_: json.dumps(answer))


def test_offline_does_not_generate_or_execute_code():
    report = propose_notebook(context(), {"nbformat": 4, "cells": []})
    assert report["patches"] == []
    assert not report["model_invoked"]


@pytest.mark.parametrize(
    "change",
    [
        "unknown_evidence",
        "duplicate_cell",
        "unknown_cell",
        "extra_field",
        "oversize_source",
        "invented_quote",
    ],
)
def test_rejects_unreviewable_model_proposals(monkeypatch, change):
    monkeypatch.setenv("QUANT_AGENT_LLM_OK", "1")
    notebook = {"nbformat": 4, "cells": [{"cell_type": "code", "source": "pass"}]}
    patch = {
        "cell": 0,
        "before_sha256": hashlib.sha256(b"pass").hexdigest(),
        "source": "print('proposal only')",
        "reason": "增加诊断",
        "citations": ["E1"],
    }
    answer = {
        "findings": [{"text": "缺少字段", "citations": ["E1"], "quotes": ["缺少成交量字段"]}],
        "next_steps": [],
        "patches": [patch],
    }
    if change == "unknown_evidence":
        patch["citations"] = ["E999"]
    elif change == "duplicate_cell":
        answer["patches"].append(dict(patch))
    elif change == "unknown_cell":
        patch["cell"] = 7
    elif change == "extra_field":
        patch["execute"] = True
    elif change == "oversize_source":
        patch["source"] = "x" * 20_001
    else:
        answer["findings"][0]["quotes"] = ["没有这条证据"]
    with pytest.raises(ValueError):
        propose_notebook(context(), notebook, model="test", invoke=lambda *_: json.dumps(answer))
    assert notebook["cells"][0]["source"] == "pass"


def test_model_opt_in_is_required_before_provider_call(monkeypatch):
    monkeypatch.delenv("QUANT_AGENT_LLM_OK", raising=False)

    def should_not_invoke(*_):
        pytest.fail("provider called before explicit opt-in")

    with pytest.raises(ValueError, match="QUANT_AGENT_LLM_OK"):
        propose_notebook(
            context(), {"nbformat": 4, "cells": []}, model="test", invoke=should_not_invoke
        )
