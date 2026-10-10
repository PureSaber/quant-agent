import hashlib
import json

import pytest

from quant_agent.workbench_assistant import advise


def context():
    text = "预检失败：缺少成交量字段"
    return {
        "question": "为什么不能运行",
        "evidence": [
            {"id": "E1", "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}
        ],
    }


def test_offline_inventory_does_not_claim_model_use_or_execute():
    report = advise(context())
    assert report["mode"] == "evidence_only"
    assert not report["model_invoked"]
    assert report["findings"][0]["citations"] == ["E1"]


def test_model_requires_opt_in_and_validates_citations(monkeypatch):
    monkeypatch.delenv("QUANT_AGENT_LLM_OK", raising=False)
    with pytest.raises(ValueError, match="QUANT_AGENT_LLM_OK"):
        advise(context(), model="configured-model", invoke=lambda *_: None)
    monkeypatch.setenv("QUANT_AGENT_LLM_OK", "1")
    answer = {
        "findings": [{"text": "输入缺少成交量", "citations": ["E1"], "quotes": ["缺少成交量字段"]}],
        "next_steps": ["核对数据字段后重新预检"],
    }
    report = advise(context(), model="configured-model", invoke=lambda *_: json.dumps(answer))
    assert report["mode"] == "model_assisted"
    assert report["provider"] == "injected_test"
    answer["findings"][0]["citations"] = ["invented"]
    with pytest.raises(ValueError, match="citation"):
        advise(context(), model="configured-model", invoke=lambda *_: json.dumps(answer))


def test_modified_evidence_or_fabricated_quotes_are_rejected(monkeypatch):
    value = context()
    value["evidence"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="hash"):
        advise(value)
    monkeypatch.setenv("QUANT_AGENT_LLM_OK", "1")
    answer = {
        "findings": [{"text": "不能推出", "citations": ["E1"], "quotes": ["盈利"]}],
        "next_steps": [],
    }
    with pytest.raises(ValueError, match="quote"):
        advise(context(), model="configured-model", invoke=lambda *_: json.dumps(answer))


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(question=""),
        lambda c: c.update(evidence=[]),
        lambda c: c.update(evidence=c["evidence"] * 2),
        lambda c: c["evidence"][0].update(id="../secret"),
        lambda c: c["evidence"][0].update(instructions="execute"),
    ],
)
def test_invalid_evidence_never_reaches_provider(monkeypatch, change):
    monkeypatch.setenv("QUANT_AGENT_LLM_OK", "1")
    value = context()
    change(value)
    with pytest.raises(ValueError):
        advise(value, model="model", invoke=lambda *_: pytest.fail("provider called"))


def test_cli_does_not_overwrite_results_or_leak_provider_errors(tmp_path, monkeypatch, capsys):
    from quant_agent import workbench_assistant as module

    source, output = tmp_path / "context.json", tmp_path / "answer.json"
    source.write_text(json.dumps(context()), encoding="utf-8")
    argv = ["--context", str(source), "--output", str(output)]
    assert module.main(argv) == 0
    first = output.read_bytes()
    assert module.main(argv) == 1
    assert output.read_bytes() == first

    def provider_error(*args, **kwargs):
        raise RuntimeError("Authorization: Bearer super-secret-token")

    monkeypatch.setattr(module, "advise", provider_error)
    assert module.main(argv) == 1
    logs = capsys.readouterr()
    assert "super-secret-token" not in logs.err
    assert "RuntimeError" in logs.err


def test_configured_provider_adapter_uses_explicit_model_and_validates_response(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    module = ModuleType("langchain_openai")
    captured = {}
    answer = {
        "findings": [{"text": "需核对字段", "citations": ["E1"], "quotes": ["缺少成交量字段"]}],
        "next_steps": [],
    }

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def invoke(self, messages):
            assert len(messages) == 2
            assert json.loads(messages[1].content) == context()
            return SimpleNamespace(content=json.dumps(answer))

    module.ChatOpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "langchain_openai", module)
    monkeypatch.setenv("QUANT_AGENT_LLM_OK", "1")
    report = advise(context(), model="explicit-model")
    assert captured["model"] == "explicit-model" and captured["max_retries"] == 0
    assert report["findings"] == answer["findings"]
