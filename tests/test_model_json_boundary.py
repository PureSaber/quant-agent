"""Public assistant boundaries accept one complete JSON result, never extract prose."""

import hashlib
import json

import pytest

from quant_agent.notebook_advisor import propose_notebook
from quant_agent.workbench_assistant import advise


def request():
    text = "Synthetic fixture is missing volume."
    return {
        "question": "What input is missing?",
        "evidence": [
            {"id": "E1", "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}
        ],
    }


def answer(notebook):
    value = {
        "findings": [{"text": "Check volume.", "citations": ["E1"], "quotes": ["missing volume"]}],
        "next_steps": ["Inspect the input schema."],
    }
    if notebook:
        value["patches"] = []
    return value


def call(notebook, **kwargs):
    if notebook:
        return propose_notebook(request(), {"nbformat": 4, "cells": []}, **kwargs)
    return advise(request(), **kwargs)


@pytest.mark.parametrize("notebook", [False, True], ids=["workbench", "notebook"])
@pytest.mark.parametrize("wrapper", ["{}", " \n```json\n{}\n```\t", "```\r\n{}\r\n```"])
def test_accepts_only_complete_json_result(monkeypatch, notebook, wrapper):
    monkeypatch.setenv("QUANT_AGENT_LLM_OK", "1")
    expected = answer(notebook)
    raw = wrapper.format(json.dumps(expected))
    report = call(notebook, model="test", invoke=lambda *_: raw)
    assert report["findings"] == expected["findings"]
    assert report["next_steps"] == expected["next_steps"]
    assert report["model_invoked"] is False and report["provider"] == "injected_test"
    if notebook:
        assert report["patches"] == []


@pytest.mark.parametrize("notebook", [False, True], ids=["workbench", "notebook"])
@pytest.mark.parametrize(
    "kind",
    [
        "prose",
        "suffix",
        "two_fences",
        "two_objects",
        "malformed",
        "unclosed",
        "wrong_language",
        "oversized",
        "multibyte_oversized",
        "non_object",
        "non_text",
        "unknown_citation",
        "invented_quote",
    ],
)
def test_invalid_result_fails_without_retry(monkeypatch, notebook, kind):
    monkeypatch.setenv("QUANT_AGENT_LLM_OK", "1")
    value = answer(notebook)
    if kind == "unknown_citation":
        value["findings"][0]["citations"] = ["E2"]
    elif kind == "invented_quote":
        value["findings"][0]["quotes"] = ["profitable"]
    raw = json.dumps(value)
    fenced = "```json\n" + raw + "\n```"
    invalid = {
        "prose": "Here is the answer:\n" + fenced,
        "suffix": fenced + "\nPlease run this.",
        "two_fences": fenced + "\n" + fenced,
        "two_objects": raw + raw,
        "malformed": '```json\n{"findings":\n```',
        "unclosed": "```json\n" + raw,
        "wrong_language": "```python\n" + raw + "\n```",
        # A valid body does not permit bypassing the original raw-byte limit with whitespace.
        "oversized": " " * (120_000 if notebook else 60_000) + fenced,
        "multibyte_oversized": "中" * (40_000 if notebook else 20_000) + fenced,
        "non_object": "```json\n[]\n```",
        "non_text": {"text": fenced},
        "unknown_citation": fenced,
        "invented_quote": fenced,
    }[kind]
    calls = []

    def invoke(*_):
        calls.append(True)
        return invalid

    with pytest.raises(TypeError if kind == "non_object" else ValueError):
        call(notebook, model="test", invoke=invoke)
    assert calls == [True]


@pytest.mark.parametrize("notebook", [False, True], ids=["workbench", "notebook"])
def test_real_sdk_requests_json_mode_and_keeps_local_validation(monkeypatch, notebook):
    import httpx
    import langchain_openai

    monkeypatch.setenv("QUANT_AGENT_LLM_OK", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://synthetic-provider.invalid/v1")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    expected = answer(notebook)
    captured, requests = {}, []
    actual_client = langchain_openai.ChatOpenAI

    def transport(request):
        assert request.url.path == "/v1/chat/completions"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "synthetic-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "synthetic-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "```json\n" + json.dumps(expected) + "\n```",
                        },
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(transport), trust_env=False) as client:

        def factory(**kwargs):
            captured.update(kwargs)
            return actual_client(**kwargs, http_client=client)

        monkeypatch.setattr(langchain_openai, "ChatOpenAI", factory)
        report = call(notebook, model="synthetic-model")
    assert captured == {
        "model": "synthetic-model",
        "temperature": 0,
        "timeout": 60,
        "max_retries": 0,
    }
    assert len(requests) == 1
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert "tools" not in requests[0]
    assert report["findings"] == expected["findings"]
    assert report["model_invoked"] is True and report["provider"] == "configured_provider"
