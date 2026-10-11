import copy
import json
from decimal import Decimal
from pathlib import Path

import pytest

from quant_agent.assessment_baseline import respond
from quant_agent.assistant_assessment import (
    identity,
    load,
    main,
    public_packet,
    score,
    self_test,
)

SUITE = Path(__file__).resolve().parents[1] / "benchmarks/assistant-v1"


def data():
    return (
        load(SUITE / "public/cases.json"),
        load(SUITE / "evaluator/labels.json"),
        load(SUITE / "controls/good.json"),
    )


def test_offline_baseline_and_independent_mutation_controls(tmp_path):
    report = self_test(SUITE, tmp_path / "report")
    assert report["passed"] and report["baseline_passed"]
    assert len(report["controls"]) == 11
    assert report["offline_baseline"]["cases_passed"] == 13
    assert report["online_model_quality"] == "not_measured"
    assert report["online_model_called"] is False
    assert (tmp_path / "report/baseline/report.html").exists()
    assert "Online model quality: not measured" in (tmp_path / "report/index.html").read_text()


def test_literal_financial_annotations_checked_with_independent_decimal_arithmetic():
    _, labels, _ = data()
    values = {
        row["case_id"]: {claim["key"]: claim["value"] for claim in row["claims"]}
        for row in labels["cases"]
    }
    assert Decimal(values["net-return"]["net_pnl"]) == Decimal(2400) - Decimal(600)
    assert Decimal(values["net-return"]["net_return"]) == Decimal(1800) / Decimal(100000)
    assert Decimal(values["cost-reverses-profit"]["net_return"]) == -Decimal(1000) / Decimal(100000)
    assert Decimal(values["external-flow"]["time_weighted_return"]) == (
        Decimal(110) / Decimal(100) * Decimal(168) / Decimal(160) - 1
    )
    assert Decimal(values["multiple-testing"]["adjusted_threshold"]) == Decimal(".05") / 100


def test_export_does_not_expose_evaluator_fields(tmp_path):
    public, labels, _ = data()
    public["secret_labels"] = labels
    public["cases"][0]["expected"] = "hidden-answer-sentinel"
    public["cases"][0]["evidence"][0]["gold"] = "hidden-answer-sentinel"
    source = tmp_path / "source.json"
    source.write_text(json.dumps(public))
    target = tmp_path / "packet.json"
    assert main(["export", "--cases", str(source), "--output", str(target)]) == 0
    text = target.read_text()
    assert "hidden-answer-sentinel" not in text and "rationale" not in text
    assert "expected" not in text and "secret_labels" not in text
    with pytest.raises(SystemExit):
        main(["export", "--cases", str(source), "--output", str(target)])


def test_baseline_is_not_case_id_lookup_and_responds_to_evidence_changes():
    public, _, _ = data()
    public["cases"] = [public["cases"][0]]
    public["cases"][0]["case_id"] = "unseen-random-id"
    public["cases"][0]["evidence"][0]["content"]["fees"] = "1400"
    result = respond(public)
    claims = result["responses"][0]["claims"]
    assert claims[0]["value"] == "1000"
    assert claims[1]["value"] == "0.01"
    assert result["responses"][0]["case_id"] == "unseen-random-id"


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda r: r[0]["claims"][0].update(value=True), "wrong_claim:net_pnl"),
        (lambda r: r[0]["claims"][0].update(value="NaN"), "wrong_claim:net_pnl"),
        (
            lambda r: r[0]["claims"][0].update(evidence=["ledger#/not-present"]),
            "unsupported_evidence:net_pnl",
        ),
        (lambda r: r[0]["claims"].pop(0), "missing_claim:net_pnl"),
        (lambda r: r[0].update(prose="unsupported conclusion outside schema"), "response_schema"),
        (lambda r: r[0]["acknowledged_failures"].append("invented-attempt"), "invented_failures"),
        (lambda r: r[0]["caveats"].append("synthetic_fixture"), "response_schema"),
    ],
)
def test_semantics_and_closed_contract_fail_even_with_valid_quoted_evidence(mutation, error):
    public, labels, submission = data()
    mutation(submission["responses"])
    result = score(public, labels, submission)
    assert not result["passed"]
    assert any(error in message for row in result["cases"] for message in row["errors"])


def test_missing_and_extra_cases_cannot_improve_pass_rate():
    public, labels, submission = data()
    submission["responses"].pop()
    extra = copy.deepcopy(submission["responses"][0])
    extra["case_id"] = "extra"
    submission["responses"].append(extra)
    result = score(public, labels, submission)
    assert not result["passed"] and result["global_errors"]
    assert result["summary"]["cases"] == 13
    assert result["cases"][-1]["passed"] is False


def test_evidence_or_submission_identity_tampering_refused():
    public, labels, submission = data()
    public["cases"][0]["evidence"][0]["content"]["fees"] = "0"
    with pytest.raises(ValueError, match="identity mismatch"):
        score(public, labels, submission)
    public, labels, submission = data()
    submission["public_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different evidence"):
        score(public, labels, submission)


@pytest.mark.parametrize("text", ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}'])
def test_ambiguous_json_refused(tmp_path, text):
    source = tmp_path / "ambiguous.json"
    source.write_text(text)
    with pytest.raises(ValueError):
        load(source)


def test_score_cli_positive_negative_and_immutable_reports(tmp_path):
    args = [
        "score",
        "--cases",
        str(SUITE / "public/cases.json"),
        "--labels",
        str(SUITE / "evaluator/labels.json"),
        "--submission",
    ]
    assert main([*args, str(SUITE / "controls/good.json"), "--output", str(tmp_path / "good")]) == 0
    assert (
        main(
            [
                *args,
                str(SUITE / "controls/wrong-net-return.json"),
                "--output",
                str(tmp_path / "bad"),
            ]
        )
        == 1
    )
    with pytest.raises(SystemExit):
        main([*args, str(SUITE / "controls/good.json"), "--output", str(tmp_path / "good")])
    public, _, _ = data()
    assert identity(public_packet(public)) == load(tmp_path / "good/report.json")["public_sha256"]
