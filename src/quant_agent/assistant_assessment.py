"""Label-separated, offline financial-research response assessment.

This evaluates a closed structured response contract, not arbitrary prose. Gold
labels never enter the exported candidate packet or the deterministic reasoner.
Self-test controls validate the evaluator; they do not measure an online LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, InvalidOperation
from html import escape
from pathlib import Path


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def identity(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Non-finite JSON number: " + value)

    return json.loads(
        Path(path).read_text("utf-8"), object_pairs_hook=pairs, parse_constant=invalid
    )


def _unique(items, key):
    if not isinstance(items, list):
        raise TypeError("Expected a list")
    result = {}
    for item in items:
        name = item[key]
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("Missing/duplicate identifier: " + str(name))
        result[name] = item
    return result


def public_packet(bundle):
    """Explicit allowlist rather than recursively removing known answer keys."""
    if bundle.get("schema") != "quant.assistant-cases/v1":
        raise ValueError("Unsupported case schema")
    cases = _unique(bundle["cases"], "case_id")
    exported = []
    for case in cases.values():
        sources = _unique(case["evidence"], "source_id")
        exported.append(
            {
                "case_id": case["case_id"],
                "question": case["question"],
                "sample_kind": case["sample_kind"],
                "evidence": [
                    {key: source[key] for key in ("source_id", "type", "content")}
                    for source in sources.values()
                ],
            }
        )
    return {"schema": bundle["schema"], "bundle_id": bundle["bundle_id"], "cases": exported}


def _resolve(case, reference):
    source_id, pointer = reference.split("#", 1)
    source = _unique(case["evidence"], "source_id")[source_id]["content"]
    if not pointer:
        return source
    if not pointer.startswith("/"):
        raise ValueError("Evidence reference requires a JSON pointer")
    for part in pointer[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(source, list):
            if not part.isdecimal():
                raise ValueError("Array pointer requires a nonnegative index")
            source = source[int(part)]
        else:
            source = source[part]
    return source


def _same(actual, expected):
    if "tolerance" not in expected:
        return type(actual) is type(expected["value"]) and actual == expected["value"]
    if type(actual) not in {str, int, float}:
        return False
    try:
        number, target, tolerance = (
            Decimal(str(actual)),
            Decimal(str(expected["value"])),
            Decimal(expected["tolerance"]),
        )
        return (
            number.is_finite()
            and target.is_finite()
            and tolerance.is_finite()
            and tolerance >= 0
            and abs(number - target) <= tolerance
        )
    except InvalidOperation:
        return False


def _string_set(value):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("Expected a string list")
    if len(value) != len(set(value)):
        raise ValueError("Duplicate list values")
    return set(value)


def score(bundle, labels, submission):
    bundle = public_packet(bundle)
    bundle_hash = identity(bundle)
    if labels.get("schema") != "quant.assistant-labels/v1":
        raise ValueError("Unsupported evaluator label schema")
    if labels.get("public_sha256") != bundle_hash or labels.get("bundle_id") != bundle["bundle_id"]:
        raise ValueError("Public evidence/label identity mismatch")
    cases, gold = _unique(bundle["cases"], "case_id"), _unique(labels["cases"], "case_id")
    if cases.keys() != gold.keys():
        raise ValueError("Gold labels must cover each case exactly")
    if set(submission) != {"schema", "public_sha256", "candidate", "responses"}:
        raise ValueError("Submission has missing or unknown fields")
    if submission["schema"] != "quant.assistant-submission/v1":
        raise ValueError("Unsupported submission schema")
    if submission["public_sha256"] != bundle_hash:
        raise ValueError("Submission was produced against different evidence")
    candidate = submission["candidate"]
    if (
        not isinstance(candidate, dict)
        or set(candidate) != {"id", "kind", "code_sha256"}
        or not isinstance(candidate["id"], str)
        or not candidate["id"]
        or candidate["kind"]
        not in {"offline_deterministic", "external_submission", "regression_control"}
    ):
        raise ValueError("Invalid candidate declaration")
    responses = _unique(submission["responses"], "case_id")
    global_errors = ["Unexpected response case: " + key for key in responses.keys() - cases.keys()]
    rows = []
    for case_id, case in cases.items():
        expected, response = gold[case_id], responses.get(case_id)
        errors, checks = [], []
        if response is None:
            errors.append("missing_response")
            response = {
                "case_id": case_id,
                "decision": None,
                "claims": [],
                "acknowledged_failures": [],
                "caveats": [],
            }
        if set(response) != {"case_id", "decision", "claims", "acknowledged_failures", "caveats"}:
            errors.append("response_schema: unknown or missing fields")
        if response.get("decision") != expected["decision"]:
            errors.append("wrong_decision")
        try:
            claims = _unique(response.get("claims"), "key")
            failures = _string_set(response.get("acknowledged_failures"))
            caveats = _string_set(response.get("caveats"))
        except (ValueError, KeyError, TypeError) as exc:
            errors.append("response_schema: " + str(exc))
            claims, failures, caveats = {}, set(), set()
        expected_claims = _unique(expected["claims"], "key")
        for key in claims.keys() - expected_claims.keys():
            errors.append("unsupported_claim:" + key)
        for key, annotation in expected_claims.items():
            claim = claims.get(key)
            correct, grounded = False, False
            if claim is None:
                errors.append("missing_claim:" + key)
            else:
                if set(claim) != {"key", "value", "evidence"}:
                    errors.append("claim_schema:" + key)
                correct = _same(claim.get("value"), annotation)
                if not correct:
                    errors.append("wrong_claim:" + key)
                try:
                    refs = _string_set(claim.get("evidence"))
                    for ref in refs:
                        _resolve(case, ref)
                    grounded = refs == set(annotation["evidence"])
                except (ValueError, KeyError, IndexError, TypeError):
                    grounded = False
                if not grounded:
                    errors.append("unsupported_evidence:" + key)
            checks.append(
                {
                    "key": key,
                    "correct": correct,
                    "grounded": grounded,
                    "expected": annotation["value"],
                    "actual": claim.get("value") if claim else None,
                }
            )
        missing_failures = set(expected["acknowledged_failures"]) - failures
        invented_failures = failures - set(expected["acknowledged_failures"])
        if missing_failures:
            errors.append("omitted_failures:" + ",".join(sorted(missing_failures)))
        if invented_failures:
            errors.append("invented_failures:" + ",".join(sorted(invented_failures)))
        if caveats != set(expected["caveats"]):
            errors.append("wrong_or_missing_caveats")
        rows.append(
            {
                "case_id": case_id,
                "question": case["question"],
                "passed": not errors,
                "errors": errors,
                "claim_checks": checks,
                "decision": response.get("decision"),
                "expected_decision": expected["decision"],
                "rationale": expected["rationale"],
                "dimensions": expected["dimensions"],
                "expected_failure_count": len(expected["acknowledged_failures"]),
                "acknowledged_failure_count": len(
                    failures & set(expected["acknowledged_failures"])
                ),
            }
        )
    claim_checks = [check for row in rows for check in row["claim_checks"]]
    dimensions = sorted({dimension for row in rows for dimension in row["dimensions"]})
    return {
        "schema": "quant.assistant-assessment/v1",
        "evidence_kind": "offline_response_assessment",
        "bundle_id": bundle["bundle_id"],
        "public_sha256": bundle_hash,
        "labels_sha256": identity(labels),
        "submission_sha256": identity(submission),
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate": candidate,
        "passed": not global_errors and all(row["passed"] for row in rows),
        "global_errors": global_errors,
        "summary": {
            "cases": len(rows),
            "cases_passed": sum(row["passed"] for row in rows),
            "claims_expected": len(claim_checks),
            "claims_correct_and_grounded": sum(
                c["correct"] and c["grounded"] for c in claim_checks
            ),
            "dimensions": {
                d: {
                    "passed": sum(r["passed"] for r in rows if d in r["dimensions"]),
                    "total": sum(d in r["dimensions"] for r in rows),
                }
                for d in dimensions
            },
        },
        "cases": rows,
        "limits": [
            "Closed structured claims; arbitrary prose and open-domain reasoning are not scored.",
            "Fixture labels are separately specified regression annotations, not an external human audit.",
            "Answers are withheld from exported candidate packets; repository labels are public regression data.",
            "This is not an unseen model holdout, trading validation or real-market evidence.",
        ],
        "online_model_called": False,
        "online_model_quality": "not_measured",
    }


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    content = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>",
        "<title>Financial research assistant assessment</title>",
        (
            "<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}"
            "td,th{border:1px solid #ccc;padding:10px;text-align:left;vertical-align:top}"
            "table{border-collapse:collapse;width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
            ".pass{color:#17683e}.fail{color:#ac2020}</style>"
        ),
        "<h1>Financial research assistant assessment</h1>",
        "<p>Offline evidence assessment. <strong>Online model quality: not measured.</strong></p>",
        "<pre>"
        + escape(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "candidate",
                        "summary",
                        "public_sha256",
                        "labels_sha256",
                        "submission_sha256",
                        "scorer_sha256",
                    )
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        + "</pre>",
        "<table><tr><th>Case</th><th>Result</th><th>Independent annotation / observed errors</th></tr>",
    ]
    for row in report["cases"]:
        content.append(
            "<tr><td>"
            + escape(row["case_id"] + " — " + row["question"])
            + "</td>"
            + ("<td class='pass'>PASS" if row["passed"] else "<td class='fail'>FAIL")
            + "</td><td>"
            + escape(row["rationale"])
            + "<pre>"
            + escape(
                json.dumps(
                    {"errors": row["errors"], "claims": row["claim_checks"]},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            + "</pre></td></tr>"
        )
    content.append("</table><h2>Scope and limitations</h2><ul>")
    content.extend("<li>" + escape(limit) + "</li>" for limit in report["limits"])
    content.append("</ul></html>")
    (output / "report.html").write_text("".join(content), "utf-8")


def self_test(suite, output):
    from quant_agent.assessment_baseline import respond

    suite, output = Path(suite), Path(output)
    public = public_packet(load(suite / "public/cases.json"))
    # Candidate computation happens using only the public packet, before label loading.
    baseline = respond(public)
    labels = load(suite / "evaluator/labels.json")
    inventory = load(suite / "evaluator/controls.json")
    output.mkdir(parents=True, exist_ok=False)
    (output / "candidate-packet.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2), "utf-8"
    )
    (output / "baseline-submission.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), "utf-8"
    )
    baseline_report = score(public, labels, baseline)
    write_report(baseline_report, output / "baseline")
    controls = []
    for control in inventory["controls"]:
        submission = load(suite / "controls" / control["file"])
        report = score(public, labels, submission)
        write_report(report, output / control["id"])
        controls.append(
            {
                "id": control["id"],
                "expected_accept": control["expected_accept"],
                "actual_accept": report["passed"],
                "evaluator_correct": report["passed"] is control["expected_accept"],
            }
        )
    result = {
        "schema": "quant.assistant-evaluator-regression/v1",
        "evidence_kind": "evaluator_regression_controls",
        "controls": controls,
        "passed": all(item["evaluator_correct"] for item in controls),
        "offline_baseline": baseline_report["summary"],
        "baseline_passed": baseline_report["passed"],
        "online_model_quality": "not_measured",
        "online_model_called": False,
    }
    (output / "self-test.json").write_text(json.dumps(result, indent=2), "utf-8")
    links = "".join(
        "<li><a href='"
        + escape(control["id"])
        + "/report.html'>"
        + escape(control["id"])
        + "</a>: "
        + str(control["evaluator_correct"])
        + " (evaluator classified as expected)</li>"
        for control in controls
    )
    (output / "index.html").write_text(
        "<!doctype html><html><meta charset='utf-8'><title>Assessment regression</title>"
        "<h1>Offline assessment regression</h1><p>Online model quality: not measured.</p>"
        "<p><a href='baseline/report.html'>Label-blind deterministic baseline report</a></p>"
        "<ul>" + links + "</ul><pre>" + escape(json.dumps(result, indent=2)) + "</pre></html>",
        "utf-8",
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "baseline"):
        command = commands.add_parser(name)
        command.add_argument("--cases", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("score")
    for name in ("cases", "labels", "submission", "output"):
        command.add_argument("--" + name, type=Path, required=True)
    command = commands.add_parser("self-test")
    command.add_argument("--suite", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "self-test":
            result = self_test(args.suite, args.output)
            print(json.dumps(result, indent=2))
            return 0 if result["passed"] and result["baseline_passed"] else 1
        public = public_packet(load(args.cases))
        if args.command in {"export", "baseline"}:
            result = public
            if args.command == "baseline":
                from quant_agent.assessment_baseline import respond

                result = respond(public)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
            return 0
        report = score(public, load(args.labels), load(args.submission))
        write_report(report, args.output)
        print(json.dumps(report["summary"], indent=2))
        return 0 if report["passed"] else 1
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Assessment refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
