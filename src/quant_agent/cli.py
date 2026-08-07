"""CLI entry: quant-review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quant_agent.adapters.base import detect_project
from quant_agent.graph import run_review


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-review", description="Review quant run outputs")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Review a completed run directory")
    run_p.add_argument("--project", help="Project id (e.g. a-share-multifactor)")
    run_p.add_argument("--run-dir", required=True, type=Path, help="Path to run output directory")
    run_p.add_argument("--config", type=Path, help="Original run config yaml")
    run_p.add_argument(
        "--agent-config",
        type=Path,
        help="agent_review.yaml path (default: package configs/agent_review.yaml)",
    )
    run_p.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode (skip LLM). Default unless --llm is passed.",
    )
    run_p.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM nodes (requires OPENAI_API_KEY and quant-agent[llm])",
    )
    run_p.add_argument("--json", action="store_true", help="Print manifest JSON to stdout")

    detect_p = sub.add_parser("detect", help="Detect project from run directory")
    detect_p.add_argument("run_dir", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "detect":
        project = detect_project(args.run_dir)
        if project:
            print(project)
            return 0
        print("unknown", file=sys.stderr)
        return 1

    if args.command == "run":
        project = args.project or detect_project(args.run_dir)
        if not project:
            print(
                "Could not detect project; pass --project a-share-multifactor",
                file=sys.stderr,
            )
            return 1

        offline = not args.llm if args.llm else True
        result = run_review(
            project=project,
            run_dir=args.run_dir,
            config_path=args.config,
            agent_config_path=args.agent_config,
            offline=offline,
        )

        if args.json:
            manifest = {
                "report_path": result.get("report_path"),
                "manifest_path": result.get("manifest_path"),
                "rule_passed": result.get("rule_passed"),
                "finding_count": len(result.get("rule_findings") or []),
            }
            print(json.dumps(manifest, indent=2))
        else:
            print(f"Review written: {result.get('report_path')}")
            print(f"Manifest: {result.get('manifest_path')}")
            findings = result.get("rule_findings") or []
            print(f"Findings: {len(findings)} (passed={result.get('rule_passed')})")

        return 0 if result.get("rule_passed", True) else 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
