"""Local CLI for the independent research workflow extension."""

import argparse
import json
from pathlib import Path

from .contracts import read_json, verify_archive, write_json
from .evidence import search_evidence_graph
from .lifecycle import ResearchLifecycle
from .ml import run_study
from .robustness import load_robustness


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ml = commands.add_parser("ml")
    ml.add_argument("--plan", type=Path, required=True)
    ml.add_argument("--dataset", type=Path, required=True)
    ml.add_argument("--output-root", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("archive", type=Path)
    robust = commands.add_parser("robustness")
    robust.add_argument("bundle", type=Path)
    robust.add_argument("--output", type=Path, required=True)
    life = commands.add_parser("lifecycle")
    life.add_argument("action", choices=["init", "state", "transition"])
    life.add_argument("--database", type=Path, required=True)
    life.add_argument("--strategy", required=True)
    life.add_argument("--event", type=Path)
    graph = commands.add_parser("graph")
    graph.add_argument("query")
    graph.add_argument("--assertions", type=Path, required=True)
    graph.add_argument("--database", type=Path)
    graph.add_argument("--studies-root", type=Path)
    graph.add_argument("--study-path", type=Path, action="append", default=[])
    graph.add_argument("--limit", type=int, default=50)
    graph.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "ml":
            result = run_study(args.plan, args.dataset, args.output_root)
        elif args.command == "verify":
            result = verify_archive(args.archive)
        elif args.command == "robustness":
            result = load_robustness(args.bundle)
            write_json(args.output, result)
        elif args.command == "lifecycle":
            service = ResearchLifecycle(args.database, create=args.action == "init")
            if args.action == "transition":
                if args.event is None:
                    raise ValueError("transition requires --event JSON")
                result = service.transition(args.strategy, **read_json(args.event))
            else:
                result = service.state(args.strategy)
        else:
            result = search_evidence_graph(
                args.query,
                read_json(args.assertions),
                database=args.database,
                studies_root=args.studies_root,
                study_paths=args.study_path,
                limit=args.limit,
            )
            write_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2 if result.get("status") in {"failed", "incomplete_family"} else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "reason": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
