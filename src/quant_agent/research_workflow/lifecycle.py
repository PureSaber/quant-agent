"""Independent research state journal. It cannot grant or change trading permissions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from quant_agent.research_history import verify_citations

from .contracts import canonical, digest, identifier, timestamp

TRANSITIONS = {
    None: {"candidate"},
    "candidate": {"review", "paused", "retired"},
    "review": {"candidate", "observe", "paused", "retired"},
    "observe": {"review", "paused", "retired"},
    "paused": {"review", "retired"},
    "retired": set(),
}


class ResearchLifecycle:
    def __init__(self, database, *, create=False):
        self.path = Path(database).resolve()
        if not self.path.exists():
            if not create:
                raise FileNotFoundError(self.path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Claim only a new independent DB, never initialize a Lab/account DB.
            with self.path.open("xb"):
                pass
            with sqlite3.connect(self.path) as db:
                db.execute("CREATE TABLE workflow_meta (schema TEXT NOT NULL)")
                db.execute("INSERT INTO workflow_meta VALUES ('quant.research-lifecycle/v1')")
                db.execute(
                    "CREATE TABLE workflow_events (strategy TEXT NOT NULL, sequence INTEGER NOT NULL, payload TEXT NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY(strategy,sequence))"
                )
        with self._read() as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables != {"workflow_meta", "workflow_events"} or db.execute(
                "SELECT schema FROM workflow_meta"
            ).fetchall() != [("quant.research-lifecycle/v1",)]:
                raise ValueError("not an independent research lifecycle database")

    def _read(self):
        return sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)

    def _state(self, db, strategy):
        identifier(strategy)
        rows = db.execute(
            "SELECT sequence,payload,sha256 FROM workflow_events WHERE strategy=? ORDER BY sequence",
            (strategy,),
        ).fetchall()
        previous, state, events = None, None, []
        last_time = None
        for expected, (sequence, payload, sha) in enumerate(rows, 1):
            event = json.loads(payload)
            if (
                sequence != expected
                or digest(event) != sha
                or event["previous_sha256"] != previous
                or event["from_state"] != state
                or event["to_state"] not in TRANSITIONS[state]
                or event["execution_authorized"] is not False
                or event["strategy_id"] != strategy
            ):
                raise ValueError("corrupt research event chain")
            when = timestamp(event["recorded_at"])
            if last_time is not None and when <= last_time:
                raise ValueError("event timestamps must increase")
            last_time, previous, state = when, sha, event["to_state"]
            events.append({**event, "sha256": sha})
        citations = [c for event in events for c in event["evidence"]]
        return {
            "schema": "quant.research-lifecycle/v1",
            "strategy_id": strategy,
            "state": state,
            "head_sha256": previous,
            "events": events,
            "evidence_verification": verify_citations(citations),
            "execution_authorized": False,
            "authority": "research workflow only; actor names are asserted, not authenticated identities",
        }

    def state(self, strategy):
        with self._read() as db:
            return self._state(db, strategy)

    def transition(
        self, strategy, to_state, *, actor, reason, evidence, recorded_at, expected_head
    ):
        if (
            not isinstance(actor, str)
            or not actor.strip()
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError("actor and explicit reason required")
        when = timestamp(recorded_at)
        if not isinstance(evidence, list) or not evidence or verify_citations(evidence)["invalid"]:
            raise ValueError("at least one currently verified evidence citation required")
        with sqlite3.connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            state = self._state(db, strategy)
            if state["head_sha256"] != expected_head:
                raise ValueError("stale expected event head; reread before transitioning")
            if to_state not in TRANSITIONS[state["state"]]:
                raise ValueError("invalid research state transition")
            if state["events"] and when <= timestamp(state["events"][-1]["recorded_at"]):
                raise ValueError("event timestamps must increase")
            if to_state == "observe" and state["evidence_verification"]["invalid"]:
                raise ValueError("changed historical evidence prevents observation promotion")
            event = {
                "strategy_id": strategy,
                "from_state": state["state"],
                "to_state": to_state,
                "actor": actor.strip(),
                "reason": reason.strip(),
                "evidence": evidence,
                "recorded_at": when.isoformat(),
                "previous_sha256": expected_head,
                "execution_authorized": False,
            }
            sha = digest(event)
            db.execute(
                "INSERT INTO workflow_events VALUES (?,?,?,?)",
                (strategy, len(state["events"]) + 1, canonical(event), sha),
            )
        return self.state(strategy)
