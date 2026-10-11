"""Reuse Lab readers with OS/SQLite-enforced read-only connections, no constructor writes."""

import hashlib
import sqlite3
from pathlib import Path

from quant_lab.family_evidence import collect_family
from quant_lab.trials import TrialRegistry, canonical


class ReadOnlyTrialRegistry(TrialRegistry):
    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def connect(self):
        db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=30)
        db.execute("PRAGMA query_only=ON")
        return db

    @staticmethod
    def _verify(result):
        actual = hashlib.sha256(canonical(result["definition"]).encode()).hexdigest()
        if actual != result["sha256"]:
            raise ValueError("registered definition digest mismatch")
        return result

    def family(self, family_id):
        return self._verify(super().family(family_id))

    def definition(self, study_id):
        return self._verify(super().definition(study_id))


def registered_family(database, family_id):
    """Return Lab's complete inventory/matrix; retries and absent attempts remain visible."""
    registry = ReadOnlyTrialRegistry(database)
    # Base Lab treats ValueError from a missing definition as not_registered.
    # Verify existing rows first so corruption cannot be mislabeled as absence.
    family = registry.family(family_id)
    with registry.connect() as db:
        existing = {row[0] for row in db.execute("SELECT study_id FROM studies")}
    for study in family["definition"]["study_ids"]:
        if study in existing:
            registry.definition(study)
    values, audit = collect_family(registry, family_id)
    # Detect a concurrent registry change instead of silently combining two inventories.
    if audit["inventory"] != registry.family_inventory(family_id):
        raise ValueError("registry changed during collection; retry the read-only snapshot")
    return values, audit
