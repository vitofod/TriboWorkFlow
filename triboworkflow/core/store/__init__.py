# Author: Vito Foderà
# Created: 2026-10-22

"""Per-calculation records, aggregated JSON and atomic writes.

The store keeps the data of one system in one directory, and never reads the
directories of other systems::

    <system>/
    ├── system.json           aggregated document of the system
    └── <step>/<label>/
        └── record.json       record of one calculation

- ``record.py``: the record of a calculation, its separate states, the reuse
  rule and the check on energy models;
- ``fingerprint.py``: digests of inputs, data and files;
- ``system.py``: the aggregated document, the provenance of its entries and
  its single writer;
- ``atomic.py``: atomic JSON and text writes;
- ``legacy.py``: the historic database calls on top of the store, for the
  ported workflow code only; it is not imported here.

Preparation, execution and analysis use the store separately: each phase
reads the files left by the previous one, and no orchestrator is required.
"""

from .atomic import StoreError, check_json_value, read_json, write_json, write_text
from .fingerprint import (bytes_digest, canonical_json, check_fingerprint, content_digest,
                          file_digest, fingerprint)
from .record import (Attempt, AttemptFile, Engine, Record, Status, can_reuse,
                     check_energy_model, identity_differences, iter_records, read_record,
                     record_path, reuse_problem, unknown_parts, write_record)
from .system import (LockError, PrerequisiteError, Provenance, SystemDocument, SystemWriter,
                     check_prerequisites, entry_is_current, is_done, read_system)

__all__ = ['Attempt', 'AttemptFile', 'Engine', 'LockError', 'PrerequisiteError', 'Provenance', 'Record',
           'Status', 'StoreError', 'SystemDocument', 'SystemWriter', 'bytes_digest', 'can_reuse',
           'canonical_json', 'check_energy_model', 'check_fingerprint', 'check_json_value',
           'check_prerequisites', 'content_digest', 'entry_is_current', 'file_digest',
           'fingerprint', 'identity_differences', 'is_done', 'iter_records', 'read_json',
           'read_record', 'read_system',
           'record_path', 'reuse_problem', 'unknown_parts', 'write_json', 'write_record',
           'write_text']
