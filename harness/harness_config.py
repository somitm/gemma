"""The editable surface — every behavioral knob, in one versioned data file.

Until now the harness's behavior was smeared across the code as module-level
constants: the system prompt in ``agent.py``, the clamp size in ``limits.py``,
the compaction prompt in ``compaction.py``, ctor defaults buried in signatures.
Changing behavior meant editing *code*, and knowing which of five files to edit.

This module declares those knobs as a first-class primitive: one JSON file
(``harness_config.json``, next to this module) is the *entire* editable surface.
Want a different prompt, a bigger window budget, another gated tool? Edit the
data file — never the code. That boundary is the point: an editor (human or
model) that may touch only this file can retune the whole harness and nothing
else, and the diff of what changed is the diff of one small file.

Three properties make the surface safe to hand over:

- **Versioned.** The file carries an integer ``version``; bump it on every
  change. Rollback is reverting the file; comparing two behaviors is diffing
  two versions of it. Cheap, because the surface is pure data.
- **Frozen.** ``HarnessConfig`` is a frozen dataclass, so "applying an edit" is
  explicitly *construct a new instance from a new file version and swap it in* —
  never in-place mutation. Nothing can quietly drift a knob at runtime.
- **Validated at the door.** ``load_config`` rejects unknown keys, missing
  fields, and wrong types — loudly, at import. A malformed surface must fail
  the run, not silently fall back to defaults the editor thought it replaced.

Sets travel as JSON arrays and land as ``frozenset``; the ``@path`` attach
pattern travels as a regex *string* and is compiled at its use site
(``context.py``). ``CONFIG`` is loaded once at import and is the single source
of truth — the old module-level names still exist, but only as re-exports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).with_name("harness_config.json")


@dataclass(frozen=True)
class HarnessConfig:
    """The harness's behavioral knobs, loaded from ``harness_config.json``."""

    version: int  # bump on every edit; rollback = revert the file
    system_prompt: str  # the agent's default system prompt
    max_tool_steps: int  # tool-call rounds per turn before the loop gives up
    default_context_limit: int  # ~token budget before compaction fires
    approval_tools: frozenset[str]  # tools the approval gate guards
    code_extensions: frozenset[str]  # a write/edit of one of these arms the test gate
    verify_attempts: int  # re-prompts before the gate marks a turn unverified
    require_run: bool  # enforce an observed passing test run after code changes
    max_item_chars: int  # per-item clamp applied at the door (limits.py)
    compaction_prompt: str  # the summarizer's instructions (compaction.py)
    memory_search_limit: int  # max hits returned by cross-session recall
    attach_pattern: str  # regex (as a string) for @path references; compiled in context.py


# name -> expected JSON type; list-typed fields are validated as arrays of
# strings and converted to frozenset below.
_SCHEMA: dict[str, type] = {
    "version": int,
    "system_prompt": str,
    "max_tool_steps": int,
    "default_context_limit": int,
    "approval_tools": list,
    "code_extensions": list,
    "verify_attempts": int,
    "require_run": bool,
    "max_item_chars": int,
    "compaction_prompt": str,
    "memory_search_limit": int,
    "attach_pattern": str,
}
_SET_FIELDS = {"approval_tools", "code_extensions"}


def _check_type(key: str, value: object, expected: type) -> None:
    """Reject a wrong-typed value loudly. ``bool`` is a subclass of ``int`` in
    Python, so integer knobs must explicitly refuse booleans (and vice versa —
    ``bool`` fields accept only real booleans, which isinstance already ensures)."""
    if expected is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif expected is list:
        ok = isinstance(value, list) and all(isinstance(x, str) for x in value)
    else:
        ok = isinstance(value, expected)
    if not ok:
        raise ValueError(
            f"harness config: field {key!r} must be {expected.__name__}"
            f"{' of str' if expected is list else ''}, got {value!r}"
        )


def load_config(path: str | Path = CONFIG_PATH) -> HarnessConfig:
    """Read and structurally validate the editable surface; fail loudly.

    Unknown keys, missing fields, and wrong types are all errors — the loader
    never silently defaults, because a silent default would mean the file on
    disk and the behavior in memory disagree, which is exactly the drift the
    surface exists to prevent."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"harness config: expected a JSON object, got {type(raw).__name__}")
    unknown = sorted(set(raw) - set(_SCHEMA))
    if unknown:
        raise ValueError(f"harness config: unknown keys {unknown}")
    missing = sorted(set(_SCHEMA) - set(raw))
    if missing:
        raise ValueError(f"harness config: missing fields {missing}")
    kwargs: dict[str, Any] = {}
    for key, expected in _SCHEMA.items():
        _check_type(key, raw[key], expected)
        kwargs[key] = frozenset(raw[key]) if key in _SET_FIELDS else raw[key]
    return HarnessConfig(**kwargs)


CONFIG = load_config()
