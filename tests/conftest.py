"""Test isolation (a ch-03 consequence).

From ch-03 the agent auto-loads ``AGENTS.md`` from its working directory. The suite
runs from the repo root, which *has* an ``AGENTS.md``, so a bare ``Agent()`` in an
earlier chapter's test would silently pick up the real project instructions and skew
its assertions. This is test infrastructure, not an agent primitive.

The fix is surgical on purpose: ignore only the *ambient* AGENTS.md (the default
``agents_dir="."``). We don't chdir or touch the real working directory, because
other chapters' tests legitimately rely on it (ch-08 reads ``pyproject.toml`` from
the repo root to prove read_file is workspace-scoped). Tests that exercise AGENTS.md
pass an explicit ``agents_dir`` and flow through to the real loader unchanged.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_ambient_agents_md(monkeypatch):
    import harness.agent as agent_mod

    real = agent_mod.load_agents_md

    def guarded(directory: str = ".", *args, **kwargs) -> str:
        if str(directory) in (".", ""):  # the ambient default — don't read the repo's file
            return ""
        return real(directory, *args, **kwargs)

    monkeypatch.setattr(agent_mod, "load_agents_md", guarded)
