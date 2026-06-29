"""ch-08 — Execution environment.

Capability: a bash tool runs commands inside a sandbox — scoped workdir and a
scrubbed environment, so host credentials don't leak in. (Forced to the local
backend here for determinism; Docker is exercised live when the daemon is up.)

The hardening that makes the boundary trustworthy rides along: read_file confined
to the workspace, the verifier scrubbed of host env, and compaction keyed off the
model's reported token usage.
"""

import os
from unittest.mock import patch

import harness.agent as agent_mod
from harness import compaction
from harness.sandbox import Sandbox, bash_tool
from harness.tools import read_file
from harness.verification import run_python
from model import LLMResponse


def test_command_runs_and_returns_output():
    sb = Sandbox(prefer_docker=False)
    r = sb.run("echo hello-from-sandbox")
    assert r.backend == "local"
    assert r.exit_code == 0
    assert "hello-from-sandbox" in r.stdout


def test_environment_is_scrubbed():
    os.environ["SANDBOX_SECRET"] = "POULTRY-FARM"
    try:
        r = Sandbox(prefer_docker=False).run("printenv SANDBOX_SECRET || echo CLEAN")
    finally:
        del os.environ["SANDBOX_SECRET"]
    assert "POULTRY-FARM" not in r.stdout
    assert "CLEAN" in r.stdout


def test_runs_in_isolated_workdir():
    r = Sandbox(prefer_docker=False).run("pwd")
    assert "sandbox-" in r.stdout  # the fresh temp workdir, not the repo


def test_bash_tool_wraps_sandbox():
    tool = bash_tool(Sandbox(prefer_docker=False))
    assert tool.name == "bash"
    out = tool.func(command="echo hi")
    assert "hi" in out and "via local" in out


# --- hardening fixes ---------------------------------------------------------
# 1. read_file scoping
def test_read_file_blocks_outside_workspace():
    assert read_file("/etc/passwd").startswith("error: path outside")


def test_read_file_allows_inside_workspace():
    out = read_file("pyproject.toml")  # cwd is the repo root under pytest
    assert "[project]" in out


# 2. verifier containment
def test_verifier_does_not_inherit_host_env():
    os.environ["LEAKY_SECRET"] = "SHOULD-NOT-LEAK"
    try:
        res = run_python("import os", "assert os.getenv('LEAKY_SECRET') is None")
    finally:
        del os.environ["LEAKY_SECRET"]
    assert res.passed


# 3. compaction on reported usage
def test_agent_tracks_reported_usage():
    with patch.object(
        agent_mod, "chat", return_value=LLMResponse(content="ok", usage={"total_tokens": 1234})
    ):
        a = agent_mod.Agent()
        a.send("hi")
    assert a._last_tokens == 1234


def test_compaction_triggers_on_reported_usage():
    # The char estimate stays tiny, but reported usage is huge -> compaction fires.
    def fake_chat(messages, **kwargs):
        first = messages[0].get("content", "") if messages else ""
        if first.startswith("You are a context summarizer"):
            return LLMResponse(content="SUMMARY")
        return LLMResponse(content="ok", usage={"total_tokens": 99999})

    with (
        patch.object(agent_mod, "chat", side_effect=fake_chat),
        patch.object(compaction, "chat", side_effect=fake_chat),
    ):
        a = agent_mod.Agent(context_limit=500)  # estimate of short msgs stays well under this
        for _ in range(8):
            a.send("x")

    assert any(str(m.get("content", "")).startswith("[summary") for m in a.messages)
