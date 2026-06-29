"""Execution environment — the harness runs code, the model never does.

When a tool shells out, the command runs inside this sandbox, not on the host
shell. The model only ever sees the captured stdout/stderr and exit code.

This is the *minimal* form: it prefers Docker (an isolated container) and falls
back to a local subprocess when no Docker daemon is around. Give it a ``workdir``
and the command runs in that persistent directory — so a bash command can see a
file a write tool just created (the workspace seam) instead of a throwaway dir.

The real boundary — no network, a non-root user, a scrubbed environment with no
inherited credentials — is the hardening that lands at ch-08. Here the point is
just the seam: code execution goes through one chokepoint the harness controls.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass

from harness.tools import Tool


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    backend: str


class Sandbox:
    def __init__(
        self,
        image: str = "busybox",
        timeout: float = 15.0,
        prefer_docker: bool = True,
    ) -> None:
        self.image = image
        self.timeout = timeout
        self.prefer_docker = prefer_docker
        self._docker: bool | None = None

    def _docker_up(self) -> bool:
        if not self.prefer_docker:
            return False
        if self._docker is None:
            try:
                self._docker = (
                    subprocess.run(
                        ["docker", "info"],
                        capture_output=True,
                        timeout=5,
                    ).returncode
                    == 0
                )
            except (OSError, subprocess.SubprocessError):
                self._docker = False
        return self._docker

    def run(self, command: str, workdir: str | None = None) -> SandboxResult:
        # A workdir makes the sandbox operate on a persistent workspace (bind-mounted
        # in docker, cwd locally) instead of a throwaway dir.
        if self._docker_up():
            return self._run_docker(command, workdir)
        return self._run_local(command, workdir)

    def _run_docker(self, command: str, workdir: str | None) -> SandboxResult:
        work = ["-v", f"{workdir}:/work"] if workdir else ["--tmpfs", "/work:rw,size=16m"]
        argv = [
            "docker", "run", "--rm",
            *work,
            "-w", "/work",
            self.image,
            "sh", "-c", command,
        ]  # fmt: skip
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        return SandboxResult(proc.stdout, proc.stderr, proc.returncode, "docker")

    def _run_local(self, command: str, workdir: str | None) -> SandboxResult:
        cwd = workdir or tempfile.mkdtemp(prefix="sandbox-")
        proc = subprocess.run(
            ["bash", "-c", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        return SandboxResult(proc.stdout, proc.stderr, proc.returncode, "local")


def bash_tool(sandbox: Sandbox, workdir: str | None = None) -> Tool:
    """A bash tool whose commands run inside the sandbox. With a workdir, commands
    run in the persistent workspace (so they see files the edit tools wrote)."""

    def run_bash(command: str) -> str:
        r = sandbox.run(command, workdir=workdir)
        body = (r.stdout + r.stderr).strip()
        return f"[exit {r.exit_code} via {r.backend}]\n{body}"

    return Tool(
        name="bash",
        description="Run a shell command in an isolated sandbox and return its output.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        func=run_bash,
    )
