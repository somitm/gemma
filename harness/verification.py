"""Verification helpers — introduced here, wired into the loop later (ch-12).

ch-08 is the execution environment: the place where untrusted code runs behind a
boundary. ``run_python`` is the same start-closed posture as the bash sandbox —
candidate code runs in a fresh process with a *scrubbed* environment and a scoped
temp workdir, so we never hand model-written code our credentials.

The module lands now because the sandbox exercise needs it, but the agent loop
does not call it yet. Turning these into a self-checking feedback loop (run the
model's code against an assertion, feed failures back, correct) is the
verification primitive that lands at ch-12.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


@dataclass
class VerificationResult:
    passed: bool
    output: str


def extract_code(text: str) -> str:
    """Pull a python code block from model output, or return the text as-is."""
    match = _FENCE.search(text)
    return (match.group(1) if match else text).strip()


def run_python(code: str, check: str, timeout: float = 10.0) -> VerificationResult:
    """Run candidate ``code`` then an assertion ``check`` in a fresh process.

    Model-written code runs with a *scrubbed* environment and a scoped temp
    workdir — the same start-closed posture as the bash sandbox, so we don't hand
    untrusted code our credentials (a real Docker sandbox would also cut network).
    """
    script = f"{code}\n\n{check}\nprint('VERIFICATION_OK')\n"
    workdir = Path(tempfile.mkdtemp(prefix="verify-"))
    candidate = workdir / "candidate.py"
    candidate.write_text(script)
    scrubbed_env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(workdir), "LC_ALL": "C"}
    try:
        proc = subprocess.run(
            [sys.executable, str(candidate)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workdir),
            env=scrubbed_env,
        )
    except subprocess.TimeoutExpired:
        return VerificationResult(False, "error: timed out")
    output = (proc.stdout + proc.stderr).strip()
    passed = proc.returncode == 0 and "VERIFICATION_OK" in proc.stdout
    return VerificationResult(passed=passed, output=output)
