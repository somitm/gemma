"""The agent — the harness drive loop. Grows one primitive per chapter.

ch-13 — Observability. The agent has done real work for many chapters; now we can
finally *see* it. A ``Tracer`` (``harness/observability.py``) threads through the
loop and records every step: each model call with its tokens, latency, finish
reason, and cost; each tool call with its arguments, result, and status; each
verification with pass/fail. The turn is wrapped in one parent span so the whole
run reads as a tree (OTel GenAI semantic conventions, in ``harness/events.py``).

The seam is deliberate. The default ``Tracer`` is silent and offline (a
``NullExporter``), so ``verify`` stays deterministic; drop in an OTLP-backed
exporter and the same spans flow to Jaeger/Honeycomb. Cost comes from
``model/pricing.py``. Trace persistence, dormant in ``memory.py`` since durable
state landed, fires now: a resumed session restores its trace too. The hooks are
additive — pass no tracer and the loop runs exactly as before.

Everything from ch-12 is unchanged: the enforced-run verification gate (the model
runs the test with bash; the harness will not accept "done" without an observed
passing run), subagents/fan-out, durable sessions, the hardened sandbox,
usage-based compaction, skills, the approval gate, and ``@path`` injection.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

from harness.compaction import compact, estimate_tokens
from harness.context import deliver
from harness.instructions import load_agents_md, test_command
from harness.limits import clamp
from harness.memory import DEFAULT_DIR, load_session, load_trace, save_session, save_trace
from harness.observability import Tracer
from harness.skills import Skill, skills_prompt
from harness.tools import ToolRegistry
from model import Provider, chat

DEFAULT_SYSTEM = (
    "You are a concise, helpful coding assistant. Use tools when they help. "
    "When you change code, verify it before reporting done: run the project's test "
    "command with the bash tool and show the real result. Never claim it works on "
    "your word alone — if you haven't run it, run it."
)
MAX_TOOL_STEPS = 6
DEFAULT_CONTEXT_LIMIT = 4000  # ~tokens; compact the history above this
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".rb",
    ".c",
    ".cpp",
    ".cc",
    ".h",
    ".hpp",
    ".cs",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
}  # a write/edit of one of these arms the test gate (a code change to verify)


class Agent:
    """A model wrapped in memory, a system prompt, context delivery, and tools."""

    def __init__(
        self,
        model: str | None = None,
        provider: Provider | None = None,
        system: str | None = None,
        agents_dir: str = ".",
        tools: ToolRegistry | None = None,
        approve: Callable[[str, str], bool] | None = None,
        approval_required: set[str] | None = None,
        context_limit: int = DEFAULT_CONTEXT_LIMIT,
        skills: list[Skill] | None = None,
        session: str | None = None,
        sessions_dir: str = DEFAULT_DIR,
        verify_attempts: int = 3,
        require_run: bool = True,
        tracer: Tracer | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.system = system
        self.agents_dir = agents_dir  # where AGENTS.md is auto-loaded from
        self.tools = tools
        self.approve = approve
        self.approval_required = approval_required or set()
        self.context_limit = context_limit
        self.skills = skills or []
        self._last_tokens = 0  # model-reported usage from the last call (ch-08)
        self.session = session
        self.sessions_dir = sessions_dir
        # Resume: load prior conversation from disk if this session exists (ch-09).
        self.messages: list[dict] = load_session(session, sessions_dir) if session else []
        # Set true whenever the last turn triggered compaction — the REPL reads
        # this to surface that the window was managed (a demoable, visible event).
        self.just_compacted = False
        self.verify_attempts = verify_attempts
        # ch-12: when a turn changes code, refuse "done" until a real passing run
        # of the project's declared test command is observed. require_run opts out.
        self.require_run = require_run
        self.tracer = tracer
        # ch-13: restore the persisted trace too, so it isn't empty on resume.
        if self.tracer is not None and session:
            self.tracer.load_events(load_trace(session, sessions_dir))

    def _approved(self, name: str, args: str) -> bool:
        # Fail closed: a tool marked as requiring approval with no approver is denied.
        return self.approve(name, args) if self.approve else False

    def _save(self) -> None:
        # Durable state: persist the full history so a restart can resume it.
        if self.session:
            save_session(self.session, self.messages, self.sessions_dir)
            if self.tracer is not None:
                save_trace(self.session, self.tracer.dump_events(), self.sessions_dir)

    def _maybe_compact(self) -> None:
        # ch-08: prefer the model's reported usage; fall back to an estimate on turn one.
        self.just_compacted = False
        window = self._last_tokens or estimate_tokens(self.messages)
        if window > self.context_limit:
            self.messages = compact(self.messages, model=self.model)
            self._last_tokens = 0  # recomputed from the next response
            self.just_compacted = True

    def _system_text(self) -> str:
        """Instruction layer = system prompt + project AGENTS.md + skills menu."""
        parts = [
            p
            for p in (
                self.system,
                load_agents_md(self.agents_dir),
                skills_prompt(self.skills),
            )
            if p
        ]
        return "\n\n".join(parts)

    def _payload(self) -> list[dict]:
        """System prompt first (if any), then the full conversation history."""
        sys_text = self._system_text()
        head = [{"role": "system", "content": sys_text}] if sys_text else []
        return head + self.messages

    def send(self, user_text: str) -> str:
        """Inject @path files, run the loop, then — if this turn changed code —
        enforce a real passing run of the project's tests before returning."""
        if self.tracer:
            self.tracer.turn_start()  # ch-13: nest this turn's steps under one span
        for block in deliver(user_text):  # @file references → injected context
            self.messages.append({"role": "user", "content": f"Context file:\n{block}"})
        self.messages.append({"role": "user", "content": user_text})
        turn_start = len(self.messages)
        reply = self._run()
        reply = self._enforce_run(reply, turn_start)  # gate "done" on a real test run
        self._save()  # durable state: persist after every turn
        return reply

    def _changed_code(self, turn_start: int) -> bool:
        """Did this turn write or edit a source file? The trigger for the gate — a
        code change to verify, not a prose file like facts.txt (by extension, the
        way a pre-commit hook decides what to run on)."""
        for m in self.messages[turn_start:]:
            if m.get("role") != "assistant" or not m.get("tool_calls"):
                continue
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                if fn.get("name") in ("write_file", "edit_file"):
                    try:
                        path = json.loads(fn.get("arguments", "{}")).get("path", "")
                    except json.JSONDecodeError:
                        path = ""
                    if any(path.endswith(ext) for ext in CODE_EXTENSIONS):
                        return True
        return False

    def _enforce_run(self, reply: str, turn_start: int) -> str:
        """If this turn changed code, refuse "done" until a real passing run of the
        project's declared test command (AGENTS.md ``## Testing``) is observed in the
        transcript. The model runs it with bash; the harness only watches the
        receipts. Capped at verify_attempts so a model that will not run it cannot
        hang the loop. No declared command, or no code change → no gate."""
        if not self.require_run:
            return reply
        command = test_command(self.agents_dir)
        if not command or not self._changed_code(turn_start):
            return reply
        for _ in range(self.verify_attempts):
            passed = self._observed_pass(command, turn_start)
            if self.tracer:
                self.tracer.record_verify(passed, 0.0, f"required: {command}")
            if passed:
                return reply
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        "You changed code but I don't see a passing run of the "
                        f"project's tests. Run `{command}` with the bash tool now — "
                        "it must exit 0 before you report done. Show the real output."
                    ),
                }
            )
            reply = self._run()
        return reply  # attempts exhausted — return the last reply (accept stays red)

    def _observed_pass(self, command: str, turn_start: int) -> bool:
        """True iff this turn's transcript holds a bash call running ``command`` that
        exited 0 — paired by tool_call_id so a failed run is not counted as a pass."""
        ran_ids: set[str] = set()
        for m in self.messages[turn_start:]:
            if m.get("role") != "assistant" or not m.get("tool_calls"):
                continue
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                if fn.get("name") == "bash" and command in fn.get("arguments", ""):
                    ran_ids.add(tc.get("id", ""))
        if not ran_ids:
            return False
        return any(
            m.get("role") == "tool"
            and m.get("tool_call_id") in ran_ids
            and str(m.get("content", "")).startswith("[exit 0")
            for m in self.messages[turn_start:]
        )

    def _run(self) -> str:
        """Drive the model, executing tool calls until it produces a final answer."""
        self._maybe_compact()
        specs = self.tools.specs() if self.tools else None
        for _ in range(MAX_TOOL_STEPS):
            t0 = time.perf_counter()
            payload = self._payload()
            resp = chat(payload, model=self.model, tools=specs, provider=self.provider)
            self._last_tokens = int(resp.usage.get("total_tokens", 0)) or self._last_tokens
            if self.tracer:
                self.tracer.record_llm(
                    resp.usage,
                    time.perf_counter() - t0,
                    finish_reason=resp.finish_reason,
                    request_model=self.model,
                    messages=payload,  # optional content; captured only when enabled
                    output=resp.content,
                )
            if resp.tool_calls and self.tools is not None:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": resp.content or "",
                        "tool_calls": resp.tool_calls,
                    }
                )
                for tc in resp.tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", "")
                    t1 = time.perf_counter()
                    if name in self.approval_required and not self._approved(name, args):
                        result = "[denied by approval gate]"
                        status = "denied"
                    else:
                        result = self.tools.call(name, args)
                        status = "error" if result.startswith("error") else "ok"
                    if self.tracer:
                        self.tracer.record_tool(
                            name, time.perf_counter() - t1, args=args, result=result, status=status
                        )
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc.get("id", ""), "content": clamp(result)}
                    )
                continue
            self.messages.append({"role": "assistant", "content": resp.content})
            return resp.content
        return "error: exceeded tool-step budget"


def main() -> None:
    from harness.memory import search_memory_tool
    from harness.orchestrator import Orchestrator
    from harness.sandbox import Sandbox, bash_tool
    from harness.skills import load_skills
    from harness.subagents import delegate_tool, fan_out_tool
    from harness.tools import default_tools
    from harness.workspace import Workspace, edit_file_tool, git_worktree, write_file_tool

    # Work in a real project: a git worktree of this repo (your checkout stays
    # pristine — edits land in a throwaway worktree), or a scratch dir when we're
    # not in a git repo. This is the coding-agent posture: it works in your code.
    wt = git_worktree(".")
    if wt is not None:
        workspace, cleanup = wt
        print(f"working in a git worktree of this repo — {workspace.root}")
    else:
        workspace, cleanup = Workspace(), (lambda: None)
        print(f"not a git repo — working in a scratch dir — {workspace.root}")

    tools = default_tools()
    tools.register(write_file_tool(workspace))
    tools.register(edit_file_tool(workspace))
    # Trusted bash: your own project, your own test command — run for real, gated
    # by approval, not by network-none isolation (that stays for untrusted code).
    tools.register(bash_tool(Sandbox(trusted=True, timeout=120), workdir=str(workspace.root)))
    tools.register(search_memory_tool())  # episodic recall across past sessions
    tools.register(delegate_tool())  # hand a self-contained subtask to a fresh subagent
    tools.register(fan_out_tool())  # split into independent subtasks, run them in parallel

    def approve(name: str, args: str) -> bool:
        return input(f"  approve {name}({args})? [y/N] ").strip().lower() in ("y", "yes")

    parser = argparse.ArgumentParser(prog="agent")
    parser.add_argument(
        "--context-limit",
        type=int,
        default=DEFAULT_CONTEXT_LIMIT,
        help="token budget before the window is compacted (default: %(default)s). "
        "Set it low, e.g. 400, to watch compaction fire live.",
    )
    args = parser.parse_args()

    tracer = Tracer()  # ch-13: record every step; print the timeline after each turn
    agent = Agent(
        system=DEFAULT_SYSTEM,
        tools=tools,
        approve=approve,
        approval_required={"bash", "write_file", "edit_file"},
        context_limit=args.context_limit,
        skills=load_skills("skills"),
        session="repl",
        agents_dir=str(workspace.root),  # read AGENTS.md (incl. ## Testing) from the project
        tracer=tracer,
    )
    print(
        "agent ready (ch-13) — observable runs (a trace with tokens + cost after each "
        "turn); change code and the harness enforces the project's tests before 'done'; "
        "/plan; durable sessions, approval gate, skills. Ctrl-D to exit."
    )
    orchestrator = Orchestrator()
    try:
        while True:
            try:
                user = input("you> ")
            except EOFError:
                print()
                break
            if not user.strip():
                continue
            if user.startswith("/plan "):
                task = user[len("/plan ") :].strip()
                if not task:
                    print("usage: /plan <task>")
                    continue
                result = orchestrator.run(task)
                print("plan:")
                for i, step in enumerate(result.plan, 1):
                    print(f"  {i}. {step}")
                print("results:")
                for i, (step, res) in enumerate(zip(result.plan, result.results, strict=False), 1):
                    print(f"  {i}. {step}\n     → {res}")
                continue
            reply = agent.send(user)
            if agent.just_compacted:
                print("[context compacted — kept the start and end, summarized the middle]")
            print("bot>", reply)
            print(tracer.timeline())
    finally:
        cleanup()


if __name__ == "__main__":
    main()
