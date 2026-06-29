"""The agent — the harness drive loop. Grows one primitive per chapter.

ch-08 — Execution environment. The model never runs code; the harness does,
inside a boundary (``harness/sandbox.py``, now hardened: no network, non-root, a
scrubbed environment with no inherited credentials). Two more boundary fixes land
with it: ``read_file`` is confined to the workspace (no ``/etc/passwd``), and the
verifier (``harness/verification.py``, exercised by tools, not yet by the loop)
runs candidate code in a scrubbed process.

The loop's only change is how compaction earns its token count. Until now it
*estimated* the window from message length. Now that the boundary work has us
reading the model's reported usage, ``_run`` captures ``resp.usage`` into
``self._last_tokens`` and ``_maybe_compact`` *prefers that real number*, falling
back to the estimate only on turn one. We were guessing; now the provider tells
us. Everything from ch-07 is unchanged: skills, the managed window's door control
(``clamp``), ``@path`` injection, tools behind the approval gate.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from harness.compaction import compact, estimate_tokens
from harness.context import deliver
from harness.instructions import load_agents_md
from harness.limits import clamp
from harness.skills import Skill, skills_prompt
from harness.tools import ToolRegistry
from model import Provider, chat

DEFAULT_SYSTEM = "You are a concise, helpful coding assistant. Use tools when they help."
MAX_TOOL_STEPS = 6
DEFAULT_CONTEXT_LIMIT = 4000  # ~tokens; compact the history above this


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
        self.messages: list[dict] = []
        # Set true whenever the last turn triggered compaction — the REPL reads
        # this to surface that the window was managed (a demoable, visible event).
        self.just_compacted = False

    def _approved(self, name: str, args: str) -> bool:
        # Fail closed: a tool marked as requiring approval with no approver is denied.
        return self.approve(name, args) if self.approve else False

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
        """Inject any @path files, append the turn, then drive the tool loop."""
        for block in deliver(user_text):  # @file references → injected context
            self.messages.append({"role": "user", "content": f"Context file:\n{block}"})
        self.messages.append({"role": "user", "content": user_text})
        return self._run()

    def _run(self) -> str:
        """Drive the model, executing tool calls until it produces a final answer."""
        self._maybe_compact()
        specs = self.tools.specs() if self.tools else None
        for _ in range(MAX_TOOL_STEPS):
            resp = chat(self._payload(), model=self.model, tools=specs, provider=self.provider)
            self._last_tokens = int(resp.usage.get("total_tokens", 0)) or self._last_tokens
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
                    # A boundary-crossing tool must clear the approval gate first.
                    if name in self.approval_required and not self._approved(name, args):
                        result = "[denied by approval gate]"
                    else:
                        result = self.tools.call(name, args)
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc.get("id", ""), "content": clamp(result)}
                    )
                continue
            self.messages.append({"role": "assistant", "content": resp.content})
            return resp.content
        return "error: exceeded tool-step budget"


def main() -> None:
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import default_tools
    from harness.workspace import Workspace, edit_file_tool, write_file_tool

    # The REPL owns a scratch workspace: the file tools write into it and bash runs
    # over the same dir, so a command sees the file the model just wrote.
    workspace = Workspace()
    tools = default_tools()
    tools.register(write_file_tool(workspace))
    tools.register(edit_file_tool(workspace))
    tools.register(bash_tool(Sandbox(), workdir=str(workspace.root)))

    def approve(name: str, args: str) -> bool:
        return input(f"  approve {name}({args})? [y/N] ").strip().lower() in ("y", "yes")

    from harness.skills import load_skills

    parser = argparse.ArgumentParser(prog="agent")
    parser.add_argument(
        "--context-limit",
        type=int,
        default=DEFAULT_CONTEXT_LIMIT,
        help="token budget before the window is compacted (default: %(default)s). "
        "Set it low, e.g. 400, to watch compaction fire live.",
    )
    args = parser.parse_args()

    agent = Agent(
        system=DEFAULT_SYSTEM,
        tools=tools,
        approve=approve,
        approval_required={"bash", "write_file", "edit_file"},
        context_limit=args.context_limit,
        skills=load_skills("skills"),
    )
    print(
        "agent ready (ch-08) — sandboxed tools, approval gate, managed window, skills. "
        "Ctrl-D to exit."
    )
    while True:
        try:
            user = input("you> ")
        except EOFError:
            print()
            break
        if not user.strip():
            continue
        reply = agent.send(user)
        if agent.just_compacted:
            print("[context compacted — kept the start and end, summarized the middle]")
        print("bot>", reply)


if __name__ == "__main__":
    main()
