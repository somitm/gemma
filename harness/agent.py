"""The agent — the harness drive loop. Grows one primitive per chapter.

ch-05 — Tools. Until now the model could only talk; now it can *act*. It returns
tool calls, the harness runs them, feeds the results back, and the model keeps
going until it has a final answer. The model decides *what* to do; the harness
decides *how* — and is the one that actually runs anything.

``send`` now ends in ``_run``: a bounded loop (``MAX_TOOL_STEPS``) that calls the
model with the available tool specs. If the reply asks for tools, the harness
appends the assistant turn (with its tool calls), executes each call, appends the
raw result as a ``tool`` message, and loops. Otherwise it returns the final text.

Some tools cross a boundary (a shell, the filesystem). Those can be named in
``approval_required``; before such a tool runs, the harness asks an ``approve``
callback, and feeds back ``[denied by approval gate]`` if refused. With no
approver wired, a gated tool *fails closed* — it is denied, never run.

The instruction layers (ch-03) and context delivery (ch-04) are unchanged: the
system prompt + AGENTS.md head the payload, and ``@path`` files are injected
before the turn. The single ``chat`` call still goes through the ``model/`` seam.
"""

from __future__ import annotations

from collections.abc import Callable

from harness.context import deliver
from harness.instructions import load_agents_md
from harness.tools import ToolRegistry
from model import Provider, chat

DEFAULT_SYSTEM = "You are a concise, helpful coding assistant. Use tools when they help."
MAX_TOOL_STEPS = 6


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
    ) -> None:
        self.model = model
        self.provider = provider
        self.system = system
        self.agents_dir = agents_dir  # where AGENTS.md is auto-loaded from
        self.tools = tools
        self.approve = approve
        self.approval_required = approval_required or set()
        self.messages: list[dict] = []

    def _approved(self, name: str, args: str) -> bool:
        # Fail closed: a tool marked as requiring approval with no approver is denied.
        return self.approve(name, args) if self.approve else False

    def _system_text(self) -> str:
        """The instruction layer = built-in system prompt + project AGENTS.md."""
        parts = [p for p in (self.system, load_agents_md(self.agents_dir)) if p]
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
        specs = self.tools.specs() if self.tools else None
        for _ in range(MAX_TOOL_STEPS):
            resp = chat(self._payload(), model=self.model, tools=specs, provider=self.provider)
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
                        {"role": "tool", "tool_call_id": tc.get("id", ""), "content": result}
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

    agent = Agent(
        system=DEFAULT_SYSTEM,
        tools=tools,
        approve=approve,
        approval_required={"bash", "write_file", "edit_file"},
    )
    print("agent ready (ch-05) — with tools + an approval gate. Ctrl-D to exit.")
    while True:
        try:
            user = input("you> ")
        except EOFError:
            print()
            break
        if not user.strip():
            continue
        print("bot>", agent.send(user))


if __name__ == "__main__":
    main()
