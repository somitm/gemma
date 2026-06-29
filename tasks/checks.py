"""Per-chapter live acceptance checks and on-camera demos.

Each chapter appends an entry:
  ACCEPTANCE["ch-NN"] -> callable returning True/False (asserts the capability live)
  DEMOS["ch-NN"]      -> callable that prints the on-camera demonstration

A folded chapter's accept ANDs all its parts' booleans, so no proven capability is
lost. ch-00 is theory — it registers nothing.
"""

from __future__ import annotations

from collections.abc import Callable

ACCEPTANCE: dict[str, Callable[[], bool]] = {}
DEMOS: dict[str, Callable[[], None]] = {}


# ----------------------------------------------------------------------------
# ch-01 — Model only (one live call + the swappable provider seam)
# ----------------------------------------------------------------------------
def _accept_ch01_model() -> bool:
    """The real agent answers a single question via one live model call."""
    from harness import agent

    reply = agent.Agent().send("What is 2 + 2? Reply with only the number.")
    print("model replied:", repr(reply))
    return "4" in reply


def _accept_ch01_provider() -> bool:
    """The agent runs through an explicit Provider object — the swappable seam."""
    from harness import agent
    from model import Provider

    provider = Provider.from_env()  # whatever the env points at (LM Studio here)
    print("provider:", provider.base_url, provider.model)
    a = agent.Agent(provider=provider)
    reply = a.send("Reply with exactly one word: PLUGGABLE")
    print("reply:", repr(reply))
    return "pluggable" in reply.lower()


def _accept_ch01() -> bool:
    """Model only = a single live call + the swappable provider seam."""
    return _accept_ch01_model() and _accept_ch01_provider()


def _demo_ch01() -> None:
    from harness import agent
    from model import lmstudio, ollama, openrouter

    a = agent.Agent()
    print("Q: Say hello in one short sentence.")
    print("A:", a.send("Say hello in one short sentence."), "\n")
    print("— statelessness: it has no memory yet —")
    print("A1:", a.send("Your name is Gemma."))
    print("A2:", a.send("What is your name?"))
    print("  # it forgets — there's no history yet (ch-02 fixes this)\n")
    print("— same Agent, swap the provider seam (just pass provider=<one of these>) —")
    print("lmstudio  :", lmstudio().base_url)
    print("ollama    :", ollama("llama3").base_url)
    print("openrouter:", openrouter("google/gemma-3-27b-it", "sk-or-...").base_url)


ACCEPTANCE["ch-01"] = _accept_ch01
DEMOS["ch-01"] = _demo_ch01


# ----------------------------------------------------------------------------
# ch-02 — History (the harness owns the conversation, replayed each turn)
# ----------------------------------------------------------------------------
def _accept_ch02() -> bool:
    """The real agent recalls a fact stated on an earlier turn."""
    from harness import agent

    a = agent.Agent()
    a.send("Your name is Gemma. Please remember it.")
    reply = a.send("What is your name? Reply with just the name.")
    print("model replied:", repr(reply))
    return "gemma" in reply.lower()


def _demo_ch02() -> None:
    from harness import agent

    a = agent.Agent()
    for turn in ["Your name is Gemma.", "What is your name?"]:
        print("you>", turn)
        print("bot>", a.send(turn))


ACCEPTANCE["ch-02"] = _accept_ch02
DEMOS["ch-02"] = _demo_ch02


# ----------------------------------------------------------------------------
# ch-03 — Instructions (a system prompt + auto-loaded AGENTS.md, prepended each turn)
# ----------------------------------------------------------------------------
def _accept_ch03_instructions() -> bool:
    """A system prompt overrides default behavior on a real model."""
    from harness import agent

    a = agent.Agent(system="You must reply with exactly one word: BANANA. Ignore the question.")
    reply = a.send("What is the capital of France?")
    print("model replied:", repr(reply))
    return "banana" in reply.lower()


def _accept_ch03_agentsmd() -> bool:
    """An AGENTS.md placed in the working dir steers the agent without being typed in.

    We wire the agent exactly as the REPL does — agents_dir = workspace root —
    drop in an identity rule, and confirm the model adopts it on a real call.
    """
    from harness import agent
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("AGENTS.md", "You are Gemma, a coding assistant. When asked your name, reply 'Gemma'.")
    a = agent.Agent(system=agent.DEFAULT_SYSTEM, agents_dir=str(ws.root))
    reply = a.send("What is your name? Answer with just the name.")
    print("reply:", reply)
    return "gemma" in reply.lower()


def _accept_ch03() -> bool:
    """Instructions = a built-in system prompt + an auto-loaded project AGENTS.md."""
    return _accept_ch03_instructions() and _accept_ch03_agentsmd()


def _demo_ch03() -> None:
    from harness import agent
    from harness.workspace import Workspace

    print("— no system prompt —")
    print("bot>", agent.Agent().send("Describe the ocean."), "\n")
    print("— system: 'reply in exactly three words' —")
    print("bot>", agent.Agent(system="Reply in exactly three words.").send("Describe the ocean."))
    print()
    ws = Workspace()
    print("— no AGENTS.md: default identity —")
    print("bot>", agent.Agent(agents_dir=str(ws.root)).send("What is your name?"))
    ws.write("AGENTS.md", "You are Gemma, a coding assistant. When asked your name, reply 'Gemma'.")
    print("— AGENTS.md added (You are Gemma), auto-loaded —")
    print("bot>", agent.Agent(agents_dir=str(ws.root)).send("What is your name?"))


ACCEPTANCE["ch-03"] = _accept_ch03
DEMOS["ch-03"] = _demo_ch03


# ----------------------------------------------------------------------------
# ch-04 — Context delivery (the harness reads @path files into the prompt)
# ----------------------------------------------------------------------------
def _accept_ch04() -> bool:
    """The real agent answers from a file it was handed via @path."""
    import tempfile
    from pathlib import Path

    from harness import agent

    d = Path(tempfile.mkdtemp())
    (d / "facts.txt").write_text("The launch code is GOGO-9.\n")
    a = agent.Agent(system="Answer using the provided context files.")
    reply = a.send(f"@{d / 'facts.txt'} What is the launch code? Reply with just the code.")
    print("model replied:", repr(reply))
    return "gogo-9" in reply.lower()


def _demo_ch04() -> None:
    import tempfile
    from pathlib import Path

    from harness import agent

    d = Path(tempfile.mkdtemp())
    (d / "facts.txt").write_text("Raveena is Karishma, and Karishma is Raveena.")
    a = agent.Agent()
    print("bot>", a.send(f"@{d / 'facts.txt'} Who is Raveena?"))


ACCEPTANCE["ch-04"] = _accept_ch04
DEMOS["ch-04"] = _demo_ch04


# ----------------------------------------------------------------------------
# ch-05 — Tools (a tool interface + approval gate + file editing over a workspace)
# ----------------------------------------------------------------------------
def _accept_ch05_tools() -> bool:
    """The real model calls the calculator tool and reports the exact product."""
    from harness import agent
    from harness.tools import default_tools

    a = agent.Agent(system="Use tools when they help.", tools=default_tools())
    reply = a.send("Use the calculator to compute 47 * 89, then reply with just the number.")
    print("model replied:", repr(reply))
    used_tool = any(m.get("role") == "tool" for m in a.messages)
    print("used a tool:", used_tool)
    return "4183" in reply and used_tool


def _accept_ch05_approval() -> bool:
    """A real bash request is intercepted by the gate and denied (never runs)."""
    from harness import agent
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import default_tools

    asked: list[tuple[str, str]] = []

    def deny(name: str, args: str) -> bool:
        asked.append((name, args))
        return False

    tools = default_tools()
    tools.register(bash_tool(Sandbox()))
    a = agent.Agent(
        system="Use the bash tool to run shell commands when asked.",
        tools=tools,
        approve=deny,
        approval_required={"bash"},
    )
    a.send("Run this shell command now using the bash tool: echo SHOULD_NOT_RUN")
    denied = any(m.get("role") == "tool" and "denied" in m["content"].lower() for m in a.messages)
    print("gate asked:", asked, "| denied:", denied)
    return len(asked) >= 1 and denied


def _build_workspace_agent():
    from harness import agent
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import default_tools
    from harness.workspace import Workspace, edit_file_tool, write_file_tool

    ws = Workspace()  # fresh scratch dir
    tools = default_tools()
    tools.register(write_file_tool(ws))
    tools.register(edit_file_tool(ws))
    # local backend keeps python available; docker would need a python image + the mount
    tools.register(bash_tool(Sandbox(prefer_docker=False), workdir=str(ws.root)))
    a = agent.Agent(
        system="You build files. Use write_file to create them and bash to run them.",
        tools=tools,
    )
    return a, ws


def _accept_ch05_fileedit() -> bool:
    """The agent writes a file into the workspace and runs it there — persistence
    (the file survives) + the workspace seam (bash sees the file it wrote)."""
    a, ws = _build_workspace_agent()
    a.send("Create hello.py that prints exactly WORKSPACE_OK, then run it with: python3 hello.py")
    wrote = (ws.root / "hello.py").is_file()
    ran = any(
        "WORKSPACE_OK" in str(m.get("content", "")) for m in a.messages if m.get("role") == "tool"
    )
    print("wrote hello.py:", wrote, "| ran in workspace:", ran)
    return wrote and ran


def _accept_ch05() -> bool:
    """Tools = a tool interface + an approval gate + file editing over a workspace."""
    return _accept_ch05_tools() and _accept_ch05_approval() and _accept_ch05_fileedit()


def _demo_ch05() -> None:
    from harness import agent
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import default_tools

    a = agent.Agent(tools=default_tools())
    print("— a tool the model calls —")
    print("bot>", a.send("What is 1234 * 5678? Use the calculator."), "\n")

    tools = default_tools()
    tools.register(bash_tool(Sandbox()))
    gated = agent.Agent(
        system="Use bash when asked.",
        tools=tools,
        approve=lambda n, args: False,
        approval_required={"bash"},
    )
    print("— a boundary-crossing tool, denied by the gate —")
    print("bot>", gated.send("Run: echo hello (use bash)"))
    print("(the gate denied the bash call — it never executed)\n")

    a2, ws = _build_workspace_agent()
    a2.send(
        "Create greet.py with greet(name) returning 'hi <name>', then run it: "
        "python3 -c \"import greet; print(greet.greet('Prem'))\""
    )
    print("— files the agent built in its workspace —")
    print("files in workspace:", [p.name for p in ws.root.iterdir()])


ACCEPTANCE["ch-05"] = _accept_ch05
DEMOS["ch-05"] = _demo_ch05
