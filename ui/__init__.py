"""UI package — the only package allowed to import textual/rich.

Dependencies point ui -> core (harness/model), never core -> ui. The TUI consumes
what the harness ``Tracer`` records; it does not reimplement the agent.
"""
