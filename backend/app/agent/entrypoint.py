"""Canonical conversational-agent entry point declaration.

Channels must call ``app.agent.pipeline.process_message``. The two earlier
agent implementations remain only as historical references until they can be
deleted in a dedicated cleanup release.
"""

CANONICAL_AGENT_ENTRYPOINT = "app.agent.pipeline.process_message"
QUARANTINED_AGENT_ENTRYPOINTS = frozenset({
    "app.agent.nourishbot_agent.get_nourishbot_response",
    "app.agent.orchestrator.get_bot_response",
})


def reject_legacy_agent(entrypoint: str) -> None:
    raise RuntimeError(
        f"{entrypoint} is quarantined; use {CANONICAL_AGENT_ENTRYPOINT}"
    )
