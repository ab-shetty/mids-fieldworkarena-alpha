"""A2A executor that bridges the green agent ↔ FWAAgent."""
from __future__ import annotations

import logging
import os
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Message,
    Part,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)

from .agent import FWAAgent
from .multimodal import parts_to_input

logger = logging.getLogger(__name__)

TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class FWAExecutor(AgentExecutor):
    def __init__(self) -> None:
        self._agent: FWAAgent | None = None

    def _ensure_agent(self) -> FWAAgent:
        if self._agent is None:
            self._agent = FWAAgent()
        return self._agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        if not message or not message.parts:
            logger.warning("Empty A2A message")
            return

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            return

        task_id = context.task_id or "unknown"
        context_id = context.context_id or "unknown"

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                status=TaskStatus(
                    state=TaskState.working,
                    message=Message(
                        messageId=uuid4().hex,
                        role="agent",
                        parts=[Part(root=TextPart(kind="text", text="Analyzing inputs…"))],
                    ),
                ),
                final=False,
            )
        )

        try:
            agent_input = parts_to_input(message.parts)
            # Reconstruct the goal text from text parts (the green agent puts
            # the whole goal in a single TextPart, but we join in case it's
            # split across multiple).
            goal = "\n".join(agent_input.text_blocks[: max(1, len(agent_input.text_blocks))])
            # The goal is actually the FIRST text block (the rest are
            # PDF/text-file extracts joined in by parts_to_input).  Identify
            # which blocks are the goal vs file extracts.
            goal_blocks = [t for t in agent_input.text_blocks if not t.startswith("=== ")]
            file_blocks = [t for t in agent_input.text_blocks if t.startswith("=== ")]
            goal = "\n\n".join(goal_blocks) if goal_blocks else (agent_input.text_blocks[0] if agent_input.text_blocks else "")
            agent_input.text_blocks = file_blocks

            agent = self._ensure_agent()
            result = await _to_thread(agent.answer, goal, agent_input)
            response = result.answer or ""
            logger.info(
                "Answered task %s: format=%s images=%d text=%d len=%d",
                task_id, result.output_format, result.n_images, result.n_text_blocks, len(response),
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Agent failure: %s", e)
            response = f"(error: {e})"

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                status=TaskStatus(
                    state=TaskState.completed,
                    message=Message(
                        messageId=uuid4().hex,
                        role="agent",
                        parts=[Part(root=TextPart(kind="text", text=response))],
                    ),
                ),
                final=True,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError(message="Cancellation not supported")


async def _to_thread(fn, *args, **kwargs):
    import asyncio
    return await asyncio.to_thread(fn, *args, **kwargs)
