"""Entry point: A2A server hosting the FieldWorkArena purple agent."""
from __future__ import annotations

import argparse
import logging
import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from .executor import FWAExecutor


def main() -> None:
    parser = argparse.ArgumentParser(description="FieldWorkArena Purple Agent (gpt-5-mini, multimodal)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9019)
    parser.add_argument("--card-url", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    skill = AgentSkill(
        id="fieldworkarena_agent",
        name="FieldWorkArena field-ops analyst",
        description=(
            "Multimodal field-operations analyst that answers FieldWorkArena "
            "factory/warehouse/retail tasks from images, PDFs, and videos."
        ),
        tags=["field-work", "multimodal", "factory", "warehouse", "retail", "ppe"],
        examples=[
            "Are the workers wearing aprons in this image?",
            "How many incidents were there regarding 'distance between the cart and the worker less than 1 meter'?",
            "Please extract and list the items for the 'pre-shift inspection of electrical insulating rubber gloves' in this PDF file.",
        ],
    )

    agent_card = AgentCard(
        name="FieldWorkArena Purple Agent (gpt-5-mini)",
        description=(
            "Multimodal purple agent for the FieldWorkArena AgentBeats benchmark. "
            "Uses gpt-5-mini via the OpenAI Responses API to read images, PDFs, "
            "and video frames, and produces format-faithful answers (text or JSON)."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="0.1.0",
        skills=[skill],
        default_input_modes=[
            "text", "text/plain",
            "application/pdf",
            "image/jpeg", "image/png",
            "video/mp4",
        ],
        default_output_modes=["text", "text/plain"],
        capabilities=AgentCapabilities(streaming=True),
    )

    handler = DefaultRequestHandler(
        agent_executor=FWAExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

    uvicorn.run(app.build(), host=args.host, port=args.port, timeout_keep_alive=300)


if __name__ == "__main__":
    main()
