"""FieldWorkArena reasoning agent.

The green agent sends a goal string of the form:

    # Question
    <natural-language question>

    # Input Data
    file_a.jpg
    file_b.pdf
    ...

    # Output Format
    text   |   json   |   <other hint>

…plus the actual file bytes as A2A FileParts. The green judge then grades
our reply with one of: fuzzy_match, numerical_match, json_match,
exact_match, must_include, must_exclude.

Strategy:
- Detect the output format (text vs json) from the goal.
- Build a single multimodal Responses-API call with all images inline plus
  any extracted PDF/text content as text blocks.
- Use a system prompt that pushes hard on (a) following the output format
  exactly, (b) precision in numerical answers, (c) no hedging, no prose
  preamble, (d) JSON validity when format=json.
- For json output, request structured output via response_format=json_object
  so it parses cleanly for json_match.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

from .multimodal import AgentInput
from .providers import LLMProvider, make_provider

logger = logging.getLogger(__name__)

# Model is submitter-selectable. `OPENAI_MODEL` keeps backward compat with the
# original variable name; `LLM_MODEL` is the provider-neutral alias.
MODEL = os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-5-mini")
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "medium")
REASONING_EFFORT_JSON = os.environ.get("REASONING_EFFORT_JSON", "high")
REASONING_EFFORT_NUMERIC = os.environ.get("REASONING_EFFORT_NUMERIC", "high")
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "4096"))


SYSTEM_BASE = """You are a meticulous field-operations analyst working on the
FieldWorkArena benchmark. You answer questions about factory, warehouse, and
retail scenes from images, PDF documents, and (occasionally) video frames.

You will be given:
  - A QUESTION from the operator.
  - INPUT DATA: a list of attached files. Their actual contents are provided
    as image attachments (for images and video frames) and inline text (for
    PDFs and text files). Always treat the attached content as authoritative.
  - An OUTPUT FORMAT hint (usually "text" or "json").

Answering rules:
1. Read the attached evidence carefully before answering. For images, look at
   workers, posture, PPE (helmets/gloves/aprons/masks/goggles/safety vests),
   tools, distances, signage, fluids/leaks, and any visible numbers.
2. For YES/NO questions, answer "Yes." or "No." followed by ONE short
   restating clause that confirms WHAT you affirmed/denied. The grading judge
   penalises bare "Yes."/"No." answers when the gold is more substantive.
   Examples:
     Q: "Are the workers wearing aprons in this image?"
        → "Yes, the workers are wearing aprons."
     Q: "Is the worker holding a tool?"
        → "No, the worker is not holding a tool."
3. For COUNT questions, answer "There were N." (or "N <object(s)>.") with N
   in Arabic digits. Examples: "There were 2.", "3 violations.", "0".
4. For DISTANCE questions, ALWAYS estimate a specific numeric distance with
   units, even if the question is phrased boolean ("is X within 1 meter?"
   → "No, it is 1.5 meters."). Use these reference scales when estimating
   from images:
     - average worker shoulder-to-shoulder ≈ 0.45 m
     - average worker height ≈ 1.7 m
     - typical pallet ≈ 1.0–1.2 m wide
     - typical industrial cart wheel ≈ 0.15 m
   Round to one decimal place in meters (e.g. "0.7 meters", "1.5 meters").
5. For TIME / TIMESTAMP questions, use HH:MM:SS for instants and
   "from HH:MM:SS to HH:MM:SS" for ranges.
6. For LIST/EXTRACTION questions over PDFs, give every required item from
   the source document, copying terminology verbatim where possible. Use
   the same enumeration style (①②③, "1.", "-") as the source if visible.
7. Never hedge with multiple candidate values. Pick one. Do not write
   "approximately X or Y" or "between X and Y" unless that range IS the
   answer the question asks for.
8. Never refuse. Never apologise. Never say "I cannot determine". Make your
   best, most specific answer from the evidence.
9. Do not invent file timestamps, IDs, or measurements that are not in the
   evidence — but DO commit to a best estimate where the question requires
   one (rule 4).
10. Answer in the same language as the question (English in / English out).
11. Output ONLY the answer. No preamble like "Sure, here is…", no markdown
    code fences, no extra commentary outside the answer itself.
"""

SYSTEM_JSON_SUFFIX = """

This task requires a JSON output. Emit a SINGLE JSON object that exactly
matches the schema described in the question. Common pitfalls to avoid:
  - Use the EXACT field names quoted in the question (case-sensitive).
  - When the question says "starting from N", the FIRST item's ID is N,
    the next is N+1, and so on. Do NOT reuse IDs.
  - When the question asks for "total_violations", that field must equal
    the length of the details list.
  - String fields like "Filename" must contain the file basename verbatim
    (with extension), as listed in the INPUT DATA section.
  - Numeric fields stay numeric (no quoting numbers); identifier fields
    that are clearly strings ("ID": "1120") stay strings.
  - When a field's value is unknown but required, copy what the question
    suggests; do not invent values.
  - No trailing commas, no comments, no markdown fences. The very first
    character of your reply must be "{" and the very last must be "}".
"""

SYSTEM_NUMERIC_SUFFIX = """

This task will be graded on the numerical value(s) you produce. Be exact:
  - Do not round unless the question asks for a rounded answer.
  - Include units when the question implies them (e.g. "0.7 meters",
    "00:03:18 to 00:03:24"), and write numbers as Arabic digits.
  - When counting, give a single integer.
  - When measuring distances/times, use the SAME unit as the question.

For distance estimation in images, anchor on a reference object with known
size (worker shoulders ≈ 0.45 m, worker height ≈ 1.7 m, hard hat ≈ 0.3 m,
pallet ≈ 1.0 m wide, standard floor tile ≈ 0.6 m square, cart body ≈ 0.6 m
wide). Most workplace clearances in this benchmark are between 0.3 m and
3.0 m — values outside that range are usually wrong.

For boolean+quantity questions ("is X less than 1 m?"), DO NOT FLIP the
boolean. First decide the boolean from common sense (close = yes, far =
no), then attach a quantity that is consistent with that boolean. If you
say "No, X is greater than 1 m", your distance must be > 1 m.
"""


_FORMAT_RE = re.compile(r"^#\s*Output\s*Format\s*\n([^\n#]+)", re.IGNORECASE | re.MULTILINE)
_QUESTION_RE = re.compile(r"#\s*Question\s*\n(.*?)(?=\n#\s|\Z)", re.IGNORECASE | re.DOTALL)


def _detect_output_format(goal: str) -> str:
    m = _FORMAT_RE.search(goal or "")
    if m:
        v = m.group(1).strip().lower()
        if "json" in v:
            return "json"
        return "text"
    return "text"


def _detect_numeric_intent(goal: str) -> bool:
    q = ""
    m = _QUESTION_RE.search(goal or "")
    if m:
        q = m.group(1).lower()
    triggers = (
        "how many", "count", "number of", "how long", "how much",
        "what time", "start time", "end time", "timestamp",
        "distance", "duration", "ratio", "percent",
    )
    return any(t in q for t in triggers)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_to_json(text: str) -> str:
    """Best-effort: extract a single JSON object from a model reply."""
    if not text:
        return text
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _JSON_OBJECT_RE.search(text)
    if m:
        return m.group(0).strip()
    return text.strip()


@dataclass
class AgentResult:
    answer: str
    output_format: str
    n_images: int
    n_text_blocks: int


class FWAAgent:
    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider or make_provider(MODEL)

    def _system_prompt(self, output_format: str, numeric: bool) -> str:
        sys = SYSTEM_BASE
        if output_format == "json":
            sys += SYSTEM_JSON_SUFFIX
        if numeric:
            sys += SYSTEM_NUMERIC_SUFFIX
        return sys

    def _build_user_text(self, goal: str, agent_input: AgentInput) -> str:
        text = goal
        if agent_input.text_blocks:
            text += "\n\n# Attached document content\n" + "\n\n".join(agent_input.text_blocks)
        if agent_input.file_summary:
            text += "\n\n# Attachment summary\n" + "\n".join(agent_input.file_summary)
        return text

    def answer(self, goal: str, agent_input: AgentInput) -> AgentResult:
        output_format = _detect_output_format(goal)
        numeric = _detect_numeric_intent(goal)
        system = self._system_prompt(output_format, numeric)

        if output_format == "json":
            effort = REASONING_EFFORT_JSON
        elif numeric:
            effort = REASONING_EFFORT_NUMERIC
        else:
            effort = REASONING_EFFORT

        user_text = self._build_user_text(goal, agent_input)

        logger.info(
            "FWA call: model=%s effort=%s format=%s images=%d text_blocks=%d",
            MODEL, effort, output_format, len(agent_input.images), len(agent_input.text_blocks),
        )

        raw = self.provider.respond(
            system=system,
            text=user_text,
            images=agent_input.images,
            output_format=output_format,
            effort=effort,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        if output_format == "json":
            cleaned = _strip_to_json(raw)
            # validate JSON; if it fails, fall back to raw
            try:
                json.loads(cleaned)
                raw = cleaned
            except json.JSONDecodeError:
                logger.warning("Model JSON failed to parse; sending raw text.")

        return AgentResult(
            answer=raw,
            output_format=output_format,
            n_images=len(agent_input.images),
            n_text_blocks=len(agent_input.text_blocks),
        )
