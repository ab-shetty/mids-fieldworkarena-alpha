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
import base64
from io import BytesIO
from dataclasses import dataclass

from PIL import Image

from .multimodal import AgentInput
from .providers import LLMProvider, make_provider

logger = logging.getLogger(__name__)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "medium")
REASONING_EFFORT_JSON = os.environ.get("REASONING_EFFORT_JSON", "high")
REASONING_EFFORT_NUMERIC = os.environ.get("REASONING_EFFORT_NUMERIC", "high")
REASONING_EFFORT_VERIFY = os.environ.get("REASONING_EFFORT_VERIFY", "low")
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "4096"))
ENABLE_SECOND_PASS_VERIFY = os.environ.get("ENABLE_SECOND_PASS_VERIFY", "0").lower() not in {"0", "false", "no"}
ENABLE_BBOX_JSON_CONFIRM = os.environ.get("ENABLE_BBOX_JSON_CONFIRM", "1").lower() not in {"0", "false", "no"}


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

SYSTEM_BOX_SUFFIX = """

This question uses image coordinates / a bounding box.
  - Treat coordinates as image pixels with origin at the TOP-LEFT corner.
  - x increases to the right; y increases downward.
  - Evaluate only the rectangular region inside the specified coordinates.
  - Ignore workers outside the rectangle, even if they are more visually salient.
  - For "is a worker located within the area" questions, answer based on
    whether a worker is visibly inside that rectangle, not merely nearby.
  - If derived crop attachments are provided, they are zoomed views of the
    specified rectangle. Use them as the primary evidence for boxed-region
    judgements and use the full image only for context.
"""

SYSTEM_FACING_SUFFIX = """

This question asks about facing direction inside a bounding box.
  - Judge the orientation of the worker INSIDE the specified rectangle only.
  - Use head / torso direction, not the direction of travel of another worker.
  - If the worker's body orientation is ambiguous, prefer the clearest visible
    orientation cue in the boxed region.
"""

SYSTEM_LONG_SLEEVE_SUFFIX = """

This question asks about long-sleeved clothing.
  - Long sleeves can be partly obscured by an apron, gloves, tools, or pose.
  - Check whether the garment appears to extend from shoulder toward wrist on
    both arms before concluding "No".
  - Do not confuse a dark apron or vest layer with bare forearms.
"""

SYSTEM_JSON_ISSUE_SUFFIX = """

This JSON task is an incident / violation report.
  - DO NOT try to match the final benchmark schema directly.
  - Instead, emit this normalized JSON schema only:
    {"items":[
      {
        "filename":"<exact attached filename>",
        "category":"violation|incident",
        "short_description":"<brief label>",
        "description":"<brief detail>",
        "distance_meters": <number or null>
      }
    ]}
  - Include ONLY items that should appear in the final report.
  - If there are no violations/incidents, return exactly: {"items":[]}
  - Use the exact attached filename including extension.
  - Use "incident" for Category when the question says incident(s); otherwise
    use "violation".
  - For distance reports, set "distance_meters" when a distance is relevant.
  - Do not invent timestamps, previous-task findings, or missing-step claims
    that are not directly supported by the current attachments.
"""

SYSTEM_EVIDENCE_ONLY_SUFFIX = """

Base the answer ONLY on the attachments included in the current task.
  - If the question mentions previous tasks, a video clip, or timestamps, but
    those materials are not actually attached here, do not invent missing-step
    or timestamp-based violations from absent evidence.
  - Do not assume unobserved steps failed simply because they are not visible
    in a still image.
"""

SYSTEM_VERIFY = """

You are verifying a draft answer to a FieldWorkArena task using the SAME
attached evidence.

Review the evidence again before deciding whether the draft is correct. Focus
on common failure modes:
  - wrong worker/object pair
  - wrong inside/outside judgement for the specified coordinate rectangle
  - wrong facing direction for the worker inside the rectangle
  - sleeves partly hidden by apron, gloves, or pose
  - boolean distance statement inconsistent with the numeric distance
  - numeric distance obviously too small or too large for the visible scene

If the draft answer is correct, repeat it exactly.
If it is wrong, output a corrected final answer.
Output ONLY the final answer. No explanation.
"""


_FORMAT_RE = re.compile(r"^#\s*Output\s*Format\s*\n([^\n#]+)", re.IGNORECASE | re.MULTILINE)
_QUESTION_RE = re.compile(r"#\s*Question\s*\n(.*?)(?=\n#\s|\Z)", re.IGNORECASE | re.DOTALL)
_INPUT_DATA_RE = re.compile(r"#\s*Input\s*Data\s*\n(.*?)(?=\n#\s|\Z)", re.IGNORECASE | re.DOTALL)
_COORD_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")

MAX_COORDINATE_CROPS = int(os.environ.get("MAX_COORDINATE_CROPS", "4"))
COORDINATE_CROP_MARGIN_RATIO = float(os.environ.get("COORDINATE_CROP_MARGIN_RATIO", "0.18"))


def _detect_output_format(goal: str) -> str:
    m = _FORMAT_RE.search(goal or "")
    if m:
        v = m.group(1).strip().lower()
        if "json" in v:
            return "json"
        return "text"
    return "text"


def _is_no_answer(text: str) -> bool:
    return text.strip().lower().startswith("no")


def _detect_numeric_intent(goal: str) -> bool:
    q = _question_text(goal)
    triggers = (
        "how many", "count", "number of", "how long", "how much",
        "what time", "start time", "end time", "timestamp",
        "distance", "duration", "ratio", "percent",
    )
    return any(t in q for t in triggers)


def _question_text_raw(goal: str) -> str:
    m = _QUESTION_RE.search(goal or "")
    if not m:
        return ""
    return m.group(1).strip()


def _question_text(goal: str) -> str:
    return _question_text_raw(goal).lower()


def _is_box_task(question: str) -> bool:
    return any(p in question for p in ("bounding box", "defined area", "defined by coordinates", "coordinates ("))


def _is_facing_task(question: str) -> bool:
    return "facing left" in question or "facing right" in question or "facing within" in question


def _is_long_sleeve_task(question: str) -> bool:
    return "long-sleeved" in question or "long sleeved" in question


def _is_issue_report_json_task(question: str, output_format: str) -> bool:
    if output_format != "json":
        return False
    return any(
        phrase in question
        for phrase in (
            "create a new issue",
            "incident report",
            "report the findings",
            "identified violations",
            "identified incidents",
        )
    )


def _is_bbox_issue_report_json_task(question: str, output_format: str) -> bool:
    if not _is_issue_report_json_task(question, output_format):
        return False
    return "bounding box" in question or "defined area" in question or "coordinates (" in question


def _should_verify(question: str, output_format: str, numeric: bool, agent_input: AgentInput) -> bool:
    if not ENABLE_SECOND_PASS_VERIFY:
        return False
    if output_format != "text":
        return False
    if not agent_input.images:
        return False
    return any(
        (
            numeric,
            _is_box_task(question),
            _is_facing_task(question),
            _is_long_sleeve_task(question),
        )
    )


def _mentions_missing_context(question: str) -> bool:
    return any(
        phrase in question
        for phrase in (
            "previous task",
            "previous tasks",
            "video clip",
            "video's timestamp",
            "video timestamp",
            "start and end times",
        )
    )


def _extract_starting_id(question: str) -> int | None:
    m = re.search(r"starting from\s+(\d+)", question)
    if not m:
        m = re.search(r"numbered sequentially from\s+(\d+)", question)
    if not m:
        m = re.search(r"sequentially numbered from\s+(\d+)", question)
    if not m:
        return None
    return int(m.group(1))


def _extract_input_files(goal: str) -> list[str]:
    m = _INPUT_DATA_RE.search(goal or "")
    if not m:
        return []
    lines = [line.strip() for line in m.group(1).splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _is_image_file(name: str) -> bool:
    lower = name.lower()
    return lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"))


def _distance_short_description(question: str) -> str | None:
    lead = re.sub(r"\s+", " ", question).strip().lower()
    patterns = (
        r'distance between the (.+?) and the (.+?) less than',
        r'estimate the minimum distance between the (.+?) and the (.+?)(?:,| if | next,| then,|$)',
        r'distance between the (.+?) and the (.+?)(?:,| if | next,| then,|$)',
    )
    for pattern in patterns:
        m = re.search(pattern, lead)
        if not m:
            continue
        left = m.group(1).strip()
        right = m.group(2).strip()
        right = re.sub(r"\.\s*if this distance is.*", "", right).strip()
        right = re.sub(r"\bif this distance is.*", "", right).strip()
        right = re.sub(r"\bnext,.*", "", right).strip()
        right = re.sub(r"\bthen,.*", "", right).strip()
        left = left.replace("nearest worker", "worker")
        right = right.replace("nearest worker", "worker")
        return f"Distance between the {left} and the {right}"
    return None


def _title_field_name(question: str, field: str) -> str:
    pattern = re.compile(rf'"({re.escape(field)}|{re.escape(field.lower())}|{re.escape(field.title())})"')
    m = pattern.search(question)
    if m:
        return m.group(1)
    return field


def _wants_field(question: str, field: str) -> bool:
    lower = question.lower()
    field_lower = field.lower()
    if f'field "{field_lower}"' in lower:
        return True
    return re.search(rf'\b{re.escape(field_lower)}\b\s*\(', lower) is not None


def _filename_field_name(question: str) -> str:
    if "image filename" in question.lower():
        return "Image filename"
    return _title_field_name(question, "Filename")


def _field_constant(question: str, field: str) -> str | None:
    patterns = (
        rf'value of "([^"]+)" for field "{re.escape(field)}"',
        rf'{re.escape(field.lower())}\s*\("([^"]+)"\)',
    )
    for pattern in patterns:
        m = re.search(pattern, question, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _decode_maybe_base64_text(text: str) -> str:
    stripped = text.strip()
    if "(" in stripped or "\t" in stripped:
        return text
    try:
        decoded = base64.b64decode(stripped, validate=True).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return text
    if "(" in decoded or "\t" in decoded:
        return decoded
    return text


def _parse_coordinate_map(agent_input: AgentInput) -> dict[str, tuple[int, int, int, int]]:
    mapping: dict[str, tuple[int, int, int, int]] = {}
    for block in agent_input.text_blocks:
        if "Image_Bounding_Box_Coordinates.txt" not in block:
            continue
        content = block.split("\n", 1)[1] if "\n" in block else block
        content = _decode_maybe_base64_text(content)
        for line in content.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("image"):
                continue
            parts = line.split("\t")
            filename = parts[0].strip().strip('"')
            box = _extract_coordinate_box(line)
            if filename and box is not None:
                mapping[filename] = box
    return mapping


def _single_image_input(agent_input: AgentInput, filename: str) -> AgentInput | None:
    for name, jpeg in agent_input.images:
        if name != filename:
            continue
        size = agent_input.original_sizes.get(name)
        if size is None:
            continue
        return AgentInput(
            text_blocks=[],
            images=[(name, jpeg)],
            file_summary=[f"image: {name}"],
            original_sizes={name: size},
            derived_views=[],
        )
    return None


def _fallback_description(question: str, filename: str, short_description: str) -> str:
    q = question.lower()
    if "no worker was detected within the bounding box" in q:
        return f"In {filename}, no worker was detected within the bounding box."
    if "no worker facing left was detected within the bounding box" in q:
        return f"In {filename}, no worker facing left was detected within the bounding box."
    if "distance between the worker and the cart was less than 1 meter" in q:
        return f"In {filename}, the distance between the worker and the cart was less than 1 meter."
    return short_description


def _canonical_issue_key(key: str) -> str | None:
    norm = key.strip().lower().replace("_", " ").replace("-", " ")
    mapping = {
        "id": "ID",
        "category": "Category",
        "short description": "Short description",
        "filename": "Filename",
        "image filename": "Filename",
        "description": "Description",
    }
    return mapping.get(norm)


def _extract_coordinate_box(question: str) -> tuple[int, int, int, int] | None:
    coords = [(int(x), int(y)) for x, y in _COORD_RE.findall(question)]
    if len(coords) < 2:
        return None
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _jpeg_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _scaled_box(
    current_size: tuple[int, int],
    original_size: tuple[int, int],
    box: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    cur_w, cur_h = current_size
    orig_w, orig_h = original_size
    if cur_w <= 0 or cur_h <= 0 or orig_w <= 0 or orig_h <= 0:
        return None

    sx = cur_w / orig_w
    sy = cur_h / orig_h
    x1, y1, x2, y2 = box
    left = max(0, min(cur_w - 1, int(round(x1 * sx))))
    top = max(0, min(cur_h - 1, int(round(y1 * sy))))
    right = max(left + 1, min(cur_w, int(round(x2 * sx))))
    bottom = max(top + 1, min(cur_h, int(round(y2 * sy))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _expand_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    margin_x = max(8, int(round(width * margin_ratio)))
    margin_y = max(8, int(round(height * margin_ratio)))
    img_w, img_h = image_size
    return (
        max(0, left - margin_x),
        max(0, top - margin_y),
        min(img_w, right + margin_x),
        min(img_h, bottom + margin_y),
    )


def _with_coordinate_crops(goal: str, agent_input: AgentInput) -> AgentInput:
    question = _question_text(goal)
    if not _is_box_task(question):
        return agent_input

    box = _extract_coordinate_box(question)
    if box is None or not agent_input.images:
        return agent_input

    derived_images: list[tuple[str, bytes]] = []
    derived_views = list(agent_input.derived_views)
    original_sizes = dict(agent_input.original_sizes)

    for name, jpeg in agent_input.images[:MAX_COORDINATE_CROPS]:
        original_size = agent_input.original_sizes.get(name)
        if original_size is None:
            continue

        try:
            with Image.open(BytesIO(jpeg)) as img:
                img = img.convert("RGB")
                scaled = _scaled_box(img.size, original_size, box)
                if scaled is None:
                    continue

                crop = img.crop(scaled)
                crop_name = f"{name}#bbox_crop"
                crop_bytes = _jpeg_bytes(crop)
                derived_images.append((crop_name, crop_bytes))
                original_sizes[crop_name] = crop.size
                derived_views.append(
                    f"{crop_name}: exact crop of the coordinate rectangle from {name}."
                )

                context_box = _expand_box(scaled, img.size, COORDINATE_CROP_MARGIN_RATIO)
                context = img.crop(context_box)
                context_name = f"{name}#bbox_context"
                context_bytes = _jpeg_bytes(context)
                derived_images.append((context_name, context_bytes))
                original_sizes[context_name] = context.size
                derived_views.append(
                    f"{context_name}: expanded context crop around the coordinate rectangle from {name}."
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to derive coordinate crops for %s: %s", name, e)

    if not derived_images:
        return agent_input

    return AgentInput(
        text_blocks=list(agent_input.text_blocks),
        images=derived_images + list(agent_input.images),
        file_summary=list(agent_input.file_summary),
        original_sizes=original_sizes,
        derived_views=derived_views,
    )


def _normalize_issue_report_json(text: str, question: str) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(data, dict):
        return text

    raw_items = data.get("items")
    if raw_items is None:
        raw_items = data.get("details")
    if raw_items is None:
        for key in ("violations", "incidents", "reports"):
            value = data.get(key)
            if isinstance(value, list):
                raw_items = value
                break
    if raw_items is None:
        return text
    if not isinstance(raw_items, list):
        return text

    start_id = _extract_starting_id(question) or 1100
    category_constant = _field_constant(question, "Category")
    default_category = category_constant or ("incident" if "incident" in question.lower() and "violation" not in question.lower() else "violation")
    short_description_constant = _distance_short_description(question) or _field_constant(question, "Short description")
    filename_field = _filename_field_name(question)
    include_category = True
    include_description = _wants_field(question, "Description")
    include_short_description = True

    rendered_details: list[dict[str, object]] = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue

        filename = str(item.get("filename") or item.get("Filename") or item.get("image_filename") or item.get("Image filename") or "").strip()
        if not filename:
            continue

        short_description = str(
            short_description_constant
            or item.get("short_description")
            or item.get("Short description")
            or ""
        ).strip()
        description = str(item.get("description") or item.get("Description") or "").strip()
        distance_value = item.get("distance_meters")
        category = str(item.get("category") or item.get("Category") or default_category).strip() or default_category

        if _mentions_missing_context(question):
            desc_lower = description.lower()
            if any(token in desc_lower for token in ("missing step", "timestamp", "previous task", "previous tasks", "video clip")):
                continue

        detail: dict[str, object] = {"ID": str(start_id + len(rendered_details))}

        if include_category:
            detail[_title_field_name(question, "Category")] = category_constant or category
        if include_short_description:
            detail[_title_field_name(question, "Short description")] = short_description

        detail[filename_field] = filename

        if include_description:
            if distance_value is not None:
                try:
                    distance_num = float(distance_value)
                    description = f"{distance_num:g}meters"
                except (TypeError, ValueError):
                    pass
            if not description:
                description = _fallback_description(question, filename, short_description)
            else:
                m = re.search(r"(\d+(?:\.\d+)?)\s*meters?", description.lower())
                if m and ("distance" in question.lower() or "less than" in question.lower()):
                    description = f"{m.group(1)}meters"
            detail[_title_field_name(question, "Description")] = description

        rendered_details.append(detail)

    normalized = {
        "total_violations": len(rendered_details),
        "details": rendered_details,
    }
    return json.dumps(normalized, ensure_ascii=False)


def _render_issue_report_from_items(items: list[dict[str, object]], question: str) -> str:
    payload = json.dumps({"items": items}, ensure_ascii=False)
    return _normalize_issue_report_json(payload, question)


def _legacy_normalize_issue_report_json(text: str, question: str) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(data, dict):
        return text

    details = data.get("details")
    if details is None:
        for value in data.values():
            if isinstance(value, list):
                details = value
                break
    if not isinstance(details, list):
        return text

    start_id = _extract_starting_id(question) or 1100
    category = "incident" if "incident" in question and "violation" not in question else "violation"
    short_description_hint = _distance_short_description(question)

    normalized_details: list[dict] = []
    for idx, item in enumerate(details):
        if not isinstance(item, dict):
            continue

        normalized: dict[str, object] = {}
        for key, value in item.items():
            canonical = _canonical_issue_key(str(key))
            if canonical:
                normalized[canonical] = value

        normalized["ID"] = str(normalized.get("ID", start_id + idx))
        normalized["Category"] = str(normalized.get("Category", category))

        short_description = str(normalized.get("Short description", "")).strip()
        if (not short_description or short_description.lower() == "distance violation") and short_description_hint:
            normalized["Short description"] = short_description_hint

        description = normalized.get("Description")
        if short_description_hint and description is not None:
            m = re.search(r"(\d+(?:\.\d+)?)\s*meters?", str(description).lower())
            if m:
                normalized["Description"] = f"{m.group(1)}meters"

        filename = normalized.get("Filename")
        if filename is not None:
            normalized["Filename"] = str(filename)

        normalized_details.append(normalized)

    normalized = {
        "total_violations": len(normalized_details),
        "details": normalized_details,
    }
    return json.dumps(normalized, ensure_ascii=False)


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

    def _system_prompt(self, goal: str, output_format: str, numeric: bool) -> str:
        sys = SYSTEM_BASE
        question = _question_text(goal)
        if output_format == "json":
            sys += SYSTEM_JSON_SUFFIX
        if numeric:
            sys += SYSTEM_NUMERIC_SUFFIX
        if _is_box_task(question):
            sys += SYSTEM_BOX_SUFFIX
        if _is_facing_task(question):
            sys += SYSTEM_FACING_SUFFIX
        if _is_long_sleeve_task(question):
            sys += SYSTEM_LONG_SLEEVE_SUFFIX
        if _is_issue_report_json_task(question, output_format):
            sys += SYSTEM_JSON_ISSUE_SUFFIX
        if _mentions_missing_context(question):
            sys += SYSTEM_EVIDENCE_ONLY_SUFFIX
        return sys

    def _build_user_text(self, goal: str, agent_input: AgentInput) -> str:
        text = goal
        if agent_input.text_blocks:
            text += "\n\n# Attached document content\n" + "\n\n".join(agent_input.text_blocks)
        if agent_input.derived_views:
            text += "\n\n# Derived views\n" + "\n".join(f"- {note}" for note in agent_input.derived_views)
        if agent_input.file_summary:
            text += "\n\n# Attachment summary\n" + "\n".join(agent_input.file_summary)
        return text

    def _select_effort(self, output_format: str, numeric: bool) -> str:
        if output_format == "json":
            return REASONING_EFFORT_JSON
        if numeric:
            return REASONING_EFFORT_NUMERIC
        return REASONING_EFFORT

    def _respond(
        self,
        *,
        system: str,
        user_text: str,
        agent_input: AgentInput,
        output_format: str,
        effort: str,
    ) -> str:
        return self.provider.respond(
            system=system,
            text=user_text,
            images=agent_input.images,
            output_format=output_format,
            effort=effort,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

    def _verify_answer(
        self,
        *,
        goal: str,
        draft_answer: str,
        agent_input: AgentInput,
        question: str,
        numeric: bool,
    ) -> str:
        system = SYSTEM_VERIFY
        if numeric:
            system += SYSTEM_NUMERIC_SUFFIX
        if _is_box_task(question):
            system += SYSTEM_BOX_SUFFIX
        if _is_facing_task(question):
            system += SYSTEM_FACING_SUFFIX
        if _is_long_sleeve_task(question):
            system += SYSTEM_LONG_SLEEVE_SUFFIX

        user_text = self._build_user_text(goal, agent_input)
        user_text += f"\n\n# Draft answer\n{draft_answer or '(empty)'}"

        logger.info("FWA verify pass: model=%s effort=%s", MODEL, REASONING_EFFORT_VERIFY)
        return self._respond(
            system=system,
            user_text=user_text,
            agent_input=agent_input,
            output_format="text",
            effort=REASONING_EFFORT_VERIFY,
        )

    def _bbox_subgoal(self, filename: str, box: tuple[int, int, int, int], question: str) -> str:
        x1, y1, x2, y2 = box
        if _is_facing_task(question):
            subquestion = (
                f"Is there a worker facing left within the area defined by coordinates "
                f"({x1},{y1}), ({x2},{y1}), ({x1},{y2}), and ({x2},{y2}) in this image?"
            )
        else:
            subquestion = (
                f"Is a worker located within the area defined by coordinates "
                f"({x1},{y1}), ({x2},{y1}), ({x1},{y2}), and ({x2},{y2}) in this image?"
            )
        return f"# Question\n{subquestion}\n\n# Input Data\n{filename}\n\n# Output Format\ntext\n"

    def _confirm_bbox_issue(self, subgoal: str, agent_input: AgentInput, draft_answer: str) -> bool:
        if not _is_no_answer(draft_answer):
            return False
        if not ENABLE_BBOX_JSON_CONFIRM:
            return True
        prepared_input = _with_coordinate_crops(subgoal, agent_input)
        verified = self._verify_answer(
            goal=subgoal,
            draft_answer=draft_answer,
            agent_input=prepared_input,
            question=_question_text(subgoal),
            numeric=False,
        )
        return _is_no_answer(verified)

    def _solve_bbox_issue_report_json(self, goal: str, question_raw: str, agent_input: AgentInput) -> str | None:
        input_files = _extract_input_files(goal)
        image_files = [name for name in input_files if _is_image_file(name)]
        if not image_files:
            return None

        explicit_box = _extract_coordinate_box(question_raw)
        coordinate_map = _parse_coordinate_map(agent_input)
        short_description = _field_constant(question_raw, "Short description") or ""
        category = _field_constant(question_raw, "Category") or (
            "incident" if "incident" in question_raw.lower() and "violation" not in question_raw.lower() else "violation"
        )

        items: list[dict[str, object]] = []
        for filename in image_files:
            box = explicit_box or coordinate_map.get(filename)
            if box is None:
                continue

            image_input = _single_image_input(agent_input, filename)
            if image_input is None:
                continue

            subgoal = self._bbox_subgoal(filename, box, question_raw.lower())
            subresult = self.answer(subgoal, image_input)
            answer = (subresult.answer or "").strip()
            if not self._confirm_bbox_issue(subgoal, image_input, answer):
                continue

            items.append(
                {
                    "filename": filename,
                    "category": category,
                    "short_description": short_description,
                    "description": "",
                }
            )

        return _render_issue_report_from_items(items, question_raw)

    def answer(self, goal: str, agent_input: AgentInput) -> AgentResult:
        output_format = _detect_output_format(goal)
        numeric = _detect_numeric_intent(goal)
        question = _question_text(goal)
        question_raw = _question_text_raw(goal)
        if _is_bbox_issue_report_json_task(question, output_format):
            rendered = self._solve_bbox_issue_report_json(goal, question_raw, agent_input)
            if rendered is not None:
                return AgentResult(
                    answer=rendered,
                    output_format=output_format,
                    n_images=len(agent_input.images),
                    n_text_blocks=len(agent_input.text_blocks),
                )

        agent_input = _with_coordinate_crops(goal, agent_input)
        system = self._system_prompt(goal, output_format, numeric)
        effort = self._select_effort(output_format, numeric)

        user_text = self._build_user_text(goal, agent_input)

        logger.info(
            "FWA call: model=%s effort=%s format=%s images=%d text_blocks=%d",
            MODEL, effort, output_format, len(agent_input.images), len(agent_input.text_blocks),
        )

        raw = self._respond(
            system=system,
            user_text=user_text,
            agent_input=agent_input,
            output_format=output_format,
            effort=effort,
        )

        if output_format == "json":
            cleaned = _strip_to_json(raw)
            if _is_issue_report_json_task(question, output_format):
                cleaned = _normalize_issue_report_json(cleaned, question_raw)
            # validate JSON; if it fails, fall back to raw
            try:
                json.loads(cleaned)
                raw = cleaned
            except json.JSONDecodeError:
                logger.warning("Model JSON failed to parse; sending raw text.")
        elif _should_verify(question, output_format, numeric, agent_input):
            try:
                verified = self._verify_answer(
                    goal=goal,
                    draft_answer=raw,
                    agent_input=agent_input,
                    question=question,
                    numeric=numeric,
                ).strip()
                if verified:
                    raw = verified
            except Exception as e:  # noqa: BLE001
                logger.warning("Verification pass failed; keeping draft answer: %s", e)

        return AgentResult(
            answer=raw,
            output_format=output_format,
            n_images=len(agent_input.images),
            n_text_blocks=len(agent_input.text_blocks),
        )
