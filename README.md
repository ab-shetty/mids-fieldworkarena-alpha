# FieldWorkArena Agent — MIDS Alpha

A multimodal reasoning agent for the [FieldWorkArena AgentBeats benchmark](https://agentbeats.dev/agentbeater/fieldworkarena). Given factory-floor images, PDFs, and video clips, the agent answers safety-compliance questions: Is the worker wearing a hard hat? How many violations are visible? Is the cart within 1 meter of the worker?

---

## The Problem

FieldWorkArena tasks arrive as A2A messages carrying a natural-language question, attached files, and an output-format hint (`text` or `json`). A judge agent grades the response — using fuzzy match, numerical match, or structured JSON comparison — against a gold answer derived from human annotation.

The factory category has 79 tasks across three grading types:

| Grading | Count | What it tests |
|---|---|---|
| `fuzzy_match` | 43 | PPE compliance, yes/no, identification |
| `numerical_match` | 24 | Counts, distances, timestamps |
| `json_match` | 12 | Structured incident/violation reports |

Input files skew heavily visual: 98 images, 14 PDFs, 2 text files. Getting the right answer requires reading images carefully — and producing answers in exactly the format the grader expects.

---

## What We Built

A single-call reasoning agent that:

1. **Understands the task type** from the question before sending anything to the model.
2. **Composes a targeted system prompt** — the base rules plus only the guidance relevant to that task type.
3. **Enhances the visual evidence** by cropping bounding-box regions out of images so the model focuses on exactly the right area.
4. **Calls an OpenAI reasoning model** (gpt-5-mini or gpt-5) via the Responses API with appropriate effort.
5. **Post-processes the output** to match the benchmark's exact JSON schemas, including field names, ID sequences, and distance formats.

There is no tool use, no retrieval, no multi-step chain — one well-prepared call is enough for almost all tasks.

---

## Agent Architecture

```
A2A message (question + files)
          │
          ▼
  ┌───────────────┐
  │  Multimodal   │  Images → JPEG (≤1568px)
  │  Converter    │  PDFs  → extracted text
  └──────┬────────┘  Video → sampled frames
         │
         ▼
  ┌───────────────┐
  │  Task         │  Detect: json / numeric / bbox /
  │  Classifier   │  facing / long-sleeve / issue-report
  └──────┬────────┘
         │
         ▼
  ┌───────────────┐
  │  System       │  Base rules + only the relevant
  │  Prompt       │  task-specific guidance blocks
  │  Composer     │
  └──────┬────────┘
         │
         ▼
  ┌───────────────┐
  │  Image        │  For bbox tasks: crop the exact
  │  Cropper      │  rectangle + an expanded context view
  └──────┬────────┘
         │
         ▼
  ┌───────────────┐
  │  Reasoning    │  OpenAI Responses API
  │  Model        │  effort: medium / high based on task
  └──────┬────────┘
         │
         ▼
  ┌───────────────┐
  │  JSON         │  Normalize field names, ID sequences,
  │  Normalizer   │  distance formats to match grader schema
  └───────────────┘
```

---

## Key Design Decisions

### Dynamic system prompt composition

A single long prompt is wasteful and noisy. Instead, a base prompt covering universal rules (YES/NO format, count format, distance estimation anchors, no hedging) is extended with narrow guidance blocks that activate only when needed:

- **Numeric suffix** — reminds the model to commit to one value with correct units, and provides scale references (worker height ≈ 1.7 m, pallet ≈ 1.0 m wide) for distance estimation.
- **Bounding-box suffix** — instructs the model to evaluate only the specified pixel rectangle, ignore workers outside it, and use cropped views as primary evidence.
- **Facing-direction suffix** — addresses a common failure where the model reports the direction a *different* worker is walking rather than the posture of the worker inside the box.
- **JSON issue-report suffix** — gives the model a simplified intermediate schema to emit, which the post-processor then reshapes into the benchmark's exact schema. This avoids asking the model to track complex field names and ID sequences directly.
- **Evidence-only suffix** — activated when the question references "previous tasks" or "video timestamps" that aren't actually attached; instructs the model not to invent missing-step violations from absent evidence.

### Coordinate-aware image cropping

Bounding-box tasks specify a pixel rectangle like `(120,45), (380,210)`. Rather than relying on the model to mentally zoom in on the correct region of a full-resolution image, the agent:

1. Rescales the original coordinates to the downscaled JPEG dimensions.
2. Crops the exact rectangle and an 18%-expanded context window.
3. Prepends both crops to the image list with descriptive labels.

The model sees the zoomed evidence first and the full image second — a meaningful accuracy improvement on tasks where the worker of interest is small in the frame.

### Per-image bbox JSON solver

JSON incident-report tasks that reference bounding boxes are broken into per-image sub-problems:

1. For each image, formulate: *"Is a worker located within the area defined by (x1,y1)…(x2,y2)?"*
2. Run that single-image sub-call.
3. Confirm "No" answers with a second verification pass to catch false negatives.
4. Assemble confirmed positives into the final JSON report.

This avoids asking the model to simultaneously track multiple images, multiple coordinate rectangles, and a strict output schema in one call.

### JSON schema normalization

The benchmark's grader uses exact field-name and format matching. Rather than over-specify the schema in the prompt (which produces inconsistent results), the agent asks the model for a simple intermediate format and then applies deterministic post-processing:

- Renames fields to match benchmark case (e.g. `"short_description"` → `"Short description"`).
- Assigns sequential IDs starting from the value the question specifies (e.g. `starting from 1100`).
- Extracts numeric distance values from prose descriptions and formats them as `"1.5meters"` — the exact string the `numerical_match` grader expects inside JSON.
- Filters out invented violations that reference absent evidence (missing video clips, unattached previous-task findings).

---

## Results

Local evaluation on a random factory sample (seed=42, n=6), graded with the benchmark's own evaluators:

| Model | Score rate | Notes |
|---|---|---|
| `gpt-5-mini` | **0.45** | ~5 s/task |
| `gpt-5` | **0.50** | ~30 s/task |

The 24 numerical tasks are the hardest: distance estimation from still images carries inherent noise, and every model we tested misses some ±0.5 m clearance judgements.

---

## Project Structure

```
src/
  server.py      — A2A HTTP server (Starlette + uvicorn)
  executor.py    — A2A task executor; bridges protocol to agent
  multimodal.py  — File converter: images, PDFs, video frames
  agent.py       — Core reasoning logic (classifiers, prompt, crops, normalization)
  providers.py   — OpenAI Responses API client
tests/
  harness.py     — Local benchmark runner against the HuggingFace dataset
amber-manifest.json5  — Deployment manifest for AgentBeats submission
```
