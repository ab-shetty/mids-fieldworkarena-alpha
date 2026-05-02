# Purple Agent — FieldWorkArena (gpt-5-mini, multimodal)

Multimodal A2A purple agent for the [FieldWorkArena AgentBeats benchmark](https://agentbeats.dev/agentbeater/fieldworkarena).
Targets the **factory** category (79 tasks: 43 fuzzy_match / 24 numerical_match / 12 json_match — 98 image inputs, 14 PDFs, 2 text).

## How it works

1. **A2A server** (`src/server.py`) on port 9019 — same protocol the green agent expects.
2. **Multimodal converter** (`src/multimodal.py`) turns each `FilePart`
   into an OpenAI Responses-API content block:
   - JPG/PNG → `input_image` (downscaled to 1568px max edge, re-encoded JPEG)
   - PDF → `input_text` extracted via pypdf
   - TXT → `input_text` decoded
   - MP4 → up to 24 evenly-spaced JPEG frames via opencv (factory subset has no videos, but warehouse/retail do)
3. **Reasoner** (`src/agent.py`) — single Responses-API call to gpt-5-mini with:
   - A system prompt that enforces no hedging, no preamble, format-faithful answers
   - JSON-mode (`text.format=json_object`) when the goal hints `# Output Format\njson`
   - Reasoning effort: `medium` for free-text, `high` for json/numerical
4. **A2A executor** (`src/executor.py`) emits `TaskState.completed` with the answer in a single `TextPart`.

## Local sample evaluation

```bash
export OPENAI_API_KEY=…
export HUGGINGFACE_TOKEN=…   # gates access to Fujitsu/FieldWorkArena_Dataset
python -m tests.harness --n 6
```

The harness reuses the green agent's `automatic_evaluation.py` so scores match the leaderboard's grading exactly.

## Submission

Build & push the image:

```bash
docker build -t ghcr.io/<you>/purple-agent-fwa:v0.1 .
docker push ghcr.io/<you>/purple-agent-fwa:v0.1
```

Then point `amber-manifest.json5` at the immutable digest and submit per the AgentBeats workflow.

## Tunable env vars (also exposed via `amber-manifest.json5` config_schema)

| var | default | notes |
|---|---|---|
| `OPENAI_API_KEY` | — | **required**, marked `secret` in the manifest |
| `OPENAI_MODEL` | `gpt-5-mini` | submitter-selectable: `gpt-5-mini`, `gpt-5`, etc. |
| `REASONING_EFFORT` | `medium` | text answers |
| `REASONING_EFFORT_JSON` | `high` | json output |
| `REASONING_EFFORT_NUMERIC` | `high` | how-many / how-long / how-much |
| `MAX_IMAGE_EDGE` | `1568` | image-downscale ceiling, pixels |
| `MAX_VIDEO_FRAMES` | `24` | only used for warehouse/retail (factory has no videos) |
| `MAX_OUTPUT_TOKENS` | `4096` | per call |

### Local benchmark snapshot (factory, mixed seed samples)

| model | sample | score_rate | notes |
|---|---|---|---|
| `gpt-5-mini` | seed=42, n=6 | 0.45 | ~5s/task; ~7 min for full 79 |
| `gpt-5`      | seed=42, n=6 | 0.50 | ~30s/task; ~30–40 min for full 79 |
| `gemini-2.5-flash` (free tier) | seed=42, n=6 | 0.00 | self-contradictory wording, hallucinated JSON violations |
| `gemini-2.5-flash` (free tier) | seed=11, n=6 | 0.17 | rate-limited at 5 req/min |

Numerical-distance tasks (24/79) are the structural weak point — every model we tested misses some `0.5 m`/`1.5 m` distance estimates by ±20%. `gemini-2.5-pro` (paid) is plausibly the leaderboard's choice but we couldn't test it on the free key. Submitter should pick model+effort by accuracy/cost.

### Provider selection

The submitter picks a backend at run time via the manifest config:

```json5
{ llm_model: "gpt-5",            openai_api_key: "sk-…" }   // OpenAI
{ llm_model: "gemini-2.5-pro",   gemini_api_key: "AIza…" }   // Gemini
{ llm_model: "gpt-5-mini",       openai_api_key: "sk-…" }   // default
```

Provider is auto-detected from the model name prefix (`gemini-*` → Gemini, else OpenAI). Override via `llm_provider` if needed.
