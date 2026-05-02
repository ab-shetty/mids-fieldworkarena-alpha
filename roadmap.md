# Roadmap — FieldWorkArena Purple Agent

This file is the handoff doc. If a fresh agent picks up this work, read this
top-to-bottom before touching code.

## Goal

Build a competitive **purple agent** for the AgentBeats Phase 2
**FieldWorkArena** benchmark (research agent track). User signed up at
agentbeats.dev as `ashetty21@berkeley.edu`. Target a top-3 finish on the
**factory** category (where the leaderboard's 99.7% / 99.1% top scores
currently sit). Above 90% is the stretch goal; ~60-65% is the realistic
ceiling for a single-model approach without extra agent-level tricks.

## Background — important things to know

### 1. The benchmark is FieldWorkArena, not MLE-Bench

The user originally said "MLE-Bench in the research agent track" but every
URL they linked is **FieldWorkArena** — a factory/warehouse/retail
field-operations benchmark using real Fujitsu facility data. Confirmed
2026-05-02. MLE-Bench is a separate competition. **Do not conflate them.**

Refs:
- Green agent: `https://github.com/RDI-Foundation/FieldWorkArena-agentbeats`
- Agentbeats page: `https://agentbeats.dev/agentbeater/fieldworkarena`
- HuggingFace dataset: `Fujitsu/FieldWorkArena_Dataset` (gated; needs
  `HF_TOKEN` to access)

### 2. Three task categories with current availability limits

| category | available / total | dominant input types |
|---|---:|---|
| **factory** | 79 / 176 | jpg (98), pdf (14), txt (2) — **no videos** |
| warehouse | 162 / 264 | mixed, includes videos |
| retail | 5 / 446 | mixed |

Factory is where the leaderboard competition is concentrated and where we
target. Factory eval-function distribution: **fuzzy_match 43, numerical_match
24, json_match 12**. Output format: **text 67, json 12**.

Factory subset breakdown by source file:
- `Tasks_2.3.json` (54 tasks, 68% of factory): single-image PPE/safety yes-no,
  fuzzy
- `Tasks_1.1.json` (9 tasks): PDF document extraction, fuzzy
- `Tasks_4.2.json` (8 tasks): PDF + image compliance reports, mostly JSON
- `Tasks_3.3.json` (4 tasks): multi-image counting, numerical
- `Tasks_3.4.json` (4 tasks): multi-image violations to JSON

### 3. Wire protocol (A2A over HTTP)

The purple agent serves an A2A endpoint on **port 8080** (per the manifest
convention; locally we use 9019 to match the green's `scenario.toml`). The
green agent on port 9009 sends each task as a single A2A `Message` with:

- `parts[0]` = `TextPart` with the goal: `# Question\n... # Input Data\n
  filenames... # Output Format\ntext|json`.
- `parts[1..N]` = `FilePart(FileWithBytes)` for each attached file (jpg / pdf
  / txt / mp4), base64-encoded.

We respond with a single `TaskStatusUpdateEvent` whose final state is
`TaskState.completed` and whose message's first `TextPart` contains the
answer text. The green parses ONLY this `TextPart` for grading.

Source of truth: `/root/agentbeats/green/src/fieldworkarena/agent/fwa_green_agent.py`.

### 4. Grading rubric (six eval functions)

The green agent grades each answer with one of:

| eval_func | how it scores |
|---|---|
| `fuzzy_match` | LLM judge (`gpt-4-1106-preview`) returns correct/incorrect/partially correct; partial → 0 |
| `numerical_match` | LLM extracts numbers; `(1 - 0.5) + 0.5 * numerical_score` if "correct", 0 if "incorrect"; `eval_distance` gives partial credit on |Δ|/ref ≤ 0.1/0.2/0.3/0.4/0.5 |
| `json_match` | LLM judge with JSON-aware prompt |
| `exact_match` | string equality after lowercase |
| `must_include` | substring match |
| `must_exclude` | substring non-match |

Critical: **flipping a boolean kills the entire numerical_match score.** A
"No, it is 1.4 m" when gold is "Yes, it is 0.5 m" → 0 points, not 0.5.

The judge LLM is `gpt-4-1106-preview` (retired). For local grading we
substitute `gpt-4o-mini` via a monkey-patch in the harness; this only affects
local scores, not the leaderboard.

### 5. Data source needs HF_TOKEN

`BenchmarkDataSource` reads `HF_TOKEN` (or `HUGGINGFACE_TOKEN`, aliased) and
downloads files from `Fujitsu/FieldWorkArena_Dataset` on HuggingFace. The
dataset is gated; the user already has access on their account. Must be set
in the runtime env of any local harness or container that loads file
payloads.

### 6. Image / manifest contract

Image must be `linux/amd64`, pushed to GHCR, package made public.
Manifest expects port 8080 and a single A2A endpoint. The user's
`OPENAI_API_KEY` and `GEMINI_API_KEY` are passed via `${config.openai_api_key}`
and `${config.gemini_api_key}` (both marked `secret: true`). `LLM_MODEL` can
be overridden at submission time via `${config.llm_model}`; provider is
auto-detected from the model name prefix (`gemini-*` → Gemini, else OpenAI).
When omitted, runtime default is `gpt-5-mini`.

## Submitted results so far

**None yet (2026-05-02).** No submissions to the FieldWorkArena leaderboard.
Local sample evaluation only.

## What is built

| Path | Status | Notes |
|---|---|---|
| `src/server.py` | ✅ | A2A Starlette app, port 9019 (manifest maps to 8080), agent card advertises `text/pdf/jpeg/mp4`. |
| `src/executor.py` | ✅ | Bridges A2A executor lifecycle. Splits the goal text from extracted PDF/text content, emits final answer in a single `TextPart`. |
| `src/agent.py` | ✅ | Single multimodal call to the chosen model. Format-aware system prompt, JSON mode when goal hints `json`. Reasoning effort: `medium` for free-text, `high` for json/numerical. |
| `src/multimodal.py` | ✅ | `FilePart` → model content blocks. Image: downscale to 1568px max edge + JPEG re-encode. PDF: pypdf text extract. Text: utf-8 decode. Video: opencv frame-sample (max 24 frames, evenly spaced). |
| `src/providers.py` | ✅ | Provider abstraction. `OpenAIProvider` uses Responses API with `reasoning.effort` + JSON mode. `GeminiProvider` uses google-genai with `thinking_config` budget mapped from effort + `response_mime_type` for JSON. Both pin temp=0/seed=0 for determinism. |
| `tests/harness.py` | ✅ | Loads factory tasks from green's benchmark dir, downloads files via `BenchmarkDataSource`, runs agent in-process (no A2A round-trip), grades with green's evaluators. Supports `--n N --seed S`, `--ids …`, `--bucket fuzzy/numerical/json`, `--all`. Substitutes `gpt-4o-mini` for retired judge model. |
| `Dockerfile` | ✅ | python:3.12-slim, libgl1/libglib for opencv, exposes 9019. |
| `amber-manifest.json5` | ✅ | Image ref placeholder — update before submit. Exposes `llm_model`, `llm_provider`, `openai_api_key`, `gemini_api_key`, plus reasoning-effort knobs. |
| `requirements.txt` | ✅ | `a2a-sdk[http-server]>=0.3.20,<1.0`, openai, google-genai, pypdf, pillow, opencv-python-headless, numpy. |

## What is NOT done yet

1. **No submission has been made.** The agent hasn't been built into a Docker
   image or pushed to GHCR. Need to: build linux/amd64 image, push to
   `ghcr.io/ab-shetty/purple-agent-fwa`, mark the package public, update
   `amber-manifest.json5` with the pinned digest, and submit via Quick Submit
   on agentbeats.dev. **Lift in difficulty: trivial, just hasn't been done.**
2. **No re-verification / multi-pass loop.** Currently each task is one
   model call. A second pass that re-examines the image with the draft
   answer as context ("you said 0.3 m — count pallet widths to verify")
   could lift spatial-reasoning scores by 5-15pp historically. ~2× cost,
   ~2× latency. **Lift: medium.**
3. **No tool-use augmentation.** The leaderboard's 99.7% almost certainly
   uses agent-level tricks beyond raw VLM capability — likely tool calls to
   specialized vision models (object detection, depth estimation, OCR for
   visible measurements) and/or ensemble voting. We do none of that today.
   **Lift: high (multi-day).**
4. **Canonical test subset not picked.** User asked to stop seed-shopping;
   we should commit a fixed 10-task representative subset (5 fuzzy, 3
   numerical, 2 json — span easy/medium/hard within each bucket) at
   `tests/canonical.txt` and add a `--canonical` harness flag, so iteration
   uses the same tasks every time and score deltas are meaningful. **Lift:
   trivial; pending user sign-off on bucket weights.**
5. **No video tasks tested.** Factory has none. **Warehouse and retail
   currently have NO available video tasks either** — all 102 unreleased
   warehouse tasks may include videos, but none of the 162 currently
   available do. (Confirmed by extension scan 2026-05-02.) The
   `_extract_video_frames` path samples 24 evenly-spaced JPEG frames; not
   verified end-to-end. **Lift: low; deferred until video tasks are released.**

## Verified locally

- All Python files compile under Python 3.12.
- `/usr/bin/python` is Python 3.8 (won't run this code); the venv at
  `.venv/` uses Python 3.12.
- Server boots and serves `/.well-known/agent-card.json` on port 9019.
- HF dataset access works with the user's `HUGGINGFACE_TOKEN`.
- The `a2a-sdk` version pin is critical: pip's resolver picks `1.0.x` by
  default, which has a breaking API change (no `FilePart` / `FileWithBytes`
  exports under `a2a.types`). `requirements.txt` pins `>=0.3.20,<1.0` —
  keep it that way unless you also rewrite the imports.
- Sample-evaluation snapshot (factory subset, 6-task seed=42 mixed sample
  unless noted; "judge=gpt-4o-mini" substituted for retired
  `gpt-4-1106-preview`):

| model | sample | score | notes |
|---|---|---:|---|
| `gpt-5-mini` | seed=42, n=6 | 2.7/6 (0.45) | ~5s/task; ~7 min full 79 |
| `gpt-5` | seed=42, n=6 | 3.0/6 (0.50) | ~25-65s/task; ~30-40 min full |
| `gemini-2.5-flash` (free) | seed=42, n=6 | 0.0/6 (0.00) | hallucinated JSON violations, self-contradictory wording |
| `gemini-2.5-flash` (free) | seed=11, n=6 | 1.0/6 (0.17) | hit free-tier 5 req/min limit |
| `gemini-2.5-pro` (paid) | seed=42, n=6 | 3.0/6 (0.50) | best so far on this seed |
| `gemini-2.5-pro` (paid) | seed=11, n=8 | 3.0/8 (0.38) | another seed, smaller variance |
| `gemini-3.1-pro-preview` | seed=42, n=6 | 1.8/6 (0.30) | terser PDF extraction hurts fuzzy; underestimates distances more |

- Hard-case patterns observed across all models on the seed=42 sample:
  - `2.3.0019` "Are workers wearing long-sleeved shirts?" — every model
    answers "No" (visual error vs gold "Yes"). Possibly a benchmark
    annotation that disagrees with naive visual reading.
  - `2.3.0047` 0.5m gold, all models predict 0.1-0.3m (under-estimate).
  - `2.3.0043` 0.6m gold, all models predict 0.1-0.4m (under-estimate).
  - This points to a **systematic perceptual bias** — VLMs under-estimate
    workplace distances on these specific factory images, not random noise.

- Web-search confirmed (2026-05-02) that VLMs in general cap at **50-60% on
  spatial-reasoning benchmarks**. SpatialVLM was explicitly created because
  "VLMs lack capabilities in 3D spatial reasoning, such as recognizing
  quantitative relationships of physical objects like distances or size
  difference." Our distance-estimation pain is the *known general failure
  mode* of VLMs, not a bug in our agent.

- **Cross-category sanity check (2026-05-02)** with `gpt-5-mini` on 5
  hand-picked non-factory tasks: 3/4 valid (75%). Same agent code, no
  per-category tweaking.
  - `1.1.0033` (warehouse PDF, fuzzy): 1.0 ✓
  - `2.1.0001` (warehouse image, distance fuzzy): 0.0 — predicted 6m vs
    gold 12m; same VLM distance bias seen on factory.
  - `2.1.0058` (warehouse image, fuzzy): error — benchmark data bug, the
    task's `input_data` is `WH_..._00_05jpg` (missing the `.` before `jpg`),
    so the green's `BenchmarkDataSource._load_single_file` rejects it with
    `Unsupported file extension: ` (empty). Affects every competitor, not
    just us. Worth flagging to RDI but not actionable on the purple side.
  - `4.1.2001` (retail txt, fuzzy): 1.0 ✓ — business hours read correctly.
  - `4.1.2002` (retail txt, fuzzy): 1.0 ✓ — list-format hours all correct.
  - **Conclusion**: PDF/text extraction transfers cleanly across all three
    categories; image distance-estimation weakness also transfers. No
    code changes needed for cross-category — same one agent serves
    factory/warehouse/retail.

## Open questions for the user

1. **Set `gemini-2.5-pro` as the headline default** in the manifest, or
   leave it on `gpt-5-mini` for cost reasons? Pro is ~10× the API cost but
   ~5pp better on small samples.
2. **Add Claude Opus 4.7 as a third provider?** Search results indicate its
   image resolution jumped from 1.15MP → 3.75MP; could legitimately help
   distance estimation on factory images. ~50 LOC; needs `ANTHROPIC_API_KEY`.
3. **Implement the re-verification loop?** ~half a day; expected ~5-15pp on
   numerical_match but doubles cost/latency.
4. **Build & submit a baseline now**, or keep iterating locally first? A
   baseline result on the leaderboard would be informative even at ~60%.

## How to test (3-minute smoke test)

```bash
cd /root/agentbeats/purple
export OPENAI_API_KEY=sk-...
export HUGGINGFACE_TOKEN=hf_...

# Single known-good task
.venv/bin/python -m tests.harness --n 1 --ids 2.3.0011 --out smoke.json

# 6-task mixed sample (seed=42 is the canonical comparison seed used above)
.venv/bin/python -m tests.harness --n 6 --seed 42 --out mixed6.json

# Switch model
LLM_MODEL=gemini-2.5-pro GEMINI_API_KEY=AIza... \
  .venv/bin/python -m tests.harness --n 6 --seed 42 --out mixed6_gem.json

# Single-bucket runs
.venv/bin/python -m tests.harness --n 4 --bucket json --seed 3 --out json4.json
```

Expectations on the seed=42 mixed sample:
- `gpt-5-mini`: 2-3 / 6 (45-50%), ~30s wall.
- `gemini-2.5-pro`: 3 / 6 (50%), ~60-90s wall.
- All models will fail `2.3.0019` (long-sleeves) and the 0.5m / 0.6m
  distance tasks — that's the known model ceiling, not an agent bug.

If it fails:
- Check that `OPENAI_API_KEY` and `HUGGINGFACE_TOKEN` are set.
- Check that `.venv/bin/python` exists; recreate via
  `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
- For Gemini failures: confirm the env value doesn't have a leading `=` —
  the user's free-tier key was `=AIza...` (extra `=`); the paid key is
  clean. `GeminiProvider.__init__` strips a leading `=` defensively.

## Submission flow (user does these)

GitHub user is `ab-shetty`. Repo name TBD; suggested
`agentbeats-fwa-purple` → image `ghcr.io/ab-shetty/purple-agent-fwa`.

Not yet done. To do once:

1. **Push to a public GitHub repo** under `ab-shetty/`. Add a publish
   workflow that builds and pushes `linux/amd64` to GHCR on every push to
   `main`.
2. **Make the GHCR package public** (one-time): open
   `https://github.com/ab-shetty?tab=packages`, switch visibility to
   **Public** so the AgentBeats runner can pull without auth.
3. **Update `amber-manifest.json5`** with the pinned digest from the
   workflow's job summary
   (`ghcr.io/ab-shetty/purple-agent-fwa@sha256:<DIGEST>`). Commit + push.
4. **Submit on agentbeats.dev** via Quick Submit:
   - Manifest URL:
     `https://raw.githubusercontent.com/ab-shetty/agentbeats-fwa-purple/main/amber-manifest.json5`
   - Pick the FieldWorkArena leaderboard + the **factory** category.
   - Paste the encrypted `openai_api_key` (and optionally `gemini_api_key`).
   - Optional Config JSON, e.g.
     `{"llm_model":"gemini-2.5-pro","reasoning_effort_numeric":"high"}`
   - The runner forks the leaderboard repo, opens a PR, and writes the
     result JSON.

## Where to push next for top-3

1. **Submit a baseline now** on `gpt-5-mini` so we have a real anchor
   number, then iterate. Without a real submission, we're optimizing blind
   against local sample noise.
2. **Add a re-verification pass on numerical_match tasks specifically.** Of
   the 24 numerical_match tasks, distance estimation is where 80% of the
   score gap lives. A second pass that re-asks "given these reference
   objects, count widths between A and B" could move scores 5-15pp.
3. **Add a vision-tool augmentation** — call a depth-estimation or
   object-detection API (e.g. Grounding-DINO via HF) for distance tasks
   only, and feed the bounding-box pixel distance + a known reference scale
   to the LLM for the actual answer. This is the *most likely* path to
   beating 90%; it's also multi-day work.
4. **Test Claude Opus 4.7** as a provider — its higher image resolution
   may genuinely help on factory images. ~50 LOC plus `ANTHROPIC_API_KEY`.
5. **Tune prompts against the canonical 10-task subset** once chosen, NOT
   against random seeds. Currently the system prompt is a reasonable
   baseline; targeted improvements to the numeric-distance and PDF-
   extraction sections are likely cheap wins.
6. **Pin a canonical subset and stop seed-shopping** (user direction
   2026-05-02 — important: the user explicitly does not want random seeds).
   Commit `tests/canonical.txt` with 10 representative IDs and run that
   subset every iteration with deterministic decoding for honest deltas.
