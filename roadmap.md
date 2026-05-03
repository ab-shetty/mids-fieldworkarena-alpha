# Roadmap — FieldWorkArena Purple Agent

This is the working handoff doc for the purple agent in
`/root/agentbeats/purple`. Read this before changing code.

## Goal

Build a competitive purple agent for AgentBeats Phase 2 on the
**FieldWorkArena** benchmark, with primary focus on the **factory**
category. The practical near-term goal is steady local improvement on a
fixed representative slice and then a real leaderboard baseline.

## Non-negotiable guardrails

These came directly from the user and must be preserved:

1. **Never inspect competitors' code.**
2. **Do not cheat.** No task-ID branching, no hardcoded answers, no prompt
   stuffing with benchmark-specific gold outputs.
3. Use the user's existing environment credentials:
   - `OPENAI_API_KEY`
   - `HUGGINGFACE_TOKEN` / `HF_TOKEN`
4. Use `gpt-5-mini` for local testing unless the user explicitly asks for a
   model comparison.
5. Prefer a representative local slice that takes roughly 10-15 minutes,
   not full-benchmark runs for every iteration.
6. Keep GitHub runner constraints in mind:
   - CPU-only friendly
   - lightweight dependencies
   - no large local vision stacks unless clearly justified

## Benchmark reality

The user originally mentioned MLE-Bench, but the linked benchmark, green
agent, leaderboard, and dataset are all for **FieldWorkArena**. Treat
FieldWorkArena as the benchmark of record.

Relevant repos and assets:

- Green agent repo:
  `https://github.com/RDI-Foundation/FieldWorkArena-agentbeats`
- Agentbeats benchmark page:
  `https://agentbeats.dev/agentbeater/fieldworkarena`
- Leaderboard repo:
  `https://github.com/RDI-Foundation/FieldWorkArena-agentbeats-leaderboard`
- Dataset:
  `Fujitsu/FieldWorkArena_Dataset` on Hugging Face

Factory remains the focus because that is where the public competition is
concentrated and where the current agent work has been evaluated.

## Current agent architecture

### Runtime pieces

- [src/server.py](/root/agentbeats/purple/src/server.py)
  A2A HTTP server.
- [src/executor.py](/root/agentbeats/purple/src/executor.py)
  Bridges the incoming A2A task to the local agent.
- [src/multimodal.py](/root/agentbeats/purple/src/multimodal.py)
  Converts A2A file parts into image/text inputs.
- [src/providers.py](/root/agentbeats/purple/src/providers.py)
  OpenAI Responses API wrapper.
- [src/agent.py](/root/agentbeats/purple/src/agent.py)
  Main task logic, prompt selection, crop derivation, JSON normalization,
  bbox-specific routing.
- [tests/harness.py](/root/agentbeats/purple/tests/harness.py)
  Local benchmark harness against green's evaluators.

### What the agent now does

1. Parses the green agent's `# Question / # Input Data / # Output Format`
   goal string.
2. Converts image, PDF, and text attachments into a single multimodal input.
3. Uses prompt suffixes specialized for:
   - numeric tasks
   - box / coordinate tasks
   - facing-direction tasks
   - long-sleeve tasks
   - evidence-only constraints
   - JSON issue-report tasks
4. For coordinate-based image tasks, derives:
   - exact bbox crop
   - padded context crop
5. For JSON issue-report tasks, normalizes model output into a deterministic
   benchmark-style schema.
6. For bbox-based JSON incident tasks, bypasses generic JSON generation and
   instead:
   - solves each image individually with the crop-aware text path
   - confirms candidate incidents with one extra verification pass
   - renders final JSON deterministically

## Important current behaviors

### 1. Crop tool is live and on by default

Coordinate-based tasks now get derived image views in
[src/agent.py](/root/agentbeats/purple/src/agent.py:559) using Pillow only.
This is lightweight and GitHub-runner safe.

### 2. Deterministic JSON rendering is live and on by default

Issue-report tasks are no longer expected to emit the final benchmark schema
directly from the model. The model first emits a normalized `{"items":[...]}`
shape, and code renders the final `{"total_violations": ..., "details": ...}`
payload deterministically.

### 3. Bbox-specific JSON path is live and on by default

Tasks like bbox-presence / bbox-facing incident reports now route through
`_solve_bbox_issue_report_json(...)` instead of one-shot JSON generation.

### 4. Global second-pass verifier exists but is OFF by default

`ENABLE_SECOND_PASS_VERIFY=1` enables a broader self-check on text answers,
but measured results were negative. Leave it off unless explicitly testing it.

### 5. Narrow bbox incident confirmation is ON by default

`ENABLE_BBOX_JSON_CONFIRM=1` is the current default. It only triggers when a
bbox JSON route is about to emit an incident for a specific image. This was a
measured improvement, not a speculative one.

## Current environment knobs

Main env vars in use:

- `OPENAI_MODEL` default: `gpt-5-mini`
- `REASONING_EFFORT`
- `REASONING_EFFORT_JSON`
- `REASONING_EFFORT_NUMERIC`
- `REASONING_EFFORT_VERIFY`
- `MAX_OUTPUT_TOKENS`
- `ENABLE_SECOND_PASS_VERIFY` default `0`
- `ENABLE_BBOX_JSON_CONFIRM` default `1`
- `MAX_COORDINATE_CROPS`
- `COORDINATE_CROP_MARGIN_RATIO`

## Fixed evaluation slice

The user explicitly asked to stop seed-shopping. That is now implemented.

[tests/canonical.txt](/root/agentbeats/purple/tests/canonical.txt) contains
the current fixed 18-task slice:

```text
1.1.0024
1.1.0028
2.3.0011
2.3.0014
2.3.0019
2.3.0020
2.3.0051
2.3.0054
2.3.0071
2.3.0083
2.3.0021
2.3.0026
2.3.0041
2.3.0043
3.3.0003
3.4.0003
4.2.0001
4.2.0026
```

This slice mixes PDF extraction, PPE / spatial yes-no, numerical distance,
multi-image counting, and JSON reports. It is the default local iteration set.

Run it with:

```bash
env OPENAI_MODEL=gpt-5-mini .venv/bin/python -m tests.harness --canonical --out canonical.json
```

Use this slice for honest deltas. Do not go back to random-seed comparisons
unless the user explicitly asks for broader sampling.

## Measured progress so far

All scores below are local harness scores, not leaderboard submissions.

### Canonical progression

- `baseline_fixed18.json`: `9/18` (`0.500`)
- `rerun_fixed18.json`: `10/18` (`0.556`)
- `canonical_with_crop.json`: `11/18` (`0.611`)

Interpretation:

- Prompt and JSON cleanup moved the slice from `9/18` to `10/18`.
- The coordinate crop tool produced the next real gain to `11/18`.

### Second-pass verifier results

- `noverify_spatial8.json`: `4/8`
- `verify_spatial8.json`: `3/8`

Interpretation:

- The broad text-answer verifier currently hurts score.
- Keep `ENABLE_SECOND_PASS_VERIFY=0` by default.

### Deterministic JSON work

Focused JSON checks established that output rendering is no longer the main
problem for several tasks:

- `4.2.0001` was fixed by deterministic empty-output handling.
- `4.2.0026` now renders the right schema shape but still misses because the
  underlying distance estimate is wrong.
- `3.4.0003` still fails because the model overcalls incidents.

### Bbox JSON route results

The bbox-specific route is now the strongest recent improvement.

- `bbox_json_path_v3.json`: `4/4`
  - `3.4.0005` pass
  - `3.4.0007` pass
  - `4.2.0028` pass
  - `4.2.0030` pass

This route now combines:

1. per-image bbox subquestions
2. crop-backed evidence
3. deterministic final JSON rendering
4. narrow confirmation of candidate incidents

### Mixed JSON suite

- `json_suite_v3.json`: `5/7` (`0.714`)

Breakdown:

- Pass:
  - `3.4.0005`
  - `3.4.0007`
  - `4.2.0001`
  - `4.2.0028`
  - `4.2.0030`
- Fail:
  - `3.4.0003`
  - `4.2.0026`

Interpretation:

- Bbox-report JSON tasks are now substantially better.
- The remaining misses are still mostly **distance perception**, not JSON
  formatting.

## What is currently working well

1. **PDF / text extraction**
   The document path is stable and transfers across categories.
2. **Coordinate-region reasoning**
   The crop tool materially improved bbox and facing tasks.
3. **Bbox-based JSON incident reports**
   The specialized route is now a measured win.
4. **OpenAI JSON robustness**
   `src/providers.py` now retries lower-effort JSON calls when OpenAI exhausts
   `max_output_tokens` before visible output.

## What is still weak

### 1. Distance estimation

This is still the main score bottleneck. Known problematic tasks include:

- `2.3.0021`
- `2.3.0026`
- `2.3.0043`
- `3.4.0003`
- `4.2.0026`

Current pattern:

- The model often picks the wrong worker-object pair or underestimates
  spatial distance.
- JSON rendering now preserves these wrong facts cleanly, which makes the
  remaining problem easier to isolate.

### 2. Long-sleeve detection

`2.3.0019` is still a stubborn visual miss.

### 3. Broad self-verification

The generic second-pass verifier is not yet selective enough to help overall.

## Code changes that matter most

### [src/providers.py](/root/agentbeats/purple/src/providers.py)

- Better extraction of visible text from Responses API output objects.
- Retry path for incomplete JSON responses when `max_output_tokens` is hit.

### [src/multimodal.py](/root/agentbeats/purple/src/multimodal.py)

- `AgentInput` now carries:
  - `original_sizes`
  - `derived_views`
- These are required for coordinate-aware crop derivation.

### [src/agent.py](/root/agentbeats/purple/src/agent.py)

Major additions:

- task-type prompt suffixes for box, facing, long-sleeve, evidence-only, and
  JSON issue-report tasks
- coordinate crop derivation
- deterministic issue-report JSON rendering
- bbox-coordinate map parsing from `Image_Bounding_Box_Coordinates.txt`
- bbox-specific JSON routing
- narrow bbox incident confirmation pass

### [tests/harness.py](/root/agentbeats/purple/tests/harness.py)

- `--canonical` support for fixed-slice evaluation

### [tests/canonical.txt](/root/agentbeats/purple/tests/canonical.txt)

- fixed representative task list for iteration

## Immediate next steps

Priority is based on expected score lift per unit complexity under the
user's constraints.

### 1. Build a distance-specific helper

This is the next highest-value item.

Target tasks:

- `3.4.0003`
- `4.2.0026`
- distance-heavy `2.3.*` tasks in the canonical slice

Direction:

- keep it lightweight and CPU-friendly
- avoid heavy local models unless clearly worth it
- likely use simple geometry / object-selection scaffolding rather than a
  generic second-pass self-check

Good shape:

1. isolate the relevant object pair explicitly
2. ask for the minimum distance only between those two objects
3. optionally run a narrow verification pass on the selected pair

### 2. Make distance verification selective, not global

The current broad verifier is too blunt. If verification comes back, it
should probably be restricted to:

- numeric distance tasks
- candidate bbox incidents
- maybe long-sleeve edge cases

### 3. Re-run the canonical 18-task slice after the next distance change

Do not trust narrow wins alone. The next real milestone should be a
canonical improvement beyond `11/18`.

### 4. Consider a real submission once the next distance lift lands

There is still no leaderboard submission as of 2026-05-02. A baseline
submission becomes more worthwhile once the local agent is no longer losing
easy bbox JSON tasks.

## Explicitly deprioritized for now

1. **Competitor inspection**
   Forbidden by user instruction.
2. **Task-answer hardcoding**
   Forbidden by user instruction.
3. **Heavy local vision stacks**
   Not a first move because the final benchmark runs on GitHub runners.
4. **Generic web-search tools**
   Not the bottleneck.
5. **Broader provider expansion**
   Not the highest-leverage next step while distance errors are still local
   and specific.

## Submission status

No submission has been made yet.

When ready:

1. push the repo to a public GitHub repository
2. build and publish a `linux/amd64` image to GHCR
3. make the GHCR package public
4. update `amber-manifest.json5` with the pinned digest
5. submit via Agentbeats Quick Submit against FieldWorkArena factory

## How to test

### Syntax check

```bash
python3.13 -m compileall src tests
```

### Canonical slice

```bash
env OPENAI_MODEL=gpt-5-mini .venv/bin/python -m tests.harness --canonical --out canonical.json
```

### Bbox JSON regression slice

```bash
env OPENAI_MODEL=gpt-5-mini .venv/bin/python -m tests.harness \
  --ids 3.4.0005 3.4.0007 4.2.0028 4.2.0030 \
  --out bbox_json_path_v3.json
```

### Mixed JSON suite

```bash
env OPENAI_MODEL=gpt-5-mini .venv/bin/python -m tests.harness \
  --ids 3.4.0003 3.4.0005 3.4.0007 4.2.0001 4.2.0026 4.2.0028 4.2.0030 \
  --out json_suite_v3.json
```

### Optional toggles

Broad verifier off by default:

```bash
env OPENAI_MODEL=gpt-5-mini ENABLE_SECOND_PASS_VERIFY=1 \
  .venv/bin/python -m tests.harness --canonical --out verify_run.json
```

Bbox incident confirmation on by default, but can be disabled for A/B tests:

```bash
env OPENAI_MODEL=gpt-5-mini ENABLE_BBOX_JSON_CONFIRM=0 \
  .venv/bin/python -m tests.harness \
  --ids 3.4.0005 3.4.0007 4.2.0028 4.2.0030 \
  --out bbox_json_no_confirm.json
```

## Handoff summary

If a fresh agent picks this up, the current situation is:

1. The crop tool is real and beneficial.
2. Deterministic JSON rendering is real and beneficial.
3. Bbox JSON routing is now a clear measured win.
4. The next score bottleneck is distance perception, not formatting.
5. Do not inspect competitor code.
6. Do not hardcode answers.
7. Use `gpt-5-mini` for local test loops unless the user redirects.
