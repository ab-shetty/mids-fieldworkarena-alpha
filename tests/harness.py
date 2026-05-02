"""Local sample-evaluation harness.

Loads a small subset of FieldWorkArena factory tasks from the green-agent
benchmark, downloads their attached files from the Fujitsu HuggingFace
dataset, runs the FWAAgent directly (skipping the A2A round-trip for speed),
and grades using the green agent's own evaluators.

Usage:
    cd /root/agentbeats/purple
    python -m tests.harness --n 6
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

# Make `src.*` importable from anywhere
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# Add the green agent so we can re-use its data source + evaluators
GREEN = ROOT.parent / "green"
sys.path.insert(0, str(GREEN / "src"))

from a2a.types import FilePart, FileWithBytes, Part, TextPart  # noqa: E402

from src.agent import FWAAgent  # noqa: E402
from src.multimodal import parts_to_input  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("harness")


def _load_factory_task_ids() -> list[str]:
    import tomllib
    with open(GREEN / "benchmark" / "all_task_ids.toml", "rb") as f:
        data = tomllib.load(f)
    return [".".join(t.split(".")[-3:]) for t in data["factory"]]


def _load_all_tasks() -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    for f in sorted((GREEN / "benchmark" / "tasks" / "group2").glob("Tasks_*.json")):
        try:
            data = json.loads(f.read_text())
            for t in data:
                tid = t.get("id")
                if tid:
                    tasks[tid] = t
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping %s: %s", f, e)
    return tasks


def _build_goal(task: dict) -> str:
    """Mirror green/src/fieldworkarena/agent/metrics/tasks/task_loader.py:build_goal."""
    query = ""
    answer = ""
    for c in task.get("conversations", []):
        if c["from"] == "human":
            query = c["value"]
        elif c["from"] == "gpt":
            answer = c["value"]
    input_data = task["input_data"]
    output_format = task["output_format"]
    goal = "# Question\n" + query + "\n\n# Input Data\n"
    files = input_data if isinstance(input_data, list) else input_data.split()
    for fn in files:
        goal += fn.strip() + "\n"
    goal += f"\n# Output Format\n{output_format}\n"
    return goal, query, answer, files


def _make_a2a_parts(goal: str, file_payloads) -> list[Part]:
    parts: list[Part] = [Part(root=TextPart(kind="text", text=goal))]
    for fp in file_payloads:
        parts.append(Part(root=FilePart(kind="file", file=fp)))
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6, help="Number of tasks to sample")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--all", action="store_true", help="Run all factory tasks (ignores --n/--seed)")
    parser.add_argument("--ids", nargs="*", help="Specific task IDs to run instead of sampling")
    parser.add_argument("--out", default="harness_results.json")
    parser.add_argument("--skip", nargs="*", default=[], help="Task IDs to skip")
    parser.add_argument(
        "--bucket",
        choices=["all", "fuzzy", "numerical", "json"],
        default="all",
        help="Restrict sample to one eval_func bucket",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    if not os.environ.get("HUGGINGFACE_TOKEN") and not os.environ.get("HF_TOKEN"):
        sys.exit("HUGGINGFACE_TOKEN / HF_TOKEN not set")

    # The green data source reads HF_TOKEN; alias if only HUGGINGFACE_TOKEN is present
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    # Imports that need HF_TOKEN to be set
    from fieldworkarena.agent.metrics import automatic as auto_eval_pkg  # noqa: F401
    from fieldworkarena.agent.metrics.automatic import automatic_evaluation as auto_eval
    from fieldworkarena.agent.metrics.tasks.data_source import BenchmarkDataSource

    # Override judge model — the green code uses gpt-4-1106-preview, which is
    # retired. Locally we use gpt-4o-mini as the substitute judge. The actual
    # leaderboard run uses RDI's container with their own judge model, so this
    # local substitution only affects our offline scoring.
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    _orig = auto_eval.generate_from_openai_chat_completion
    def _patched(messages, model, **kw):  # noqa: ANN001
        return _orig(messages=messages, model=judge_model, **kw)
    auto_eval.generate_from_openai_chat_completion = _patched
    logger.info("Judge model override: %s", judge_model)

    factory_ids = set(_load_factory_task_ids())
    all_tasks = _load_all_tasks()
    factory_tasks = [t for tid, t in all_tasks.items() if tid in factory_ids]
    if args.bucket != "all":
        wanted = {"fuzzy": "fuzzy_match", "numerical": "numerical_match", "json": "json_match"}[args.bucket]
        factory_tasks = [t for t in factory_tasks if t.get("eval_func") == wanted]

    if args.all:
        sample = sorted(factory_tasks, key=lambda t: t["id"])
    elif args.ids:
        # When IDs are explicit, allow any task from any category, not just factory.
        ids = set(args.ids)
        sample = [t for tid, t in all_tasks.items() if tid in ids]
    else:
        random.seed(args.seed)
        sample = random.sample(factory_tasks, k=min(args.n, len(factory_tasks)))
    sample = [t for t in sample if t["id"] not in set(args.skip)]
    logger.info("Running %d tasks: %s", len(sample), [t["id"] for t in sample])

    ds = BenchmarkDataSource(access_token=os.environ["HF_TOKEN"])
    ds.validate_access()
    agent = FWAAgent()

    results = []
    t_start = time.time()
    for i, task in enumerate(sample, 1):
        tid = task["id"]
        eval_func = task["eval_func"]
        goal, query, gold, file_names = _build_goal(task)
        logger.info("[%d/%d] task=%s eval=%s files=%s", i, len(sample), tid, eval_func, file_names)
        try:
            file_payloads = ds.load_file_payload(task["input_data"])
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to load files for %s: %s", tid, e)
            results.append({"id": tid, "score": 0.0, "error": str(e), "eval_func": eval_func})
            continue

        parts = _make_a2a_parts(goal, file_payloads)
        try:
            agent_input = parts_to_input(parts)
            goal_blocks = [t for t in agent_input.text_blocks if not t.startswith("=== ")]
            file_blocks = [t for t in agent_input.text_blocks if t.startswith("=== ")]
            goal_only = "\n\n".join(goal_blocks)
            agent_input.text_blocks = file_blocks

            t0 = time.time()
            res = agent.answer(goal_only, agent_input)
            dt = time.time() - t0
            pred = res.answer
        except Exception as e:  # noqa: BLE001
            logger.exception("Agent error on %s: %s", tid, e)
            results.append({"id": tid, "score": 0.0, "error": str(e), "eval_func": eval_func})
            continue

        # Grade exactly as green agent does
        try:
            if eval_func == "fuzzy_match":
                score, _ = auto_eval.llm_fuzzy_match(pred, gold, query)
            elif eval_func == "exact_match":
                score, _ = auto_eval.exact_match(gold, pred)
            elif eval_func == "must_include":
                score, _ = auto_eval.must_include(gold, pred)
            elif eval_func == "must_exclude":
                score, _ = auto_eval.must_exclude(gold, pred)
            elif eval_func == "json_match":
                score, _ = auto_eval.json_match(pred, gold, query)
            elif eval_func == "numerical_match":
                score, _ = auto_eval.numerical_match(pred, gold, query)
            else:
                score = 0.0
        except Exception as e:  # noqa: BLE001
            logger.exception("Eval failure on %s: %s", tid, e)
            score = 0.0

        # numerical_match returns float, possibly with non-tuple
        score = float(score) if not isinstance(score, tuple) else float(score[0])

        logger.info("  → score=%.3f (%.1fs) pred=%r", score, dt, pred[:240].replace("\n", "\\n"))
        results.append({
            "id": tid,
            "eval_func": eval_func,
            "score": score,
            "pred": pred,
            "gold": gold,
            "query": query,
            "n_images": res.n_images,
            "n_text_blocks": res.n_text_blocks,
            "agent_seconds": round(dt, 2),
        })

    total = sum(r["score"] for r in results)
    rate = total / len(results) if results else 0.0
    elapsed = time.time() - t_start
    summary = {
        "n_tasks": len(results),
        "total_score": total,
        "score_rate": rate,
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    logger.info("=" * 60)
    logger.info("DONE: %d/%d  score_rate=%.4f  elapsed=%.1fs  → %s",
                int(total), len(results), rate, elapsed, args.out)


if __name__ == "__main__":
    main()
