# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28,<1", "pyarrow>=20,<24"]
# ///
"""Seeded, zero-shot MMLU-Pro evaluation through Jev or OpenJev's choice endpoint."""

import argparse
import asyncio
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pyarrow.parquet as pq

REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
DATA_URL = (
    f"https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/{REVISION}/"
    "data/test-00000-of-00001.parquet"
)
MODEL = "~typesafe/jev-latest"
OPENJEV_MODEL = "Qwen/Qwen3.6-35B-A3B"
OPENJEV_URL = "https://ekzhang--openjev-sglang-openjev.us-west.modal.direct"
INSTRUCTIONS = "Choose the correct answer to the multiple-choice question in the state."


def payload(row, model=MODEL):
    # Deliberately select input fields; never send answer, answer_index or cot_content.
    return {
        "model": model,
        "state": row["question"],
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": {chr(65 + i): text for i, text in enumerate(row["options"])},
            }
        },
    }


def summarize(rows, manifest, output):
    name = "Jev" if manifest["model"] == MODEL else "OpenJev"
    if sorted(r["index"] for r in rows) != sorted(manifest["indices"]):
        raise ValueError("Report requires every sampled question exactly once")
    n = len(rows)
    correct = sum(r["prediction"] == r["answer"] for r in rows)
    accuracy = correct / n
    z = 1.96
    center = (accuracy + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(accuracy * (1 - accuracy) / n + z * z / (4 * n * n))
    radius /= 1 + z * z / n
    groups = defaultdict(list)
    for r in rows:
        groups[r["category"]].append(r["prediction"] == r["answer"])
    result = {
        "manifest": manifest,
        "correct": correct,
        "count": n,
        "accuracy": accuracy,
        "wilson_95": [center - radius, center + radius],
        "response_model_ids": sorted({r["model"] for r in rows}),
        "reported_cost_usd": (
            sum(r["usage"]["cost"] for r in rows)
            if all("cost" in r.get("usage", {}) for r in rows)
            else None
        ),
        "requests_with_retries": sum(r["attempts"] > 1 for r in rows),
        "subjects": {
            name: {
                "correct": sum(values),
                "count": len(values),
                "accuracy": sum(values) / len(values),
            }
            for name, values in sorted(groups.items())
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        f"# {name} on a random MMLU-Pro subset",
        "",
        f"**{accuracy:.2%} accuracy ({correct:,}/{n:,})**; approximate 95% Wilson interval "
        f"{center - radius:.2%}–{center + radius:.2%}.",
        "",
        f"Sampled {n:,} test questions uniformly without replacement, seed {manifest['seed']}. "
        "This is a subset estimate, not a full-test score. The interval treats questions as "
        "independent and does not apply a finite-population correction.",
        "",
        f"Zero-shot direct answers through {name}'s `choice` endpoint. "
        "State contains only the question; criteria map original answer letters to option text. "
        "No gold answers or provided rationales enter the request. The API's selected choice "
        "is scored by exact match. No retries based on correctness.",
        "",
        "This differs from the benchmark's usual five-shot chain-of-thought setup; "
        "do not treat it as a reproduction of published leaderboard scores. "
        "Public benchmark training overlap is unknown.",
        "",
        f"Resolved model: {', '.join(result['response_model_ids'])}. "
        + (
            f"Successful-response cost: ${result['reported_cost_usd']:.4f}. "
            if result["reported_cost_usd"] is not None
            else "API response cost unavailable. "
        )
        + f"Saved responses with in-run retries: {result['requests_with_retries']} "
        "(excludes attempts from interrupted collection segments).",
        "",
        "| Subject | Correct | Questions | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name, g in result["subjects"].items():
        lines.append(f"| {name} | {g['correct']} | {g['count']} | {g['accuracy']:.2%} |")
    lines += [
        "",
        "Source: [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) "
        "([official evaluation](https://github.com/TIGER-AI-Lab/MMLU-Pro)). "
        "The pinned revision, dataset hash, sampled row indices, prompt, and request settings "
        "are recorded in metrics.json. Raw dataset text is not committed.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(lines[2], flush=True)


async def main(args):
    is_jev = args.provider == "jev"
    model = MODEL if is_jev else OPENJEV_MODEL
    endpoint = "https://openrouter.ai" if is_jev else OPENJEV_URL
    route = "/api/alpha/decisions" if is_jev else "/v1/systemone"
    args.output = args.output or Path(f"evals/runs/mmlu-pro-{args.provider}")
    args.report = args.report or Path(f"evals/runs/mmlu-pro-{args.provider}-report")
    data_path = Path(__file__).parent / "data/mmlu-pro-test.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(follow_redirects=True, timeout=120) as downloader:
        if not data_path.exists():
            response = await downloader.get(DATA_URL)
            response.raise_for_status()
            data_path.write_bytes(response.content)
    dataset = pq.read_table(data_path).to_pylist()
    indices = random.Random(args.seed).sample(range(len(dataset)), args.count)
    manifest = {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "revision": REVISION,
        "data_url": DATA_URL,
        "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "population": len(dataset),
        "count": args.count,
        "seed": args.seed,
        "indices": indices,
        "model": model,
        "instructions": INSTRUCTIONS,
        "state": "{question}",
        "criteria": "Original letters A… mapped to options in order",
        "shots": 0,
        "endpoint": endpoint + route,
        "concurrency": args.concurrency,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(k) != v for k, v in manifest.items()):
            raise ValueError("Different run parameters; choose a new output directory")
        manifest = previous
    else:
        manifest["started_at"] = datetime.now(UTC).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    path = args.output / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {r["index"] for r in rows}
    if len(done) != len(rows) or not done.issubset(indices):
        raise ValueError("Invalid saved predictions")
    queue = asyncio.Queue()
    for i in indices:
        if i not in done:
            queue.put_nowait(i)
    headers = {"Modal-Session-ID": "openjev-mmlu-pro-eval"}
    if is_jev:
        headers = {
            "Authorization": "Bearer "
            + Path("~/.openrouter-api-key").expanduser().read_text().strip(),
            "X-OpenRouter-Title": "OpenJev MMLU-Pro evaluation",
        }
    started = time.monotonic()
    async with httpx.AsyncClient(base_url=endpoint, headers=headers, timeout=150) as client:
        deadline = time.monotonic() + 1200
        while not is_jev and not queue.empty():
            try:
                ready = await client.get("/health", timeout=10)
                if ready.is_success:
                    (args.output / "health.json").write_text(ready.text + "\n")
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Backend did not become ready in 20 minutes")
            await asyncio.sleep(5)
        with path.open("a", buffering=1) as stream:

            async def worker():
                while not queue.empty():
                    i = queue.get_nowait()
                    row = dataset[i]
                    body = payload(row, model)
                    for attempt in range(9):
                        try:
                            response = await client.post(route, json=body)
                            response.raise_for_status()
                            data = response.json()
                            if not is_jev:
                                assert data["model"] == model
                                assert data["usage"]["output_tokens"] == 2
                            answer = data["answers"]["answer"]
                            if answer["choice"] not in body["questions"]["answer"]["criteria"]:
                                raise ValueError("Unexpected answer choice")
                            break
                        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                            retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                                exc.response.status_code in {429, 500, 502, 503, 504, 520, 529}
                            )
                            if not retryable or attempt == 8:
                                raise
                            print(f"Retry index={i}, attempt={attempt + 1}: {exc}", flush=True)
                            await asyncio.sleep(min(30, 2 ** (attempt + 1)))
                    record = {
                        "index": i,
                        "question_id": row["question_id"],
                        "category": row["category"],
                        "answer": row["answer"],
                        "prediction": answer["choice"],
                        "probabilities": answer.get("probabilities"),
                        "model": data.get("model", model),
                        "usage": data.get("usage", {}),
                        "attempts": attempt + 1,
                    }
                    rows.append(record)
                    stream.write(json.dumps(record) + "\n")
                    if len(rows) % 100 == 0 or len(rows) == args.count:
                        print(
                            f"Completed {len(rows)}/{args.count} in "
                            f"{time.monotonic() - started:.1f}s",
                            flush=True,
                        )
                    await asyncio.sleep(0.1)

            async with asyncio.TaskGroup() as group:
                for _ in range(args.concurrency):
                    group.create_task(worker())
    summarize(rows, manifest, args.report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=16, choices=range(1, 65))
    parser.add_argument("--provider", choices=["jev", "openjev"], default="jev")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    asyncio.run(main(parser.parse_args()))
