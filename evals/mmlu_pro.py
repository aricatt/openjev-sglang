# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28,<1", "pyarrow>=20,<24"]
# ///
"""Seeded, zero-shot MMLU-Pro evaluation through Jev's choice endpoint."""

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
INSTRUCTIONS = "Choose the correct answer to the multiple-choice question in the state."


def payload(row):
    # Deliberately select input fields; never send answer, answer_index or cot_content.
    return {
        "model": MODEL,
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
        "reported_cost_usd": sum(r.get("usage", {}).get("cost", 0) for r in rows),
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
        "# Jev on a random MMLU-Pro subset",
        "",
        f"**{accuracy:.2%} accuracy ({correct:,}/{n:,})**; approximate 95% Wilson interval "
        f"{center - radius:.2%}–{center + radius:.2%}.",
        "",
        f"Sampled {n:,} test questions uniformly without replacement, seed {manifest['seed']}. "
        "This is a subset estimate, not a full-test score. The interval treats questions as "
        "independent and does not apply a finite-population correction.",
        "",
        "Zero-shot direct answers through OpenRouter's Jev `choice` endpoint. "
        "State contains only the question; criteria map original answer letters to option text. "
        "No gold answers or provided rationales enter the request. The API's selected choice "
        "is scored by exact match. No retries based on correctness.",
        "",
        "This differs from the benchmark's usual five-shot chain-of-thought setup; "
        "do not treat it as a reproduction of published leaderboard scores. "
        "Public benchmark training overlap is unknown.",
        "",
        f"Resolved model: {', '.join(result['response_model_ids'])}. "
        f"Successful-response cost: ${result['reported_cost_usd']:.4f}. "
        f"Saved responses with in-run retries: {result['requests_with_retries']} "
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
        "model": MODEL,
        "instructions": INSTRUCTIONS,
        "state": "{question}",
        "criteria": "Original letters A… mapped to options in order",
        "shots": 0,
        "endpoint": "https://openrouter.ai/api/alpha/decisions",
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
    headers = {
        "Authorization": "Bearer " + Path("~/.openrouter-api-key").expanduser().read_text().strip(),
        "X-OpenRouter-Title": "OpenJev MMLU-Pro evaluation",
    }
    started = time.monotonic()
    async with httpx.AsyncClient(
        base_url="https://openrouter.ai", headers=headers, timeout=150
    ) as client:
        with path.open("a", buffering=1) as stream:

            async def worker():
                while not queue.empty():
                    i = queue.get_nowait()
                    row = dataset[i]
                    body = payload(row)
                    for attempt in range(9):
                        try:
                            response = await client.post("/api/alpha/decisions", json=body)
                            response.raise_for_status()
                            data = response.json()
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
                        "model": data.get("model", MODEL),
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
    parser.add_argument("--output", type=Path, default=Path("evals/runs/mmlu-pro-jev"))
    parser.add_argument("--report", type=Path, default=Path("evals/runs/mmlu-pro-jev-report"))
    asyncio.run(main(parser.parse_args()))
