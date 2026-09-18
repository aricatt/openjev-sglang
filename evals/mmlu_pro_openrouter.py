# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28,<1", "pyarrow>=20,<24", "numpy>=2,<3", "matplotlib>=3.10,<4"]
# ///
"""One-token hosted MMLU-Pro accuracy, paired with the saved Jev/OpenJev sample."""

import argparse
import asyncio
import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pyarrow.parquet as pq
from compare_mmlu_pro import load, paired, wilson

INSTRUCTION = (
    "Choose the correct answer to the multiple-choice question. "
    "Answer with only the uppercase option letter, without explanation."
)
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def messages(row):
    # Select only question/options: gold labels and dataset rationales stay local.
    choices = "\n".join(f"{chr(65 + i)}: {s}" for i, s in enumerate(row["options"]))
    return [{"role": "user", "content": f"{INSTRUCTION}\n\n{row['question']}\n\n{choices}"}]


def score_response(data, labels):
    usage = data["usage"]
    if usage["completion_tokens"] != 1:
        raise ValueError("Provider did not generate exactly one token")
    if usage.get("completion_tokens_details", {}).get("reasoning_tokens") != 0:
        raise ValueError("Provider did not explicitly report zero reasoning tokens")
    choice = data["choices"][0]
    if choice["message"].get("reasoning") or choice["message"].get("reasoning_details"):
        raise ValueError("Unexpected generated reasoning")
    content = (choice.get("logprobs") or {}).get("content") or []
    if len(content) != 1 or not content[0].get("top_logprobs"):
        raise ValueError("Expected exactly one position with top-token logprobs")
    top = content[0]["top_logprobs"]
    if any(not math.isfinite(t["logprob"]) for t in top):
        raise ValueError("Nonfinite logprob")
    # Exact single-token A-J labels, without stripping whitespace or merging variants.
    found = {t["token"]: t["logprob"] for t in top if t["token"] in labels}
    # A returned valid label beats every omitted label, because this is a top-k list.
    # No valid label means unresolved, counted wrong rather than silently excluded.
    prediction = max(found, key=found.get) if found else None
    return prediction, found


def report(args, manifest, rows):
    reference, baseline = load(args.reference)
    jev_manifest, jev = load(args.jev)
    indices = sorted(manifest["indices"])
    rows = sorted(rows, key=lambda r: r["index"])
    if [r["index"] for r in rows] != indices:
        raise ValueError("Incomplete/duplicate predictions")
    for other_manifest in [reference, jev_manifest]:
        if other_manifest["sha256"] != manifest["sha256"]:
            raise ValueError("Dataset mismatch")
    baseline = [r for r in baseline if r["index"] in indices]
    jev = [r for r in jev if r["index"] in indices]

    def identity(data):
        return [(r["index"], r["question_id"], r["answer"], r["category"]) for r in data]

    if identity(rows) != identity(baseline) or identity(rows) != identity(jev):
        raise ValueError("Mismatched question identities or gold labels")
    runs = {"OpenJev · Qwen3.6-35B-A3B": baseline, "Qwen3.8-27B · Parasail": rows, "Jev": jev}
    correct = {
        name: np.array([r["prediction"] == r["answer"] for r in rr]) for name, rr in runs.items()
    }
    overall = {
        name: {
            "correct": int(v.sum()),
            "count": len(v),
            "accuracy": float(v.mean()),
            "wilson_95": wilson(int(v.sum()), len(v)),
        }
        for name, v in correct.items()
    }
    ours = correct["Qwen3.8-27B · Parasail"]
    subjects = {
        c: {
            "count": sum(r["category"] == c for r in rows),
            **{
                name: float(v[[r["category"] == c for r in rows]].mean())
                for name, v in correct.items()
            },
        }
        for c in sorted({r["category"] for r in rows})
    }
    metrics = {
        "overall": overall,
        "subjects": subjects,
        "qwen38_minus_openjev": paired(correct["OpenJev · Qwen3.6-35B-A3B"], ours),
        "qwen38_minus_jev": paired(correct["Jev"], ours),
        "unresolved": sum(r["prediction"] is None for r in rows),
        "all_labels_returned": sum(r["all_labels_returned"] for r in rows),
        "response_models": sorted({r["model"] for r in rows}),
        "response_providers": sorted({r["provider"] for r in rows}),
        "completion_tokens": sum(r["usage"]["completion_tokens"] for r in rows),
        "reasoning_tokens": sum(
            r["usage"]["completion_tokens_details"]["reasoning_tokens"] for r in rows
        ),
        "reported_cost_usd": sum(r["usage"].get("cost", 0) for r in rows),
        "requests_with_retries": sum(r["attempts"] > 1 for r in rows),
        "request_latency_median_s": float(np.median([r["latency_s"] for r in rows])),
        "manifest": manifest,
    }
    args.report.mkdir(parents=True, exist_ok=True)
    (args.report / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    lines = [
        "# Qwen3.8-27B: one-token MMLU-Pro",
        "",
        "| Model/service | Correct | Accuracy | 95% Wilson CI |",
        "|---|---:|---:|---:|",
    ]
    for name, m in overall.items():
        lo, hi = m["wilson_95"]
        lines.append(
            f"| {name} | {m['correct']}/{m['count']} | {m['accuracy']:.1%} | {lo:.1%}–{hi:.1%} |"
        )
    lines += [
        "",
        f"Same {len(rows)} test questions from the earlier seed-42 sample. "
        "Qwen3.8 uses zero-shot plain multiple-choice chat through OpenRouter, pinned to Parasail, "
        "thinking disabled, max_tokens=1, temperature=1, top_p=1, no logit bias. "
        "Prediction is the highest-logprob exact uppercase answer label, not the sampled token.",
        "",
        "The services share questions and option order, but their prompt wrappers and backends "
        "differ. OpenJev uses its state/question wrapper and an answer prefix; this hosted "
        "run uses the plain prompt saved in the manifest. Jev's internal computation is unknown. "
        "This is a service comparison, not an isolated model ablation or a reproduction of "
        "five-shot chain-of-thought leaderboard scores.",
        "",
        f"Unresolved (no valid label in top-20): {metrics['unresolved']}; these count wrong. "
        f"All answer labels returned: {metrics['all_labels_returned']}/{len(rows)}. "
        "If any valid label is returned, its argmax over valid labels is determined even when "
        "other labels are omitted. Missing probabilities are not treated as zero; no calibration "
        "metrics are computed from truncated distributions.",
        "",
        f"Reported output: {metrics['completion_tokens']} tokens, "
        f"{metrics['reasoning_tokens']} reasoning tokens. "
        f"Successful-response cost: ${metrics['reported_cost_usd']:.4f}. "
        f"Requests with retries: {metrics['requests_with_retries']}.",
        "",
    ]
    for key in ["qwen38_minus_openjev", "qwen38_minus_jev"]:
        m = metrics[key]
        lo, hi = m["difference_95"]
        lines.append(
            f"{key}: {m['right_minus_left'] * 100:+.1f} percentage points "
            f"(paired bootstrap 95% interval {lo * 100:+.1f} to {hi * 100:+.1f}; "
            "10,000 draws, seed 42, resampled question pairs)."
        )
    lines += [
        "",
        "![Comparison](comparison.png)",
        "",
        "## Subject breakdown",
        "",
        "| Subject | Questions | OpenJev | Qwen3.8 | Jev |",
        "|---|---:|---:|---:|---:|",
    ]
    for c, m in subjects.items():
        values = " | ".join(f"{m[name]:.1%}" for name in runs)
        lines.append(f"| {c} | {m['count']} | {values} |")
    (args.report / "report.md").write_text("\n".join(lines) + "\n")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [1.8, 1]})
    colors = ["#2671bd", "#289675", "#cb583e"]
    cats = sorted(subjects, key=lambda c: subjects[c]["Qwen3.8-27B · Parasail"])
    for name, color in zip(runs, colors, strict=True):
        ax.scatter(
            [subjects[c][name] * 100 for c in cats], range(len(cats)), label=name, color=color, s=40
        )
    ax.set(
        yticks=range(len(cats)),
        yticklabels=[f"{c.replace('_', ' ').title()} ({subjects[c]['count']})" for c in cats],
        xlim=(0, 103),
        xlabel="Accuracy (%)",
        title="Accuracy by subject (question count)",
    )
    for i, m in enumerate(overall.values()):
        value = m["accuracy"] * 100
        lo, hi = np.array(m["wilson_95"]) * 100
        bx.errorbar(
            value, i, xerr=[[value - lo], [hi - value]], fmt="o", color=colors[i], capsize=4
        )
        bx.text(value, i + 0.18, f"{value:.1f}%", ha="center", color=colors[i], fontweight="bold")
    bx.set(
        yticks=range(3),
        yticklabels=["OpenJev", "Qwen3.8-27B", "Jev"],
        ylim=(-0.5, 2.6),
        xlim=(40, 95),
        xlabel="Accuracy (%) · 95% Wilson intervals",
        title="Overall",
    )
    for a in (ax, bx):
        a.grid(axis="x", alpha=0.2)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"MMLU-Pro · same {len(rows):,} questions", fontsize=19)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        ncols=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.04),
    )
    fig.text(
        0.5,
        0.025,
        "Zero-shot service comparison; prompt wrappers differ. "
        "Qwen runs generate one answer token; Jev internals unknown.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    fig.savefig(args.report / "comparison.png", dpi=160)
    fig.savefig(args.report / "comparison.pdf")
    plt.close(fig)
    print(
        json.dumps(
            {k: v for k, v in metrics.items() if k not in {"subjects", "manifest"}}, indent=2
        ),
        flush=True,
    )


async def main(args):
    reference = json.loads((args.reference / "manifest.json").read_text())
    data_path = Path(__file__).parent / "data/mmlu-pro-test.parquet"
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != reference["sha256"]:
        raise ValueError("Dataset hash differs from baseline")
    dataset = pq.read_table(data_path).to_pylist()
    settings = {
        "model": "qwen/qwen3.8-27b",
        "max_tokens": 1,
        "temperature": 1,
        "top_p": 1,
        "logprobs": True,
        "top_logprobs": 20,
        "reasoning": {"enabled": False},
        "provider": {"only": ["parasail"], "allow_fallbacks": False, "require_parameters": True},
    }
    if not 1 <= args.count <= len(reference["indices"]):
        raise ValueError("Count outside reference sample")
    manifest = {
        k: reference[k] for k in ["dataset", "split", "revision", "sha256", "seed", "population"]
    }
    manifest.update(
        indices=reference["indices"][: args.count],
        count=args.count,
        settings=settings,
        endpoint=ENDPOINT,
        instruction=INSTRUCTION,
        prompt="{instruction}\\n\\n{question}\\n\\n{A: option...}",
        shots=0,
        concurrency=args.concurrency,
        scoring="argmax exact uppercase labels in top-20; no label counts wrong",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(k) != v for k, v in manifest.items()):
            raise ValueError("Run parameters changed; use a new directory")
        manifest = previous
    else:
        manifest["started_at"] = datetime.now(UTC).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    path = args.output / "predictions.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done = {r["index"] for r in rows}
    if len(done) != len(rows) or not done.issubset(manifest["indices"]):
        raise ValueError("Invalid saved predictions")
    queue = asyncio.Queue()
    for i in manifest["indices"]:
        if i not in done:
            queue.put_nowait(i)
    key = Path("~/.openrouter-api-key").expanduser().read_text().strip()
    headers = {"Authorization": "Bearer " + key, "X-OpenRouter-Title": "OpenJev one-token MMLU-Pro"}
    started = time.monotonic()
    # No redirects: credentials are sent only to the fixed OpenRouter endpoint.
    async with httpx.AsyncClient(headers=headers, timeout=120) as client:
        with path.open("a", buffering=1) as stream:

            async def worker():
                while not queue.empty():
                    i = queue.get_nowait()
                    row = dataset[i]
                    body = {**settings, "messages": messages(row)}
                    for attempt in range(6):
                        call_started = time.monotonic()
                        try:
                            response = await client.post(ENDPOINT, json=body)
                            response.raise_for_status()
                            data = response.json()
                            if "error" in data:
                                raise ValueError(f"Provider error: {data['error']}")
                            break
                        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                            if isinstance(
                                exc, httpx.HTTPStatusError
                            ) and exc.response.status_code not in {
                                429,
                                500,
                                502,
                                503,
                                504,
                                520,
                                529,
                            }:
                                raise
                            if attempt == 5:
                                raise
                            print(f"Transient retry index={i}, attempt={attempt + 1}", flush=True)
                            await asyncio.sleep(min(30, 2 ** (attempt + 1)))
                    labels = list("ABCDEFGHIJ"[: len(row["options"])])
                    if data["model"] != settings["model"] or data["provider"] != "Parasail":
                        raise ValueError("Unexpected model/provider routing")
                    prediction, found = score_response(data, labels)
                    record = {
                        "index": i,
                        "question_id": row["question_id"],
                        "category": row["category"],
                        "answer": row["answer"],
                        "prediction": prediction,
                        "label_logprobs": found,
                        "all_labels_returned": len(found) == len(labels),
                        "model": data["model"],
                        "provider": data["provider"],
                        "usage": data["usage"],
                        "attempts": attempt + 1,
                        "latency_s": time.monotonic() - call_started,
                        "response": data,
                    }
                    stream.write(json.dumps(record) + "\n")
                    rows.append(record)
                    if len(rows) % 100 == 0 or len(rows) == args.count:
                        print(
                            f"Completed {len(rows)}/{args.count} "
                            f"in {time.monotonic() - started:.1f}s",
                            flush=True,
                        )

            async with asyncio.TaskGroup() as group:
                for _ in range(args.concurrency):
                    group.create_task(worker())
    report(args, manifest, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=16, choices=range(1, 65))
    parser.add_argument(
        "--reference", type=Path, default=Path("evals/runs/mmlu-pro-openjev-simple-options")
    )
    parser.add_argument("--jev", type=Path, default=Path("evals/runs/mmlu-pro-jev"))
    parser.add_argument("--output", type=Path, default=Path("evals/runs/qwen38-27b"))
    parser.add_argument("--report", type=Path, default=Path("evals/results/qwen38-27b"))
    asyncio.run(main(parser.parse_args()))
