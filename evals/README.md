# BoolQ calibration evaluation

Run all 3,270 labeled development examples against OpenJev and optionally real Jev.
These are standalone uv scripts: their dependencies stay separate from the API's
environment. No server redeployment is needed to run an evaluation.

```sh
# OpenJev: 16 concurrent requests, with a short pause per worker.
uv run evals/boolq.py --concurrency 16

# Real Jev: reads ~/.openrouter-api-key locally; sends it only to OpenRouter.
uv run evals/boolq.py --provider jev --concurrency 16 --output evals/runs/boolq-jev

# Paired report, PNG and PDF plots, and machine-readable metrics.
uv run evals/report_boolq.py --jev evals/runs/boolq-jev

# OpenJev report without a comparison endpoint.
uv run evals/report_boolq.py
```

Collection writes each successful prediction immediately to JSONL and resumes
completed rows on rerun. It refuses to reuse a directory with a different
protocol. Transient network and capacity errors receive bounded retries; a
permanent failure stops the run. The report requires all rows, so failed requests
cannot silently disappear from the denominator. `--limit 5 --output
evals/runs/pilot` runs a small pilot. `--concurrency 1` reduces load further.
The default concurrency is 2; up to 64 is configurable. Concurrency can change
when resuming a run; `execution.jsonl` records each segment's settings and starting
row count, while the manifest preserves the original run settings.

The source is [Google's BoolQ](https://github.com/google-research-datasets/boolean-questions),
mirrored at `google/boolq` on Hugging Face, pinned to revision
`35b264d03638db9f4ce671b711558bf7ff0f80d5`. The original Google download currently
requires access; the public parquet mirror supplies passage, question and answer.
Its `validation` split corresponds to BoolQ's labeled development set; official
test answers are hidden. The dataset license is CC BY-SA 3.0. Downloaded data and
raw run outputs are ignored by git.

The passage is `state`; the question instruction is `Based on the passage, answer
this yes/no question:\n{question}`. Both providers receive explicit Yes/No
criteria. The gold answer is never sent to the API. We score `noul` as P(yes).
No calibration parameters are fitted, and the entropy-based `confidence` field
is not used. The providers can use different internal prompts and inference
procedures despite receiving equivalent payloads.

Metrics include accuracy, binary Brier score, log loss, two forms of ten-bin ECE,
mean overconfidence, and errors above 90%, 95% and 99% selected-answer probability.
The report includes 2,000 paired bootstrap samples clustered by passage, reliability
plots with approximate binwise Wilson intervals, and bin counts. Brier/log loss
measure predictive skill as well as calibration. ECE depends on binning and has
finite-sample bias. Results describe passage-grounded yes/no classification;
public benchmark overlap with model training is unknown.

Every run records the dataset byte hash, source revision, exact prompt, provider,
requested model ID, response model IDs, usage, retries and request IDs. API key
contents are never written to the results.

## Saved results

- [BoolQ: OpenJev versus Jev](results/boolq-2026-09-18/comparison/report.md), with
  [PNG](results/boolq-2026-09-18/comparison/reliability.png) and
  [PDF](results/boolq-2026-09-18/comparison/reliability.pdf) calibration curves.
- [Jev: 1,000 random MMLU-Pro questions](results/mmlu-pro-2026-09-18/report.md).
- [OpenJev: the same MMLU-Pro questions](results/mmlu-pro-2026-09-18/openjev/report.md).

## MMLU-Pro subset

```sh
uv run evals/mmlu_pro.py --count 1000 --seed 42 --concurrency 16
uv run evals/mmlu_pro.py --provider openjev --count 1000 --seed 42 --concurrency 16
```

This standalone script samples test questions uniformly without replacement and
uses Jev's or OpenJev's `choice` endpoint with zero-shot direct answers. Each
provider gets its own output directory; OpenJev waits for server readiness and
never reads or sends the OpenRouter key. It reports exact-match
accuracy, an approximate Wilson interval, and subject breakdowns. It is not a
reproduction of the usual five-shot chain-of-thought benchmark protocol. The
manifest records the pinned dataset revision, byte hash and all sampled indices.
Predictions resume on rerun; incomplete runs cannot produce a final report.
