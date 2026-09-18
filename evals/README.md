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
- [MMLU-Pro comparison after prompt cleanup](results/mmlu-pro-2026-09-18/comparison/report.md):
  subject accuracy, paired answer outcomes, and the before/after prompt result.

## MMLU-Pro subset

```sh
uv run evals/mmlu_pro.py --count 1000 --seed 42 --concurrency 16
uv run evals/mmlu_pro.py --provider openjev --count 1000 --seed 42 --concurrency 16

# Preserve the previous run when testing a new deployment.
uv run evals/mmlu_pro.py --provider openjev --output evals/runs/mmlu-pro-openjev-simple-options
uv run evals/compare_mmlu_pro.py
```

This standalone script samples test questions uniformly without replacement and
uses Jev's or OpenJev's `choice` endpoint with zero-shot direct answers. Each
provider gets its own output directory; OpenJev waits for server readiness and
never reads or sends the OpenRouter key. It reports exact-match
accuracy, an approximate Wilson interval, and subject breakdowns. It is not a
reproduction of the usual five-shot chain-of-thought benchmark protocol. The
manifest records the pinned dataset revision, byte hash and all sampled indices.
Predictions resume on rerun; incomplete runs cannot produce a final report.
The comparison script verifies matching questions and payload settings, then
produces a subject dot plot, overall Wilson intervals, a paired outcome grid,
and paired bootstrap intervals for score differences. Its default inputs are the
saved Jev run, the original OpenJev run, and the new prompt run shown above.

## One-token prompt research

[Research findings and results](results/mmlu-pro-research-2026-09-18/report.md)
include the five prompt variants, a KV-cache audit, and a proposed bounded-reasoning
experiment.

`prompt_probe.py` compares the deployed prompt, a plain MCQ, a JSON answer prefix,
five validation examples with direct answers, and five validation examples with
rationales. Every test answer still uses exactly one generated token with thinking
disabled. This diagnostic calls the internal SGLang endpoint through an existing
Modal container; it does not change the deployment or expose a new endpoint.

```sh
# First cache test data using mmlu_pro.py. Also download the validation split:
curl -L --fail -o evals/data/mmlu-pro-validation.parquet \
  https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/b189ec765aa7ed75c8acfea42df31fdae71f97be/data/validation-00000-of-00001.parquet
uv run modal container list
uv run evals/prompt_probe.py --container ta-... --output evals/runs/prompt-research
```

The default 256 questions exclude the earlier 1,000-question sample. This is an
exploratory comparison, not a new headline benchmark: any selected improvement
needs confirmation on untouched questions. Test gold answers remain local; only
the official validation examples supply demonstration answers. The script records
all five variants and paired bootstrap intervals, and fails if any result is missing.
Use a new output directory for each run. Raw output is streamed into `remote.log`;
this bounded diagnostic does not resume or retry failed requests.
