# Direct semantic-audio attention boost: 50 paired questions

This experiment increases actual transformer attention to semantic audio during
decoding. It does not change the confidence-based commitment priority. Evaluate
the unchanged baseline and one frozen ramp on the same 50 new GSM8K Speech
questions. Higher accuracy is the hypothesis, not an assumed result.

## Frozen intervention

Historical `gsm_diagnose20_v13` examples need a mean **87.15 forwards** to reveal
answer content (not the 256-forward fixed canvas). Using the user's requested
half-mean start gives `ceil(87.15 / 2) = 44`; the ramp ends at `ceil(87.15) = 88`.
These are **zero-based forward indices**. Bias is zero through index 44,
first positive at index 45, and stays capped after index 88.

At layers 24–31, add the same bias to semantic-audio key logits for **all answer
query rows**. Prefix query rows, acoustic keys, prompt keys, and unmasking
selection rules are unchanged mathematically. The bias is

```text
b(s) = 0.14742952727445446 * clamp((s - 44) / 44, 0, 1)
```

The historical matched 18-question probe cohort has semantic mass 0.15382076 in
the 25–50% content-progress quarter and 0.11922369 in the last quarter, averaged
over layers 24–31. The maximum bias is **half their log-odds difference**:

```text
0.5 * (logit(0.15382076) - logit(0.11922369)) = 0.14742953
exp(max_bias) = 1.15885161
```

This increases audio-versus-other attention odds by at most **15.89%** at fixed
Q/K. It would move the historical late aggregate mass from 11.92% to roughly
13.56% under that simplified fixed-distribution calculation; actual trajectories
can change. Half compensation is a conservative design choice, not an optimized
strength. Source hashes and historical IDs are stored in
`direct_boost_protocol_v24.json`. Calibration uses attention measurements only,
not previous correctness labels.

The additive mask uses the model's original attention implementation and never
rewrites upstream source. A zero-bias call follows the exact original mask-free
path. Active intervention can use a different attention kernel because the mask
is nonempty; the report measures warmed synchronized autonomous decode time, including the actual bias implementation overhead.
Persistent high-attention positions receive the boost too; no sink is silently
excluded or identified from future evaluation data.

## Run on the existing Vast audio environment

Copy the supplied source bundle into the experiment checkout. No push, new GPU
rental, or modification of an existing running job is required. If needed, set
up its dependencies first with `bash setup.sh`.

```bash
cd /workspace/llada-audio-drift
bash scripts/vast_audio_direct_boost.sh plan

# Optional end-to-end GPU check, separate output directory:
RUN_NAME=audio_direct_boost2_smoke_v24 SAMPLES=2 \
  bash scripts/vast_audio_direct_boost.sh run

# The requested paired 50-question evaluation:
nohup bash scripts/vast_audio_direct_boost.sh run \
  > audio_direct_boost50_v24.log 2>&1 &
tail -f audio_direct_boost50_v24.log
```

The launcher samples exactly 50 questions with seed 20261024 from the pinned
dataset revision, filtering clips to at most 30 seconds. It excludes **all 130
previously used question IDs** (30 development plus 100 independent-v21 IDs).
Selection never consults correctness. The prepared manifest and source WAVs are
reused on rerun. The two-question smoke consumes the first two of this same
fixed pool, so those two are not independent validation if inspected; do not
tune from the smoke outputs. `SAMPLES=50`, `LENGTH=512`, `BLOCK_LENGTH=32`, and
`STEPS=256` are fixed by the protocol for the requested evaluation.

Both conditions stop only when EOS is present with all preceding answer slots
revealed. The first two questions additionally check exact zero-bias identity
and baseline agreement with the official fixed-budget decoder's visible output.
Saved activation logs fail if the run reaches the active interval but applies
no bias. Questions finishing before activation remain in the primary score.

## Results and inspection

`runs/audio_direct_boost50_v24/REPORT.md` and `summary.json` update after each
complete paired example. They report baseline/ramp accuracy, recoveries, losses,
parse failures, forward counts, number of activated examples, and an exact
paired test. The primary final-answer extractor is the already frozen
`explicit-conclusion-v1`; original strict marker scores are retained separately.
Neither scorer validates every reasoning step or proves acoustic grounding.

Per-example `result.json` saves both complete answers and token IDs.
`baseline_steps.json` and `ramp_steps.json` preserve commit trajectories.
`ramp_attention.json` records applied bias and multiplier without dense attention diagnostics in the timed path. The first two questions also run a separate untimed diagnostic pass, saved as `ramp_diagnostic_attention.json`, measuring per-layer semantic mass on up to eight current active masked probes at every forward. That pass must reproduce the timed boosted answer exactly. The
`local_unbiased_semantic_mass` entry is computed with the same live Q/K without
the local bias; it is **not** a separate baseline trajectory or a causal
importance score. The bias acts on all answer queries, not only these probes.

Completed pairs are skipped on an unchanged rerun. Interrupted pairs restart.
Changes to sources, inputs, protocol, exclusions, or runtime versions require a
new output directory. A result with higher attention alone is not a successful
accuracy result; compare all 50 paired final answers, including regressions.

## Local validation

```bash
DIFFA_ROOT=/path/to/pinned/DIFFA \
  python -m unittest discover -s tests -p test_direct_attention_boost.py -v
```

Tests check the explicit softmax/logit-bias formula, intervention scope, the
15.89% maximum odds increase, schedule boundaries, exact zero identity, hook
cleanup, and a tiny real upstream LLaDA transformer's nonzero output effect.
CPU checks do not establish full-size CUDA behavior or answer accuracy.

## Throughput reporting

The refreshed source bundle reports `answer_tokens_per_second`, `decode_seconds`,
`seconds_per_question`, and accuracy for both conditions. Answer TPS is the sum
of non-special generated answer tokens before EOS divided by the sum of timed
decode seconds across every completed question, including incorrect answers.
It includes reasoning text and never counts masked padding slots as generated
text. It is not the mean of per-question TPS values.

Each input shape warms the baseline and maximum-bias attention paths before
timing. Both timed decodes use the same stopping rule. Condition order alternates
by question. Timers synchronize CUDA at both ends and exclude model loading,
audio preparation, warmup, diagnostic reconstruction, official/zero verification,
scoring, and disk writes. Each condition has one timed decode per question;
50 questions provide a paired workload, not repeated latency confidence bounds.
The summary JSON also supplies TPS including that question's separately measured
audio preparation. TPS differences can reflect different reasoning lengths and
forward counts, not just per-forward implementation cost.

Use a fresh output name for this timing-enabled source version:

```bash
RUN_NAME=audio_direct_boost50_v24_tps bash scripts/vast_audio_direct_boost.sh run
.venv-audio/bin/python -m experiments.audio_drift.run_direct_boost \
  --out runs/audio_direct_boost50_v24_tps --report-only
cat runs/audio_direct_boost50_v24_tps/REPORT.md
```
