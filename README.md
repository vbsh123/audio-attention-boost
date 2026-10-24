# Audio attention boost

Paired evaluation of a mild, increasing semantic-audio attention bias in
DIFFA-2 / LLaDA-8B-Instruct on **50 new spoken GSM8K questions**. Reports final
answer **accuracy, answer tokens per second, decode time, and paired wins/losses**.
Accuracy improvement is a hypothesis; no full-size GPU results are included.

## Clone, set up, and run on Vast

Use Python 3.10–3.12 and a BF16-capable CUDA GPU with enough memory for the
8B model plus Whisper (the prior experiments used an A40).

```bash
git clone https://github.com/vbsh123/audio-attention-boost.git
cd audio-attention-boost
bash setup.sh
```

Setup automatically starts the 50-question paired experiment in the background
after all installation and GPU checks succeed. No separate run command is needed.
It prints the PID and writes progress to `run.log`. Follow progress if desired:

```bash
tail -f run.log
```

`Ctrl+C` exits the log viewer; the background experiment continues. Setup creates
`.venv-audio`, installs the pinned CUDA PyTorch pair and dependencies, and checks
out immutable upstream DIFFA source. The run downloads/caches model weights and
dataset shards, then prepares a reproducible 50-question audio manifest. Models
and audio remain outside version control.

The report updates after every complete baseline/boosted question pair:

```bash
bash report.sh
```

Results are in `runs/audio_direct_boost50_v24_tps/REPORT.md` and `summary.json`.
Each sample includes both generated answers, token IDs, commit trajectories, and
boost activation logs. The first two questions also receive separate untimed
attention diagnostics, zero-bias identity checks, and official-decoder checks.

To package the final results:

```bash
python3 -m zipfile -c results.zip runs/audio_direct_boost50_v24_tps run.log
```

## Frozen experiment

- Excludes all 130 questions from earlier development/evaluation runs.
- Baseline and one ramp policy on identical questions and decoding budgets.
- Historical mean answer-content forwards: **87.15**.
- Bias is zero through forward index **44**, rises linearly, and caps at **88**.
- Maximum additive attention-logit bias **0.14742953**, about **15.89% higher
  semantic-audio-versus-other attention odds** at fixed query/key scores.
- Bias acts on semantic-audio keys for answer queries in layers **24–31**.
- Strength compensates half the measured midpoint-to-late log-odds decline;
  it is fixed before new evaluation and is not claimed to be optimal.
- 512 answer slots, 32-slot blocks, at most 256 forwards; safe EOS stopping.
- Frozen final-numerical-answer grading; original marker scores retained too.

This changes attention itself, not the previously tested commitment-priority
heuristic. See [the full protocol](experiments/audio_drift/DIRECT_ATTENTION_BOOST.md)
and [frozen calibration](experiments/audio_drift/direct_boost_protocol_v24.json).

## TPS definition

Answer TPS = **total non-special answer tokens before EOS / total synchronized
decode seconds**, including reasoning text and all questions regardless of
correctness. Warmup, model loading, audio preparation, grading, disk writes, and
diagnostic attention reconstruction are excluded. Each shape warms both paths;
paired order alternates across questions. The summary also includes throughput
with audio preparation added. One timed decode per condition/question is used.

Answer lengths and forward counts can change, so TPS is reported alongside
accuracy and seconds/question. It does not isolate per-forward kernel speed.

## Optional controls and resuming

```bash
# Inspect the frozen plan without installing model dependencies:
bash run.sh plan

# Optional two-question GPU smoke check, separate output name:
RUN_NAME=audio_direct_boost2_smoke_v24_tps SAMPLES=2 bash run.sh

# Resume an interrupted 50-question run with unchanged sources/settings:
nohup bash run.sh > run-resumed.log 2>&1 &
```

Completed pairs are skipped; incomplete pairs restart. A changed source or
protocol requires a new `RUN_NAME`. The smoke check uses the first two questions
of the same fixed evaluation pool; do not tune settings from its answers.

`PYTHON=python3.12 bash setup.sh` selects a specific Python.
`AUTO_RUN=0 bash setup.sh` installs and checks the environment without starting
an experiment. A checkout lock prevents two run commands from using the GPU
concurrently in this checkout. Repeated starts append to `run.log`.
`AUDIO_VENV=/path/to/existing/audio/environment` reuses an existing environment.
`DIFFA_ROOT` can select an existing clean pinned upstream checkout. Upstream
source is verified and never edited.

## Tests

Six CPU tests cover the logit-bias formula, intervention scope, timing/diagnostic
output equivalence, schedule, hook cleanup, real upstream tiny-model behavior,
BF16 paired-run reporting, official decoding, and resume behavior. CPU tests
do not establish full-size GPU accuracy or throughput.

```bash
source .venv-audio/bin/activate
DIFFA_ROOT="$PWD/external/DIFFA" \
  python -m unittest discover -s tests -p test_direct_attention_boost.py -v
```
