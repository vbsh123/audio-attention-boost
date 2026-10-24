"""Freeze a mild direct-attention intervention from historical measurements."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def bias_at(step, protocol):
    start, end = protocol['start_forward'], protocol['end_forward']
    if start < 0 or end <= start:
        raise ValueError('Invalid boost interval')
    maximum = protocol['max_logit_bias']
    if not math.isfinite(maximum) or maximum < 0:
        raise ValueError('Invalid logit bias')
    return maximum * min(1., max(0., (step - start) / (end - start)))


def calibrate(baseline_root, depth_summary, compensation_fraction=.5):
    if not 0 < compensation_fraction <= 1:
        raise ValueError('Compensation fraction must be in (0, 1]')
    paths = sorted(Path(baseline_root).glob('samples/*/baseline.json'))
    if not paths:
        raise ValueError('No historical baselines')
    baselines = [json.loads(p.read_text()) for p in paths]
    forwards = [b['baseline_content_revealed_step'] + 1 for b in baselines]
    mean = statistics.mean(forwards)
    with Path(depth_summary).open() as source:
        rows = [r for r in csv.DictReader(source) if r['group'] == 'semantic'
                and int(r['layer_start']) == 24 and int(r['layer_end']) == 31]
    by_quarter = {int(r['quarter']): r for r in rows}
    if set(by_quarter) != {0, 1, 2, 3}:
        raise ValueError('Need all four semantic-attention quarters at layers 24–31')
    midpoint, late = (float(by_quarter[q]['mass_mean']) for q in (1, 3))
    if not 0 < late < midpoint < 1:
        raise ValueError('Expected a mid-to-late decline in semantic attention mass')
    deficit = math.log(midpoint / (1 - midpoint)) - math.log(late / (1 - late))
    maximum = compensation_fraction * deficit
    return dict(version='direct-audio-logit-bias-v24', source_dataset='mistralai/gsm8k_speech',
                source_revision='e94caa2154433a54cd430702666c454977df3083',
                calibration_ids=[b['id'] for b in baselines], historical_content_forwards=forwards,
                mean_content_forwards=mean, start_forward=math.ceil(mean / 2),
                end_forward=math.ceil(mean), max_logit_bias=maximum,
                maximum_audio_odds_multiplier=math.exp(maximum), layers=list(range(24, 32)),
                key_group='semantic', query_scope='all_answer_positions',
                compensation_fraction=compensation_fraction,
                calibration=dict(midpoint_quarter=1, late_quarter=3, midpoint_mass=midpoint,
                    late_mass=late, full_log_odds_deficit=deficit,
                    attention_examples=int(by_quarter[1]['examples']),
                    depth_summary_sha256=hashlib.sha256(Path(depth_summary).read_bytes()).hexdigest(),
                    baselines_sha256={p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),
                interpretation='Half the historical midpoint-to-late log-odds deficit is a conservative design choice, not a fitted optimum. Attention calibration uses fixed probes, not evidence of causal grounding.',
                generation=dict(length=512, block_length=32, steps=256), samples=50, seed=20261024,
                grading='explicit-conclusion-v1', zero_identity_samples=2, official_verify_samples=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root', type=Path, required=True)
    p.add_argument('--depth-summary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    protocol = calibrate(args.baseline_root, args.depth_summary)
    args.out.write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({k: protocol[k] for k in ('mean_content_forwards', 'start_forward',
          'end_forward', 'max_logit_bias', 'maximum_audio_odds_multiplier')}, indent=2))


if __name__ == '__main__':
    main()
