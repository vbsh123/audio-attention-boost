import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from experiments.audio_drift.direct_boost_protocol import bias_at
from experiments.audio_drift.compare_tuning import compare

ROOT=Path(__file__).resolve().parents[1]/'experiments/audio_drift'


def load(name):
    return json.loads((ROOT/name).read_text())


class TuningTests(unittest.TestCase):
    def test_profiles_change_only_requested_schedule_strength_and_layers(self):
        original=load('direct_boost_protocol_v24.json')
        deep=load('direct_boost_early15_deep_v25.json')
        all_layers=load('direct_boost_early15_all_v25.json')
        self.assertEqual(original['start_forward'],44)
        self.assertAlmostEqual(original['maximum_audio_odds_multiplier'],1.158851614520242)
        for p in [deep,all_layers]:
            self.assertEqual(p['start_forward'],math.ceil(original['mean_content_forwards']*.25))
            self.assertEqual(p['end_forward'],88)
            self.assertEqual(bias_at(22,p),0)
            self.assertGreater(bias_at(23,p),0)
            self.assertAlmostEqual(math.exp(bias_at(88,p)),1.5)
            self.assertEqual(bias_at(88,p),bias_at(255,p))
            for field in ['generation','seed','calibration_ids','key_group','query_scope']:
                self.assertEqual(p[field],original[field])
        self.assertEqual(deep['layers'],list(range(24,32)))
        self.assertEqual(all_layers['layers'],list(range(32)))

    def test_comparison_requires_identical_inputs_and_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)
            for label in ['deep','all']:
                p=r/label;p.mkdir()
                config=dict(protocol=load(f'direct_boost_early15_{label}_v25.json'),
                    inputs=[dict(id='a')],models=dict(llm='pinned'),torch='fixture',
                    extractor_sha256='same',source_sha256='same',samples=1)
                (p/'config.json').write_text(json.dumps(config))
                metrics=dict(correct=1,accuracy=1.,answer_tokens_per_second=8.,seconds_per_question=10.,forwards=80)
                (p/'summary.json').write_text(json.dumps(dict(completed=1,expected=1,
                    conditions=dict(baseline=metrics,ramp=metrics))))
                result=dict(id='a',baseline=dict(answer_ids=[1,2]),ramp=dict(secondary=dict(correct=True)))
                (p/'samples/00000').mkdir(parents=True)
                (p/'samples/00000/result.json').write_text(json.dumps(result))
            with contextlib.redirect_stdout(io.StringIO()):compare(r/'deep',r/'all',r/'out')
            self.assertTrue((r/'out/COMPARISON.md').exists())
            path=r/'all/samples/00000/result.json'
            result=json.loads(path.read_text());result['baseline']['answer_ids']=[3,4]
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'Baseline token mismatch'):
                compare(r/'deep',r/'all',r/'invalid')
            self.assertFalse((r/'invalid').exists())


if __name__=='__main__':unittest.main()
