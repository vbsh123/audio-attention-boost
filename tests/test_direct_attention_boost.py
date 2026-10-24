"""Numerical tests of direct attention intervention, not only schedule plumbing."""
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from experiments.audio_drift.decode import Prepared, decode, forward_logits
from experiments.audio_drift.direct_attention_boost import DirectAttentionBoost
from experiments.audio_drift.direct_boost_protocol import bias_at


PROTOCOL=json.loads((Path(__file__).resolve().parents[1]/'experiments/audio_drift/direct_boost_protocol_v24.json').read_text())


class Block:
    def _scaled_dot_product_attention(self,q,k,v,**kwargs):
        self.last_mask=kwargs.get('attn_mask')
        return F.scaled_dot_product_attention(q,k,v,**kwargs)


class AttrDict(dict):
    __getattr__=dict.__getitem__


class DirectBoostTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(18)
        self.blocks=[Block(),Block()]
        self.llm=SimpleNamespace(model=SimpleNamespace(transformer=AttrDict(blocks=self.blocks)),
                                 config=SimpleNamespace(n_layers=2))
        self.prepared=Prepared(torch.tensor([[1,2,3]]),torch.randn(1,4,4),1,1,31)
        self.protocol={**PROTOCOL,'layers':[1]}
        self.x=torch.tensor([[1,2,3,31,31,31,31]])
        self.q,self.k,self.v=[torch.randn(1,2,11,4) for _ in range(3)]

    def begin(self,controller,step=88):
        controller.begin(dict(step=step,block=0,block_length=4),self.prepared.ids_with_audio(self.x))

    def test_schedule_frozen_boundaries_and_monotonicity(self):
        self.assertEqual(PROTOCOL['start_forward'],44)
        self.assertEqual(PROTOCOL['end_forward'],88)
        self.assertAlmostEqual(PROTOCOL['mean_content_forwards'],87.15)
        values=[bias_at(s,PROTOCOL) for s in range(256)]
        self.assertEqual(values[44],0)
        self.assertGreater(values[45],0)
        self.assertEqual(values[88],values[255])
        self.assertEqual(values,sorted(values))
        self.assertAlmostEqual(math.exp(values[-1]),1.158851614520242)

    def test_attention_matches_explicit_logit_bias_and_scope(self):
        base=F.scaled_dot_product_attention(self.q,self.k,self.v)
        logits=self.q@self.k.transpose(-1,-2)/math.sqrt(4)
        logits[:,:,self.prepared.answer_start:,self.prepared.group_positions('semantic')]+=self.protocol['max_logit_bias']
        expected=logits.softmax(-1)@self.v
        with DirectAttentionBoost(self.llm,self.prepared,self.protocol) as boost:
            self.begin(boost)
            untouched=self.blocks[0]._scaled_dot_product_attention(self.q,self.k,self.v)
            actual=self.blocks[1]._scaled_dot_product_attention(self.q,self.k,self.v)
            boost.end()
        torch.testing.assert_close(untouched,base,rtol=0,atol=0)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=2e-7)
        torch.testing.assert_close(actual[:,:,:self.prepared.answer_start],base[:,:,:self.prepared.answer_start])
        mask=self.blocks[1].last_mask
        self.assertEqual(int((mask!=0).sum()),4*3)
        self.assertEqual(float(mask[:self.prepared.answer_start].sum()),0)
        self.assertEqual(float(mask[:,self.prepared.group_positions('acoustic')].sum()),0)
        r=boost.records[0]['layers'][0]
        self.assertGreater(r['biased_semantic_mass'],r['local_unbiased_semantic_mass'])
        with DirectAttentionBoost(self.llm,self.prepared,self.protocol,monitor=False) as clean:
            self.begin(clean)
            no_diagnostics=self.blocks[1]._scaled_dot_product_attention(self.q,self.k,self.v)
            clean.end()
        torch.testing.assert_close(no_diagnostics,actual,rtol=0,atol=0)
        self.assertFalse(clean.records[0]['attention_diagnostics'])

    def test_zero_and_before_start_exact_identity_and_restoration(self):
        base=F.scaled_dot_product_attention(self.q,self.k,self.v)
        for zero,step in [(True,100),(False,44)]:
            with DirectAttentionBoost(self.llm,self.prepared,self.protocol,zero=zero) as boost:
                self.begin(boost,step)
                actual=self.blocks[1]._scaled_dot_product_attention(self.q,self.k,self.v)
                boost.end()
                torch.testing.assert_close(actual,base,rtol=0,atol=0)
                self.assertIsNone(self.blocks[1].last_mask)
            self.assertNotIn('_scaled_dot_product_attention',self.blocks[1].__dict__)
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with DirectAttentionBoost(self.llm,self.prepared,self.protocol):
                raise RuntimeError('intentional')
        self.assertNotIn('_scaled_dot_product_attention',self.blocks[1].__dict__)

    def test_rejects_unsupported_attention(self):
        with DirectAttentionBoost(self.llm,self.prepared,self.protocol) as boost:
            self.begin(boost)
            with self.assertRaises(ValueError):
                self.blocks[1]._scaled_dot_product_attention(self.q,self.k,self.v,is_causal=True)


@unittest.skipUnless(os.environ.get('DIFFA_ROOT'),'Set DIFFA_ROOT for the actual upstream model test')
class UpstreamDirectBoostTests(unittest.TestCase):
    def tiny(self):
        from experiments.audio_drift.checkout import verify_checkout
        sys.dont_write_bytecode=True
        sys.path.insert(0,str(verify_checkout(os.environ['DIFFA_ROOT'])))
        from src.fast_dllm.model.configuration_llada import LLaDAConfig
        from src.fast_dllm.model.modeling_llada import LLaDAModelLM
        torch.manual_seed(10)
        config=LLaDAConfig(d_model=16,n_heads=2,n_kv_heads=2,n_layers=2,mlp_hidden_size=32,
            vocab_size=32,embedding_size=32,block_type='llama',block_group_size=1,rope=True,
            activation_type='silu',max_sequence_length=64,flash_attention=False,
            weight_tying=False,mask_token_id=31,eos_token_id=30,
            attention_dropout=0.,residual_dropout=0.,embedding_dropout=0.)
        llm=LLaDAModelLM(config,init_params=True).eval()
        output=torch.nn.Linear(16,32,bias=True)
        with torch.no_grad():output.bias[30:]=-100
        llm.model.transformer.ff_out=output
        prepared=Prepared(torch.tensor([[1,2,3]]),torch.randn(1,3,16),1,1,31)
        return llm,prepared

    def test_real_transformer_zero_identity_nonzero_effect_and_cleanup(self):
        llm,prepared=self.tiny()
        protocol={**PROTOCOL,'layers':[1],'start_forward':0,'end_forward':2}
        with torch.no_grad():
            ordinary,_,_=decode(llm,prepared,4,2,4,save_states=False)
            with DirectAttentionBoost(llm,prepared,protocol,zero=True) as zero:
                actual,_,_=decode(llm,prepared,4,2,4,recorder=zero,save_states=False)
            torch.testing.assert_close(actual,ordinary,rtol=0,atol=0)
            x=torch.tensor([[1,2,3,31,31,31,31]])
            baseline_logits=forward_logits(llm,prepared,x)
            with DirectAttentionBoost(llm,prepared,protocol) as boost:
                boost.begin(dict(step=2,block=0,block_length=2),prepared.ids_with_audio(x))
                changed_logits=forward_logits(llm,prepared,x)
                boost.end()
            self.assertGreater(float((changed_logits-baseline_logits).abs().max()),1e-7)
            torch.testing.assert_close(forward_logits(llm,prepared,x),baseline_logits,rtol=0,atol=0)

    def test_complete_runner_reports_and_resume_on_bf16_upstream_cpu(self):
        from contextlib import ExitStack
        from experiments.audio_drift.run_direct_boost import main
        llm,prepared=self.tiny()
        llm=llm.to(torch.bfloat16)
        prepared.speech=prepared.speech.to(torch.bfloat16)
        tokenizer=SimpleNamespace(eos_token_id=30,decode=lambda ids,skip_special_tokens: '#### 7')
        protocol={**PROTOCOL,'layers':[1],'start_forward':0,'end_forward':2,
                  'generation':dict(length=4,block_length=2,steps=4),'calibration_ids':[]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'sample.wav').write_bytes(b'fixture')
            (root/'protocol.json').write_text(json.dumps(protocol))
            (root/'excluded.json').write_text(json.dumps(dict(ids=['old-question'])))
            rows=[dict(id=f'new-question-{i}',audio='sample.wav',task='gsm8k_speech',metric='numeric',
                       prompt='Solve the audio.',target='7',source_dataset=protocol['source_dataset'],
                       source_revision=protocol['source_revision']) for i in range(2)]
            (root/'manifest.json').write_text(json.dumps(rows))
            argv=['runner','--manifest',str(root/'manifest.json'),'--protocol',str(root/'protocol.json'),
                  '--exclusions',str(root/'excluded.json'),'--out',str(root/'out'),'--samples','2',
                  '--diffa-root',os.environ['DIFFA_ROOT']]
            with ExitStack() as stack:
                # GPU preflight is mocked on a CPU model; keep Dynamo from
                # querying a nonexistent CUDA RNG while checking that preflight.
                stack.enter_context(torch._dynamo.config.patch(disable=True))
                for target,value in [('torch.cuda.is_available',True),('torch.cuda.is_bf16_supported',True),
                    ('torch.cuda.get_device_name','CPU fixture'),
                    ('torch.cuda.get_device_properties',SimpleNamespace(total_memory=0)),
                    ('torch.cuda.synchronize',None),('torch.cuda.manual_seed_all',None)]:
                    stack.enter_context(patch(target,return_value=value))
                stack.enter_context(patch('experiments.audio_drift.model.load_model',
                    return_value=(SimpleNamespace(llm_model=llm),tokenizer,None,dict(fixture=True))))
                stack.enter_context(patch('experiments.audio_drift.model.prepare',return_value=prepared))
                stack.enter_context(patch('experiments.audio_drift.inputs.load_audio',return_value=(None,1.)))
                stack.enter_context(patch.object(sys,'argv',argv))
                main()
                summary=json.loads((root/'out/summary.json').read_text())
                self.assertEqual(summary['completed'],2)
                self.assertEqual(summary['activated_examples'],2)
                for condition in ('baseline','ramp'):
                    metrics=summary['conditions'][condition]
                    self.assertGreater(metrics['decode_seconds'],0)
                    self.assertAlmostEqual(metrics['answer_tokens_per_second'],
                        metrics['visible_answer_tokens']/metrics['decode_seconds'])
                    self.assertEqual(metrics['accuracy'],1.)
                for i in range(2):
                    result=json.loads((root/f'out/samples/{i:05d}/result.json').read_text())
                    self.assertTrue(result['official_baseline_match'])
                    self.assertTrue(result['zero_identity'])
                # Matching completed pairs must be preserved on restart.
                before=(root/'out/samples/00000/result.json').read_bytes()
                main()
                self.assertEqual(before,(root/'out/samples/00000/result.json').read_bytes())


if __name__=='__main__':unittest.main()
