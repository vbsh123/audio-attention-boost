"""Add semantic-key logit bias to answer queries at the actual attention call.

This changes attention itself, not unmasking priority. Zero bias delegates to the
original call without a mask, preserving its backend and exact output.
"""
from contextlib import AbstractContextManager
import math

import torch

from .direct_boost_protocol import bias_at
from .hooks import blocks_of


class DirectAttentionBoost(AbstractContextManager):
    def __init__(self, llm, prepared, protocol, zero=False, probe_count=8, monitor=True):
        self.blocks = blocks_of(llm)
        self.prepared, self.protocol = prepared, protocol
        self.layers = set(protocol['layers'])
        if not self.layers or not self.layers <= set(range(len(self.blocks))):
            raise ValueError('Invalid boost layers')
        if protocol['key_group'] != 'semantic' or protocol['query_scope'] != 'all_answer_positions':
            raise ValueError('Unsupported direct-boost scope')
        bias_at(0, protocol)
        self.positions = prepared.group_positions('semantic')
        if not self.positions or probe_count < 1:
            raise ValueError('Semantic keys and positive probe count required')
        self.zero, self.probe_count = zero, probe_count
        self.monitor = monitor
        self.originals, self.masks, self.records = [], {}, []
        self.meta = None

    def begin(self, meta, ids_with_audio):
        self.meta = dict(meta)
        self.bias = 0. if self.zero else bias_at(meta['step'], self.protocol)
        self.observed = {}
        self.called_layers = set()
        if not self.monitor:
            return
        # Diagnostics use current active masked positions, never reference text,
        # a baseline future state, or the eventual answer length.
        answer = ids_with_audio[0, self.prepared.answer_start:]
        start = meta['block'] * meta['block_length']
        end = min(start + meta['block_length'], len(answer))
        candidates = (answer[start:end] == self.prepared.mask_id).nonzero().flatten() + start
        if not len(candidates):
            raise ValueError('No active masked answer queries')
        n = min(self.probe_count, len(candidates))
        indices = torch.linspace(0, len(candidates)-1, n, device=candidates.device).long()
        self.queries = candidates[indices] + self.prepared.answer_start

    def _call(self, layer, original, q, k, v, attn_mask=None, dropout_p=0., is_causal=False):
        if self.meta is None:
            raise RuntimeError('Call begin() before model forward')
        if (q.shape[0] != 1 or k.shape[-2] != q.shape[-2]
                or attn_mask is not None or dropout_p or is_causal):
            raise ValueError('Expected batch-one uncached unmasked eval attention')
        if k.shape[1] != q.shape[1] or v.shape[1] != q.shape[1]:
            raise ValueError('Pinned DIFFA requires equal query/key/value head counts')
        self.called_layers.add(layer)
        if self.bias:
            # Broadcast across heads; never bias prefix queries or acoustic keys.
            cache_key = (q.device, q.dtype, q.shape[-2], self.bias)
            if cache_key not in self.masks:
                # Bound memory: only the current forward's mask is retained.
                self.masks.clear()
                mask = torch.zeros((q.shape[-2], k.shape[-2]), device=q.device, dtype=torch.float32)
                mask[self.prepared.answer_start:, self.positions] = self.bias
                self.masks[cache_key] = mask
            output = original(q, k, v, attn_mask=self.masks[cache_key],
                              dropout_p=dropout_p, is_causal=is_causal)
        else:
            output = original(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=is_causal)
        if not self.monitor:
            return output
        # Counterfactual local probabilities under the SAME Q/K; these are not
        # an unperturbed baseline trajectory once earlier layers were boosted.
        selected = q[0].index_select(-2, self.queries.to(q.device)).float()
        probabilities = (selected @ k[0].float().transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1)
        before = probabilities[..., self.positions].sum(-1)
        multiplier = math.exp(self.bias)
        after = multiplier * before / (1 + (multiplier - 1) * before)
        self.observed[layer] = torch.stack((before.mean(), after.mean(), after.min(), after.max())).detach()
        return output

    def end(self, layers=None):
        if self.called_layers != self.layers:
            raise RuntimeError('Not all selected attention layers were called')
        if not self.monitor:
            self.records.append(dict(**self.meta, logit_bias=self.bias,
                audio_odds_multiplier=math.exp(self.bias), attention_diagnostics=False))
            return
        # One synchronized small transfer per forward, rather than per layer.
        values = torch.stack([self.observed[l] for l in sorted(self.layers)]).cpu().tolist()
        self.records.append(dict(**self.meta, logit_bias=self.bias,
            audio_odds_multiplier=math.exp(self.bias), probe_offsets=(self.queries-self.prepared.answer_start).tolist(),
            layers=[dict(layer=l, local_unbiased_semantic_mass=x[0], biased_semantic_mass=x[1],
                         biased_mass_min=x[2], biased_mass_max=x[3]) for l, x in zip(sorted(self.layers), values)]))

    def __enter__(self):
        try:
            for layer in sorted(self.layers):
                block = self.blocks[layer]
                name = '_scaled_dot_product_attention'
                existed, original = name in block.__dict__, getattr(block, name)
                self.originals.append((block, name, existed, original))
                def attention(q, k, v, attn_mask=None, dropout_p=0., is_causal=False,
                              _layer=layer, _original=original):
                    return self._call(_layer, _original, q, k, v, attn_mask, dropout_p, is_causal)
                setattr(block, name, attention)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for block, name, existed, original in reversed(self.originals):
            if existed:
                setattr(block, name, original)
            else:
                delattr(block, name)
        self.originals.clear()
        self.masks.clear()
        return False
