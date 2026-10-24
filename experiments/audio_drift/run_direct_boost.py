"""Paired 50-question baseline versus a frozen, increasing audio logit bias."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

from .direct_boost_protocol import bias_at
from .inputs import canonical_number, read_manifest, score
from .numeric_answer_report import VERSION, extract_answer


HERE = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def summarize(out):
    results = [json.loads(p.read_text()) for p in sorted((out/'samples').glob('*/result.json'))
               if not p.parent.name.startswith('.')]
    if not results:
        return
    wins = sum(not r['baseline']['secondary']['correct'] and r['ramp']['secondary']['correct'] for r in results)
    losses = sum(r['baseline']['secondary']['correct'] and not r['ramp']['secondary']['correct'] for r in results)
    n = wins + losses
    pvalue = min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses)+1)) / 2**n) if n else 1.
    summary = dict(completed=len(results), expected=json.loads((out/'config.json').read_text())['samples'],
                   recovered=wins, lost=losses, net=wins-losses, exact_paired_p=pvalue,
                   activated_examples=sum(r['ramp']['boosted_forwards']>0 for r in results), conditions={})
    for name in ('baseline', 'ramp'):
        elapsed=sum(r[name]['decode_seconds'] for r in results)
        tokens=sum(r[name]['visible_answer_tokens'] for r in results)
        with_audio=sum(r[name]['decode_seconds']+r['audio_preparation_seconds'] for r in results)
        summary['conditions'][name] = dict(correct=sum(r[name]['secondary']['correct'] for r in results),
            accuracy=sum(r[name]['secondary']['correct'] for r in results)/len(results),
            original_marker_correct=sum(r[name]['score']['correct'] for r in results),
            parse_failures=sum(r[name]['secondary']['prediction'] is None for r in results),
            forwards=sum(r[name]['forwards'] for r in results),
            missing_eos=sum(r[name]['missing_eos'] for r in results),
            visible_answer_tokens=tokens, decode_seconds=elapsed, answer_tokens_per_second=tokens/elapsed,
            seconds_per_question=elapsed/len(results),
            answer_tokens_per_second_with_audio_preparation=tokens/with_audio)
    summary['ramp_decode_speedup']=summary['conditions']['baseline']['decode_seconds']/summary['conditions']['ramp']['decode_seconds']
    write_json(out/'summary.json', summary)
    lines=['# Direct audio attention boost: paired evaluation', '',
           f"Completed {len(results)}/{summary['expected']}. Frozen primary extractor: {VERSION}.", '',
           '| Condition | Correct | Accuracy | Answer TPS | Decode seconds | Seconds/question | Parse failures | Forwards |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in summary['conditions'].items():
        lines.append(f"| {name} | {r['correct']}/{len(results)} | {100*r['accuracy']:.1f}% | {r['answer_tokens_per_second']:.2f} | {r['decode_seconds']:.2f} | {r['seconds_per_question']:.2f} | {r['parse_failures']} | {r['forwards']} |")
    lines += ['', f'Recovered: {wins}; lost: {losses}; net: {wins-losses:+d}. Exact paired two-sided p={pvalue:.4g}.',
              f"Nonzero boost on {summary['activated_examples']}/{len(results)} examples; short nonactivated examples remain in the primary score.",
              f"Ramp decode speedup: {summary['ramp_decode_speedup']:.3f}x (below 1 means slower).",
              'Answer TPS = total non-special answer tokens before EOS / total synchronized decode seconds; ratio of sums, not a mean of per-question rates. Includes reasoning text. Audio preparation, warmup, verification, diagnostics, scoring, and disk writes are outside decode timing. Each shape warms both paths first; paired order alternates across questions. One timed decode per condition/question, so GPU noise remains.',
              'This grades final numerical answers, not full reasoning or acoustic grounding. Different answer lengths and forward counts can affect TPS and total latency; this is not a matched-input per-forward speed comparison.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=HERE/'direct_boost_protocol_v24.json')
    p.add_argument('--exclusions', type=Path, default=HERE/'direct_boost_excluded_ids_v24.json')
    p.add_argument('--samples', type=int, default=50)
    p.add_argument('--diffa-root', type=Path, default=HERE.parents[1]/'external'/'DIFFA')
    p.add_argument('--plan', action='store_true')
    p.add_argument('--report-only', action='store_true', help='Regenerate TPS/accuracy reports without loading the model')
    args=p.parse_args()
    if args.report_only:
        summarize(args.out)
        return
    if args.manifest is None and not args.plan:
        p.error('--manifest is required for evaluation')
    protocol=json.loads(args.protocol.read_text())
    if args.samples < 1 or args.samples > protocol['samples']:
        raise ValueError('Use 1–50 questions with this frozen protocol')
    bias_at(0, protocol)
    print(json.dumps(dict(samples=args.samples, start=protocol['start_forward'], end=protocol['end_forward'],
        max_bias=protocol['max_logit_bias'], max_odds_multiplier=protocol['maximum_audio_odds_multiplier'],
        layers=protocol['layers'], generation=protocol['generation'], grading=VERSION), indent=2), flush=True)
    if args.plan:
        return
    rows=read_manifest(args.manifest, args.samples)
    if len(rows)!=args.samples:
        raise ValueError('Manifest has fewer questions than requested')
    exclusions=json.loads(args.exclusions.read_text())
    excluded=set(exclusions['ids']) | set(protocol['calibration_ids'])
    if excluded & {r['id'] for r in rows}:
        raise ValueError('Evaluation overlaps previous questions/calibration')
    if any(r.get('source_dataset')!=protocol['source_dataset'] or
           r.get('source_revision')!=protocol['source_revision'] or r.get('metric')!='numeric' for r in rows):
        raise ValueError('Expected pinned GSM8K Speech numeric questions')
    import torch
    from types import SimpleNamespace
    from .checkout import COMMIT, verify_checkout
    from .decode import decode
    from .direct_attention_boost import DirectAttentionBoost
    from .model import (load_model, prepare, DIFFA_REPO, DIFFA_REVISION, LLM_REPO, LLM_REVISION)
    from .inputs import load_audio
    verify_checkout(args.diffa_root)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Run on a BF16-capable CUDA GPU (e.g. the existing Vast audio environment)')
    torch.manual_seed(protocol['seed']); torch.cuda.manual_seed_all(protocol['seed'])
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    digest=hashlib.sha256()
    for path in sorted(HERE.glob('*.py')):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    config=dict(protocol=protocol, protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        source_sha256=digest.hexdigest(), samples=args.samples, inputs=rows,
        timing_protocol='synchronized-decode-no-probes-v1', paired_order='alternating',
        warmup='one unboosted and one maximum-bias forward per input shape', diagnostic_samples=2,
        upstream_commit=COMMIT, torch=torch.__version__, python=platform.python_version(),
        extractor_sha256=hashlib.sha256((HERE/'numeric_answer_report.py').read_bytes()).hexdigest(),
        exclusions_sha256=hashlib.sha256(args.exclusions.read_bytes()).hexdigest(),
        models=dict(diffa=DIFFA_REVISION, llm=LLM_REVISION, whisper='06f233fe06e710322aca913c1bc4249a0d71fce1'))
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'config.json').exists():
        if json.loads((args.out/'config.json').read_text())!=config:
            raise ValueError('Existing config differs: use a fresh run directory')
    elif any(args.out.iterdir()):
        raise ValueError('Output is nonempty without a matching config')
    else:
        write_json(args.out/'config.json',config)
    (args.out/'samples').mkdir(exist_ok=True)
    model, tokenizer, extractor, report=load_model(args.diffa_root, DIFFA_REPO, DIFFA_REVISION,
        LLM_REPO, LLM_REVISION, 'openai/whisper-large-v3', config['models']['whisper'])
    write_json(args.out/'load_report.json', report)
    write_json(args.out/'device.json',dict(name=torch.cuda.get_device_name(),
        memory_bytes=torch.cuda.get_device_properties(0).total_memory))
    llm=model.llm_model; gen=protocol['generation']
    from src.origin_generate import generate as official_generate
    from .decode import forward_logits

    def visible_ids(x, prepared):
        ids=x[0,prepared.prompt.shape[1]:].tolist()
        return ids[:ids.index(tokenizer.eos_token_id)+1] if tokenizer.eos_token_id in ids else ids

    for i,row in enumerate(rows):
        destination=args.out/'samples'/f'{i:05d}'
        if (destination/'result.json').exists():
            print('Skip completed',row['id'],flush=True); continue
        destination.mkdir(exist_ok=True)
        print(f"[{i+1}/{len(rows)}] {row['id']}",flush=True)
        audio, seconds=load_audio(row['audio'])
        torch.cuda.synchronize(); preparation_start=time.perf_counter()
        prepared=prepare(model, tokenizer, extractor, audio, row['prompt'], gen['length'])
        torch.cuda.synchronize(); preparation_seconds=time.perf_counter()-preparation_start
        result=dict(id=row['id'], target=row['target'], audio_seconds=seconds,
                    audio_preparation_seconds=preparation_seconds,
                    semantic_positions=len(prepared.group_positions('semantic')))
        # Prime the actual unmasked and additive-mask backend paths for this
        # sequence shape. No generated answer, future state, or reference is used.
        warm_x=prepared.prompt.new_full((1,prepared.prompt.shape[1]+gen['length']),prepared.mask_id)
        warm_x[:,:prepared.prompt.shape[1]]=prepared.prompt
        with torch.no_grad():
            warm=forward_logits(llm,prepared,warm_x); del warm
            with DirectAttentionBoost(llm,prepared,protocol,monitor=False) as warm_boost:
                warm_boost.begin(dict(step=protocol['end_forward'],block=0,block_length=gen['block_length']),
                                 prepared.ids_with_audio(warm_x))
                warm=forward_logits(llm,prepared,warm_x); del warm
                warm_boost.end()
        del warm_x
        outputs={}
        result['condition_order']=['baseline','ramp'] if i%2==0 else ['ramp','baseline']
        for condition in result['condition_order']:
            recorder=DirectAttentionBoost(llm,prepared,protocol,monitor=False) if condition=='ramp' else None
            torch.cuda.synchronize(); start=time.perf_counter()
            from contextlib import nullcontext
            with recorder if recorder is not None else nullcontext():
                final,_,events=decode(llm,prepared,gen['length'],gen['block_length'],gen['steps'],
                    recorder=recorder, save_states=False, eos_id=tokenizer.eos_token_id,
                    progress_every=0, progress_label=condition)
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            ids=final[0,prepared.prompt.shape[1]:].tolist()
            visible=visible_ids(final,prepared)
            text=tokenizer.decode(visible,skip_special_tokens=True).strip()
            specials=set(getattr(tokenizer,'all_special_ids',[tokenizer.eos_token_id]))
            token_count=sum(t not in specials for t in visible)
            prediction,method=extract_answer(text)
            result[condition]=dict(text=text,answer_ids=ids,forwards=len(events),missing_eos=tokenizer.eos_token_id not in ids,
                score=score(text,row['target'],'numeric'),decode_seconds=elapsed,
                visible_answer_tokens=token_count,answer_tokens_per_second=token_count/elapsed,
                secondary=dict(version=VERSION,prediction=prediction,extraction=method,
                               correct=prediction==canonical_number(row['target'])),
                boosted_forwards=sum(r['logit_bias']>0 for r in recorder.records) if recorder else 0)
            write_json(destination/f'{condition}_steps.json',events)
            if recorder:
                write_json(destination/'ramp_attention.json',recorder.records)
                if len(events)>protocol['start_forward']+1 and not result[condition]['boosted_forwards']:
                    raise RuntimeError('Expected nonzero intervention but none applied')
            outputs[condition]=final
            print(condition,result[condition]['secondary'], 'forwards',len(events),flush=True)
        if i<config['diagnostic_samples']:
            with DirectAttentionBoost(llm,prepared,protocol,monitor=True) as diagnostics:
                diagnostic,_,_=decode(llm,prepared,gen['length'],gen['block_length'],gen['steps'],
                    recorder=diagnostics,save_states=False,eos_id=tokenizer.eos_token_id)
            if not torch.equal(diagnostic,outputs['ramp']):
                raise RuntimeError('Diagnostic pass changed the boosted answer')
            write_json(destination/'ramp_diagnostic_attention.json',diagnostics.records)
            del diagnostic
        if i<protocol['zero_identity_samples']:
            with DirectAttentionBoost(llm,prepared,protocol,zero=True) as control:
                zero,_,_=decode(llm,prepared,gen['length'],gen['block_length'],gen['steps'],recorder=control,
                               save_states=False,eos_id=tokenizer.eos_token_id)
            result['zero_identity']=torch.equal(zero,outputs['baseline'])
            if not result['zero_identity']:
                raise RuntimeError('Zero bias does not reproduce baseline exactly')
        if i<protocol['official_verify_samples']:
            official=official_generate(SimpleNamespace(llm_model=llm,device=llm.device),prepared.prompt,
                prepared.prompt.new_tensor([prepared.insertion]),prepared.speech,steps=gen['steps'],
                gen_length=gen['length'],block_length=gen['block_length'],temperature=0.,cfg_scale=0.,mask_id=prepared.mask_id)
            result['official_baseline_match']=visible_ids(official,prepared)==visible_ids(outputs['baseline'],prepared)
            if not result['official_baseline_match']:
                raise RuntimeError('Baseline differs from official decoder')
            del official
        write_json(destination/'result.json',result)
        summarize(args.out)
        del outputs,prepared


if __name__=='__main__':
    main()
