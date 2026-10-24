"""Compare two completed tuning runs without treating them as independent data."""
import argparse
import json
from pathlib import Path


def compare(deep_run, all_run, out):
    paths={'deep':Path(deep_run),'all':Path(all_run)}
    configs={name:json.loads((p/'config.json').read_text()) for name,p in paths.items()}
    summaries={name:json.loads((p/'summary.json').read_text()) for name,p in paths.items()}
    results={name:{r['id']:r for p in sorted((root/'samples').glob('*/result.json'))
                   if not p.parent.name.startswith('.') for r in [json.loads(p.read_text())]}
             for name,root in paths.items()}
    for field in ('inputs','models','torch','extractor_sha256','source_sha256'):
        if configs['deep'][field]!=configs['all'][field]:
            raise ValueError('Runs differ in '+field)
    for field in ('start_forward','end_forward','max_logit_bias','generation','seed'):
        if configs['deep']['protocol'][field]!=configs['all']['protocol'][field]:
            raise ValueError('Protocols differ beyond layers: '+field)
    if configs['deep']['protocol']['layers']!=list(range(24,32)) or configs['all']['protocol']['layers']!=list(range(32)):
        raise ValueError('Expected deep layers 24–31 versus all layers 0–31')
    ids=set(results['deep'])
    if ids!=set(results['all']) or len(ids)!=configs['deep']['samples']:
        raise ValueError('Both runs must have all requested paired questions completed')
    for name in paths:
        if summaries[name]['completed']!=len(ids) or summaries[name]['expected']!=len(ids):
            raise ValueError('Incomplete summary: '+name)
    for id in ids:
        if results['deep'][id]['baseline']['answer_ids']!=results['all'][id]['baseline']['answer_ids']:
            raise ValueError('Baseline token mismatch on '+id)
    all_wins=sorted(i for i in ids if results['all'][i]['ramp']['secondary']['correct'] and not results['deep'][i]['ramp']['secondary']['correct'])
    deep_wins=sorted(i for i in ids if results['deep'][i]['ramp']['secondary']['correct'] and not results['all'][i]['ramp']['secondary']['correct'])
    lines=['# Earlier, stronger audio boost: layer comparison','',
           f'{len(ids)} shared development questions. Same input fingerprints, baseline token IDs, strength and schedule; only the boosted layers differ.','',
           '| Run | Condition | Correct | Accuracy | Answer TPS | Seconds/question | Forwards |',
           '|---|---|---:|---:|---:|---:|---:|']
    for name,summary in summaries.items():
        for condition,r in summary['conditions'].items():
            lines.append(f"| {name} | {condition} | {r['correct']}/{len(ids)} | {100*r['accuracy']:.1f}% | {r['answer_tokens_per_second']:.2f} | {r['seconds_per_question']:.2f} | {r['forwards']} |")
    lines += ['',f'All-layers correct / deep wrong: {len(all_wins)}. Deep correct / all-layers wrong: {len(deep_wins)}.',
              'These are tuning results on the same questions, not independent replications. TPS includes reasoning and can change with answer length. Keep settings fixed before testing the larger validation pool.']
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    (out/'comparison.json').write_text(json.dumps(dict(questions=len(ids),summaries=summaries,
        all_wins=all_wins,deep_wins=deep_wins),indent=2)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deep-run',type=Path,required=True)
    p.add_argument('--all-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();compare(args.deep_run,args.all_run,args.out)
