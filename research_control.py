"""Submit bounded jobs or inspect the active study using current Modal configuration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['status','submit','stop','collect'])
    parser.add_argument('run_id')
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--job')
    parser.add_argument('--name')
    parser.add_argument('--output',default='outputs')
    args=parser.parse_args()
    state=modal.Dict.from_name('ensemble-desktop-localization-study-state')
    control=modal.Queue.from_name('ensemble-desktop-localization-study-control')
    if args.command=='status':
        if args.name:
            print(json.dumps(state.get(f'{args.run_id}:result:{args.name}',
                                       state.get(f'{args.run_id}:progress:{args.name}',{})),indent=2))
        else:
            print(json.dumps({str(i):state.get(f'{args.run_id}:worker:{i}',{}) for i in range(2)},indent=2))
    elif args.command=='submit':
        job=json.loads(args.job)
        if args.worker not in [0,1] or job.get('split','dev') not in ['dev','validation']:
            raise ValueError('Invalid worker or split')
        control.put(job,partition=f'{args.run_id}-{args.worker}')
        print('Submitted',job['name'])
    elif args.command=='stop':
        control.put({'stop':True},partition=f'{args.run_id}-{args.worker}')
        print('Stop requested after the current job')
    else:
        summary={}
        for key,value in state.items():
            if key.startswith(f'{args.run_id}:result:') or key.startswith(f'{args.run_id}:worker:'):
                summary[key]=value
        output=Path(args.output)/f'{args.run_id}-summaries.json'
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(summary,indent=2)+'\n')
        print(output)


if __name__=='__main__':
    main()

