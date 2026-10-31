from __future__ import annotations

import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

from grounding import study_policies as policies
from grounding.client import ModelPool
from grounding.evaluation import score_point


def record_cost(records):
    return {"model_calls": len(records),
            "prompt_tokens": sum(r.get('usage',{}).get('prompt_tokens',0) for r in records),
            "output_tokens": sum(r.get('usage',{}).get('completion_tokens',0) for r in records),
            "model_seconds": sum(r.get('wall_seconds',0) for r in records)}


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups['overall'].append(row)
        for key in ['ui_type','application','platform','group','target_size']:
            groups[f'{key}:{row.get(key,"unknown")}'].append(row)
    summary = {}
    for name, examples in groups.items():
        summary[name] = {
            'n':len(examples),'correct':sum(r['correct'] for r in examples),
            'accuracy':sum(r['correct'] for r in examples)/len(examples),
            'errors':sum('error' in r for r in examples),
            'used_python':sum(bool(r.get('prediction',{}).get('used_python')) for r in examples),
            'python_steps':sum(sum(e.get('type')=='python' for e in r.get('prediction',{}).get('events',[])) for r in examples),
            'python_errors':sum(sum(e.get('type')=='python' and bool(e.get('error'))
                                    for e in r.get('prediction',{}).get('events',[])) for r in examples),
            'mean_calls':sum(r['cost']['model_calls'] for r in examples)/len(examples),
            'mean_model_seconds':sum(r['cost']['model_seconds'] for r in examples)/len(examples),
            'mean_prompt_tokens':sum(r['cost']['prompt_tokens'] for r in examples)/len(examples),
            'mean_output_tokens':sum(r['cost']['output_tokens'] for r in examples)/len(examples),
        }
    return summary


class StudyRunner:
    def __init__(self, config, data_root, output_root, run_id, state, deadline):
        self.config, self.data_root, self.output_root = config, Path(data_root), Path(output_root)
        self.run_id, self.state, self.deadline = run_id, state, deadline
        self.memo = {}

    def baseline(self, ctx, alias):
        key = f'{self.run_id}:baseline:{ctx["split"]}:{alias}'
        if key not in self.memo:
            self.memo[key] = self.state.get(key, {})
        saved = self.memo[key].get(ctx['id'])
        if saved is not None:
            ctx['cached_records'].extend(saved['records'])
            return saved['prediction']
        before = len(ctx['pool'].records)
        prediction = policies.direct(ctx, alias)
        ctx['baseline_produced'][alias] = {'prediction':prediction,
                                         'records':self.compact_records(ctx['pool'].records[before:])}
        return prediction

    @staticmethod
    def compact_records(records):
        return [{k:r[k] for k in ['model','usage','wall_seconds','error'] if k in r} for r in records]

    def run(self, job):
        name, split = job['name'], job.get('split','dev')
        output = self.output_root / name
        output.mkdir(parents=True, exist_ok=False)
        (output/'job.json').write_text(json.dumps(job,indent=2)+'\n')
        rows = [json.loads(line) for line in (self.data_root/f'{split}.jsonl').read_text().splitlines()]
        offset = job.get('offset',0)
        rows = rows[offset:offset+job.get('limit',len(rows))]
        results, produced = [], defaultdict(dict)
        start = time.monotonic()
        for index, row in enumerate(rows):
            if time.monotonic() > self.deadline - 90:
                break
            path = self.data_root/row['image']
            with Image.open(path) as source:
                image = source.convert('RGB')
            pool = ModelPool(self.config, max_calls=job.get('max_calls',6),
                             max_output_tokens=job.get('max_output_tokens',768), timeout=45,
                             trace_path=output/f'{index:04d}-calls.jsonl')
            started = time.monotonic()
            with tempfile.TemporaryDirectory(prefix='grounding-policy-') as directory:
                ctx = {'id':row['id'],'split':split,'path':path,'instruction':row['instruction'],
                       'image':image,'pool':pool,'scratch':Path(directory),'cached_records':[],
                       'baseline_produced':{}}
                try:
                    kind = job['policy']
                    if kind == 'direct':
                        if job.get('space') or job.get('bbox'):
                            prediction = policies.direct(ctx,job.get('alias','qwen'),
                                                         space=job.get('space'),bbox=job.get('bbox',False))
                        else:
                            prediction = self.baseline(ctx,job.get('alias','qwen'))
                    elif kind == 'refine':
                        first = self.baseline(ctx,'qwen')
                        prediction = policies.repeat_refine(ctx,first,mode=job.get('mode','full'),
                                                            fraction=job.get('fraction',0.3))
                        prediction['first_point'] = first['point']
                    elif kind == 'select':
                        candidates = {alias:self.baseline(ctx,alias) for alias in ['qwen','ministral','lfm']}
                        prediction = policies.select_candidate(ctx,candidates,judge=job.get('judge','qwen'),
                                                               annotated=job.get('annotated',True))
                        prediction['candidates'] = {a:p['point'] for a,p in candidates.items()}
                    elif kind == 'agent':
                        prediction = policies.agent(ctx,controller=job.get('controller','qwen'),
                                                    python=job.get('python',True),
                                                    force_python=job.get('force_python',False),
                                                    max_steps=job.get('max_steps',5))
                    else:
                        raise ValueError(f'Unknown policy {kind}')
                    correct = score_point(prediction.get('point'),row['bbox_xyxy'],*image.size)
                    result = {'prediction':prediction,'correct':correct}
                except Exception as error:
                    result = {'correct':False,'error':f'{type(error).__name__}: {str(error)[:1200]}'}
                for alias,value in ctx['baseline_produced'].items():
                    produced[alias][row['id']] = value
                records = list(ctx['cached_records']) + pool.records
                result.update(id=row['id'],image=row['img_filename'],bbox_xyxy=row['bbox_xyxy'],
                              width=image.width,height=image.height,instruction=row['instruction'],
                              ui_type=row.get('ui_type'),application=row.get('application'),
                              platform=row.get('platform'),group=row.get('group'),
                              new_model_calls=pool.calls,cost=record_cost(records),
                              wall_seconds=time.monotonic()-started)
                box=row['bbox_xyxy']
                relative=(box[2]-box[0])*(box[3]-box[1])/(image.width*image.height)
                result['target_size']='tiny' if relative<0.0005 else ('small' if relative<0.0025 else 'larger')
                results.append(result)
                with (output/'predictions.jsonl').open('a') as stream:
                    stream.write(json.dumps(result)+'\n')
            if (index+1)%16==0 or index+1==len(rows):
                progress={'job':name,'done':index+1,'total':len(rows),
                          'correct':sum(r['correct'] for r in results),
                          'elapsed_seconds':time.monotonic()-start}
                print(json.dumps(progress),flush=True)
                self.state[f'{self.run_id}:progress:{name}']=progress
        for alias,values in produced.items():
            key=f'{self.run_id}:baseline:{split}:{alias}'
            previous=self.state.get(key,{})
            previous.update(values)
            self.state[key]=previous
            self.memo[key]=previous
        summary={'name':name,'job':job,'complete':len(results)==len(rows),'evaluated':len(results),
                 'planned':len(rows),'elapsed_seconds':time.monotonic()-start,
                 'new_model_calls':sum(r['new_model_calls'] for r in results),
                 'metrics':aggregate(results)}
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        self.state[f'{self.run_id}:result:{name}']=summary
        return summary

