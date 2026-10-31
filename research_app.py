"""Bounded two-worker study. Uses the caller's Modal CLI configuration."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

from modal_app import ROOT, base, cache, results

base = base.add_local_file(ROOT / 'modal_app.py', '/root/modal_app.py')
app=modal.App('ensemble-desktop-localization')
control=modal.Queue.from_name('ensemble-desktop-localization-study-control',create_if_missing=True)
state=modal.Dict.from_name('ensemble-desktop-localization-study-state',create_if_missing=True)
DATA_ROOT='/cache/screenspot-study-v1'


@app.function(image=base,cpu=8,memory=16384,timeout=1800,volumes={'/cache':cache})
def prepare_data():
    from huggingface_hub import snapshot_download

    from grounding.config import load_config
    from grounding.study_data import prepare_study
    for model in load_config('/root/models.json')['models']:
        snapshot_download(model['repo_id'],revision=model['revision'],
                          ignore_patterns=['*.md','.gitattributes','*.pth','*.pt','original/*'])
    metadata=prepare_study(DATA_ROOT,dev_count=128,validation_count=512)
    cache.commit()
    return metadata


@app.function(image=base,gpu='H100',cpu=8,memory=65536,timeout=5300,max_containers=2,
              single_use_containers=True,name='ensemble-desktop-localization',
              volumes={'/cache':cache,'/results':results})
def worker(run_id:str,index:int,active_seconds:int=5100):
    started=time.monotonic()
    import hashlib
    import importlib.metadata
    import traceback

    from grounding.config import load_config
    from grounding.servers import MemoryMonitor, ServerGroup, gpu_memory
    from grounding.study_runner import StudyRunner

    if not 1<=active_seconds<=5100:
        raise ValueError('Worker active time exceeds the aggregate three-H100-hour budget')
    cache.reload()
    output=Path('/results')/run_id/f'worker-{index}'
    output.mkdir(parents=True,exist_ok=True)
    config=load_config('/root/models.json')
    metadata={'run_id':run_id,'worker':index,'hardware':gpu_memory(),'config':config,
              'packages':{p:importlib.metadata.version(p) for p in ['vllm','transformers','Pillow','numpy','httpx']},
              'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in Path('/root/grounding').glob('study_*.py')},
              'active_seconds_limit':active_seconds,'jobs':[],'status':'running'}
    monitor=MemoryMonitor(output/'gpu-memory.jsonl').start()
    state[f'{run_id}:worker:{index}']={'status':'starting','started_unix':time.time()}
    try:
        with ServerGroup(config,output/'servers') as servers:
            metadata['startup_memory']=servers.snapshots
            runner=StudyRunner(config,DATA_ROOT,output,run_id,state,started+active_seconds)
            while time.monotonic()<started+active_seconds-120:
                state[f'{run_id}:worker:{index}']={'status':'ready','elapsed_seconds':time.monotonic()-started}
                try:
                    job=control.get(timeout=15,partition=f'{run_id}-{index}')
                except Exception as error:
                    if 'Empty' in type(error).__name__:
                        continue
                    raise
                if job is None:
                    continue
                if job.get('stop'):
                    break
                state[f'{run_id}:worker:{index}']={'status':'running','job':job['name'],
                                                  'elapsed_seconds':time.monotonic()-started}
                result=runner.run(job)
                metadata['jobs'].append(result['name'])
                metadata['elapsed_seconds']=time.monotonic()-started
                (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
                results.commit()
            metadata['status']='completed'
    except Exception:
        metadata['status']='failed'
        metadata['error']=traceback.format_exc()
        print(metadata['error'],flush=True)
    finally:
        monitor.close()
        metadata['memory_monitor']=monitor.summary()
        metadata['elapsed_seconds']=time.monotonic()-started
        metadata['measured_h100_hours']=metadata['elapsed_seconds']/3600
        (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        results.commit()
        state[f'{run_id}:worker:{index}']=metadata
    return metadata


def initial_jobs():
    # The optimization agent edits Python programs; no VLM emits executable Python.
    source=(ROOT/'candidates/self_consistency.py').read_text()
    zero=[{'name':'direct-mai-dev','policy':'direct','alias':'mai','limit':128},
          {'name':'direct-venus-dev','policy':'direct','alias':'venus','limit':128},
          {'name':'direct-venus-4mp-dev','policy':'direct','alias':'venus','request_pixels':4194304,'limit':128}]
    for name,mode,fraction,count in [('repeat-full','full',0.3,128),('python-detail-30','detail',0.3,128),
                                     ('python-detail-50','detail',0.5,64),
                                     ('python-detail-context','detail-context',0.4,64)]:
        zero.append({'name':name+'-venus-dev','policy':'refine','alias':'venus','mode':mode,
                     'fraction':fraction,'fallback':True,'limit':count})
    one=[{'name':'direct-jedi-dev','policy':'direct','alias':'jedi','limit':128}]
    for label,samplers in [('venus',['venus']*5),('mixed',['venus','jedi','venus','jedi','mai']),
                           ('mai',['mai']*5)]:
        one.append({'name':f'sampling-{label}-dev','policy':'custom',
                    'source':source.replace("SAMPLERS = ['venus'] * 5",f'SAMPLERS = {samplers!r}'),
                    'limit':64,'max_calls':6,'max_output_tokens':512})
    for annotated in [True,False]:
        one.append({'name':f'selector-specialists-{"annotated" if annotated else "text"}-dev',
                    'policy':'select','aliases':['mai','venus','jedi'],'judge':'mai',
                    'annotated':annotated,'limit':64})
    return [zero,one]


@app.local_entrypoint()
def main(prepare_only:bool=False,skip_prepare:bool=False,active_seconds:int=5100,
         prior_gpu_seconds:float=0,resume_baselines_from:str=''):
    if not skip_prepare:
        data=prepare_data.remote()
        print(json.dumps(data,indent=2))
    if prepare_only:
        return
    if not 1<=active_seconds<=5100 or 2*active_seconds+prior_gpu_seconds+200>10800:
        raise ValueError('Requested runs exceed three aggregate H100-hours including safety margin')
    run_id=datetime.now(timezone.utc).strftime('study-%Y%m%dT%H%M%SZ')
    state[f'{run_id}:plan']={'run_id':run_id,'max_h100_hours':3.0,'workers':2,
                           'worker_active_seconds':active_seconds,'prior_gpu_seconds':prior_gpu_seconds,
                           'initial_jobs':initial_jobs(),
                           'selection':'development only; validation starts after policy freeze'}
    for index,jobs in enumerate(initial_jobs()):
        for job in jobs:
            control.put(job,partition=f'{run_id}-{index}')
    if resume_baselines_from:
        for split in ['dev','validation']:
            for alias in ['mai','venus','jedi']:
                value=state.get(f'{resume_baselines_from}:baseline:{split}:{alias}',{})
                if value:
                    state[f'{run_id}:baseline:{split}:{alias}']=value
    (ROOT/'outputs').mkdir(exist_ok=True)
    (ROOT/'outputs'/f'{run_id}-launch.json').write_text(json.dumps({'run_id':run_id,'max_h100_hours':3.0})+'\n')
    print(f'STUDY_RUN_ID={run_id}',flush=True)
    calls=[worker.spawn(run_id,index,active_seconds) for index in range(2)]
    reports=[call.get() for call in calls]
    summary={'run_id':run_id,'workers':reports,
             'prior_gpu_seconds':prior_gpu_seconds,
             'measured_h100_hours':prior_gpu_seconds/3600+sum(r['measured_h100_hours'] for r in reports)}
    (ROOT/'outputs'/f'{run_id}-compute.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

