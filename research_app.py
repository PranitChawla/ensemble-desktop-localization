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
    from grounding.study_data import prepare_study
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
    zero=[{'name':f'direct-{alias}-dev','policy':'direct','alias':alias,'limit':128}
          for alias in ['qwen','ministral','lfm']]
    zero += [{'name':f'format-{alias}-{space}-dev','policy':'direct','alias':alias,'space':space,'limit':48}
             for alias,spaces in [('ministral',['norm1000','norm01']),('lfm',['norm1000','pixels'])]
             for space in spaces]
    zero += [{'name':'repeat-full-dev','policy':'refine','mode':'full','limit':128},
             {'name':'python-detail-30-dev','policy':'refine','mode':'detail','fraction':0.3,'limit':128},
             {'name':'python-detail-50-dev','policy':'refine','mode':'detail','fraction':0.5,'limit':128},
             {'name':'direct-qwen-4mp-dev','policy':'direct','alias':'qwen','request_pixels':4194304,'limit':128},
             {'name':'direct-qwen-box-dev','policy':'direct','alias':'qwen','bbox':True,'limit':128},
             {'name':'python-detail-context-dev','policy':'refine','mode':'detail-context','fraction':0.4,
              'fallback':True,'limit':128}]
    one=[]
    for controller in ['qwen','ministral']:
        for mode in ['no-python','optional-python','required-python']:
            one.append({'name':f'agent-{controller}-{mode}-pilot','policy':'agent','controller':controller,
                        'python':mode!='no-python','force_python':mode=='required-python','limit':32,
                        'max_calls':4,'max_steps':3,'max_output_tokens':256,'structured':True})
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
            for alias in ['qwen','ministral','lfm']:
                value=state.get(f'{resume_baselines_from}:baseline:{split}:{alias}',{})
                if value:
                    state[f'{run_id}:baseline:{split}:{alias}']=value
    (ROOT/'outputs').mkdir(exist_ok=True)
    (ROOT/'outputs'/f'{run_id}-launch.json').write_text(json.dumps({'run_id':run_id,'max_h100_hours':3.0})+'\n')
    print(f'STUDY_RUN_ID={run_id}',flush=True)
    calls=[worker.spawn(run_id,index,active_seconds) for index in range(2)]
    reports=[call.get() for call in calls]
    summary={'run_id':run_id,'workers':reports,
             'measured_h100_hours':sum(r['measured_h100_hours'] for r in reports)}
    (ROOT/'outputs'/f'{run_id}-compute.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

