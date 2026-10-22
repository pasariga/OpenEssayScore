"""Real loopback HTTP load, child-only resource probes, no external API traffic."""
import concurrent.futures,json,os,socket,subprocess,sys,threading,time,urllib.request,resource
from pathlib import Path
import numpy as np,psutil
from .client import ScoreClient,ScoreClientError

ROOT=Path(os.environ.get('OPENESSAYSCORE_WORKSPACE', str(Path.cwd() / 'workspace'))).resolve();OUT=ROOT/'reports/round5-service-v1'
STORE=ROOT/'services/score_service/model_store';REVIEW=ROOT/'checkpoints/WenHeng-Review-r4-prototype'

def http_json(url):
    with urllib.request.urlopen(url,timeout=1) as response:return json.load(response)
def stats(values):return {'p50_ms':float(np.percentile(values,50)),'p95_ms':float(np.percentile(values,95)),'max_ms':float(max(values))}

def main():
    rows=[json.loads(line) for line in (ROOT/'data/processed/cfms34-v1/train.jsonl').read_text().splitlines()][:100]
    payloads=[{'text':r['text'],'authorization_scope':'public_research_dataset','writing_prompt':r['title'] or '任务文本上下文',
       'text_verified':True,'request_id':'engineering_loopback','include_review':False} for r in rows]
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    allowed=sorted(os.sched_getaffinity(0));cpu=allowed[0]
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    command=[sys.executable,'-m','openessayscore.service.server','--store',str(STORE),'--review-bundle',str(REVIEW),'--port',str(port)]
    log=(OUT/'http_server.log').open('w')
    start=time.perf_counter()
    process=subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,stderr=log,preexec_fn=lambda:os.sched_setaffinity(0,{cpu}))
    sampler_stop=threading.Event();samples=[]
    def sample():
        p=psutil.Process(process.pid)
        while not sampler_stop.wait(.01):
            try:m=p.memory_info();samples.append((time.perf_counter(),m.rss,m.vms))
            except psutil.Error:return
    sampler=threading.Thread(target=sample,daemon=True);sampler.start()
    try:
        base=f'http://127.0.0.1:{port}'
        deadline=time.monotonic()+20
        while True:
            if process.poll() is not None:raise RuntimeError('Owned service exited during startup')
            try:
                ready=http_json(base+'/health/ready')
                if ready['ready']:break
            except Exception:pass
            if time.monotonic()>deadline:raise TimeoutError('Owned service readiness timeout')
            time.sleep(.02)
        cold=time.perf_counter()-start;client=ScoreClient(base,3.)
        for payload in payloads[:10]:client.score(payload)
        baseline_rss=psutil.Process(process.pid).memory_info().rss
        timings=[];scores=[]
        for payload in payloads:
            t=time.perf_counter();r=client.score(payload);timings.append((time.perf_counter()-t)*1000);scores.append(r['prediction']['overall_score'])
        concurrent_results=[]
        for workers in [1,4,8,16]:
            def call(i):
                t=time.perf_counter()
                try:
                    r=client.score(payloads[i%len(payloads)])
                    return {'ok':True,'ms':(time.perf_counter()-t)*1000,'score':r['prediction']['overall_score'],'index':i%len(payloads)}
                except ScoreClientError as exc:return {'ok':False,'code':exc.code,'status':exc.status,'ms':(time.perf_counter()-t)*1000}
            t=time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:responses=list(pool.map(call,range(200)))
            elapsed=time.perf_counter()-t;success=[r for r in responses if r['ok']]
            assert all(abs(r['score']-scores[r['index']])<1e-6 for r in success)
            errors={code:sum(not r['ok'] and r.get('code')==code for r in responses) for code in {r.get('code') for r in responses if not r['ok']}}
            concurrent_results.append({'concurrency':workers,'requests':200,'success':len(success),'errors':errors,
              'wall_seconds':elapsed,'successful_rps':len(success)/elapsed,'success_latency':stats([r['ms'] for r in success]) if success else None})
        rss_before_repeat=psutil.Process(process.pid).memory_info().rss;repeat=[]
        for _ in range(500):repeat.append(client.score(payloads[0])['prediction']['overall_score'])
        rss_after_repeat=psutil.Process(process.pid).memory_info().rss
        assert max(repeat)-min(repeat)<1e-12
        review_t=[]
        for payload in payloads[:30]:
            t=time.perf_counter();r=client.score({**payload,'include_review':True});review_t.append((time.perf_counter()-t)*1000)
            assert r['review']['enabled'] and r['review']['candidate_interval'] is None
        final_ready=http_json(base+'/health/ready');assert final_ready['ready']
        output={'scope':'real DSW loopback HTTP, owned server pinned to one allowed logical CPU; no production host contacted',
          'cold_start_to_ready_seconds':cold,'model_initialization_ms':ready['load_ms'],'sequential_100':stats(timings),
          'review_enabled_30':stats(review_t),'concurrency':concurrent_results,
          'repeat_500':{'identical_scores':True,'rss_before':rss_before_repeat,'rss_after':rss_after_repeat,'rss_delta':rss_after_repeat-rss_before_repeat},
          'server_rss_after_warmup':baseline_rss,'server_peak_sampled_rss':max(r[1] for r in samples),
          'server_peak_sampled_vms':max(r[2] for r in samples),'final_health':final_ready,
          'cpu_affinity_count':1,'gpu_used':False,'paid_api_calls':0,
          'limits':'short engineering stress, not a multi-day soak; loopback timings exclude cross-host network, reverse proxy, TLS and concurrent production workloads'}
        (OUT/'http_benchmark.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    finally:
        sampler_stop.set();sampler.join(timeout=2)
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        log.close()
    # Virtual address-space limits on disposable owned children, not host memory settings.
    probes=[]
    probe="""
import json,psutil
from openessayscore.service.core import ScoreService
s=ScoreService('services/score_service/model_store','checkpoints/WenHeng-Review-r4-prototype')
print(json.dumps({'ready':s.status()['ready'],'rss':psutil.Process().memory_info().rss,'vms':psutil.Process().memory_info().vms}))
"""
    for cap in [128,256,512,1024]:
        def limit(cap=cap):
            resource.setrlimit(resource.RLIMIT_AS,(cap*1024**2,cap*1024**2));os.sched_setaffinity(0,{cpu})
        t=time.perf_counter();p=subprocess.run([sys.executable,'-c',probe],env=env,cwd=ROOT,preexec_fn=limit,capture_output=True,text=True,timeout=20)
        try:parsed=json.loads(p.stdout) if p.returncode==0 else None
        except ValueError:parsed=None
        probes.append({'address_space_limit_mb':cap,'exit_code':p.returncode,'result':parsed,'wall_seconds':time.perf_counter()-t,
           'failure_summary':p.stderr[-300:] if p.returncode else None})
    (OUT/'resource_probes.json').write_text(json.dumps({'kind':'RLIMIT_AS virtual address space, not cgroup/RAM quota','probes':probes},indent=2),encoding='utf-8')
    print(json.dumps({'http':output,'resource_probes':probes,'owned_server_stopped':process.poll() is not None},indent=2))

if __name__=='__main__':main()
