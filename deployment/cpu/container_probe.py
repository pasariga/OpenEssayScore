"""Container-side technical workloads. No teacher labels or private text."""
import argparse,concurrent.futures,json,math,os,sys,time
from pathlib import Path
sys.path.insert(0,'/opt/shadow')
from unix_client import UnixConnection,UnixScoreClient

CONTROL='这是内存硬限制与并发行为验证使用的技术控制文本，不是学生作文，没有教师评分，不能作为准确率验证。'


def cgroup_memory():
    result={'version':None,'limit_bytes':None,'current_bytes':None,'peak_bytes':None,'oom_kill':None}
    v2=Path('/sys/fs/cgroup/memory.max')
    if v2.exists():
        result['version']=2
        for key,file in [('limit_bytes','memory.max'),('current_bytes','memory.current'),('peak_bytes','memory.peak')]:
            try:result[key]=int((v2.parent/file).read_text().strip())
            except (OSError,ValueError):pass
        try:events=dict(x.split() for x in (v2.parent/'memory.events').read_text().splitlines());result['oom_kill']=int(events.get('oom_kill',0))
        except OSError:pass
    else:
        root=Path('/sys/fs/cgroup/memory');result['version']=1 if root.exists() else None
        for key,file in [('limit_bytes','memory.limit_in_bytes'),('current_bytes','memory.usage_in_bytes'),('peak_bytes','memory.max_usage_in_bytes')]:
            try:result[key]=int((root/file).read_text().strip())
            except (OSError,ValueError):pass
        try:events=dict(x.split() for x in (root/'memory.oom_control').read_text().splitlines());result['oom_kill']=int(events.get('oom_kill',0))
        except OSError:pass
    return result


def request(sock):
    begin=time.perf_counter();c=UnixConnection(sock,timeout=2)
    try:
        payload={'text':CONTROL,'writing_prompt':'technical resource control','authorization_scope':'authorized_research','text_verified':True}
        c.request('POST','/v1/score',json.dumps(payload).encode(),{'Content-Type':'application/json'})
        r=c.getresponse();data=r.read(131073)
        if r.status!=200:return {'status':r.status}
        if len(data)>131072:raise ValueError('Oversize result')
        d=json.loads(data);score=d['prediction']['overall_score']
        assert isinstance(score,(int,float)) and not isinstance(score,bool) and math.isfinite(score) and 0<=score<=10
        assert d['prediction']['confidence'] is None and not d['review']['experimental_risk_routing_enabled']
        assert d['model']['score_range']==[0,10]
        return {'status':200,'score':score,'ms':(time.perf_counter()-begin)*1000}
    except Exception as e:return {'status':'transport_or_validation_failure','kind':type(e).__name__}
    finally:c.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['health','load','allocate'],default='health');p.add_argument('--socket',default='/run/shadow/score.sock')
    p.add_argument('--requests',type=int,default=100);p.add_argument('--concurrency',type=int,default=1);a=p.parse_args()
    if not 1<=a.requests<=2000 or not 1<=a.concurrency<=16:raise SystemExit('Bounded workload required')
    if a.mode=='health':
        try:ready=UnixScoreClient(a.socket,timeout=1).ready().get('ready') is True
        except Exception:ready=False
        print(json.dumps({'ready':ready,'cgroup':cgroup_memory()}));return 0 if ready else 2
    if a.mode=='allocate':
        # Called only in an already verified 512MiB container. Touch real pages.
        assert cgroup_memory()['limit_bytes']==512*1024**2
        blocks=[]
        try:
            for _ in range(128):
                b=bytearray(8*1024**2)
                for i in range(0,len(b),4096):b[i]=1
                blocks.append(b);time.sleep(.01)
        except MemoryError:
            print(json.dumps({'python_memory_error':True,'cgroup':cgroup_memory()}));return 3
        print(json.dumps({'unexpected_allocation_completed':True}));return 4
    before=cgroup_memory();begin=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:results=list(pool.map(lambda _:request(a.socket),range(a.requests)))
    codes={str(k):sum(r['status']==k for r in results) for k in {r['status'] for r in results}};ok=[r for r in results if r['status']==200]
    ms=sorted(r['ms'] for r in ok)
    print(json.dumps({'requests':a.requests,'concurrency':a.concurrency,'status_counts':codes,'success':len(ok),
      'p95_success_ms':ms[int(.95*(len(ms)-1))] if ms else None,'control_scores_consistent':len({r['score'] for r in ok})<=1,
      'elapsed_seconds':time.perf_counter()-begin,'cgroup_before':before,'cgroup_after':cgroup_memory(),
      'teacher_accuracy_evidence':False}))
    return 0 if ok else 2


if __name__=='__main__':raise SystemExit(main())
