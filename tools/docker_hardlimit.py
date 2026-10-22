"""Bounded, nonproduction Docker verification. Uses an existing image/model store.

No privileged containers, networking, system reconfiguration, production targets,
credential access, image pruning, or stable-model promotion. Only created test
containers and the new temporary model export are removed afterwards.
"""
import argparse,json,os,re,shutil,subprocess,tempfile,time,uuid
from pathlib import Path
MIB=1024**2
OWNER='openessayscore-cpu-verification'


def validate_limit(docker_memory,kernel_memory):
    if docker_memory!=512*MIB or kernel_memory!=512*MIB:raise ValueError('512MiB kernel enforcement not verified')


def cmd(args,timeout=40,check=True):
    p=subprocess.run(['docker',*args],capture_output=True,text=True,timeout=timeout)
    if check and p.returncode:raise RuntimeError('Docker command failed: '+args[0]+' (exit '+str(p.returncode)+')')
    return p


def inspect(container):
    fmt='{"state":{{json .State}},"memory":{{json .HostConfig.Memory}},"memory_swap":{{json .HostConfig.MemorySwap}},"restart_count":{{json .RestartCount}}}'
    # State contains no application env. Do not print host config or image env.
    return json.loads(cmd(['inspect','--format',fmt,container]).stdout)


def probe(container,*args,check=True):
    p=cmd(['exec',container,'python','/opt/container_probe.py',*args],timeout=60,check=False)
    try:d=json.loads(p.stdout)
    except ValueError:d={'probe_exit':p.returncode}
    if check and p.returncode:raise RuntimeError('Container probe failed (exit '+str(p.returncode)+')')
    return p.returncode,d


def main():
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--store',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nonproduction-confirmed',action='store_true');a=p.parse_args()
    if not a.nonproduction_confirmed:raise SystemExit('Explicit nonproduction confirmation required')
    if os.name!='posix':raise SystemExit('Run on Linux Docker host (including an approved WSL environment), not native Windows')
    if not re.fullmatch(r'(sha256:[a-f0-9]{64}|[A-Za-z0-9./:_-]+@sha256:[a-f0-9]{64})',a.image):raise SystemExit('Immutable local image ID/digest required')
    a.store=a.store.resolve()
    if not (a.store/'registry.json').is_file():raise SystemExit('A separately authorized, registered model store is required')
    if a.output.exists():raise SystemExit('Choose a new output path; historical results are immutable')
    cmd(['info','--format','{{.ServerVersion}}'])
    containers=[];result={'scope':'actual nonproduction Docker cgroup test, not accuracy evaluation','passed':False,'cold_starts':[],
      'image':a.image,'gpu_training_seconds':0,'paid_api_calls':0}
    with tempfile.TemporaryDirectory(prefix='oes-cpu-') as td:
        export=Path(td)/'models';shutil.copytree(a.store,export)
        uid=os.getuid() or 10001;gid=os.getgid() if os.getuid() else 10001
        # Only this newly created copy is changed; original model permissions remain intact.
        for f in [Path(td),export,*export.rglob('*')]:
            if f.is_symlink():raise ValueError('Model export symlinks forbidden')
            if os.getuid()==0:os.chown(f,uid,gid)
            f.chmod(0o700 if f.is_dir() else 0o600)
        def start(memory=512):
            name='oes-cpu-'+uuid.uuid4().hex[:12]
            args=['run','-d','--pull','never','--name',name,'--label',OWNER,'--network','none','--read-only','--restart','no',
              '--memory',str(memory)+'m','--memory-swap',str(memory)+'m','--cpus','1','--pids-limit','64',
              '--cap-drop','ALL','--security-opt','no-new-privileges','--user',str(uid)+':'+str(gid),
              '--tmpfs',f'/run/shadow:rw,noexec,nosuid,nodev,size=8m,uid={uid},gid={gid},mode=0700',
              '--tmpfs',f'/tmp:rw,noexec,nosuid,nodev,size=32m,uid={uid},gid={gid},mode=0700',
              '--mount','type=bind,src='+str(export)+',dst=/models,readonly',a.image]
            begin=time.perf_counter();container=cmd(args).stdout.strip();containers.append(container)
            ready=False;kernel=None
            for _ in range(100):
                state=inspect(container)
                if not state['state']['Running']:break
                code,d=probe(container,'--mode','health',check=False)
                if d.get('ready'):
                    ready=True;kernel=d.get('cgroup');break
                time.sleep(.2)
            return container,{'ready':ready,'cold_seconds':time.perf_counter()-begin,'kernel':kernel,'state':inspect(container)}
        try:
            for _ in range(3):
                container,cold=start();result['cold_starts'].append(cold)
                assert cold['ready'];validate_limit(cold['state']['memory'],cold['kernel']['limit_bytes'])
                assert cold['state']['memory_swap']==512*MIB
                if len(result['cold_starts'])<3:cmd(['stop','-t','3',container])
            result['loads']=[]
            for n,concurrency in [(100,1),(1000,1),(200,4),(200,8)]:
                _,d=probe(container,'--mode','load','--requests',str(n),'--concurrency',str(concurrency));result['loads'].append(d)
                assert d['control_scores_consistent'] and d['success']>0
                if concurrency==1:assert d['success']==n
            result['before_oom']=probe(container,'--mode','health')[1]
            code,allocation=probe(container,'--mode','allocate',check=False);result['oom_injection']={'exit':code,'probe':allocation}
            after=inspect(container);result['after_oom_state']=after
            _,health_after=probe(container,'--mode','health',check=False) if after['state']['Running'] else (1,{})
            before_oom=result['before_oom'].get('cgroup',{}).get('oom_kill')
            after_oom=health_after.get('cgroup',{}).get('oom_kill')
            result['kernel_oom_observed']=after['state']['OOMKilled'] or (before_oom is not None and after_oom is not None and after_oom>before_oom)
            assert result['kernel_oom_observed'] or allocation.get('python_memory_error')
            cmd(['stop','-t','3',container],check=False)
            recovered,recovery=start();result['recreated_512_recovery']=recovery;assert recovery['ready']
            _,recovered_scores=probe(recovered,'--mode','load','--requests','20');assert recovered_scores['success']==20
            cmd(['stop','-t','3',recovered])
            low,negative=start(128);result['negative_128m']=negative
            assert not negative['ready'],'Negative control unexpectedly ready; review actual footprint'
            result['passed']=True
        except Exception as e:result['failure_kind']=type(e).__name__;raise
        finally:
            # Remove only IDs returned by our own successful docker run calls.
            for container in containers:cmd(['rm','-f',container],check=False)
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'passed':result['passed'],'output':str(a.output)}))


if __name__=='__main__':main()
