"""One-shot by default; emits metrics/recommended action, never stops services."""
import argparse,datetime,json,os,time
from pathlib import Path


def recommendation(mem_available_mib,cpu_percent,disk_free_gib,oom_delta):
    reasons=[]
    if mem_available_mib<256:reasons.append('HOST_MEMORY_HEADROOM_LOW')
    if cpu_percent>75:reasons.append('HOST_CPU_BUSY')
    if disk_free_gib<5:reasons.append('HOST_DISK_HEADROOM_LOW')
    if oom_delta>0:reasons.append('NEW_OOM_KILL_OBSERVED')
    return {'stop_shadow_recommended':bool(reasons),'reasons':reasons,'automatic_production_action':False}


def read():
    mem={k:int(v.split()[0]) for line in Path('/proc/meminfo').read_text().splitlines() for k,v in [line.split(':',1)]}
    cpu=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    vm=dict(x.split() for x in Path('/proc/vmstat').read_text().splitlines());s=os.statvfs('/')
    return {'mem_available_mib':mem['MemAvailable']/1024,'cpu':cpu,'oom':int(vm.get('oom_kill',0)),'disk_free_gib':s.f_bavail*s.f_frsize/1024**3}


def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=int,default=1);p.add_argument('--interval',type=float,default=30);a=p.parse_args()
    if not 1<=a.samples<=120 or not 1<=a.interval<=60:raise SystemExit('Use 1..120 bounded samples, interval1..60s')
    old=read()
    for i in range(a.samples):
        time.sleep(1 if i==0 else a.interval);now=read();delta=sum(now['cpu'])-sum(old['cpu'])
        idle=now['cpu'][3]+now['cpu'][4]-old['cpu'][3]-old['cpu'][4];busy=100*(1-idle/delta) if delta else 0
        result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cpu_busy_percent':busy,
          'available_mib':now['mem_available_mib'],'disk_free_gib':now['disk_free_gib'],
          **recommendation(now['mem_available_mib'],busy,now['disk_free_gib'],now['oom']-old['oom'])}
        print(json.dumps(result),flush=True);old=now


if __name__=='__main__':main()
