"""Bounded shadow-only dispatch. Does not accept or mutate the primary result.

Permission objects must come from a trusted authorization registry/curator,
never from student-supplied request booleans. This is an engineering gate,
not independent verification of consent or anonymization.
"""
import collections,concurrent.futures,datetime,json,math,os,random,threading,time,uuid
from dataclasses import dataclass
from pathlib import Path

VERSION='wenheng-score-v1-20261022'
RUBRIC='CFMS34-holistic-sum-0to10'
WEIGHTS_SHA='c28c5010ae4f1170aba860ecbe9f2393f5a723311c59f89db5ebbc2f773193cb'
PREPROCESS='v1-nfkc-redact-email-mobile-1'


@dataclass(frozen=True)
class Permission:
    purpose: str
    verified: bool
    expires_at: float
    withdrawn: bool = False
    minors_or_unknown_age: bool = True
    guardian_verified: bool = False
    deidentification_verified: bool = False


class Journal:
    """Only allowlisted events, capped in bytes; no raw request/result logging."""
    def __init__(self,path,max_bytes=10*1024*1024):
        self.path=Path(path);self.max_bytes=max_bytes;self.lock=threading.Lock()

    def append(self,event):
        allowed={'event_id','utc','status','native_score','model_version','rubric_id','weight','teacher_review_required'}
        if not set(event)<=allowed:raise ValueError('Unexpected event fields')
        data=(json.dumps(event,allow_nan=False,separators=(',',':'))+'\n').encode()
        with self.lock:
            self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            flags=os.O_WRONLY|os.O_APPEND|os.O_CREAT|getattr(os,'O_NOFOLLOW',0)
            fd=os.open(self.path,flags,0o600)
            try:
                if os.fstat(fd).st_size+len(data)>self.max_bytes:raise OSError('SHADOW_JOURNAL_CAPACITY')
                os.write(fd,data)
            finally:os.close(fd)


class ShadowDispatcher:
    def __init__(self,policy,score_call,journal,*,rng=random.random,clock=time.monotonic,wall=time.time):
        if policy.get('aggregation_weight')!=0 or policy.get('automatic_risk_routing_enabled') is not False:
            raise ValueError('Unsafe shadow policy')
        self.policy=dict(policy);self.call=score_call;self.journal=journal;self.rng=rng;self.clock=clock;self.wall=wall
        if not 0<=self.policy['sample_rate']<=1 or self.policy['max_pending'] not in [1,2]:raise ValueError('Invalid bounded policy')
        self.lock=threading.Lock();self.slots=threading.BoundedSemaphore(self.policy['max_pending'])
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='isolated-shadow')
        self.hour=collections.deque();self.failures=0;self.open_until=0;self.counts=collections.Counter();self.futures=set()

    def submit(self,payload,permission):
        # Nonblocking admission: no request waits for a score or a free queue slot.
        with self.lock:
            now=self.clock();stamp=self.wall()
            if not self.policy['enabled']:return 'disabled'
            if not isinstance(permission,Permission) or permission.verified is not True or permission.withdrawn is not False:
                return 'authorization_required'
            if not isinstance(permission.expires_at,(int,float)) or not math.isfinite(permission.expires_at) or permission.expires_at<=stamp:
                return 'authorization_required'
            if permission.purpose!='isolated_shadow_evaluation' or permission.deidentification_verified is not True:
                return 'authorization_required'
            if permission.minors_or_unknown_age is not False and permission.guardian_verified is not True:return 'authorization_required'
            if now<self.open_until:return 'circuit_open'
            if self.rng()>=self.policy['sample_rate']:return 'not_sampled'
            while self.hour and now-self.hour[0]>=3600:self.hour.popleft()
            if len(self.hour)>=self.policy['max_per_hour']:return 'hourly_cap'
            if not isinstance(payload,dict) or set(payload)-{'text','writing_prompt'}:return 'invalid_payload'
            text=payload.get('text');prompt=payload.get('writing_prompt')
            if not isinstance(text,str) or not 20<=len(text)<=20000 or not isinstance(prompt,str) or not prompt.strip() or len(prompt)>1000:
                return 'invalid_payload'
            if not self.slots.acquire(blocking=False):return 'queue_full'
            self.hour.append(now)
        request={'text':text,'writing_prompt':prompt,'authorization_scope':'authorized_research','text_verified':True,
                 'rubric_id':RUBRIC,'model_version':VERSION,'include_review':False,'deadline_ms':200}
        event_id=str(uuid.uuid4())
        try:future=self.pool.submit(self._work,request,event_id)
        except RuntimeError:self.slots.release();return 'worker_unavailable'
        with self.lock:self.futures.add(future)
        future.add_done_callback(self._finished)
        return 'scheduled'

    def _work(self,request,event_id):
        event={'event_id':event_id,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'status':'unavailable','weight':0,'teacher_review_required':True}
        failed=True
        try:
            result=self.call(request);model=result.get('model',{});prediction=result.get('prediction',{})
            score=prediction.get('overall_score')
            if result.get('contract_version')!='openessayscore/1.0' or model.get('version')!=VERSION or model.get('rubric_id')!=RUBRIC or model.get('score_range')!=[0,10]:
                raise ValueError('Incompatible model contract')
            if model.get('weights_sha256')!=WEIGHTS_SHA or model.get('preprocessing_version')!=PREPROCESS:raise ValueError('Artifact mismatch')
            if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=10:
                raise ValueError('Invalid native score')
            if result.get('review',{}).get('experimental_risk_routing_enabled') is not False or prediction.get('confidence') is not None:
                raise ValueError('Unsupported confidence/risk capability')
            event.update(status='available',native_score=score,model_version=VERSION,rubric_id=RUBRIC);failed=False
        except Exception:
            # Deliberately omit exception text: third-party exceptions may contain inputs.
            pass
        try:self.journal.append(event)
        except (OSError,ValueError):failed=True
        with self.lock:
            self.counts['failed' if failed else 'completed']+=1
            self.failures=self.failures+1 if failed else 0
            if self.failures>=self.policy['failure_threshold']:self.open_until=self.clock()+self.policy['cooldown_seconds']

    def _finished(self,future):
        with self.lock:self.futures.discard(future)
        self.slots.release()

    def close(self,drain=True):self.pool.shutdown(wait=True,cancel_futures=not drain)
