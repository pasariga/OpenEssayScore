import copy,hashlib,json,math,re,threading,time,unicodedata,uuid
from pathlib import Path
from pydantic import BaseModel,ConfigDict,Field,StrictBool,StrictInt,StrictStr,ValidationError
from typing import Literal
from .registry import ModelRegistry,ModelLoadError,RUBRIC,PREPROCESS,V1_WEIGHT_SHA

MAX_CHARS=20000
BUILD_SHA=hashlib.sha256(b''.join((Path(__file__).parent/name).read_bytes() for name in ['core.py','registry.py','api.py'])).hexdigest()
class ServiceError(Exception):
    def __init__(self,code,status=422):self.code=code;self.status=status;super().__init__(code)

class ScoreRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    text:StrictStr=Field(min_length=1,max_length=MAX_CHARS)
    authorization_scope:Literal['public_research_dataset','authorized_research']
    rubric_id:StrictStr=RUBRIC
    model_version:StrictStr|None=None
    request_id:StrictStr|None=Field(default=None,pattern='^[A-Za-z0-9_-]{1,64}$')
    writing_prompt:StrictStr=Field(min_length=1,max_length=1000)
    text_verified:StrictBool
    grade_level:StrictInt|None=Field(default=None,ge=1,le=6)
    include_review:StrictBool=False
    deadline_ms:StrictInt=Field(default=5000,ge=10,le=10000)

def clean(text):
    text=unicodedata.normalize('NFKC',text).strip()
    text=re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}','[EMAIL]',text)
    return re.sub(r'(?<!\d)1[3-9]\d{9}(?!\d)','[PHONE]',text)

class ScoreService:
    def __init__(self,store,review_bundle=None,journal=None,max_inflight=4,memory_budget_mb=None):
        self.registry=ModelRegistry(store);self.semaphore=threading.BoundedSemaphore(max_inflight)
        self.review_bundle=Path(review_bundle) if review_bundle else None;self.review=None;self.review_error=None
        self.journal=Path(journal) if journal else None;self.memory_budget_mb=memory_budget_mb
        self.model_error=None;self.metrics={'completed':0,'failed':0};self.metrics_lock=threading.Lock()
        t=time.perf_counter()
        try:self.registry.load()
        except MemoryError:self.model_error='RESOURCE_EXHAUSTED'
        except ModelLoadError as exc:self.model_error=str(exc)
        if self.review_bundle:
            try:
                from openessayscore.review.review_engine import ReviewEngine
                self.review=ReviewEngine(self.review_bundle)
            except Exception:self.review_error='REVIEW_MODULE_UNAVAILABLE'
        self.load_ms=(time.perf_counter()-t)*1000
    def status(self):
        try:
            loaded=self.registry.load();version=loaded['version'];digest=loaded['manifest_sha256'];ready=True
            if self.memory_budget_mb:
                import psutil
                ready=psutil.Process().memory_info().rss<=self.memory_budget_mb*1024**2
        except (ModelLoadError,MemoryError):version=None;digest=None;ready=False
        return {'live':True,'ready':ready,'active_model':version,'manifest_sha256':digest,
          'review_status':'experimental_guardrails_only' if self.review else ('unavailable' if self.review_error else 'disabled'),
          'model_status':'research_baseline','load_ms':self.load_ms,'counters':dict(self.metrics)}
    def _predict(self,loaded,text):return float(loaded['scorer'].predict(loaded['vectorizer'].transform([text]))[0])
    def evaluate(self,payload):
        t0=time.perf_counter()
        if not self.semaphore.acquire(blocking=False):raise ServiceError('SERVICE_BUSY',429)
        try:
            try:req=ScoreRequest.model_validate(payload)
            except ValidationError:raise ServiceError('INVALID_REQUEST',422)
            if req.rubric_id!=RUBRIC:raise ServiceError('RUBRIC_NOT_SUPPORTED',409)
            if not req.text_verified:raise ServiceError('TEXT_NOT_VERIFIED',422)
            if not req.writing_prompt.strip():raise ServiceError('TASK_REQUIRED',422)
            text=clean(req.text)
            if len(re.findall(r'[\u4e00-\u9fff]',text))<20:raise ServiceError('INSUFFICIENT_CHINESE_TEXT',422)
            if self.memory_budget_mb:
                import psutil
                if psutil.Process().memory_info().rss>self.memory_budget_mb*1024**2:raise MemoryError()
            try:loaded=self.registry.load(req.model_version)
            except ModelLoadError as exc:
                raise ServiceError(str(exc),404 if str(exc)=='MODEL_VERSION_NOT_REGISTERED' else 503)
            begin=time.perf_counter();score=self._predict(loaded,text);predict_ms=(time.perf_counter()-begin)*1000
            if not math.isfinite(score):raise ServiceError('INVALID_MODEL_RESULT',503)
            score=max(0.,min(10.,score))
            if (time.perf_counter()-t0)*1000>req.deadline_ms:raise ServiceError('DEADLINE_EXCEEDED',504)
            review={'enabled':False,'teacher_review_required':True,'confidence':None,'candidate_interval':None,
                    'reasons':['RESEARCH_MODEL_REQUIRES_TEACHER_REVIEW'],'experimental_risk_routing_enabled':False}
            if req.include_review:
                if self.review:
                    try:
                        result=self.review.assess({'text':text,'score':score,'rubric_id':RUBRIC,'score_range':[0,10],
                          'score_model_sha256':V1_WEIGHT_SHA,'authorization':req.authorization_scope,'prompt_present':True,
                          'ocr_verified':True,'domain':'external_service_request'})
                        review={**result,'enabled':True,'experimental_risk_routing_enabled':False}
                    except Exception:review['reasons'].append('REVIEW_MODULE_UNAVAILABLE')
                else:review['reasons'].append('REVIEW_MODULE_UNAVAILABLE')
            result={'contract_version':'openessayscore/1.0','evaluation_id':str(uuid.uuid4()),
              'service_build_sha256':BUILD_SHA,
              'request_id':req.request_id or uuid.uuid4().hex,'mode':'offline_research',
              'model':{'version':loaded['version'],'manifest_sha256':loaded['manifest_sha256'],
                'weights_sha256':V1_WEIGHT_SHA,'preprocessing_version':PREPROCESS,'rubric_id':RUBRIC,'score_range':[0,10],
                'capabilities':loaded['manifest']['capabilities']},
              'input':{'normalized_sha256':hashlib.sha256(text.encode()).hexdigest(),'characters':len(text)},
              'prediction':{'overall_score':score,'dimension_scores':None,'summary':None,'confidence':None},
              'review':review,'warnings':['NATIVE_CFMS_SCALE_NOT_CONVERTIBLE_WITHOUT_VALIDATED_MAPPING',
                'GRADE_AND_PROMPT_RELEVANCE_NOT_VALIDATED'],'timing_ms':{'predict':predict_ms,'total':(time.perf_counter()-t0)*1000}}
            if self.journal:
                self.journal.mkdir(parents=True,exist_ok=True)
                path=self.journal/(result['evaluation_id']+'.json')
                with path.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,allow_nan=False)
                path.chmod(0o600)
            with self.metrics_lock:self.metrics['completed']+=1
            return result
        except MemoryError:
            with self.metrics_lock:self.metrics['failed']+=1
            raise ServiceError('RESOURCE_EXHAUSTED',503)
        except ServiceError:
            with self.metrics_lock:self.metrics['failed']+=1
            raise
        except Exception:
            with self.metrics_lock:self.metrics['failed']+=1
            raise ServiceError('SCORING_UNAVAILABLE',503)
        finally:self.semaphore.release()
