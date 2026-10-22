import concurrent.futures,copy,hashlib,json,threading,time,os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from .registry import *
from .core import *
from .api import create_app
from .collaboration import *

import os
ROOT=Path(os.environ.get('OPENESSAYSCORE_WORKSPACE',str(Path.cwd()/'workspace')))
SOURCE=Path(os.environ.get('OES_V1_BUNDLE',str(ROOT/'models/v1')))
REVIEW=Path(os.environ.get('OES_REVIEW_BUNDLE',str(ROOT/'models/review_runtime')))
pytestmark=pytest.mark.skipif(not (SOURCE/'best_scorer.joblib').exists(),reason='Private model artifact required; not distributed publicly')
ROW={'text':'这是用于软件恢复与接口连通性检查的技术控制文本，不是学生作文，也没有教师分数。请勿将此文本用于证明模型准确率。','title':'Technical connectivity control'}
BASE={'text':ROW['text'],'authorization_scope':'public_research_dataset','writing_prompt':ROW['title'] or '原任务上下文待进一步核实',
      'text_verified':True,'request_id':'transport_test'}
@pytest.fixture(scope='module')
def store(tmp_path_factory):
    p=tmp_path_factory.mktemp('registry');register_v1(p,SOURCE);return p
@pytest.fixture
def service(store,tmp_path):return ScoreService(store,REVIEW,tmp_path/'journal')

def test_actual_model_parity(service):
    r=service.evaluate(BASE)
    import joblib
    expected=float(joblib.load(SOURCE/'best_scorer.joblib').predict(joblib.load(SOURCE/'lexical_vectorizer.joblib').transform([clean(BASE['text'])]))[0])
    assert r['prediction']['overall_score']==pytest.approx(expected,abs=1e-6)
    assert r['prediction']['dimension_scores'] is None and r['prediction']['confidence'] is None
    assert r['model']['capabilities']['teacher_level_verified'] is False
    assert 'text' not in json.dumps(r,ensure_ascii=False).replace('text_', '') # No raw-text key in response.

def test_review_optional_and_original_interval_is_null(service):
    first=service.evaluate(BASE);second=service.evaluate({**BASE,'include_review':True})
    assert first['prediction']['overall_score']==second['prediction']['overall_score']
    assert not first['review']['enabled'] and second['review']['enabled']
    assert second['review']['candidate_interval'] is None
    assert second['review']['teacher_review_required'] and not second['review']['experimental_risk_routing_enabled']

@pytest.mark.parametrize('patch,code',[
 ({'text':''},'INVALID_REQUEST'),({'text':'ASCII only'},'INSUFFICIENT_CHINESE_TEXT'),
 ({'text':None},'INVALID_REQUEST'),({'text':'字'*20001},'INVALID_REQUEST'),
 ({'text_verified':False},'TEXT_NOT_VERIFIED'),({'authorization_scope':'unknown'},'INVALID_REQUEST'),
 ({'grade_level':True},'INVALID_REQUEST'),({'writing_prompt':' '},'TASK_REQUIRED'),
 ({'rubric_id':'WH-P34-NARR-0.1-draft'},'RUBRIC_NOT_SUPPORTED'),
 ({'extra_field':'private'},'INVALID_REQUEST')])
def test_invalid_requests(service,patch,code):
    with pytest.raises(ServiceError) as exc:service.evaluate({**BASE,**patch})
    assert exc.value.code==code

def test_http_contract_and_error_redaction(service):
    with TestClient(create_app(service)) as c:
        assert c.get('/health/live').status_code==200 and c.get('/health/ready').status_code==200
        assert c.post('/v1/score',json=BASE).status_code==200
        assert c.post('/v1/score',content='not json',headers={'Content-Type':'application/json'}).status_code==400
        assert c.post('/v1/score',content='NaN',headers={'Content-Type':'application/json'}).status_code==400
        assert c.post('/v1/score',content='x').status_code==415
        assert c.post('/v1/score',content='x'*131073,headers={'Content-Type':'application/json'}).status_code==413
        r=c.post('/v1/score',json={**BASE,'text':'DO_NOT_ECHO_PRIVATE_TEXT','grade_level':'bad'})
        assert r.status_code==422 and 'DO_NOT_ECHO' not in r.text

def test_missing_model_liveness_vs_readiness(tmp_path):
    s=ScoreService(tmp_path/'missing')
    with TestClient(create_app(s)) as c:
        assert c.get('/health/live').status_code==200 and c.get('/health/ready').status_code==503
        assert c.post('/v1/score',json=BASE).status_code==503

def test_tampered_weights_fail_closed(tmp_path):
    register_v1(tmp_path,SOURCE)
    (tmp_path/'versions'/VERSION/'best_scorer.joblib').write_bytes(b'not a model')
    s=ScoreService(tmp_path)
    with pytest.raises(ServiceError) as exc:s.evaluate(BASE)
    assert exc.value.code=='MODEL_WEIGHT_HASH_MISMATCH'

def test_unknown_version_and_path_injection(service):
    for value in ['not-registered','../anything']:
        with pytest.raises(ServiceError) as exc:service.evaluate({**BASE,'model_version':value})
        assert exc.value.status==404

def test_memory_fault_recovery(service,monkeypatch):
    original=service._predict
    def fail(*args):raise MemoryError()
    monkeypatch.setattr(service,'_predict',fail)
    with pytest.raises(ServiceError) as exc:service.evaluate(BASE)
    assert exc.value.code=='RESOURCE_EXHAUSTED'
    monkeypatch.setattr(service,'_predict',original)
    assert service.evaluate(BASE)['prediction']['overall_score']>=0

def test_configured_memory_pressure(store):
    service=ScoreService(store,memory_budget_mb=1)
    assert not service.status()['ready']
    with pytest.raises(ServiceError) as exc:service.evaluate(BASE)
    assert exc.value.status==503

def test_concurrent_repeated_scores(service):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:responses=list(pool.map(lambda _:service.evaluate(BASE),range(32)))
    assert len({r['prediction']['overall_score'] for r in responses})==1
    assert len({r['evaluation_id'] for r in responses})==32

def test_overload_rejects_without_unbounded_queue(store,monkeypatch):
    s=ScoreService(store,max_inflight=1);entered=threading.Event();release=threading.Event();original=s._predict
    def waiting(*args):entered.set();assert release.wait(3);return original(*args)
    monkeypatch.setattr(s,'_predict',waiting)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(s.evaluate,BASE);assert entered.wait(2)
        try:
            with pytest.raises(ServiceError) as exc:s.evaluate(BASE)
            assert exc.value.status==429
        finally:release.set()
        assert future.result()['prediction']['overall_score']>=0

def test_deadline_and_review_failure(service,monkeypatch):
    original=service._predict
    def slow(*args):time.sleep(.025);return original(*args)
    monkeypatch.setattr(service,'_predict',slow)
    with pytest.raises(ServiceError) as exc:service.evaluate({**BASE,'deadline_ms':10})
    assert exc.value.status==504
    monkeypatch.setattr(service,'_predict',original)
    service.review=None;service.review_error='injected'
    r=service.evaluate({**BASE,'include_review':True})
    assert 'REVIEW_MODULE_UNAVAILABLE' in r['review']['reasons'] and r['prediction']['overall_score']>=0

def test_versions_rollback_and_immutable_history(tmp_path):
    store=tmp_path/'models';register_v1(store,SOURCE)
    s=ScoreService(store,journal=tmp_path/'history');r1=s.evaluate(BASE)
    artifact=tmp_path/'history'/(r1['evaluation_id']+'.json');before=digest(artifact)
    duplicate='engineering-copy-same-v1';register_v1(store,SOURCE,duplicate,engineering_fixture=True)
    s.registry.activate(duplicate);r2=s.evaluate(BASE)
    s.registry.activate(VERSION);r3=s.evaluate(BASE)
    assert r2['model']['version']==duplicate and r3['model']['version']==VERSION
    assert digest(artifact)==before and json.loads(artifact.read_text())['model']['version']==VERSION
    assert r1['prediction']['overall_score']==r2['prediction']['overall_score']==r3['prediction']['overall_score']
    assert len(list((tmp_path/'history').glob('*.json')))==3

def peer_fixture(total=72):
    # Pure transport fixture, not a teacher assessment or an accuracy experiment.
    return {'fixture':'transport_only_not_human','scoring_standard_version':'example-four-traits-100-v1','result':{
      'overall_score':total,'summary':'接口测试占位，不是教师评语',
      'dimension_scores':[{'dimension':d,'score':total/4,'max_score':25,'rationale':'接口字段校验，不是作文评价','evidence':[]} for d in sorted(DIMENSIONS)],
      'annotations':[],'improvement_suggestions':[]}}

def test_offline_contract_scale_and_failure_fallback(service):
    final={'overall_score':72,'version':'transport_fixture'};snapshot=copy.deepcopy(final)
    calls={name:(lambda:peer_fixture()) for name in PEERS}
    r=integrate_offline(lambda:service.evaluate(BASE),calls,final)
    assert r['native_sidecar']['aggregation_weight']==0 and r['new_combined_score'] is None
    assert r['legacy_final']==snapshot and final==snapshot
    with pytest.raises(MappingUnavailable):map_native_score(7.2,100)
    def timeout():raise TimeoutError()
    calls['provider_b']=timeout;calls['provider_c']=lambda:peer_fixture(99)
    r=integrate_offline(lambda:service.evaluate(BASE),calls,final)
    assert 'provider_b:TIMEOUT' in r['issues'] and 'PEER_SCORE_DISAGREEMENT_REQUIRES_REVIEW' in r['issues']
    def failure():raise MemoryError()
    r=integrate_offline(failure,calls,final)
    assert r['native_sidecar']['status']=='unavailable' and r['legacy_final']==snapshot
    r=integrate_offline(lambda:service.evaluate(BASE),{name:timeout for name in PEERS},None)
    assert r['new_combined_score'] is None and 'NO_COMPARABLE_PROVIDER_RESULTS' in r['issues']
    assert r['paid_api_calls']==0

def test_peer_schema_does_not_accept_missing_dimensions():
    e=peer_fixture();e['result']['dimension_scores'][0].pop('rationale')
    assert not validate_peer(e)

def test_original_preprocessing_parity_in_separate_process(service):
    import subprocess,sys,os
    texts=[BASE['text'],BASE['text']+'\n联络占位 test@example.com 13800000000 ＡＢＣ']
    # Numeric/control strings are preprocessing fixtures, not fabricated teacher scores.
    code='import sys,json;sys.path.insert(0,sys.argv[1]);from predict import Scorer;print(json.dumps(Scorer(sys.argv[1]).predict(json.load(sys.stdin)).tolist()))'
    child=subprocess.run([sys.executable,'-c',code,str(SOURCE)],input=json.dumps(texts),text=True,capture_output=True,timeout=15)
    assert child.returncode==0
    expected=json.loads(child.stdout)
    actual=[service.evaluate({**BASE,'text':text})['prediction']['overall_score'] for text in texts]
    assert np_allclose(actual,expected)

def np_allclose(a,b):return all(abs(x-y)<1e-6 for x,y in zip(a,b))

def test_client_network_guard_and_timeout():
    from .client import ScoreClient,ScoreClientError
    for host in ['https://api.provider_a.com','http://example.com','file:///tmp/x']:
        with pytest.raises(ValueError):ScoreClient(host)
    # Only an owned loopback socket is used for the timeout path.
    import http.server
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):time.sleep(.1);self.send_response(200);self.end_headers()
        def log_message(self,*args):pass
    server=http.server.HTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=server.handle_request,daemon=True);worker.start()
    try:
        with pytest.raises(ScoreClientError) as exc:ScoreClient(f'http://127.0.0.1:{server.server_port}',timeout=.02).score(BASE)
        assert exc.value.code=='LOCAL_SCORE_SERVICE_UNAVAILABLE'
    finally:worker.join(timeout=1);server.server_close()

def test_http_admission_is_bounded_before_worker_queue(store,monkeypatch):
    s=ScoreService(store);entered=threading.Event();release=threading.Event();original=s._predict
    def waiting(*args):entered.set();assert release.wait(3);return original(*args)
    monkeypatch.setattr(s,'_predict',waiting)
    with TestClient(create_app(s,max_http_inflight=1)) as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(client.post,'/v1/score',json=BASE);assert entered.wait(2)
            try:assert client.post('/v1/score',json=BASE).status_code==429
            finally:release.set()
            assert future.result().status_code==200

def test_inflight_request_keeps_original_release_during_activation(tmp_path,monkeypatch):
    root=tmp_path/'models';register_v1(root,SOURCE);register_v1(root,SOURCE,'same-v1-engineering-fixture',engineering_fixture=True)
    s=ScoreService(root);original=s._predict;entered=threading.Event();release=threading.Event()
    def hold(*args):entered.set();assert release.wait(3);return original(*args)
    monkeypatch.setattr(s,'_predict',hold)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(s.evaluate,BASE);assert entered.wait(2)
        try:s.registry.activate('same-v1-engineering-fixture')
        finally:release.set()
        old=future.result()
    monkeypatch.setattr(s,'_predict',original)
    new=s.evaluate(BASE)
    assert old['model']['version']==VERSION and new['model']['version']=='same-v1-engineering-fixture'

def test_failed_activation_keeps_active_pointer(tmp_path):
    register_v1(tmp_path,SOURCE);before=(tmp_path/'registry.json').read_bytes()
    with pytest.raises(ModelLoadError):ModelRegistry(tmp_path).activate('unknown-release')
    assert (tmp_path/'registry.json').read_bytes()==before
