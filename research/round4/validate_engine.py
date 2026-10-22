"""Functional guards, role isolation, cost and metric tests; no fabricated teacher data."""
import copy,csv,json,math,os,subprocess,sys,tempfile,time
from pathlib import Path
import joblib,numpy as np
from sklearn.metrics import roc_auc_score
from risk_core import *
from review_engine import ReviewEngine,V1_SHA
ROOT=Path(os.environ.get('WENHENG_WORKSPACE', str(Path.cwd() / 'workspace'))).resolve();RUN='round4-trust-review-v1'
OUT=ROOT/'reports'/RUN;CKPT=ROOT/'checkpoints'/RUN
fold=CKPT/'fold_0'
dump(fold/'engine_manifest.json',{'files':{name:sha(fold/name) for name in ['risk_bundle.joblib','metadata.json']}})
engine=ReviewEngine(fold)
example=json.loads((OUT/'internal_example_request.json').read_text())
cases=[]
def check(name,request,expected,reason=None):
    r=engine.assess(request)
    assert r['decision']==expected,(name,r)
    assert r['teacher_review_required'] and r['confidence'] is None and r['error_probability'] is None
    if reason:assert reason in r['reasons'] or reason in r['warnings'],(name,r)
    cases.append({'name':name,'passed':True})
check('real held-out internal feature request',example,'teacher_review')
check('wrong input type',[], 'reject_input','REQUEST_NOT_OBJECT')
for name,key,val,reason in [
 ('empty technical string','text','','MISSING_TEXT'),
 ('unverified rights','authorization','pending','AUTHORIZATION_NOT_DECLARED'),
 ('third-round scale mismatch','rubric_id','WH-P34-NARR-0.1-draft','RUBRIC_OR_SCALE_MISMATCH'),
 ('nonfinite score','score',float('nan'),'INVALID_SCORE'),
 ('out of range score','score',11,'INVALID_SCORE'),
 ('unknown model version','score_model_sha256','0'*64,'UNKNOWN_SCORE_MODEL_HASH'),
 ('no task','prompt_present',False,'MISSING_TASK'),
 ('OCR not verified','ocr_verified',False,'TEXT_OR_OCR_NOT_VERIFIED'),
 ('missing feature vector','features',None,'FEATURES_MISSING_INVALID_OR_MISMATCHED')]:
    req=copy.deepcopy(example);req[key]=val;check(name,req,'reject_input' if reason not in ['FEATURES_MISSING_INVALID_OR_MISMATCHED'] else 'teacher_review',reason)
# String fixtures test input controls only, not essay/grade data or labeled OOD samples.
for name,text,warning in [('non Chinese control','plain ASCII control string','INSUFFICIENT_CHINESE_OR_LANGUAGE_OUTSIDE_SCOPE'),
                          ('repetition control','字'*1000,'EXTREME_REPETITION')]:
    req=copy.deepcopy(example);req['text']=text;req['score_model_sha256']=V1_SHA
    check(name,req,'teacher_review',warning)
req=copy.deepcopy(example);req['score_model_sha256']=V1_SHA
r=engine.assess(req);assert r['candidate_interval'] is None and r['experimental_review_flag'] is None
cases.append({'name':'original v1 never receives replica calibration','passed':True})
req=copy.deepcopy(example);req['domain']='unseen-source'
r=engine.assess(req);assert r['candidate_interval'] is None and 'DOMAIN_NOT_EXTERNALLY_VALIDATED' in r['warnings']
cases.append({'name':'domain shift suppresses candidate interval','passed':True})
req=copy.deepcopy(example);req['features'][4]+=1
r=engine.assess(req);assert r['candidate_interval'] is None and 'FEATURES_MISSING_INVALID_OR_MISMATCHED' in r['reasons']
cases.append({'name':'score feature tampering fails closed','passed':True})
r=engine.assess(example,experimental_risk=True)
assert r['teacher_review_required'] and not r['research_risk_method_supported'] and r['confidence'] is None
cases.append({'name':'experimental switch cannot enable automated scoring','passed':True})
# Finite-sample quantile, including small-n failure and exact order statistic.
assert conformal_quantile(np.arange(1,11),.1)==10
try:conformal_quantile(np.arange(1,5),.1);raise AssertionError('small-n should fail')
except ValueError:pass
cases.append({'name':'finite sample conformal quantile boundaries','passed':True})
with tempfile.TemporaryDirectory(dir=OUT) as tmp:
    folder=Path(tmp)
    for name in ['risk_bundle.joblib','metadata.json','engine_manifest.json']:(folder/name).write_bytes((fold/name).read_bytes())
    (folder/'metadata.json').write_text('{}')
    try:ReviewEngine(folder);raise AssertionError('tampering should fail')
    except ValueError:pass
cases.append({'name':'artifact tamper rejected before model deserialization','passed':True})

prereg=json.loads((OUT/'preregistration.json').read_text())
parts=[set(p) for p in prereg['partitions']]
assert len(set.union(*parts))==751 and sum(map(len,parts))==751
for k in range(5):
    roles=[parts[k],parts[(k+1)%5],parts[(k+2)%5],set.union(*[parts[j] for j in range(5) if j not in [k,(k+1)%5,(k+2)%5]])]
    assert all(not roles[a]&roles[b] for a in range(4) for b in range(a+1,4))
with (OUT/'validation_predictions.csv').open(newline='') as f:records=list(csv.DictReader(f))
assert len(records)==751 and len({r['id'] for r in records})==751
y=np.array([float(r['human_score']) for r in records]);p=np.array([float(r['replica_score']) for r in records]);event=np.abs(p-y)>1
results=json.loads((OUT/'results.json').read_text())
for rule in prereg['rules']:
    review=np.array([int(r[rule+'_review20'])==1 for r in records]);rank=np.array([float(r[rule]) for r in records])
    stored=results['results'][rule]['0.2']
    assert np.isclose(review.mean(),stored['review_rate'])
    assert np.isclose(np.sum(review&event)/np.sum(event),stored['error_capture_rate'])
    assert np.isclose(roc_auc_score(event,rank),stored['auroc'])
cases.append({'name':'roles disjoint and all 751 validation metrics independently recomputed','passed':True})

# Cost of review only, existing precomputed features, no scoring or BGE extraction included.
lat=[]
for _ in range(5):engine.assess(example)
for _ in range(100):
    t=time.perf_counter();engine.assess(example);lat.append(time.perf_counter()-t)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
t=time.perf_counter();child=subprocess.run([sys.executable,str(Path(__file__).parent/'review_engine.py'),'--bundle-dir',str(fold),
    '--request',str(OUT/'internal_example_request.json')],capture_output=True,text=True,env=env,timeout=20)
assert child.returncode==0;cli=json.loads(child.stdout);assert cli['teacher_review_required']
dump(OUT/'cli_smoke_output.json',cli)
import psutil
benchmark={'warm_review_p50_ms':float(np.percentile(lat,50)*1000),'warm_review_p95_ms':float(np.percentile(lat,95)*1000),
 'fresh_process_cli_seconds':time.perf_counter()-t,'rss_bytes_validation_process':psutil.Process().memory_info().rss,
 'scope':'review engine on existing features; excludes score provider, semantic embedding and teacher work',
 'gpu_seconds':0}
dump(OUT/'engine_benchmark.json',benchmark)
preserved={path:sha(ROOT/path)==digest for path,digest in prereg['v1_baseline_hashes'].items()}
assert all(preserved.values())
dump(OUT/'tests.json',{'passed':True,'checks':cases,'count':len(cases),'original_v1_hashes_preserved':preserved,
 'synthetic_strings_are_control_fixtures_only':True,'real_new_teacher_labels':0})
print(json.dumps({'test_count':len(cases),'passed':True,'benchmark':benchmark},indent=2))

if __name__=='__main__':pass
