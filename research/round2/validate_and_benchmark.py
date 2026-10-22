import collections,csv,hashlib,importlib.metadata,json,os,sys,time
from pathlib import Path
import joblib,numpy as np
from sklearn.metrics import cohen_kappa_score
ROOT=Path(os.environ.get('WENHENG_WORKSPACE', str(Path.cwd() / 'workspace'))).resolve();OUT=ROOT/'reports/round2-audit-cv-v1'
CKPT=ROOT/'checkpoints/round2-audit-cv-v1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'wenheng_score'))
from features import structural,Encoder

def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
config=json.loads((OUT/'cv_preregistration.json').read_text());results=json.loads((OUT/'cv_results.json').read_text())
rows=[json.loads(line) for line in (ROOT/'data/processed/cfms34-v1/train.jsonl').read_text().splitlines()]
ids=[r['id'] for r in rows];lookup={r['id']:r for r in rows}
import unicodedata
norm=lambda t:''.join(c for c in unicodedata.normalize('NFKC',t) if c.isalnum()).lower()
held_out=[]
for fold in config['folds']:
    a,b=set(fold['fit_ids']),set(fold['validation_ids'])
    assert not a&b and a|b==set(ids)
    assert not {norm(lookup[i]['title']) for i in a}&{norm(lookup[i]['title']) for i in b}
    held_out.extend(b)
assert collections.Counter(held_out)==collections.Counter(ids)
with (OUT/'cv_oof_predictions.csv').open(newline='') as f:records=list(csv.DictReader(f))
assert [r['id'] for r in records]==ids
y=np.array([r['score'] for r in rows]);pred={name:np.array([float(r[name]) for r in records]) for name in config['families']}
for name,p in pred.items():
    c=np.clip(p,0,10);q=cohen_kappa_score(np.rint(y).astype(int),np.rint(c).astype(int),labels=list(range(11)),weights='quadratic')
    assert abs(np.mean(np.abs(y-c))-results['families'][name]['pooled_oof']['mae'])<1e-12
    assert abs(q-results['families'][name]['pooled_oof']['qwk'])<1e-12

# Conditional, title-clustered bootstrap on existing OOF predictions: exploratory only.
groups=collections.defaultdict(list)
for i,row in enumerate(rows):groups[norm(row['title'])].append(i)
clusters=list(groups.values());rng=np.random.default_rng(config['seed']);deltas={name:[] for name in ['late_fusion','shared_multitask']}
for _ in range(1000):
    chosen=rng.integers(0,len(clusters),len(clusters));indices=np.concatenate([clusters[i] for i in chosen])
    baseline=np.abs(np.clip(pred['lexical'][indices],0,10)-y[indices])
    for name in deltas:deltas[name].append(float(np.mean(np.abs(np.clip(pred[name][indices],0,10)-y[indices])-baseline)))
comparison={name:{'mae_difference_vs_lexical':results['families'][name]['pooled_oof']['mae']-results['families']['lexical']['pooled_oof']['mae'],
                  'conditional_clustered_bootstrap95':[float(v) for v in np.percentile(values,[2.5,97.5])],
                  'caveat':'conditional on these OOF predictions; correlated CV training and family selection not accounted for; not new test evidence'} for name,values in deltas.items()}
save('paired_oof_comparison.json',comparison)

# Verify exact reduced-rank ridge algebra against an augmented least-squares SVD.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'round2'))
from cv_research import rrrfit,rrrpredict
x=rng.normal(size=(80,12));targets=rng.normal(size=(80,6));alpha=10.;rank=3
rr=rrrfit(x,targets,alpha,rank);xs=rr['scaler'].transform(x);yc=targets-targets.mean(0)
augx=np.vstack([xs,np.sqrt(alpha)*np.eye(xs.shape[1])]);augy=np.vstack([yc,np.zeros((xs.shape[1],yc.shape[1]))])
coef=np.linalg.lstsq(augx,augy,rcond=None)[0];_,_,vt=np.linalg.svd(augx@coef,full_matrices=False)
expected=coef@vt[:rank].T@vt[:rank]
assert np.allclose(rr['coef'],expected,atol=1e-9)
assert np.linalg.matrix_rank(rr['coef'],tol=1e-8)<=rank

# All cost measurements here use training texts only, with GPU hidden.
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch,psutil
torch.set_num_threads(2)
assert not torch.cuda.is_available()
model=joblib.load(CKPT/'outer_fold_0.joblib')
start=time.perf_counter();enc=Encoder(ROOT/'checkpoints/pretrained/bge-small-zh-v1.5',device='cpu',batch_size=1);load_seconds=time.perf_counter()-start
texts=[r['text'] for r in rows[:20]]
enc.encode(texts[:1]);latencies=[]
for text in texts:
    start=time.perf_counter();e,_=enc.encode([text]);dense=np.concatenate([e,structural([text])],axis=1)
    lexical=model['lexical'].predict(model['lexical_vectorizer'].transform([text]))
    sem=model['semantic_structure']['regressor'].predict(model['semantic_structure']['scaler'].transform(dense))
    p=(1-model['late_fusion_semantic_weight'])*lexical+model['late_fusion_semantic_weight']*sem
    assert np.isfinite(p).all()
    latencies.append(time.perf_counter()-start)
benchmark={'scope':'CPU fusion research prototype, outer-fold-0 models, 20 original-training essays',
           'gpu_hidden':True,'cuda_available':False,'load_seconds':load_seconds,
           'single_essay_warm_ms_p50':float(np.percentile(latencies,50)*1000),
           'single_essay_warm_ms_p95':float(np.percentile(latencies,95)*1000),
           'process_rss_bytes':psutil.Process().memory_info().rss,
           'threads':2,'caveat':'DSW CPU only, small sample, includes pretrained libraries; not production concurrency or independent scoring evaluation'}
save('fusion_cpu_benchmark.json',benchmark)
integrity=json.loads((OUT/'v1_integrity_before.json').read_text())
assert all(sha(ROOT/item['path'])==item['expected'] for item in integrity['files'])
validation={'passed':True,'v1_74_files_still_unchanged':True,'outer_validation_exactly_once':True,
            'title_groups_disjoint':True,'oof_metrics_recomputed':True,'reduced_rank_shared_objective_verified':True,
            'no_new_test_evaluation':True,'source_hashes_post_run':{p.name:sha(p) for p in (ROOT/'src/round2').glob('*.py')},
            'package_versions_post_run':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','joblib','torch','transformers']},
            'advance_gate_passed':bool(results['families']['lexical']['pooled_oof']['mae']-results['families']['late_fusion']['pooled_oof']['mae']>=.05)}
save('validation.json',validation)
print(json.dumps({'validation':validation,'paired_comparison':comparison,'cpu_benchmark':benchmark},ensure_ascii=False,indent=2))
