import os
import collections,csv,hashlib,json,sys,unicodedata
from pathlib import Path
import joblib,numpy as np
from sklearn.metrics import cohen_kappa_score
ROOT=Path(os.environ.get('WENHENG_WORKSPACE', str(Path.cwd() / 'workspace'))).resolve()
OUT=ROOT/'reports/round2-audit-cv-v1'
OUT.mkdir(parents=True,exist_ok=True)
REV='933489269eb9dbf5f6f00f8e8be6ba0ccebc3dbb'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'wenheng_score'))
from features import structural

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
prior=json.loads((ROOT/'backups/wenheng-score-v1-20261022T024733Z.manifest.json').read_text())
integrity=[]
for info in prior['files']:
    path=ROOT/info['path']
    integrity.append({'path':info['path'],'expected':info['sha256'],'actual':sha(path) if path.is_file() else None})
assert all(r['expected']==r['actual'] for r in integrity),'Original v1 artifacts changed'
save('v1_integrity_before.json',{'files':integrity,'all_unchanged':True})
raw={};processed={};stats={};trait_errors=[]
trait_ids=[]
with (ROOT/'data/raw/cfms34'/REV/'trait_map.csv').open(encoding='utf-8-sig',newline='') as f:
    trait_ids=[r['hifts_id'] for r in csv.DictReader(f)]
for split in ['train','dev','test']:
    with (ROOT/'data/raw/cfms34'/REV/f'{split}.csv').open(encoding='utf-8-sig',newline='') as f:raw[split]=list(csv.DictReader(f))
    processed[split]=[json.loads(s) for s in (ROOT/'data/processed/cfms34-v1'/f'{split}.jsonl').read_text().splitlines()]
    mapping={r['编号']:r for r in raw[split]}
    for row in processed[split]:assert row['score']==float(mapping[row['id']]['总分_1'])+float(mapping[row['id']]['总分_2'])
    a=np.array([float(r['总分_1']) for r in raw[split]]);b=np.array([float(r['总分_2']) for r in raw[split]])
    for row in raw[split]:
        for trait in trait_ids:
            for suffix in ['1','2']:
                try:
                    value=float(row[f'{trait}_{suffix}'])
                    assert np.isfinite(value) and 0<=value<=5
                except Exception:trait_errors.append({'id':row['编号'],'column':f'{trait}_{suffix}'})
    stats[split]={'n':len(a),'sum_score_counts':dict(sorted(collections.Counter((a+b).tolist()).items())),
                  'rater1_counts':dict(sorted(collections.Counter(a.tolist()).items())),
                  'rater2_counts':dict(sorted(collections.Counter(b.tolist()).items())),
                  'exact_agreement_n':int(np.sum(a==b)),'disagreement_n':int(np.sum(a!=b)),
                  'abs_difference_counts':dict(sorted(collections.Counter(np.abs(a-b).tolist()).items())),
                  'human_human_qwk_0_5':float(cohen_kappa_score(a.astype(int),b.astype(int),labels=list(range(6)),weights='quadratic'))}
norm=lambda t:''.join(c for c in unicodedata.normalize('NFKC',t) if c.isalnum()).lower()
overlap={}
for a,b in [('train','dev'),('train','test'),('dev','test')]:
    overlap[a+'__'+b]={'id_overlap':len({r['id'] for r in processed[a]}&{r['id'] for r in processed[b]}),
                         'text_hash_overlap':len({r['text_hash'] for r in processed[a]}&{r['text_hash'] for r in processed[b]}),
                         'normalized_title_overlap':len({norm(r['title']) for r in processed[a]}&{norm(r['title']) for r in processed[b]})}
ckpt=ROOT/'checkpoints/round1-cfms34-bge-v1'
vectorizer=joblib.load(ckpt/'lexical_vectorizer.joblib')
cached=np.load(ckpt/'embedding_cache.npz')
clipping={}
for family in ['length_ridge','character_tfidf_ridge','semantic_ridge','semantic_structure_ridge']:
    model=joblib.load(ckpt/f'{family}.joblib');clipping[family]={}
    for split,rows in processed.items():
        texts=[r['text'] for r in rows];y=np.array([r['score'] for r in rows])
        if family=='length_ridge':x=structural(texts)[:,:1]
        elif family=='character_tfidf_ridge':x=vectorizer.transform(texts)
        elif family=='semantic_ridge':x=cached[split]
        else:x=np.concatenate([cached[split],structural(texts)],axis=1)
        p=model.predict(x);c=np.clip(p,0,10)
        clipping[family][split]={'raw_min':float(p.min()),'raw_max':float(p.max()),
            'below_0':int(np.sum(p<0)),'above_10':int(np.sum(p>10)),
            'raw_mae':float(np.mean(np.abs(p-y))),'clipped_mae':float(np.mean(np.abs(c-y))),
            'signed_bias':float(np.mean(c-y)),
            'gold_9plus_n':int(np.sum(y>=9)),
            'gold_9plus_signed_bias':float(np.mean(c[y>=9]-y[y>=9])) if np.sum(y>=9) else None}
v1=json.loads((ROOT/'reports/round1-cfms34-bge-v1/metrics.json').read_text())
selection=json.loads((ROOT/'reports/round1-cfms34-bge-v1/selection_frozen.json').read_text())
best=max(selection['all_validation_candidates'],key=lambda r:(r['dev']['qwk'],-r['dev']['mae']))
assert best['family']==selection['selected_family']
with (ROOT/'reports/round1-cfms34-bge-v1/test_predictions.csv').open(newline='') as f:preds=list(csv.DictReader(f))
y=np.array([float(r['human_score']) for r in preds]);p=np.array([float(r[best['family']]) for r in preds])
O=np.zeros((11,11))
for a,b in zip(np.rint(y).astype(int),np.rint(p).astype(int)):O[a,b]+=1
E=np.outer(O.sum(1),O.sum(0))/len(y);W=(np.arange(11)[:,None]-np.arange(11)[None,:])**2
manual=float(1-(W*O).sum()/(W*E).sum())
assert abs(manual-v1['models'][best['family']]['test']['qwk'])<1e-12
report={'v1_files_verified':len(integrity),'score_definition':'rater1+rater2, 0..10; no scale conversion',
        'raw_stats':stats,'trait_count':len(trait_ids),'invalid_trait_cells':trait_errors,
        'overlap':overlap,'clipping':clipping,'qwk':{'manual_recalculation':manual,
             'rounding':'numpy.rint, ties-to-even','labels':list(range(11)),'weights':'quadratic',
             'single_gold_score_slice_qwk':'uninformative; do not interpret as group accuracy'},
        'selection_matches_dev_only_rule':True,
        'issues':['official test selected for rater agreement; dev distribution also differs',
                  'student/school/prompt independence not verifiable; titles are not verified prompts',
                  'train/test performance gap and high-score shrinkage are real limitations',
                  'round1 metadata claims encoder_frozen also for lexical winner; model family must be read',
                  'round1 predict.py falls back to best weights when requested family file missing; do not reuse fallback in v2',
                  'round1 CPU file manifest predates MODEL_CARD; archive manifest covers it',
                  'round1 multioutput-independent Ridge would not by itself couple tasks',
                  'no immutable test access barrier in v1; v2 runner must only load training partition'],
        'audit_only_test_usage':'historical clipping/metric verification, no candidate selection or parameter tuning'}
save('audit.json',report)
print(json.dumps(report,ensure_ascii=False,indent=2))
