import os
"""CPU-only nested CV on the original training split. Never opens dev/test labels."""
import collections,csv,hashlib,json,sys,time,unicodedata
from pathlib import Path
import joblib,numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import cohen_kappa_score
ROOT=Path(os.environ.get('WENHENG_WORKSPACE', str(Path.cwd() / 'workspace'))).resolve();RUN='round2-audit-cv-v1'
OUT=ROOT/'reports'/RUN;CKPT=ROOT/'checkpoints'/RUN
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'wenheng_score'))
from features import structural
SEED=20261022
ALPHAS=[1.,10.,100.,1000.]
RANKS=[4,8]
FAMILIES=['mean','length','lexical','semantic','semantic_structure','semantic_structure_completecase','shared_multitask','late_fusion']

def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(t):return ''.join(c for c in unicodedata.normalize('NFKC',t) if c.isalnum()).lower()
def metrics(y,raw):
    p=np.clip(raw,0,10);q=cohen_kappa_score(np.rint(y).astype(int),np.rint(p).astype(int),labels=list(range(11)),weights='quadratic')
    high=y>=9
    return {'mae':float(np.mean(np.abs(p-y))),'rmse':float(np.sqrt(np.mean((p-y)**2))),
            'qwk':float(q) if np.isfinite(q) else None,'bias':float(np.mean(p-y)),
            'raw_mae':float(np.mean(np.abs(raw-y))),'clipped_n':int(np.sum((raw<0)|(raw>10))),
            'high_9plus_n':int(high.sum()),'high_9plus_bias':float(np.mean(p[high]-y[high])) if high.sum() else None}
def order(y,p):
    m=metrics(y,p)
    return (m['mae'],-(m['qwk'] if m['qwk'] is not None else -1))

def densefit(x,y,alpha):
    scaler=StandardScaler();xs=scaler.fit_transform(x)
    model=Ridge(alpha=alpha);model.fit(xs,y)
    return {'scaler':scaler,'regressor':model}
def densepredict(model,x):return model['regressor'].predict(model['scaler'].transform(x))

def rrrfit(x,Y,alpha,rank):
    # Exact reduced-rank ridge: eigenvectors of Y'X(X'X+lambda I)^-1X'Y.
    scaler=StandardScaler();xs=scaler.fit_transform(x).astype(np.float64)
    mean=Y.mean(0);yc=Y-mean
    xy=xs.T@yc
    coef=np.linalg.solve(xs.T@xs+alpha*np.eye(xs.shape[1]),xy)
    gram=xy.T@coef;gram=(gram+gram.T)/2
    _,vectors=np.linalg.eigh(gram);V=vectors[:,-rank:]
    return {'scaler':scaler,'coef':coef@V@V.T,'target_mean':mean,'rank':rank,'alpha':alpha}
def rrrpredict(model,x):return model['scaler'].transform(x)@model['coef']+model['target_mean']


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=True);CKPT.mkdir(exist_ok=True)
    assert not (OUT/'cv_results.json').exists(),'Do not overwrite a completed experiment'
    # Source of truth is the train file ONLY. No dev/test CSV or predictions loaded.
    trainfile=ROOT/'data/processed/cfms34-v1/train.jsonl'
    rows=[json.loads(line) for line in trainfile.read_text().splitlines()]
    texts=np.array([r['text'] for r in rows],dtype=object)
    y=np.array([r['score'] for r in rows],dtype=np.float64)
    groups=np.array([norm(r['title']) or r['text_hash'] for r in rows])
    rawfile=ROOT/'data/raw/cfms34/933489269eb9dbf5f6f00f8e8be6ba0ccebc3dbb/train.csv'
    with rawfile.open(encoding='utf-8-sig',newline='') as f:raw={r['编号']:r for r in csv.DictReader(f)}
    with (rawfile.parent/'trait_map.csv').open(encoding='utf-8-sig',newline='') as f:
        traits=[r['hifts_id'] for r in csv.DictReader(f) if r['hifts_id']!='Cv04']
    Y=np.full((len(rows),len(traits)+1),np.nan);Y[:,0]=y
    invalid=[]
    for i,row in enumerate(rows):
        for j,trait in enumerate(traits,1):
            a,b=float(raw[row['id']][trait+'_1']),float(raw[row['id']][trait+'_2'])
            if np.isfinite(a+b) and 0<=a<=5 and 0<=b<=5:Y[i,j]=a+b
            else:invalid.append({'id':row['id'],'trait':trait,'rater1':a,'rater2':b})
    complete=np.isfinite(Y).all(1)
    # Freeze exact IDs against the original run's recorded training order.
    original_manifest=ROOT/'data/processed/cfms34-v1/split_manifest.json'
    manifest=json.loads(original_manifest.read_text())
    assert [r['id'] for r in rows]==manifest['splits']['train']['ids']
    assert sha(trainfile)==manifest['splits']['train']['sha256']
    cachefile=ROOT/'checkpoints/round1-cfms34-bge-v1/embedding_cache.npz'
    # Explicitly read train array only, never access the other arrays.
    with np.load(cachefile) as cache:semantic=np.asarray(cache['train'],dtype=np.float64)
    assert semantic.shape==(len(rows),512) and np.isfinite(semantic).all()
    structure=structural(texts).astype(np.float64)
    dense=np.concatenate([semantic,structure],axis=1)
    DISAGREE=np.array([r['rater1']!=r['rater2'] for r in rows])
    outer=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(texts,y.astype(int),groups))
    folds=[{'fold':i,'fit_ids':[rows[j]['id'] for j in a],'validation_ids':[rows[j]['id'] for j in b]} for i,(a,b) in enumerate(outer)]
    config={'run':RUN,'training_rows':len(rows),'seed':SEED,'outer_folds':5,'inner_folds':3,
            'groups':'NFKC/alnum normalized essay title, a weak proxy only; not verified prompt/student ID',
            'alphas':ALPHAS,'shared_ranks':RANKS,'fusion_semantic_weights':[.25,.5,.75],
            'primary_selection':'lowest inner pooled MAE, tie higher fixed-label QWK',
            'families':FAMILIES,'targets':['holistic']+traits,
            'excluded_trait':'Cv04 handwriting: no images available',
            'bad_trait_policy':'quarantine out-of-range labels; no guessed corrections; complete-case fits only',
            'invalid_training_traits':invalid,'complete_case_n':int(complete.sum()),
            'cv_scope':'original 751 training essays only, no new independent test',
            'gpu_seconds':0,'cpu_deadline_seconds':600,
            'advance_gate':'development only: seek MAE reduction >=0.05 vs nested lexical baseline, no QWK decline and reduced high-score bias; no production promotion without sealed external test',
            'inputs':{'processed_train_sha256':sha(trainfile),'raw_train_sha256':sha(rawfile),
                      'embedding_cache_sha256':sha(cachefile)},
            'access_policy':'runner does not load dev/test partitions or old test predictions',
            'folds':folds}
    save(OUT/'cv_preregistration.json',config)
    predictions={k:np.full(len(rows),np.nan) for k in FAMILIES}
    trait_predictions=np.full_like(Y,np.nan)
    fold_results=[];fit_count=0

    for fold,(train_idx,val_idx) in enumerate(outer):
        assert not set(groups[train_idx])&set(groups[val_idx])
        inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED+fold+1).split(texts[train_idx],y[train_idx].astype(int),groups[train_idx]))
        inner_predictions={name:{a:np.full(len(train_idx),np.nan) for a in ALPHAS} for name in ['length','lexical','semantic','semantic_structure','semantic_structure_completecase']}
        mt_predictions={(a,r):np.full(len(train_idx),np.nan) for a in [10.,100.] for r in RANKS}
        for inner_fit,inner_val in inner:
            a,b=train_idx[inner_fit],train_idx[inner_val]
            assert not set(groups[a])&set(groups[b])
            vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(1,3),min_df=2,max_features=40000,sublinear_tf=True,dtype=np.float64)
            xa=vectorizer.fit_transform(texts[a]);xb=vectorizer.transform(texts[b])
            for alpha in ALPHAS:
                lex=Ridge(alpha=alpha,solver='lsqr').fit(xa,y[a]);fit_count+=1
                inner_predictions['lexical'][alpha][inner_val]=lex.predict(xb)
                for family,array in [('length',structure[:,:1]),('semantic',semantic),('semantic_structure',dense),('semantic_structure_completecase',dense)]:
                    fit_indices=a[complete[a]] if family.endswith('completecase') else a
                    model=densefit(array[fit_indices],y[fit_indices],alpha);fit_count+=1
                    inner_predictions[family][alpha][inner_val]=densepredict(model,array[b])
            good=a[complete[a]]
            for alpha in [10.,100.]:
                for rank in RANKS:
                    model=rrrfit(dense[good],Y[good],alpha,rank);fit_count+=1
                    mt_predictions[(alpha,rank)][inner_val]=rrrpredict(model,dense[b])[:,0]
            assert time.perf_counter()-start<600,'CPU budget exceeded'
        chosen={family:min(scores,key=lambda alpha:order(y[train_idx],scores[alpha])) for family,scores in inner_predictions.items()}
        chosen_mt=min(mt_predictions,key=lambda key:order(y[train_idx],mt_predictions[key]))
        lex_inner=inner_predictions['lexical'][chosen['lexical']]
        sem_inner=inner_predictions['semantic_structure'][chosen['semantic_structure']]
        weight=min([.25,.5,.75],key=lambda w:order(y[train_idx],(1-w)*lex_inner+w*sem_inner))

        vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(1,3),min_df=2,max_features=40000,sublinear_tf=True,dtype=np.float64)
        xa=vectorizer.fit_transform(texts[train_idx]);xb=vectorizer.transform(texts[val_idx])
        models={'lexical_vectorizer':vectorizer}
        predictions['mean'][val_idx]=y[train_idx].mean()
        lex=Ridge(alpha=chosen['lexical'],solver='lsqr').fit(xa,y[train_idx]);fit_count+=1
        models['lexical']=lex;predictions['lexical'][val_idx]=lex.predict(xb)
        for family,array in [('length',structure[:,:1]),('semantic',semantic),('semantic_structure',dense),('semantic_structure_completecase',dense)]:
            good=train_idx[complete[train_idx]] if family.endswith('completecase') else train_idx
            model=densefit(array[good],y[good],chosen[family]);fit_count+=1
            predictions[family][val_idx]=densepredict(model,array[val_idx]);models[family]=model
        good=train_idx[complete[train_idx]]
        mt=rrrfit(dense[good],Y[good],*chosen_mt);fit_count+=1
        models['shared_multitask']=mt
        trait_predictions[val_idx]=rrrpredict(mt,dense[val_idx])
        predictions['shared_multitask'][val_idx]=trait_predictions[val_idx,0]
        predictions['late_fusion'][val_idx]=(1-weight)*predictions['lexical'][val_idx]+weight*predictions['semantic_structure'][val_idx]
        models['late_fusion_semantic_weight']=weight
        joblib.dump(models,CKPT/f'outer_fold_{fold}.joblib')
        result={'fold':fold,'fit_n':len(train_idx),'validation_n':len(val_idx),
                'hyperparameters':{'alphas':chosen,'shared_multitask':list(chosen_mt),'fusion_semantic_weight':weight},
                'models':{family:metrics(y[val_idx],pred[val_idx]) for family,pred in predictions.items()},
                'elapsed_seconds':time.perf_counter()-start}
        fold_results.append(result);save(OUT/'fold_progress.json',fold_results)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    assert all(np.isfinite(p).all() for p in predictions.values())
    full={}
    for name,pred in predictions.items():
        full[name]={'pooled_oof':metrics(y,pred),
                    'fold_mae_mean':float(np.mean([r['models'][name]['mae'] for r in fold_results])),
                    'fold_mae_std':float(np.std([r['models'][name]['mae'] for r in fold_results],ddof=1)),
                    'agreement_cases':metrics(y[~DISAGREE],pred[~DISAGREE]),
                    'disagreement_cases':metrics(y[DISAGREE],pred[DISAGREE])}
    # No automatic promotion: these CV results are model-development evidence only.
    selected=min(full,key=lambda name:order(y,predictions[name]))
    auxiliary={}
    for j,trait in enumerate(traits,1):
        valid=np.isfinite(Y[:,j]);auxiliary[trait]={'n':int(valid.sum()),'oof':metrics(Y[valid,j],trait_predictions[valid,j])}
    with (OUT/'cv_oof_predictions.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','human_score','human_disagreement']+FAMILIES);writer.writeheader()
        for i,row in enumerate(rows):writer.writerow({'id':row['id'],'human_score':y[i],'human_disagreement':int(DISAGREE[i]),**{k:float(v[i]) for k,v in predictions.items()}})
    np.savez_compressed(CKPT/'training_only_oof.npz',**predictions,auxiliary=trait_predictions)
    results={'scope':'nested grouped CV within original training set; NOT independent test',
             'training_n':len(rows),'families':full,'development_candidate':selected,
             'outer_fold_results':fold_results,'auxiliary_trait_metrics':auxiliary,
             'fit_count':fit_count,'wall_seconds':time.perf_counter()-start,'gpu_seconds':0,
             'v1_preserved':True,'production_model_replaced':False,
             'limitations':['title grouping is not verified prompt, author or school grouping',
                            'one fixed-seed nested CV, small dataset; fold errors are correlated',
                            'family chosen after comparing OOF results; winner estimate can be optimistic',
                            'no new external human-scored dataset acquired for evaluation',
                            'no gold trait scores are inputs; low-rank shared output is the only task coupling']}
    save(OUT/'cv_results.json',results)
    print(json.dumps({'summary':{name:r['pooled_oof'] for name,r in full.items()},'development_candidate':selected,
                      'fit_count':fit_count,'wall_seconds':results['wall_seconds'],'gpu_seconds':0},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
