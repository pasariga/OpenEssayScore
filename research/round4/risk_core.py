"""Reusable, CPU-only feature and conformal helpers. Never reads labels implicitly."""
import hashlib,json,re,unicodedata
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

FEATURE_NAMES=['log_chars','cjk_ratio','unique_character_ratio','lexical_oov_fraction',
               'provided_score','distance_from_fit_mean_score','semantic_knn_distance',
               'lexical_semantic_prediction_gap','length_mad_distance']
BASIC=[0,1,2,3,4,5,8]
RUBRIC='CFMS34-holistic-sum-0to10'

def norm(text):return unicodedata.normalize('NFKC',text).strip()
def groups_title(text):return ''.join(c for c in norm(text) if c.isalnum()).lower()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def make_reference(texts,embeddings,vectorizer,y):
    length=np.log1p([len(t) for t in texts]);median=float(np.median(length))
    mad=max(float(np.median(np.abs(length-median))),.05)
    nn=NearestNeighbors(n_neighbors=5,metric='cosine').fit(embeddings)
    return {'nn':nn,'length_median':median,'length_mad':mad,'mean_score':float(np.mean(y)),
            'vectorizer':vectorizer,'min_chars':min(map(len,texts)),'max_chars':max(map(len,texts))}

def feature_matrix(texts,embeddings,score,auxiliary,reference):
    analyzer=reference['vectorizer'].build_analyzer();vocab=reference['vectorizer'].vocabulary_
    distance=reference['nn'].kneighbors(embeddings,return_distance=True)[0].mean(1)
    result=[]
    for t,p,a,d in zip(texts,score,auxiliary,distance):
        n=max(len(t),1);tokens=analyzer(t)
        oov=sum(x not in vocab for x in tokens)/max(len(tokens),1)
        log=np.log1p(n)
        result.append([log,len(re.findall(r'[\u4e00-\u9fff]',t))/n,len(set(t))/n,oov,
                       p,abs(p-reference['mean_score']),d,abs(p-a),
                       abs(log-reference['length_median'])/reference['length_mad']])
    return np.asarray(result,dtype=np.float64)

def fit_risk(x,event,columns):
    scaler=StandardScaler();xs=scaler.fit_transform(x[:,columns])
    assert len(np.unique(event))==2,'Risk development partition needs both outcomes'
    learner=LogisticRegression(C=1.,max_iter=500,random_state=20261022).fit(xs,event)
    return {'scaler':scaler,'learner':learner,'columns':columns}

def risk_margin(model,x):
    return model['learner'].decision_function(model['scaler'].transform(x[:,model['columns']]))

def fit_local_scale(x,residual):
    scaler=StandardScaler();xs=scaler.fit_transform(x)
    learner=Ridge(alpha=10.).fit(xs,np.log(residual+.25))
    return {'scaler':scaler,'learner':learner}

def local_scale(model,x):
    return np.clip(np.exp(model['learner'].predict(model['scaler'].transform(x))),.25,5.)

def conformal_quantile(residual,alpha=.1):
    residual=np.asarray(residual);n=len(residual);rank=int(np.ceil((n+1)*(1-alpha)))
    if n==0 or rank>n:raise ValueError('Insufficient finite calibration data')
    assert np.isfinite(residual).all() and (residual>=0).all()
    return float(np.sort(residual)[rank-1])

def interval(score,halfwidth):return np.maximum(0,score-halfwidth),np.minimum(10,score+halfwidth)

def random_rank(ids,seed=20261022):
    return np.array([int(hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest()[:13],16)/16**13 for i in ids])
