"""Fixed-recipe CPU reproduction on authorized processed data; never reads test."""
import argparse,json,sys,time
from pathlib import Path
import joblib,numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,cohen_kappa_score
sys.path.insert(0,str(Path(__file__).parent/'wenheng_score'))
from predict import clean


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--run-id',required=True)
    a=p.parse_args()
    if not __import__('re').fullmatch('[A-Za-z0-9_-]{1,80}',a.run_id):raise ValueError('Invalid run ID')
    out=a.workspace/'checkpoints'/a.run_id;out.mkdir(parents=True,exist_ok=False)
    data=a.workspace/'data/processed/cfms34-v1'
    rows={s:[json.loads(x) for x in (data/(s+'.jsonl')).read_text(encoding='utf-8').splitlines()] for s in ['train','dev']}
    started=time.perf_counter()
    vec=TfidfVectorizer(analyzer='char',ngram_range=(1,3),min_df=2,max_features=40000,sublinear_tf=True,dtype=np.float64)
    x=vec.fit_transform([clean(r['text']) for r in rows['train']]);y=np.array([r['score'] for r in rows['train']])
    model=Ridge(alpha=.1,solver='lsqr');model.fit(x,y)
    actual=np.array([r['score'] for r in rows['dev']]);pred=np.clip(model.predict(vec.transform([clean(r['text']) for r in rows['dev']])),0,10)
    result={'status':'local_reproduction_not_stable_model','test_read':False,'fit_seconds':time.perf_counter()-started,
      'train_n':len(y),'dev_n':len(actual),'dev_mae':float(mean_absolute_error(actual,pred)),
      'dev_qwk':float(cohen_kappa_score(np.rint(actual).astype(int),np.rint(pred).astype(int),labels=list(range(11)),weights='quadratic'))}
    joblib.dump(model,out/'best_scorer.joblib');joblib.dump(vec,out/'lexical_vectorizer.joblib')
    (out/'reproduction.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
