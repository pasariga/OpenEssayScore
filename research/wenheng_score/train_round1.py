"""Train CPU-deployable supervised scorers, select on dev, then evaluate test once."""
import argparse
import collections
import csv
import datetime
import importlib.metadata
import json
from pathlib import Path
import random
import time
import joblib
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score
from common import ROOT, SEED, PROCESSED, PRETRAINED, dump, load_split, text_input, checksum
from features import structural, FEATURE_NAMES, Encoder


def metrics(y,p):
    p=np.clip(np.asarray(p),0,10)
    qwk=cohen_kappa_score(np.rint(y).astype(int),np.rint(p).astype(int),labels=list(range(11)),weights='quadratic')
    return {'mae':float(mean_absolute_error(y,p)), 'rmse':float(np.sqrt(mean_squared_error(y,p))),
            'qwk':float(qwk) if np.isfinite(qwk) else None,
            'pearson':float(pearsonr(y,p).statistic) if np.std(p)>1e-8 and np.std(y)>1e-8 else None,
            'spearman':float(spearmanr(y,p).statistic) if np.std(p)>1e-8 and np.std(y)>1e-8 else None,
            'within_1_point':float(np.mean(np.abs(y-p)<=1)),
            'rounded_exact':float(np.mean(np.rint(y)==np.rint(p)))}


def selection_key(record):
    m=record['dev']
    return (m['qwk'] if m['qwk'] is not None else -1,-m['mae'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',default='round1-cfms34-bge-v1')
    args=parser.parse_args()
    run=ROOT/'reports'/args.run_id
    ckpt=ROOT/'checkpoints'/args.run_id
    assert not (run/'selection_frozen.json').exists(), 'Use a new run-id; do not overwrite evaluated runs'
    run.mkdir(parents=True,exist_ok=True); ckpt.mkdir(parents=True,exist_ok=True)
    random.seed(SEED); np.random.seed(SEED)
    import torch
    torch.manual_seed(SEED); torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    train,dev=load_split('train'),load_split('dev')
    # Test labels are not loaded into fitting or selection functions.
    tests=[{'text':r['text'],'id':r['id']} for r in load_split('test')]
    ys={'train':np.asarray([r['score'] for r in train]),'dev':np.asarray([r['score'] for r in dev])}
    texts={s:[text_input(r) for r in rows] for s,rows in [('train',train),('dev',dev),('test',tests)]}
    config={'run_id':args.run_id,'seed':SEED,'target_scale':[0,10],
            'target':'sum of dual human ratings','features':'essay text only',
            'model_id':'BAAI/bge-small-zh-v1.5', 'encoder_frozen':True,
            'pooling':'L2 normalized CLS per <=510 token chunk; L2 normalized chunk sum',
            'max_tokens_per_chunk':512,'max_chunks':'all','encoder_cuda_dtype':'float16',
            'supervised_scorer':'Ridge; trained on official training split only',
            'alpha_grid':[.1,1,10,100,1000],
            'selection':'highest dev QWK, tie lower dev MAE; fixed round-to-nearest integer on 0..10',
            'test_policy':'one frozen evaluation, no tuning or thresholds from test',
            'gpu_dense_budget_seconds':1800,'gpu_watchdog_seconds':240,
            'structural_features':FEATURE_NAMES,
            'processed_manifest_sha256':checksum(PROCESSED/'split_manifest.json'),
            'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','numpy','scikit-learn','scipy','joblib']}}
    dump(ckpt/'training_config.json',config)
    all_candidates=[]; fitted={}; representations={}

    def fit_family(name,xs,scaled):
        candidates=[]
        begin=time.perf_counter()
        for alpha in config['alpha_grid']:
            model=make_pipeline(StandardScaler(),Ridge(alpha=alpha)) if scaled else Ridge(alpha=alpha,solver='lsqr')
            start=time.perf_counter(); model.fit(xs['train'],ys['train'])
            record={'family':name,'alpha':alpha,'fit_seconds':time.perf_counter()-start,
                    'train':metrics(ys['train'],model.predict(xs['train'])),
                    'dev':metrics(ys['dev'],model.predict(xs['dev']))}
            candidates.append((record,model)); all_candidates.append(record)
        record,model=max(candidates,key=lambda pair:selection_key(pair[0]))
        fitted[name]=(record,model)
        representations[name]=xs
        joblib.dump(model,ckpt/f'{name}.joblib')
        print(json.dumps({'family_selected':record,'family_fit_wall_seconds':time.perf_counter()-begin}),flush=True)

    structure={s:structural(t) for s,t in texts.items()}
    lengths={s:x[:,:1] for s,x in structure.items()}
    fit_family('length_ridge',lengths,True)
    lexical=TfidfVectorizer(analyzer='char',ngram_range=(1,3),min_df=2,max_features=40000,sublinear_tf=True,dtype=np.float32)
    lexical.fit(texts['train'])
    lex={s:lexical.transform(t) for s,t in texts.items()}
    joblib.dump(lexical,ckpt/'lexical_vectorizer.joblib')
    fit_family('character_tfidf_ridge',lex,False)

    # Watchdog is scoped to this PID, never touches unrelated GPU processes.
    import os, signal, threading
    watchdog=threading.Timer(240,lambda:os.kill(os.getpid(),signal.SIGTERM))
    watchdog.daemon=True; watchdog.start()
    gpu_begin=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    encoder=Encoder(PRETRAINED,device='cuda',batch_size=16)
    embeddings={}; encoding_logs={}
    for split in ['train','dev','test']:
        embeddings[split],encoding_logs[split]=encoder.encode(texts[split])
        print(json.dumps({'embedding':split,**encoding_logs[split]}),flush=True)
    torch.cuda.synchronize()
    gpu_seconds=time.perf_counter()-gpu_begin
    peak_gpu_bytes=torch.cuda.max_memory_allocated()
    parameters=encoder.parameter_count
    del encoder; torch.cuda.empty_cache(); watchdog.cancel()
    np.savez_compressed(ckpt/'embedding_cache.npz',**embeddings)
    dump(run/'gpu_usage.json',{'gpu_wall_seconds':gpu_seconds,'peak_torch_gpu_bytes':peak_gpu_bytes,
                             'encoder_parameters':parameters,'encoding':encoding_logs,
                             'scope':'model load to GPU, all embedding passes, synchronization; Ridge training runs on CPU',
                             'budget_seconds':1800,'watchdog_seconds':240})
    fit_family('semantic_ridge',embeddings,True)
    semantic_structure={s:np.concatenate([embeddings[s],structure[s]],axis=1) for s in embeddings}
    fit_family('semantic_structure_ridge',semantic_structure,True)
    selected_name=max(fitted,key=lambda name:selection_key(fitted[name][0]))
    selected_record,selected_model=fitted[selected_name]
    freeze={'selected_family':selected_name,'selected_config':selected_record,
            'selection_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'test_not_used_for_selection':True,'all_validation_candidates':all_candidates}
    dump(run/'selection_frozen.json',freeze)
    joblib.dump(selected_model,ckpt/'best_scorer.joblib')
    dump(ckpt/'model_metadata.json',{'selected_family':selected_name,'source_model_path':str(PRETRAINED),
                                  'training_config':'training_config.json','score_range':[0,10],
                                  'status':'research baseline; not validated across grades or schools',
                                  'dataset_license':'CC-BY-4.0','encoder_license':'MIT'})

    # Independent test evaluation begins only after writing the selection freeze.
    test_records=load_split('test')
    ytest=np.asarray([r['score'] for r in test_records])
    predictions={}; test_results={}
    mean=float(ys['train'].mean())
    predictions['training_mean']=np.full(len(ytest),mean)
    test_results['training_mean']={'test':metrics(ytest,predictions['training_mean']),
                                 'dev':metrics(ys['dev'],np.full(len(dev),mean)),
                                 'train':metrics(ys['train'],np.full(len(train),mean))}
    for name,(record,model) in fitted.items():
        predictions[name]=np.clip(model.predict(representations[name]['test']),0,10)
        test_results[name]={'test':metrics(ytest,predictions[name]),'dev':record['dev'],
                            'train':record['train'],'alpha':record['alpha']}
    selected_pred=predictions[selected_name]
    rng=np.random.default_rng(SEED)
    boots=[]
    for _ in range(1000):
        indices=rng.integers(0,len(ytest),len(ytest))
        boots.append(metrics(ytest[indices],selected_pred[indices]))
    intervals={metric:[float(np.percentile([b[metric] for b in boots if b[metric] is not None],q)) for q in [2.5,97.5]]
               for metric in ['mae','qwk']}
    errors=[]
    for r,p in zip(test_records,selected_pred):
        errors.append({'id':r['id'],'human_score':r['score'],'prediction':float(p),
                       'absolute_error':float(abs(r['score']-p)),'characters':len(r['text'])})
    errors.sort(key=lambda r:r['absolute_error'],reverse=True)
    score_slices={str(score):{'n':int(np.sum(ytest==score)),**metrics(ytest[ytest==score],selected_pred[ytest==score])}
                  for score in np.unique(ytest)}
    length_slices={}
    for label,lo,hi in [('short',0,350),('medium',350,650),('long',650,100000)]:
        mask=np.array([lo<=len(r['text'])<hi for r in test_records])
        if mask.sum():length_slices[label]={'n':int(mask.sum()),**metrics(ytest[mask],selected_pred[mask])}
    with (run/'test_predictions.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','human_score']+list(predictions));writer.writeheader()
        for i,r in enumerate(test_records):writer.writerow({'id':r['id'],'human_score':r['score'],**{k:float(p[i]) for k,p in predictions.items()}})
    result={'selected_family':selected_name,'target_scale':[0,10],'models':test_results,
            'selected_bootstrap95':intervals,'score_slices':score_slices,'length_slices':length_slices,
            'largest_errors':errors[:15], 'gpu_wall_seconds':gpu_seconds,
            'limitations':['100 test essays; agreement-selected official test distribution',
                          'dev distribution differs sharply from train and test',
                          'test only scores 6,8,10 in this release; lower scores untested',
                          'no student/school-disjoint split can be verified',
                          'no prompt relevance or plagiarism detector',
                          'encoder frozen; only supervised scorer trained on human ratings',
                          'label-shift and grade generalization not validated']}
    dump(run/'metrics.json',result)
    # Export readable, portable ridge coefficients in addition to joblib.
    for name,(_,model) in fitted.items():
        if hasattr(model,'named_steps'):
            scaler=model.named_steps['standardscaler']; ridge=model.named_steps['ridge']
            np.savez(ckpt/f'{name}_weights.npz',coef=ridge.coef_,intercept=ridge.intercept_,
                     scaler_mean=scaler.mean_,scaler_scale=scaler.scale_)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
