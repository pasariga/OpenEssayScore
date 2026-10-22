"""Four disjoint roles per fold, original training partition only, CPU only."""
import csv,importlib.metadata,json,os,sys,time
from pathlib import Path
import joblib,numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,average_precision_score
from risk_core import *

ROOT=Path(os.environ.get('WENHENG_WORKSPACE', str(Path.cwd() / 'workspace'))).resolve();RUN='round4-trust-review-v1'
OUT=ROOT/'reports'/RUN;CKPT=ROOT/'checkpoints'/RUN
RULES=['random','length_outlier','lexical_novelty','semantic_novelty','disagreement','learned_basic','learned_full','local_interval_width']
BUDGETS=[.1,.2,.3]

def summary(y,p,rank,review):
    error=np.abs(p-y);event=error>1.;keep=~review
    return {'n':len(y),'base_mae':float(error.mean()),'large_error_n':int(event.sum()),
      'review_n':int(review.sum()),'review_rate':float(review.mean()),'retained_coverage':float(keep.mean()),
      'error_capture_rate':float(np.sum(review&event)/event.sum()) if event.sum() else None,
      'review_precision':float(np.mean(event[review])) if review.any() else None,
      'retained_mae':float(error[keep].mean()) if keep.any() else None,
      'retained_large_error_rate':float(event[keep].mean()) if keep.any() else None,
      'auroc':float(roc_auc_score(event,rank)) if len(np.unique(event))==2 else None,
      'average_precision':float(average_precision_score(event,rank)) if event.any() else None,
      'event_prevalence':float(event.mean()),'teacher_corrected_accuracy':'not measured; no teacher review performed'}

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','Run with GPU hidden'
    t0=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True);CKPT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'results.json').exists(),'Completed run must not be overwritten'
    trainfile=ROOT/'data/processed/cfms34-v1/train.jsonl'
    rows=[json.loads(line) for line in trainfile.read_text().splitlines()]
    ids=np.array([r['id'] for r in rows]);texts=np.array([r['text'] for r in rows],dtype=object)
    y=np.array([r['score'] for r in rows]);group=np.array([groups_title(r['title']) or r['text_hash'] for r in rows])
    assert len(rows)==751
    cachefile=ROOT/'checkpoints/round1-cfms34-bge-v1/embedding_cache.npz'
    with np.load(cachefile) as cache:embedding=cache['train'].astype(np.float64)
    manifest=json.loads((ROOT/'data/processed/cfms34-v1/split_manifest.json').read_text())
    assert sha(trainfile)==manifest['splits']['train']['sha256'] and ids.tolist()==manifest['splits']['train']['ids']
    partitions=[b for _,b in StratifiedGroupKFold(5,shuffle=True,random_state=20261024).split(texts,y.astype(int),group)]
    config={'run':RUN,'source_partition':'original 751 train only, already used in previous development',
      'score_recipe':'frozen v1 TFIDF char1..3 min_df2 max40000 sublinearTF + Ridge alpha0.1, refit replicas only',
      'v1_files_modified':False,'outer_partitions':5,'seed':20261024,
      'roles':'eval=k; interval calibration=(k+1)%5; risk development=(k+2)%5; score fitting=other 2 partitions',
      'title_group_limit':'title is only a weak proxy, student/school/prompt independence unknown',
      'risk_features':FEATURE_NAMES,'binary_target':'absolute score error >1.0 on CFMS0..10',
      'risk_model':'fixed C1 logistic raw margin, never expose predict_proba as confidence',
      'local_scale':'fixed alpha10 Ridge on log(abs_error+.25), clip scale .25..5',
      'semantic_auxiliary':'frozen cached BGE, fixed Ridge alpha1000; no auxiliary trait labels',
      'rules':RULES,'review_budgets':BUDGETS,'primary_budget':.2,
      'thresholds':'quantile 1-budget of calibration RANKS only, no calibration error-label tuning',
      'pooled_ranking':'calibration-feature empirical percentile per fold; this is a rank, not error probability',
      'interval_alpha':.1,'conformal':'split absolute residual and separately normalized residual; score fit and scale fit exclude calibration',
      'cpu_timeout_seconds':300,'gpu_seconds':0,
      'advance_gate':'learned full: AUROC >=.60, large-error capture at primary budget >=random+.05, retained MAE <=all MAE-.05; no deployment promotion',
      'inputs':{'training_sha256':sha(trainfile),'embedding_cache_sha256':sha(cachefile)},
      'v1_baseline_hashes':{p:sha(ROOT/p) for p in ['checkpoints/WenHeng-Score-v1-cpu/best_scorer.joblib','src/wenheng_score/predict.py']},
      'partitions':[[str(ids[i]) for i in part] for part in partitions]}
    dump(OUT/'preregistration.json',config)
    predicted=np.full(len(y),np.nan);features=np.full((len(y),len(FEATURE_NAMES)),np.nan)
    ranks={r:np.full(len(y),np.nan) for r in RULES}
    flags={r:{b:np.zeros(len(y),dtype=bool) for b in BUDGETS} for r in RULES}
    bounds={name:np.full((len(y),2),np.nan) for name in ['global','local']}
    allfold=[]
    for k in range(5):
      ev=partitions[k];cal=partitions[(k+1)%5];rd=partitions[(k+2)%5]
      fit=np.concatenate([partitions[j] for j in range(5) if j not in [k,(k+1)%5,(k+2)%5]])
      role_sets=[set(group[idx]) for idx in [fit,rd,cal,ev]]
      assert all(not role_sets[a]&role_sets[b] for a in range(4) for b in range(a+1,4))
      vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(1,3),min_df=2,max_features=40000,sublinear_tf=True,dtype=np.float64)
      xf=vectorizer.fit_transform(texts[fit]);scorer=Ridge(alpha=.1,solver='lsqr').fit(xf,y[fit])
      sem_scaler=StandardScaler().fit(embedding[fit]);sem=Ridge(alpha=1000.).fit(sem_scaler.transform(embedding[fit]),y[fit])
      reference=make_reference(texts[fit],embedding[fit],vectorizer,y[fit])
      pp={};xx={}
      for name,idx in [('risk',rd),('cal',cal),('eval',ev)]:
        pp[name]=np.clip(scorer.predict(vectorizer.transform(texts[idx])),0,10)
        aux=np.clip(sem.predict(sem_scaler.transform(embedding[idx])),0,10)
        xx[name]=feature_matrix(texts[idx],embedding[idx],pp[name],aux,reference)
      residual=np.abs(pp['risk']-y[rd]);event=(residual>1.).astype(int)
      basic=fit_risk(xx['risk'],event,BASIC);full=fit_risk(xx['risk'],event,list(range(len(FEATURE_NAMES))))
      scale_model=fit_local_scale(xx['risk'],residual)
      calerr=np.abs(pp['cal']-y[cal]);q=conformal_quantile(calerr)
      qlocal=conformal_quantile(calerr/local_scale(scale_model,xx['cal']))
      for label,idx in [('cal',cal),('eval',ev)]:
        x=xx[label]
        dd={'random':random_rank(ids[idx]),'length_outlier':x[:,8],'lexical_novelty':x[:,3],
          'semantic_novelty':x[:,6],'disagreement':x[:,7],
          'learned_basic':risk_margin(basic,x),'learned_full':risk_margin(full,x),
          'local_interval_width':local_scale(scale_model,x)}
        if label=='cal':cal_ranks=dd
        else:ev_ranks=dd
      thresholds={rule:{str(b):float(np.quantile(cal_ranks[rule],1-b,method='higher')) for b in BUDGETS} for rule in RULES}
      for rule in RULES:
        ranks[rule][ev]=np.searchsorted(np.sort(cal_ranks[rule]),ev_ranks[rule],side='right')/(len(cal)+1)
        for b in BUDGETS:flags[rule][b][ev]=ev_ranks[rule]>thresholds[rule][str(b)]
      predicted[ev]=pp['eval'];features[ev]=xx['eval']
      for label,half in [('global',np.full(len(ev),q)),('local',qlocal*local_scale(scale_model,xx['eval']))]:
        lo,hi=interval(pp['eval'],half);bounds[label][ev]=np.column_stack([lo,hi])
      fold_dir=CKPT/f'fold_{k}';fold_dir.mkdir(exist_ok=True)
      joblib.dump({'vectorizer':vectorizer,'scorer':scorer,'semantic_scaler':sem_scaler,'semantic_scorer':sem},fold_dir/'score_replica.joblib')
      score_sha=sha(fold_dir/'score_replica.joblib')
      joblib.dump({'reference':reference,'basic':basic,'full':full,'scale_model':scale_model,
                  'global_q':q,'local_q':qlocal,'thresholds':thresholds,
                  'calibration_ranks':{r:np.sort(cal_ranks[r]) for r in RULES},'score_model_sha256':score_sha},fold_dir/'risk_bundle.joblib')
      dump(fold_dir/'metadata.json',{'fold':k,'rubric_id':RUBRIC,'score_range':[0,10],
        'score_model_sha256':score_sha,'feature_names':FEATURE_NAMES,'calibration_n':len(cal),
        'status':'internal research replica; not original v1; not validated for external essays'})
      result={'fold':k,'role_counts':{'score_fit':len(fit),'risk_fit':len(rd),'calibration':len(cal),'validation':len(ev)},
        'calibration_quantile':q,'local_normalized_quantile':qlocal,
        'primary':{r:summary(y[ev],pp['eval'],ev_ranks[r],flags[r][.2][ev]) for r in RULES},
        'elapsed_seconds':time.perf_counter()-t0}
      allfold.append(result);dump(OUT/'fold_progress.json',allfold)
      print(json.dumps({'fold':k,'elapsed_seconds':result['elapsed_seconds'],'risk_validation_complete':True}),flush=True)
      assert time.perf_counter()-t0<300,'CPU budget exceeded'
    assert np.isfinite(predicted).all() and np.isfinite(features).all()
    results={r:{str(b):summary(y,predicted,ranks[r],flags[r][b]) for b in BUDGETS} for r in RULES}
    intervals={}
    for name,intervals_ in bounds.items():
      covered=(y>=intervals_[:,0])&(y<=intervals_[:,1])
      intervals[name]={'empirical_internal_coverage':float(covered.mean()),
        'mean_width':float(np.mean(intervals_[:,1]-intervals_[:,0])),
        'median_width':float(np.median(intervals_[:,1]-intervals_[:,0])),
        'external_coverage_guarantee':False,'calibration_target':.9}
    associations={name:float(spearmanr(features[:,j],np.abs(predicted-y)).statistic) for j,name in enumerate(FEATURE_NAMES)}
    selected=results['learned_full']['0.2'];random=results['random']['0.2']
    gate=selected['auroc']>=.6 and selected['error_capture_rate']>=random['error_capture_rate']+.05 and selected['retained_mae']<=selected['base_mae']-.05
    output={'scope':'four-role grouped internal validation on original training data; NOT a new independent test',
      'n':len(y),'all_scoring_predictions_are_out_of_score_fit':True,'results':results,'intervals':intervals,
      'feature_error_spearman':associations,'primary_full_advance_gate':bool(gate),
      'stopped_if_unsupported':not bool(gate),'no_automatic_production_promotion':True,
      'old_v1_weights_unchanged':all(sha(ROOT/p)==v for p,v in config['v1_baseline_hashes'].items()),
      'wall_seconds':time.perf_counter()-t0,'gpu_seconds':0,
      'limitations':['original training data already inspected in earlier research',
       'title grouping is not verified student or prompt grouping; conformal exchangeability uncertain',
       'replicas fit about300 essays, not the original full751 v1; triage and intervals do not transfer automatically',
       'no labeled real OOD set and no actual teacher re-review outcomes',
       'scores remain unchanged; retained-subset MAE is conditional, not total system accuracy improvement',
       'pooled scores use label-free calibration percentiles, not error probabilities']}
    dump(OUT/'results.json',output)
    with (OUT/'validation_predictions.csv').open('w',encoding='utf-8',newline='') as f:
      writer=csv.writer(f);writer.writerow(['id','human_score','replica_score']+RULES+[r+'_review20' for r in RULES])
      for i in range(len(y)):writer.writerow([ids[i],y[i],predicted[i]]+[ranks[r][i] for r in RULES]+[int(flags[r][.2][i]) for r in RULES])
    np.savez_compressed(CKPT/'validation_arrays.npz',features=features,y=y,prediction=predicted,
      global_bounds=bounds['global'],local_bounds=bounds['local'])
    # One real held-out internal feature request for CLI smoke tests, no fabricated score.
    i=int(partitions[0][0]);meta=json.loads((CKPT/'fold_0/metadata.json').read_text())
    request={'essay_id':str(ids[i]),'text':str(texts[i]),'score':float(predicted[i]),
      'features':features[i].tolist(),'feature_names':FEATURE_NAMES,'rubric_id':RUBRIC,'score_range':[0,10],
      'score_model_sha256':meta['score_model_sha256'],'authorization':'public_research_dataset',
      'prompt_present':True,'ocr_verified':True,'domain':'CFMS34-research'}
    dump(OUT/'internal_example_request.json',request)
    print(json.dumps({'primary20':{r:results[r]['0.2'] for r in RULES},'intervals':intervals,
      'advance_gate':bool(gate),'wall_seconds':output['wall_seconds'],'gpu_seconds':0},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
