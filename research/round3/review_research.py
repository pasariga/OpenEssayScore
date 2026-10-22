import os
"""Describe existing training-only OOF errors. No fitting or new predictions."""
import csv,hashlib,json,collections,time
from pathlib import Path
import numpy as np
ROOT=Path(os.environ.get('WENHENG_WORKSPACE', str(Path.cwd() / 'workspace'))).resolve()
OUT=ROOT/'reports/round3-rubric-protocol-v1';OUT.mkdir(exist_ok=True)
source=ROOT/'reports/round2-audit-cv-v1/cv_oof_predictions.csv'
with source.open(newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==751
y=np.array([float(r['human_score']) for r in rows])
p=np.array([float(r['late_fusion']) for r in rows])
error=np.abs(np.clip(p,0,10)-y)
spread=np.array([max(float(r[k]) for k in ['lexical','semantic_structure','shared_multitask'])-
                 min(float(r[k]) for k in ['lexical','semantic_structure','shared_multitask']) for r in rows])
# Fixed exploratory rule. No threshold search or old-test data access.
flag=spread>=1.0
report={'scope':'descriptive reuse of round2 original-training OOF predictions only',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'n':len(rows),
        'residual_quantiles':{str(q):float(np.quantile(error,q,method='higher')) for q in [.5,.8,.9,.95]},
        'fixed_disagreement_rule':{'max_minus_min_at_least':1.0,'scale':'CFMS sum 0..10',
             'flagged_n':int(flag.sum()),'flagged_rate':float(flag.mean()),
             'flagged_mae':float(error[flag].mean()) if flag.any() else None,
             'unflagged_mae':float(error[~flag].mean()) if (~flag).any() else None,
             'large_error_definition':'absolute error > 1.0',
             'large_error_recall':float(np.sum(flag&(error>1))/np.sum(error>1)) if np.sum(error>1) else None},
        'not_a_calibrated_interval':True,'coverage_guarantee_claimed':False,
        'new_independent_test_n':0,'training_runs':0,'gpu_seconds':0,
        'limitations':['OOF models share fitting data across folds; residuals cannot be treated as an independent split-conformal calibration set',
                      'flag behavior here is descriptive and already inspected; not a validated triage performance estimate',
                      'these scores use CFMS 0..10 and must not be doubled or transferred to the new 0..20 rubric',
                      'prediction disagreement is a warning signal, not a probability or proof of error']}
(OUT/'review_research.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
