"""Decoupled review CLI: consumes a score; never changes or produces essay scores."""
import argparse,json,re
from pathlib import Path
import joblib,numpy as np
from risk_core import FEATURE_NAMES,RUBRIC,risk_margin,local_scale,interval,norm,sha

V1_SHA='c28c5010ae4f1170aba860ecbe9f2393f5a723311c59f89db5ebbc2f773193cb'

class ReviewEngine:
    def __init__(self,bundle_dir):
        self.directory=Path(bundle_dir)
        self.metadata=json.loads((self.directory/'metadata.json').read_text())
        self.integrity=json.loads((self.directory/'engine_manifest.json').read_text())
        for filename,digest in self.integrity['files'].items():
            if sha(self.directory/filename)!=digest:raise ValueError('Review artifact hash mismatch: '+filename)
        self.bundle=joblib.load(self.directory/'risk_bundle.joblib')

    def assess(self,request,experimental_risk=False):
        reasons=[];warnings=[]
        result={'module':'WenHeng review prototype r4','mode':'guardrails_only',
          'decision':'teacher_review','teacher_review_required':True,'score_changed':False,
          'experimental_review_flag':None,'error_probability':None,'confidence':None,
          'candidate_interval':None,'reasons':reasons,'warnings':warnings,
          'production_ready':False,'research_risk_method_supported':False}
        if not isinstance(request,dict):
            result['decision']='reject_input';reasons.append('REQUEST_NOT_OBJECT');return result
        text=request.get('text')
        if not isinstance(text,str) or not text.strip():reasons.append('MISSING_TEXT')
        elif len(text)>20000:reasons.append('TEXT_EXCEEDS_INPUT_LIMIT')
        if request.get('authorization') not in ['public_research_dataset','authorized_research']:reasons.append('AUTHORIZATION_NOT_DECLARED')
        if request.get('rubric_id')!=RUBRIC or request.get('score_range')!=[0,10]:reasons.append('RUBRIC_OR_SCALE_MISMATCH')
        if request.get('prompt_present') is not True:reasons.append('MISSING_TASK')
        if request.get('ocr_verified') is not True:reasons.append('TEXT_OR_OCR_NOT_VERIFIED')
        score=request.get('score')
        if isinstance(score,bool) or not isinstance(score,(float,int)) or not np.isfinite(score) or not 0<=score<=10:reasons.append('INVALID_SCORE')
        supplied_sha=request.get('score_model_sha256')
        calibrated_sha=self.bundle['score_model_sha256']
        if supplied_sha not in [calibrated_sha,V1_SHA]:reasons.append('UNKNOWN_SCORE_MODEL_HASH')
        if reasons:
            result['decision']='reject_input';return result
        clean=norm(text);n=max(len(clean),1);cjk=len(re.findall(r'[\u4e00-\u9fff]',clean))
        if cjk<20 or cjk/n<.3:warnings.append('INSUFFICIENT_CHINESE_OR_LANGUAGE_OUTSIDE_SCOPE')
        if '\ufffd' in clean or sum(ord(c)<32 and c not in '\n\r\t' for c in clean)/n>.01:warnings.append('TEXT_ENCODING_SUSPECT')
        if len(set(clean))/n<.03:warnings.append('EXTREME_REPETITION')
        reference=self.bundle['reference']
        if not reference['min_chars']<=n<=reference['max_chars']:warnings.append('LENGTH_OUTSIDE_FITTING_RANGE')
        if request.get('domain')!='CFMS34-research':warnings.append('DOMAIN_NOT_EXTERNALLY_VALIDATED')
        if supplied_sha==V1_SHA:
            reasons.append('ORIGINAL_V1_HAS_NO_MATCHING_R4_CALIBRATION')
            reasons.append('EXTERNAL_TEACHER_VALIDATION_PENDING')
            return result
        features=request.get('features');names=request.get('feature_names')
        try:
            if names!=FEATURE_NAMES:raise ValueError('feature schema')
            x=np.asarray(features,dtype=np.float64)
            if x.shape!=(len(FEATURE_NAMES),) or not np.isfinite(x).all():raise ValueError('feature shape/value')
            # Recompute all inexpensive features rather than trust arbitrary score metadata.
            analyzer=reference['vectorizer'].build_analyzer();tokens=analyzer(clean)
            oov=sum(t not in reference['vectorizer'].vocabulary_ for t in tokens)/max(1,len(tokens))
            expected={0:np.log1p(n),1:cjk/n,2:len(set(clean))/n,3:oov,4:score,
                      5:abs(score-reference['mean_score']),8:abs(np.log1p(n)-reference['length_median'])/reference['length_mad']}
            if any(not np.isclose(x[i],value,atol=1e-6) for i,value in expected.items()):raise ValueError('feature/text mismatch')
            if not (0<=x[6]<=2 and 0<=x[7]<=10):raise ValueError('semantic feature range')
        except (ValueError,TypeError):
            reasons.extend(['FEATURES_MISSING_INVALID_OR_MISMATCHED','EXTERNAL_TEACHER_VALIDATION_PENDING']);return result
        if x[6]>np.quantile(self.bundle['calibration_ranks']['semantic_novelty'],.95,method='higher'):warnings.append('SEMANTIC_REFERENCE_DISTANCE_HIGH')
        if x[3]>np.quantile(self.bundle['calibration_ranks']['lexical_novelty'],.95,method='higher'):warnings.append('LEXICAL_REFERENCE_COVERAGE_LOW')
        if x[7]>np.quantile(self.bundle['calibration_ranks']['disagreement'],.95,method='higher'):warnings.append('MODELS_DISAGREE_NOT_PROOF_OF_ERROR')
        margin=float(risk_margin(self.bundle['full'],x[None,:])[0])
        cal=self.bundle['calibration_ranks']['learned_full']
        percentile=float(np.searchsorted(cal,margin,side='right')/(len(cal)+1))
        experimental=margin>self.bundle['thresholds']['learned_full']['0.2']
        lo,hi=interval(np.array([score]),np.array([self.bundle['global_q']]))
        result['candidate_interval']={'low':float(lo[0]),'high':float(hi[0]),'scale':[0,10],
          'nominal_target':.9,'scope':'internal split-conformal candidate, bound to this research score replica',
          'external_coverage_guarantee':False,'per_essay_correctness_probability':None}
        interval_blockers=['INSUFFICIENT_CHINESE_OR_LANGUAGE_OUTSIDE_SCOPE','TEXT_ENCODING_SUSPECT',
            'EXTREME_REPETITION','LENGTH_OUTSIDE_FITTING_RANGE','DOMAIN_NOT_EXTERNALLY_VALIDATED',
            'SEMANTIC_REFERENCE_DISTANCE_HIGH','LEXICAL_REFERENCE_COVERAGE_LOW']
        if any(w in warnings for w in interval_blockers):
            result['candidate_interval']=None
            reasons.append('REFERENCE_SHIFT_OR_INPUT_ANOMALY_INTERVAL_SUPPRESSED')
        result['diagnostics']={'uncalibrated_margin':margin,'calibration_rank_percentile':percentile,
           'percentile_is_error_probability':False,'feature_provider_semantic_values_require_trusted_adapter':True}
        if experimental_risk:
            result['mode']='experimental_diagnostics';result['experimental_review_flag']=bool(experimental)
            reasons.append('UNSUPPORTED_RESEARCH_RANKER_DO_NOT_AUTOMATE')
        reasons.append('EXTERNAL_TEACHER_VALIDATION_PENDING')
        return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle-dir',type=Path,required=True)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--experimental-risk',action='store_true')
    args=parser.parse_args()
    try:
        request=json.loads(args.request.read_text(encoding='utf-8-sig'))
        output=ReviewEngine(args.bundle_dir).assess(request,args.experimental_risk)
    except (ValueError,OSError) as exc:
        output={'decision':'reject_input','teacher_review_required':True,'reason':'INVALID_REQUEST_OR_ARTIFACT',
                'detail':str(exc),'confidence':None,'error_probability':None}
    print(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
