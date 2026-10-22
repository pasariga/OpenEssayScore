"""Optional read-only bridge. v1 produces a score; review remains a separate module."""
import argparse,json,sys
from pathlib import Path
from .review_engine import ReviewEngine,V1_SHA
from .risk_core import RUBRIC,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--v1-bundle',type=Path,required=True)
    p.add_argument('--review-bundle',type=Path,required=True);p.add_argument('--request',type=Path,required=True)
    a=p.parse_args()
    try:
        request=json.loads(a.request.read_text(encoding='utf-8-sig'))
        if not isinstance(request,dict) or request.get('authorization') not in ['public_research_dataset','authorized_research']:
            raise ValueError('Research authorization must be declared')
        if request.get('rubric_id')!=RUBRIC or request.get('score_range')!=[0,10]:raise ValueError('v1 supports CFMS0..10 only')
        if not request.get('prompt_present') or not request.get('ocr_verified'):raise ValueError('Task and verified text required')
        if sha(a.v1_bundle/'best_scorer.joblib')!=V1_SHA:raise ValueError('Original v1 scorer hash differs')
        if sha(a.v1_bundle/'character_tfidf_ridge.joblib')!=V1_SHA:raise ValueError('Original v1 family weights differ')
        if sha(a.v1_bundle/'lexical_vectorizer.joblib')!='e098d09fb07b81320e87a1c2e96cba1387525f64ca93414c279d66a0070a667b':raise ValueError('Original v1 vocabulary differs')
        sys.path.insert(0,str(a.v1_bundle))
        from predict import Scorer
        text=request.get('text')
        if not isinstance(text,str) or not text.strip() or len(text)>20000:raise ValueError('Invalid text')
        value=float(Scorer(a.v1_bundle).predict([text])[0])
        request['score']=value;request['score_model_sha256']=V1_SHA
        review=ReviewEngine(a.review_bundle).assess(request)
        output={'v1_score_0_10':value,'review':review,'score_model_modified':False,
                'note':'v1 interval calibration unavailable; do not transfer research-replica intervals'}
    except (OSError,ValueError,TypeError,KeyError) as exc:
        output={'decision':'reject_input','teacher_review_required':True,'reason':str(exc)}
    print(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
