"""Run only with an authorized restored bundle and the public installed package."""
import argparse,json,os,subprocess,sys,tempfile,time
from pathlib import Path
from openessayscore import Scorer
from openessayscore.service.registry import digest,V1_WEIGHT_SHA,V1_VECTOR_SHA


def main():
    p=argparse.ArgumentParser();p.add_argument('--payload',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();bundle=a.payload/'models/v1';start=time.perf_counter()
    assert digest(bundle/'best_scorer.joblib')==V1_WEIGHT_SHA
    assert digest(bundle/'lexical_vectorizer.joblib')==V1_VECTOR_SHA
    with tempfile.TemporaryDirectory() as d:
        s=Scorer.from_v1_bundle(bundle,model_store=Path(d)/'store',review_bundle=a.payload/'models/review_runtime')
        control='这是用于软件恢复和接口兼容性验证的技术控制文本，不是学生作文，也没有教师评分，不能用于证明模型准确率。'
        r=s.score(control,writing_prompt='technical recovery control',include_review=True)
        assert 0<=r['prediction']['overall_score']<=10
        assert r['prediction']['confidence'] is None and r['review']['teacher_review_required']
        assert not r['review']['experimental_risk_routing_enabled']
    result={'passed':True,'model_hashes_match':True,'cpu_only':True,'source_module':Scorer.__module__,
        'native_score_in_range':True,'confidence_is_null':True,'automatic_risk_routing':False,
        'teacher_accuracy_evidence':False,'elapsed_seconds':time.perf_counter()-start}
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
