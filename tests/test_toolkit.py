import json,os,subprocess,sys
from pathlib import Path
import pytest
from openessayscore import Scorer,ServiceError
from openessayscore.service.registry import VERSION

CONTROL='这是接口与恢复测试使用的技术控制文本，不是学生作文，没有人工评分，不能据此说明模型评分准确率。'
BUNDLE=Path(os.environ.get('OES_V1_BUNDLE','workspace/models/v1'))

def test_cli_missing_registry_is_honest(tmp_path):
    r=subprocess.run([sys.executable,'-m','openessayscore','status','--store',str(tmp_path/'absent')],capture_output=True,text=True)
    assert r.returncode==0 and json.loads(r.stdout)['ready'] is False

def test_library_validation_without_weights(tmp_path):
    with pytest.raises(ServiceError) as exc:Scorer(tmp_path).score(' ',writing_prompt='control')
    assert exc.value.code=='INSUFFICIENT_CHINESE_TEXT'

@pytest.mark.skipif(not (BUNDLE/'best_scorer.joblib').exists(),reason='Authorized private v1 artifacts required')
def test_python_cli_http_identical(tmp_path):
    from fastapi.testclient import TestClient
    from openessayscore.service.api import create_app
    store=tmp_path/'models';scorer=Scorer.from_v1_bundle(BUNDLE,model_store=store)
    result=scorer.score(CONTROL,writing_prompt='technical control')
    assert result['model']['version']==VERSION and result['contract_version']=='openessayscore/1.0'
    p=tmp_path/'control.txt';p.write_text(CONTROL,encoding='utf-8')
    r=subprocess.run([sys.executable,'-m','openessayscore','score','--store',str(store),'--text-file',str(p),'--prompt','technical control'],capture_output=True,text=True)
    assert r.returncode==0
    cli=json.loads(r.stdout)
    with TestClient(create_app(scorer.service)) as c:
        http=c.post('/v1/score',json=dict(text=CONTROL,writing_prompt='technical control',authorization_scope='authorized_research',text_verified=True)).json()
    assert cli['prediction']==http['prediction']==result['prediction']
    assert result['prediction']['confidence'] is None
