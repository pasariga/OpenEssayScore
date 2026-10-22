import importlib.util,json,sys,zipfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from openessayscore.service.api import create_app
from openessayscore.service.core import ScoreService
from openessayscore.service.collaboration import map_native_score,MappingUnavailable,validate_peer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from release_guard import scan,create_archive,verify_archive

def test_source_only_health_is_honest(tmp_path):
    service=ScoreService(tmp_path/'no_weights')
    with TestClient(create_app(service)) as client:
        assert client.get('/health/live').status_code==200
        assert client.get('/health/ready').status_code==503
        assert client.post('/v1/score',json={}).status_code==422
def test_no_unvalidated_mapping():
    with pytest.raises(MappingUnavailable):map_native_score(7,100)
def test_peer_rejects_missing_traits():assert not validate_peer({'scoring_standard_version':'example-four-traits-100-v1','result':{'overall_score':70}})
def test_scanner_never_prints_credentials(tmp_path):
    secret='gh'+'p_'+'Q'*40;(tmp_path/'bad.txt').write_text(secret)
    result=scan(tmp_path)
    assert not result['passed'] and secret not in json.dumps(result)
def test_student_rows_rejected(tmp_path):
    (tmp_path/'table.csv').write_text('essay_id,total\ncontrol,0\n')
    assert not scan(tmp_path)['passed']
def test_private_binary_requires_exact_allowlist(tmp_path):
    from release_guard import sha
    (tmp_path/'a.joblib').write_bytes(b'opaque-test-bytes')
    assert not scan(tmp_path,private=True)['passed']
    assert scan(tmp_path,private=True,approved_binaries={'a.joblib':sha(tmp_path/'a.joblib')})['passed']
def test_manifest_roundtrip_and_tamper(tmp_path):
    root=tmp_path/'input';root.mkdir();(root/'sample.txt').write_text('Technical archive control, not an essay.')
    archive=tmp_path/'snapshot.zip';create_archive(root,archive)
    assert verify_archive(archive,tmp_path/'restore')['files_verified']==1
    with pytest.raises(ValueError):create_archive(root,archive)
    with zipfile.ZipFile(archive,'a') as z:z.writestr('unlisted.txt','x')
    with pytest.raises(ValueError):verify_archive(archive)
def test_source_tree_compiles():
    root=Path(__file__).resolve().parents[1]
    for folder in ['src','research','tools']:
        for p in (root/folder).rglob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
