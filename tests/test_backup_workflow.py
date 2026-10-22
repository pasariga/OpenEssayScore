import importlib.util,subprocess,sys,json
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import backup

def test_transient_failure_retries_without_printing_secrets(monkeypatch):
    responses=iter([subprocess.CompletedProcess([],1,'','DO_NOT_PRINT'),subprocess.CompletedProcess([],0,'ok','')])
    monkeypatch.setattr(backup.subprocess,'run',lambda *a,**k:next(responses))
    monkeypatch.setattr(backup.time,'sleep',lambda _:None)
    assert backup.command(['git','push'],retries=2)=='ok'

def test_visibility_mismatch_stops_upload(monkeypatch):
    monkeypatch.setattr(backup,'command',lambda *a,**k:json.dumps({'visibility':'PUBLIC'}))
    with pytest.raises(ValueError):backup.visibility('owner/archive','PRIVATE')

def test_existing_release_asset_verified_not_overwritten(tmp_path,monkeypatch):
    data=b'technical archive bytes';asset=tmp_path/'a.zip';asset.write_bytes(data);calls=[]
    monkeypatch.setattr(backup.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess([],0))
    def command(args,**kwargs):
        calls.append(args)
        if 'view' in args:return json.dumps({'assets':[{'name':'a.zip'}]})
        if 'download' in args:Path(args[args.index('--output')+1]).write_bytes(data);return ''
        raise AssertionError('Unexpected upload/overwrite')
    monkeypatch.setattr(backup,'command',command)
    backup.publish_release('owner/archive','backup-test',[asset],tmp_path/'notes.md')
    assert not any('upload' in c for c in calls)
