"""Technical transport fixtures only: no teacher scores, students, or real consent."""
import copy,json,os,subprocess,sys,threading,time
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'shadow'));sys.path.insert(0,str(ROOT/'tools'))
from dispatcher import ShadowDispatcher,Journal,Permission,VERSION,RUBRIC,WEIGHTS_SHA,PREPROCESS
from capacity_gate import assess
from monitor_resources import recommendation

CONTROL={'text':'这是软件影子流程的技术控制文本，不是学生作文，没有真实教师评分和真实授权记录，不能用于准确率验证。','writing_prompt':'transport control'}


def policy(**patch):
    d=json.loads((ROOT/'shadow/policy.json').read_text());d.update(enabled=True,sample_rate=1,max_per_hour=100);d.update(patch);return d


def permission(**patch):
    d=dict(purpose='isolated_shadow_evaluation',verified=True,expires_at=time.time()+3600,guardian_verified=True,deidentification_verified=True);d.update(patch);return Permission(**d)


def response():
    return {'contract_version':'openessayscore/1.0','model':{'version':VERSION,'rubric_id':RUBRIC,'score_range':[0,10],
       'weights_sha256':WEIGHTS_SHA,'preprocessing_version':PREPROCESS},'prediction':{'overall_score':4.5,'confidence':None},
       'review':{'experimental_risk_routing_enabled':False},'input':{'text':'DO_NOT_LOG_ORIGINAL_INPUT'}}


def wait_count(d,total):
    end=time.monotonic()+3
    while sum(d.counts.values())<total and time.monotonic()<end:time.sleep(.005)
    assert sum(d.counts.values())>=total


def test_default_disabled_never_calls(tmp_path):
    d=ShadowDispatcher(policy(enabled=False),lambda _:pytest.fail('must not score'),Journal(tmp_path/'events'))
    try:assert d.submit(CONTROL,permission())=='disabled'
    finally:d.close()
    assert not (tmp_path/'events').exists()


@pytest.mark.parametrize('patch',[{'verified':False},{'verified':'true'},{'withdrawn':True},{'expires_at':0},{'expires_at':float('nan')},
  {'purpose':'ordinary_site_scoring'},{'guardian_verified':False},{'deidentification_verified':False}])
def test_permission_fail_closed(tmp_path,patch):
    d=ShadowDispatcher(policy(),lambda _:pytest.fail('unauthorized'),Journal(tmp_path/'events'))
    try:assert d.submit(CONTROL,permission(**patch))=='authorization_required'
    finally:d.close()


def test_bounded_queue_is_nonblocking_and_does_not_take_primary_result(tmp_path):
    started=threading.Event();release=threading.Event();primary={'overall':81,'opaque':['unchanged']};before=copy.deepcopy(primary)
    def call(_):started.set();assert release.wait(3);return response()
    d=ShadowDispatcher(policy(max_pending=2),call,Journal(tmp_path/'events'))
    try:
        assert d.submit(CONTROL,permission())=='scheduled';assert started.wait(1)
        assert d.submit(CONTROL,permission())=='scheduled'
        start=time.monotonic();assert d.submit(CONTROL,permission())=='queue_full';assert time.monotonic()-start<.1
        assert primary==before
    finally:release.set();d.close()
    assert d.counts['completed']==2


def test_sample_and_hourly_limits(tmp_path):
    d=ShadowDispatcher(policy(sample_rate=0),lambda _:response(),Journal(tmp_path/'events'))
    try:assert d.submit(CONTROL,permission())=='not_sampled'
    finally:d.close()
    d=ShadowDispatcher(policy(max_per_hour=1),lambda _:response(),Journal(tmp_path/'events'))
    try:
        assert d.submit(CONTROL,permission())=='scheduled';wait_count(d,1)
        assert d.submit(CONTROL,permission())=='hourly_cap'
    finally:d.close()


def test_failures_trip_circuit_without_logging_exception_text(tmp_path):
    now=[0.]
    def bad(_):raise TimeoutError('DO_NOT_LOG_PRIVATE_EXCEPTION')
    journal=tmp_path/'events';d=ShadowDispatcher(policy(),bad,Journal(journal),clock=lambda:now[0])
    try:
        for i in range(3):
            assert d.submit(CONTROL,permission())=='scheduled';wait_count(d,i+1)
        assert d.submit(CONTROL,permission())=='circuit_open'
        now[0]=61;d.call=lambda _:response()
        assert d.submit(CONTROL,permission())=='scheduled';wait_count(d,4)
    finally:d.close()
    text=journal.read_text();assert 'DO_NOT_LOG' not in text and CONTROL['text'] not in text


@pytest.mark.parametrize('patch',['bad_scale','bad_version','bad_weights','bad_score','confidence','risk_routing'])
def test_reject_unapproved_outputs(tmp_path,patch):
    r=response()
    if patch=='bad_scale':r['model']['score_range']=[0,100]
    if patch=='bad_version':r['model']['version']='not-v1'
    if patch=='bad_weights':r['model']['weights_sha256']='unknown'
    if patch=='bad_score':r['prediction']['overall_score']=float('nan')
    if patch=='confidence':r['prediction']['confidence']=.9
    if patch=='risk_routing':r['review']['experimental_risk_routing_enabled']=True
    d=ShadowDispatcher(policy(),lambda _:r,Journal(tmp_path/'events'))
    try:assert d.submit(CONTROL,permission())=='scheduled'
    finally:d.close()
    assert d.counts['failed']==1
    assert json.loads((tmp_path/'events').read_text())['status']=='unavailable'


def test_log_allowlist_capacity_and_permissions(tmp_path):
    journal=tmp_path/'events';d=ShadowDispatcher(policy(),lambda _:response(),Journal(journal))
    try:assert d.submit(CONTROL,permission())=='scheduled'
    finally:d.close()
    event=json.loads(journal.read_text());assert event['weight']==0 and event['teacher_review_required']
    assert 'DO_NOT_LOG' not in journal.read_text() and 'input' not in event and 'confidence' not in event
    assert journal.stat().st_mode&0o077==0
    with pytest.raises(ValueError):Journal(journal).append({'text':'forbidden'})
    with pytest.raises(OSError):Journal(journal,max_bytes=1).append({'status':'unavailable'})


def test_invalid_input_and_unsafe_policy(tmp_path):
    with pytest.raises(ValueError):ShadowDispatcher(policy(aggregation_weight=1),None,None)
    d=ShadowDispatcher(policy(),lambda _:response(),Journal(tmp_path/'events'))
    try:
        assert d.submit({**CONTROL,'student_id':'forbidden'},permission())=='invalid_payload'
        assert d.submit({**CONTROL,'text':'x'*20001},permission())=='invalid_payload'
    finally:d.close()


def test_resource_gates_do_not_claim_peak_coverage():
    audit={'summary':{'minimum_available_memory_bytes':897.4*1024**2},'container_configuration':[{'compose_service':'backend','memory_limit_bytes':600*1024**2}]}
    r=assess(audit,126.8);assert not r['passed'] and 340<r['additional_available_mib_needed']<350
    audit['summary']['minimum_available_memory_bytes']=1500*1024**2;assert assess(audit,126.8)['passed']
    assert r['peak_observation_complete'] is False
    for values in [(200,1,10,0),(900,90,10,0),(900,1,2,0),(900,1,10,1)]:
        assert recommendation(*values)['stop_shadow_recommended']
    assert not recommendation(900,1,10,0)['automatic_production_action']


def test_deploy_and_rollback_default_no_mutation(tmp_path):
    bindir=tmp_path/'bin';bindir.mkdir();log=tmp_path/'calls'
    docker=bindir/'docker';docker.write_text('#!/usr/bin/env python3\nimport os,sys,json\nwith open(os.environ["CALL_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\nif any(x in sys.argv for x in ["up","stop","down","restart"]):raise SystemExit(91)\n');docker.chmod(0o700)
    env={**os.environ,'PATH':str(bindir)+os.pathsep+os.environ['PATH'],'CALL_LOG':str(log),
      'OES_SHADOW_IMAGE':'example.invalid/offline@sha256:'+'1'*64,'OES_MODEL_STORE':'/srv/oes-shadow/models','OES_RESULTS_DIR':'/srv/oes-shadow/results','OES_CONTROL_DIR':'/srv/oes-shadow/control'}
    for name in ['activate-shadow.sh','rollback-shadow.sh']:
        p=subprocess.run(['bash',str(ROOT/'deploy'/name)],env=env,capture_output=True,text=True)
        assert p.returncode==0,p.stderr
    assert 'up' not in log.read_text() and 'stop' not in log.read_text()
    p=subprocess.run(['bash',str(ROOT/'deploy/activate-shadow.sh'),'--execute-after-approval'],env=env,capture_output=True,text=True)
    assert p.returncode==3 and 'up' not in log.read_text()


def test_compose_isolated_no_network_database_or_site_volumes():
    import yaml
    d=yaml.safe_load((ROOT/'deploy/compose.shadow.yaml').read_text());s=d['services']['score']
    assert d['name']=='oes-shadow-r6' and s['network_mode']=='none' and 'ports' not in s
    assert s['mem_limit']=='512m' and s['memswap_limit']=='512m' and s['cpus']==.25
    assert s['read_only'] and s['restart']=='no' and s['cap_drop']==['ALL']
    assert 'env_file' not in s and 'depends_on' not in s


def test_path_alias_cannot_expose_model_as_results(tmp_path):
    env={**os.environ,'OES_SHADOW_IMAGE':'example.invalid/offline@sha256:'+'1'*64,
      'OES_MODEL_STORE':'/srv/oes-shadow/models','OES_RESULTS_DIR':'/srv/oes-shadow/control/../models',
      'OES_CONTROL_DIR':'/srv/oes-shadow/control'}
    p=subprocess.run(['bash',str(ROOT/'deploy/activate-shadow.sh')],env=env,capture_output=True,text=True)
    assert p.returncode==2 and 'overlap' in p.stderr


def test_replay_malformed_and_unauthorized_inputs_do_not_score_or_echo(tmp_path):
    result=tmp_path/'never-created'
    p=subprocess.run([sys.executable,str(ROOT/'shadow/replay.py'),'--policy',str(ROOT/'shadow/policy.json'),
      '--authorizations',str(ROOT/'shadow/authorizations.example.json'),'--socket',str(tmp_path/'absent'),
      '--results',str(result)],input='["DO_NOT_ECHO"]\n'+json.dumps({'authorization_ref':'missing',**CONTROL})+'\n',
      capture_output=True,text=True)
    assert p.returncode==0 and 'DO_NOT_ECHO' not in p.stdout+p.stderr and CONTROL['text'] not in p.stdout+p.stderr
    assert json.loads(p.stdout)['admission']['invalid_or_unauthorized']==2 and not result.exists()


def test_rollback_only_owned_score_container_without_image_configuration(tmp_path):
    bindir=tmp_path/'bin';bindir.mkdir();log=tmp_path/'calls'
    docker=bindir/'docker';docker.write_text('#!/usr/bin/env python3\nimport os,sys,json\na=sys.argv[1:]\nwith open(os.environ["CALL_LOG"],"a") as f:f.write(json.dumps(a)+"\\n")\nif a[0]=="ps":\n assert "label=com.docker.compose.project=oes-shadow-r6" in a and "label=com.docker.compose.service=score" in a\n print("abcdef123456")\nelif a[0]=="stop":\n assert a==["stop","-t","5","abcdef123456"]\nelse:raise SystemExit(92)\n');docker.chmod(0o700)
    env={k:v for k,v in os.environ.items() if not k.startswith('OES_')};env.update(PATH=str(bindir)+os.pathsep+env['PATH'],CALL_LOG=str(log),OES_CHANGE_APPROVED='yes')
    p=subprocess.run(['bash',str(ROOT/'deploy/rollback-shadow.sh'),'--execute-after-approval'],env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert len(log.read_text().splitlines())==2


def test_rollback_reports_docker_failure_instead_of_false_success(tmp_path):
    bindir=tmp_path/'bin';bindir.mkdir();docker=bindir/'docker';docker.write_text('#!/usr/bin/env bash\nexit 17\n');docker.chmod(0o700)
    env={**os.environ,'PATH':str(bindir)+os.pathsep+os.environ['PATH'],'OES_CHANGE_APPROVED':'yes'}
    p=subprocess.run(['bash',str(ROOT/'deploy/rollback-shadow.sh'),'--execute-after-approval'],env=env,capture_output=True,text=True)
    assert p.returncode==17
