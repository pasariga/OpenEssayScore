"""Actual v1 CPU/UDS engineering test on a technical text, not human-labelled data."""
import json,os,stat,subprocess,sys,tempfile,time
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'shadow'))
from unix_client import UnixScoreClient
from dispatcher import ShadowDispatcher,Journal,Permission


@pytest.mark.skipif(not (Path(os.environ.get('OES_V1_BUNDLE','workspace/models/v1'))/'best_scorer.joblib').exists(),reason='Authorized private v1 bundle required')
def test_actual_v1_unix_service_and_sidecar(tmp_path):
    import psutil
    from openessayscore.service.registry import register_v1,digest,V1_WEIGHT_SHA,V1_VECTOR_SHA
    source=Path(os.environ['OES_V1_BUNDLE']);hashes={n:digest(source/n) for n in ['best_scorer.joblib','lexical_vectorizer.joblib']}
    assert hashes['best_scorer.joblib']==V1_WEIGHT_SHA and hashes['lexical_vectorizer.joblib']==V1_VECTOR_SHA
    store=tmp_path/'model-store';register_v1(store,source)
    with tempfile.TemporaryDirectory(prefix='r6uds-') as sd:
        directory=Path(sd);directory.chmod(0o700);sock=directory/'score.sock'
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        affinity=[min(os.sched_getaffinity(0))]
        child=subprocess.Popen([sys.executable,str(ROOT/'shadow/serve_shadow.py'),'--store',str(store),'--socket',str(sock)],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
        started=time.perf_counter();process=psutil.Process(child.pid);process.cpu_affinity(affinity)
        peak=0;timings=[];client=UnixScoreClient(sock,timeout=2)
        try:
            ready=False;end=time.monotonic()+15
            while child.poll() is None and time.monotonic()<end:
                peak=max(peak,process.memory_info().rss)
                try:ready=client.ready()['ready']
                except Exception:pass
                if ready:break
                time.sleep(.02)
            assert ready,'Owned UDS scorer failed readiness'
            cold=time.perf_counter()-started
            assert stat.S_IMODE(sock.stat().st_mode)==0o600
            assert not [x for x in process.net_connections(kind='inet') if x.status==psutil.CONN_LISTEN]
            health=subprocess.run([sys.executable,str(ROOT/'shadow/healthcheck.py'),'--socket',str(sock)],env=env,text=True,capture_output=True,timeout=3)
            assert health.returncode==0
            technical={'text':'这是影子评分接口的工程控制文本，不是学生作文，没有真实教师评分，用于检验旁路运行和故障隔离。','writing_prompt':'technical transport test'}
            request={**technical,'authorization_scope':'authorized_research','text_verified':True,'include_review':False}
            scores=[]
            for _ in range(50):
                t=time.perf_counter();r=client.score(request);timings.append((time.perf_counter()-t)*1000);scores.append(r['prediction']['overall_score']);peak=max(peak,process.memory_info().rss)
                assert r['prediction']['confidence'] is None and not r['review']['experimental_risk_routing_enabled']
            assert len(set(scores))==1
            policy=json.loads((ROOT/'shadow/policy.json').read_text());policy.update(enabled=True,sample_rate=1)
            journal=tmp_path/'isolated-results.jsonl';dispatcher=ShadowDispatcher(policy,client.score,Journal(journal))
            try:
                # Unit/transport permission only; not a real student authorization record.
                permission=Permission('isolated_shadow_evaluation',True,time.time()+60,guardian_verified=True,deidentification_verified=True)
                assert dispatcher.submit(technical,permission)=='scheduled'
            finally:dispatcher.close()
            event=json.loads(journal.read_text());assert event['native_score']==scores[0] and event['weight']==0
            assert technical['text'] not in journal.read_text() and 'normalized_sha256' not in journal.read_text()
            summary={'scope':'new R6 UDS + dispatcher CPU engineering check, no accuracy evidence','cold_to_ready_seconds':cold,
              'serial_requests':len(timings),'p95_ms':sorted(timings)[int(.95*(len(timings)-1))],
              'sampled_peak_rss_bytes':peak,'one_cpu_affinity':True,'tcp_listeners':0,'socket_mode':'0600',
              'hard_docker_cgroup_enforcement_tested':False,'cpu_quota_025_tested':False,'gpu_training_seconds':0}
            Path(os.environ.get('OES_EVIDENCE_DIR',str(tmp_path)),'unix-integration.json').write_text(json.dumps(summary,indent=2))
        finally:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=3)
        assert not sock.exists()
    assert all(digest(source/n)==v for n,v in hashes.items())


def test_unix_missing_socket_fails_closed(tmp_path):
    with pytest.raises(OSError):UnixScoreClient(tmp_path/'absent').ready()
