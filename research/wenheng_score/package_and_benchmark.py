"""Export selected CPU scorer and measure fresh-process and warm inference."""
import csv
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import sys
import time
import numpy as np
from common import ROOT, load_split, dump, checksum

RUN='round1-cfms34-bge-v1'
checkpoint=ROOT/'checkpoints'/RUN
reports=ROOT/'reports'/RUN
bundle=ROOT/'checkpoints/WenHeng-Score-v1-cpu'
bundle.mkdir(exist_ok=True)
for name in ['best_scorer.joblib','character_tfidf_ridge.joblib','lexical_vectorizer.joblib',
             'model_metadata.json','training_config.json']:
    shutil.copy2(checkpoint/name,bundle/name)
for name in ['predict.py','features.py']:
    shutil.copy2(Path(__file__).parent/name,bundle/name)
requirements=['numpy','scipy','scikit-learn','joblib']
(bundle/'requirements.txt').write_text(''.join(f'{p}=={importlib.metadata.version(p)}\n' for p in requirements),encoding='utf-8')
(bundle/'ATTRIBUTION.md').write_text('''# Attribution and scope

WenHeng-Score v1 is a supervised Chinese essay scoring research baseline.
It uses human holistic labels from CFMS-34 by Shihang Yang, Sanwoo Lee,
Ningning Zhao and Yunfang Wu, released under CC BY 4.0.
Source: https://github.com/Atiyahsama/HiFTS
Revision: 933489269eb9dbf5f6f00f8e8be6ba0ccebc3dbb
License: https://creativecommons.org/licenses/by/4.0/
Changes: normalization, privacy-pattern masking, holistic-only target,
Ridge regression fit on the 751-essay training split. No AI labels or feedback used.

The selected CPU package contains TF-IDF and Ridge weights, no pretrained encoder.
The semantic comparison used BAAI/bge-small-zh-v1.5 (MIT), kept separately.
This package is a research artifact, not a validated production scoring service.
Exact grades, schools and student identities are not available; no claim of
generalization to all primary/secondary schools or school exam scales is made.
''',encoding='utf-8')

test=load_split('test')
sample=reports/'cpu_smoke_input.txt'
sample.write_text(test[0]['text'],encoding='utf-8')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
         USE_TF='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
start=time.perf_counter()
child=subprocess.run([sys.executable,str(bundle/'predict.py'),'--essay-file',str(sample)],
                     env=env,cwd=str(bundle),capture_output=True,text=True,timeout=30)
assert child.returncode==0,child.stderr
smoke=json.loads(child.stdout)
cold=time.perf_counter()-start
assert smoke['family']=='character_tfidf_ridge' and smoke['device']=='cpu'

from predict import Scorer
import psutil
process=psutil.Process()
rss_before=process.memory_info().rss
start=time.perf_counter();scorer=Scorer(bundle);load_time=time.perf_counter()-start
texts=[r['text'] for r in test]
scorer.predict(texts[:3])
latencies=[];scores=[]
for text in texts:
    start=time.perf_counter();p=scorer.predict([text]);latencies.append(time.perf_counter()-start);scores.append(float(p[0]))
start=time.perf_counter();batch=scorer.predict(texts);batch_seconds=time.perf_counter()-start
with (reports/'test_predictions.csv').open(encoding='utf-8',newline='') as f:
    saved=list(csv.DictReader(f))
expected=np.array([float(r['character_tfidf_ridge']) for r in saved])
assert np.allclose(scores,expected,atol=1e-5)
assert np.allclose(batch,expected,atol=1e-5)
cpu_files={str(p.relative_to(bundle)):{'bytes':p.stat().st_size,'sha256':checksum(p)} for p in bundle.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
selection=json.loads((reports/'selection_frozen.json').read_text())
cpu_fit=sum(r['fit_seconds'] for r in selection['all_validation_candidates'])
result={'cpu_bundle':str(bundle),'selected_family':smoke['family'],'gpu_hidden_during_smoke_test':True,
        'fresh_process_smoke_seconds':cold,'model_load_seconds_in_existing_process':load_time,
        'single_essay_warm_ms_p50':float(np.percentile(latencies,50)*1000),
        'single_essay_warm_ms_p95':float(np.percentile(latencies,95)*1000),
        'batch_100_seconds':batch_seconds,'batch_essays_per_second':len(texts)/batch_seconds,
        'process_rss_before_model_bytes':rss_before,'process_rss_after_benchmark_bytes':process.memory_info().rss,
        'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'bundle_bytes':sum(r['bytes'] for r in cpu_files.values()),'files':cpu_files,
        'all_20_cpu_ridge_candidate_fits_seconds':cpu_fit,
        'prediction_equivalence_to_gpu_embedding_experiment_output':True,
        'measurement_context':'DSW Linux 8 logical CPUs; BLAS/OMP limited to 2; baseline package imports no torch',
        'limitations':'CPU latency on DSW only; not a measurement on target production CPU; not HTTP or concurrency performance'}
dump(reports/'cpu_benchmark.json',result)
dump(bundle/'file_manifest.json',cpu_files)
dump(reports/'cpu_smoke_output.json',smoke)
print(json.dumps(result,ensure_ascii=False,indent=2))
