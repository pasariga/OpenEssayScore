"""Independent checks on split integrity, selection and reported score metrics."""
import csv
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from common import ROOT, PROCESSED, load_split, checksum, dump
from predict import Scorer

run=ROOT/'reports/round1-cfms34-bge-v1'
ckpt=ROOT/'checkpoints/round1-cfms34-bge-v1'
manifest=json.loads((PROCESSED/'split_manifest.json').read_text())
rows={s:load_split(s) for s in ['train','dev','test']}
for name,records in rows.items():
    assert len(records)==manifest['splits'][name]['count']
    assert [r['id'] for r in records]==manifest['splits'][name]['ids']
    assert checksum(PROCESSED/f'{name}.jsonl')==manifest['splits'][name]['sha256']
for a,b in [('train','dev'),('train','test'),('dev','test')]:
    assert not ({r['id'] for r in rows[a]} & {r['id'] for r in rows[b]})
    assert not ({r['text_hash'] for r in rows[a]} & {r['text_hash'] for r in rows[b]})
freeze=json.loads((run/'selection_frozen.json').read_text())
winner=max(freeze['all_validation_candidates'],key=lambda r:(r['dev']['qwk'],-r['dev']['mae']))
assert winner['family']==freeze['selected_family']
assert checksum(ckpt/'best_scorer.joblib')==checksum(ckpt/f"{winner['family']}.joblib")
with (run/'test_predictions.csv').open(encoding='utf-8',newline='') as f:
    preds=list(csv.DictReader(f))
assert [r['id'] for r in preds]==[r['id'] for r in rows['test']]
gold=np.array([r['score'] for r in rows['test']])
pred=np.array([float(r[freeze['selected_family']]) for r in preds])
observed=np.zeros((11,11))
for y,p in zip(np.rint(gold).astype(int),np.rint(pred).astype(int)):observed[y,p]+=1
expected=np.outer(observed.sum(axis=1),observed.sum(axis=0))/observed.sum()
weights=(np.arange(11)[:,None]-np.arange(11)[None,:])**2
manual_qwk=float(1-(weights*observed).sum()/(weights*expected).sum())
manual_mae=float(np.mean(np.abs(gold-pred)))
metrics=json.loads((run/'metrics.json').read_text())
reported=metrics['models'][freeze['selected_family']]['test']
assert np.isclose(manual_qwk,reported['qwk'])
assert np.isclose(manual_mae,reported['mae'])
scorer=Scorer(ROOT/'checkpoints/WenHeng-Score-v1-cpu')
assert 'torch' not in sys.modules
for text in ['', 'English-only essay without Chinese.']:
    try:
        scorer.predict([text])
        raise AssertionError('Invalid input should fail')
    except ValueError:
        pass
gpu=json.loads((run/'gpu_usage.json').read_text())
assert gpu['gpu_wall_seconds']<1800
result={'passed':True,'checks':['split file checksums and IDs','no cross-split ID/text-hash overlap',
          'frozen selection agrees with dev-only criterion','saved best weights match selection',
          'test prediction IDs match held-out IDs','MAE independently recomputed',
          'QWK independently recomputed from confusion matrix','selected CPU scorer imports no torch',
          'empty/non-Chinese inputs rejected','GPU phase under 30-minute budget'],
        'manual_test_mae':manual_mae,'manual_test_qwk':manual_qwk}
dump(run/'artifact_validation.json',result)
print(json.dumps(result,indent=2))
