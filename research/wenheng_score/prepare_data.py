"""Validate publisher data, audit overlap and freeze a leakage-aware split."""
import collections
import csv
import hashlib
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import ROOT, DATA_REVISION, PROCESSED, dump, normalized, redact, checksum

raw = ROOT / 'data/raw/cfms34' / DATA_REVISION
PROCESSED.mkdir(parents=True, exist_ok=True)
all_rows, rejected = [], []
for split in ['train', 'dev', 'test']:
    with (raw / f'{split}.csv').open(encoding='utf-8-sig', newline='') as f:
        entries = list(csv.DictReader(f))
    for item in entries:
        try:
            r1, r2 = float(item['总分_1']), float(item['总分_2'])
            assert np.isfinite(r1) and np.isfinite(r2) and 0 <= r1 <= 5 and 0 <= r2 <= 5
            assert item['编号'] and len(normalized(item['作文内容'])) >= 20
            text = redact(item['作文内容'])
            all_rows.append({'id': item['编号'], 'split': split, 'text': text,
                             'title': redact(item['作文题目']), 'rater1': r1, 'rater2': r2,
                             'score': r1 + r2, 'text_hash': hashlib.sha256(normalized(text).encode()).hexdigest()})
        except Exception as exc:
            rejected.append({'id': item.get('编号'), 'split': split, 'reason': str(exc)})
assert len(all_rows) == 951, 'Unexpected publisher release: review before proceeding'
assert len({r['id'] for r in all_rows}) == len(all_rows), 'Duplicate IDs require review'

# Audit-only TF-IDF uses all texts, never passed to a scoring model.
audit_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(5, 5), min_df=1, dtype=np.float32)
matrix = audit_vectorizer.fit_transform([normalized(r['text']) for r in all_rows])
similarity = (matrix @ matrix.T).tocoo()
parents = list(range(len(all_rows)))
def find(i):
    while parents[i] != i:
        parents[i] = parents[parents[i]]
        i = parents[i]
    return i
def union(a,b):
    a,b=find(a),find(b)
    parents[max(a,b)]=min(a,b)
near_pairs=[]
for a,b,value in zip(similarity.row,similarity.col,similarity.data):
    if a < b and value >= .90:
        union(int(a),int(b))
        near_pairs.append({'id1':all_rows[a]['id'],'id2':all_rows[b]['id'],
                           'split1':all_rows[a]['split'],'split2':all_rows[b]['split'],
                           'cosine_5gram':float(value)})
exact=collections.defaultdict(list)
for i,row in enumerate(all_rows):
    exact[row['text_hash']].append(i)
for indices in exact.values():
    for i in indices[1:]: union(indices[0],i)
groups=collections.defaultdict(list)
for i in range(len(all_rows)): groups[find(i)].append(i)

kept=[]
priority={'test':0,'dev':1,'train':2}
for group in groups.values():
    # Preserve official held-out membership; remove linked copies from fitting data.
    selected=min(group,key=lambda i:(priority[all_rows[i]['split']],all_rows[i]['id']))
    for i in group:
        if i != selected:
            rejected.append({'id':all_rows[i]['id'],'split':all_rows[i]['split'],
                             'reason':'exact/near-duplicate component','retained_id':all_rows[selected]['id']})
    kept.append(all_rows[selected])

manifest={'source_revision':DATA_REVISION,'target':'sum of two human holistic ratings',
          'score_range':[0,10], 'feature_columns':['text'],
          'normalization':'NFKC, trim, redact email and PRC mobile-number patterns',
          'dedup_rule':'normalized exact hash + character 5gram TFIDF cosine >=0.90 connected components; preserve test then dev then train',
          'audit_vectorizer_is_not_used_by_models':True,'splits':{}}
stats={}
for split in ['train','dev','test']:
    rows=sorted([r for r in kept if r['split']==split],key=lambda r:r['id'])
    path=PROCESSED/f'{split}.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    scores=np.array([r['score'] for r in rows])
    lengths=np.array([len(r['text']) for r in rows])
    manifest['splits'][split]={'count':len(rows),'ids':[r['id'] for r in rows],'sha256':checksum(path)}
    stats[split]={'count':len(rows),'score_distribution':dict(sorted(collections.Counter(scores.tolist()).items())),
                  'score_mean':float(scores.mean()),'score_std':float(scores.std()),
                  'character_length_min':int(lengths.min()),'character_length_median':float(np.median(lengths)),
                  'character_length_max':int(lengths.max()),
                  'unique_titles':len({normalized(r['title']) for r in rows}),
                  'rater_exact_agreement':float(np.mean([r['rater1']==r['rater2'] for r in rows]))}
sets=[{r['text_hash'] for r in kept if r['split']==s} for s in ['train','dev','test']]
assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
audit={'statistics':stats,'rejected':rejected,'near_duplicate_pairs':near_pairs,
       'exact_duplicate_groups':[[all_rows[i]['id'] for i in v] for v in exact.values() if len(v)>1],
       'phone_replacement_count':sum(r['text'].count('[PHONE]') for r in all_rows),
       'email_replacement_count':sum(r['text'].count('[EMAIL]') for r in all_rows),
       'limitations':['exact school grade, age, author/student ID and school ID not released',
                     'official test only contains overall-score agreement cases',
                     'essay-disjoint, not demonstrated student/school/prompt-disjoint',
                     'public pretrained encoder may have undisclosed benchmark contamination',
                     'no generated feedback or AI scores are used']}
dump(PROCESSED/'split_manifest.json',manifest)
dump(ROOT/'reports/round1/data_audit.json',audit)
print(json.dumps(audit,ensure_ascii=False,indent=2))
