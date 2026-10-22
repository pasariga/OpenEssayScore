"""Validate real future annotations; an empty template is explicitly not data."""
import argparse,csv,json,re
from pathlib import Path
from content import SCHEMAS,RUBRIC_ID
parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--out',type=Path)
args=parser.parse_args();errors=[];data={}
for name,header in SCHEMAS.items():
    path=args.directory/f'{name}.csv'
    if not path.is_file():errors.append(f'{name}: missing file');data[name]=[];continue
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=header:errors.append(f'{name}: header mismatch')
        data[name]=[{k:(v or '').strip() for k,v in row.items()} for row in reader if any((v or '').strip() for v in row.values())]
def index(name,key):
    result={}
    for n,row in enumerate(data[name],2):
        value=row.get(key,'')
        if not value or value in result:errors.append(f'{name}:{n}: missing/duplicate {key}')
        result[value]=row
    return result
prompts=index('prompts','prompt_id');essays=index('essays','essay_id');permissions=index('permissions','authorization_id')
ratings={side:index('ratings_'+side,'essay_id') for side in ['A','B']}
def score_check(table,row,n,total_field):
    try:
        values=[float(row[k]) for k in ['A_content','B_structure','C_language','D_conventions']]
        assert all(0<=v<=5 and v.is_integer() for v in values)
        assert abs(sum(values)-float(row[total_field]))<1e-9
    except (ValueError,KeyError,AssertionError):errors.append(f'{table}:{n}: invalid/missing scores or total')
for i,(eid,row) in enumerate(essays.items(),2):
    if row['grade'] not in ['3','4'] or row['rubric_id']!=RUBRIC_ID:errors.append(f'essays:{i}: scope/version mismatch')
    if row['prompt_id'] not in prompts:errors.append(f'essays:{i}: prompt not found')
    elif prompts[row['prompt_id']]['grade']!=row['grade']:errors.append(f'essays:{i}: grade differs from task')
    if not re.fullmatch('[0-9a-fA-F]{64}',row['text_sha256']):errors.append(f'essays:{i}: missing/invalid text hash')
    if row['privacy_status']!='passed':errors.append(f'essays:{i}: privacy not checked')
    auth=permissions.get(row['authorization_id'])
    required=['institution_permission','guardian_consent','separate_sensitive_consent','student_assent']
    if not auth or any(auth[k]!='yes' for k in required):errors.append(f'essays:{i}: required authorization not confirmed')
    if not row['student_key'] or not row['school_key']:errors.append(f'essays:{i}: grouping identifiers missing')
for side in ['A','B']:
    for i,(eid,row) in enumerate(ratings[side].items(),2):
        if eid not in essays:errors.append(f'ratings_{side}:{i}: unknown essay')
        if row['rubric_id']!=RUBRIC_ID or not row['rater_id']:errors.append(f'ratings_{side}:{i}: missing rater/version')
        if row['ratable_status']=='ratable':
            score_check('ratings_'+side,row,i,'total20')
            if not row['evidence']:errors.append(f'ratings_{side}:{i}: missing evidence')
        elif row['ratable_status'] in ['review','unratable']:
            if any(row[k] for k in ['A_content','B_structure','C_language','D_conventions','total20']):errors.append(f'ratings_{side}:{i}: unratable/review must leave scores blank')
        else:errors.append(f'ratings_{side}:{i}: missing ratability')
review_needed=[]
for eid in essays:
    if eid not in ratings['A'] or eid not in ratings['B']:errors.append(f'{eid}: two independent ratings missing')
    elif ratings['A'][eid]['rater_id']==ratings['B'][eid]['rater_id']:errors.append(f'{eid}: raters must differ')
    else:
        a,b=ratings['A'][eid],ratings['B'][eid]
        if a['ratable_status']!='ratable' or b['ratable_status']!='ratable':review_needed.append(eid)
        else:
            try:
                if abs(float(a['total20'])-float(b['total20']))>=4 or any(abs(float(a[k])-float(b[k]))>=2 for k in ['A_content','B_structure','C_language','D_conventions']):review_needed.append(eid)
            except ValueError:pass
seen_students={};seen_tasks={}
for n,row in enumerate(data['split_registry'],2):
    partition=row['partition'];eid=row['essay_id']
    if eid not in essays:errors.append(f'split_registry:{n}: unknown essay')
    else:
        for key in ['student_key','prompt_id','source_group','text_sha256']:
            if row[key]!=essays[eid][key]:errors.append(f'split_registry:{n}: {key} differs from essay record')
        auth=permissions.get(essays[eid]['authorization_id'],{})
        purpose='research_training' if partition=='development' else 'closed_evaluation'
        if auth.get(purpose)!='yes':errors.append(f'split_registry:{n}: purpose not authorized')
        if not auth.get('retention_until') or not auth.get('withdrawal_channel'):errors.append(f'split_registry:{n}: retention or withdrawal channel missing')
    if partition not in ['development','calibration','sealed_test']:errors.append(f'split_registry:{n}: unknown partition')
    student=row['student_key']
    if student in seen_students and seen_students[student]!=partition:errors.append(f'split_registry:{n}: student leakage')
    seen_students[student]=partition
    if partition=='sealed_test' and not all(row[k] for k in ['locked_at','custodian_id','dataset_version','text_sha256']):errors.append(f'split_registry:{n}: test custody incomplete')
for n,row in enumerate(data['adjudications'],2):
    if row['essay_id'] not in essays:errors.append(f'adjudications:{n}: unknown essay')
    if row['reference_rule']=='adjudicated':score_check('adjudications',row,n,'reference_total20')
registered=[r['essay_id'] for r in data['split_registry']]
if len(registered)!=len(set(registered)):errors.append('split_registry: duplicate essay')
if set(essays)-set(registered):errors.append('split_registry: some essays have no partition')
empty=not any(data.values())
report={'status':'EMPTY_TEMPLATES_NOT_READY' if empty else ('INVALID' if errors else 'STRUCTURAL_CHECKS_PASSED_NEEDS_HUMAN_REVIEW'),
        'real_essay_n':len(essays),'rating_rows':sum(len(x) for x in ratings.values()),'errors':errors,'review_needed_ids':review_needed,
        'limitations':['does not verify actual consent documents, reidentification risk, rater independence in practice or text-file hash contents','does not grant permission or activate model training']}
if args.out:args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
if errors:raise SystemExit(2)
