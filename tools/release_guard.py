"""Fail-closed source/asset scanning. Findings never print matched secrets."""
import csv,hashlib,io,json,re,zipfile
from pathlib import Path

RULES={
 'private_key':r'-{5}BEGIN (?:[A-Z]+ )?PRIVATE KEY-{5}',
 'github_credential':r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})',
 'cloud_credential':r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|\bLTAI[A-Za-z0-9]{12,}\b',
 'dsw_identity':r'\bdsw-[a-z0-9-]{10,}\b|--instance-id\s*[= ]|wenheng-'+'score-train',
 'private_local_path':r'[A-Za-z]:[/\\]+Users[/\\]+'+'|/'+'root/'+'|/opt/'+'wenheng',
 'private_auth_config':r'\.'+'proxyclientconfig|wenheng_'+'dsw_ed25519|'+r'\.ssh[/\\](?:config|id_)',
}
DENIED_NAMES={'.env','auth.json','local-config.json','.'+'proxyclientconfig'}
BINARY={'.joblib','.pkl','.pickle','.npy','.npz','.bin','.safetensors','.pt','.pth'}
FORBIDDEN_JSON={'largest_errors','fit_ids','validation_ids','student_key','essay_id','evaluated_text'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def scan(root,private=False,approved_binaries=None):
    root=Path(root);approved_binaries=approved_binaries or {};findings=[];files=[]
    for p in sorted(root.rglob('*')):
        if '.git' in p.parts or '__pycache__' in p.parts or not p.is_file():continue
        rel=p.relative_to(root).as_posix();files.append(p)
        if p.is_symlink():findings.append({'path':rel,'rule':'symlink_not_allowed'});continue
        if p.name in DENIED_NAMES or p.name.startswith('.env.') or any(x in p.parts for x in ['raw','processed','fixtures_with_real_text']):
            findings.append({'path':rel,'rule':'excluded_path'});continue
        if p.suffix.lower() in BINARY:
            if not private or approved_binaries.get(rel)!=sha(p):findings.append({'path':rel,'rule':'unapproved_binary'})
            continue
        if p.suffix.lower() in ['.zip','.gz','.bz2','.xz','.7z','.tar']:
            findings.append({'path':rel,'rule':'nested_archive_not_allowed'});continue
        try:text=p.read_text(encoding='utf-8-sig')
        except UnicodeError:findings.append({'path':rel,'rule':'unknown_binary'});continue
        for name,pattern in RULES.items():
            if re.search(pattern,text,re.I if name in ['dsw_identity','private_local_path'] else 0):findings.append({'path':rel,'rule':name})
        if p.suffix=='.csv' and len(list(csv.reader(io.StringIO(text))))>1:findings.append({'path':rel,'rule':'nonempty_csv_not_approved'})
        if p.suffix=='.json':
            try:obj=json.loads(text)
            except ValueError:findings.append({'path':rel,'rule':'invalid_json'});continue
            def walk(value):
                if isinstance(value,dict):
                    for k,v in value.items():
                        if k in FORBIDDEN_JSON and v not in [None,[],{}]:findings.append({'path':rel,'rule':'record_level_json'})
                        if k in ['text','作文内容','essay_text'] and isinstance(v,str) and len(v)>20:findings.append({'path':rel,'rule':'possible_essay_text'})
                        walk(v)
                elif isinstance(value,list):
                    for v in value:walk(v)
            # Schemas describe fields; they are not collected observations.
            if 'schema' not in p.name.lower():walk(obj)
    return {'passed':not findings,'findings':findings,'files_scanned':len(files)}

def manifest(root):
    root=Path(root)
    return {'files':[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
      for p in sorted(root.rglob('*')) if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.name!='SHA256SUMS.json']}

def create_archive(source,target):
    source=Path(source);target=Path(target)
    if target.exists():raise ValueError('Archive already exists; use a new version')
    m=manifest(source)
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for row in m['files']:z.write(source/row['path'],arcname=row['path'])
        z.writestr('SHA256SUMS.json',json.dumps(m,ensure_ascii=False,indent=2))
    verify_archive(target)
    return m

def verify_archive(archive,destination=None):
    destination=Path(destination).resolve() if destination else None
    if destination and destination.exists() and any(destination.iterdir()):raise ValueError('Restore destination must be empty')
    with zipfile.ZipFile(archive) as z:
        m=json.loads(z.read('SHA256SUMS.json'))
        expected={r['path'] for r in m['files']}|{'SHA256SUMS.json'}
        if set(z.namelist())!=expected or len(z.namelist())!=len(expected):raise ValueError('Archive contains unlisted or duplicate files')
        if any((i.external_attr>>16)&0o170000==0o120000 for i in z.infolist()):raise ValueError('Archive symlinks are forbidden')
        for row in m['files']:
            name=row['path'];path=Path(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:raise ValueError('Unsafe archive member')
            data=z.read(name)
            if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Checksum mismatch')
        if destination:
            destination.mkdir(parents=True,exist_ok=True)
            for name in expected:
                p=(destination/name).resolve()
                if not p.is_relative_to(destination):raise ValueError('Unsafe restore path')
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
    return {'passed':True,'files_verified':len(m['files']),'archive_sha256':sha(archive)}
