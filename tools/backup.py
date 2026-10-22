"""On-demand immutable GitHub backup. gh handles authentication; no token access."""
import argparse,json,os,re,subprocess,sys,time,shlex
from pathlib import Path
from release_guard import scan,sha,manifest,create_archive,verify_archive

def command(args,cwd=None,retries=1):
    for attempt in range(retries):
        p=subprocess.run(args,cwd=cwd,text=True,capture_output=True,encoding='utf-8',errors='replace')
        if p.returncode==0:return p.stdout.strip()
        if attempt+1<retries:time.sleep(min(2**attempt,4))
    raise RuntimeError(f'Command failed: {args[0]} {args[1]} (exit {p.returncode}); inspect local command diagnostics, never send credentials')

def gh_path():return os.environ.get('GH_CLI','gh')
def visibility(repo,expected):
    d=json.loads(command([gh_path(),'repo','view',repo,'--json','visibility,nameWithOwner']))
    if d['visibility']!=expected:raise ValueError('Repository visibility mismatch')

def push_git(folder,repo):
    folder=Path(folder)
    if not (folder/'.git').exists():command(['git','init','-b','main'],folder)
    current=command(['git','remote'],folder)
    if 'origin' not in current.splitlines():command(['git','remote','add','origin',f'https://github.com/{repo}.git'],folder)
    elif command(['git','remote','get-url','origin'],folder)!=f'https://github.com/{repo}.git':raise ValueError('Wrong Git origin')
    user=json.loads(command([gh_path(),'api','user','--jq','{login:.login,id:.id}']))
    command(['git','config','user.name',user['login']],folder)
    command(['git','config','user.email',f"{user['id']}+{user['login']}@users.noreply.github.com"],folder)
    command(['git','add','--all'],folder)
    if command(['git','status','--porcelain'],folder):command(['git','commit','-m','Archive reviewed OpenEssayScore sources and reproducibility records'],folder)
    # No force push, reset or history rewriting. Divergence fails for human/agent review.
    helper='!'+shlex.quote(Path(gh_path()).as_posix())+' auth git-credential' if os.environ.get('GH_CLI') else '!gh auth git-credential'
    command(['git','-c','credential.helper=','-c','credential.helper='+helper,'push','-u','origin','main'],folder,retries=3)
    return command(['git','rev-parse','HEAD'],folder)

def publish_release(repo,tag,assets,notes):
    visible=subprocess.run([gh_path(),'release','view',tag,'--repo',repo,'--json','tagName'],capture_output=True)
    if visible.returncode:
        command([gh_path(),'release','create',tag,'--repo',repo,'--title',tag,'--notes-file',str(notes)],retries=3)
    for asset in assets:
        # Upload is append-only. Existing names are verified instead of clobbered.
        listing=json.loads(command([gh_path(),'release','view',tag,'--repo',repo,'--json','assets']))
        existing=next((a for a in listing['assets'] if a['name']==Path(asset).name),None)
        if existing:
            verification=Path(asset).parent/('existing-'+tag+'-'+Path(asset).name)
            if verification.exists():verification=verification.with_name(str(time.time_ns())+'-'+verification.name)
            command([gh_path(),'release','download',tag,'--repo',repo,'--pattern',Path(asset).name,'--output',str(verification)],retries=3)
            if sha(verification)!=sha(asset):raise ValueError('Existing release asset differs; choose a new tag')
        else:command([gh_path(),'release','upload',tag,'--repo',repo,str(asset)],retries=3)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('scan');q.add_argument('source',type=Path)
    q=sub.add_parser('verify');q.add_argument('archive',type=Path);q.add_argument('--restore',type=Path)
    q=sub.add_parser('sync');q.add_argument('--profile',type=Path,required=True);q.add_argument('--tag',required=True)
    a=p.parse_args()
    if a.action=='scan':
        result=scan(a.source);print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 2)
    if a.action=='verify':print(json.dumps(verify_archive(a.archive,a.restore),indent=2));return
    if not re.fullmatch(r'backup-\d{8}T\d{6}Z(?:-[a-z0-9]+)?',a.tag):raise ValueError('Use a new UTC backup timestamp tag')
    cfg=json.loads(a.profile.read_text(encoding='utf-8-sig'));pub=Path(cfg['public_dir']);arc=Path(cfg['archive_dir']);payload=Path(cfg['payload_dir']);out=Path(cfg['output_dir'])
    out.mkdir(parents=True,exist_ok=True)
    allow=json.loads(Path(cfg['approved_binaries']).read_text(encoding='utf-8'))
    for root,is_private in [(pub,False),(arc,False),(payload,True)]:
        checked=scan(root,private=is_private,approved_binaries=allow)
        if not checked['passed']:print(json.dumps(checked,indent=2));raise ValueError('Sensitive-content scan rejected export')
    visibility(cfg['public_repo'],'PUBLIC');visibility(cfg['archive_repo'],'PRIVATE')
    asset=out/(a.tag+'-research-archive.zip')
    if not asset.exists():create_archive(payload,asset)
    else:
        verify_archive(asset)
        with __import__('zipfile').ZipFile(asset) as z:old=json.loads(z.read('SHA256SUMS.json'))
        if old!=manifest(payload):raise ValueError('Payload changed under existing tag; use a new tag')
    checksum=out/(a.tag+'-SHA256SUMS.json')
    checksum.write_text(json.dumps({'asset':asset.name,'sha256':sha(asset),'scope':'private sanitized research archive; raw student data excluded'},indent=2),encoding='utf-8')
    note=out/(a.tag+'-notes.md');note.write_text('Sanitized research backup. No raw essays, individual embeddings, credentials or private infrastructure configuration. All originals retained offline. See manifests and exclusions.\n',encoding='utf-8')
    version_dir=arc/'versions'/a.tag
    if not version_dir.exists():version_dir.mkdir(parents=True)
    (version_dir/'SHA256SUMS.json').write_bytes(checksum.read_bytes())
    (version_dir/'file_manifest.json').write_text(json.dumps(manifest(payload),ensure_ascii=False,indent=2),encoding='utf-8')
    public_sha=push_git(pub,cfg['public_repo']);archive_sha=push_git(arc,cfg['archive_repo'])
    publish_release(cfg['archive_repo'],a.tag,[asset,checksum],note)
    # Recover uploaded bytes into a new directory, never verify only the upload response.
    restored=out/('download-'+a.tag)
    restored.mkdir(exist_ok=True)
    download=restored/asset.name
    if not download.exists():command([gh_path(),'release','download',a.tag,'--repo',cfg['archive_repo'],'--pattern',asset.name,'--dir',str(restored)],retries=3)
    if sha(download)!=sha(asset):raise ValueError('Downloaded asset checksum mismatch')
    verified=verify_archive(download)
    receipt={'tag':a.tag,'public_commit':public_sha,'archive_commit':archive_sha,'release_asset_sha256':sha(asset),
       'download_verification':verified,'private_release':f"https://github.com/{cfg['archive_repo']}/releases/tag/{a.tag}"}
    (out/(a.tag+'-receipt.json')).write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
