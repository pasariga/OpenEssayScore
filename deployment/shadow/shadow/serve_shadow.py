"""Single CPU worker, protected UDS only. Existing scorer remains unchanged."""
import argparse,os,socket,stat,signal
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--store',type=Path,required=True);p.add_argument('--socket',type=Path,required=True)
    a=p.parse_args();parent=a.socket.parent
    if not parent.is_dir() or parent.is_symlink() or parent.stat().st_mode&0o077:
        raise SystemExit('Socket directory must exist, be owned by operator, and be mode0700')
    if parent.stat().st_uid!=os.getuid() or a.socket.exists() or a.socket.is_symlink():raise SystemExit('Unsafe or occupied socket path')
    from openessayscore.service.core import ScoreService
    from openessayscore.service.api import create_app
    import uvicorn
    s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);inode=None
    def terminate(signum,frame):raise SystemExit(0)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    try:
        s.bind(str(a.socket));os.chmod(a.socket,0o600);inode=a.socket.stat().st_ino
        service=ScoreService(a.store,max_inflight=1,memory_budget_mb=384)
        if not service.status()['ready']:raise SystemExit('Model not ready; refusing shadow startup')
        uvicorn.run(create_app(service,max_http_inflight=1),fd=s.fileno(),workers=1,limit_concurrency=4,
                    access_log=False,log_level='warning')
    finally:
        s.close()
        if inode is not None and a.socket.exists() and stat.S_ISSOCK(a.socket.lstat().st_mode) and a.socket.stat().st_ino==inode and a.socket.stat().st_uid==os.getuid():a.socket.unlink()


if __name__=='__main__':main()
